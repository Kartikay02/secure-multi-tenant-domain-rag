import re
from collections.abc import Sequence

from app.core.config import GuardrailSettings, get_settings
from app.core.exceptions import GuardrailViolationError
from app.core.logging import get_logger
from app.rag.context.domain import AssembledContext, ContextDocument
from app.rag.validation.domain import ClaimValidation, ValidationResult
from app.rag.validation.evaluator import (
    FRAMING_PREFIX_PATTERN,
    STOPWORDS,
    DeterministicClaimEvaluator,
)
from app.rag.validation.interfaces import ClaimEvaluatorProtocol, GroundingValidatorProtocol

logger = get_logger("app.rag.validation.validator")

UNCERTAINTY_INDICATORS = [
    "do not have sufficient information",
    "cannot answer",
    "not enough information",
    "insufficient context",
    "no relevant context",
    "provided context does not contain",
    "documentation does not mention",
]


def _is_uncertainty_response(text: str) -> bool:
    """Check if the answer explicitly declares uncertainty or insufficient context."""
    text_lower = text.lower()
    return any(indicator in text_lower for indicator in UNCERTAINTY_INDICATORS)


class GroundingValidator(GroundingValidatorProtocol):
    """Production grounding guardrail evaluating factual consistency, citations, and evidence.

    Confidence Score Methodology:
    -----------------------------
    The confidence score S in [0.0, 1.0] is a composite index computed from three factors:
      1. Claim Grounding Ratio (G_ratio, weight 0.60):
         Fraction of substantive claims validated by cited evidence chunks.
      2. Citation Validity Ratio (C_ratio, weight 0.25):
         Penalizes nonexistent/phantom citations and uncited assertions.
      3. Evidence Retrieval Quality (E_quality, weight 0.15):
         Mean relevance/rerank score of cited context chunks.

    Strict Safety Invariants:
      - Phantom citations cap confidence at <= 0.40 and immediately force grounded=False.
      - Unsupported claims exceeding threshold immediately force grounded=False.
      - If grounded=False, conservative fallback replaces answer with refusal.
    """

    def __init__(
        self,
        settings: GuardrailSettings | None = None,
        claim_evaluator: ClaimEvaluatorProtocol | None = None,
    ) -> None:
        guardrail_settings = settings or get_settings().guardrails

        self.enabled = guardrail_settings.enabled
        self.grounding_threshold = guardrail_settings.grounding_threshold
        self.confidence_threshold = guardrail_settings.confidence_threshold
        self.min_claim_overlap = guardrail_settings.min_claim_overlap
        self.require_citations = guardrail_settings.require_citations
        self.fallback_mode = guardrail_settings.fallback_mode
        self.fallback_refusal = guardrail_settings.fallback_refusal_message

        self.claim_evaluator = claim_evaluator or DeterministicClaimEvaluator()

        logger.info(
            "Initialized GroundingValidator",
            extra={
                "enabled": self.enabled,
                "grounding_threshold": self.grounding_threshold,
                "confidence_threshold": self.confidence_threshold,
                "fallback_mode": self.fallback_mode,
            },
        )

    def validate(
        self,
        query: str,
        answer: str,
        context: AssembledContext,
    ) -> ValidationResult:
        """Validate answer against context evidence and citation integrity."""
        if not self.enabled:
            if context.total_chunks == 0 or not context.documents or context.is_empty:
                return ValidationResult(
                    grounded=False,
                    confidence_score=0.0,
                    final_answer=answer,
                    fallback_applied=False,
                    insufficient_context=True,
                )
            return ValidationResult(
                grounded=True,
                confidence_score=1.0,
                final_answer=answer,
                fallback_applied=False,
                insufficient_context=False,
            )

        clean_answer = answer.strip()

        # 1. Check if model explicitly declared uncertainty
        if _is_uncertainty_response(clean_answer):
            logger.info("Answer detected as valid explicit uncertainty declaration.")
            return ValidationResult(
                grounded=True,
                confidence_score=1.0,
                unsupported_claims=[],
                citation_errors=[],
                claims=[],
                fallback_applied=False,
                final_answer=clean_answer,
                insufficient_context=True,
                metrics={"claim_ratio": 1.0, "citation_ratio": 1.0, "evidence_quality": 1.0},
            )

        # 2. Check if context is completely empty while model attempted to answer
        if context.total_chunks == 0 or not context.documents:
            logger.warning("Answer provided when context was empty; flagging as ungrounded.")
            return self._build_ungrounded_result(
                answer=clean_answer,
                unsupported_claims=[clean_answer],
                citation_errors=["No context documents available to support answer."],
                claims=[],
                metrics={"claim_ratio": 0.0, "citation_ratio": 0.0, "evidence_quality": 0.0},
                insufficient_context=True,
                forced_confidence=0.0,
            )

        # 3. Citation and Claim Decomposition (protect fenced code blocks from splitting)
        code_blocks: list[str] = []

        def _save_code(match: re.Match[str]) -> str:
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks)-1}__"

        text_for_split = re.sub(r"```[\s\S]*?```(?:\s*\[\d+\])?", _save_code, clean_answer)
        raw_sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])(?!\s*\[\d+\])\s+|\n+\s*", text_for_split)
            if s.strip()
        ]
        sentences: list[str] = []
        for s in raw_sentences:
            for idx, block in enumerate(code_blocks):
                s = s.replace(f"__CODE_BLOCK_{idx}__", block)
            sentences.append(s)

        validated_claims: list[ClaimValidation] = []
        unsupported_claims: list[str] = []
        citation_errors: list[str] = []

        all_citations_in_answer: list[int] = []
        invalid_citations_set: set[int] = set()

        for s in sentences:
            # Extract citation IDs for this sentence
            raw_cids = [int(m) for m in re.findall(r"\[(\d+)\]", s)]
            all_citations_in_answer.extend(raw_cids)

            # Check for invalid (phantom) citations
            invalid_cids = [cid for cid in raw_cids if context.get_citation(cid) is None]
            valid_cids = [cid for cid in raw_cids if context.get_citation(cid) is not None]

            for inv in invalid_cids:
                invalid_citations_set.add(inv)
                citation_errors.append(f"Citation [{inv}] does not exist in context.")

            # Missing citation check
            missing_citation = False
            clean_s = FRAMING_PREFIX_PATTERN.sub("", s).strip()
            content_words = [
                w
                for w in re.findall(r"\b[a-zA-Z0-9_\-]+\b", clean_s.lower())
                if w not in STOPWORDS and len(w) > 1
            ]
            is_substantive = bool(content_words)

            if self.require_citations and not raw_cids and is_substantive:
                missing_citation = True
                citation_errors.append(f"Claim lacks citation: '{s[:60]}...'")

            # Determine evidence to evaluate against
            evidence_docs: Sequence[ContextDocument]
            if valid_cids:
                evidence_docs = [
                    doc for cid in valid_cids if (doc := context.get_citation(cid)) is not None
                ]
            else:
                # If no citations, evaluate against all context chunks
                evidence_docs = context.documents

            supported, support_score, reason = self.claim_evaluator.evaluate_claim(
                claim=s,
                evidence=evidence_docs,
                min_overlap=self.min_claim_overlap,
            )

            # Invariant enforcement:
            # 1. References to phantom/non-existent citations always invalidate the claim
            if invalid_cids:
                supported = False
                reason = f"References invalid citation(s): {invalid_cids}"
            # 2. If entire answer contains no citations, or claim is unsupported by context
            elif missing_citation:
                if not all_citations_in_answer or not supported:
                    supported = False
                    reason = "Missing required citation identifier."

            claim_val = ClaimValidation(
                claim=s,
                citations=raw_cids,
                supported=supported,
                support_score=support_score,
                missing_citations=missing_citation,
                invalid_citations=invalid_cids,
                unsupported_reason=reason,
            )
            validated_claims.append(claim_val)

            if not supported:
                unsupported_claims.append(s)

        # 4. Multi-Factor Calibrated Confidence Calculation
        total_claims = len(validated_claims)
        supported_claims = sum(1 for c in validated_claims if c.supported)
        g_ratio = supported_claims / total_claims if total_claims > 0 else 1.0

        # Citation validity ratio
        total_citations = len(all_citations_in_answer)
        missing_count = sum(1 for c in validated_claims if c.missing_citations)
        invalid_count = len(invalid_citations_set)

        if total_citations == 0 and missing_count > 0:
            c_ratio = 0.0
        elif total_citations == 0:
            c_ratio = 1.0
        else:
            c_penalty = (invalid_count * 2 + missing_count) / max(
                1, total_citations + missing_count
            )
            c_ratio = max(0.0, 1.0 - c_penalty)

        # Evidence quality from cited chunks
        valid_chunk_scores = [
            doc.score
            for cid in all_citations_in_answer
            if (doc := context.get_citation(cid)) is not None
        ]
        e_quality = sum(valid_chunk_scores) / len(valid_chunk_scores) if valid_chunk_scores else 0.5

        # Weighted composite score
        raw_confidence = (0.60 * g_ratio) + (0.25 * c_ratio) + (0.15 * min(1.0, e_quality))
        calibrated_confidence = round(max(0.0, min(1.0, raw_confidence)), 4)

        # 5. Strict Invariants & Grounding Verdict
        grounded = True

        if invalid_citations_set:
            grounded = False
            calibrated_confidence = min(calibrated_confidence, 0.40)

        if g_ratio < self.grounding_threshold:
            grounded = False

        if calibrated_confidence < self.confidence_threshold:
            grounded = False

        metrics = {
            "claim_grounding_ratio": round(g_ratio, 4),
            "citation_validity_ratio": round(c_ratio, 4),
            "evidence_quality": round(e_quality, 4),
        }

        # 6. Fallback Behavior
        if not grounded:
            return self._build_ungrounded_result(
                answer=clean_answer,
                unsupported_claims=unsupported_claims,
                citation_errors=citation_errors,
                claims=validated_claims,
                metrics=metrics,
                insufficient_context=False,
                forced_confidence=calibrated_confidence,
            )

        return ValidationResult(
            grounded=True,
            confidence_score=calibrated_confidence,
            unsupported_claims=[],
            citation_errors=[],
            claims=validated_claims,
            fallback_applied=False,
            final_answer=clean_answer,
            insufficient_context=False,
            metrics=metrics,
        )

    def _build_ungrounded_result(
        self,
        answer: str,
        unsupported_claims: list[str],
        citation_errors: list[str],
        claims: list[ClaimValidation],
        metrics: dict[str, float],
        insufficient_context: bool,
        forced_confidence: float,
    ) -> ValidationResult:
        """Construct fallback response when validation fails."""
        if self.fallback_mode == "raise":
            logger.error("Grounding validation failed; raising GuardrailViolationError.")
            raise GuardrailViolationError(
                f"Grounding validation failed: {len(unsupported_claims)} unsupported claims.",
                details={
                    "unsupported_claims": unsupported_claims,
                    "citation_errors": citation_errors,
                    "confidence_score": forced_confidence,
                },
            )

        if self.fallback_mode == "annotate":
            warning_header = f"[GUARDRAIL WARNING: Answer contains {len(unsupported_claims)} unsupported claim(s)]\n\n"
            final_ans = warning_header + answer
            fallback_applied = True
        else:
            # Default: refusal
            final_ans = self.fallback_refusal
            fallback_applied = True

        logger.warning(
            f"Grounding validation failed (confidence={forced_confidence:.2f}); applied fallback '{self.fallback_mode}'",
            extra={
                "unsupported_count": len(unsupported_claims),
                "citation_errors_count": len(citation_errors),
                "fallback_mode": self.fallback_mode,
            },
        )

        return ValidationResult(
            grounded=False,
            confidence_score=forced_confidence,
            unsupported_claims=unsupported_claims,
            citation_errors=citation_errors,
            claims=claims,
            fallback_applied=fallback_applied,
            final_answer=final_ans,
            insufficient_context=insufficient_context,
            metrics=metrics,
        )
