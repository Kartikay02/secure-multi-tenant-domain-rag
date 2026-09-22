"""Claim evaluators verifying factual propositions against source evidence."""

import re
from collections.abc import Sequence

from app.rag.context.domain import ContextDocument
from app.rag.validation.interfaces import ClaimEvaluatorProtocol

# Standard English stopwords to focus evaluation on informative semantic terms
STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren't",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "can't",
    "cannot",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "isn't",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    # Meta-discourse and conversational framing words
    "based",
    "provided",
    "documentation",
    "document",
    "documents",
    "context",
    "according",
    "stated",
    "mentioned",
    "indicates",
    "shows",
    "explains",
    "handled",
    "using",
    "mechanism",
    "mechanisms",
    "follows",
    "following",
    "overview",
    "summary",
    "topic",
    "information",
    "answer",
    "question",
}

FRAMING_PREFIX_PATTERN = re.compile(
    r"^(?:"
    r"(?:based on|according to|as stated in|as mentioned in|from)\s+(?:the\s+)?(?:provided\s+)?(?:context|documentation|documents?|'[^']*'|\"[^\"]*\"|[^\s,:]+)|"
    r"the\s+(?:provided\s+)?(?:context|documentation|documents?)\s+(?:states?|indicates?|shows?|explains?)|"
    r"in\s+(?:the\s+)?(?:provided\s+)?(?:context|documentation|documents?)|"
    r"(?:the\s+)?(?:api|system|service|specification)\s+(?:exposes?|provides?|defines?|lists?|contains?|includes?)(?:\s+(?:several|the\s+following|key|various))?(?:\s+(?:endpoints?|methods?|operations?|fields?|properties?))?(?:\s*[:,]?\s*(?:including|namely)?)?|"
    r"(?:available|key|supported)\s+(?:endpoints?|methods?|operations?|features?)\s+(?:are|include|consist of)"
    r")\s*[:,]?\s*",
    re.IGNORECASE,
)


MARKDOWN_LINK_PATTERN = re.compile(
    r"!?\[(?!\d+\])([^\]\n]+)\]\s*\((?:[^)\n]+)\)",
)
ORPHAN_ANCHOR_PATTERN = re.compile(
    r"\s*\([#\-][a-zA-Z0-9_\-]+\)",
)


def normalize_markdown_links(text: str) -> str:
    """Normalize Markdown navigation links into plain readable text.

    Transforms:
        [System Architecture](-system-architecture) -> System Architecture
        [System Architecture](#-system-architecture) -> System Architecture
        [System Architecture] (-system-architecture) -> System Architecture
        [Health](#health-readiness) -> Health
        [API](https://example.com) -> API

    Preserves:
        Citations: [1], [2]
        Technical paths: GET /health
        JSON values and code identifiers
    """
    if not text:
        return text
    text = MARKDOWN_LINK_PATTERN.sub(r"\1", text)
    text = ORPHAN_ANCHOR_PATTERN.sub("", text)
    return text


class DeterministicClaimEvaluator(ClaimEvaluatorProtocol):
    """Deterministic, rule-based claim evaluator for grounding verification.

    Evaluates:
    1. Strict numeric token preservation: flags claims with numbers not present in cited chunks.
    2. Informative lexical term recall: measures the fraction of key terms supported by evidence.
    3. Verbatim phrase matches: rewards phrase fidelity.
    """

    def evaluate_claim(
        self,
        claim: str,
        evidence: Sequence[ContextDocument],
        min_overlap: float = 0.60,
    ) -> tuple[bool, float, str | None]:
        """Evaluate whether a claim is substantiated by cited evidence documents."""
        if not evidence:
            return False, 0.0, "No evidence documents provided for claim."

        # Strip citation tags like [1], [2] from claim text
        clean_claim = re.sub(r"\[\d+\]", "", claim).strip()
        # Normalize any Markdown navigation links to readable text
        clean_claim = normalize_markdown_links(clean_claim)
        # Strip leading bullet indicators (e.g. "- ", "* ", "1. ")
        clean_claim = re.sub(r"^(?:[-*•]|\d+\.)\s*", "", clean_claim).strip()
        # Strip conversational framing prefix (e.g. "Based on the provided documentation,", "From 'file.txt':")
        # Apply iteratively in case multiple framing prefixes are stacked (e.g. "Based on ..., From ...:")
        while True:
            stripped = FRAMING_PREFIX_PATTERN.sub("", clean_claim).strip()
            if stripped == clean_claim:
                break
            clean_claim = stripped

        if not clean_claim:
            return True, 1.0, None

        # 1. Extract and check numeric tokens (including percentages like 99.99%)
        claim_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?", clean_claim))
        combined_evidence_text = " ".join(
            f"{doc.title} {doc.source_id} {doc.content}" for doc in evidence
        ).lower()
        evidence_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?", combined_evidence_text))

        if claim_numbers:
            missing_numbers = claim_numbers - evidence_numbers
            if missing_numbers:
                return (
                    False,
                    0.20,
                    f"Numeric claim mismatch: numbers {sorted(missing_numbers)} not found in cited evidence.",
                )

        # 2. Extract content terms (excluding stopwords)
        raw_words = re.findall(r"\b[a-zA-Z0-9_\-]+\b", clean_claim.lower())
        content_words = [w for w in raw_words if w not in STOPWORDS and len(w) > 1]

        # If claim contains only conversational / transition words, consider it supported
        if not content_words:
            return True, 1.0, None

        evidence_words = set(re.findall(r"\b[a-zA-Z0-9_\-]+\b", combined_evidence_text))

        # 3. Calculate semantic term recall
        unique_content_words = set(content_words)
        matching_words = unique_content_words.intersection(evidence_words)
        recall = len(matching_words) / len(unique_content_words)

        # 4. Check for phrase fidelity (verbatim substring boost)
        clean_lower_claim = clean_claim.lower()
        if clean_lower_claim in combined_evidence_text:
            return True, 1.0, None

        if recall >= min_overlap:
            return True, round(recall, 4), None

        missing_words = sorted(unique_content_words - evidence_words)[:5]
        return (
            False,
            round(recall, 4),
            f"Insufficient term overlap ({recall:.2f} < {min_overlap:.2f}); missing key terms: {missing_words}",
        )


class MockClaimEvaluator(ClaimEvaluatorProtocol):
    """Mock evaluator allowing fine-grained simulation of claim support in tests."""

    def __init__(
        self,
        default_supported: bool = True,
        default_score: float = 0.95,
        fail_on_keywords: list[str] | None = None,
    ) -> None:
        self.default_supported = default_supported
        self.default_score = default_score
        self.fail_on_keywords = fail_on_keywords or []

    def evaluate_claim(
        self,
        claim: str,
        evidence: Sequence[ContextDocument],
        min_overlap: float = 0.60,
    ) -> tuple[bool, float, str | None]:
        if not evidence:
            return False, 0.0, "No evidence provided."

        for kw in self.fail_on_keywords:
            if kw.lower() in claim.lower():
                return False, 0.2, f"Simulated failure on keyword '{kw}'"

        return self.default_supported, self.default_score, None
