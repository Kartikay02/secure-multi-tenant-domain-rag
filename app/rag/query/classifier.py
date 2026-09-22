"""Zero-latency intent classification engine for search queries."""

import re
from dataclasses import dataclass
from enum import StrEnum


class QueryIntent(StrEnum):
    """Semantic intent category for incoming user queries."""

    FACTUAL = "factual"
    SUMMARY = "summary"
    COMPARISON = "comparison"
    MULTI_HOP = "multi_hop"
    DEFINITION = "definition"
    EXPLORATORY = "exploratory"


@dataclass(frozen=True)
class ClassificationResult:
    """Outcome of query intent classification."""

    intent: QueryIntent
    confidence: float
    reason: str
    matched_pattern: str | None = None


class QueryClassifier:
    """Production intent classifier applying regex heuristics and token patterns.

    Operates in sub-millisecond time (< 0.2ms) without external API overhead.
    """

    # Intent regex patterns ordered by specificity
    _SUMMARY_PATTERNS = [
        re.compile(
            r"\b(summarize|summary|overview|high[\s-]level|tl;?dr|main points|key takeaways|what is (this|the) (document|file|text|repo) about|give me an overview)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(briefly explain all|walk me through|outline the)\b", re.IGNORECASE),
    ]

    _COMPARISON_PATTERNS = [
        re.compile(
            r"\b(compare|comparison|versus|vs\.?|differences? between|pros and cons|trade[\s-]?offs?|how does .* differ from)\b",
            re.IGNORECASE,
        ),
    ]

    _DEFINITION_PATTERNS = [
        re.compile(
            r"^\s*(what is|what are|define|definition of|what does .* mean)\b", re.IGNORECASE
        ),
        re.compile(r"\bmeaning of\b", re.IGNORECASE),
    ]

    _MULTI_HOP_PATTERNS = [
        re.compile(
            r"\b(and how does|which then|after .* what happens|relates to .* and|how does .* connect to .* and)\b",
            re.IGNORECASE,
        ),
    ]

    _FACTUAL_PATTERNS = [
        re.compile(
            r"^\s*(how many|how much|when was|who is|where is|which version|what is the value of|what parameters?|what port|list the)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(exact|specific|constant|threshold|default value)\b", re.IGNORECASE),
    ]

    def classify(self, query: str) -> ClassificationResult:
        """Classify user query into a primary QueryIntent."""
        cleaned = query.strip()
        if not cleaned:
            return ClassificationResult(
                intent=QueryIntent.EXPLORATORY,
                confidence=0.5,
                reason="Empty query defaulted to exploratory.",
            )

        # 1. Check Summary intent
        for pat in self._SUMMARY_PATTERNS:
            match = pat.search(cleaned)
            if match:
                return ClassificationResult(
                    intent=QueryIntent.SUMMARY,
                    confidence=0.95,
                    reason=f"Matched summary pattern: '{match.group(0)}'",
                    matched_pattern=match.group(0),
                )

        # 2. Check Comparison intent
        for pat in self._COMPARISON_PATTERNS:
            match = pat.search(cleaned)
            if match:
                return ClassificationResult(
                    intent=QueryIntent.COMPARISON,
                    confidence=0.90,
                    reason=f"Matched comparison pattern: '{match.group(0)}'",
                    matched_pattern=match.group(0),
                )

        # 3. Check Multi-hop intent
        for pat in self._MULTI_HOP_PATTERNS:
            match = pat.search(cleaned)
            if match:
                return ClassificationResult(
                    intent=QueryIntent.MULTI_HOP,
                    confidence=0.82,
                    reason=f"Matched multi-hop pattern: '{match.group(0)}'",
                    matched_pattern=match.group(0),
                )

        # 4. Check Factual intent (specific attributes, values, metrics, parameters)
        for pat in self._FACTUAL_PATTERNS:
            match = pat.search(cleaned)
            if match:
                return ClassificationResult(
                    intent=QueryIntent.FACTUAL,
                    confidence=0.85,
                    reason=f"Matched factual pattern: '{match.group(0)}'",
                    matched_pattern=match.group(0),
                )

        # 5. Check Definition intent (conceptual explanations, term definitions)
        for pat in self._DEFINITION_PATTERNS:
            match = pat.search(cleaned)
            if match:
                return ClassificationResult(
                    intent=QueryIntent.DEFINITION,
                    confidence=0.88,
                    reason=f"Matched definition pattern: '{match.group(0)}'",
                    matched_pattern=match.group(0),
                )

        # Fallback to Exploratory
        return ClassificationResult(
            intent=QueryIntent.EXPLORATORY,
            confidence=0.70,
            reason="No explicit intent trigger matched; classified as general exploratory query.",
        )
