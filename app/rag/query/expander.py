"""Query expansion, keyword extraction, and decomposition engine."""

import re


class QueryExpander:
    """Query expansion utility for hybrid search and lexical retrieval optimization."""

    _STOP_WORDS = {
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
        "can't",
        "cannot",
        "could",
        "couldn't",
        "did",
        "didn't",
        "do",
        "does",
        "doesn't",
        "doing",
        "don't",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "hadn't",
        "has",
        "hasn't",
        "have",
        "haven't",
        "having",
        "he",
        "he'd",
        "he'll",
        "he's",
        "her",
        "here",
        "here's",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "how's",
        "i",
        "i'd",
        "i'll",
        "i'm",
        "i've",
        "if",
        "in",
        "into",
        "is",
        "isn't",
        "it",
        "it's",
        "its",
        "itself",
        "let's",
        "me",
        "more",
        "most",
        "mustn't",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "ought",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "shan't",
        "she",
        "she'd",
        "she'll",
        "she's",
        "should",
        "shouldn't",
        "so",
        "some",
        "such",
        "than",
        "that",
        "that's",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "there's",
        "these",
        "they",
        "they'd",
        "they'll",
        "they're",
        "they've",
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
        "wasn't",
        "we",
        "we'd",
        "we'll",
        "we're",
        "we've",
        "were",
        "weren't",
        "what",
        "what's",
        "when",
        "when's",
        "where",
        "where's",
        "which",
        "while",
        "who",
        "who's",
        "whom",
        "why",
        "why's",
        "with",
        "won't",
        "would",
        "wouldn't",
        "you",
        "you'd",
        "you'll",
        "you're",
        "you've",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "please",
        "explain",
        "tell",
        "give",
        "show",
    }

    _DOMAIN_ACRONYMS: dict[str, list[str]] = {
        "rag": ["retrieval-augmented generation", "retrieval augmented generation"],
        "rrf": ["reciprocal rank fusion"],
        "hnsw": ["hierarchical navigable small world", "approximate nearest neighbor"],
        "llm": ["large language model"],
        "pii": ["personally identifiable information"],
        "rbac": ["role based access control"],
        "nli": ["natural language inference"],
        "bm25": ["best matching 25", "lexical search"],
        "pgvector": ["postgres vector", "vector extension"],
        "sse": ["server-sent events", "streaming"],
        "ann": ["approximate nearest neighbor"],
    }

    def extract_keywords(self, query: str) -> list[str]:
        """Extract salient search keywords from a query, preserving casing and removing noise words."""
        tokens = re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", query)
        keywords: list[str] = []
        seen = set()
        for token in tokens:
            lower = token.lower()
            if lower not in self._STOP_WORDS and lower not in seen:
                seen.add(lower)
                keywords.append(token)
        return keywords

    def expand_for_lexical(self, query: str) -> str:
        """Enrich a user query with domain expansions while maintaining high relevance."""
        keywords = self.extract_keywords(query)
        expansions: list[str] = list(keywords)

        for kw in keywords:
            lower = kw.lower()
            if lower in self._DOMAIN_ACRONYMS:
                expansions.extend(self._DOMAIN_ACRONYMS[lower])

        # Return concatenated keywords and expansions
        return " ".join(dict.fromkeys(expansions)) if expansions else query

    def decompose_query(self, query: str) -> list[str]:
        """Decompose complex, compound, or multi-hop queries into discrete search queries."""
        cleaned = query.strip()
        sub_queries = [cleaned]

        # Check for compound conjunctions: " ... and how ... ", " ... versus ... ", " ... compared to ... "
        conjunction_patterns = [
            r"\s+and\s+(?:how|what|why|where|when|can|does|is)\s+",
            r"\s+(?:versus|vs\.?)\s+",
            r"\s+compared\s+to\s+",
            r"\s+as\s+well\s+as\s+",
        ]

        for pattern in conjunction_patterns:
            parts = re.split(pattern, cleaned, flags=re.IGNORECASE)
            if len(parts) > 1:
                for part in parts:
                    sub = part.strip().rstrip("?.,")
                    if len(sub) > 5 and sub not in sub_queries:
                        sub_queries.append(sub)
                break

        return sub_queries
