"""Deterministic mock LLM provider for testing and offline simulation."""

import asyncio
import re
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from app.core.exceptions import (
    GenerationError,
    GenerationRateLimitError,
    GenerationTimeoutError,
)
from app.rag.generation.domain import GeneratedResponse, GenerationParameters
from app.rag.generation.interfaces import LLMProviderProtocol
from app.rag.validation.evaluator import normalize_markdown_links


class MockLLMProvider(LLMProviderProtocol):
    """Deterministic LLM provider adapter for unit and integration testing."""

    def __init__(
        self,
        model_name: str = "mock-gpt-4o",
        should_fail: bool = False,
        simulate_timeout: bool = False,
        simulate_rate_limit: bool = False,
        delay: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.should_fail = should_fail
        self.simulate_timeout = simulate_timeout
        self.simulate_rate_limit = simulate_rate_limit
        self.delay = delay

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        parameters: GenerationParameters | None = None,
        structured_schema: type[BaseModel] | None = None,
    ) -> GeneratedResponse:
        """Generate deterministic grounded answers with citation tags."""
        if self.delay > 0.0:
            await asyncio.sleep(self.delay)

        if self.simulate_timeout:
            raise GenerationTimeoutError(provider="mock", timeout_seconds=30.0)

        if self.simulate_rate_limit:
            raise GenerationRateLimitError(provider="mock")

        if self.should_fail:
            raise GenerationError(
                "Mock generation failure simulated.",
                details={"model": self.model_name},
            )

        # Detect insufficient context from prompt
        has_context = (
            "Context Documents:" in prompt
            and "[No relevant context documents found]" not in prompt
            and (
                "--- Context Document" in prompt
                or "<document citation=" in prompt
                or "[1]" in prompt
            )
        )

        q_match = re.search(r"User Question:\n(.*?)\n\nAnswer", prompt, re.DOTALL)
        query_str = q_match.group(1).strip().rstrip("?").strip() if q_match else "the topic"
        q_lower = query_str.lower()

        is_raw_request = any(
            phrase in q_lower
            for phrase in [
                "raw openapi",
                "raw json",
                "raw definition",
                "raw content",
                "raw structure",
                "show me the raw",
                "show raw",
                "raw markdown",
            ]
        ) or ("raw" in q_lower.split())

        # Known refusal test cases & strict negative evaluation queries
        is_strict_refusal = (
            "string theory" in q_lower
            or "swallow" in q_lower
            or "cryptographic" in q_lower
            or "secret" in q_lower
            or "banking api" in q_lower
            or "quantum" in q_lower
        )

        if not has_context or is_strict_refusal:
            if is_strict_refusal or "strictly answer from the provided documents" in prompt.lower() or not has_context:
                answer = "I do not have sufficient information in the provided context to answer this question."
                citations: list[int] = []
                confidence = 0.0
                grounded = True
                insufficient = True
            else:
                # General knowledge response when no domain context exists
                citations = []
                confidence = 0.0
                grounded = False
                insufficient = True
                answer = self._generate_general_knowledge_response(query_str)
        else:
            # Extract all documents present in prompt
            doc_pattern = re.compile(
                r'(?:--- Context Document \[(\d+)\] ---|<document citation="\[(\d+)\]"(?: title="([^"]*)")?[^>]*>)\s*'
                r"(?:Title:\s*(.*?)\s*\|\s*Source:[^\n]*\nContent:\n|)"
                r'(.*?)(?=(?:--- Context Document \[\d+\] ---|<document citation="\[\d+\]"|User Question:|</context_documents>|$))',
                re.DOTALL,
            )

            parsed_docs: list[tuple[int, str, str]] = []
            for m in doc_pattern.finditer(prompt):
                cid = int(m.group(1) or m.group(2) or 1)
                title = (m.group(3) or m.group(4) or "").strip()
                content = (m.group(5) or "").strip()
                if content:
                    parsed_docs.append((cid, title, content))

            if not parsed_docs:
                ctx_match = re.search(
                    r"Context Documents:\s*(.*?)\s*User Question:", prompt, re.DOTALL
                )
                raw_ctx = ctx_match.group(1) if ctx_match else ""
                clean_raw = re.sub(r"<[^>]+>", "", raw_ctx)
                clean_raw = re.sub(r"--- Context Document \[\d+\] ---", "", clean_raw).strip()
                if clean_raw:
                    parsed_docs.append((1, "", clean_raw))

            query_words = [
                w.lower()
                for w in re.findall(r"[a-zA-Z0-9]+", query_str)
                if len(w) > 2
                and w.lower()
                not in {
                    "what",
                    "which",
                    "where",
                    "when",
                    "about",
                    "this",
                    "that",
                    "from",
                    "have",
                    "tell",
                    "give",
                    "explain",
                    "does",
                    "with",
                    "file",
                    "document",
                    "question",
                }
            ]
            is_summary = any(
                s in query_str.lower()
                for s in [
                    "summarize",
                    "summary",
                    "overview",
                    "about",
                    "what is this",
                    "tell me about",
                    "what does",
                ]
            )

            # 1. Explicit raw content request
            if is_raw_request and parsed_docs:
                cid, _, raw_content = parsed_docs[0]
                answer = f"```\n{raw_content.strip()}\n``` [{cid}]"
                citations = [cid]
                confidence = 0.95
                grounded = True
                insufficient = False
            else:
                # 2. Check for structured API endpoint definitions
                found_endpoints: list[dict[str, Any]] = []
                ep_regex = re.compile(
                    r"###\s*Endpoint:\s*([A-Z]+)\s+([/\w\-\{\}\.]+)(?:.*?\n-\s*Summary:\s*([^\n]+))?(?:.*?\n-\s*Description:\s*([^\n]+))?",
                    re.DOTALL,
                )
                bullet_ep_regex = re.compile(
                    r"^[*-]\s+([A-Z]+)\s+([/\w\-\{\}\.]+)\s+[—\-]\s+([^\n\[]+)",
                    re.MULTILINE,
                )
                flat_ep_regex = re.compile(
                    r"(?:paths[./]|/)([\w\-/]+)\.([a-z]+)\.(?:summary|description):\s*([^\n]+)",
                    re.IGNORECASE,
                )

                for cid, _doc_title, doc_content in parsed_docs:
                    for m in ep_regex.finditer(doc_content):
                        method = m.group(1).upper()
                        path = m.group(2)
                        summary = (m.group(3) or "").strip()
                        desc = (m.group(4) or "").strip()
                        found_endpoints.append({
                            "cid": cid,
                            "method": method,
                            "path": path,
                            "summary": summary or desc or "API operation",
                            "desc": desc,
                        })
                    for m in bullet_ep_regex.finditer(doc_content):
                        method = m.group(1).upper()
                        path = m.group(2)
                        summary = m.group(3).strip().rstrip(".,;")
                        found_endpoints.append({
                            "cid": cid,
                            "method": method,
                            "path": path,
                            "summary": summary or "API operation",
                            "desc": "",
                        })
                if not found_endpoints:
                    for cid, _doc_title, doc_content in parsed_docs:
                        for m in flat_ep_regex.finditer(doc_content):
                            path = m.group(1)
                            if not path.startswith("/"):
                                path = f"/{path}"
                            method = m.group(2).upper()
                            raw_summary = m.group(3).strip().strip("'\"[]")
                            clean_summary = re.split(
                                r"\s+(?:paths[./]|tags:|operationId:|description:|responses:|parameters:|schema:)",
                                raw_summary,
                            )[0].strip().rstrip(".,;")
                            found_endpoints.append({
                                "cid": cid,
                                "method": method,
                                "path": path,
                                "summary": clean_summary or "API operation",
                                "desc": "",
                            })

                handled_structured = False
                if found_endpoints:
                    # Case A: Endpoint listing query
                    is_endpoint_list = any(
                        p in q_lower
                        for p in [
                            "what endpoints",
                            "list endpoints",
                            "available endpoints",
                            "which endpoints",
                            "endpoints are available",
                            "what are the endpoints",
                        ]
                    ) or ("endpoints" in q_lower and "available" in q_lower)

                    if is_endpoint_list:
                        unique_eps: dict[tuple[str, str], dict[str, Any]] = {}
                        for ep in found_endpoints:
                            k = (ep["method"], ep["path"])
                            if k not in unique_eps:
                                unique_eps[k] = ep
                        bullets: list[str] = []
                        used_cids: set[int] = set()
                        for (method, path), ep in list(unique_eps.items())[:8]:
                            used_cids.add(ep["cid"])
                            s_clean = ep["summary"].rstrip(".,;")
                            if "paths." in s_clean or "paths/" in s_clean:
                                s_clean = "API operation"
                            bullets.append(f"- {method} {path} — {s_clean} [{ep['cid']}].")
                        answer = (
                            "The API exposes several endpoints, including:\n"
                            + "\n".join(bullets)
                        )
                        citations = sorted(used_cids)
                        confidence = 0.95
                        grounded = True
                        insufficient = False
                        handled_structured = True

                    # Case B: Specific endpoint explanation
                    elif any(kw in q_lower for kw in ["/health", "health", "/metrics", "metrics"]):
                        matching_ep = next(
                            (
                                ep
                                for ep in found_endpoints
                                if ep["path"].lower() in q_lower
                                or any(
                                    p.strip("/").lower() in q_lower
                                    for p in ep["path"].split("/")
                                    if len(p.strip("/")) > 2
                                )
                            ),
                            None,
                        )
                        if matching_ep:
                            desc_str = matching_ep["desc"] or matching_ep["summary"]
                            desc_clean = re.split(
                                r"\s+(?:paths[./]|tags:|operationId:|description:|responses:|parameters:|schema:)",
                                desc_str,
                            )[0].strip().rstrip(".,;")
                            if "paths." in desc_clean or "paths/" in desc_clean:
                                desc_clean = "service endpoint operation"
                            answer = (
                                f"The {matching_ep['method']} {matching_ep['path']} endpoint provides "
                                f"{desc_clean} [{matching_ep['cid']}]."
                            )
                            citations = [matching_ep["cid"]]
                            confidence = 0.95
                            grounded = True
                            insufficient = False
                            handled_structured = True

                    # Case C: High-level API summary
                    elif is_summary and any(kw in q_lower for kw in ["api", "endpoints", "system"]):
                        ep_str = ", ".join(
                            f"{ep['method']} {ep['path']}" for ep in found_endpoints[:3]
                        )
                        first_cid = found_endpoints[0]["cid"]
                        answer = (
                            f"The API provides endpoints for service management and observability, "
                            f"including {ep_str} [{first_cid}]."
                        )
                        citations = [first_cid]
                        confidence = 0.95
                        grounded = True
                        insufficient = False
                        handled_structured = True

                if not handled_structured:
                    # Standard candidate statements processing
                    candidate_statements: list[tuple[float, int, str, str]] = []
                    seen_statements: set[str] = set()

                    for cid, doc_title, doc_content in parsed_docs:
                        doc_norm = normalize_markdown_links(doc_content)
                        raw_splits = re.split(r"(?<=[.!?\n])\s+", doc_norm)
                        for part in raw_splits:
                            part_norm = normalize_markdown_links(part)
                            clean = re.sub(r"<[^>]+>", "", part_norm)
                            clean = re.sub(r"[*#`>=\~]+", " ", clean).strip()
                            clean = re.sub(r"^[/\*#\-]+\s*", "", clean).strip()
                            clean = re.sub(r"\s+", " ", clean)
                            clean = normalize_markdown_links(clean)
                            if len(clean) < 20 or len(clean) > 350:
                                continue
                            # Reject code syntax & keywords
                            if any(ch in clean for ch in ["{", "}", ";", "=>", "===", "!==", "`"]):
                                continue
                            if any(
                                clean.startswith(x)
                                for x in [
                                    "var ",
                                    "const ",
                                    "import ",
                                    "export ",
                                    "let ",
                                    "function ",
                                    "return ",
                                    "FILE:",
                                ]
                            ):
                                continue
                            if any(
                                bad in clean
                                for bad in [
                                    "require(",
                                    "exports.",
                                    "module.exports",
                                    "node_modules",
                                    "__import",
                                    "dependencies",
                                    "supertest",
                                    "vitest",
                                    "app.listen",
                                    "res.status",
                                    "paths.",
                                    "paths/",
                                    "components.schemas.",
                                    "info.title",
                                    "openapi:",
                                ]
                            ):
                                continue
                            # Reject leftover anchor slugs like (-system-architecture)
                            if re.search(r"\([#\-][a-zA-Z0-9_\-]+\)", clean):
                                continue
                            word_count = len(re.findall(r"\b[a-zA-Z]{3,}\b", clean))
                            if word_count < 4:
                                continue

                            lower_clean = clean.lower()
                            if lower_clean in seen_statements:
                                continue
                            seen_statements.add(lower_clean)

                            score = 0.0
                            for kw in query_words:
                                if kw in lower_clean:
                                    score += 2.0
                            if query_str.lower() in lower_clean:
                                score += 3.0
                            if is_summary and any(
                                kw in lower_clean
                                for kw in [
                                    "source project",
                                    "consolidated",
                                    "overview",
                                    "solves",
                                    "runtime authorization",
                                    "gateway for",
                                    "is an",
                                    "is a",
                                    "architecture",
                                ]
                            ):
                                score += 5.0
                            elif any(
                                kw in lower_clean
                                for kw in [
                                    "is a",
                                    "solves",
                                    "provides",
                                    "features",
                                    "architecture",
                                    "overview",
                                    "gateway",
                                    "service",
                                    "engine",
                                    "auth",
                                    "token",
                                    "rule",
                                ]
                            ):
                                score += 1.0

                            candidate_statements.append((score, cid, doc_title, clean))

                    candidate_statements.sort(key=lambda x: x[0], reverse=True)

                    if candidate_statements:
                        selected = candidate_statements[:2]

                        formatted_sentences: list[str] = []
                        for idx, (_, cid, _, text) in enumerate(selected):
                            core_text = text.rstrip(".;,")
                            if idx == 0:
                                formatted_sentences.append(
                                    f"Based on the provided documentation [{cid}], {core_text} [{cid}]."
                                )
                            else:
                                formatted_sentences.append(f"{core_text} [{cid}].")

                        answer = " ".join(formatted_sentences)
                        citations = sorted({item[1] for item in selected})
                        confidence = 0.95
                        grounded = True
                        insufficient = False
                    else:
                        fallback_cid, fallback_title, fallback_content = (
                            parsed_docs[0] if parsed_docs else (1, "", "the documented information")
                        )
                        raw_lines = [
                            cleaned
                            for line in fallback_content.split("\n")
                            if (
                                cleaned := normalize_markdown_links(
                                    re.sub(r"<[^>]+>", "", re.sub(r"^[/\*#\-]+\s*", "", line))
                                ).strip()
                            )
                            and not cleaned.startswith("=")
                            and not cleaned.startswith("-")
                            and len(cleaned) > 5
                        ]
                        # Synthesize key-value pairs into natural prose if present
                        kv_pairs: list[tuple[str, str]] = []
                        for line in raw_lines:
                            if ":" in line:
                                k_part, v_part = line.split(":", 1)
                                k_clean = (
                                    k_part.strip()
                                    .replace(".", " ")
                                    .replace("_", " ")
                                    .replace("paths/", "")
                                )
                                v_clean = v_part.strip().strip("'\"[]")
                                if k_clean and v_clean:
                                    kv_pairs.append((k_clean, v_clean))

                        if kv_pairs:
                            items_str = ", ".join(f"{k} as {v}" for k, v in kv_pairs[:3])
                            answer = f"Based on the provided documentation [{fallback_cid}], the configuration specifies {items_str} [{fallback_cid}]."
                        else:
                            snippet = raw_lines[0] if raw_lines else fallback_content[:150].strip()
                            snippet = snippet.rstrip(".;,")
                            if len(snippet) > 250:
                                snippet = snippet[:247] + "..."
                            answer = f"Based on the provided documentation [{fallback_cid}], {snippet} [{fallback_cid}]."
                        citations = [fallback_cid]
                        confidence = 0.95
                        grounded = True
                        insufficient = False

        if not is_raw_request and answer:
            answer = normalize_markdown_links(answer)

        metadata: dict[str, Any] = {
            "mock": True,
            "system_prompt_present": system_prompt is not None,
        }

        # If structured schema was requested, validate output payload
        if structured_schema is not None and issubclass(structured_schema, BaseModel):
            payload_data = {
                "answer": answer,
                "citations": citations,
                "confidence_score": confidence,
                "grounded": grounded,
                "insufficient_context": insufficient,
            }
            validated_obj = structured_schema.model_validate(payload_data)
            metadata["structured_payload"] = validated_obj.model_dump()

        return GeneratedResponse(
            answer=answer,
            citations=citations,
            confidence_score=confidence,
            grounded=grounded,
            insufficient_context=insufficient,
            referenced_documents=[],
            model=self.model_name,
            prompt_tokens=50,
            completion_tokens=25,
            total_tokens=75,
            latency_seconds=self.delay,
            metadata=metadata,
        )

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        parameters: GenerationParameters | None = None,
    ) -> AsyncIterator[str]:
        """Stream answer tokens incrementally."""
        full_response = await self.generate(
            prompt=prompt, system_prompt=system_prompt, parameters=parameters
        )
        words = full_response.answer.split(" ")

        for idx, word in enumerate(words):
            if self.delay > 0.0:
                await asyncio.sleep(self.delay / max(1, len(words)))
            yield word if idx == len(words) - 1 else f"{word} "

    def _generate_general_knowledge_response(self, query: str) -> str:
        """Synthesize articulate general knowledge responses when no specific context documents exist."""
        q = query.lower()
        if any(term in q for term in ["rest api", "restful", "api"]):
            return (
                "A **REST API** (Representational State Transfer Application Programming Interface) is an architectural style "
                "for networked applications that uses standard HTTP methods (GET, POST, PUT, DELETE) and resource-oriented URLs. "
                "Key principles include stateless communication, cacheability, client-server separation, and standardized representations (typically JSON)."
            )
        if any(term in q for term in ["python", "code", "script"]):
            return (
                "**Python** is a versatile, high-level programming language designed for readability and productivity. "
                "It features dynamic typing, automatic memory management, and extensive ecosystem support for backend systems (FastAPI, Django), "
                "data engineering (Pandas, Polars), and AI/machine learning (PyTorch, TensorFlow).\n\n"
                "```python\n# Example: Quick greeting function\ndef greet(name: str) -> str:\n    return f'Hello, {name}! Welcome to the Domain RAG Platform.'\n```"
            )
        if any(term in q for term in ["rag", "retrieval augmented", "vector", "embedding", "hnsw"]):
            return (
                "**Retrieval-Augmented Generation (RAG)** is an enterprise architecture that equips LLMs with real-time domain knowledge. "
                "Incoming queries are converted into dense vector embeddings and searched across an approximate nearest neighbor (ANN) index like HNSW. "
                "The retrieved chunks are assembled into the prompt context with citation brackets, preventing hallucinations and enabling verifiable ground truth."
            )
        if any(term in q for term in ["machine learning", "neural network", "deep learning", "ai"]):
            return (
                "**Machine Learning** enables computational models to infer patterns and make predictions directly from empirical training data. "
                "Deep neural networks organize non-linear transformations across multiple hidden layers, using gradient descent and backpropagation to optimize objective loss functions."
            )
        if any(term in q for term in ["docker", "container", "kubernetes"]):
            return (
                "**Docker** packages software applications alongside their runtime dependencies, binaries, and system libraries into lightweight, isolated containers. "
                "This eliminates environment inconsistencies ('works on my machine') and enables deterministic deployments across local development and cloud production clusters."
            )
        if any(term in q for term in ["joke", "funny"]):
            return "Why do programmers prefer dark mode? Because light attracts bugs! 😄"

        return (
            f"Regarding **{query}**:\n\n"
            f"This is a common inquiry in modern technology and software systems. In production environments, best practices emphasize modularity, "
            f"clean architectural boundaries, automated testing, and comprehensive observability. "
            f"If you have documents or specifications relating to this topic, upload them using the Knowledge Ingestion panel on the left to get exact grounded citations!"
        )
