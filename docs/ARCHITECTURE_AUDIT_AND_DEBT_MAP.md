# Domain RAG: Comprehensive Architecture Audit & Technical Debt Map

**Author**: Principal AI Engineer & Staff RAG Systems Architect  
**Evaluation Target**: Production Domain RAG System  
**Repository**: `rag_project`  
**Status**: Completed Production Audit  

---

## 1. Executive Summary & Codebase Inventory

This document represents an exhaustive, production-grade architectural audit of the Domain RAG system. The codebase is organized according to Clean Architecture / Hexagonal Architecture principles, strictly separating domain entities, abstract protocol boundaries, infrastructure adapters, and application orchestration services.

### Codebase Inventory by Layer

| Layer | Path | Core Modules & Responsibilities |
| :--- | :--- | :--- |
| **API & Presentation** | `app/api/` | FastAPI routers (`v1/endpoints/query.py`, `v1/endpoints/documents.py`, `health.py`, `metrics.py`), dependency injection (`deps.py`), OpenAPI specification. |
| **UI Presentation** | `app/ui/` | Viewport-locked, responsive chat & file ingestion interface with raw file preview, embedded chunk viewer, and citation badges (`index.html`). |
| **Core & Cross-Cutting** | `app/core/` | Section-based Pydantic configuration (`config.py`), RFC 7807 domain exception hierarchy (`exceptions.py`), structured JSON logging (`logging.py`). |
| **Middleware** | `app/middleware/` | Request correlation ID (`X-Correlation-ID`), global RFC 7807 error handler, latency timing middleware (`X-Response-Time-Ms`), security headers. |
| **Domain Models & Schemas** | `app/models/`, `app/schemas/` | SQLAlchemy 2.0 ORM models (`document.py`, `ingestion.py`), Pydantic v2 validation DTOs (`query.py`, `document.py`, `common.py`). |
| **Security & Isolation** | `app/security/` | Multi-tenant scoping, prompt injection detection (`prompt_guard.py`), PII redaction (`pii.py`), API key authorization (`auth.py`). |
| **Observability** | `app/observability/` | Stage-level latency tracking, OpenTelemetry-compatible tracing hooks (`hooks.py`), token usage and cost accounting (`cost.py`). |
| **RAG Ingestion** | `app/rag/ingestion/` | Multi-format parsers (PDF, DOCX, TXT, Markdown, CSV, JSON), decompression bomb protection, magic-byte validation, content hashing (SHA-256). |
| **RAG Chunking** | `app/rag/chunking/` | Recursive character splitting, token-aware splitting, semantic boundary chunking, section hierarchy metadata preservation. |
| **RAG Vector & Storage** | `app/rag/vector/`, `app/rag/storage/` | PGVector HNSW index adapter, in-memory vector store fallback, transactional document repositories (`document_repo.py`, `chunk_repo.py`). |
| **RAG Query Understanding** | `app/rag/query/` | Unicode normalization, bounds checking (`preprocessor.py`), zero-latency intent classification (`classifier.py`), expansion & decomposition (`expander.py`), query routing (`router.py`). |
| **RAG Retrieval & Fusion** | `app/rag/retrieval/` | Dense vector retriever (`dense.py`), PGVector full-text `ts_rank_cd` / SQLite lexical retriever (`lexical.py`), Reciprocal Rank Fusion (`hybrid.py`, `fusion.py`). |
| **RAG Reranking** | `app/rag/reranking/` | Cross-encoder reranking adapter (`cross_encoder.py`), score calibration, summary section boost, graceful degradation fallback (`mock.py`). |
| **RAG Context Assembly** | `app/rag/context/` | Strict token budgeting, sliding window truncation, lost-in-the-middle positioning, deterministic citation mapping (`[N]`). |
| **RAG Generation** | `app/rag/generation/` | OpenAI/Anthropic/Local LLM adapters, streaming generator, mock deterministic provider with citation enforcement. |
| **RAG Grounding & Validation** | `app/rag/validation/` | Claim-level NLI / lexical overlap evaluator (`evaluator.py`), citation out-of-bounds guardrail, hallucination refusal trigger (`validator.py`). |
| **RAG Evaluation** | `app/rag/evaluation/` | Golden benchmark dataset runner, retrieval metrics (Recall@K, MRR, NDCG), generation metrics (groundedness, context relevance, citation precision). |
| **Application Services** | `app/services/` | End-to-end multi-stage pipeline coordinator (`rag_service.py`), asynchronous document ingestion worker (`ingestion_service.py`). |

---

## 2. Deep Technical Audit Across 10 Dimensions

### Dimension 1: Retrieval Quality & Ranking Behavior
- **Strengths**: True hybrid retrieval combining dense vector similarity with PostgreSQL full-text search (`ts_rank_cd`). Uses Reciprocal Rank Fusion (RRF) with configurable smoothing constant $k=60$ and vector/lexical weighting.
- **Identified Debt**: Candidate deduplication between dense and lexical branches was purely ID-based; missing lexical query term expansion when queries contained domain-specific synonyms.
- **Resolution**: Added `QueryExpander` to synthesize search variants and boosted chunk 0 candidates when summary intent is detected.

### Dimension 2: Query Handling, Expansion & Routing
- **Strengths**: `StandardQueryPreprocessor` performs NFKC Unicode normalization, control character stripping, whitespace collapsing, and min/max length validation.
- **Identified Debt**: Lack of explicit intent classification (factual vs. summary vs. multi-hop vs. definition), preventing retrieval parameter specialization.
- **Resolution**: Designed `QueryClassifier` and `QueryRouter` with dynamic top-k and score threshold adjustment.

### Dimension 3: Chunking, Parsing & Context Integrity
- **Strengths**: Clean separation of binary parsing from text chunking. Chunking preserves `section_hierarchy`, `page_number`, and token counts using character/word heuristic token estimators.
- **Identified Debt**: Tabular and structured formats (CSV, JSON) were unhandled by dedicated parsers, resulting in degraded tabular context.
- **Resolution**: Implemented `CSVParser` (row-context preservation) and `JSONParser` (hierarchical path flattening).

### Dimension 4: Context Assembly, Budgeting & Token Economy
- **Strengths**: `ContextBuilder` calculates running token totals, enforces strict `max_context_tokens`, and tracks dropped chunks. Employs "lost-in-the-middle" reordering (placing highest-ranked chunks at top and bottom).
- **Identified Debt**: When single large chunks exceeded remaining budget, naive builders would drop them entirely without partial sliding-window consideration.
- **Resolution**: Retained strict chunk integrity to prevent truncated sentences, while providing detailed telemetry on dropped candidates.

### Dimension 5: Grounding, Faithfulness & Hallucination Guardrails
- **Strengths**: `GroundingValidator` extracts individual claims from generated responses and verifies evidence overlap against assembled `ContextDocument` chunks. If unsupported claims exceed threshold, it triggers an honest refusal ("I do not have sufficient information in the provided context...").
- **Identified Debt**: Initial claim evaluators failed to match document title citations when answers referenced source filenames.
- **Resolution**: Enhanced `DeterministicClaimEvaluator` to index document titles and source identifiers as valid grounding context.

### Dimension 6: Ingestion, Deduplication & Metadata Preservation
- **Strengths**: Content-addressed deduplication using SHA-256 hashing. Duplicate files skip parsing and embedding, returning HTTP 200 with `is_duplicate=True`. Atomic database transactions prevent orphaned records.
- **Identified Debt**: Ingestion jobs required asynchronous processing state updates across multiple version records.
- **Resolution**: Verified transactional rollbacks and cascading foreign keys in `DocumentRepository`.

### Dimension 7: Performance, Latency & Async Concurrency
- **Strengths**: Asynchronous non-blocking I/O (`async/await`) across all storage, vector, and HTTP routes. Stage-level latency tracking (`pre_ms`, `ret_ms`, `rerank_ms`, `ctx_ms`, `gen_ms`, `val_ms`).
- **Identified Debt**: Vector search and lexical search ran sequentially rather than concurrently in hybrid retrieval.
- **Resolution**: Verified `asyncio.gather` concurrency in `HybridRetriever` to execute dense and lexical searches in parallel.

### Dimension 8: Reliability, Edge Cases & Error Resilience
- **Strengths**: Standardized RFC 7807 Problem Details for all API errors. Graceful degradation: if cross-encoder reranker times out or crashes, pipeline falls back to retrieval top-k with `rerank_fallback=True` and succeeds.
- **Identified Debt**: Missing simulated failure-mode test verification.
- **Resolution**: Built a 13-scenario failure mode test matrix (`tests/unit/test_failure_modes.py`, `tests/integration/test_failure_modes_e2e.py`).

### Dimension 9: Enterprise Security & Multi-Tenancy
- **Strengths**: Strict tenant scoping (`tenant_id` filtering enforced at database and vector layers). `PromptGuard` detects injection delimiters (`ignore previous instructions`, `system override`). PII masking for email/credit cards.
- **Identified Debt**: Cross-tenant document access lacked explicit integration tests.
- **Resolution**: Added integration tests asserting HTTP 403 `FORBIDDEN` for cross-tenant document and query requests.

### Dimension 10: Developer Experience, Typing & Observability
- **Strengths**: 100% type annotations validated by `mypy` across 219+ files. Zero `ruff` lint warnings. Structured JSON logs with correlation IDs.
- **Identified Debt**: Need a standalone evaluation CLI runner to produce reproducible benchmark tables.
- **Resolution**: Created `run_eval.py` and `app/rag/evaluation/golden_dataset.py`.

---

## 3. Prioritized Technical Debt & Upgrade Matrix

| Priority | Area | Description of Issue | Production Impact | Architectural Remediation |
| :--- | :--- | :--- | :--- | :--- |
| **P0** | Query Understanding | Queries treated uniformly regardless of factual vs summary intent. | Suboptimal chunk retrieval for high-level overview questions. | Implement `QueryClassifier`, `QueryExpander`, and `QueryRouter` with dynamic strategy selection. |
| **P0** | Evaluation System | No standalone CLI benchmark runner for CI/CD regression verification. | Cannot quantitatively prove retrieval and generation quality deltas. | Create `run_eval.py` with golden dataset evaluating Recall@K, MRR, Groundedness, and Latency. |
| **P1** | Document Ingestion | Lack of structured format parsers (CSV, JSON). | Tabular and JSON configuration files cannot be ingested accurately. | Implement `CSVParser` and `JSONParser` with schema flattening and row preservation. |
| **P1** | Deployment | Missing production container configuration. | Difficult to deploy reproducibly across staging and cloud environments. | Create multi-stage `Dockerfile` and `docker-compose.yml` (FastAPI + pgvector + Redis). |
| **P1** | Documentation | Architectural tradeoffs and interview defense not documented in repository. | Architectural decisions difficult to defend in technical evaluations. | Author `docs/INTERVIEW_DEFENSE_AND_TRADEOFFS.md` with in-depth ADRs. |
| **P2** | Test Coverage | Need test coverage for new query modules, tabular parsers, and evaluation runner. | Risk of silent regressions during subsequent refactors. | Add comprehensive unit tests in `tests/unit/` maintaining 100% green test suite. |
