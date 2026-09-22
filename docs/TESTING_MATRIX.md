# Domain RAG Comprehensive Testing Matrix

This matrix documents the multi-layered test architecture for the enterprise Domain RAG system, detailing the 9 verification layers, component coverage, test scenarios, mock vs real integration dependencies, and deterministic isolation guarantees.

---

## 1. Multi-Layer Testing Architecture

| Layer # | Testing Layer | Scope & Responsibilities | Key Test Files | Isolation & Dependency Mode |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Unit Tests** | Tests discrete functions, domain logic, data models, parsing algorithms, chunkers, and mathematical ranking logic in pure isolation. | `tests/unit/test_parsers.py`<br>`tests/unit/test_chunker.py`<br>`tests/unit/test_fusion_strategies.py`<br>`tests/unit/test_context_*.py` | 100% Mocked / In-memory, zero I/O, deterministic seeds. |
| **2** | **Integration Tests** | Validates inter-component contracts, end-to-end multi-stage pipeline flow, storage synchronization, and repository interactions. | `tests/integration/test_chunking_pipeline.py`<br>`tests/integration/test_embedding_pipeline.py`<br>`tests/integration/test_ingestion_pipeline.py` | In-memory SQLite (`sqlite+aiosqlite:///:memory:`), mock embedding/LLM providers, temp file paths. |
| **3** | **API Tests** | Validates FastAPI HTTP endpoints, RFC 7807 problem details, status codes, query streaming (SSE), authentication headers, and schema validation. | `tests/integration/test_document_api.py`<br>`tests/unit/test_api_query.py`<br>`tests/unit/test_api_documents_v1.py`<br>`tests/integration/test_api_e2e_rag.py` | `httpx.AsyncClient` with `ASGITransport`, injected dependency overrides, dummy API keys. |
| **4** | **Database Tests** | Validates transactional atomicity, session rollbacks on integrity violations, foreign key cascading deletions, and repository error mappings. | `tests/integration/test_database_layer.py`<br>`tests/unit/test_repositories.py` | In-memory async SQLite engine with clean session rollback fixtures per test. |
| **5** | **Retrieval Tests** | Validates dense vector search, PostgreSQL full-text/trigram lexical search, reciprocal rank fusion (RRF), score thresholds, and zero-match handling. | `tests/integration/test_retrieval_resilience.py`<br>`tests/integration/test_hybrid_retrieval.py`<br>`tests/integration/test_vector_retrieval.py`<br>`tests/unit/test_lexical_retriever.py` | In-memory PGVector/Mock vector stores, transactional SQL sessions, mock embedding vectors. |
| **6** | **Ingestion Tests** | Validates document upload, magic byte verification, decompression bomb prevention, metadata extraction, text normalization, and job tracking. | `tests/integration/test_ingestion_pipeline.py`<br>`tests/unit/test_parsers.py`<br>`tests/unit/test_validator.py`<br>`tests/unit/test_sanitizer.py` | Temp filesystem directories (`tmp_path`), real binary parsers (pypdf, python-docx), mock storage. |
| **7** | **RAG Pipeline Tests** | Validates the complete 7-stage orchestrator (preprocessing → retrieval → reranking → context assembly → generation → grounding → telemetry). | `tests/integration/test_rag_orchestration_pipeline.py`<br>`tests/unit/test_rag_orchestrator_service.py` | Replaceable provider adapters, deterministic stage latency tracking, correlation ID propagation. |
| **8** | **Grounding Tests** | Validates fact verification guardrails, claim-evidence overlap calculation, citation bounds checking, uncertainty detection, and fallback substitutions. | `tests/unit/test_grounding_validator.py`<br>`tests/integration/test_grounding_pipeline.py` | Deterministic claim evaluator with zero hallucinated false-positives and strict refusal rules. |
| **9** | **Failure-Mode Tests** | Explicitly verifies system resilience under corrupted inputs, timeouts, upstream service crashes, rate limits, and malicious requests. | `tests/unit/test_failure_modes.py`<br>`tests/integration/test_failure_modes_e2e.py` | Fault-injecting mock providers (`FailingEmbeddingProvider`, `FailingLLMProvider`, `FailingReranker`, `FailingVectorStore`). |

---

## 2. Failure-Mode Test Matrix (13 Scenarios)

Every failure scenario required for production resilience is tested across unit, integration, and HTTP API levels:

| Scenario | Description & Failure Trigger | Unit Test | Integration / API Test | Expected System Behavior & Recovery |
| :--- | :--- | :--- | :--- | :--- |
| **1. Invalid Document** | Unsupported file extension (`.exe`) or spoofed file header (missing `%PDF-` / ZIP header). | `test_scenario_1_invalid_document_unsupported_extension`<br>`test_scenario_1_invalid_document_spoofed_pdf_magic_bytes` | `test_e2e_unsupported_file_extension_returns_415`<br>`test_e2e_corrupted_pdf_header_returns_400` | Rejects file immediately with HTTP 400 (`FILE_VALIDATION_ERROR`) or 415 (`UNSUPPORTED_FILE_TYPE`). No storage leakage. |
| **2. Duplicate Document** | Uploading an identical document with matching SHA-256 content hash. | `test_scenario_2_duplicate_document_skips_reprocessing` | `test_e2e_duplicate_document_returns_200_is_duplicate_true` | Idempotent detection: skips parsing, chunking, and embedding; returns HTTP 200 with `is_duplicate=True` and existing document metadata. |
| **3. Empty Document** | Uploading a 0-byte payload. | `test_scenario_3_empty_document_raises_validation_error` | `test_e2e_empty_document_upload_returns_400` | Rejected during file validation with HTTP 400 (`FILE_VALIDATION_ERROR`). |
| **4. Parser Failure** | Binary stream corrupted mid-file, truncated archive, or corrupted syntax. | `test_scenario_4_parser_failure_pdf_corrupted_stream`<br>`test_scenario_4_parser_failure_docx_corrupted_archive` | `test_ingestion_pipeline_corrupted_file` | Raises `ParsingError`, updates `IngestionJob` to `FAILED` with specific error message, marks version `FAILED`. |
| **5. Embedding Failure** | Upstream embedding provider throws 502, rate limit 429, or network disconnect. | `test_scenario_5_embedding_provider_error`<br>`test_scenario_5_embedding_provider_timeout` | `test_embedding_pipeline_failure` | Wrapped as `EmbeddingError` (502) or `EmbeddingTimeoutError` (504); safe retries if transient, terminates job with clean error trace. |
| **6. Vector DB Failure** | Vector database connection drop or syntax error during similarity search or upsert. | `test_scenario_6_vector_database_failure`<br>`test_scenario_6_dense_retriever_propagates_vector_failure` | `test_retrieval_resilience.py` | Wrapped cleanly as `DatabaseError` (500), preventing raw internal SQL or driver exceptions from leaking. |
| **7. Reranker Failure** | Cross-encoder reranker times out or throws unhandled upstream exception. | `test_scenario_7_reranker_failure_triggers_graceful_fallback`<br>`test_scenario_7_orchestrator_rerank_failure_fallback_metadata` | `test_e2e_reranker_failure_graceful_fallback_returns_200` | **Graceful Fallback**: pipeline catches error, logs warning, and seamlessly falls back to retrieval top-k candidates with `rerank_fallback=True`. Query succeeds with 200 OK. |
| **8. LLM Timeout** | LLM provider response latency exceeds configured timeout budget (`timeout_seconds`). | `test_scenario_8_llm_timeout` | `test_e2e_query_timeout_returns_504` | Terminated via `asyncio.timeout` and raised as HTTP 504 `QUERY_TIMEOUT`. Does not hang server workers. |
| **9. Insufficient Context** | Retrieved chunks are empty, below score threshold, or irrelevant to query. | `test_scenario_9_insufficient_context_honest_refusal`<br>`test_scenario_9_empty_context_unsupported_claim_forces_zero_confidence` | `test_e2e_insufficient_context_query_returns_200` | Returns explicit refusal ("I do not have sufficient information in the provided context..."), sets `insufficient_context=True`, `citations=[]`, `confidence_score=0.0`. |
| **10. Hallucinated Answer** | Generated answer contains claims unsupported by the retrieved context. | `test_scenario_10_hallucinated_answer_detected_and_refused`<br>`test_case_3_hallucinated_answer` | `test_grounding_pipeline.py` | `GroundingValidator` detects unsupported claims, sets `grounded=False`, forces confidence to 0.0, and substitutes conservative fallback refusal. |
| **11. Invalid Citation** | Answer references non-existent citations (e.g. `[99]`) or malformed brackets. | `test_scenario_11_invalid_citation_detected`<br>`test_case_5_invalid_citation_out_of_bounds` | `test_grounding_pipeline.py` | Recorded in `citation_errors`, triggers conservative fallback refusal or ungrounded flag, strips orphan citations. |
| **12. Malformed Request** | Empty or whitespace-only query, missing required request fields, or negative top-k. | `test_scenario_12_malformed_query_empty_or_whitespace` | `test_e2e_malformed_query_request_body_returns_422` | Request validation layer or preprocessor rejects request with HTTP 400 (`QUERY_VALIDATION_ERROR`) or HTTP 422. |
| **13. Unauthorized Access** | Request lacking credentials, invalid API key, or caller accessing another tenant's document. | `test_scenario_13_unauthorized_cross_tenant_document_access` | `test_e2e_unauthenticated_request_returns_401`<br>`test_e2e_invalid_api_key_returns_401`<br>`test_e2e_cross_tenant_document_access_returns_403` | Rejects missing/bad auth with HTTP 401 (`UNAUTHORIZED`); rejects cross-tenant access with HTTP 403 (`FORBIDDEN`). |

---

## 3. Database Layer Integrity & Resilience Verification

| Area | Test Function | Description |
| :--- | :--- | :--- |
| **Transaction Rollbacks** | `test_transaction_rollback_on_integrity_violation` | Verifies that when a database integrity violation occurs (e.g., clashing primary key), the session rolls back cleanly without corrupting subsequent database transactions. |
| **Session Isolation** | `test_session_rollback_cleans_uncommitted_state` | Verifies that uncommitted session changes are completely discarded on rollback, ensuring no dirty reads or orphaned records. |
| **Cascading Deletions** | `test_cascading_delete_document_and_associated_entities` | Verifies that deleting a `Document` cascades atomically through `DocumentVersion`, `DocumentChunk`, and `IngestionJob` records via SQLAlchemy relationship rules. |
| **Job State Tracking** | `test_ingestion_job_state_transitions` | Verifies asynchronous job progression (`QUEUED` → `PROCESSING` → `FAILED`), capturing pipeline stage timestamps and error messages. |
| **Batch Ordering** | `test_chunk_batch_ordered_retrieval` | Verifies that chunks inserted in arbitrary order are retrieved deterministically sorted by `chunk_index ASC`. |

---

## 4. Test Guarantees & Constraints

1. **Deterministic Execution**:
   - Tests do not rely on non-deterministic external LLM responses. Mock providers (`MockLLMProvider`, `MockEmbeddingProvider`, `MockReranker`) use deterministic algorithmic heuristics.
   - Text splitters and claim evaluators execute identical deterministic logic on every run.

2. **Full Test Isolation**:
   - Every database test operates in an isolated in-memory SQLite database (`sqlite+aiosqlite:///:memory:`). Tables are created per test run with auto-rollback after test execution.
   - Local storage fixtures write to pytest `tmp_path` factories that are cleaned up automatically.

3. **Zero Real Production Secrets**:
   - All tests use dummy credentials (`test-api-key`, `test-tenant`, dummy auth tokens). No external API keys (OpenAI, Cohere, Qdrant) are required or used.

4. **Meaningful Assertions**:
   - Tests assert exact HTTP status codes, structured error codes (`FILE_VALIDATION_ERROR`, `FORBIDDEN`, `QUERY_TIMEOUT`), boolean flags (`grounded`, `is_duplicate`, `insufficient_context`), and latency breakdown metrics.
