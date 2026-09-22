# Domain RAG: Architectural Decision Records & Interview Defense Guide

**Author**: Principal AI Engineer & Staff RAG Systems Architect  
**Document Classification**: Production System Architecture & Technical Defense Guide  
**Target Audience**: Engineering Leadership, System Architects, Technical Interview Panels  

---

## Table of Contents
1. [ADR-001: Hybrid Retrieval (Dense Vector + BM25 Lexical) via Reciprocal Rank Fusion](#adr-001-hybrid-retrieval)
2. [ADR-002: Cross-Encoder Reranking & Score Calibration](#adr-002-cross-encoder-reranking)
3. [ADR-003: Chunking Architecture & Token Budgeting](#adr-003-chunking-architecture)
4. [ADR-004: Grounding Verification, Claim-Level Attributions & Hallucination Guardrails](#adr-004-grounding-verification)
5. [ADR-005: RAG vs. Fine-Tuning vs. Long-Context LLMs: Strategic Tradeoffs](#adr-005-rag-vs-fine-tuning-vs-long-context)
6. [ADR-006: Multi-Tenancy Isolation & Prompt Injection Defenses](#adr-006-multi-tenancy-isolation)
7. [Production Failure Modes & Mitigation Matrix (10 Scenarios)](#production-failure-modes--mitigation-matrix)

---

<a name="adr-001-hybrid-retrieval"></a>
## ADR-001: Hybrid Retrieval (Dense Vector + BM25 Lexical) via Reciprocal Rank Fusion

### Context
Dense vector embeddings excel at semantic similarity, conceptual abstraction, and paraphrased queries. However, they consistently fail on exact token matching, domain-specific acronyms (e.g. `HNSW`, `RRF`, `CVE-2024-3094`), part numbers, and code identifiers. Conversely, sparse lexical search (PostgreSQL `tsvector` with `ts_rank_cd` or BM25) excels at keyword precision but misses semantic equivalents and synonyms.

### Decision
Implement parallel hybrid retrieval fusing dense HNSW vector similarity with full-text lexical ranking using Reciprocal Rank Fusion (RRF):

$$RRF\_Score(d) = \sum_{m \in M} \frac{w_m}{k + \text{rank}_m(d)}$$

Where:
- $M = \{\text{dense}, \text{lexical}\}$
- $k = 60$ (standard Cormack smoothing constant preventing top-rank skew)
- $w_{\text{dense}} = 0.5$, $w_{\text{lexical}} = 0.5$ (dynamically tuned by `QueryRouter`)

### Tradeoff Analysis & Interview Defense

| Dimension | Pure Dense (Vector Only) | Pure Sparse (BM25/Lexical) | Hybrid RRF (This System) |
| :--- | :--- | :--- | :--- |
| **Semantic Recall** | High (captures conceptual intent) | Low (vocabulary mismatch problem) | **Highest** (combines semantic + lexical) |
| **Keyword/Acronym Precision**| Poor (out-of-vocabulary drift) | High (exact inverted index hits) | **Highest** (lexical rank rescues precision) |
| **Query Latency** | ~15-25ms | ~5-12ms | ~22-30ms (concurrent `asyncio.gather`) |
| **Index Complexity** | Embedding compute + vector store | Inverted text index | Dual indexing (Postgres `pgvector` + `tsvector`) |

**Defense Question**: *"Why did you choose Reciprocal Rank Fusion over simple linear score combination ($s = \alpha s_{\text{dense}} + (1-\alpha) s_{\text{lexical}}$)?"*  
**Defense Answer**:  
Linear score combination requires calibrating raw cosine similarity (range $[-1, 1]$ or $[0, 1]$) against unbounded BM25 / `ts_rank_cd` scores (range $[0, \infty)$). In production, lexical scores fluctuate drastically based on document length and term frequency distribution, requiring complex per-query min-max normalization. RRF is scale-invariant: it operates strictly on ordinal rank positions, guaranteeing monotonic robustness without hyperparameter fragility across disparate corpora.

---

<a name="adr-002-cross-encoder-reranking"></a>
## ADR-002: Cross-Encoder Reranking & Score Calibration

### Context
Bi-encoders compute query embeddings and chunk embeddings independently:

$$\text{score}_{\text{bi}} = \cos(\mathbf{E}(q), \mathbf{E}(d))$$

Because bi-encoders never allow query tokens to attend directly to document tokens, they suffer from semantic compression bottlenecks.

### Decision
Introduce a two-stage retrieval pipeline:
1. **Stage 1 (Coarse Retrieval)**: Hybrid dense + lexical retrieval retrieves $K_1 = 15$ candidates.
2. **Stage 2 (Fine Reranking)**: Cross-encoder performs deep all-to-all cross-attention over concatenated tokens $[CLS] \circ q \circ [SEP] \circ d \circ [SEP]$, outputting a calibrated relevance probability $P(\text{relevant} \mid q, d)$, pruning down to $K_2 = 5$ chunks.

```
Query ────► [Stage 1: Hybrid Retrieval] ── (Top 15 Chunks) ──► [Stage 2: Cross-Encoder] ── (Top 5 Chunks) ──► Context Assembly
                 (Dense + Lexical RRF)                               (Deep Cross-Attention)
```

### Tradeoff Analysis & Interview Defense

| Parameter | Bi-Encoder (First Stage) | Cross-Encoder (Second Stage) |
| :--- | :--- | :--- |
| **Attention Mechanism** | Independent: $q \to \mathbf{u}$, $d \to \mathbf{v}$ | Joint Cross-Attention: $Softmax\left(\frac{Q K^T}{\sqrt{d_k}}\right)$ over $(q, d)$ |
| **Throughput** | 10,000+ chunks/sec (ANN index) | ~50-200 pairs/sec (heavy GPU/CPU inference) |
| **Scoring Quality (NDCG@10)**| Baseline (~0.68) | State-of-the-Art (~0.84, +23% improvement) |
| **Fault Resilience** | Critical path | Non-critical with **Graceful Degradation Fallback** |

**Defense Question**: *"What happens if your cross-encoder reranker times out or experiences an OOM crash under load?"*  
**Defense Answer**:  
Our `RAGOrchestratorService` wraps reranking in an explicit circuit-breaking fallback block. If the reranker throws `RerankTimeoutError` or an unexpected runtime exception, the pipeline catches it, emits a warning metric with correlation ID, sets `rerank_fallback=True`, truncates candidates to `retrieved_chunks[:resolved_rerank_top_k]`, and seamlessly continues context assembly. The API returns HTTP 200 with zero client downtime.

---

<a name="adr-003-chunking-architecture"></a>
## ADR-003: Chunking Architecture & Token Budgeting

### Context
Fixed-size token chunking splits sentences arbitrarily, bisecting key factual claims across boundaries. Conversely, giant chunks dilute embedding vector density and waste context token budget.

### Decision
1. **Multi-Strategy Chunking**:
   - **Markdown/Header-Aware Chunking**: Splits text along `#`, `##`, `###` headings, preserving section hierarchy metadata.
   - **Recursive Character Splitting**: Separates text hierarchically by `["\n\n", "\n", ". ", " ", ""]` with target chunk size $C = 512$ tokens and overlap $O = 64$ tokens.
   - **Tabular Parsing**: CSV and JSON parsers convert tabular records into key-value context rows before chunking.
2. **Context Budgeting & Lost-in-the-Middle Mitigation**:
   - Enforces a hard token limit `max_context_tokens` (default 2,048 tokens).
   - Arranges retrieved chunks using bidirectional interleaving: the highest-relevance chunk is placed at the very beginning of the context window (Position 1), the second-highest at the very end (Position N), and lower-ranked chunks in the center.

```
Context Window Ordering:
[System Prompt]
├── Chunk 1 (Highest Relevance, Score: 0.95)   <── Primacy Position
├── Chunk 3 (Medium Relevance, Score: 0.84)
├── Chunk 4 (Lower Relevance, Score: 0.78)
└── Chunk 2 (Second Highest, Score: 0.91)      <── Recency Position
[User Instruction & Question]
```

### Tradeoff Analysis
- **Small Chunks (128 tokens)**: High vector search precision; poor generation context (lacks ambient narrative).
- **Large Chunks (1024 tokens)**: High context coherence; poor retrieval precision (embedding space vector dilution).
- **Sweet Spot (512 tokens with 64 token overlap)**: Optimal balance of semantic distinctiveness and self-contained factual assertions.

---

<a name="adr-004-grounding-verification"></a>
## ADR-004: Grounding Verification, Claim-Level Attributions & Hallucination Guardrails

### Context
LLMs suffer from confabulation (hallucination), synthesizing plausible yet factually unsupported claims. Standard RAG architectures blindly return generated text to the user without post-generation verification.

### Decision
Implement deterministic post-generation validation before returning answers to clients:
1. **Citation Enforcement**: The LLM prompt strictly requires every claim to be tagged with an inline citation bracket `[N]`.
2. **Claim Extraction & Evidence Overlap**: `DeterministicClaimEvaluator` decomposes the generated response into individual claims using sentence boundary tokenization and computes evidence overlap against the assembled `ContextDocument` chunks.
3. **Threshold Gate & Conservative Refusal**: If unsupported claims exceed tolerance or citation errors exist (e.g. referencing non-existent `[99]`), the system overrides the answer with an honest refusal:
   > *"I do not have sufficient information in the provided context to answer this question."*
   Sets `grounded=False`, `confidence_score=0.0`, and `insufficient_context=True`.

---

<a name="adr-005-rag-vs-fine-tuning-vs-long-context"></a>
## ADR-005: RAG vs. Fine-Tuning vs. Long-Context LLMs: Strategic Tradeoffs

### Context
When building enterprise knowledge systems, engineers often debate between RAG, Parameter-Efficient Fine-Tuning (PEFT/LoRA), and massive Long-Context LLM windows (e.g. Gemini 1.5 Pro 1M tokens).

### Decision & Comparative Matrix

| Criteria | RAG (This Architecture) | Fine-Tuning (LoRA / SFT) | Long-Context LLMs (1M+ Tokens) |
| :--- | :--- | :--- | :--- |
| **Knowledge Dynamism** | **Instantaneous**: Documents indexed in seconds; zero model retraining. | **Static**: Requires batch data collection, validation, training cycle ($ hours to days). | **Per-Query**: Requires transmitting entire repository on every prompt. |
| **Factual Verifiability** | **Exact**: Every assertion traceable to citation `[N]` and source chunk ID. | **Opaque**: Knowledge baked into weights; cannot provide deterministic citations. | **Traceable**: Can cite context, but prone to "needle-in-a-haystack" degradation. |
| **Per-Query Latency** | **50 - 150ms** (retrieval + small prompt generation). | **30 - 100ms** (generation only). | **3,000 - 15,000ms** (processing 500k+ input tokens). |
| **Serving Cost per Query** | **Low** ($0.0005 - $0.002 per query). | **Medium** (dedicated inference GPU hosting). | **Extremely High** ($0.05 - $0.20 per query). |
| **Data Security & Multi-Tenancy** | **Built-in**: Query-level `tenant_id` filtering at vector & SQL layer. | **Difficult**: Requires separate fine-tuned adapter per tenant to prevent data leaks. | **Vulnerable**: High risk of tenant data leakage if contexts are merged. |

**Strategic Verdict**:  
- **Fine-tuning** is for teaching models *style, tone, syntax, and specialized task behavior*.
- **RAG** is for providing models *factual knowledge, up-to-date domain data, and access-controlled enterprise documents*.
- This system uses RAG for enterprise knowledge retrieval and relies on prompt contracts for format adherence.

---

<a name="adr-006-multi-tenancy-isolation"></a>
## ADR-006: Multi-Tenancy Isolation & Prompt Injection Defenses

### Multi-Tenancy Design
1. **Relational Layer**: Every document, version, and chunk possesses a mandatory `tenant_id` column. All queries executed by `DocumentRepository` inject `tenant_id == active_tenant`.
2. **Vector Layer**: Similarity searches pass metadata filters `{"tenant_id": active_tenant}`, restricting HNSW graph traversal to chunks belonging exclusively to the caller's organization.
3. **Cross-Tenant Security Validation**: Accessing another tenant's document returns HTTP 403 `FORBIDDEN` via `ForbiddenError`.

### Prompt Injection Guardrails
`PromptGuard` runs at Stage 1 before retrieval or generation:
1. **Delimiters & Control Tokens**: Detects and rejects sequences such as `system:`, `<|im_start|>`, `[INST]`, `ignore previous instructions`, `bypass safety protocols`.
2. **Context Boundary Isolation**: In `ContextBuilder`, context documents are wrapped in distinct XML-style markdown delimiters (`<context_document id="..."> ... </context_document>`), instructing the model that contents inside these blocks are passive reference data, not executable instructions.

---

<a name="production-failure-modes--mitigation-matrix"></a>
## Production Failure Modes & Mitigation Matrix (10 Scenarios)

| # | Failure Mode | Root Cause | Concrete Architectural Mitigation in Codebase |
| :--- | :--- | :--- | :--- |
| **1** | **Reranker Timeout / Crash** | GPU memory pressure or external reranker API latency spike. | `RAGOrchestratorService` catches exception, logs alert, sets `rerank_fallback=True`, and uses retrieval top-k. Returns 200 OK. |
| **2** | **Embedding Provider 502/429** | Upstream embedding endpoint rate-limited or unavailable. | Exponential backoff retry loop with jitter. If exhausted, raises `EmbeddingError` (502) or `EmbeddingTimeoutError` (504) without database corruption. |
| **3** | **Zero-Match / Irrelevant Retrieval** | Query concepts do not exist in indexed corpus. | Score threshold prunes irrelevant candidates. If retrieved pool is empty, `GroundingValidator` sets `insufficient_context=True`, returns conservative refusal. |
| **4** | **Hallucinated Generation** | LLM generates assertions unsupported by retrieved context. | `DeterministicClaimEvaluator` flags unsupported sentences. Grounding score drops to 0.0, triggering refusal substitution. |
| **5** | **Out-of-Bounds Citations** | LLM invents non-existent citations (e.g. `[99]`). | Citation resolution validates all declared citations against active context documents. Invalid citations recorded in `citation_errors`. |
| **6** | **Decompression Bomb Upload** | Malicious ZIP or DOCX with tiny compressed size expanding to gigabytes. | `UploadSecurityGuard` enforces `max_decompression_ratio=20.0` and `max_uncompressed_bytes=50MB`. Rejects upload with HTTP 400. |
| **7** | **Database Transaction Drop** | Network partition or constraint conflict during document ingestion. | Async SQLAlchemy session auto-rollbacks on exception; foreign keys cascade clean deletions. No orphan records left in database. |
| **8** | **Prompt Injection Hijack** | Malicious user query attempts to override system prompt. | `PromptGuard` sanitizes input and rejects adversarial patterns with HTTP 400 (`SECURITY_VALIDATION_ERROR`). |
| **9** | **Cross-Tenant Data Snooping** | Tenant A attempts to retrieve or query Tenant B's documents. | Mandatory `tenant_id` query scoping at both PostgreSQL and Vector index layers. Mismatched access raises HTTP 403 `FORBIDDEN`. |
| **10**| **Context Window Overflow** | High candidate count or long documents exceeding LLM token budget. | `ContextBuilder` tracks exact running token counts, bounds context at `max_tokens`, drops excess candidates, and logs `dropped_chunks_count`. |
