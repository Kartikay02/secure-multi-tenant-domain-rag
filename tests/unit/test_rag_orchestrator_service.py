"""Unit tests for RAGOrchestratorService using dependency-injected mocks."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.exceptions import PipelineStageError, QueryValidationError
from app.core.logging import set_correlation_id
from app.rag.context.domain import AssembledContext, ContextDocument
from app.rag.generation.domain import GeneratedResponse, GenerationParameters
from app.rag.orchestration.domain import RAGRequest
from app.rag.validation.domain import ValidationResult
from app.rag.vector.domain import RetrievalResult
from app.services.rag_service import RAGOrchestratorService, create_rag_orchestrator


@pytest.fixture
def mock_retrieval_results() -> list[RetrievalResult]:
    doc_id = uuid.uuid4()
    return [
        RetrievalResult(
            chunk_id=uuid.uuid4(),
            document_id=doc_id,
            text="HNSW indexes provide approximate nearest neighbor search with logarithmic scaling.",
            score=0.92,
            metadata={"title": "Vector Search Docs", "page_number": 1, "source_id": "doc1.pdf"},
        ),
        RetrievalResult(
            chunk_id=uuid.uuid4(),
            document_id=doc_id,
            text="PostgreSQL pgvector extension enables cosine and inner-product distance metrics.",
            score=0.88,
            metadata={"title": "Vector Search Docs", "page_number": 2, "source_id": "doc1.pdf"},
        ),
    ]


@pytest.fixture
def mock_context_documents(mock_retrieval_results: list[RetrievalResult]) -> list[ContextDocument]:
    return [
        ContextDocument(
            citation_id=1,
            citation_label="[1]",
            chunk_id=mock_retrieval_results[0].chunk_id,
            document_id=mock_retrieval_results[0].document_id,
            source_id="doc1.pdf",
            title="Vector Search Docs",
            page_number=1,
            score=0.92,
            content=mock_retrieval_results[0].text,
            token_count=18,
        ),
        ContextDocument(
            citation_id=2,
            citation_label="[2]",
            chunk_id=mock_retrieval_results[1].chunk_id,
            document_id=mock_retrieval_results[1].document_id,
            source_id="doc1.pdf",
            title="Vector Search Docs",
            page_number=2,
            score=0.88,
            content=mock_retrieval_results[1].text,
            token_count=16,
        ),
    ]


@pytest.fixture
def mock_assembled_context(mock_context_documents: list[ContextDocument]) -> AssembledContext:
    doc_map = {doc.citation_id: doc for doc in mock_context_documents}
    return AssembledContext(
        formatted_context="[1] HNSW indexes provide...\n\n[2] PostgreSQL pgvector...",
        documents=mock_context_documents,
        citation_map=doc_map,
        total_tokens=34,
        total_chunks=2,
        truncated=False,
        dropped_chunks_count=0,
    )


class TestRAGOrchestratorService:
    """Comprehensive test suite for RAGOrchestratorService dependency injection and stage workflows."""

    @pytest.mark.asyncio
    async def test_full_pipeline_happy_path(
        self,
        mock_retrieval_results: list[RetrievalResult],
        mock_assembled_context: AssembledContext,
    ) -> None:
        """Verify end-to-end execution through all 6 stages with latency capture and correlation ID."""
        # 1. Setup Stage Mocks
        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock(return_value=mock_retrieval_results)

        mock_reranker = MagicMock()
        mock_reranker.rerank = AsyncMock(return_value=mock_retrieval_results)

        mock_builder = MagicMock()
        mock_builder.build_context = MagicMock(return_value=mock_assembled_context)

        mock_gen_service = MagicMock()
        mock_gen_service.generate_answer = AsyncMock(
            return_value=GeneratedResponse(
                answer="HNSW enables logarithmic approximate nearest neighbor search [1].",
                citations=[1],
                confidence_score=0.95,
                grounded=True,
                insufficient_context=False,
                model="gpt-4o-mini",
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
            )
        )

        mock_validator = MagicMock()
        mock_validator.validate = MagicMock(
            return_value=ValidationResult(
                grounded=True,
                confidence_score=0.95,
                final_answer="HNSW enables logarithmic approximate nearest neighbor search [1].",
                fallback_applied=False,
                insufficient_context=False,
                unsupported_claims=[],
                citation_errors=[],
            )
        )

        orchestrator = RAGOrchestratorService(
            retriever=mock_retriever,
            reranker=mock_reranker,
            context_builder=mock_builder,
            generation_service=mock_gen_service,
            grounding_validator=mock_validator,
        )

        # 2. Execute
        response = await orchestrator.execute(
            query="  What is HNSW?  ",
            request_id="test-corr-123",
        )

        # 3. Assertions
        assert response.query == "What is HNSW?"
        assert response.raw_query == "  What is HNSW?  "
        assert (
            response.answer == "HNSW enables logarithmic approximate nearest neighbor search [1]."
        )
        assert response.grounded is True
        assert response.confidence_score == 0.95
        assert response.insufficient_context is False
        assert response.fallback_applied is False
        assert response.citations == [1]
        assert len(response.referenced_documents) == 1
        assert response.referenced_documents[0].citation_id == 1
        assert response.request_id == "test-corr-123"
        assert response.model == "gpt-4o-mini"
        assert response.retrieved_chunks_count == 2
        assert response.context_chunks_count == 2

        # Assert stage latencies
        assert response.stage_latencies.preprocessing_ms >= 0.0
        assert response.stage_latencies.retrieval_ms >= 0.0
        assert response.stage_latencies.reranking_ms >= 0.0
        assert response.stage_latencies.context_assembly_ms >= 0.0
        assert response.stage_latencies.generation_ms >= 0.0
        assert response.stage_latencies.validation_ms >= 0.0
        assert response.stage_latencies.total_ms >= 0.0

        # Verify mocks were called
        mock_retriever.retrieve.assert_awaited_once()
        mock_reranker.rerank.assert_awaited_once()
        mock_builder.build_context.assert_called_once()
        mock_gen_service.generate_answer.assert_awaited_once()
        mock_validator.validate.assert_called_once()

    @pytest.mark.asyncio
    async def test_correlation_id_propagation_and_fallback(
        self,
        mock_retrieval_results: list[RetrievalResult],
        mock_assembled_context: AssembledContext,
    ) -> None:
        """Verify correlation ID is set from context or auto-generated if missing."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock(return_value=mock_retrieval_results)
        mock_builder = MagicMock()
        mock_builder.build_context = MagicMock(return_value=mock_assembled_context)
        mock_gen_service = MagicMock()
        mock_gen_service.generate_answer = AsyncMock(
            return_value=GeneratedResponse(answer="Answer", citations=[])
        )

        orchestrator = RAGOrchestratorService(
            retriever=mock_retriever,
            context_builder=mock_builder,
            generation_service=mock_gen_service,
        )

        # Case 1: auto-generated UUID
        set_correlation_id("")
        res1 = await orchestrator.execute(query="What is vector search?")
        assert res1.request_id != ""
        assert len(res1.request_id) == 32  # hex format

        # Case 2: from context var
        set_correlation_id("existing-ctx-id-456")
        res2 = await orchestrator.execute(query="What is vector search?")
        assert res2.request_id == "existing-ctx-id-456"

    @pytest.mark.asyncio
    async def test_reranker_disabled_by_parameter(
        self,
        mock_retrieval_results: list[RetrievalResult],
        mock_assembled_context: AssembledContext,
    ) -> None:
        """Verify reranker is skipped when enable_reranking=False."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock(return_value=mock_retrieval_results)
        mock_reranker = MagicMock()
        mock_reranker.rerank = AsyncMock()
        mock_builder = MagicMock()
        mock_builder.build_context = MagicMock(return_value=mock_assembled_context)
        mock_gen_service = MagicMock()
        mock_gen_service.generate_answer = AsyncMock(
            return_value=GeneratedResponse(answer="Answer [1]", citations=[1])
        )

        orchestrator = RAGOrchestratorService(
            retriever=mock_retriever,
            reranker=mock_reranker,
            context_builder=mock_builder,
            generation_service=mock_gen_service,
        )

        response = await orchestrator.execute(
            query="Test query",
            enable_reranking=False,
            rerank_top_k=2,
        )

        mock_reranker.rerank.assert_not_awaited()
        # Verify candidate list passed directly to context builder
        mock_builder.build_context.assert_called_once()
        candidates_passed = mock_builder.build_context.call_args[1]["candidates"]
        assert len(candidates_passed) == 2
        assert response.metadata.get("reranker_applied") is False

    @pytest.mark.asyncio
    async def test_reranker_graceful_fallback_on_failure(
        self,
        mock_retrieval_results: list[RetrievalResult],
        mock_assembled_context: AssembledContext,
    ) -> None:
        """Verify that a reranker exception does not fail the pipeline but falls back to retrieval results."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock(return_value=mock_retrieval_results)
        mock_reranker = MagicMock()
        mock_reranker.rerank = AsyncMock(side_effect=RuntimeError("Reranker connection timeout"))
        mock_builder = MagicMock()
        mock_builder.build_context = MagicMock(return_value=mock_assembled_context)
        mock_gen_service = MagicMock()
        mock_gen_service.generate_answer = AsyncMock(
            return_value=GeneratedResponse(answer="Answer without reranker.", citations=[])
        )

        orchestrator = RAGOrchestratorService(
            retriever=mock_retriever,
            reranker=mock_reranker,
            context_builder=mock_builder,
            generation_service=mock_gen_service,
        )

        response = await orchestrator.execute(query="Test query", enable_reranking=True)

        assert response.answer == "Answer without reranker."
        assert response.metadata.get("rerank_fallback") is True
        assert "Reranker connection timeout" in response.metadata.get("rerank_error", "")

    @pytest.mark.asyncio
    async def test_grounding_validation_disabled_by_parameter(
        self,
        mock_retrieval_results: list[RetrievalResult],
        mock_assembled_context: AssembledContext,
    ) -> None:
        """Verify grounding validator is skipped when enable_grounding=False."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock(return_value=mock_retrieval_results)
        mock_builder = MagicMock()
        mock_builder.build_context = MagicMock(return_value=mock_assembled_context)
        mock_gen_service = MagicMock()
        mock_gen_service.generate_answer = AsyncMock(
            return_value=GeneratedResponse(answer="Raw generated answer.", citations=[])
        )
        mock_validator = MagicMock()
        mock_validator.validate = MagicMock()

        orchestrator = RAGOrchestratorService(
            retriever=mock_retriever,
            context_builder=mock_builder,
            generation_service=mock_gen_service,
            grounding_validator=mock_validator,
            mandatory_grounding=False,
        )

        response = await orchestrator.execute(query="Test query", enable_grounding=False)

        mock_validator.validate.assert_not_called()
        assert response.answer == "Raw generated answer."

    @pytest.mark.asyncio
    async def test_grounding_validation_fallback_applied(
        self,
        mock_retrieval_results: list[RetrievalResult],
        mock_assembled_context: AssembledContext,
    ) -> None:
        """Verify that when validator triggers conservative fallback, the answer is replaced with refusal."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock(return_value=mock_retrieval_results)
        mock_builder = MagicMock()
        mock_builder.build_context = MagicMock(return_value=mock_assembled_context)
        mock_gen_service = MagicMock()
        mock_gen_service.generate_answer = AsyncMock(
            return_value=GeneratedResponse(answer="Hallucinated claims [99].", citations=[99])
        )
        mock_validator = MagicMock()
        mock_validator.validate = MagicMock(
            return_value=ValidationResult(
                grounded=False,
                confidence_score=0.20,
                final_answer="I cannot answer this question based on available documentation.",
                fallback_applied=True,
                insufficient_context=True,
                unsupported_claims=["Hallucinated claims [99]."],
                citation_errors=["Citation [99] does not exist in context."],
            )
        )

        orchestrator = RAGOrchestratorService(
            retriever=mock_retriever,
            context_builder=mock_builder,
            generation_service=mock_gen_service,
            grounding_validator=mock_validator,
        )

        response = await orchestrator.execute(query="Test query")

        assert response.grounded is False
        assert response.fallback_applied is True
        assert response.answer == "I cannot answer this question based on available documentation."
        assert response.citations == []
        assert response.referenced_documents == []
        assert len(response.unsupported_claims) == 1
        assert len(response.citation_errors) == 1

    @pytest.mark.asyncio
    async def test_configurable_runtime_overrides(
        self,
        mock_retrieval_results: list[RetrievalResult],
        mock_assembled_context: AssembledContext,
    ) -> None:
        """Verify that runtime parameter overrides are correctly forwarded to every stage."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock(return_value=mock_retrieval_results)
        mock_reranker = MagicMock()
        mock_reranker.rerank = AsyncMock(return_value=mock_retrieval_results)
        mock_builder = MagicMock()
        mock_builder.build_context = MagicMock(return_value=mock_assembled_context)
        mock_gen_service = MagicMock()
        mock_gen_service.generate_answer = AsyncMock(
            return_value=GeneratedResponse(answer="Answer", citations=[])
        )

        orchestrator = RAGOrchestratorService(
            retriever=mock_retriever,
            reranker=mock_reranker,
            context_builder=mock_builder,
            generation_service=mock_gen_service,
        )

        custom_params = GenerationParameters(temperature=0.7, max_tokens=512)
        await orchestrator.execute(
            query="Test query",
            retrieval_top_k=25,
            score_threshold=0.75,
            rerank_top_k=8,
            max_context_tokens=1500,
            max_context_chunks=4,
            generation_parameters=custom_params,
            filter_metadata={"department": "engineering"},
        )

        mock_retriever.retrieve.assert_awaited_once_with(
            query="Test query",
            top_k=25,
            score_threshold=0.75,
            filter_metadata={"department": "engineering"},
        )
        mock_reranker.rerank.assert_awaited_once_with(
            query="Test query",
            candidates=mock_retrieval_results,
            top_k=8,
        )
        mock_builder.build_context.assert_called_once_with(
            query="Test query",
            candidates=mock_retrieval_results,
            max_tokens=1500,
            max_chunks=4,
        )
        mock_gen_service.generate_answer.assert_awaited_once_with(
            query="Test query",
            context=mock_assembled_context,
            parameters=custom_params,
        )

    @pytest.mark.asyncio
    async def test_execute_request_payload(
        self,
        mock_retrieval_results: list[RetrievalResult],
        mock_assembled_context: AssembledContext,
    ) -> None:
        """Verify executing via RAGRequest object."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock(return_value=mock_retrieval_results)
        mock_builder = MagicMock()
        mock_builder.build_context = MagicMock(return_value=mock_assembled_context)
        mock_gen_service = MagicMock()
        mock_gen_service.generate_answer = AsyncMock(
            return_value=GeneratedResponse(answer="Answer", citations=[])
        )

        orchestrator = RAGOrchestratorService(
            retriever=mock_retriever,
            context_builder=mock_builder,
            generation_service=mock_gen_service,
        )

        request = RAGRequest(
            query="What is vector indexing?",
            request_id="req-payload-123",
            retrieval_top_k=7,
        )
        response = await orchestrator.execute_request(request)

        assert response.query == "What is vector indexing?"
        assert response.request_id == "req-payload-123"
        mock_retriever.retrieve.assert_awaited_once_with(
            query="What is vector indexing?",
            top_k=7,
            score_threshold=None,
            filter_metadata=None,
        )

    @pytest.mark.asyncio
    async def test_stage_failure_query_preprocessing(self) -> None:
        """Verify that an empty or whitespace query raises QueryValidationError."""
        mock_retriever = MagicMock()
        mock_builder = MagicMock()
        mock_gen_service = MagicMock()

        orchestrator = RAGOrchestratorService(
            retriever=mock_retriever,
            context_builder=mock_builder,
            generation_service=mock_gen_service,
        )

        with pytest.raises(QueryValidationError) as exc_info:
            await orchestrator.execute(query="   \n\t  ")
        assert exc_info.value.error_code == "QUERY_VALIDATION_ERROR"

    @pytest.mark.asyncio
    async def test_stage_failure_retrieval(self) -> None:
        """Verify that a retrieval backend failure raises PipelineStageError with stage='retrieval'."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock(side_effect=ConnectionError("Database unreachable"))
        mock_builder = MagicMock()
        mock_gen_service = MagicMock()

        orchestrator = RAGOrchestratorService(
            retriever=mock_retriever,
            context_builder=mock_builder,
            generation_service=mock_gen_service,
        )

        with pytest.raises(PipelineStageError) as exc_info:
            await orchestrator.execute(query="Valid query", request_id="fail-req-1")

        assert exc_info.value.stage == "retrieval"
        assert exc_info.value.request_id == "fail-req-1"
        assert "Database unreachable" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_stage_failure_context_assembly(
        self,
        mock_retrieval_results: list[RetrievalResult],
    ) -> None:
        """Verify that a context builder failure raises PipelineStageError with stage='context_assembly'."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock(return_value=mock_retrieval_results)
        mock_builder = MagicMock()
        mock_builder.build_context = MagicMock(side_effect=ValueError("Token bounding failed"))
        mock_gen_service = MagicMock()

        orchestrator = RAGOrchestratorService(
            retriever=mock_retriever,
            context_builder=mock_builder,
            generation_service=mock_gen_service,
        )

        with pytest.raises(PipelineStageError) as exc_info:
            await orchestrator.execute(query="Valid query")

        assert exc_info.value.stage == "context_assembly"

    @pytest.mark.asyncio
    async def test_stage_failure_generation(
        self,
        mock_retrieval_results: list[RetrievalResult],
        mock_assembled_context: AssembledContext,
    ) -> None:
        """Verify that an LLM failure raises PipelineStageError with stage='generation'."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock(return_value=mock_retrieval_results)
        mock_builder = MagicMock()
        mock_builder.build_context = MagicMock(return_value=mock_assembled_context)
        mock_gen_service = MagicMock()
        mock_gen_service.generate_answer = AsyncMock(side_effect=TimeoutError("LLM API timed out"))

        orchestrator = RAGOrchestratorService(
            retriever=mock_retriever,
            context_builder=mock_builder,
            generation_service=mock_gen_service,
        )

        with pytest.raises(PipelineStageError) as exc_info:
            await orchestrator.execute(query="Valid query")

        assert exc_info.value.stage == "generation"
        assert "LLM API timed out" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_empty_retrieval_candidates(self) -> None:
        """Verify handling when retrieval yields 0 candidate chunks."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock(return_value=[])

        empty_context = AssembledContext(
            formatted_context="",
            documents=[],
            citation_map={},
            total_tokens=0,
            total_chunks=0,
            truncated=False,
            dropped_chunks_count=0,
        )
        mock_builder = MagicMock()
        mock_builder.build_context = MagicMock(return_value=empty_context)

        mock_gen_service = MagicMock()
        mock_gen_service.generate_answer = AsyncMock(
            return_value=GeneratedResponse(
                answer="I cannot answer this question based on available documentation.",
                citations=[],
                insufficient_context=True,
            )
        )

        orchestrator = RAGOrchestratorService(
            retriever=mock_retriever,
            context_builder=mock_builder,
            generation_service=mock_gen_service,
        )

        response = await orchestrator.execute(query="Unknown topic query")

        assert response.retrieved_chunks_count == 0
        assert response.context_chunks_count == 0
        assert response.insufficient_context is True
        assert response.citations == []
        assert response.referenced_documents == []

    def test_factory_function(self) -> None:
        """Verify create_rag_orchestrator factory constructs service with defaults."""
        mock_retriever = MagicMock()
        orchestrator = create_rag_orchestrator(retriever=mock_retriever)

        assert isinstance(orchestrator, RAGOrchestratorService)
        assert orchestrator.retriever is mock_retriever
        assert orchestrator.default_enable_reranking is True
        assert orchestrator.mandatory_grounding is True

    @pytest.mark.asyncio
    async def test_client_cannot_disable_mandatory_grounding(
        self,
        mock_retrieval_results: list[RetrievalResult],
        mock_assembled_context: AssembledContext,
    ) -> None:
        """Verify client enable_grounding=False cannot disable mandatory server grounding."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock(return_value=mock_retrieval_results)
        mock_builder = MagicMock()
        mock_builder.build_context = MagicMock(return_value=mock_assembled_context)
        mock_gen_service = MagicMock()
        mock_gen_service.generate_answer = AsyncMock(
            return_value=GeneratedResponse(
                answer="PostgreSQL WAL replication is active [1].",
                citations=[1],
                grounded=True,
                confidence_score=0.9,
            )
        )
        mock_validator = MagicMock()
        mock_validator.validate = MagicMock(
            return_value=ValidationResult(
                grounded=True,
                confidence_score=0.95,
                final_answer="Validated answer [1].",
                fallback_applied=False,
                insufficient_context=False,
            )
        )

        orchestrator = RAGOrchestratorService(
            retriever=mock_retriever,
            context_builder=mock_builder,
            generation_service=mock_gen_service,
            grounding_validator=mock_validator,
            mandatory_grounding=True,
        )

        # Client explicitly passes enable_grounding=False
        response = await orchestrator.execute(query="Test query", enable_grounding=False)

        # Grounding validator MUST still have been invoked
        mock_validator.validate.assert_called_once()
        assert response.grounded is True
        assert response.answer == "Validated answer [1]."

    @pytest.mark.asyncio
    async def test_empty_context_forces_ungrounded_in_orchestrator(self) -> None:
        """Verify empty context is strictly marked ungrounded with 0 confidence."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock(return_value=[])

        empty_context = AssembledContext(
            formatted_context="",
            documents=[],
            citation_map={},
            total_tokens=0,
            total_chunks=0,
            truncated=False,
            dropped_chunks_count=0,
        )
        mock_builder = MagicMock()
        mock_builder.build_context = MagicMock(return_value=empty_context)

        mock_gen_service = MagicMock()
        mock_gen_service.generate_answer = AsyncMock(
            return_value=GeneratedResponse(
                answer="An ungrounded claim without context.",
                citations=[],
                grounded=True,  # LLM erroneously claimed grounded
                confidence_score=0.99,
            )
        )

        orchestrator = RAGOrchestratorService(
            retriever=mock_retriever,
            context_builder=mock_builder,
            generation_service=mock_gen_service,
            grounding_validator=None,
            mandatory_grounding=False,
        )

        # Client also requests enable_grounding=False
        response = await orchestrator.execute(query="Test query", enable_grounding=False)

        # Server-side policy MUST force grounded=False, confidence_score=0.0
        assert response.grounded is False
        assert response.confidence_score == 0.0
        assert response.insufficient_context is True

    @pytest.mark.asyncio
    async def test_streaming_mandatory_grounding_cannot_be_disabled(
        self,
        mock_retrieval_results: list[RetrievalResult],
        mock_assembled_context: AssembledContext,
    ) -> None:
        """Verify streaming query enforces mandatory grounding even if client passes False."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock(return_value=mock_retrieval_results)
        mock_builder = MagicMock()
        mock_builder.build_context = MagicMock(return_value=mock_assembled_context)

        async def _mock_stream(*args, **kwargs):
            yield "Token 1 "
            yield "Token 2"

        mock_gen_service = MagicMock()
        mock_gen_service.stream_answer = _mock_stream

        mock_validator = MagicMock()
        mock_validator.validate = MagicMock(
            return_value=ValidationResult(
                grounded=True,
                confidence_score=0.92,
                final_answer="Validated stream answer",
                fallback_applied=False,
                insufficient_context=False,
            )
        )

        orchestrator = RAGOrchestratorService(
            retriever=mock_retriever,
            context_builder=mock_builder,
            generation_service=mock_gen_service,
            grounding_validator=mock_validator,
            mandatory_grounding=True,
        )

        events = []
        async for evt in orchestrator.execute_stream(
            query="Streaming query", enable_grounding=False
        ):
            events.append(evt)

        mock_validator.validate.assert_called_once()
        done_event = [e for e in events if e.get("type") in ("complete", "done")][0]
        assert done_event["grounded"] is True
        assert done_event["confidence_score"] == 0.92
