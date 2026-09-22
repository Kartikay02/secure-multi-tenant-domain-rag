"""Production observability dashboard recommendations, SLI/SLO definitions, and alerting rules."""

from typing import Any

# Production Service Level Indicators (SLIs) and Objectives (SLOs)
PRODUCTION_SLOS: dict[str, dict[str, str]] = {
    "Availability": {
        "target": "99.9% success rate",
        "description": "Fraction of HTTP /query requests completing with status < 500",
        "sli": "sum(rate(rag_requests_total{status=~'2..'}[5m])) / sum(rate(rag_requests_total[5m]))",
    },
    "Latency_p95": {
        "target": "< 2500ms at p95",
        "description": "95th percentile end-to-end question-answering duration",
        "sli": "rag_request_latency_ms{quantile='0.95', endpoint='/api/v1/query'}",
    },
    "Grounding_Pass_Rate": {
        "target": ">= 98.0%",
        "description": "Percentage of generated answers verified as fully grounded in evidence",
        "sli": "1 - (sum(rate(rag_grounding_failures_total[5m])) / sum(rate(rag_requests_total{endpoint='rag_query'}[5m])))",
    },
    "Insufficient_Context_Rate": {
        "target": "< 5.0%",
        "description": "Percentage of user queries failing to find adequate source documentation",
        "sli": "sum(rate(rag_insufficient_context_total[1h])) / sum(rate(rag_requests_total{endpoint='rag_query'}[1h]))",
    },
}

# Recommended PromQL Alerting Rules
PROMETHEUS_ALERT_RULES: list[dict[str, Any]] = [
    {
        "alert": "RAGHighErrorRate",
        "expr": "sum(rate(rag_errors_total[5m])) / sum(rate(rag_requests_total[5m])) > 0.02",
        "for": "2m",
        "labels": {"severity": "critical"},
        "annotations": {
            "summary": "RAG HTTP error rate exceeds 2% over 5m",
            "description": "High failure rate indicates database downtime, external LLM provider throttling, or unhandled exceptions.",
        },
    },
    {
        "alert": "RAGHighP95Latency",
        "expr": "rag_request_latency_ms{quantile='0.95', endpoint='/api/v1/query'} > 3500",
        "for": "5m",
        "labels": {"severity": "warning"},
        "annotations": {
            "summary": "RAG query p95 latency exceeds 3.5s",
            "description": "Check retrieval and LLM generation stage latencies to identify pipeline bottlenecks.",
        },
    },
    {
        "alert": "RAGGroundingFailureSpike",
        "expr": "rate(rag_grounding_failures_total[10m]) / rate(rag_requests_total{endpoint='rag_query'}[10m]) > 0.05",
        "for": "5m",
        "labels": {"severity": "warning"},
        "annotations": {
            "summary": "Answer grounding failure rate exceeds 5%",
            "description": "Grounding guardrail is rejecting answers frequently. Check retrieval threshold or model hallucination drift.",
        },
    },
    {
        "alert": "RAGTokenConsumptionSpike",
        "expr": "sum(rate(rag_tokens_total[15m])) > 10000",
        "for": "10m",
        "labels": {"severity": "warning"},
        "annotations": {
            "summary": "Unusual surge in LLM token consumption",
            "description": "Token usage rate has spiked, which may lead to rapid API budget depletion.",
        },
    },
]


def get_grafana_dashboard_spec() -> dict[str, Any]:
    """Return recommended Grafana dashboard JSON layout definition."""
    return {
        "title": "Domain RAG System - Production Observability",
        "refresh": "10s",
        "schemaVersion": 36,
        "panels": [
            {
                "title": "Request Throughput & Error Rate",
                "type": "timeseries",
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                "targets": [
                    {
                        "expr": "sum(rate(rag_requests_total[1m])) by (status)",
                        "legendFormat": "HTTP {{status}}",
                    },
                    {
                        "expr": "sum(rate(rag_errors_total[1m])) by (stage)",
                        "legendFormat": "Errors ({{stage}})",
                    },
                ],
            },
            {
                "title": "End-to-End Latency Percentiles (ms)",
                "type": "timeseries",
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                "targets": [
                    {
                        "expr": "rag_request_latency_ms{quantile='0.5'}",
                        "legendFormat": "p50 (median)",
                    },
                    {"expr": "rag_request_latency_ms{quantile='0.9'}", "legendFormat": "p90"},
                    {"expr": "rag_request_latency_ms{quantile='0.95'}", "legendFormat": "p95"},
                    {"expr": "rag_request_latency_ms{quantile='0.99'}", "legendFormat": "p99"},
                ],
            },
            {
                "title": "Pipeline Stage Latency Breakdown (Watermark)",
                "type": "timeseries",
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                "targets": [
                    {
                        "expr": "rag_retrieval_latency_ms{quantile='0.95'}",
                        "legendFormat": "Retrieval (p95)",
                    },
                    {
                        "expr": "rag_reranking_latency_ms{quantile='0.95'}",
                        "legendFormat": "Reranking (p95)",
                    },
                    {
                        "expr": "rag_generation_latency_ms{quantile='0.95'}",
                        "legendFormat": "Generation (p95)",
                    },
                ],
            },
            {
                "title": "Quality & Safety: Grounding & Context Refusals",
                "type": "timeseries",
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                "targets": [
                    {
                        "expr": "rate(rag_grounding_failures_total[5m])",
                        "legendFormat": "Grounding Failures / sec",
                    },
                    {
                        "expr": "rate(rag_insufficient_context_total[5m])",
                        "legendFormat": "Insufficient Context / sec",
                    },
                ],
            },
            {
                "title": "LLM Token Usage & Cost Driver",
                "type": "timeseries",
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16},
                "targets": [
                    {
                        "expr": "sum(rate(rag_tokens_total{type='prompt'}[5m]))",
                        "legendFormat": "Prompt Tokens / sec",
                    },
                    {
                        "expr": "sum(rate(rag_tokens_total{type='completion'}[5m]))",
                        "legendFormat": "Completion Tokens / sec",
                    },
                ],
            },
            {
                "title": "Ingestion Pipeline Latency & Volume",
                "type": "timeseries",
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16},
                "targets": [
                    {
                        "expr": "rag_ingestion_latency_ms{quantile='0.95'}",
                        "legendFormat": "Ingestion Duration (p95)",
                    },
                    {
                        "expr": "rate(rag_chunks_created_total[5m])",
                        "legendFormat": "Chunks Created / sec",
                    },
                ],
            },
        ],
    }
