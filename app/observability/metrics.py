"""Metrics collection engine, metric definitions, and Prometheus OpenMetrics exporter."""

import math
from threading import Lock

from app.observability.interfaces import MetricsCollectorProtocol

# Standardized Metric Identifiers
METRIC_REQUESTS_TOTAL = "rag_requests_total"
METRIC_ERRORS_TOTAL = "rag_errors_total"
METRIC_REQUEST_LATENCY_MS = "rag_request_latency_ms"
METRIC_INGESTION_LATENCY_MS = "rag_ingestion_latency_ms"
METRIC_RETRIEVAL_LATENCY_MS = "rag_retrieval_latency_ms"
METRIC_RERANKING_LATENCY_MS = "rag_reranking_latency_ms"
METRIC_GENERATION_LATENCY_MS = "rag_generation_latency_ms"
METRIC_TOKENS_TOTAL = "rag_tokens_total"
METRIC_RETRIEVAL_TOP_K = "rag_retrieval_top_k"
METRIC_GROUNDING_FAILURES_TOTAL = "rag_grounding_failures_total"
METRIC_INSUFFICIENT_CONTEXT_TOTAL = "rag_insufficient_context_total"

METRIC_HELP: dict[str, str] = {
    METRIC_REQUESTS_TOTAL: "Total number of HTTP and pipeline requests processed",
    METRIC_ERRORS_TOTAL: "Total number of pipeline and HTTP errors encountered",
    METRIC_REQUEST_LATENCY_MS: "End-to-end request duration in milliseconds",
    METRIC_INGESTION_LATENCY_MS: "Document ingestion and chunking duration in milliseconds",
    METRIC_RETRIEVAL_LATENCY_MS: "Candidate chunk retrieval stage duration in milliseconds",
    METRIC_RERANKING_LATENCY_MS: "Cross-encoder reranking stage duration in milliseconds",
    METRIC_GENERATION_LATENCY_MS: "LLM answer generation stage duration in milliseconds",
    METRIC_TOKENS_TOTAL: "Total prompt and completion LLM tokens consumed",
    METRIC_RETRIEVAL_TOP_K: "Number of candidate chunks retrieved from index",
    METRIC_GROUNDING_FAILURES_TOTAL: "Total number of queries failing answer grounding verification",
    METRIC_INSUFFICIENT_CONTEXT_TOTAL: "Total number of queries resulting in insufficient context refusal",
}


def _tags_to_key(tags: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
    if not tags:
        return ()
    return tuple(sorted((str(k), str(v)) for k, v in tags.items()))


def _format_tags(tags: tuple[tuple[str, str], ...]) -> str:
    if not tags:
        return ""
    pairs = [f'{k}="{v}"' for k, v in tags]
    return "{" + ",".join(pairs) + "}"


class InMemoryMetricsCollector(MetricsCollectorProtocol):
    """Thread-safe in-memory metrics registry computing running percentiles and Prometheus exposition."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._histograms: dict[tuple[str, tuple[tuple[str, str], ...]], list[float]] = {}

    def increment_counter(
        self,
        name: str,
        value: float = 1.0,
        tags: dict[str, str] | None = None,
    ) -> None:
        key = (name, _tags_to_key(tags))
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + value

    def record_gauge(
        self,
        name: str,
        value: float,
        tags: dict[str, str] | None = None,
    ) -> None:
        key = (name, _tags_to_key(tags))
        with self._lock:
            self._gauges[key] = float(value)

    def record_histogram(
        self,
        name: str,
        value: float,
        tags: dict[str, str] | None = None,
    ) -> None:
        key = (name, _tags_to_key(tags))
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = []
            self._histograms[key].append(float(value))

    def get_counter(self, name: str, tags: dict[str, str] | None = None) -> float:
        key = (name, _tags_to_key(tags))
        with self._lock:
            return self._counters.get(key, 0.0)

    def get_gauge(self, name: str, tags: dict[str, str] | None = None) -> float | None:
        key = (name, _tags_to_key(tags))
        with self._lock:
            return self._gauges.get(key)

    def get_histogram_summary(
        self,
        name: str,
        tags: dict[str, str] | None = None,
    ) -> dict[str, float]:
        key = (name, _tags_to_key(tags))
        with self._lock:
            values = list(self._histograms.get(key, []))

        if not values:
            return {
                "count": 0.0,
                "sum": 0.0,
                "mean": 0.0,
                "p50": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        sorted_vals = sorted(values)
        count = len(sorted_vals)
        total = sum(sorted_vals)

        def _p(p: float) -> float:
            k = (count - 1) * (p / 100.0)
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return sorted_vals[int(k)]
            return sorted_vals[int(f)] * (c - k) + sorted_vals[int(c)] * (k - f)

        return {
            "count": float(count),
            "sum": round(total, 3),
            "mean": round(total / count, 3),
            "p50": round(_p(50.0), 3),
            "p90": round(_p(90.0), 3),
            "p95": round(_p(95.0), 3),
            "p99": round(_p(99.0), 3),
            "min": round(sorted_vals[0], 3),
            "max": round(sorted_vals[-1], 3),
        }

    def generate_prometheus_text(self) -> str:
        """Render all recorded metrics in Prometheus / OpenMetrics line text format."""
        lines: list[str] = []

        with self._lock:
            # Group counters by name
            counter_groups: dict[str, list[tuple[tuple[tuple[str, str], ...], float]]] = {}
            for (name, tags), val in self._counters.items():
                counter_groups.setdefault(name, []).append((tags, val))

            for name, c_entries in sorted(counter_groups.items()):
                help_text = METRIC_HELP.get(name, f"Counter {name}")
                lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} counter")
                for tags, val in sorted(c_entries):
                    tag_str = _format_tags(tags)
                    lines.append(f"{name}{tag_str} {val}")

            # Group gauges by name
            gauge_groups: dict[str, list[tuple[tuple[tuple[str, str], ...], float]]] = {}
            for (name, tags), val in self._gauges.items():
                gauge_groups.setdefault(name, []).append((tags, val))

            for name, g_entries in sorted(gauge_groups.items()):
                help_text = METRIC_HELP.get(name, f"Gauge {name}")
                lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} gauge")
                for tags, val in sorted(g_entries):
                    tag_str = _format_tags(tags)
                    lines.append(f"{name}{tag_str} {val}")

            # Group histograms by name
            hist_groups: dict[str, list[tuple[tuple[tuple[str, str], ...], list[float]]]] = {}
            for (name, tags), vals in self._histograms.items():
                hist_groups.setdefault(name, []).append((tags, vals))

            for name, h_entries in sorted(hist_groups.items()):
                help_text = METRIC_HELP.get(name, f"Histogram summary {name}")
                lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} summary")
                for tags, vals in sorted(h_entries):
                    if not vals:
                        continue
                    sorted_v = sorted(vals)
                    cnt = len(sorted_v)
                    tot = sum(sorted_v)

                    base_tag_dict = dict(tags)
                    for q_val in (0.5, 0.9, 0.95, 0.99):
                        idx = int((cnt - 1) * q_val)
                        q_tags = dict(base_tag_dict)
                        q_tags["quantile"] = str(q_val)
                        tag_str = _format_tags(_tags_to_key(q_tags))
                        lines.append(f"{name}{tag_str} {sorted_v[idx]:.2f}")

                    base_tag_str = _format_tags(tags)
                    lines.append(f"{name}_sum{base_tag_str} {tot:.2f}")
                    lines.append(f"{name}_count{base_tag_str} {cnt}")

        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        """Clear all metric registers (primarily used in test fixtures)."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


class NoOpMetricsCollector(MetricsCollectorProtocol):
    """No-op metrics collector for environments where telemetry is disabled."""

    def increment_counter(
        self, name: str, value: float = 1.0, tags: dict[str, str] | None = None
    ) -> None:
        pass

    def record_gauge(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        pass

    def record_histogram(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        pass

    def generate_prometheus_text(self) -> str:
        return "# Metrics disabled\n"

    def reset(self) -> None:
        pass


_global_metrics_collector: MetricsCollectorProtocol = InMemoryMetricsCollector()


def get_metrics_collector() -> MetricsCollectorProtocol:
    """Retrieve global metrics collector singleton."""
    return _global_metrics_collector


def set_metrics_collector(collector: MetricsCollectorProtocol) -> None:
    """Override global metrics collector instance."""
    global _global_metrics_collector
    _global_metrics_collector = collector
