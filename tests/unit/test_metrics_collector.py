"""Unit tests for InMemoryMetricsCollector and Prometheus OpenMetrics exporter."""

from app.observability.metrics import (
    InMemoryMetricsCollector,
    NoOpMetricsCollector,
)


def test_counter_increments_and_tags() -> None:
    collector = InMemoryMetricsCollector()

    collector.increment_counter(
        "requests_total", value=1.0, tags={"method": "GET", "status": "200"}
    )
    collector.increment_counter(
        "requests_total", value=2.0, tags={"method": "GET", "status": "200"}
    )
    collector.increment_counter(
        "requests_total", value=1.0, tags={"method": "POST", "status": "201"}
    )

    assert collector.get_counter("requests_total", {"method": "GET", "status": "200"}) == 3.0
    assert collector.get_counter("requests_total", {"method": "POST", "status": "201"}) == 1.0
    assert collector.get_counter("requests_total", {"method": "DELETE", "status": "200"}) == 0.0


def test_gauge_recording() -> None:
    collector = InMemoryMetricsCollector()

    collector.record_gauge("active_connections", value=10.0, tags={"pool": "primary"})
    assert collector.get_gauge("active_connections", {"pool": "primary"}) == 10.0

    collector.record_gauge("active_connections", value=5.0, tags={"pool": "primary"})
    assert collector.get_gauge("active_connections", {"pool": "primary"}) == 5.0
    assert collector.get_gauge("active_connections", {"pool": "secondary"}) is None


def test_histogram_percentiles_and_aggregates() -> None:
    collector = InMemoryMetricsCollector()

    # Record 100 latency samples: 1.0ms, 2.0ms, ... 100.0ms
    for i in range(1, 101):
        collector.record_histogram("latency_ms", value=float(i), tags={"route": "search"})

    summary = collector.get_histogram_summary("latency_ms", {"route": "search"})

    assert summary["count"] == 100.0
    assert summary["sum"] == 5050.0
    assert summary["mean"] == 50.5
    assert summary["min"] == 1.0
    assert summary["max"] == 100.0
    assert summary["p50"] == 50.5
    assert summary["p90"] == 90.1
    assert summary["p95"] == 95.05
    assert summary["p99"] == 99.01


def test_prometheus_openmetrics_text_format() -> None:
    collector = InMemoryMetricsCollector()

    collector.increment_counter("rag_requests_total", value=42.0, tags={"endpoint": "/query"})
    collector.record_gauge("rag_active_workers", value=4.0)
    collector.record_histogram("rag_request_latency_ms", value=150.0, tags={"endpoint": "/query"})

    text = collector.generate_prometheus_text()

    assert "# HELP rag_requests_total" in text
    assert "# TYPE rag_requests_total counter" in text
    assert 'rag_requests_total{endpoint="/query"} 42.0' in text

    assert "# HELP rag_active_workers" in text
    assert "# TYPE rag_active_workers gauge" in text
    assert "rag_active_workers 4.0" in text

    assert "# HELP rag_request_latency_ms" in text
    assert "# TYPE rag_request_latency_ms summary" in text
    assert 'rag_request_latency_ms_count{endpoint="/query"} 1' in text
    assert 'rag_request_latency_ms_sum{endpoint="/query"} 150.00' in text
    assert 'rag_request_latency_ms{endpoint="/query",quantile="0.95"} 150.00' in text


def test_metrics_collector_reset() -> None:
    collector = InMemoryMetricsCollector()
    collector.increment_counter("counter_a", 5.0)
    collector.reset()
    assert collector.get_counter("counter_a") == 0.0


def test_noop_metrics_collector() -> None:
    noop = NoOpMetricsCollector()
    noop.increment_counter("any_counter")
    noop.record_gauge("any_gauge", 1.0)
    noop.record_histogram("any_hist", 10.0)
    assert "# Metrics disabled" in noop.generate_prometheus_text()
