"""Metrics exposition endpoint for Prometheus scrapers and monitoring systems."""

from fastapi import APIRouter, Response

from app.observability.metrics import get_metrics_collector

router = APIRouter()


@router.get("", response_class=Response, summary="Expose Prometheus metrics")
@router.get("/", response_class=Response, include_in_schema=False)
def get_prometheus_metrics() -> Response:
    """Return application metrics formatted for Prometheus scrapers (OpenMetrics text format)."""
    collector = get_metrics_collector()
    content = collector.generate_prometheus_text()
    return Response(
        content=content,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
