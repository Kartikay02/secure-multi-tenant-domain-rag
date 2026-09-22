"""Aggregated API Version 1 Router."""

from fastapi import APIRouter

from app.api.v1.endpoints import documents, health, metrics, query

api_router = APIRouter()

# Register endpoint sub-routers
api_router.include_router(health.router, tags=["System Health"])
api_router.include_router(metrics.router, prefix="/metrics", tags=["Metrics & Observability"])
api_router.include_router(documents.router, prefix="/documents", tags=["Documents & Ingestion"])
api_router.include_router(query.router, prefix="/query", tags=["RAG Query & Generation"])
