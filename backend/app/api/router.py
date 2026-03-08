from fastapi import APIRouter

from app.api.routes import documents, health, query, upload

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(upload.router, prefix="/upload", tags=["upload"])
api_router.include_router(query.router, prefix="/query", tags=["query"])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
