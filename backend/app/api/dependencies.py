from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db_session
from app.services.document_service import DocumentService
from app.services.ingestion_service import IngestionService
from app.services.query_service import QueryService


async def get_document_service(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> DocumentService:
    return DocumentService(
        session=session,
        vector_store=request.app.state.vector_store,
    )


async def get_ingestion_service(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> IngestionService:
    return IngestionService(
        session=session,
        parser=request.app.state.pdf_parser,
        chunker=request.app.state.chunker,
        embedder=request.app.state.embedder,
        vector_store=request.app.state.vector_store,
        settings=request.app.state.settings,
    )


async def get_query_service(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> QueryService:
    return QueryService(
        session=session,
        embedder=request.app.state.embedder,
        vector_store=request.app.state.vector_store,
        prompt_builder=request.app.state.prompt_builder,
        ollama_client=request.app.state.ollama_client,
        settings=request.app.state.settings,
    )
