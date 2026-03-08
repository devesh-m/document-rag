from fastapi import APIRouter, Depends

from app.api.dependencies import get_query_service
from app.schemas import QueryRequest, QueryResponse
from app.services.query_service import QueryService

router = APIRouter()


@router.post("", response_model=QueryResponse)
async def query_documents(
    payload: QueryRequest,
    service: QueryService = Depends(get_query_service),
) -> QueryResponse:
    return await service.answer_question(payload)
