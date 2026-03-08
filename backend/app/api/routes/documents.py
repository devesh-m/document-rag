from fastapi import APIRouter, Depends

from app.api.dependencies import get_document_service
from app.schemas import DeleteDocumentResponse, DocumentResponse
from app.services.document_service import DocumentService

router = APIRouter()


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    service: DocumentService = Depends(get_document_service),
) -> list[DocumentResponse]:
    return await service.list_documents()


@router.delete("/{document_id}", response_model=DeleteDocumentResponse)
async def delete_document(
    document_id: int,
    service: DocumentService = Depends(get_document_service),
) -> DeleteDocumentResponse:
    return await service.delete_document(document_id)
