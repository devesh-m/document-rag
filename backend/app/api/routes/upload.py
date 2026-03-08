from fastapi import APIRouter, Depends, File, UploadFile, status

from app.api.dependencies import get_ingestion_service
from app.schemas import UploadResponse
from app.services.ingestion_service import IngestionService

router = APIRouter()


@router.post("", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile = File(...),
    service: IngestionService = Depends(get_ingestion_service),
) -> UploadResponse:
    file_bytes = await file.read()
    return await service.ingest_document(file.filename, file_bytes)
