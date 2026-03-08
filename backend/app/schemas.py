from datetime import datetime

from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    id: int
    filename: str
    upload_time: datetime
    is_deleted: bool


class UploadResponse(BaseModel):
    message: str
    document: DocumentResponse | None = None
    page_count: int = 0
    chunk_count: int = 0


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2_000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class CitationResponse(BaseModel):
    chunk_id: int
    document_id: int
    filename: str
    page_number: int | None = None
    snippet: str
    similarity_score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    retrieved_chunks: int


class DeleteDocumentResponse(BaseModel):
    message: str
    document_id: int


class HealthResponse(BaseModel):
    status: str
    app_name: str
    environment: str


class ErrorResponse(BaseModel):
    detail: str
