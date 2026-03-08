import logging
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.database.repositories import ChunkRepository, DocumentRepository
from app.exceptions import NotFoundError
from app.rag.vector_store import FAISSVectorStore
from app.schemas import DeleteDocumentResponse, DocumentResponse


logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(self, session: AsyncSession, vector_store: FAISSVectorStore) -> None:
        self.session = session
        self.vector_store = vector_store
        self.document_repository = DocumentRepository(session)
        self.chunk_repository = ChunkRepository(session)

    async def list_documents(self) -> list[DocumentResponse]:
        documents = await self.document_repository.list_documents()
        return [
            DocumentResponse(
                id=document.id,
                filename=document.filename,
                upload_time=document.upload_time,
                is_deleted=document.is_deleted,
            )
            for document in documents
        ]

    async def delete_document(self, document_id: int) -> DeleteDocumentResponse:
        document = await self.document_repository.get_document(document_id)
        if document is None:
            raise NotFoundError(f"Document {document_id} was not found.")

        chunk_ids = await self.chunk_repository.get_document_chunk_ids(document_id)
        await self.vector_store.remove_embeddings(chunk_ids)
        await self.document_repository.soft_delete_document(document)
        await self.session.commit()

        file_path = Path(document.storage_path)
        if file_path.exists():
            file_path.unlink()

        logger.info("Document soft-deleted", extra={"document_id": document_id})
        return DeleteDocumentResponse(
            message="Document deleted from metadata and vector index.",
            document_id=document_id,
        )
