import logging
from pathlib import Path
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.database.repositories import ChunkRepository, DocumentRepository
from app.exceptions import BadRequestError
from app.rag.chunker import TextChunker
from app.rag.embedder import EmbeddingService
from app.rag.pdf_parser import PDFParser
from app.rag.vector_store import FAISSVectorStore
from app.schemas import DocumentResponse, UploadResponse


logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(
        self,
        session: AsyncSession,
        parser: PDFParser,
        chunker: TextChunker,
        embedder: EmbeddingService,
        vector_store: FAISSVectorStore,
        settings: Settings,
    ) -> None:
        self.session = session
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.settings = settings
        self.document_repository = DocumentRepository(session)
        self.chunk_repository = ChunkRepository(session)

    async def ingest_document(self, filename: str, file_bytes: bytes) -> UploadResponse:
        safe_filename = Path(filename).name
        if not safe_filename.lower().endswith(".pdf"):
            raise BadRequestError("Only PDF uploads are supported.")

        pages = self.parser.parse(file_bytes)
        if not pages:
            raise BadRequestError("The uploaded PDF did not contain extractable text.")

        chunks = self.chunker.split_pages(pages)
        if not chunks:
            raise BadRequestError("No chunks could be produced from the uploaded PDF.")

        stored_path = self._build_storage_path(safe_filename)
        stored_path.write_bytes(file_bytes)

        created_chunk_ids: list[int] = []
        try:
            document = await self.document_repository.create_document(
                filename=safe_filename,
                storage_path=str(stored_path),
            )
            chunk_records = await self.chunk_repository.create_chunks(document.id, chunks)
            created_chunk_ids = [chunk.id for chunk in chunk_records]

            embeddings = await self.embedder.embed_texts([chunk.text for chunk in chunks])
            await self.vector_store.add_embeddings(embeddings, created_chunk_ids)
            await self.chunk_repository.assign_vector_indices(chunk_records)
            await self.session.commit()

            logger.info(
                "Document ingested",
                extra={
                    "document_id": document.id,
                    "document_filename": safe_filename,
                    "page_count": len(pages),
                    "chunk_count": len(chunks),
                },
            )
            return UploadResponse(
                message=f"Indexed {safe_filename} into {len(chunks)} chunks.",
                document=DocumentResponse(
                    id=document.id,
                    filename=document.filename,
                    upload_time=document.upload_time,
                    is_deleted=document.is_deleted,
                ),
                page_count=len(pages),
                chunk_count=len(chunks),
            )
        except Exception:
            await self.session.rollback()
            if created_chunk_ids:
                await self.vector_store.remove_embeddings(created_chunk_ids)
            if stored_path.exists():
                stored_path.unlink()
            raise

    def _build_storage_path(self, filename: str) -> Path:
        unique_name = f"{uuid4().hex}_{filename}"
        self.settings.upload_path.mkdir(parents=True, exist_ok=True)
        return self.settings.upload_path / unique_name
