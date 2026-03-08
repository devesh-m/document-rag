from sqlalchemy import select
from sqlalchemy.orm import joinedload
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Chunk, Document
from app.rag.chunker import TextChunk


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list_documents(self) -> list[Document]:
        result = await self.session.execute(
            select(Document).order_by(Document.upload_time.desc())
        )
        return list(result.scalars().all())

    async def get_document(self, document_id: int) -> Document | None:
        return await self.session.get(Document, document_id)

    async def create_document(self, filename: str, storage_path: str) -> Document:
        document = Document(filename=filename, storage_path=storage_path)
        self.session.add(document)
        await self.session.flush()
        await self.session.refresh(document)
        return document

    async def soft_delete_document(self, document: Document) -> Document:
        document.is_deleted = True
        await self.session.flush()
        await self.session.refresh(document)
        return document


class ChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_chunks(self, document_id: int, chunks: list[TextChunk]) -> list[Chunk]:
        chunk_models = [
            Chunk(
                document_id=document_id,
                chunk_text=chunk.text,
                chunk_order=chunk.chunk_order,
                page_number=chunk.page_number,
            )
            for chunk in chunks
        ]
        self.session.add_all(chunk_models)
        await self.session.flush()
        return chunk_models

    async def assign_vector_indices(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            chunk.vector_index = chunk.id
        await self.session.flush()

    async def get_document_chunk_ids(self, document_id: int) -> list[int]:
        result = await self.session.execute(
            select(Chunk.id)
            .join(Document, Chunk.document_id == Document.id)
            .where(Chunk.document_id == document_id)
            .where(Document.is_deleted.is_(False))
        )
        return list(result.scalars().all())

    async def get_active_chunks_by_ids(self, chunk_ids: list[int]) -> dict[int, Chunk]:
        if not chunk_ids:
            return {}

        result = await self.session.execute(
            select(Chunk)
            .options(joinedload(Chunk.document))
            .join(Document, Chunk.document_id == Document.id)
            .where(Chunk.id.in_(chunk_ids))
            .where(Document.is_deleted.is_(False))
        )
        chunks = result.scalars().unique().all()
        return {chunk.id: chunk for chunk in chunks}
