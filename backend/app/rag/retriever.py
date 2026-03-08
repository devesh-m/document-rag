from dataclasses import dataclass

from app.database.models import Chunk
from app.database.repositories import ChunkRepository
from app.rag.embedder import EmbeddingService
from app.rag.vector_store import FAISSVectorStore


@dataclass(slots=True)
class RetrievedChunk:
    chunk: Chunk
    similarity_score: float


class Retriever:
    def __init__(
        self,
        embedder: EmbeddingService,
        vector_store: FAISSVectorStore,
        chunk_repository: ChunkRepository,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunk_repository = chunk_repository

    async def retrieve(self, question: str, top_k: int) -> list[RetrievedChunk]:
        query_embedding = await self.embedder.embed_query(question)
        search_results = await self.vector_store.search(query_embedding, top_k=top_k * 3)
        if not search_results:
            return []

        chunk_ids = [chunk_id for chunk_id, _ in search_results]
        chunk_map = await self.chunk_repository.get_active_chunks_by_ids(chunk_ids)

        retrieved: list[RetrievedChunk] = []
        for chunk_id, score in search_results:
            chunk = chunk_map.get(chunk_id)
            if chunk is None:
                continue
            retrieved.append(RetrievedChunk(chunk=chunk, similarity_score=score))
            if len(retrieved) >= top_k:
                break
        return retrieved
