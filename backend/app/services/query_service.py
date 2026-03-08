from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.database.repositories import ChunkRepository
from app.rag.embedder import EmbeddingService
from app.rag.ollama_client import OllamaClient
from app.rag.prompt_builder import PromptBuilder
from app.rag.retriever import Retriever
from app.rag.vector_store import FAISSVectorStore
from app.schemas import CitationResponse, QueryRequest, QueryResponse


class QueryService:
    def __init__(
        self,
        session: AsyncSession,
        embedder: EmbeddingService,
        vector_store: FAISSVectorStore,
        prompt_builder: PromptBuilder,
        ollama_client: OllamaClient,
        settings: Settings,
    ) -> None:
        self.retriever = Retriever(
            embedder=embedder,
            vector_store=vector_store,
            chunk_repository=ChunkRepository(session),
        )
        self.prompt_builder = prompt_builder
        self.ollama_client = ollama_client
        self.settings = settings

    async def answer_question(self, payload: QueryRequest) -> QueryResponse:
        top_k = payload.top_k or self.settings.retrieval_top_k
        retrieved_chunks = await self.retriever.retrieve(payload.question, top_k=top_k)

        if not retrieved_chunks:
            return QueryResponse(
                answer="I could not find relevant content in the indexed documents.",
                citations=[],
                retrieved_chunks=0,
            )

        citations = [
            CitationResponse(
                chunk_id=item.chunk.id,
                document_id=item.chunk.document_id,
                filename=item.chunk.document.filename,
                page_number=item.chunk.page_number,
                snippet=item.chunk.chunk_text[:500],
                similarity_score=round(item.similarity_score, 4),
            )
            for item in retrieved_chunks
        ]
        prompt = self.prompt_builder.build(payload.question, citations)
        answer = await self.ollama_client.generate(prompt)
        return QueryResponse(
            answer=answer.strip(),
            citations=citations,
            retrieved_chunks=len(citations),
        )
