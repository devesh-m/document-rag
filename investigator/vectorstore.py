from __future__ import annotations

import uuid

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from investigator.config import Settings


EMBED_DIM = 768


class PassageStore:
    def __init__(self, settings: Settings) -> None:
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        self.settings = settings
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.gemini_api_key,
        )
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        names = [item.name for item in self.client.get_collections().collections]
        if self.settings.qdrant_collection in names:
            return
        self.client.create_collection(
            collection_name=self.settings.qdrant_collection,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )

    def upsert_chunks(self, *, document_id: int, filename: str, chunks: list[dict]) -> int:
        texts = [item["text"] for item in chunks]
        vectors = self.embeddings.embed_documents(texts)
        points = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk_id = chunk["chunk_id"]
            points.append(
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id)),
                    vector=vector,
                    payload={
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "filename": filename,
                        "page": chunk["page"],
                        "text": chunk["text"],
                    },
                )
            )
        self.client.upsert(collection_name=self.settings.qdrant_collection, points=points)
        return len(points)

    def search(self, query: str, limit: int | None = None) -> list[dict]:
        vector = self.embeddings.embed_query(query)
        hits = self.client.search(
            collection_name=self.settings.qdrant_collection,
            query_vector=vector,
            limit=limit or self.settings.retrieval_top_k,
            with_payload=True,
        )
        results = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                {
                    "chunk_id": payload.get("chunk_id", ""),
                    "document_id": payload.get("document_id"),
                    "filename": payload.get("filename", ""),
                    "page": payload.get("page"),
                    "text": payload.get("text", ""),
                    "score": float(hit.score or 0),
                }
            )
        return results

    def get_chunk(self, chunk_id: str) -> dict | None:
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
        points = self.client.retrieve(
            collection_name=self.settings.qdrant_collection,
            ids=[point_id],
            with_payload=True,
        )
        if not points:
            return None
        payload = points[0].payload or {}
        return {
            "chunk_id": payload.get("chunk_id", chunk_id),
            "document_id": payload.get("document_id"),
            "filename": payload.get("filename", ""),
            "page": payload.get("page"),
            "text": payload.get("text", ""),
        }
