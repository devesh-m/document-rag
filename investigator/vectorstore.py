from __future__ import annotations

import uuid
from urllib.parse import urlparse

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from investigator.config import Settings


EMBED_DIM = 768


def connect_qdrant(url: str, api_key: str) -> QdrantClient:
    raw = (url or "").strip()
    if not raw:
        raise RuntimeError("QDRANT_URL is not set.")
    if "://" not in raw:
        raw = ("https://" if api_key else "http://") + raw
    parsed = urlparse(raw)
    host = parsed.hostname
    if not host:
        raise RuntimeError("QDRANT_URL is not a valid host.")
    https = parsed.scheme == "https" or bool(api_key)
    port = parsed.port or (443 if https else 6333)
    try:
        client = QdrantClient(
            host=host,
            port=port,
            https=https,
            api_key=api_key or None,
            timeout=30,
            prefer_grpc=False,
            check_compatibility=False,
        )
        client.get_collections()
        return client
    except TypeError:
        try:
            client = QdrantClient(url=raw, api_key=api_key or None, timeout=30)
            client.get_collections()
            return client
        except Exception as exc:
            raise RuntimeError(f"Could not reach Qdrant ({host}): {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Could not reach Qdrant ({host}:{port}): {exc}") from exc


class PassageStore:
    def __init__(self, settings: Settings) -> None:
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        self.settings = settings
        self.client = connect_qdrant(settings.qdrant_url, settings.qdrant_api_key)
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
        try:
            vectors = self.embeddings.embed_documents(texts)
        except Exception as exc:
            raise RuntimeError(f"Gemini embeddings failed: {exc}") from exc
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
