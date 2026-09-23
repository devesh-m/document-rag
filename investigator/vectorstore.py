from __future__ import annotations

import uuid
from urllib.parse import urlparse

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from investigator.config import Settings
from investigator.loop import ensure_event_loop


EMBED_DIM = 3072


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
        ensure_event_loop()
        self.settings = settings
        self.client = connect_qdrant(settings.qdrant_url, settings.qdrant_api_key)
        model = settings.resolved_embedding_model
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=model,
                google_api_key=settings.gemini_api_key,
                transport="rest",
            )
        except Exception:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=model,
                google_api_key=settings.gemini_api_key,
            )
        self._ensure_collection(EMBED_DIM)

    def _vector_size(self) -> int | None:
        try:
            info = self.client.get_collection(self.settings.qdrant_collection)
            vectors = info.config.params.vectors
        except Exception:
            return None
        if hasattr(vectors, "size"):
            return int(vectors.size)
        if isinstance(vectors, dict):
            first = next(iter(vectors.values()), None)
            if first is not None and hasattr(first, "size"):
                return int(first.size)
        return None

    def _ensure_collection(self, size: int) -> None:
        name = self.settings.qdrant_collection
        names = [item.name for item in self.client.get_collections().collections]
        if name in names:
            existing = self._vector_size()
            if existing != size:
                self.client.delete_collection(collection_name=name)
                names = [item.name for item in self.client.get_collections().collections]
        if name not in names:
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=size, distance=Distance.COSINE),
            )
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        try:
            self.client.create_payload_index(
                collection_name=self.settings.qdrant_collection,
                field_name="document_id",
                field_schema="integer",
            )
        except Exception:
            return

    def clear(self) -> None:
        name = self.settings.qdrant_collection
        names = [item.name for item in self.client.get_collections().collections]
        if name in names:
            self.client.delete_collection(collection_name=name)
        self._ensure_collection(EMBED_DIM)

    def upsert_chunks(self, *, document_id: int, filename: str, chunks: list[dict]) -> int:
        ensure_event_loop()
        texts = [item["text"] for item in chunks]
        try:
            vectors = self.embeddings.embed_documents(texts)
        except Exception as exc:
            raise RuntimeError(f"Gemini embeddings failed: {exc}") from exc
        if not vectors:
            raise RuntimeError("Gemini returned no embeddings.")
        self._ensure_collection(len(vectors[0]))
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
        ensure_event_loop()
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
        if not results:
            records, _ = self.client.scroll(
                collection_name=self.settings.qdrant_collection,
                limit=limit or self.settings.retrieval_top_k,
                with_payload=True,
                with_vectors=False,
            )
            for rec in records:
                payload = rec.payload or {}
                results.append(
                    {
                        "chunk_id": payload.get("chunk_id", ""),
                        "document_id": payload.get("document_id"),
                        "filename": payload.get("filename", ""),
                        "page": payload.get("page"),
                        "text": payload.get("text", ""),
                        "score": 0.5,
                    }
                )
        return results

    def count(self) -> int:
        try:
            return int(self.client.count(collection_name=self.settings.qdrant_collection).count)
        except Exception:
            return 0

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

    def delete_document(self, document_id: int) -> None:
        ids: list = []
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.settings.qdrant_collection,
                limit=128,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            for record in records:
                payload = record.payload or {}
                try:
                    stored = int(payload.get("document_id"))
                except (TypeError, ValueError):
                    continue
                if stored == int(document_id):
                    ids.append(record.id)
            if offset is None:
                break
        if ids:
            self.client.delete(
                collection_name=self.settings.qdrant_collection,
                points_selector=ids,
            )
