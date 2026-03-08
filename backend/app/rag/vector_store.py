import asyncio
import json
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np


class FAISSVectorStore:
    def __init__(self, index_path: Path, metadata_path: Path) -> None:
        self.index_path = index_path
        self.metadata_path = metadata_path
        self._index: faiss.IndexIDMap2 | None = None
        self._dimension: int | None = None
        self._lock = asyncio.Lock()

    async def initialize(self, dimension: int) -> None:
        async with self._lock:
            await asyncio.to_thread(self._initialize_sync, dimension)

    async def add_embeddings(self, embeddings: np.ndarray, ids: list[int]) -> None:
        if embeddings.size == 0:
            return
        async with self._lock:
            await asyncio.to_thread(self._add_embeddings_sync, embeddings, ids)

    async def remove_embeddings(self, ids: Iterable[int]) -> int:
        ids_array = np.array(list(ids), dtype="int64")
        if ids_array.size == 0:
            return 0
        async with self._lock:
            return await asyncio.to_thread(self._remove_embeddings_sync, ids_array)

    async def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if top_k <= 0:
            return []
        async with self._lock:
            return await asyncio.to_thread(self._search_sync, query_embedding, top_k)

    async def stats(self) -> dict[str, int]:
        async with self._lock:
            if self._index is None or self._dimension is None:
                return {"dimension": 0, "total_vectors": 0}
            return {"dimension": self._dimension, "total_vectors": self._index.ntotal}

    def _initialize_sync(self, dimension: int) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        if self.index_path.exists():
            loaded_index = faiss.read_index(str(self.index_path))
            if not isinstance(loaded_index, faiss.IndexIDMap2):
                loaded_index = faiss.IndexIDMap2(loaded_index)
            self._index = loaded_index
            self._dimension = loaded_index.d
            if self._dimension != dimension:
                raise RuntimeError(
                    f"Existing FAISS dimension {self._dimension} does not match embedding dimension {dimension}."
                )
            self._persist_metadata()
            return

        self._dimension = dimension
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
        faiss.write_index(self._index, str(self.index_path))
        self._persist_metadata()

    def _ensure_index(self) -> faiss.IndexIDMap2:
        if self._index is None or self._dimension is None:
            raise RuntimeError("FAISS index has not been initialized.")
        return self._index

    def _add_embeddings_sync(self, embeddings: np.ndarray, ids: list[int]) -> None:
        index = self._ensure_index()
        embeddings = np.atleast_2d(np.asarray(embeddings, dtype="float32"))
        ids_array = np.asarray(ids, dtype="int64")
        index.add_with_ids(embeddings, ids_array)
        faiss.write_index(index, str(self.index_path))
        self._persist_metadata()

    def _remove_embeddings_sync(self, ids_array: np.ndarray) -> int:
        index = self._ensure_index()
        removed = index.remove_ids(ids_array)
        faiss.write_index(index, str(self.index_path))
        self._persist_metadata()
        return int(removed)

    def _search_sync(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        index = self._ensure_index()
        if index.ntotal == 0:
            return []

        query = np.asarray([query_embedding], dtype="float32")
        scores, ids = index.search(query, min(top_k, index.ntotal))
        results: list[tuple[int, float]] = []
        for vector_id, score in zip(ids[0], scores[0], strict=False):
            if int(vector_id) == -1:
                continue
            results.append((int(vector_id), float(score)))
        return results

    def _persist_metadata(self) -> None:
        if self._dimension is None or self._index is None:
            return
        payload = {
            "dimension": self._dimension,
            "total_vectors": int(self._index.ntotal),
        }
        self.metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
