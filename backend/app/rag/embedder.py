import asyncio
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._dimension: int | None = None

    async def initialize(self) -> None:
        await asyncio.to_thread(self._load_model)

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise RuntimeError("Embedding model has not been initialized.")
        return self._dimension

    async def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype="float32")
        embeddings = await asyncio.to_thread(self._encode, list(texts))
        return embeddings

    async def embed_query(self, text: str) -> np.ndarray:
        embeddings = await self.embed_texts([text])
        return embeddings[0]

    def _load_model(self) -> None:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()

    def _encode(self, texts: list[str]) -> np.ndarray:
        self._load_model()
        assert self._model is not None
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype("float32")
