"""Local ONNX embeddings via fastembed. No torch, ~70 MB model download on first use."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from fastembed import TextEmbedding


class FastEmbedEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model_name = model_name
        self._model = TextEmbedding(model_name)
        info = next(m for m in TextEmbedding.list_supported_models() if m["model"] == model_name)
        self.dim = int(info["dim"])

    @staticmethod
    def _normalise(vec: np.ndarray) -> list[float]:
        norm = float(np.linalg.norm(vec))
        return (vec / norm if norm else vec).astype(np.float32).tolist()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._normalise(v) for v in self._model.passage_embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return self._normalise(next(self._model.query_embed(text)))
