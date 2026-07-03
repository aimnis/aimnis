"""Local text embeddings via fastembed (ONNX, CPU, no torch).

The model is loaded lazily and cached — the first call downloads/initializes it,
subsequent calls are cheap. Embeddings run locally so they never consume upstream
OpenRouter quota.
"""

from __future__ import annotations

from functools import lru_cache

from .config import settings


@lru_cache(maxsize=1)
def _model():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=settings.embedding_model)


def embed(text: str) -> list[float]:
    """Embed a single string into a `settings.embedding_dim`-length vector."""
    vec = next(iter(_model().embed([text])))
    return [float(x) for x in vec]


def embed_many(texts: list[str]) -> list[list[float]]:
    return [[float(x) for x in v] for v in _model().embed(texts)]
