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


def _supported_model_names() -> set[str]:
    from fastembed import TextEmbedding

    names: set[str] = set()
    for m in TextEmbedding.list_supported_models():
        name = m.get("model") or m.get("model_name")  # key name varies by version
        if name:
            names.add(name)
    return names


def check_model_supported() -> None:
    """Fail fast (call at startup) if AIMNIS_EMBEDDING_MODEL isn't a fastembed model.

    Without this a bad/mis-pasted model name only surfaces as a 500 on the first
    search (embedding is lazy), which is hard to diagnose. This turns it into one
    clear error in the deploy logs. Cheap: list_supported_models() is static
    metadata, no model download.
    """
    supported = _supported_model_names()
    if settings.embedding_model not in supported:
        raise RuntimeError(
            f"AIMNIS_EMBEDDING_MODEL={settings.embedding_model!r} is not a supported "
            f"fastembed model. Set it to a supported name (e.g. 'BAAI/bge-small-en-v1.5') "
            f"or unset it to use the default. Supported models: {sorted(supported)}"
        )


def embed(text: str) -> list[float]:
    """Embed a single string into a `settings.embedding_dim`-length vector."""
    vec = next(iter(_model().embed([text])))
    return [float(x) for x in vec]


def embed_many(texts: list[str]) -> list[list[float]]:
    return [[float(x) for x in v] for v in _model().embed(texts)]
