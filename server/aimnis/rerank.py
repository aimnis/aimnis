"""Cross-encoder reranking of semantic-cache candidates.

The bi-encoder used for retrieval (bge-small) embeds each query on its own, so it
is structurally weak at telling apart near-identical but opposite-intent queries
("how to enable X" vs "how to disable X" land very close). A cross-encoder jointly
encodes the (incoming query, candidate query) pair and scores their relevance
directly, which separates gross mismatches (unrelated, or reworded opposites)
sharply. It does NOT reliably split a single flipped word — that residual is
delegated to the calling agent by echoing the matched question (see resolve /
format_for_agent).

Runs locally on CPU (fastembed ONNX), so it spends no upstream quota. The model is
loaded lazily and cached; the first call downloads/initializes it.
"""

from __future__ import annotations

from functools import lru_cache
from math import exp

from .config import settings


@lru_cache(maxsize=1)
def _model():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=settings.rerank_model)


def _sigmoid(x: float) -> float:
    # Overflow-safe: cross-encoder logits can be large-magnitude negatives.
    if x >= 0:
        return 1.0 / (1.0 + exp(-x))
    e = exp(x)
    return e / (1.0 + e)


def rank(query: str, documents: list[str]) -> list[tuple[int, float]]:
    """Rescore `documents` against `query`, returning (original_index, score) pairs
    sorted by score descending. `score` is the sigmoid of the cross-encoder logit,
    normalized to (0, 1) so it can be compared against a fixed accept threshold."""
    if not documents:
        return []
    scores = list(_model().rerank(query, documents))
    ranked = [(i, _sigmoid(float(s))) for i, s in enumerate(scores)]
    ranked.sort(key=lambda t: t[1], reverse=True)
    return ranked
