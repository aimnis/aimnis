"""Cross-encoder rerank tests.

The model is monkeypatched so these run without downloading it — we test the
ordering, sigmoid normalization, and the no-op / overflow edges, not the ONNX
weights themselves.
"""

from __future__ import annotations

from aimnis import rerank


class _FakeCrossEncoder:
    """Returns a fixed logit per document string, preserving input order."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def rerank(self, query: str, documents: list[str]):
        return [self._scores[d] for d in documents]


def test_rank_orders_by_score_desc(monkeypatch):
    scores = {"enable telemetry": 5.0, "disable telemetry": 2.0, "add a nuget pkg": -8.0}
    monkeypatch.setattr(rerank, "_model", lambda: _FakeCrossEncoder(scores))

    ranked = rerank.rank("enable telemetry", list(scores))

    # Highest logit first; original indices preserved in the tuples.
    assert [i for i, _ in ranked] == [0, 1, 2]
    vals = [s for _, s in ranked]
    assert vals == sorted(vals, reverse=True)
    assert all(0.0 < v < 1.0 for v in vals)  # sigmoid-normalized


def test_rank_empty_docs_does_not_load_model(monkeypatch):
    def _boom():
        raise AssertionError("model must not be loaded for empty documents")

    monkeypatch.setattr(rerank, "_model", _boom)
    assert rerank.rank("q", []) == []


def test_sigmoid_is_overflow_safe():
    assert rerank._sigmoid(-1000.0) == 0.0
    assert 0.99 < rerank._sigmoid(1000.0) <= 1.0
    assert rerank._sigmoid(0.0) == 0.5
