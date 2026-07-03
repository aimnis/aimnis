"""Real-embedding integration: proves the local model + pgvector retrieval agree
on semantics. Skips if the model can't load (e.g. offline CI)."""

from __future__ import annotations

import pytest

from aimnis import embedding, pool
from aimnis.config import settings
from aimnis.normalize import normalize, query_hash


def test_check_model_supported_rejects_garbage(monkeypatch):
    # A mis-pasted / bogus AIMNIS_EMBEDDING_MODEL must fail fast with a clear error
    # (this is what would have turned the Railway 500 loop into one glance).
    monkeypatch.setattr(settings, "embedding_model", "XcYC_not-a-real-model")
    with pytest.raises(RuntimeError, match="not a supported"):
        embedding.check_model_supported()


def test_check_model_supported_accepts_default():
    # The shipped default must be a real fastembed model.
    embedding.check_model_supported()


def _embed_or_skip(text: str):
    try:
        from aimnis.embedding import embed

        return embed(text)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"embedding model unavailable: {exc}")


async def test_paraphrase_closer_than_unrelated(clean):
    stored = "how do I fix a python ImportError"
    paraphrase = "python import error, how to resolve it"
    unrelated = "best margherita pizza recipe in naples"

    e_stored = _embed_or_skip(stored)
    e_para = _embed_or_skip(paraphrase)
    e_unrel = _embed_or_skip(unrelated)

    norm = normalize(stored)
    await pool.insert(
        clean,
        query_text=stored,
        query_norm=norm,
        query_hash=query_hash(norm),
        embedding=e_stored,
        answer_text="check that the module is installed and on sys.path",
    )

    # With a permissive threshold both return the nearest (the stored entry);
    # the paraphrase must be strictly closer than the unrelated query.
    hit_para = await pool.lookup(clean, query_hash="x", embedding=e_para, max_distance=1.0)
    hit_unrel = await pool.lookup(clean, query_hash="x", embedding=e_unrel, max_distance=1.0)

    assert hit_para is not None and hit_para.match == "semantic"
    assert hit_unrel is not None
    assert hit_para.distance < hit_unrel.distance
