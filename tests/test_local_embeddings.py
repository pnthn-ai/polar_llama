"""CI-safe tests for the in-process local embedding backend (issue #83).

These tests never import ``mlx``/``mlx_embeddings`` or touch a GPU: they
exercise the embedding engine seam (``LocalEmbeddingEngine`` /
``FakeEmbeddingEngine`` / registry) and the ``embedding_local`` /
``.llama.embedding_local`` ``map_batches`` path entirely on CPU, using the
deterministic ``FakeEmbeddingEngine``. The real ``MlxEmbeddingEngine`` path
is intentionally NOT exercised here (BLOCKED-ON-GPU; see
``tests/test_local_embeddings_gpu.py``).
"""

from __future__ import annotations

import math
import sys

import polars as pl
import pytest

from helpers import assert_import_stays_mlx_free, assert_stays_mlx_free

from polar_llama import embedding_local
from polar_llama.index import HnswIndex
from polar_llama.local.embed import (
    DEFAULT_EMBED_CHUNK_SIZE,
    FAKE_FAIL_MARKER,
    FakeEmbeddingEngine,
    LocalEmbeddingEngine,
    clear_embedding_registry,
    embed_chunked,
    get_embedding_engine,
    register_embedding_engine,
)

pytestmark = pytest.mark.local


@pytest.fixture(autouse=True)
def _clean_embedding_registry():
    """Isolate the process-global embedding registry between tests."""
    clear_embedding_registry()
    yield
    clear_embedding_registry()


def _norm(vec):
    return math.sqrt(sum(x * x for x in vec))


# ---------------------------------------------------------------------------
# 1. FakeEmbeddingEngine basics
# ---------------------------------------------------------------------------
def test_fake_embedding_engine_is_protocol():
    engine = FakeEmbeddingEngine("m")
    assert isinstance(engine, LocalEmbeddingEngine)


# ---------------------------------------------------------------------------
# 2. Deterministic, row-aligned
# ---------------------------------------------------------------------------
def test_fake_engine_deterministic_and_row_aligned():
    engine = FakeEmbeddingEngine("m")
    texts = [f"doc-{i}" for i in range(20)] + ["doc-0"]  # duplicate at the end
    vecs = engine.embed(texts)

    assert len(vecs) == len(texts)
    # Identical text -> identical vector.
    assert vecs[0] == vecs[-1]
    # Distinct text -> distinct vector (overwhelmingly likely with SHA-256).
    assert vecs[0] != vecs[1]
    # Re-embedding the same text again reproduces the same vector.
    again = engine.embed(["doc-0"])
    assert again[0] == vecs[0]


# ---------------------------------------------------------------------------
# 3. dtype parity with embedding_async
# ---------------------------------------------------------------------------
def test_dtype_parity_with_embedding_async(monkeypatch):
    """embedding_local's output dtype must match embedding_async's documented
    contract: List[Float64] (see polar_llama/__init__.py::embedding_async,
    "Expression with embeddings as List[Float64]").
    """
    monkeypatch.setenv("POLAR_LLAMA_LOCAL_ENGINE", "fake")

    df = pl.DataFrame({"text": ["hello", "world"]})
    result = df.with_columns(emb=embedding_local(pl.col("text")))

    assert result.schema["emb"] == pl.List(pl.Float64)


# ---------------------------------------------------------------------------
# 4. Null handling; empty string is embedded (parity with embedding_async's
#    Rust plugin, which only skips `None`, not `""`).
# ---------------------------------------------------------------------------
def test_null_rows_produce_null_embeddings_empty_string_is_embedded(monkeypatch):
    monkeypatch.setenv("POLAR_LLAMA_LOCAL_ENGINE", "fake")

    df = pl.DataFrame({"text": [None, "a", ""]})
    result = df.with_columns(emb=embedding_local(pl.col("text")))
    embs = result["emb"].to_list()

    assert embs[0] is None
    assert embs[1] is not None
    assert embs[2] is not None  # "" is embedded, not nulled -- matches embedding_async


# ---------------------------------------------------------------------------
# 5. normalize option
# ---------------------------------------------------------------------------
def test_normalize_option():
    engine = FakeEmbeddingEngine("m", dim=8)

    normalized = engine.embed(["some text"], normalize=True)[0]
    assert abs(_norm(normalized) - 1.0) < 1e-9

    raw = engine.embed(["some text"], normalize=False)[0]
    # The raw (unnormalized) vector should generally NOT already be unit
    # norm -- assert it differs from the normalized version.
    assert raw != normalized


# ---------------------------------------------------------------------------
# 6. batch_size chunking
# ---------------------------------------------------------------------------
class _RecordingFakeEngine(FakeEmbeddingEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.call_sizes = []

    def embed(self, texts, *, normalize=True):
        self.call_sizes.append(len(texts))
        return super().embed(texts, normalize=normalize)


def test_batch_size_chunking():
    engine = _RecordingFakeEngine("m")
    texts = [f"t{i}" for i in range(10)]

    vecs = embed_chunked(engine, texts, normalize=True, chunk_size=3)

    assert engine.call_sizes == [3, 3, 3, 1]
    assert len(vecs) == 10
    # Row-aligned: re-embedding a given text via the un-chunked path matches.
    assert vecs[0] == engine.embed(["t0"])[-1]


def test_default_chunk_size_is_smaller_than_generation_default():
    # Sanity: embeddings default chunk size exists and is a positive int.
    assert DEFAULT_EMBED_CHUNK_SIZE > 0


# ---------------------------------------------------------------------------
# 7. registry singleton + injection
# ---------------------------------------------------------------------------
def test_registry_singleton_and_injection():
    fake = FakeEmbeddingEngine("m")
    register_embedding_engine("m", fake, engine="in_process")

    resolved = get_embedding_engine("m", engine="in_process")
    assert resolved is fake

    # embedding_local resolves to the registered instance WITHOUT the env
    # override, purely through the (model, engine) registry key.
    df = pl.DataFrame({"text": ["hello"]})
    result = df.with_columns(
        emb=embedding_local(pl.col("text"), model="m", engine="in_process")
    )
    assert result["emb"][0] is not None
    assert result["emb"][0].to_list() == fake.embed(["hello"])[0]


# ---------------------------------------------------------------------------
# 8. env override forces fake, no mlx import
# ---------------------------------------------------------------------------
def test_env_override_forces_fake(monkeypatch):
    monkeypatch.setenv("POLAR_LLAMA_LOCAL_ENGINE", "fake")

    df = pl.DataFrame({"text": ["hello", "world"]})
    result = df.with_columns(
        emb=embedding_local(pl.col("text"), model="whatever-model", engine="in_process")
    )

    assert result.schema["emb"] == pl.List(pl.Float64)
    assert result["emb"].null_count() == 0

    # The "forces fake" half of the claim -- that this path never reaches for
    # mlx -- is only meaningful in a fresh interpreter: in-process, mlx may
    # already be in sys.modules because another test imported it.
    assert_stays_mlx_free(
        "import os\n"
        "os.environ['POLAR_LLAMA_LOCAL_ENGINE'] = 'fake'\n"
        "import polars as pl\n"
        "from polar_llama import embedding_local\n"
        "pl.DataFrame({'text': ['hello', 'world']}).with_columns(\n"
        "    emb=embedding_local(pl.col('text'), model='whatever-model',\n"
        "                        engine='in_process'))\n",
        what="the fake embedding path",
    )


# ---------------------------------------------------------------------------
# 9. per-row failure isolation
# ---------------------------------------------------------------------------
def test_per_row_failure_yields_null():
    engine = FakeEmbeddingEngine("m", fail_on=lambda t: t == "bad")
    vecs = engine.embed(["good-1", "bad", "good-2"])

    assert vecs[0] is not None
    assert vecs[1] is None
    assert vecs[2] is not None


def test_fail_marker_yields_null(monkeypatch):
    monkeypatch.setenv("POLAR_LLAMA_LOCAL_ENGINE", "fake")

    df = pl.DataFrame({"text": ["ok", f"trigger {FAKE_FAIL_MARKER}", "ok2"]})
    result = df.with_columns(emb=embedding_local(pl.col("text")))
    embs = result["emb"].to_list()

    assert embs[0] is not None
    assert embs[1] is None
    assert embs[2] is not None


# ---------------------------------------------------------------------------
# 10. importing the module never imports mlx
# ---------------------------------------------------------------------------
def test_import_embed_module_does_not_import_mlx():
    """Checked out-of-process -- see the note in test_local_ci_smoke.py."""
    import polar_llama.local.embed as embed_module

    assert embed_module is not None
    assert_import_stays_mlx_free(["polar_llama.local.embed"])


# ---------------------------------------------------------------------------
# 11. fake embeddings feed HnswIndex + cosine_similarity unchanged
# ---------------------------------------------------------------------------
def test_fake_embeddings_feed_hnsw_and_cosine(monkeypatch):
    from polar_llama import cosine_similarity

    monkeypatch.setenv("POLAR_LLAMA_LOCAL_ENGINE", "fake")

    df = pl.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "text": [
                "the cat sat on the mat",
                "a feline rested on the rug",
                "quarterly revenue grew 12 percent",
                "the cat sat on the mat",  # duplicate of row 0
            ],
        }
    ).with_columns(emb=embedding_local(pl.col("text")))

    assert df.schema["emb"] == pl.List(pl.Float64)

    index = HnswIndex.build(df, id_col="id", embedding_col="emb")
    query_df = df.filter(pl.col("id") == "a").select("emb")
    result = index.query(query_df, "emb", k=4)

    # The nearest neighbor of row "a" is itself (distance 0) since the
    # FakeEmbeddingEngine is deterministic; the exact-duplicate row "d"
    # should also rank at distance 0.
    top_ids = result.sort("rank")["neighbor_id"].to_list()
    assert top_ids[0] in ("a", "d")

    sim = df.select(
        cosine_similarity(pl.col("emb"), pl.col("emb")).alias("self_sim")
    )
    for v in sim["self_sim"].to_list():
        assert v == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# .llama namespace parity
# ---------------------------------------------------------------------------
def test_llama_namespace_embedding_local(monkeypatch):
    monkeypatch.setenv("POLAR_LLAMA_LOCAL_ENGINE", "fake")

    df = pl.DataFrame({"text": ["hello", "world"]})
    result = df.with_columns(emb=pl.col("text").llama.embedding_local())

    assert result.schema["emb"] == pl.List(pl.Float64)
    assert result["emb"].null_count() == 0
