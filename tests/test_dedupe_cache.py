#!/usr/bin/env python3
"""
Test suite for in-run dedupe / persistent response caching (issue #77).

Layered the same way as `tests/test_checkpointing.py`:

- Unit-level tests drive `polar_llama.dedup.deduped_expr` directly with a
  hand-written `run_pending` stub -- no network, no API keys, no mlx.
- An end-to-end test drives the real `inference_async(..., dedupe=True)`
  wiring against a local stdlib mock OpenAI-compatible HTTP server (same
  `ThreadingHTTPServer` pattern as `test_checkpointing.py`).
- A gated real-API proof sits at the bottom, skipped unless `OPENAI_API_KEY`
  is set.
"""

from __future__ import annotations

import datetime
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from helpers import BacklogHTTPServer
from typing import Callable

import polars as pl
import pytest
from pydantic import BaseModel

from polar_llama import (
    Checkpoint,
    DedupeStats,
    Provider,
    ResponseCache,
    inference_async,
    inference_messages,
)
from polar_llama.checkpoint import CheckpointStore
from polar_llama.dedup import ResponseCacheStore, _parse_ttl, deduped_expr
from polar_llama.keys import config_fingerprint, content_key

# ============================================================================
# TTL parsing
# ============================================================================


def test_parse_ttl_accepts_all_shapes():
    assert _parse_ttl(None) is None
    assert _parse_ttl(90) == 90.0
    assert _parse_ttl(90.5) == 90.5
    assert _parse_ttl(datetime.timedelta(minutes=2)) == 120.0
    assert _parse_ttl("30m") == 1800.0
    assert _parse_ttl("24h") == 86400.0
    assert _parse_ttl("7d") == 604800.0
    assert _parse_ttl("45s") == 45.0


def test_parse_ttl_rejects_bad_strings():
    with pytest.raises(ValueError):
        _parse_ttl("bogus")
    with pytest.raises(ValueError):
        _parse_ttl("10x")


def test_response_cache_validates_ttl_at_construction(tmp_path):
    with pytest.raises(ValueError):
        ResponseCache(tmp_path / "rc", ttl="not-a-ttl")


# ============================================================================
# Unit tests -- deduped_expr with a stub run_pending, no store
# ============================================================================


def _make_counting_stub(counter: dict) -> Callable[[pl.Series], pl.Series]:
    def run_pending(s: pl.Series) -> pl.Series:
        seen = []
        for v in s.to_list():
            counter["n"] += 1
            seen.append(f"OUT:{v}")
        return pl.Series(seen, dtype=pl.Utf8)

    return run_pending


def test_forty_percent_duplicate_acceptance():
    # 100 rows, 60 unique values -> exactly 60 calls, 40 collapsed.
    counter = {"n": 0}
    stats = DedupeStats()
    fp = "fp-dedupe-40pct"

    unique_values = [f"v{i}" for i in range(60)]
    # First 60 rows are the unique values themselves; the remaining 40 rows
    # repeat the first 40 unique values.
    values = unique_values + unique_values[:40]
    df = pl.DataFrame({"prompt": values})

    expr = deduped_expr(
        pl.col("prompt"),
        run_pending=_make_counting_stub(counter),
        fingerprint=fp,
        stats=stats,
    )
    out = df.with_columns(out=expr)

    assert out["out"].to_list() == [f"OUT:{v}" for v in values]
    assert counter["n"] == 60
    assert stats.calls_made == 60
    assert stats.rows_collapsed == 40
    assert stats.rows_total == 100
    assert stats.rows_null == 0
    assert stats.cache_hits == 0


def test_result_identity_deduped_matches_undeduped():
    fp = "fp-identity"
    values = (["alpha"] * 5) + (["beta"] * 4) + (["gamma"] * 3) + ["delta"]
    df = pl.DataFrame({"prompt": values})

    def deterministic(s: pl.Series) -> pl.Series:
        return pl.Series([f"out:{v}" for v in s.to_list()], dtype=pl.Utf8)

    undeduped = df.with_columns(
        out=pl.col("prompt").map_batches(deterministic, return_dtype=pl.Utf8)
    )

    deduped = df.with_columns(
        out=deduped_expr(pl.col("prompt"), run_pending=deterministic, fingerprint=fp)
    )

    assert deduped["out"].to_list() == undeduped["out"].to_list()


def test_order_and_null_preservation():
    counter = {"n": 0}
    fp = "fp-order-null"
    values = ["a", None, "b", "a", None, "c", "b"]
    df = pl.DataFrame({"prompt": values})

    stats = DedupeStats()
    expr = deduped_expr(
        pl.col("prompt"),
        run_pending=_make_counting_stub(counter),
        fingerprint=fp,
        stats=stats,
    )
    out = df.with_columns(out=expr)

    assert out["out"].to_list() == [
        "OUT:a",
        None,
        "OUT:b",
        "OUT:a",
        None,
        "OUT:c",
        "OUT:b",
    ]
    # 3 unique non-null values -> 3 calls.
    assert counter["n"] == 3
    assert stats.rows_total == 7
    assert stats.rows_null == 2
    assert stats.calls_made == 3
    assert stats.rows_collapsed == 2  # "a" and "b" each appear twice


def test_stats_correctness_invariant():
    counter = {"n": 0}
    fp = "fp-stats-invariant"
    values = ["a", "b", "a", None, "c", "b", "b", None]
    df = pl.DataFrame({"prompt": values})

    stats = DedupeStats()
    expr = deduped_expr(
        pl.col("prompt"),
        run_pending=_make_counting_stub(counter),
        fingerprint=fp,
        stats=stats,
    )
    df.with_columns(out=expr)

    assert stats.rows_total == 8
    assert stats.rows_null == 2
    assert stats.cache_hits == 0
    # unique non-null values: a, b, c -> 3 calls
    assert stats.calls_made == 3
    # non-null rows (6) minus unique calls (3) = 3 collapsed
    assert stats.rows_collapsed == 3
    assert stats.saved_calls == stats.cache_hits + stats.rows_collapsed == 3
    non_null = stats.rows_total - stats.rows_null
    assert stats.hit_rate == stats.saved_calls / non_null


def test_stats_reset():
    stats = DedupeStats()
    stats.rows_total = 10
    stats.calls_made = 5
    stats.reset()
    assert stats.rows_total == 0
    assert stats.calls_made == 0
    assert stats.cache_hits == 0
    assert stats.rows_collapsed == 0
    assert stats.rows_null == 0


# ============================================================================
# Persistent response_cache store -- unit tests against deduped_expr + store
# ============================================================================


def test_persistent_cache_hit_across_two_calls(tmp_path):
    fp = "fp-persist"
    path = tmp_path / "rcache"
    df = pl.DataFrame({"prompt": [f"row-{i}" for i in range(10)]})

    counter = {"n": 0}
    store1 = ResponseCacheStore(path, fp)
    stats1 = DedupeStats()
    expr1 = deduped_expr(
        pl.col("prompt"),
        run_pending=_make_counting_stub(counter),
        fingerprint=fp,
        store=store1,
        stats=stats1,
    )
    out1 = df.with_columns(out=expr1)
    assert counter["n"] == 10
    assert stats1.cache_hits == 0
    assert stats1.calls_made == 10

    # Run 2: a fresh store handle over the same directory, with a stub that
    # would raise if ever called -- everything should be served from disk.
    def _boom(s: pl.Series) -> pl.Series:
        raise AssertionError("run_pending should not be called -- fully cached")

    store2 = ResponseCacheStore(path, fp)
    stats2 = DedupeStats()
    expr2 = deduped_expr(
        pl.col("prompt"), run_pending=_boom, fingerprint=fp, store=store2, stats=stats2
    )
    out2 = df.with_columns(out=expr2)

    assert out2["out"].to_list() == out1["out"].to_list()
    assert stats2.cache_hits == 10
    assert stats2.calls_made == 0
    assert stats2.rows_collapsed == 0


def test_failed_results_are_not_persisted(tmp_path):
    fp = "fp-fail-not-cached"
    path = tmp_path / "rcache"
    df = pl.DataFrame({"prompt": ["good", "bad"]})

    def flaky(s: pl.Series) -> pl.Series:
        out = []
        for v in s.to_list():
            if v == "bad":
                out.append(json.dumps({"_error": "api_error", "_details": "boom"}))
            else:
                out.append(f"OUT:{v}")
        return pl.Series(out, dtype=pl.Utf8)

    store = ResponseCacheStore(path, fp)
    out = df.with_columns(
        out=deduped_expr(
            pl.col("prompt"),
            run_pending=flaky,
            fingerprint=fp,
            store=store,
            has_schema=True,
        )
    )
    assert out["out"][0] == "OUT:good"
    assert json.loads(out["out"][1])["_error"] == "api_error"

    index = store.load_index()
    assert len(index) == 1  # only "good" persisted
    assert content_key("good", fp) in index
    assert content_key("bad", fp) not in index

    # Resume: "bad" must be re-requested (not silently served as if cached);
    # "good" must not be.
    seen = []

    def healthy(s: pl.Series) -> pl.Series:
        vals = s.to_list()
        seen.extend(vals)
        return pl.Series([f"OUT:{v}" for v in vals], dtype=pl.Utf8)

    store2 = ResponseCacheStore(path, fp)
    out2 = df.with_columns(
        out=deduped_expr(
            pl.col("prompt"),
            run_pending=healthy,
            fingerprint=fp,
            store=store2,
            has_schema=True,
        )
    )
    assert out2["out"].to_list() == ["OUT:good", "OUT:bad"]
    assert seen == ["bad"]


def test_ttl_expiry_and_manual_invalidation(tmp_path):
    fp = "fp-ttl"
    path = tmp_path / "rcache"

    store = ResponseCacheStore(path, fp)
    store.append(["k1"], ["OUT:v1"], [True], [None])

    # Fresh (unexpired): a 1-hour ttl should still see it.
    idx = store.load_index(ttl_seconds=3600)
    assert "k1" in idx

    # Backdate the entry beyond a short ttl by rewriting the part file's
    # `ts` column directly (simulating time having passed).
    part_files = store._part_paths()
    assert len(part_files) == 1
    old_df = pl.read_parquet(part_files[0])
    backdated = old_df.with_columns(
        ts=pl.lit(
            datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=2),
            dtype=pl.Datetime("us", "UTC"),
        )
    )
    backdated.write_parquet(part_files[0])

    # A 1-hour ttl now treats the entry as expired -> pending, not a hit.
    idx_expired = store.load_index(ttl_seconds=3600)
    assert "k1" not in idx_expired

    # No ttl (None) -> the entry is still visible regardless of age.
    idx_no_ttl = store.load_index(ttl_seconds=None)
    assert "k1" in idx_no_ttl

    # A deduped_expr run against the expired entry recomputes.
    def stub(s: pl.Series) -> pl.Series:
        return pl.Series([f"RECOMPUTED:{v}" for v in s.to_list()], dtype=pl.Utf8)

    df = pl.DataFrame({"prompt": ["v1"]})
    out = df.with_columns(
        out=deduped_expr(
            pl.col("prompt"),
            run_pending=stub,
            fingerprint=fp,
            store=ResponseCacheStore(path, fp, ttl_seconds=3600),
        )
    )
    assert out["out"][0] == "RECOMPUTED:v1"

    # ResponseCache.clear() empties the store entirely.
    rc = ResponseCache(path)
    rc.clear()
    assert ResponseCacheStore(path, fp).load_index() == {}


def test_prune_drops_expired_and_compacts(tmp_path):
    path = tmp_path / "rcache"
    fp = "fp-prune"
    store = ResponseCacheStore(path, fp)

    store.append(["k1"], ["OUT:v1"], [True], [None])
    store.append(["k2"], ["OUT:v2"], [True], [None])
    assert len(store._part_paths()) == 2

    # Backdate k1 beyond a 1-hour ttl; leave k2 fresh.
    for p in store._part_paths():
        df = pl.read_parquet(p)
        if df["key"][0] == "k1":
            df = df.with_columns(
                ts=pl.lit(
                    datetime.datetime.now(datetime.timezone.utc)
                    - datetime.timedelta(hours=2),
                    dtype=pl.Datetime("us", "UTC"),
                )
            )
            df.write_parquet(p)

    rc = ResponseCache(path, ttl="1h")
    kept = rc.prune()
    assert kept == 1

    remaining_store = ResponseCacheStore(path, fp)
    remaining_parts = remaining_store._part_paths()
    assert len(remaining_parts) == 1  # compacted to a single part
    index = remaining_store.load_index()
    assert "k2" in index
    assert "k1" not in index


def test_fingerprint_change_yields_zero_stale_hits(tmp_path):
    path = tmp_path / "rcache"
    store_a = ResponseCacheStore(path, "fp-a")
    store_a.append(["irrelevant"], ["OUT:x"], [True], [None])

    def stub(s: pl.Series) -> pl.Series:
        return pl.Series([f"OUT:{v}" for v in s.to_list()], dtype=pl.Utf8)

    df = pl.DataFrame({"prompt": ["hello"]})
    stats = DedupeStats()
    out = df.with_columns(
        out=deduped_expr(
            pl.col("prompt"),
            run_pending=stub,
            fingerprint="fp-b",  # different config -> different fingerprint
            store=ResponseCacheStore(path, "fp-b"),
            stats=stats,
        )
    )
    assert out["out"][0] == "OUT:hello"
    assert stats.cache_hits == 0  # no stale hit despite sharing the directory


# ============================================================================
# response_model struct fan-back
# ============================================================================


class _Recommendation(BaseModel):
    title: str
    year: int


def test_response_model_struct_fan_back_identical_across_duplicates():
    fp = "fp-struct-fanback"
    values = ["Alpha", "Beta", "Alpha", "Alpha"]
    df = pl.DataFrame({"prompt": values})

    def stub(s: pl.Series) -> pl.Series:
        return pl.Series(
            [json.dumps({"title": v, "year": 2024}) for v in s.to_list()],
            dtype=pl.Utf8,
        )

    struct_dtype = pl.Struct({"title": pl.Utf8, "year": pl.Int64})
    from polar_llama import _parse_json_to_struct  # internal, but stable enough to test

    raw_expr = deduped_expr(pl.col("prompt"), run_pending=stub, fingerprint=fp)
    out = df.with_columns(
        rec=raw_expr.map_batches(
            lambda s: _parse_json_to_struct(s, struct_dtype), return_dtype=struct_dtype
        )
    )
    recs = out["rec"].to_list()
    assert recs[0] == {"title": "Alpha", "year": 2024}
    assert recs[2] == recs[0]
    assert recs[3] == recs[0]
    assert recs[1] == {"title": "Beta", "year": 2024}


# ============================================================================
# Interop rules
# ============================================================================


def test_response_cache_and_checkpoint_raises(tmp_path):
    df = pl.DataFrame({"prompt": ["a"]})
    with pytest.raises(ValueError, match="response_cache"):
        df.with_columns(
            out=inference_async(
                pl.col("prompt"),
                model="m",
                checkpoint=Checkpoint(tmp_path / "ckpt"),
                response_cache=ResponseCache(tmp_path / "rcache"),
            )
        )


def test_response_cache_and_usage_raises(tmp_path):
    df = pl.DataFrame({"prompt": ["a"]})
    with pytest.raises(ValueError, match="response_cache"):
        df.with_columns(
            out=inference_async(
                pl.col("prompt"),
                model="m",
                usage=True,
                response_cache=ResponseCache(tmp_path / "rcache"),
            )
        )


def test_inference_messages_response_cache_and_checkpoint_raises(tmp_path):
    df = pl.DataFrame({"msgs": [json.dumps([{"role": "user", "content": "hi"}])]})
    with pytest.raises(ValueError, match="response_cache"):
        df.with_columns(
            out=inference_messages(
                pl.col("msgs"),
                model="m",
                checkpoint=Checkpoint(tmp_path / "ckpt"),
                response_cache=ResponseCache(tmp_path / "rcache"),
            )
        )


def test_dedupe_and_checkpoint_does_not_raise_and_is_subsumed(
    mock_server, monkeypatch, tmp_path
):
    server, base_url = mock_server
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)
    ckpt_path = tmp_path / "run.ckpt"

    df = pl.DataFrame({"prompt": ["a", "a", "b"]})
    out = df.with_columns(
        answer=inference_async(
            pl.col("prompt"), model="m", checkpoint=ckpt_path, dedupe=True
        )
    )
    # checkpoint already collapses duplicates on its own; dedupe=True is a
    # harmless no-op alongside it (no raise, no double-collapsing artifact).
    assert out["answer"].to_list() == ["ECHO:a", "ECHO:a", "ECHO:b"]
    assert len(server.requests_received) == 2  # one call per unique prompt


def test_dedupe_and_provider_cache_does_not_raise(mock_server, monkeypatch):
    server, base_url = mock_server
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)

    df = pl.DataFrame({"prompt": ["a", "a"]})
    out = df.with_columns(
        answer=inference_async(pl.col("prompt"), model="m", dedupe=True, cache=True)
    )
    assert out["answer"].to_list() == ["ECHO:a", "ECHO:a"]


def test_dedupe_with_usage_duplicates_carry_identical_usage_struct(
    mock_server, monkeypatch
):
    server, base_url = mock_server
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)

    df = pl.DataFrame({"prompt": ["a", "a", "b"]})
    out = df.with_columns(
        answer=inference_async(pl.col("prompt"), model="m", dedupe=True, usage=True)
    )
    rows = out["answer"].to_list()
    assert rows[0]["response"] == "ECHO:a"
    assert rows[1]["response"] == "ECHO:a"
    # Documented semantics: the collapsed duplicate carries the SAME usage
    # struct as the row that was actually computed (only one API call was
    # made for "a"), not a zeroed-out usage.
    assert rows[0]["usage"] == rows[1]["usage"]
    assert rows[0]["usage"] is not None
    # Only 2 API calls were made (one per unique prompt), for 3 rows.
    assert len(server.requests_received) == 2


# ============================================================================
# End-to-end: inference_async(dedupe=True) against a local mock server
# ============================================================================


class _MockOpenAIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002
        pass

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        body = json.loads(raw.decode("utf-8")) if raw else {}

        with self.server.lock:  # type: ignore[attr-defined]
            self.server.requests_received.append(body)  # type: ignore[attr-defined]

        status, content = self.server.responder(body)  # type: ignore[attr-defined]

        if status != 200:
            payload_bytes = json.dumps({"error": {"message": "mock failure"}}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload_bytes)))
            self.end_headers()
            self.wfile.write(payload_bytes)
            return

        payload = {
            "id": "chatcmpl-fake",
            "model": body.get("model", "mock-model"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }
        payload_bytes = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload_bytes)))
        self.end_headers()
        self.wfile.write(payload_bytes)


def _user_content(body: dict) -> str:
    for msg in reversed(body.get("messages", [])):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def _echo_responder(body: dict):
    return 200, f"ECHO:{_user_content(body)}"


@pytest.fixture
def mock_server():
    server = BacklogHTTPServer(("127.0.0.1", 0), _MockOpenAIHandler)
    server.lock = threading.Lock()
    server.requests_received = []
    server.responder = _echo_responder
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield server, f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


@pytest.fixture(autouse=True)
def _mock_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    yield
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)


def test_inference_async_dedupe_collapses_duplicate_requests(mock_server, monkeypatch):
    server, base_url = mock_server
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)

    # 40% duplicates: 10 unique prompts, repeated to 25 rows total (60% new
    # + 40% repeats of the first 10).
    unique_prompts = [f"row-{i}" for i in range(15)]
    values = unique_prompts + unique_prompts[:10]  # 25 rows, 15 unique
    df = pl.DataFrame({"prompt": values})

    stats = DedupeStats()
    out = df.with_columns(
        answer=inference_async(
            pl.col("prompt"), model="mock-model", dedupe=True, dedupe_stats=stats
        )
    )
    assert out["answer"].to_list() == [f"ECHO:{p}" for p in values]
    assert len(server.requests_received) == 15
    assert stats.calls_made == 15
    assert stats.rows_collapsed == 10


def test_inference_async_response_cache_end_to_end(mock_server, monkeypatch, tmp_path):
    server, base_url = mock_server
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)
    rc = ResponseCache(tmp_path / "rcache")

    df = pl.DataFrame({"prompt": [f"row-{i}" for i in range(5)]})
    stats1 = DedupeStats()
    out1 = df.with_columns(
        answer=inference_async(
            pl.col("prompt"),
            model="mock-model",
            response_cache=rc,
            dedupe_stats=stats1,
        )
    )
    assert len(server.requests_received) == 5
    assert stats1.cache_hits == 0

    # Second call, fresh DataFrame, same response_cache -- zero new requests.
    stats2 = DedupeStats()
    out2 = df.with_columns(
        answer=inference_async(
            pl.col("prompt"),
            model="mock-model",
            response_cache=rc,
            dedupe_stats=stats2,
        )
    )
    assert len(server.requests_received) == 5  # unchanged
    assert stats2.cache_hits == 5
    assert out2["answer"].to_list() == out1["answer"].to_list()


def test_inference_messages_dedupe_shares_keys_across_input_shapes(
    mock_server, monkeypatch
):
    server, base_url = mock_server
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)

    conversation = [{"role": "user", "content": "hello"}]
    df = pl.DataFrame(
        {
            "msgs": pl.Series(
                [conversation, conversation],
                dtype=pl.List(pl.Struct({"role": pl.Utf8, "content": pl.Utf8})),
            )
        }
    )
    out = df.with_columns(
        answer=inference_messages(pl.col("msgs"), model="m", dedupe=True)
    )
    assert out["answer"].to_list() == ["ECHO:hello", "ECHO:hello"]
    assert len(server.requests_received) == 1


def test_dedupe_none_is_unchanged_behavior(mock_server, monkeypatch):
    server, base_url = mock_server
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)

    df = pl.DataFrame({"prompt": ["hi", "hi"]})
    out = df.with_columns(answer=inference_async(pl.col("prompt"), model="m"))
    # dedupe=False (default) is byte-identical to pre-#77: every row hits
    # the backend independently, even exact duplicates.
    assert out["answer"].to_list() == ["ECHO:hi", "ECHO:hi"]
    assert len(server.requests_received) == 2


# ============================================================================
# Gated real-API smoke test
# ============================================================================


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)
def test_dedupe_real_openai_identical_to_uncached(tmp_path):
    """End-to-end proof against the real OpenAI API.

    Duplicated prompts at temperature 0 -- the deduped run's output must
    equal an uncached run's output, and must have made fewer calls.
    """
    prompts = ["Reply with just the number 7."] * 3 + [
        "Reply with just the number 8."
    ] * 2
    df = pl.DataFrame({"prompt": prompts})

    uncached = df.with_columns(
        answer=inference_async(
            pl.col("prompt"),
            provider=Provider.OPENAI,
            model="gpt-4o-mini",
        )
    )

    stats = DedupeStats()
    deduped = df.with_columns(
        answer=inference_async(
            pl.col("prompt"),
            provider=Provider.OPENAI,
            model="gpt-4o-mini",
            dedupe=True,
            dedupe_stats=stats,
        )
    )

    assert deduped["answer"].to_list() == uncached["answer"].to_list()
    assert stats.calls_made < len(prompts)
    assert stats.calls_made == 2
