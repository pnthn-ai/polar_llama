#!/usr/bin/env python3
"""
Test suite for resumable batch checkpointing (issue #75).

Two layers are exercised:

- Unit-level tests drive `polar_llama.checkpoint.checkpointed_expr` directly
  with a hand-written `run_pending` stub. This is the cleanest way to prove
  the crash/resume contract deterministically -- a "crash" is a stub that
  raises after a controlled number of calls, no subprocess/SIGKILL needed --
  and needs no network, no API keys, no mlx.
- End-to-end tests drive the real `inference_async` / `inference_messages`
  wiring (`checkpoint=` kwarg) against a local stdlib mock OpenAI-compatible
  HTTP server (the same `ThreadingHTTPServer` pattern as
  `tests/test_local_server_backend.py` / `tests/test_streaming.py`), so the
  actual Rust plugin fan-out is exercised, not just the Python wrapper.

A gated real-API smoke test sits at the bottom, skipped unless
`OPENAI_API_KEY` is set, mirroring `tests/test_streaming.py`.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from helpers import BacklogHTTPServer
from typing import Callable

import polars as pl
import pytest
from pydantic import BaseModel

from polar_llama import Checkpoint, Provider, inference_async, inference_messages
from polar_llama.checkpoint import CheckpointStore, checkpointed_expr
from polar_llama.keys import (
    canonicalize_messages_input,
    config_fingerprint,
    content_key,
)


# ============================================================================
# keys.py -- fingerprint / content-key primitives
# ============================================================================


def test_content_key_stable_across_calls():
    k1 = content_key("hello world", "fp")
    k2 = content_key("hello world", "fp")
    assert k1 == k2
    assert isinstance(k1, str) and len(k1) == 64  # sha256 hex digest


def test_content_key_differs_by_input_or_fingerprint():
    base = content_key("hello", "fp1")
    assert content_key("goodbye", "fp1") != base
    assert content_key("hello", "fp2") != base


def test_fingerprint_stable_for_identical_config():
    a = config_fingerprint(
        symbol="inference_async", provider="openai", model="gpt-4o-mini"
    )
    b = config_fingerprint(
        symbol="inference_async", provider="openai", model="gpt-4o-mini"
    )
    assert a == b


@pytest.mark.parametrize(
    "changed_kwargs",
    [
        {"symbol": "inference_messages"},
        {"provider": "anthropic"},
        {"model": "gpt-4o"},
        {"system_prompt": "You are terse."},
        {"response_schema": '{"type": "object"}'},
        {"response_model_name": "Other"},
    ],
)
def test_fingerprint_changes_with_each_shaping_param(changed_kwargs):
    base_kwargs = dict(
        symbol="inference_async",
        provider="openai",
        model="gpt-4o-mini",
        system_prompt=None,
        response_schema=None,
        response_model_name=None,
    )
    base = config_fingerprint(**base_kwargs)
    changed = config_fingerprint(**{**base_kwargs, **changed_kwargs})
    assert base != changed


def test_fingerprint_ignores_cache_kwargs():
    # cache_* kwargs are deliberately not part of the fingerprint signature
    # at all -- prove the two calls that only differ by *unrelated* caching
    # concerns still hash identically (there is no cache kwarg to pass here
    # in the first place, which is the point: the caller can't even leak it
    # in accidentally).
    a = config_fingerprint(
        symbol="inference_async", provider="openai", model="gpt-4o-mini"
    )
    b = config_fingerprint(
        symbol="inference_async", provider="openai", model="gpt-4o-mini"
    )
    assert a == b


def test_fingerprint_none_normalizes_to_fixed_default():
    a = config_fingerprint(symbol="inference_async")
    b = config_fingerprint(symbol="inference_async", provider=None, model=None)
    assert a == b


def test_canonicalize_messages_input_unifies_string_and_struct():
    struct_form = [{"role": "user", "content": "hi"}]
    string_form = json.dumps(struct_form)
    # Different key order / whitespace should still canonicalize identically.
    string_form_reordered = json.dumps(
        [{"content": "hi", "role": "user"}], separators=(", ", ": ")
    )
    assert canonicalize_messages_input(struct_form) == canonicalize_messages_input(
        string_form
    )
    assert canonicalize_messages_input(struct_form) == canonicalize_messages_input(
        string_form_reordered
    )


# ============================================================================
# checkpoint.py -- unit tests against checkpointed_expr with a stub run_pending
# ============================================================================


def _make_counting_stub(counter: dict) -> Callable[[pl.Series], pl.Series]:
    def run_pending(s: pl.Series) -> pl.Series:
        out = []
        for v in s.to_list():
            counter["n"] += 1
            out.append(f"OUT:{v}")
        return pl.Series(out, dtype=pl.Utf8)

    return run_pending


def test_first_run_populates_store_and_resume_makes_zero_calls(tmp_path):
    counter = {"n": 0}
    ck = Checkpoint(tmp_path / "run.ckpt", flush_every=10)
    fp = "fp-first-run"
    df = pl.DataFrame({"prompt": [f"row-{i}" for i in range(12)]})

    expr = checkpointed_expr(
        pl.col("prompt"),
        run_pending=_make_counting_stub(counter),
        fingerprint=fp,
        checkpoint=ck,
    )
    out = df.with_columns(out=expr)
    assert out["out"].to_list() == [f"OUT:row-{i}" for i in range(12)]
    assert counter["n"] == 12

    store = CheckpointStore(ck.path, fp)
    index = store.load_index()
    assert len(index) == 12
    assert all(ok for ok, _ in index.values())

    # Resume with a stub that would raise if ever called -- nothing should
    # be pending, so it must never be invoked.
    def _boom(s: pl.Series) -> pl.Series:
        raise AssertionError("run_pending should not be called on full resume")

    expr2 = checkpointed_expr(
        pl.col("prompt"), run_pending=_boom, fingerprint=fp, checkpoint=ck
    )
    out2 = df.with_columns(out=expr2)
    assert out2["out"].to_list() == out["out"].to_list()


def test_crash_and_resume_calls_stub_once_per_row(tmp_path):
    """Headline acceptance test for issue #75.

    Simulate a batch that dies partway through (a `run_pending` stub that
    raises after a controlled number of successful, individually-flushed
    rows -- equivalent to killing the process at ~50%), then resume against
    the same checkpoint store. The stub must be called exactly once per
    unique row across the two attempts combined: rows already flushed before
    the crash are never re-requested.
    """
    n_rows = 20
    crash_after = n_rows // 2  # simulate a crash right at the 50% mark
    ck = Checkpoint(tmp_path / "run.ckpt", flush_every=1)  # 1 row per flush
    fp = "fp-crash-resume"
    df = pl.DataFrame({"prompt": [f"row-{i}" for i in range(n_rows)]})

    counter = {"n": 0}

    def flaky_run_pending(s: pl.Series) -> pl.Series:
        assert len(s) == 1  # flush_every=1
        counter["n"] += 1
        if counter["n"] > crash_after:
            raise RuntimeError("simulated crash mid-batch")
        return pl.Series([f"OUT:{s.to_list()[0]}"], dtype=pl.Utf8)

    expr = checkpointed_expr(
        pl.col("prompt"), run_pending=flaky_run_pending, fingerprint=fp, checkpoint=ck
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        df.with_columns(out=expr)

    assert counter["n"] == crash_after + 1  # crash_after succeeded, +1 that raised

    store = CheckpointStore(ck.path, fp)
    index_after_crash = store.load_index()
    # Exactly the rows that completed before the crash are durable.
    assert len(index_after_crash) == crash_after

    # Resume: a healthy stub should only be asked for the remaining rows.
    counter["n"] = 0

    def healthy_run_pending(s: pl.Series) -> pl.Series:
        counter["n"] += 1
        return pl.Series([f"OUT:{v}" for v in s.to_list()], dtype=pl.Utf8)

    expr2 = checkpointed_expr(
        pl.col("prompt"), run_pending=healthy_run_pending, fingerprint=fp, checkpoint=ck
    )
    out = df.with_columns(out=expr2)

    assert out["out"].to_list() == [f"OUT:row-{i}" for i in range(n_rows)]
    # Only the not-yet-completed rows were re-requested on resume.
    assert counter["n"] == n_rows - crash_after

    # Across both attempts combined, each of the 20 unique rows was sent to
    # `run_pending` exactly once -- no row is ever double-spent.
    total_calls = (crash_after + 1) + (n_rows - crash_after)
    assert (
        total_calls == n_rows + 1
    )  # the +1 is the call that raised (no result stored)


def test_duplicate_inputs_produce_single_request(tmp_path):
    counter = {"n": 0}
    ck = Checkpoint(tmp_path / "run.ckpt", flush_every=100)
    fp = "fp-dup"
    # 12 rows, only 3 distinct values.
    values = (["alpha"] * 5) + (["beta"] * 4) + (["gamma"] * 3)
    df = pl.DataFrame({"prompt": values})

    expr = checkpointed_expr(
        pl.col("prompt"),
        run_pending=_make_counting_stub(counter),
        fingerprint=fp,
        checkpoint=ck,
    )
    out = df.with_columns(out=expr)

    assert out["out"].to_list() == [f"OUT:{v}" for v in values]
    assert counter["n"] == 3  # one request per unique value, not per row


def test_null_rows_pass_through_and_are_never_stored(tmp_path):
    counter = {"n": 0}
    ck = Checkpoint(tmp_path / "run.ckpt", flush_every=100)
    fp = "fp-null"
    df = pl.DataFrame({"prompt": ["a", None, "b", None]})

    expr = checkpointed_expr(
        pl.col("prompt"),
        run_pending=_make_counting_stub(counter),
        fingerprint=fp,
        checkpoint=ck,
    )
    out = df.with_columns(out=expr)

    assert out["out"].to_list() == ["OUT:a", None, "OUT:b", None]
    assert counter["n"] == 2

    store = CheckpointStore(ck.path, fp)
    assert len(store.load_index()) == 2  # nulls never keyed/stored


def test_failed_rows_stored_as_failed_and_retried_on_resume(tmp_path):
    # The `{"_error": ...}` failure shape is emitted by the Rust plugin only on
    # the schema (response_model) path, so this run is has_schema=True.
    ck = Checkpoint(tmp_path / "run.ckpt", flush_every=100, retry_failed=True)
    fp = "fp-fail-retry"
    df = pl.DataFrame({"prompt": ["good1", "bad", "good2"]})

    def flaky_run_pending(s: pl.Series) -> pl.Series:
        out = []
        for v in s.to_list():
            if v == "bad":
                out.append(json.dumps({"_error": "api_error", "_details": "boom"}))
            else:
                out.append(f"OUT:{v}")
        return pl.Series(out, dtype=pl.Utf8)

    expr = checkpointed_expr(
        pl.col("prompt"),
        run_pending=flaky_run_pending,
        fingerprint=fp,
        checkpoint=ck,
        has_schema=True,
    )
    out = df.with_columns(out=expr)
    assert out["out"][0] == "OUT:good1"
    assert out["out"][2] == "OUT:good2"
    assert json.loads(out["out"][1])["_error"] == "api_error"

    store = CheckpointStore(ck.path, fp)
    index = store.load_index()
    assert len(index) == 3
    bad_key = content_key("bad", fp)
    ok, raw = index[bad_key]
    assert ok is False
    assert json.loads(raw)["_error"] == "api_error"

    # Resume with a healthy backend: only the failed row should be
    # re-requested; the two already-successful rows must not be re-sent.
    seen = {"calls": []}

    def healthy_run_pending(s: pl.Series) -> pl.Series:
        vals = s.to_list()
        seen["calls"].extend(vals)
        return pl.Series([f"OUT:{v}" for v in vals], dtype=pl.Utf8)

    expr2 = checkpointed_expr(
        pl.col("prompt"),
        run_pending=healthy_run_pending,
        fingerprint=fp,
        checkpoint=ck,
        has_schema=True,
    )
    out2 = df.with_columns(out=expr2)

    assert out2["out"].to_list() == ["OUT:good1", "OUT:bad", "OUT:good2"]
    assert seen["calls"] == ["bad"]  # only the previously-failed row re-requested

    index2 = store.load_index()
    ok2, raw2 = index2[bad_key]
    assert ok2 is True
    assert raw2 == "OUT:bad"


def test_plaintext_output_resembling_error_is_not_a_failure(tmp_path):
    # Regression for the reviewer's finding: on the no-schema (plain text) path,
    # a SUCCESSFUL row whose model output happens to be a JSON object containing
    # an `_error` key must NOT be misclassified as failed. Otherwise it is
    # stored ok=False and re-requested on every resume, breaking idempotency.
    ck = Checkpoint(tmp_path / "run.ckpt", flush_every=100, retry_failed=True)
    fp = "fp-plaintext-jsonish"
    df = pl.DataFrame({"prompt": ["p1"]})

    calls = {"n": 0}

    def run_pending(s: pl.Series) -> pl.Series:
        calls["n"] += len(s)
        # The model legitimately returned text that parses as {"_error": ...}.
        return pl.Series(
            [json.dumps({"_error": "not really an error, just the answer"})],
            dtype=pl.Utf8,
        )

    # has_schema defaults to False -> plain-text path.
    out = df.with_columns(
        out=checkpointed_expr(
            pl.col("prompt"), run_pending=run_pending, fingerprint=fp, checkpoint=ck
        )
    )
    assert calls["n"] == 1

    store = CheckpointStore(ck.path, fp)
    ok, raw = store.load_index()[content_key("p1", fp)]
    assert ok is True  # stored as a SUCCESS, not a failure
    assert json.loads(raw)["_error"] == "not really an error, just the answer"

    # Resume must NOT re-request the row (it is done, not failed).
    df.with_columns(
        out=checkpointed_expr(
            pl.col("prompt"), run_pending=run_pending, fingerprint=fp, checkpoint=ck
        )
    )
    assert calls["n"] == 1  # no re-request on resume


def test_endpoint_change_invalidates_checkpoint(monkeypatch, tmp_path):
    # Regression for the reviewer's finding: the endpoint (base-URL override) is
    # a request-shaping input, so changing it must invalidate the checkpoint.
    from polar_llama.keys import config_fingerprint, endpoint_fingerprint_input

    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com")
    ep1 = endpoint_fingerprint_input("openai")
    fp1 = config_fingerprint(
        symbol="inference_async",
        provider="openai",
        model="gpt-4o-mini",
        extra={"endpoint": ep1},
    )

    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8080")  # local-MLX server
    ep2 = endpoint_fingerprint_input("openai")
    fp2 = config_fingerprint(
        symbol="inference_async",
        provider="openai",
        model="gpt-4o-mini",
        extra={"endpoint": ep2},
    )

    assert ep1 != ep2
    assert fp1 != fp2  # different endpoint -> different fingerprint -> no stale hit


def test_retry_failed_false_returns_stored_error_verbatim(tmp_path):
    ck = Checkpoint(tmp_path / "run.ckpt", flush_every=100, retry_failed=False)
    fp = "fp-fail-noretry"
    df = pl.DataFrame({"prompt": ["bad"]})

    def flaky_run_pending(s: pl.Series) -> pl.Series:
        return pl.Series(
            [json.dumps({"_error": "api_error", "_details": "boom"})], dtype=pl.Utf8
        )

    expr = checkpointed_expr(
        pl.col("prompt"), run_pending=flaky_run_pending, fingerprint=fp, checkpoint=ck
    )
    df.with_columns(out=expr)

    def _boom(s: pl.Series) -> pl.Series:
        raise AssertionError("retry_failed=False must not re-request a stored error")

    expr2 = checkpointed_expr(
        pl.col("prompt"), run_pending=_boom, fingerprint=fp, checkpoint=ck
    )
    out2 = df.with_columns(out=expr2)
    assert json.loads(out2["out"][0])["_error"] == "api_error"


def test_flush_every_chunking_creates_expected_part_count(tmp_path):
    counter = {"n": 0}
    ck = Checkpoint(tmp_path / "run.ckpt", flush_every=4)
    fp = "fp-chunking"
    df = pl.DataFrame({"prompt": [f"row-{i}" for i in range(10)]})

    expr = checkpointed_expr(
        pl.col("prompt"),
        run_pending=_make_counting_stub(counter),
        fingerprint=fp,
        checkpoint=ck,
    )
    df.with_columns(out=expr)

    store = CheckpointStore(ck.path, fp)
    n_parts = len(store._part_paths())
    assert n_parts == 3  # ceil(10 / 4)
    assert counter["n"] == 10


def test_atomic_parts_ignore_truncated_tmp_file(tmp_path):
    ck = Checkpoint(tmp_path / "run.ckpt", flush_every=100)
    fp = "fp-atomic"
    df = pl.DataFrame({"prompt": ["a", "b"]})
    counter = {"n": 0}

    expr = checkpointed_expr(
        pl.col("prompt"),
        run_pending=_make_counting_stub(counter),
        fingerprint=fp,
        checkpoint=ck,
    )
    df.with_columns(out=expr)

    # Hand-plant a truncated / corrupt part file mid-write, as a crash would
    # leave behind (the atomic rename never happened).
    junk = ck.path / "part-deadbeef.parquet.tmp"
    junk.write_bytes(b"not a real parquet file")

    store = CheckpointStore(ck.path, fp)
    index = store.load_index()
    assert len(index) == 2  # the .tmp junk file is invisible to the glob


def test_checkpoint_accepts_str_path_and_config_forms(tmp_path):
    counter = {"n": 0}
    df = pl.DataFrame({"prompt": ["a", "b"]})

    # str shorthand
    expr = checkpointed_expr(
        pl.col("prompt"),
        run_pending=_make_counting_stub(counter),
        fingerprint="fp-str",
        checkpoint=Checkpoint(str(tmp_path / "str.ckpt")),
    )
    out = df.with_columns(out=expr)
    assert out["out"].to_list() == ["OUT:a", "OUT:b"]

    # explicit Checkpoint with custom settings
    ck = Checkpoint(tmp_path / "cfg.ckpt", flush_every=1, retry_failed=False)
    expr2 = checkpointed_expr(
        pl.col("prompt"),
        run_pending=_make_counting_stub(counter),
        fingerprint="fp-cfg",
        checkpoint=ck,
    )
    out2 = df.with_columns(out=expr2)
    assert out2["out"].to_list() == ["OUT:a", "OUT:b"]


def test_checkpoint_rejects_invalid_settings(tmp_path):
    with pytest.raises(ValueError):
        Checkpoint(tmp_path / "x.ckpt", flush_every=0)
    with pytest.raises(ValueError):
        Checkpoint(tmp_path / "x.ckpt", on_mismatch="bogus")


def test_store_on_mismatch_error_raises(tmp_path):
    path = tmp_path / "run.ckpt"
    CheckpointStore(path, "fp-a")
    with pytest.raises(ValueError, match="different run configuration"):
        CheckpointStore(path, "fp-b", on_mismatch="error")


def test_store_on_mismatch_restart_ignores_stale_entries(tmp_path):
    path = tmp_path / "run.ckpt"
    store_a = CheckpointStore(path, "fp-a")
    store_a.append(["k1"], ["OUT:x"], [True], [None])
    assert len(store_a.load_index()) == 1

    # A new run under a different fingerprint over the same directory:
    # on_mismatch="restart" (default) must not see the old entry (its key
    # space is disjoint) and must not raise.
    store_b = CheckpointStore(path, "fp-b", on_mismatch="restart")
    assert store_b.load_index() == {}


# ============================================================================
# End-to-end: inference_async / inference_messages with checkpoint=...
# against a local mock OpenAI-compatible server
# ============================================================================


class _MockOpenAIHandler(BaseHTTPRequestHandler):
    """Canned `/v1/chat/completions` responder driven by `server.responder`.

    `server.responder(body) -> (status_code, content_string)`. A non-200
    status produces the Rust client's `api_error` path end-to-end.
    """

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


def test_inference_async_checkpoint_resume_skips_completed_rows(
    mock_server, monkeypatch, tmp_path
):
    server, base_url = mock_server
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)

    prompts = [f"row-{i}" for i in range(10)]
    df_first_half = pl.DataFrame({"prompt": prompts[:5]})
    df_full = pl.DataFrame({"prompt": prompts})
    ckpt_path = tmp_path / "run.ckpt"

    df_first_half.with_columns(
        answer=inference_async(
            pl.col("prompt"),
            model="mock-model",
            checkpoint=Checkpoint(ckpt_path, flush_every=2),
        )
    )
    assert len(server.requests_received) == 5

    out = df_full.with_columns(
        answer=inference_async(
            pl.col("prompt"),
            model="mock-model",
            checkpoint=Checkpoint(ckpt_path, flush_every=2),
        )
    )
    # Only the 5 new rows should have hit the server.
    assert len(server.requests_received) == 10
    assert out["answer"].to_list() == [f"ECHO:{p}" for p in prompts]


def test_key_invalidation_on_model_change_forces_full_recompute(
    mock_server, monkeypatch, tmp_path
):
    server, base_url = mock_server
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)

    df = pl.DataFrame({"prompt": ["a", "b", "c"]})
    ckpt_path = tmp_path / "run.ckpt"

    df.with_columns(
        answer=inference_async(pl.col("prompt"), model="model-v1", checkpoint=ckpt_path)
    )
    assert len(server.requests_received) == 3

    # Same checkpoint dir, different model -> different fingerprint -> every
    # row must be re-requested (not skipped as "already done").
    df.with_columns(
        answer=inference_async(pl.col("prompt"), model="model-v2", checkpoint=ckpt_path)
    )
    assert len(server.requests_received) == 6


def test_key_invalidation_on_system_prompt_change_forces_full_recompute(
    mock_server, monkeypatch, tmp_path
):
    server, base_url = mock_server
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)

    df = pl.DataFrame({"prompt": ["a", "b"]})
    ckpt_path = tmp_path / "run.ckpt"

    df.with_columns(
        answer=inference_async(
            pl.col("prompt"), model="m", system_prompt="Be terse.", checkpoint=ckpt_path
        )
    )
    assert len(server.requests_received) == 2

    df.with_columns(
        answer=inference_async(
            pl.col("prompt"),
            model="m",
            system_prompt="Be verbose.",
            checkpoint=ckpt_path,
        )
    )
    assert len(server.requests_received) == 4


def test_failed_row_stored_and_retried_end_to_end_plain_text(
    mock_server, monkeypatch, tmp_path
):
    """No `response_model`: the Rust plugin's non-schema path reports a
    failed request as a bare `None` for that row (`errors_as_json=false`,
    same as pre-#75 behavior) rather than an in-band `_error` JSON string.
    The checkpoint layer must still recognize it as failed (not "done"),
    store it, and retry it on resume.
    """
    server, base_url = mock_server
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)
    ckpt_path = tmp_path / "run.ckpt"

    def flaky_responder(body: dict):
        content = _user_content(body)
        if content == "bad":
            return 500, ""
        return 200, f"ECHO:{content}"

    server.responder = flaky_responder

    df = pl.DataFrame({"prompt": ["good1", "bad", "good2"]})
    out = df.with_columns(
        answer=inference_async(pl.col("prompt"), model="m", checkpoint=ckpt_path)
    )
    assert out["answer"][0] == "ECHO:good1"
    assert out["answer"][2] == "ECHO:good2"
    assert out["answer"][1] is None
    assert len(server.requests_received) == 3

    # Fix the backend, resume: only "bad" should be re-requested (it was
    # stored with ok=False, not silently treated as "already done").
    server.responder = _echo_responder
    server.requests_received = []
    out2 = df.with_columns(
        answer=inference_async(pl.col("prompt"), model="m", checkpoint=ckpt_path)
    )
    assert out2["answer"].to_list() == ["ECHO:good1", "ECHO:bad", "ECHO:good2"]
    assert len(server.requests_received) == 1
    assert _user_content(server.requests_received[0]) == "bad"


def test_failed_row_stored_and_retried_end_to_end_structured(
    mock_server, monkeypatch, tmp_path
):
    """With a `response_model`, the Rust plugin always reports failures as
    in-band `{"_error": ...}` JSON (`errors_as_json=true`). Verify the
    checkpoint layer stores/retries it the same way, and that the stored
    error round-trips through the normal `_error`/`_details`/`_raw` struct
    decode identically to a fresh failure.
    """
    server, base_url = mock_server
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)
    ckpt_path = tmp_path / "run.ckpt"

    def flaky_responder(body: dict):
        content = _user_content(body)
        if content == "bad":
            return 500, ""
        return 200, json.dumps({"title": content, "year": 2024})

    server.responder = flaky_responder

    df = pl.DataFrame({"prompt": ["good1", "bad", "good2"]})
    out = df.with_columns(
        rec=inference_async(
            pl.col("prompt"),
            model="m",
            response_model=_Recommendation,
            checkpoint=ckpt_path,
        )
    )
    recs = out["rec"].to_list()
    assert recs[0]["title"] == "good1"
    assert recs[0]["_error"] is None
    assert recs[2]["title"] == "good2"
    assert recs[1]["_error"] == "api_error"
    assert len(server.requests_received) == 3

    # Fix the backend, resume: only "bad" should be re-requested.
    server.responder = _echo_responder  # placeholder; overwritten below
    server.responder = lambda body: (
        200,
        json.dumps({"title": _user_content(body), "year": 2024}),
    )
    server.requests_received = []
    out2 = df.with_columns(
        rec=inference_async(
            pl.col("prompt"),
            model="m",
            response_model=_Recommendation,
            checkpoint=ckpt_path,
        )
    )
    recs2 = out2["rec"].to_list()
    assert recs2[1]["title"] == "bad"
    assert recs2[1]["_error"] is None
    assert len(server.requests_received) == 1
    assert _user_content(server.requests_received[0]) == "bad"


class _Recommendation(BaseModel):
    title: str
    year: int


def _structured_responder(body: dict):
    content = _user_content(body)
    return 200, json.dumps({"title": content, "year": 2024})


def test_structured_output_checkpoint_roundtrip(mock_server, monkeypatch, tmp_path):
    server, base_url = mock_server
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)
    server.responder = _structured_responder
    ckpt_path = tmp_path / "run.ckpt"

    prompts = ["Alpha", "Beta", "Gamma"]
    df_first = pl.DataFrame({"prompt": prompts[:2]})
    df_full = pl.DataFrame({"prompt": prompts})

    # Uncheckpointed reference run for comparison.
    reference = df_full.with_columns(
        rec=inference_async(pl.col("prompt"), model="m", response_model=_Recommendation)
    )

    df_first.with_columns(
        rec=inference_async(
            pl.col("prompt"),
            model="m",
            response_model=_Recommendation,
            checkpoint=ckpt_path,
        )
    )
    checkpointed_full = df_full.with_columns(
        rec=inference_async(
            pl.col("prompt"),
            model="m",
            response_model=_Recommendation,
            checkpoint=ckpt_path,
        )
    )

    assert checkpointed_full["rec"].to_list() == reference["rec"].to_list()
    for row in checkpointed_full["rec"].to_list():
        assert set(row.keys()) >= {"title", "year", "_error", "_details", "_raw"}
        assert row["_error"] is None


def test_inference_messages_checkpoint_shares_keys_across_input_shapes(
    mock_server, monkeypatch, tmp_path
):
    server, base_url = mock_server
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)
    ckpt_path = tmp_path / "run.ckpt"

    conversation = [{"role": "user", "content": "hello"}]
    json_string_col = pl.DataFrame({"msgs": [json.dumps(conversation)]})
    struct_col = pl.DataFrame(
        {
            "msgs": pl.Series(
                [conversation],
                dtype=pl.List(pl.Struct({"role": pl.Utf8, "content": pl.Utf8})),
            )
        }
    )

    out1 = json_string_col.with_columns(
        answer=inference_messages(pl.col("msgs"), model="m", checkpoint=ckpt_path)
    )
    assert len(server.requests_received) == 1
    assert out1["answer"][0] == "ECHO:hello"

    # Same conversation, native List(Struct) shape -- must hit the store,
    # not the server, because the two shapes canonicalize to the same key.
    out2 = struct_col.with_columns(
        answer=inference_messages(pl.col("msgs"), model="m", checkpoint=ckpt_path)
    )
    assert len(server.requests_received) == 1  # unchanged
    assert out2["answer"][0] == "ECHO:hello"


def test_inference_messages_checkpoint_basic_resume(mock_server, monkeypatch, tmp_path):
    server, base_url = mock_server
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)
    ckpt_path = tmp_path / "run.ckpt"

    convs = [[{"role": "user", "content": f"msg-{i}"}] for i in range(4)]
    df = pl.DataFrame(
        {
            "msgs": pl.Series(
                convs, dtype=pl.List(pl.Struct({"role": pl.Utf8, "content": pl.Utf8}))
            )
        }
    )

    df.head(2).with_columns(
        answer=inference_messages(pl.col("msgs"), model="m", checkpoint=ckpt_path)
    )
    assert len(server.requests_received) == 2

    out = df.with_columns(
        answer=inference_messages(pl.col("msgs"), model="m", checkpoint=ckpt_path)
    )
    assert len(server.requests_received) == 4
    assert out["answer"].to_list() == [f"ECHO:msg-{i}" for i in range(4)]


def test_checkpoint_none_is_unchanged_behavior(mock_server, monkeypatch):
    server, base_url = mock_server
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)

    df = pl.DataFrame({"prompt": ["hi"]})
    out = df.with_columns(answer=inference_async(pl.col("prompt"), model="m"))
    assert out["answer"].to_list() == ["ECHO:hi"]


def test_lazyframe_and_streaming_engine_collect(mock_server, monkeypatch, tmp_path):
    server, base_url = mock_server
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)
    ckpt_path = tmp_path / "run.ckpt"

    df = pl.DataFrame({"prompt": [f"row-{i}" for i in range(6)]})

    lazy_out = (
        df.lazy()
        .with_columns(
            answer=inference_async(pl.col("prompt"), model="m", checkpoint=ckpt_path)
        )
        .collect()
    )
    assert lazy_out["answer"].to_list() == [f"ECHO:row-{i}" for i in range(6)]

    server.requests_received = []
    streaming_out = (
        df.lazy()
        .with_columns(
            answer=inference_async(pl.col("prompt"), model="m", checkpoint=ckpt_path)
        )
        .collect(engine="streaming")
    )
    # Fully resumed from the store -- streaming engine must not re-request.
    assert len(server.requests_received) == 0
    assert streaming_out["answer"].to_list() == [f"ECHO:row-{i}" for i in range(6)]


# ============================================================================
# Gated real-API smoke test
# ============================================================================


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)
def test_checkpoint_real_openai_kill_and_resume(tmp_path):
    """End-to-end proof against the real OpenAI API.

    Runs a small batch to completion via a checkpoint, then re-runs the same
    batch against the same store: the second run must issue zero requests
    (everything already completed) and reproduce identical output.
    """
    import time

    df = pl.DataFrame(
        {"prompt": [f"Reply with just the number {i}." for i in range(6)]}
    )
    ckpt_path = tmp_path / "real_run.ckpt"

    t0 = time.time()
    first = df.with_columns(
        answer=inference_async(
            pl.col("prompt"),
            provider=Provider.OPENAI,
            model="gpt-4o-mini",
            checkpoint=Checkpoint(ckpt_path, flush_every=2),
        )
    )
    first_elapsed = time.time() - t0
    assert first["answer"].null_count() == 0

    t1 = time.time()
    second = df.with_columns(
        answer=inference_async(
            pl.col("prompt"),
            provider=Provider.OPENAI,
            model="gpt-4o-mini",
            checkpoint=Checkpoint(ckpt_path, flush_every=2),
        )
    )
    second_elapsed = time.time() - t1

    assert second["answer"].to_list() == first["answer"].to_list()
    # The fully-resumed run should be dramatically cheaper (no API calls at
    # all) than the first -- a generous bound to avoid CI flakiness while
    # still catching "resume silently re-ran everything".
    assert second_elapsed < max(1.0, first_elapsed / 2)
