#!/usr/bin/env python3
"""
Test suite for per-row usage/cost accounting (issue #76).

CI-safe: exercises the real `inference_async`/`inference_messages` Rust
plugin fan-out against a local stdlib mock OpenAI-compatible HTTP server
(the same `ThreadingHTTPServer` pattern as
`tests/test_checkpointing.py::_MockOpenAIHandler`), and the in-process MLX
seam via `FakeEngine` injection (the same pattern as
`tests/test_local_engine.py`). No API keys, no network, no mlx/GPU.

A gated real-API proof sits at the bottom, skipped unless `OPENAI_API_KEY`
and `ANTHROPIC_API_KEY` are set, mirroring `tests/test_checkpointing.py`.
"""

from __future__ import annotations

import json
import os
import threading
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from helpers import BacklogHTTPServer
from typing import Optional

import polars as pl
import pytest
from pydantic import BaseModel

from polar_llama import (
    USAGE_DTYPE,
    Checkpoint,
    Provider,
    inference_async,
    inference_messages,
)
from polar_llama import pricing
from polar_llama.local.engine import FakeEngine, clear_registry, register_engine
from polar_llama.local.expr import inference_local

# ============================================================================
# Mock OpenAI-compatible server with a configurable `usage` block
# ============================================================================


class _MockUsageHandler(BaseHTTPRequestHandler):
    """`/v1/chat/completions` responder driven by `server.responder`.

    `server.responder(body) -> (status_code, content_string, usage_dict_or_None)`.
    """

    def log_message(self, format, *args):  # noqa: A002
        pass

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        body = json.loads(raw.decode("utf-8")) if raw else {}

        with self.server.lock:  # type: ignore[attr-defined]
            self.server.requests_received.append(body)  # type: ignore[attr-defined]

        status, content, usage = self.server.responder(body)  # type: ignore[attr-defined]

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
        if usage is not None:
            payload["usage"] = usage
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
    return 200, f"ECHO:{_user_content(body)}", None


@pytest.fixture
def mock_server():
    server = BacklogHTTPServer(("127.0.0.1", 0), _MockUsageHandler)
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
def _mock_env(request, monkeypatch):
    # The gated real-API test needs the caller's actual credentials. Without
    # this opt-out the autouse fixture overwrote OPENAI_API_KEY with
    # "test-key" for EVERY test in the module, so that test could only ever
    # 401 -- it looked healthy purely because it is normally skipped, and
    # started failing the moment another module's load_dotenv() put real keys
    # in the environment and un-skipped it.
    if request.node.get_closest_marker("real_api"):
        # A live test must reach the real provider, so make sure no mock
        # base URL leaked in from an earlier module.
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
        yield
        return
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    yield
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)


@pytest.fixture(autouse=True)
def _clean_pricing_registry():
    """Isolate the process-wide price registry + unknown-model warn-once set."""
    saved_registry = dict(pricing._REGISTRY)
    saved_warned = set(pricing._WARNED_UNKNOWN)
    yield
    pricing._REGISTRY.clear()
    pricing._REGISTRY.update(saved_registry)
    pricing._WARNED_UNKNOWN.clear()
    pricing._WARNED_UNKNOWN.update(saved_warned)


# ============================================================================
# 1. usage=False regression: byte-identical dtype (no envelope leakage)
# ============================================================================


def test_usage_false_output_is_plain_utf8(mock_server, monkeypatch):
    server, base_url = mock_server
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)

    df = pl.DataFrame({"prompt": ["hello", "world"]})
    out = df.with_columns(answer=inference_async(pl.col("prompt"), model="mock-model"))

    assert out["answer"].dtype == pl.Utf8
    assert out["answer"].to_list() == ["ECHO:hello", "ECHO:world"]


def test_usage_false_with_response_model_is_still_plain_struct(
    mock_server, monkeypatch
):
    server, base_url = mock_server
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)

    class Answer(BaseModel):
        value: int

    def responder(body):
        return 200, json.dumps({"value": 7}), None

    server.responder = responder

    df = pl.DataFrame({"prompt": ["q"]})
    out = df.with_columns(
        answer=inference_async(
            pl.col("prompt"), model="mock-model", response_model=Answer
        )
    )
    dtype = out["answer"].dtype
    assert isinstance(dtype, pl.Struct)
    field_names = {f.name for f in dtype.fields}
    assert "usage" not in field_names
    assert out["answer"].struct.field("value")[0] == 7


# ============================================================================
# 2. usage=True: full struct populated, cost_usd exact to the cent (>=6 dp)
# ============================================================================


def test_usage_true_full_struct_and_exact_cost(mock_server, monkeypatch):
    server, base_url = mock_server
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)

    def responder(body):
        usage = {
            "prompt_tokens": 1200,
            "completion_tokens": 34,
            "prompt_tokens_details": {"cached_tokens": 1000},
        }
        return 200, f"ECHO:{_user_content(body)}", usage

    server.responder = responder

    df = pl.DataFrame({"prompt": ["hi"]})
    out = df.with_columns(
        r=inference_async(
            pl.col("prompt"), provider=Provider.OPENAI, model="gpt-4o-mini", usage=True
        )
    )

    dtype = out["r"].dtype
    assert isinstance(dtype, pl.Struct)
    field_names = {f.name for f in dtype.fields}
    assert field_names == {"response", "usage"}

    row = out["r"][0]
    assert row["response"] == "ECHO:hi"
    u = row["usage"]
    assert u["input_tokens"] == 1200
    assert u["output_tokens"] == 34
    assert u["cached_tokens"] == 1000
    assert u["latency_ms"] is not None and u["latency_ms"] >= 0

    # Hand-computed: non_cached=200 tokens @ $0.15/1M, cached=1000 @
    # $0.075/1M, output=34 @ $0.60/1M (packaged gpt-4o-mini price).
    expected = (200 * 0.15 + 1000 * 0.075 + 34 * 0.60) / 1_000_000.0
    assert u["cost_usd"] == pytest.approx(expected, abs=1e-9)
    assert round(u["cost_usd"], 6) == round(expected, 6)


def test_usage_true_no_usage_block_nulls_tokens_not_cost_exception(
    mock_server, monkeypatch
):
    server, base_url = mock_server
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)

    server.responder = _echo_responder  # no "usage" key in the mock response

    df = pl.DataFrame({"prompt": ["hi"]})
    out = df.with_columns(
        r=inference_async(
            pl.col("prompt"), provider=Provider.OPENAI, model="gpt-4o-mini", usage=True
        )
    )
    u = out["r"][0]["usage"]
    assert u["input_tokens"] is None
    assert u["output_tokens"] is None
    assert u["cached_tokens"] is None
    assert u["latency_ms"] is not None  # always measured Rust-side
    assert u["cost_usd"] is None  # no exception, just null


# ============================================================================
# 3. usage=True + response_model interop
# ============================================================================


class _Answer(BaseModel):
    value: int


def test_usage_true_with_response_model_interop(mock_server, monkeypatch):
    server, base_url = mock_server
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)

    def responder(body):
        content = _user_content(body)
        usage = {"prompt_tokens": 10, "completion_tokens": 5}
        if "bad" in content:
            return 200, "not valid json", usage
        return 200, json.dumps({"value": 42}), usage

    server.responder = responder

    df = pl.DataFrame({"prompt": ["good-row", "bad-row"]})
    out = df.with_columns(
        r=inference_async(
            pl.col("prompt"),
            provider=Provider.OPENAI,
            model="gpt-4o-mini",
            response_model=_Answer,
            usage=True,
        )
    )

    dtype = out["r"].dtype
    field_names = {f.name for f in dtype.fields}
    assert field_names == {"response", "usage"}

    good_row = out["r"][0]
    assert good_row["response"]["value"] == 42
    assert good_row["response"]["_error"] is None
    assert good_row["usage"]["input_tokens"] == 10
    assert good_row["usage"]["output_tokens"] == 5

    bad_row = out["r"][1]
    assert bad_row["response"]["_error"] == "validation_failed"
    # Usage is still present on a validation-failure row.
    assert bad_row["usage"]["input_tokens"] == 10
    assert bad_row["usage"]["output_tokens"] == 5


# ============================================================================
# 4. Unknown model -> null cost_usd + exactly one UserWarning across 2 calls
# ============================================================================


def test_unknown_model_null_cost_and_warns_once(mock_server, monkeypatch):
    server, base_url = mock_server
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)
    server.responder = _echo_responder

    df = pl.DataFrame({"prompt": ["a"]})

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        df.with_columns(
            r=inference_async(
                pl.col("prompt"),
                provider=Provider.OPENAI,
                model="totally-unknown-model-xyz",
                usage=True,
            )
        )
        df.with_columns(
            r=inference_async(
                pl.col("prompt"),
                provider=Provider.OPENAI,
                model="totally-unknown-model-xyz",
                usage=True,
            )
        )
        user_warnings = [w for w in rec if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 1
        assert "totally-unknown-model-xyz" in str(user_warnings[0].message)


# ============================================================================
# 5. price_table= override and register_model_price registry override
# ============================================================================


def test_price_table_override_changes_cost(mock_server, monkeypatch):
    server, base_url = mock_server
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)

    def responder(body):
        return 200, "text", {"prompt_tokens": 100, "completion_tokens": 10}

    server.responder = responder

    df = pl.DataFrame({"prompt": ["a"]})
    out = df.with_columns(
        r=inference_async(
            pl.col("prompt"),
            provider=Provider.OPENAI,
            model="custom-model-1",
            usage=True,
            price_table={
                "openai": {
                    "custom-model-1": {"input_per_1m": 1.0, "output_per_1m": 2.0}
                }
            },
        )
    )
    expected = (100 * 1.0 + 10 * 2.0) / 1_000_000.0
    assert out["r"][0]["usage"]["cost_usd"] == pytest.approx(expected, abs=1e-12)


def test_register_model_price_registry_override(mock_server, monkeypatch):
    server, base_url = mock_server
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)

    def responder(body):
        return 200, "text", {"prompt_tokens": 100, "completion_tokens": 10}

    server.responder = responder

    pricing.register_model_price(
        "openai", "custom-model-2", input_per_1m=3.0, output_per_1m=6.0
    )

    df = pl.DataFrame({"prompt": ["a"]})
    out = df.with_columns(
        r=inference_async(
            pl.col("prompt"),
            provider=Provider.OPENAI,
            model="custom-model-2",
            usage=True,
        )
    )
    expected = (100 * 3.0 + 10 * 6.0) / 1_000_000.0
    assert out["r"][0]["usage"]["cost_usd"] == pytest.approx(expected, abs=1e-12)


# ============================================================================
# 6. Null-row and empty-series behavior
# ============================================================================


def test_null_row_decodes_to_null_struct(mock_server, monkeypatch):
    server, base_url = mock_server
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)
    server.responder = _echo_responder

    df = pl.DataFrame({"prompt": ["hi", None]})
    out = df.with_columns(
        r=inference_async(pl.col("prompt"), model="mock-model", usage=True)
    )
    assert out["r"][0] is not None
    assert out["r"][1] is None


def test_empty_series_with_usage(mock_server, monkeypatch):
    server, base_url = mock_server
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)
    server.responder = _echo_responder

    df = pl.DataFrame({"prompt": pl.Series([], dtype=pl.Utf8)})
    out = df.with_columns(
        r=inference_async(pl.col("prompt"), model="mock-model", usage=True)
    )
    assert out.height == 0


# ============================================================================
# 7. Checkpoint + usage=True interop guard
# ============================================================================


def test_checkpoint_and_usage_together_raises(tmp_path):
    with pytest.raises(ValueError, match="usage=True"):
        inference_async(
            pl.col("prompt"),
            usage=True,
            checkpoint=Checkpoint(tmp_path / "run.ckpt"),
        )
    with pytest.raises(ValueError, match="usage=True"):
        inference_messages(
            pl.col("messages"),
            usage=True,
            checkpoint=Checkpoint(tmp_path / "run2.ckpt"),
        )


# ============================================================================
# 8. inference_messages usage=True + cache=True (cached_tokens surfaced)
# ============================================================================


def test_inference_messages_usage_true(mock_server, monkeypatch):
    server, base_url = mock_server
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)

    def responder(body):
        usage = {
            "prompt_tokens": 500,
            "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 400},
        }
        return 200, f"ECHO:{_user_content(body)}", usage

    server.responder = responder

    df = pl.DataFrame(
        {
            "messages": [
                json.dumps([{"role": "user", "content": "hello"}]),
            ]
        }
    )
    out = df.with_columns(
        r=inference_messages(
            pl.col("messages"),
            provider=Provider.OPENAI,
            model="gpt-4o-mini",
            usage=True,
        )
    )
    u = out["r"][0]["usage"]
    assert u["input_tokens"] == 500
    assert u["cached_tokens"] == 400
    assert u["output_tokens"] == 20


def test_inference_messages_usage_true_with_cache(mock_server, monkeypatch):
    # Regression guard for the cache=True + usage=True path: cache warming goes
    # through fetch_with_cache_warming (src/utils.rs), which has its own
    # envelope-wrap call sites separate from the plain path.
    server, base_url = mock_server
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)

    def responder(body):
        usage = {"prompt_tokens": 300, "completion_tokens": 15}
        return 200, f"ECHO:{_user_content(body)}", usage

    server.responder = responder

    df = pl.DataFrame(
        {"messages": [json.dumps([{"role": "user", "content": "hi there"}])]}
    )
    out = df.with_columns(
        r=inference_messages(
            pl.col("messages"),
            provider=Provider.OPENAI,
            model="gpt-4o-mini",
            usage=True,
            cache=True,
        )
    )
    u = out["r"][0]["usage"]
    assert u["input_tokens"] == 300
    assert u["output_tokens"] == 15
    # The response still round-trips correctly through the cache path.
    assert out["r"][0]["response"] == "ECHO:hi there"


# ============================================================================
# 8b. Anthropic wire format: parse_usage cache-fold normalization (runnable)
# ============================================================================


class _MockAnthropicHandler(BaseHTTPRequestHandler):
    """`/v1/messages` responder returning Anthropic message + usage shape."""

    def log_message(self, format, *args):  # noqa: A002
        pass

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0) or 0)
        self.rfile.read(length) if length else b""
        usage = self.server.usage_block  # type: ignore[attr-defined]
        payload = {
            "id": "msg_fake",
            "type": "message",
            "role": "assistant",
            "model": "claude-mock",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": usage,
        }
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def anthropic_mock_server():
    server = BacklogHTTPServer(("127.0.0.1", 0), _MockAnthropicHandler)
    server.usage_block = {}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield server, f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_anthropic_usage_cache_fold_normalization(anthropic_mock_server, monkeypatch):
    # Anthropic's input_tokens EXCLUDES cache read/write; the Rust parse_usage
    # folds both into input_tokens so cached_tokens <= input_tokens uniformly.
    server, base_url = anthropic_mock_server
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", base_url)
    server.usage_block = {
        "input_tokens": 100,  # base (excludes cache)
        "output_tokens": 25,
        "cache_read_input_tokens": 400,
        "cache_creation_input_tokens": 50,
    }

    df = pl.DataFrame({"prompt": ["hello"]})
    out = df.with_columns(
        r=inference_async(
            pl.col("prompt"),
            provider=Provider.ANTHROPIC,
            model="claude-mock",
            usage=True,
        )
    )
    u = out["r"][0]["usage"]
    # input_tokens normalized to base + cache_read + cache_creation.
    assert u["input_tokens"] == 100 + 400 + 50
    assert u["cached_tokens"] == 400  # the read subset
    assert u["output_tokens"] == 25
    assert u["cached_tokens"] <= u["input_tokens"]  # the invariant


# ============================================================================
# 9. MLX in-process engine via FakeEngine: tokens + latency + cost=0.0
# ============================================================================


@pytest.fixture(autouse=False)
def _clean_local_registry():
    clear_registry()
    yield
    clear_registry()


def test_inference_local_in_process_usage_true(_clean_local_registry):
    register_engine("dummy-model", FakeEngine("dummy-model"), engine="in_process")

    df = pl.DataFrame({"prompt": ["one two three", "four"]})
    out = df.with_columns(
        r=inference_local(
            pl.col("prompt"), model="dummy-model", engine="in_process", usage=True
        )
    )

    dtype = out["r"].dtype
    field_names = {f.name for f in dtype.fields}
    assert field_names == {"response", "usage"}

    row0 = out["r"][0]
    assert row0["response"] == "echo:one two three"
    u0 = row0["usage"]
    # FakeEngine.generate_with_usage: whitespace-token counts.
    assert u0["input_tokens"] == len("one two three".split())
    assert u0["output_tokens"] == len("echo:one two three".split())
    assert u0["latency_ms"] is not None and u0["latency_ms"] >= 0
    assert u0["cached_tokens"] is None
    assert u0["cost_usd"] == 0.0

    row1 = out["r"][1]
    assert row1["usage"]["cost_usd"] == 0.0


def test_inference_local_in_process_usage_false_unchanged(_clean_local_registry):
    register_engine("dummy-model", FakeEngine("dummy-model"), engine="in_process")

    df = pl.DataFrame({"prompt": ["hello"]})
    out = df.with_columns(
        r=inference_local(pl.col("prompt"), model="dummy-model", engine="in_process")
    )
    assert out["r"].dtype == pl.Utf8
    assert out["r"][0] == "echo:hello"


# ============================================================================
# Gated real-API proof (env-gated, not run in CI)
# ============================================================================


def _accounted_rows(column, provider: str) -> list:
    """The rows that actually came back, failing clearly if none did.

    An API error leaves that row's whole `usage=True` struct null, so indexing
    it would otherwise die with a bare ``TypeError: 'NoneType' object is not
    subscriptable``. Rows the provider refused are skipped, because a
    partially rate-limited batch is an artifact of the account's tier rather
    than a usage-accounting bug -- but every row that DID come back still has
    to account correctly.

    A batch where *nothing* came back fails loudly and deliberately is NOT a
    skip: this test exists to prove the live path works, and skipping on any
    failure would hide an auth or connectivity regression -- exactly the class
    of bug the autouse-fixture mix-up above was.
    """
    rows = [row for row in column if row is not None]
    if not rows:
        raise AssertionError(
            f"No response from {provider} for any row. The provider's own error "
            f"was printed to stderr above -- check the API key is valid and the "
            f"account has credit. (Run with `-m \"not real_api\"` to skip the "
            f"live-billing tests entirely.)"
        )
    return rows


@pytest.fixture
def _gentle_concurrency(monkeypatch):
    """Keep the live tests inside a modest provider rate limit.

    The default fan-out (64 in flight) trips 429s on a low-tier account, which
    shows up as a scatter of null rows rather than anything to do with usage
    accounting.
    """
    monkeypatch.setenv("POLAR_LLAMA_MAX_CONCURRENCY", "2")


@pytest.mark.real_api
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="requires OPENAI_API_KEY"
)
def test_real_api_usage_accounting_openai(_gentle_concurrency):
    df = pl.DataFrame(
        {"prompt": [f"Say the number {i} and nothing else." for i in range(6)]}
    )

    out = df.with_columns(
        r=inference_async(
            pl.col("prompt"), provider=Provider.OPENAI, model="gpt-4o-mini", usage=True
        )
    )
    for row in _accounted_rows(out["r"], "OpenAI"):
        u = row["usage"]
        assert u["input_tokens"] > 0
        assert u["output_tokens"] > 0
        assert u["cost_usd"] is not None and u["cost_usd"] >= 0


@pytest.mark.real_api
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="requires ANTHROPIC_API_KEY"
)
def test_real_api_usage_accounting_anthropic(_gentle_concurrency):
    """Split from the OpenAI proof: one provider being out of credit (or having
    a revoked key) must not take down the other provider's verification."""
    df = pl.DataFrame(
        {"prompt": [f"Say the number {i} and nothing else." for i in range(6)]}
    )

    out = df.with_columns(
        r=inference_async(
            pl.col("prompt"),
            provider=Provider.ANTHROPIC,
            model="claude-haiku-4-5",
            usage=True,
        )
    )
    rows = _accounted_rows(out["r"], "Anthropic")

    total_cost = 0.0
    for row in rows:
        u = row["usage"]
        assert u["input_tokens"] > 0
        assert u["output_tokens"] > 0
        assert u["cost_usd"] is not None
        total_cost += u["cost_usd"]

    # Verify the cost formula against the provider's own reported usage
    # (exact billing isn't queryable in-test, so this checks the formula,
    # not a ground-truth invoice).
    price = pricing.resolve_price("anthropic", "claude-haiku-4-5")
    hand_total = sum(
        pricing.compute_cost(
            row["usage"]["input_tokens"],
            row["usage"]["output_tokens"],
            row["usage"]["cached_tokens"],
            price,
        )
        for row in rows
    )
    assert total_cost == pytest.approx(hand_total, rel=0.01, abs=0.01)
