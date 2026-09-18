#!/usr/bin/env python3
"""
Test suite for `inference_stream` (issue #74).

The bulk of these tests run against a local stdlib mock SSE server (no
network, no API keys, no mlx) so they run in CI. A couple of gated real-API
smoke tests at the bottom are skipped unless the relevant API key is set,
mirroring the pattern in tests/test_embeddings.py.
"""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from helpers import BacklogHTTPServer

import polars as pl
import pytest

from polar_llama import STREAM_RESPONSE_DTYPE, inference_stream


# ============================================================================
# Mock SSE server
# ============================================================================


def _sleep(seconds: float):
    """A script entry that makes the mock server pause before continuing."""
    return ("__sleep__", seconds)


def openai_delta(text: str) -> bytes:
    return (
        "data: " + json.dumps({"choices": [{"delta": {"content": text}}]}) + "\n\n"
    ).encode()


def openai_done() -> bytes:
    return b"data: [DONE]\n\n"


def openai_error(message: str) -> bytes:
    return ("data: " + json.dumps({"error": {"message": message}}) + "\n\n").encode()


def anthropic_delta(text: str) -> bytes:
    payload = {
        "type": "content_block_delta",
        "delta": {"type": "text_delta", "text": text},
    }
    return f"event: content_block_delta\ndata: {json.dumps(payload)}\n\n".encode()


def anthropic_stop() -> bytes:
    return b'event: message_stop\ndata: {"type":"message_stop"}\n\n'


def anthropic_error(message: str) -> bytes:
    payload = {"type": "error", "error": {"message": message}}
    return f"event: error\ndata: {json.dumps(payload)}\n\n".encode()


class _SSEHandler(BaseHTTPRequestHandler):
    """Streams a scripted sequence of raw SSE bytes, chosen by the content of
    the request's last message. Deliberately left on HTTP/1.0 (the
    BaseHTTPRequestHandler default): with no Content-Length header, the
    client reads the body until the connection closes -- which is exactly the
    "EOF without a terminator frame" case some tests need, and requires no
    chunked-encoding bookkeeping for the rest.
    """

    def log_message(self, format, *args):  # noqa: A002 - silence request logging
        pass

    def _script_key(self, payload: dict):
        messages = payload.get("messages") or []
        if messages:
            return messages[-1].get("content")
        return None

    def do_POST(self):  # noqa: N802 - http.server naming convention
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            payload = {}

        key = self._script_key(payload)
        script = self.server.scripts.get(
            key, self.server.scripts.get("__default__", [])
        )

        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            for item in script:
                if isinstance(item, tuple) and item and item[0] == "__sleep__":
                    time.sleep(item[1])
                    continue
                self.wfile.write(item)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            # Client aborted the stream (e.g. cancellation test) -- expected.
            return


@pytest.fixture
def sse_server():
    server = BacklogHTTPServer(("127.0.0.1", 0), _SSEHandler)
    server.daemon_threads = True
    server.scripts = {}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield server, base_url
    finally:
        server.shutdown()
        thread.join(timeout=5)


@pytest.fixture(autouse=True)
def _dummy_keys(monkeypatch):
    # Every provider path reads its key via std env lookup; the mock server
    # doesn't check auth, but set dummy values so nothing falls through to a
    # real, unset-key code path.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")


# ============================================================================
# P1: multi-chunk assembly (OpenAI)
# ============================================================================


def test_multi_chunk_assembly_openai(sse_server, monkeypatch):
    server, base_url = sse_server
    server.scripts["Hello"] = [
        openai_delta("Hello"),
        openai_delta(" world"),
        openai_delta("!"),
        openai_done(),
    ]
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)

    df = pl.DataFrame({"prompt": ["Hello"]})
    result = df.with_columns(
        response=inference_stream(pl.col("prompt"), provider="openai")
    )

    assert result["response"].dtype == STREAM_RESPONSE_DTYPE
    row = result["response"][0]
    assert row == {"text": "Hello world!", "finished": True}


# ============================================================================
# P2: Anthropic message_stop terminator
# ============================================================================


def test_anthropic_stream_message_stop(sse_server, monkeypatch):
    server, base_url = sse_server
    server.scripts["Hi there"] = [
        anthropic_delta("Hi "),
        anthropic_delta("there!"),
        anthropic_stop(),
    ]
    monkeypatch.setenv("ANTHROPIC_BASE_URL", base_url)

    df = pl.DataFrame({"prompt": ["Hi there"]})
    result = df.with_columns(
        response=inference_stream(pl.col("prompt"), provider="anthropic")
    )

    row = result["response"][0]
    assert row == {"text": "Hi there!", "finished": True}


# ============================================================================
# P3: Groq reuses the OpenAI SSE decoder via GROQ_BASE_URL
# ============================================================================


def test_groq_reuses_openai_sse(sse_server, monkeypatch):
    server, base_url = sse_server
    server.scripts["yo"] = [
        openai_delta("yo"),
        openai_delta("!"),
        openai_done(),
    ]
    monkeypatch.setenv("GROQ_BASE_URL", base_url)

    df = pl.DataFrame({"prompt": ["yo"]})
    result = df.with_columns(
        response=inference_stream(pl.col("prompt"), provider="groq")
    )

    row = result["response"][0]
    assert row == {"text": "yo!", "finished": True}


# ============================================================================
# P4: mid-stream provider error event
# ============================================================================


def test_mid_stream_error_event(sse_server, monkeypatch):
    server, base_url = sse_server
    server.scripts["oops"] = [
        openai_delta("partial"),
        openai_error("rate limited"),
    ]
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)

    df = pl.DataFrame({"prompt": ["oops"]})
    with pytest.warns(RuntimeWarning):
        result = df.with_columns(
            response=inference_stream(pl.col("prompt"), provider="openai")
        )

    row = result["response"][0]
    assert row["text"] == "partial"
    assert row["finished"] is False


# ============================================================================
# P5: EOF without a completion marker
# ============================================================================


def test_eof_without_done(sse_server, monkeypatch):
    server, base_url = sse_server
    server.scripts["cutoff"] = [openai_delta("partial")]
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)

    df = pl.DataFrame({"prompt": ["cutoff"]})
    with pytest.warns(RuntimeWarning):
        result = df.with_columns(
            response=inference_stream(pl.col("prompt"), provider="openai")
        )

    row = result["response"][0]
    assert row["text"] == "partial"
    assert row["finished"] is False


# ============================================================================
# P6: on_token callback ordering per row
# ============================================================================


def test_on_token_callback_order(sse_server, monkeypatch):
    server, base_url = sse_server
    server.scripts["abc"] = [
        openai_delta("a"),
        openai_delta("b"),
        openai_delta("c"),
        openai_done(),
    ]
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)

    seen = []

    def on_token(row_index, delta):
        seen.append((row_index, delta))

    df = pl.DataFrame({"prompt": ["abc"]})
    result = df.with_columns(
        response=inference_stream(
            pl.col("prompt"), provider="openai", on_token=on_token
        )
    )

    row_deltas = [d for (r, d) in seen if r == 0]
    assert row_deltas == ["a", "b", "c"]
    assert "".join(row_deltas) == result["response"][0]["text"]


# ============================================================================
# P7: callback raise cancels the batch, keeps partials, finished=False
# ============================================================================


def test_callback_raise_cancels_keeps_partial(sse_server, monkeypatch):
    server, base_url = sse_server
    server.scripts["fast"] = [
        openai_delta("A1"),
        openai_delta("A2"),
        openai_delta("A3"),
        openai_done(),
    ]
    server.scripts["slow"] = [
        _sleep(0.4),
        openai_delta("late"),
        openai_done(),
    ]
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)

    calls = []

    def on_token(row_index, delta):
        calls.append((row_index, delta))
        if len(calls) == 2:
            raise RuntimeError("boom from callback")

    df = pl.DataFrame({"prompt": ["fast", "slow"]})
    with pytest.warns(RuntimeWarning):
        result = df.with_columns(
            response=inference_stream(
                pl.col("prompt"), provider="openai", on_token=on_token
            )
        )

    # DataFrame stays rectangular and intact.
    assert result.height == 2
    assert result["response"].dtype == STREAM_RESPONSE_DTYPE

    fast_row = result.filter(pl.col("prompt") == "fast")["response"][0]
    slow_row = result.filter(pl.col("prompt") == "slow")["response"][0]

    # Exactly the two deltas processed before the raise made it into the
    # accumulated text; the stream was cancelled before completion.
    assert fast_row["text"] == "A1A2"
    assert fast_row["finished"] is False

    # The "slow" row was still sleeping server-side when the batch was
    # aborted, so it never got any content.
    assert slow_row["text"] == ""
    assert slow_row["finished"] is False


# ============================================================================
# P8: null input rows pass through as null struct rows
# ============================================================================


def test_null_rows_pass_through(sse_server, monkeypatch):
    server, base_url = sse_server
    server.scripts["present"] = [openai_delta("ok"), openai_done()]
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)

    df = pl.DataFrame({"prompt": ["present", None]})
    result = df.with_columns(
        response=inference_stream(pl.col("prompt"), provider="openai")
    )

    assert result["response"][0] == {"text": "ok", "finished": True}
    assert result["response"][1] is None


# ============================================================================
# P9: response_model / response_format rejected
# ============================================================================


def test_response_model_rejected():
    df = pl.DataFrame({"prompt": ["hi"]})
    with pytest.raises(ValueError):
        df.with_columns(
            response=inference_stream(
                pl.col("prompt"), provider="openai", response_model=object
            )
        )
    with pytest.raises(ValueError):
        df.with_columns(
            response=inference_stream(
                pl.col("prompt"), provider="openai", response_format=object
            )
        )


# ============================================================================
# P10: multi-row alignment
# ============================================================================


def test_multi_row_alignment(sse_server, monkeypatch):
    server, base_url = sse_server
    server.scripts["row-zero"] = [openai_delta("zero"), openai_done()]
    server.scripts["row-one"] = [openai_delta("one"), openai_done()]
    server.scripts["row-two"] = [openai_delta("two"), openai_done()]
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)

    df = pl.DataFrame({"prompt": ["row-zero", "row-one", "row-two"]})
    result = df.with_columns(
        response=inference_stream(pl.col("prompt"), provider="openai")
    )

    texts = [row["text"] for row in result["response"]]
    assert texts == ["zero", "one", "two"]
    assert all(row["finished"] for row in result["response"])


# ============================================================================
# P11: messages=True mode
# ============================================================================


def test_messages_mode(sse_server, monkeypatch):
    server, base_url = sse_server
    server.scripts["msgtest"] = [openai_delta("reply"), openai_done()]
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)

    message_json = json.dumps([{"role": "user", "content": "msgtest"}])
    df = pl.DataFrame({"messages": [message_json]})
    result = df.with_columns(
        response=inference_stream(pl.col("messages"), provider="openai", messages=True)
    )

    row = result["response"][0]
    assert row == {"text": "reply", "finished": True}


# ============================================================================
# Gated real-API smoke tests (skipped in CI without the relevant key)
# ============================================================================


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)
def test_streaming_real_openai():
    df = pl.DataFrame({"prompt": ["Say the word 'hello'.", "Count to three."]})
    seen = []
    result = df.with_columns(
        response=inference_stream(
            pl.col("prompt"),
            provider="openai",
            on_token=lambda i, d: seen.append((i, d)),
        )
    )
    assert result.height == 2
    for row in result["response"]:
        assert row["finished"] is True
        assert row["text"]
    assert len(seen) >= 1


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)
def test_streaming_real_anthropic():
    df = pl.DataFrame({"prompt": ["Say the word 'hello'.", "Count to three."]})
    seen = []
    result = df.with_columns(
        response=inference_stream(
            pl.col("prompt"),
            provider="anthropic",
            on_token=lambda i, d: seen.append((i, d)),
        )
    )
    assert result.height == 2
    for row in result["response"]:
        assert row["finished"] is True
        assert row["text"]
    assert len(seen) >= 1
