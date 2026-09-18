"""CI-safe tests for the `engine="server"` local backend adapter.

Stands up a tiny in-process OpenAI-compatible mock HTTP server (stdlib
``http.server``, no network egress -- everything binds to 127.0.0.1) and
points ``inference_local_server`` at it via ``base_url``. This exercises the
*real* Rust async fan-out (``inference_async`` / ``inference_messages``)
end-to-end, just retargeted at localhost instead of a hosted provider, so it
needs no API key, no mlx/mlx-lm, and no GPU -- safe for Linux CI.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from helpers import BacklogHTTPServer

import polars as pl
import pytest

from polar_llama.local.server_backend import inference_local_server


class _MockOpenAIHandler(BaseHTTPRequestHandler):
    """Canned `/v1/chat/completions` responder.

    Echoes back the row's user-turn content (prefixed with ``ECHO:``) so
    tests can assert completions land on the correct row without relying on
    the order requests happen to arrive in (the async fan-out issues them
    concurrently).
    """

    def log_message(self, format, *args):  # noqa: A002 - silence stderr spam
        pass

    def do_POST(self):  # noqa: N802 - stdlib method name
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        body = json.loads(raw.decode("utf-8")) if raw else {}

        with self.server.lock:  # type: ignore[attr-defined]
            self.server.requests_received.append(  # type: ignore[attr-defined]
                {"path": self.path, "body": body}
            )

        messages = body.get("messages", [])
        user_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break

        payload = {
            "id": "chatcmpl-fake",
            "model": body.get("model", "fake-local-model"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": f"ECHO:{user_content}"},
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


@pytest.fixture
def mock_openai_server():
    server = BacklogHTTPServer(("127.0.0.1", 0), _MockOpenAIHandler)
    server.lock = threading.Lock()
    server.requests_received = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield server, f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


@pytest.fixture(autouse=True)
def _isolated_openai_base_url(monkeypatch):
    # OPENAI_BASE_URL is a process-global env var (read per-request by the
    # Rust client); make sure each test starts clean and nothing leaks into
    # other test modules regardless of how this test mutates it.
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    yield
    # The teardown half matters as much as the setup half. `start_server_backend`
    # assigns os.environ directly -- by design, that is how the server engine
    # points the Rust client at a local server -- so monkeypatch has no record
    # of the assignment to undo, and `delenv` above recorded nothing either
    # (the var was already absent). Without this pop, OPENAI_BASE_URL survives
    # this module still pointing at a mock server whose port is now closed, and
    # every later test that talks to OpenAI fails with a connection error.
    os.environ.pop("OPENAI_BASE_URL", None)


@pytest.mark.local
def test_completions_come_back_in_row_order(mock_openai_server):
    server, base_url = mock_openai_server

    prompts = [f"row-{i}" for i in range(8)]
    df = pl.DataFrame({"prompt": prompts})

    result = df.with_columns(
        answer=inference_local_server(
            pl.col("prompt"),
            model="fake-local-model",
            base_url=base_url,
        )
    )

    assert result["answer"].to_list() == [f"ECHO:{p}" for p in prompts]

    # The request(s) actually hit our local mock server, not some other host.
    assert len(server.requests_received) == len(prompts)
    for req in server.requests_received:
        assert req["path"] == "/v1/chat/completions"
        assert req["body"]["model"] == "fake-local-model"


@pytest.mark.local
def test_system_prompt_is_sent_as_separate_message(mock_openai_server):
    server, base_url = mock_openai_server

    prompts = ["alpha", "beta", "gamma"]
    df = pl.DataFrame({"prompt": prompts})

    result = df.with_columns(
        answer=inference_local_server(
            pl.col("prompt"),
            model="fake-local-model",
            system="You are terse.",
            base_url=base_url,
        )
    )

    assert result["answer"].to_list() == [f"ECHO:{p}" for p in prompts]
    assert len(server.requests_received) == len(prompts)
    for req in server.requests_received:
        roles = [m["role"] for m in req["body"]["messages"]]
        assert roles == ["system", "user"]
        assert req["body"]["messages"][0]["content"] == "You are terse."


@pytest.mark.local
def test_base_url_arg_sets_openai_base_url_env(mock_openai_server, monkeypatch):
    import os

    server, base_url = mock_openai_server
    df = pl.DataFrame({"prompt": ["hello"]})

    df.with_columns(
        answer=inference_local_server(
            pl.col("prompt"), model="fake-local-model", base_url=base_url
        )
    )

    assert os.environ.get("OPENAI_BASE_URL") == base_url


@pytest.mark.local
def test_raises_without_any_endpoint_configured():
    # No base_url passed and OPENAI_BASE_URL unset (see autouse fixture
    # above) -- must not silently fall through to https://api.openai.com.
    df = pl.DataFrame({"prompt": ["hi"]})
    with pytest.raises(ValueError):
        df.with_columns(
            answer=inference_local_server(pl.col("prompt"), model="fake-local-model")
        )


@pytest.mark.local
def test_non_default_sampling_params_warn_but_still_work(mock_openai_server):
    server, base_url = mock_openai_server
    df = pl.DataFrame({"prompt": ["hi"]})

    with pytest.warns(UserWarning):
        result = df.with_columns(
            answer=inference_local_server(
                pl.col("prompt"),
                model="fake-local-model",
                base_url=base_url,
                temperature=0.7,
            )
        )

    assert result["answer"].to_list() == ["ECHO:hi"]
