#!/usr/bin/env python3
"""Test suite for the Cloudflare Clef backend of the System One layer.

Clef takes the same questions as TypeSafe's Jev, so these tests only cover
what differs: the Workers AI endpoint (model in the path, account-scoped), the
Cloudflare API token, and the v4 response envelope. Everything else -- column
plans, fan-out, per-line chunking -- is shared and covered by
`tests/test_typesafe.py`.

CI-safe: the plugin is pointed at a local stdlib mock via
`CLOUDFLARE_API_BASE_URL`. A gated live test at the bottom runs only when
`CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` are set.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler

import polars as pl
import pytest
from helpers import BacklogHTTPServer

from polar_llama import (
    CLEF_MODELS,
    DEFAULT_CLEF_MODEL,
    choice,
    noul,
    score,
    typesafe_eval,
    typesafe_eval_each,
)
from polar_llama.assess import assess, ruleset

ACCOUNT = "acct-123"


class _MockWorkersAIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002
        pass

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        body = json.loads(raw.decode("utf-8")) if raw else {}

        with self.server.lock:  # type: ignore[attr-defined]
            self.server.requests_received.append(body)  # type: ignore[attr-defined]
            self.server.paths.append(self.path)  # type: ignore[attr-defined]
            self.server.auth_headers.append(self.headers.get("Authorization"))  # type: ignore[attr-defined]

        status, payload = self.server.responder(body)  # type: ignore[attr-defined]
        payload_bytes = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload_bytes)))
        self.end_headers()
        self.wfile.write(payload_bytes)


def _answer_for(question: dict) -> dict:
    qtype = question["type"]
    if qtype == "noul":
        return {"type": "noul", "noul": 0.75}
    if qtype == "choice":
        options = list(question["criteria"])
        probs = {opt: 0.0 for opt in options}
        probs[options[0]] = 1.0
        return {
            "type": "choice",
            "choice": options[0],
            "probabilities": probs,
            "confidence": 0.9,
        }
    levels = question["criteria"]
    probs = {str(i): 0.0 for i in range(len(levels))}
    probs["1"] = 1.0
    return {"type": "score", "score": 1.0, "probabilities": probs, "confidence": 0.8}


def _envelope(result: dict) -> dict:
    return {"success": True, "errors": [], "messages": [], "result": result}


def _default_responder(body: dict):
    answers = {qid: _answer_for(q) for qid, q in body["questions"].items()}
    return 200, _envelope(
        {
            "model": "@cf/cloudflare/clef",
            "answers": answers,
            "usage": {"input_tokens": 222, "output_tokens": 3},
        }
    )


@pytest.fixture
def clef_server(monkeypatch):
    server = BacklogHTTPServer(("127.0.0.1", 0), _MockWorkersAIHandler)
    server.lock = threading.Lock()
    server.requests_received = []
    server.paths = []
    server.auth_headers = []
    server.responder = _default_responder
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    monkeypatch.setenv("CLOUDFLARE_API_BASE_URL", f"http://{host}:{port}/client/v4")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", ACCOUNT)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token")
    monkeypatch.delenv("CLOUDFLARE_AI_GATEWAY_ID", raising=False)
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)


QUESTIONS = {
    "is_urgent": noul("Does this convey urgency?"),
    "department": choice("Which team?", {"billing": "Payments", "sales": None}),
    "frustration": score("How frustrated?", ["Calm", "Frustrated", "Very angry"]),
}


def _run(**kwargs) -> pl.DataFrame:
    return (
        pl.DataFrame({"m": ["hello"]})
        .with_columns(ts=typesafe_eval(pl.col("m"), provider="clef", **kwargs))
        .unnest("ts")
    )


# ============================================================================
# Python-side validation (no network)
# ============================================================================


def test_clef_constants_name_the_workers_ai_models():
    assert DEFAULT_CLEF_MODEL == "@cf/cloudflare/clef"
    assert "@cf/cloudflare/clef-flash" in CLEF_MODELS


def test_unknown_provider_is_rejected_before_any_request():
    with pytest.raises(ValueError, match="provider"):
        typesafe_eval(pl.col("m"), questions=QUESTIONS, provider="openai")


def test_schema_is_identical_across_providers():
    lf = pl.DataFrame({"m": ["a"]}).lazy()
    ts = lf.with_columns(
        ts=typesafe_eval(pl.col("m"), questions=QUESTIONS)
    ).collect_schema()
    cf = lf.with_columns(
        ts=typesafe_eval(pl.col("m"), questions=QUESTIONS, provider="clef")
    ).collect_schema()
    assert ts == cf


def test_missing_account_fails_up_front(monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    with pytest.raises(pl.exceptions.ComputeError, match="CLOUDFLARE_ACCOUNT_ID"):
        _run(questions=QUESTIONS)


# ============================================================================
# End-to-end through the real Rust plugin
# ============================================================================


def test_request_goes_to_the_model_path_with_the_cloudflare_token(clef_server):
    _run(questions=QUESTIONS)

    assert clef_server.paths == [
        f"/client/v4/accounts/{ACCOUNT}/ai/run/@cf/cloudflare/clef"
    ]
    assert clef_server.auth_headers == ["Bearer cf-token"]
    body = clef_server.requests_received[0]
    # The model is in the path, so the body is just the System One payload.
    assert set(body) == {"state", "questions"}
    assert body["state"] == "hello"
    assert set(body["questions"]) == {"is_urgent", "department", "frustration"}


def test_short_model_names_resolve_into_the_cloudflare_namespace(clef_server):
    _run(questions={"u": noul("x")}, model="clef-flash")
    assert clef_server.paths[0].endswith("/ai/run/@cf/cloudflare/clef-flash")


def test_answers_unwrap_from_the_envelope_into_typed_columns(clef_server):
    row = _run(questions=QUESTIONS, probabilities=True, usage=True).to_dicts()[0]
    assert row["is_urgent"] == 0.75
    assert row["department"] == "billing"
    assert row["department_confidence"] == 0.9
    assert row["frustration"] == 1.0
    assert row["frustration_p_1"] == 1.0
    assert row["_model"] == "@cf/cloudflare/clef"
    assert row["_input_tokens"] == 222
    assert row["_error"] is None


def test_api_key_is_accepted_as_a_fallback_token_name(clef_server, monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN")
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "legacy-name")
    _run(questions={"u": noul("x")})
    assert clef_server.auth_headers == ["Bearer legacy-name"]


def test_success_false_on_200_is_a_row_error(clef_server):
    clef_server.responder = lambda body: (
        200,
        {
            "success": False,
            "errors": [{"code": 5006, "message": "Model overloaded"}],
            "result": None,
        },
    )
    row = _run(questions={"u": noul("x")}).to_dicts()[0]
    assert row["u"] is None
    assert "Model overloaded" in row["_error"]


def test_http_errors_surface_cloudflares_reason(clef_server):
    clef_server.responder = lambda body: (
        401,
        {
            "success": False,
            "errors": [{"code": 10000, "message": "Authentication error"}],
        },
    )
    row = _run(questions={"u": noul("x")}).to_dicts()[0]
    assert "401" in row["_error"]
    assert "Authentication error" in row["_error"]
    # 401 is never retried.
    assert len(clef_server.requests_received) == 1


def test_rate_limits_are_retried(clef_server, monkeypatch):
    monkeypatch.setenv("POLAR_LLAMA_TYPESAFE_MAX_RETRIES", "2")
    calls = {"n": 0}

    def responder(body):
        calls["n"] += 1
        if calls["n"] == 1:
            return 429, {
                "success": False,
                "errors": [{"code": 3040, "message": "Capacity"}],
            }
        return _default_responder(body)

    clef_server.responder = responder
    row = _run(questions={"u": noul("x")}).to_dicts()[0]
    assert row["u"] == 0.75
    assert calls["n"] == 2


def test_per_line_evaluation_runs_on_clef(clef_server):
    out = (
        pl.DataFrame({"lines": [["a", "b", "c"]]})
        .with_columns(
            f=typesafe_eval_each(
                pl.col("lines"), questions={"u": noul("x")}, provider="clef"
            )
        )
        .explode("f")
        .unnest("f")
    )
    assert len(clef_server.requests_received) == 1
    assert out["line_id"].to_list() == [0, 1, 2]
    assert out["u"].to_list() == [0.75, 0.75, 0.75]
    assert clef_server.paths[0].endswith("/ai/run/@cf/cloudflare/clef")


def test_namespace_accessor_passes_the_provider(clef_server):
    out = (
        pl.DataFrame({"m": ["a"]})
        .with_columns(
            ts=pl.col("m").llama.typesafe_eval(
                questions={"u": noul("x")}, provider="clef"
            )
        )
        .unnest("ts")
    )
    assert out["u"][0] == 0.75
    assert len(clef_server.paths) == 1


def test_assess_runs_on_clef(clef_server):
    rules = ruleset("r", ok=noul("Is everything fine?"))
    out = assess(pl.DataFrame({"k": ["a"], "v": [1]}), rules, by="k", provider="clef")
    assert out["ok"][0] == 0.75
    assert len(clef_server.paths) == 1


# ============================================================================
# Gated real-API proof
# ============================================================================


@pytest.mark.skipif(
    not (os.getenv("CLOUDFLARE_ACCOUNT_ID") and os.getenv("CLOUDFLARE_API_TOKEN")),
    reason="CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN not set; skipping live Clef test",
)
@pytest.mark.parametrize("model", CLEF_MODELS)
def test_live_clef_evaluation(monkeypatch, model):
    monkeypatch.delenv("CLOUDFLARE_API_BASE_URL", raising=False)

    df = pl.DataFrame(
        {
            "message": [
                "Help! My payouts have been failing for 3 days.",
                "Hi, just wondering what your enterprise pricing looks like.",
            ]
        }
    )
    out = df.with_columns(
        ts=typesafe_eval(
            pl.col("message"),
            questions={
                "is_urgent": noul("Does this convey urgency?"),
                "department": choice(
                    "Which team should handle this?",
                    {
                        "billing": "Payments, invoicing, refunds",
                        "technical": "Bugs, outages, integrations",
                        "sales": "Pricing, upgrades, new accounts",
                    },
                ),
            },
            provider="clef",
            model=model,
            usage=True,
        )
    ).unnest("ts")

    assert out["_error"].null_count() == 2, out["_error"].to_list()
    assert out["is_urgent"][0] > out["is_urgent"][1]
    assert out["department"][1] == "sales"
