#!/usr/bin/env python3
"""Test suite for the TypeSafe System One inference layer.

CI-safe: the plugin-level tests drive the real Rust `typesafe_eval` expression
against a local stdlib mock of TypeSafe's `POST /v1/systemone`, reached by
pointing `TYPESAFE_BASE_URL` at it -- the same `BacklogHTTPServer` pattern
`tests/test_usage_accounting.py` uses for the chat providers. No API key and
no network are required.

A gated real-API proof sits at the bottom, skipped unless `TYPESAFE_API_KEY`
is set.
"""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler

import enum
from typing import Literal, Optional

import polars as pl
import pytest
from pydantic import BaseModel, Field

from helpers import BacklogHTTPServer

from polar_llama import (
    choice,
    choice_field,
    contract_questions,
    noul,
    score,
    score_field,
    typesafe_eval,
    typesafe_eval_each,
)
from polar_llama.typesafe import _encode_questions

# ============================================================================
# Mock TypeSafe server
# ============================================================================


class _MockSystemOneHandler(BaseHTTPRequestHandler):
    """`/v1/systemone` responder driven by `server.responder(body) -> (status, payload)`."""

    def log_message(self, format, *args):  # noqa: A002
        pass

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        body = json.loads(raw.decode("utf-8")) if raw else {}

        with self.server.lock:  # type: ignore[attr-defined]
            self.server.requests_received.append(body)  # type: ignore[attr-defined]
            self.server.auth_headers.append(self.headers.get("Authorization"))  # type: ignore[attr-defined]

        status, payload = self.server.responder(body)  # type: ignore[attr-defined]
        payload_bytes = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload_bytes)))
        self.end_headers()
        self.wfile.write(payload_bytes)


def _answer_for(question_id: str, question: dict) -> dict:
    """Produce a well-formed answer of the right type for a question."""
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
    return {
        "type": "score",
        "score": 1.0,
        "legend": {str(i): lv for i, lv in enumerate(levels)},
        "probabilities": probs,
        "confidence": 0.8,
    }


def _default_responder(body: dict):
    answers = {qid: _answer_for(qid, q) for qid, q in body["questions"].items()}
    return 200, {
        "model": "jev-1.13.0",
        "answers": answers,
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }


@pytest.fixture
def mock_server(monkeypatch):
    server = BacklogHTTPServer(("127.0.0.1", 0), _MockSystemOneHandler)
    server.lock = threading.Lock()
    server.requests_received = []
    server.auth_headers = []
    server.responder = _default_responder
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    monkeypatch.setenv("TYPESAFE_BASE_URL", f"http://{host}:{port}")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)


QUESTIONS = {
    "is_urgent": noul("Does this convey urgency?"),
    "department": choice(
        "Which team should handle this?",
        {"billing": "Payments", "technical": "Bugs", "sales": None},
    ),
    "frustration": score("How frustrated?", ["Calm", "Frustrated", "Very angry"]),
}


# ============================================================================
# Question builders (pure Python, no network)
# ============================================================================


def test_noul_omits_criteria_when_no_descriptions_given():
    assert noul("Urgent?") == {"type": "noul", "instructions": "Urgent?"}


def test_noul_uses_the_api_true_false_keys():
    spec = noul("Urgent?", true="time-sensitive", false="not")
    assert spec["criteria"] == {"true": "time-sensitive", "false": "not"}


def test_choice_accepts_a_mapping_or_a_bare_sequence():
    from_map = choice("x", {"a": "desc", "b": None})
    from_seq = choice("x", ["a", "b"])
    assert [o["name"] for o in from_map["options"]] == ["a", "b"]
    assert from_map["options"][0]["description"] == "desc"
    assert from_seq["options"][0]["description"] is None


def test_degenerate_questions_are_rejected_before_any_request():
    with pytest.raises(ValueError, match="at least 2 options"):
        choice("x", {"only": None})
    with pytest.raises(ValueError, match="at least 2 levels"):
        score("x", ["only"])
    with pytest.raises(ValueError, match="Duplicate choice options"):
        choice("x", ["a", "a"])


def test_encoding_preserves_declared_question_order():
    encoded = json.loads(_encode_questions(QUESTIONS))
    assert [q["id"] for q in encoded] == ["is_urgent", "department", "frustration"]


def test_raw_typesafe_api_json_is_accepted_unchanged():
    """An existing TypeSafe payload should work here without rewriting it."""
    encoded = json.loads(
        _encode_questions(
            {
                "dept": {
                    "type": "choice",
                    "instructions": "Who?",
                    "criteria": {"billing": "money", "tech": None},
                },
                "mood": {
                    "type": "score",
                    "instructions": "How?",
                    "criteria": ["Calm", "Angry"],
                },
                "urgent": {
                    "type": "noul",
                    "instructions": "Urgent?",
                    "criteria": {"true": "yes", "false": "no"},
                },
            }
        )
    )
    assert [o["name"] for o in encoded[0]["options"]] == ["billing", "tech"]
    assert encoded[1]["levels"] == ["Calm", "Angry"]
    assert encoded[2]["criteria"] == {"true": "yes", "false": "no"}


def test_malformed_questions_raise_before_any_request():
    with pytest.raises(ValueError, match="at least one question"):
        _encode_questions({})
    with pytest.raises(ValueError, match="unknown type"):
        _encode_questions({"q": {"type": "bogus"}})
    with pytest.raises(ValueError, match="unexpected keys"):
        _encode_questions({"q": {"type": "noul", "instructions": "x", "temperature": 1}})
    with pytest.raises(TypeError, match="must be a dict"):
        _encode_questions({"q": "just a string"})


def test_typesafe_eval_requires_a_state_expression():
    with pytest.raises(ValueError, match="at least one state expression"):
        typesafe_eval(questions=QUESTIONS)


# ============================================================================
# Schema (resolved without executing -- no requests at all)
# ============================================================================


def test_schema_is_derived_from_the_questions_without_calling_the_api():
    schema = (
        pl.DataFrame({"m": ["a"]})
        .lazy()
        .with_columns(ts=typesafe_eval(pl.col("m"), questions=QUESTIONS))
        .collect_schema()
    )
    assert dict(schema["ts"]) == {
        "is_urgent": pl.Float64,
        "department": pl.String,
        "department_confidence": pl.Float64,
        "frustration": pl.Float64,
        "frustration_confidence": pl.Float64,
        "_error": pl.String,
    }


def test_noul_contributes_no_confidence_column():
    """TypeSafe returns no confidence for a noul -- the probability is the distribution."""
    schema = (
        pl.DataFrame({"m": ["a"]})
        .lazy()
        .with_columns(ts=typesafe_eval(pl.col("m"), questions={"u": noul("x")}, probabilities=True))
        .collect_schema()
    )
    assert dict(schema["ts"]) == {"u": pl.Float64, "_error": pl.String}


def test_probabilities_and_usage_add_their_columns():
    schema = (
        pl.DataFrame({"m": ["a"]})
        .lazy()
        .with_columns(
            ts=typesafe_eval(pl.col("m"), questions=QUESTIONS, probabilities=True, usage=True)
        )
        .collect_schema()
    )
    names = dict(schema["ts"])
    for expected in (
        "department_p_billing",
        "department_p_technical",
        "department_p_sales",
        "frustration_p_0",
        "frustration_p_2",
        "_model",
        "_input_tokens",
        "_latency_ms",
    ):
        assert expected in names, expected
    assert names["_input_tokens"] == pl.Int64


def test_colliding_output_names_are_rejected_at_schema_time():
    with pytest.raises(pl.exceptions.ComputeError, match="duplicate output field"):
        pl.DataFrame({"m": ["a"]}).lazy().with_columns(
            ts=typesafe_eval(
                pl.col("m"),
                questions={"a": choice("x", ["p", "q"]), "a_confidence": noul("y")},
            )
        ).collect_schema()


# ============================================================================
# End-to-end through the real Rust plugin
# ============================================================================


def test_answers_land_in_typed_columns(mock_server):
    out = (
        pl.DataFrame({"m": ["hello"]})
        .with_columns(ts=typesafe_eval(pl.col("m"), questions=QUESTIONS, probabilities=True))
        .unnest("ts")
    )
    row = out.to_dicts()[0]
    assert row["is_urgent"] == 0.75
    assert row["department"] == "billing"
    assert row["department_confidence"] == 0.9
    assert row["department_p_billing"] == 1.0
    assert row["department_p_sales"] == 0.0
    assert row["frustration"] == 1.0
    assert row["frustration_p_1"] == 1.0
    assert row["_error"] is None


def test_every_question_rides_in_one_request_per_row(mock_server):
    """Fan-out is the point: N questions must not cost N requests."""
    pl.DataFrame({"m": ["a", "b"]}).with_columns(
        ts=typesafe_eval(pl.col("m"), questions=QUESTIONS)
    ).unnest("ts")

    assert len(mock_server.requests_received) == 2
    for body in mock_server.requests_received:
        assert set(body["questions"]) == {"is_urgent", "department", "frustration"}


def test_request_matches_the_documented_wire_format(mock_server):
    pl.DataFrame({"m": ["hello"]}).with_columns(
        ts=typesafe_eval(
            pl.col("m"),
            questions={
                "u": noul("Urgent?", true="yes", false="no"),
                "d": choice("Who?", {"billing": "money", "sales": None}),
                "f": score("How?", ["Calm", "Angry"]),
            },
            model="jev-preview",
        )
    ).unnest("ts")

    body = mock_server.requests_received[0]
    assert body["state"] == "hello"
    assert body["model"] == "jev-preview"
    assert body["questions"]["u"] == {
        "type": "noul",
        "instructions": "Urgent?",
        "criteria": {"true": "yes", "false": "no"},
    }
    assert body["questions"]["d"]["criteria"] == {"billing": "money", "sales": None}
    assert body["questions"]["f"]["criteria"] == ["Calm", "Angry"]
    assert mock_server.auth_headers[0] == "Bearer test-key"


def test_usage_columns_carry_the_resolved_model_and_tokens(mock_server):
    out = (
        pl.DataFrame({"m": ["a"]})
        .with_columns(ts=typesafe_eval(pl.col("m"), questions={"u": noul("x")}, usage=True))
        .unnest("ts")
    )
    row = out.to_dicts()[0]
    assert row["_model"] == "jev-1.13.0"
    assert row["_input_tokens"] == 100
    assert row["_output_tokens"] == 20
    assert row["_latency_ms"] is not None and row["_latency_ms"] >= 0


def test_null_states_are_never_sent_and_come_back_all_null(mock_server):
    out = (
        pl.DataFrame({"m": ["a", None, "c"]})
        .with_columns(ts=typesafe_eval(pl.col("m"), questions={"u": noul("x")}))
        .unnest("ts")
    )
    assert out["u"].to_list() == [0.75, None, 0.75]
    # A missing input is not a failure.
    assert out["_error"].to_list() == [None, None, None]
    assert len(mock_server.requests_received) == 2


def test_several_state_columns_become_one_named_object(mock_server):
    pl.DataFrame({"message": ["charged twice"], "order_id": ["A-104"], "n": [2]}).with_columns(
        ts=typesafe_eval(
            pl.col("message"),
            pl.col("order_id"),
            pl.col("n"),
            questions={"u": noul("x")},
        )
    ).unnest("ts")

    state = mock_server.requests_received[0]["state"]
    # Numbers keep their type rather than being stringified.
    assert state == {"message": "charged twice", "order_id": "A-104", "n": 2}


def test_a_list_column_is_sent_as_a_json_array(mock_server):
    """TypeSafe state accepts arrays -- e.g. a sequence of chat turns."""
    pl.DataFrame({"turns": [["Hi", "My card was charged twice."]]}).with_columns(
        ts=typesafe_eval(pl.col("turns"), questions={"u": noul("x")})
    ).unnest("ts")
    assert mock_server.requests_received[0]["state"] == ["Hi", "My card was charged twice."]


def test_a_struct_column_is_sent_as_a_json_object(mock_server):
    """TypeSafe state accepts objects -- e.g. a record, with types preserved."""
    df = pl.DataFrame({"ticket": [{"text": "charged twice", "amount": 49}]})
    df.with_columns(ts=typesafe_eval(pl.col("ticket"), questions={"u": noul("x")})).unnest("ts")
    assert mock_server.requests_received[0]["state"] == {"text": "charged twice", "amount": 49}


def test_nested_state_columns_compose_with_named_multi_column_state(mock_server):
    df = pl.DataFrame(
        {"message": ["refund please"], "charges": [[49, 49]]}
    )
    df.with_columns(
        ts=typesafe_eval(pl.col("message"), pl.col("charges"), questions={"u": noul("x")})
    ).unnest("ts")
    assert mock_server.requests_received[0]["state"] == {
        "message": "refund please",
        "charges": [49, 49],
    }


def test_a_constant_broadcasts_alongside_per_row_state(mock_server):
    """A shared policy/document next to per-row state is a natural request."""
    df = pl.DataFrame({"m": ["refund me", "just browsing"]})
    df.with_columns(
        ts=typesafe_eval(
            pl.col("m"),
            pl.lit("duplicates are refundable").alias("policy"),
            questions={"u": noul("x")},
        )
    ).unnest("ts")

    states = [r["state"] for r in mock_server.requests_received]
    assert {s["m"] for s in states} == {"refund me", "just browsing"}
    assert all(s["policy"] == "duplicates are refundable" for s in states)


def test_state_json_sends_structured_state_and_flags_bad_rows(mock_server):
    out = (
        pl.DataFrame({"s": ['{"a": 1}', "not json"]})
        .with_columns(ts=typesafe_eval(pl.col("s"), questions={"u": noul("x")}, state_json=True))
        .unnest("ts")
    )
    assert mock_server.requests_received[0]["state"] == {"a": 1}
    # The malformed row never reaches the API.
    assert len(mock_server.requests_received) == 1
    assert out["_error"][0] is None
    assert "not valid JSON" in out["_error"][1]


def test_row_order_is_preserved_across_concurrent_requests(mock_server):
    def responder(body):
        status, payload = _default_responder(body)
        payload["answers"]["u"]["noul"] = float(body["state"])
        return status, payload

    mock_server.responder = responder
    states = [str(i) for i in range(24)]
    out = (
        pl.DataFrame({"m": states})
        .with_columns(ts=typesafe_eval(pl.col("m"), questions={"u": noul("x")}))
        .unnest("ts")
    )
    assert out["u"].to_list() == [float(i) for i in range(24)]


# ============================================================================
# Failure handling
# ============================================================================


def test_a_failing_row_is_isolated_and_the_frame_still_resolves(mock_server):
    def responder(body):
        if body["state"] == "boom":
            return 422, {"detail": "bad question"}
        return _default_responder(body)

    mock_server.responder = responder
    out = (
        pl.DataFrame({"m": ["ok", "boom", "ok2"]})
        .with_columns(ts=typesafe_eval(pl.col("m"), questions={"u": noul("x")}))
        .unnest("ts")
    )
    assert out["u"].to_list() == [0.75, None, 0.75]
    assert out["_error"][0] is None
    assert "422" in out["_error"][1]
    assert out["_error"][2] is None


def test_rate_limits_are_retried_and_then_succeed(mock_server, monkeypatch):
    monkeypatch.setenv("POLAR_LLAMA_TYPESAFE_MAX_RETRIES", "3")
    calls = {"n": 0}

    def responder(body):
        calls["n"] += 1
        if calls["n"] == 1:
            return 429, {"detail": "slow down"}
        return _default_responder(body)

    mock_server.responder = responder
    out = (
        pl.DataFrame({"m": ["a"]})
        .with_columns(ts=typesafe_eval(pl.col("m"), questions={"u": noul("x")}))
        .unnest("ts")
    )
    assert out["u"][0] == 0.75
    assert out["_error"][0] is None
    assert calls["n"] == 2


def test_client_errors_are_not_retried(mock_server, monkeypatch):
    """Retrying a bad key or a malformed question only bills for it again."""
    monkeypatch.setenv("POLAR_LLAMA_TYPESAFE_MAX_RETRIES", "3")
    mock_server.responder = lambda body: (401, {"detail": "bad key"})

    started = time.monotonic()
    out = (
        pl.DataFrame({"m": ["a"]})
        .with_columns(ts=typesafe_eval(pl.col("m"), questions={"u": noul("x")}))
        .unnest("ts")
    )
    assert len(mock_server.requests_received) == 1
    assert "401" in out["_error"][0]
    assert time.monotonic() - started < 1.0


def test_retry_budget_is_finite(mock_server, monkeypatch):
    monkeypatch.setenv("POLAR_LLAMA_TYPESAFE_MAX_RETRIES", "1")
    mock_server.responder = lambda body: (529, {"detail": "overloaded"})

    out = (
        pl.DataFrame({"m": ["a"]})
        .with_columns(ts=typesafe_eval(pl.col("m"), questions={"u": noul("x")}))
        .unnest("ts")
    )
    # Initial attempt + 1 retry.
    assert len(mock_server.requests_received) == 2
    assert "529" in out["_error"][0]


def test_a_missing_answer_is_null_not_a_crash(mock_server):
    """A question TypeSafe declines to answer must not take down the row."""
    mock_server.responder = lambda body: (
        200,
        {"model": "m", "answers": {}, "usage": {"input_tokens": 1, "output_tokens": 1}},
    )
    out = (
        pl.DataFrame({"m": ["a"]})
        .with_columns(ts=typesafe_eval(pl.col("m"), questions={"u": noul("x")}))
        .unnest("ts")
    )
    assert out["u"][0] is None
    assert out["_error"][0] is None


# ============================================================================
# Namespace accessor
# ============================================================================


def test_namespace_accessor_matches_the_function(mock_server):
    out = (
        pl.DataFrame({"m": ["a"]})
        .with_columns(ts=pl.col("m").llama.typesafe_eval(questions={"u": noul("x")}))
        .unnest("ts")
    )
    assert out["u"][0] == 0.75


# ============================================================================
# Contracts (Pydantic model -> questions)
# ============================================================================


class ClauseFeatures(BaseModel):
    is_payment: bool = Field(description="Does this clause create a payment obligation?")
    category: Literal["fees", "term", "liability"] = Field(description="Clause type?")
    severity: float = score_field("How onerous?", ["Benign", "Notable", "Onerous"])


def test_python_types_pick_the_question_type():
    qs = contract_questions(ClauseFeatures)
    assert list(qs) == ["is_payment", "category", "severity"]
    assert qs["is_payment"]["type"] == "noul"
    assert qs["category"]["type"] == "choice"
    assert qs["severity"]["type"] == "score"


def test_field_description_becomes_the_instructions():
    qs = contract_questions(ClauseFeatures)
    assert qs["is_payment"]["instructions"] == "Does this clause create a payment obligation?"
    assert qs["severity"]["levels"] == ["Benign", "Notable", "Onerous"]
    assert [o["name"] for o in qs["category"]["options"]] == ["fees", "term", "liability"]


def test_an_enum_class_keeps_the_fields_own_description():
    """`_pydantic_to_json_schema` strips $ref siblings for OpenAI strict mode,
    which would drop this description -- contracts must not go through it."""

    class Category(str, enum.Enum):
        fees = "fees"
        term = "term"

    class M(BaseModel):
        category: Category = Field(description="What kind of clause is this?")

    qs = contract_questions(M)
    assert qs["category"]["instructions"] == "What kind of clause is this?"
    assert [o["name"] for o in qs["category"]["options"]] == ["fees", "term"]


def test_choice_field_supplies_per_option_rubrics():
    class M(BaseModel):
        kind: Literal["fees", "term"] = choice_field(
            "What kind?", {"fees": "Payments and invoicing", "term": "Duration"}
        )

    options = contract_questions(M)["kind"]["options"]
    assert options[0] == {"name": "fees", "description": "Payments and invoicing"}


def test_optional_fields_resolve_to_their_inner_type():
    class M(BaseModel):
        flagged: Optional[bool] = Field(default=None, description="Flagged?")

    assert contract_questions(M)["flagged"]["type"] == "noul"


def test_a_field_with_no_description_falls_back_to_its_name():
    class M(BaseModel):
        is_urgent: bool

    assert contract_questions(M)["is_urgent"]["instructions"] == "is urgent"


def test_free_text_fields_are_rejected_with_a_useful_message():
    """TypeSafe has no free-text primitive -- that limit should be explained."""

    class M(BaseModel):
        summary: str = Field(description="Summarize the clause")

    with pytest.raises(ValueError, match="Literal"):
        contract_questions(M)


def test_a_numeric_field_without_levels_is_rejected():
    class M(BaseModel):
        severity: float = Field(description="How bad?")

    with pytest.raises(ValueError, match="no levels"):
        contract_questions(M)


def test_nested_and_unsupported_shapes_are_rejected():
    class Inner(BaseModel):
        x: bool

    class M(BaseModel):
        inner: Inner

    with pytest.raises(ValueError, match="flat"):
        contract_questions(M)


def test_contract_and_questions_are_mutually_exclusive():
    with pytest.raises(ValueError, match="exactly one"):
        typesafe_eval(pl.col("m"), questions={"u": noul("x")}, contract=ClauseFeatures)
    with pytest.raises(ValueError, match="exactly one"):
        typesafe_eval(pl.col("m"))


def test_a_contract_drives_the_per_row_expression_too(mock_server):
    out = (
        pl.DataFrame({"m": ["Customer shall pay all fees."]})
        .with_columns(ts=typesafe_eval(pl.col("m"), contract=ClauseFeatures))
        .unnest("ts")
    )
    assert set(out.columns) >= {"m", "is_payment", "category", "severity"}
    assert out["is_payment"][0] == 0.75


def test_hash_is_reserved_in_question_ids():
    with pytest.raises(ValueError, match="reserved separator"):
        _encode_questions({"a#b": noul("x")})


# ============================================================================
# Per-line dimension
# ============================================================================


def _line_responder(body: dict):
    """Answer each per-line noul with the line's own index, so a scrambled
    mapping is detectable rather than plausible."""
    answers = {}
    for qid, q in body["questions"].items():
        answers[qid] = _answer_for(qid, q)
        if q["type"] == "noul":
            answers[qid]["noul"] = float(qid.split("#")[0])
    return 200, {
        "model": "jev-1.13.0",
        "answers": answers,
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }


def test_each_returns_a_list_of_structs_one_per_segment():
    schema = (
        pl.DataFrame({"lines": [["a"]]})
        .lazy()
        .with_columns(f=typesafe_eval_each(pl.col("lines"), questions={"u": noul("x")}))
        .collect_schema()
    )
    assert schema["f"] == pl.List(
        pl.Struct({"line_id": pl.Int64, "line": pl.String, "u": pl.Float64, "_error": pl.String})
    )


def test_include_segment_false_drops_the_line_column():
    schema = (
        pl.DataFrame({"lines": [["a"]]})
        .lazy()
        .with_columns(
            f=typesafe_eval_each(
                pl.col("lines"), questions={"u": noul("x")}, include_segment=False
            )
        )
        .collect_schema()
    )
    assert "line" not in dict(schema["f"].inner)


def test_every_segment_is_answered_in_one_request(mock_server):
    """The whole point: N segments must not cost N requests."""
    mock_server.responder = _line_responder
    out = (
        pl.DataFrame({"lines": [["a", "b", "c", "d"]]})
        .with_columns(f=typesafe_eval_each(pl.col("lines"), questions={"u": noul("x")}))
        .explode("f")
        .unnest("f")
    )
    assert len(mock_server.requests_received) == 1
    assert out["line_id"].to_list() == [0, 1, 2, 3]
    assert out["line"].to_list() == ["a", "b", "c", "d"]
    # Each segment got its own answer, in the right place.
    assert out["u"].to_list() == [0.0, 1.0, 2.0, 3.0]


def test_segments_go_into_state_once_and_questions_name_their_line(mock_server):
    pl.DataFrame({"lines": [["alpha", "beta"]]}).with_columns(
        f=typesafe_eval_each(pl.col("lines"), questions={"pays": noul("Payment?")})
    ).explode("f")

    body = mock_server.requests_received[0]
    # The segments are billed once, in state -- not repeated per question.
    assert body["state"]["lines"] == {"0": "alpha", "1": "beta"}
    assert set(body["questions"]) == {"0#pays", "1#pays"}
    q = body["questions"]["1#pays"]
    assert q["instructions"]["question"] == "Payment?"
    assert q["instructions"]["evaluate_line_id"] == "1"


def test_chunking_splits_requests_but_keeps_global_line_ids(mock_server):
    mock_server.responder = _line_responder
    lines = [f"line {i}" for i in range(13)]
    out = (
        pl.DataFrame({"lines": [lines]})
        .with_columns(
            f=typesafe_eval_each(pl.col("lines"), questions={"u": noul("x")}, max_questions=5)
        )
        .explode("f")
        .unnest("f")
    )
    assert len(mock_server.requests_received) == 3  # ceil(13 / 5)
    assert out["line_id"].to_list() == list(range(13))
    assert out["u"].to_list() == [float(i) for i in range(13)]
    assert out["_error"].null_count() == 13


def test_chunk_size_accounts_for_contract_width(mock_server):
    """Each segment costs one question per contract field."""
    mock_server.responder = _line_responder
    lines = [f"line {i}" for i in range(12)]
    pl.DataFrame({"lines": [lines]}).with_columns(
        f=typesafe_eval_each(
            pl.col("lines"),
            questions={"a": noul("x"), "b": noul("y"), "c": noul("z")},
            max_questions=6,
        )
    ).explode("f")
    # 6 questions / 3 per segment = 2 segments per request -> 6 requests.
    assert len(mock_server.requests_received) == 6
    for body in mock_server.requests_received:
        assert len(body["questions"]) == 6


def test_a_failing_chunk_only_marks_its_own_segments(mock_server):
    seen = {"n": 0}

    def responder(body):
        seen["n"] += 1
        if seen["n"] == 2:
            return 422, {"detail": "bad"}
        return _line_responder(body)

    mock_server.responder = responder
    lines = [f"line {i}" for i in range(9)]
    out = (
        pl.DataFrame({"lines": [lines]})
        .with_columns(
            f=typesafe_eval_each(pl.col("lines"), questions={"u": noul("x")}, max_questions=3)
        )
        .explode("f")
        .unnest("f")
    )
    errors = out["_error"].to_list()
    # Exactly one chunk of 3 failed; the other 6 segments still resolved.
    assert sum(e is not None for e in errors) == 3
    assert sum(e is None for e in errors) == 6
    assert out["line_id"].to_list() == list(range(9))


def test_context_columns_ride_in_every_chunk(mock_server):
    lines = [f"line {i}" for i in range(6)]
    pl.DataFrame({"lines": [lines], "title": ["Master Services Agreement"]}).with_columns(
        f=typesafe_eval_each(
            pl.col("lines"), pl.col("title"), questions={"u": noul("x")}, max_questions=2
        )
    ).explode("f")

    assert len(mock_server.requests_received) == 3
    for body in mock_server.requests_received:
        assert body["state"]["title"] == "Master Services Agreement"


def test_rows_are_independent_documents(mock_server):
    mock_server.responder = _line_responder
    out = (
        pl.DataFrame({"doc": ["a", "b"], "lines": [["x", "y"], ["p", "q", "r"]]})
        .with_columns(f=typesafe_eval_each(pl.col("lines"), questions={"u": noul("x")}))
        .explode("f")
        .unnest("f")
    )
    # One request per document, and line ids restart per row.
    assert len(mock_server.requests_received) == 2
    assert out["doc"].to_list() == ["a", "a", "b", "b", "b"]
    assert out["line_id"].to_list() == [0, 1, 0, 1, 2]


def test_null_and_empty_segment_lists_are_never_sent(mock_server):
    pl.DataFrame({"lines": [None, [], ["real"]]}).with_columns(
        f=typesafe_eval_each(pl.col("lines"), questions={"u": noul("x")})
    ).explode("f")
    assert len(mock_server.requests_received) == 1
    assert mock_server.requests_received[0]["state"]["lines"] == {"0": "real"}


def test_a_non_list_column_is_rejected_with_a_fix(mock_server):
    with pytest.raises(pl.exceptions.ComputeError, match="str.split"):
        pl.DataFrame({"text": ["a\nb"]}).lazy().with_columns(
            f=typesafe_eval_each(pl.col("text"), questions={"u": noul("x")})
        ).collect_schema()


# ============================================================================
# Gated real-API proof
# ============================================================================


@pytest.mark.skipif(
    not os.getenv("TYPESAFE_API_KEY"),
    reason="TYPESAFE_API_KEY not set; skipping live TypeSafe test",
)
def test_live_typesafe_evaluation(monkeypatch):
    monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)

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
                "is_urgent": noul(
                    "Does this convey urgency?",
                    true="Explicitly time-sensitive",
                    false="No urgency expressed",
                ),
                "department": choice(
                    "Which team should handle this?",
                    {
                        "billing": "Payments, invoicing, refunds",
                        "technical": "Bugs, outages, integrations",
                        "sales": "Pricing, upgrades, new accounts",
                    },
                ),
            },
            probabilities=True,
            usage=True,
        )
    ).unnest("ts")

    assert out["_error"].null_count() == 2, out["_error"].to_list()
    # The urgent payout failure should read as more urgent than a pricing ask,
    # and route somewhere other than sales.
    assert out["is_urgent"][0] > out["is_urgent"][1]
    assert out["department"][0] in {"billing", "technical"}
    assert out["department"][1] == "sales"
    # Probabilities are a distribution.
    probs = [out[f"department_p_{o}"][0] for o in ("billing", "technical", "sales")]
    assert abs(sum(probs) - 1.0) < 0.05
    assert out["_input_tokens"][0] > 0


@pytest.mark.skipif(
    not os.getenv("TYPESAFE_API_KEY"),
    reason="TYPESAFE_API_KEY not set; skipping live TypeSafe test",
)
def test_live_per_line_contract_extraction(monkeypatch):
    monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)

    text = "\n".join(
        [
            "Definitions. 'Services' means the hosted software described in an Order Form.",
            "Fees. Customer shall pay all fees within thirty (30) days of invoice date.",
            "Term. This Agreement commences on the Effective Date and continues for one year.",
            "Governing Law. This Agreement is governed by the laws of Delaware.",
        ]
    )
    out = (
        pl.DataFrame({"text": [text]})
        .with_columns(clause=pl.col("text").str.split("\n"))
        .with_columns(
            f=typesafe_eval_each(pl.col("clause"), contract=ClauseFeatures, usage=True)
        )
        .explode("f")
        .unnest("f")
    )

    assert len(out) == 4
    assert out["line_id"].to_list() == [0, 1, 2, 3]
    assert out["_error"].null_count() == 4, out["_error"].to_list()
    # Only the Fees clause is a payment obligation.
    payment = out["is_payment"].to_list()
    assert payment[1] > 0.8
    assert max(payment[0], payment[2], payment[3]) < 0.5
    assert out["category"][1] == "fees"
    # One request covered all four clauses.
    assert out["_input_tokens"].n_unique() == 1


@pytest.mark.skipif(
    not os.getenv("TYPESAFE_API_KEY"),
    reason="TYPESAFE_API_KEY not set; skipping live TypeSafe test",
)
def test_live_namespace_accessor_for_per_line(monkeypatch):
    monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)
    out = (
        pl.DataFrame({"clause": [["Customer shall pay $500 on demand.", "Notices go to Exhibit A."]]})
        .with_columns(
            f=pl.col("clause").llama.typesafe_eval_each(
                questions={"pays": noul("Does this require the customer to pay money?")}
            )
        )
        .explode("f")
        .unnest("f")
    )
    assert out["pays"][0] > 0.8
    assert out["pays"][1] < 0.5
