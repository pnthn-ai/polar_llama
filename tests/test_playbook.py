#!/usr/bin/env python3
"""Tests for playbooks: business rules evaluated across many rows.

CI-safe. The evaluation tests drive the real Rust `typesafe_eval` expression
against a local stdlib mock of `POST /v1/systemone`, reached by pointing
`TYPESAFE_BASE_URL` at it, so no API key and no network are needed. A gated
live test sits at the bottom.
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
    DEFAULT_CONSISTENCY_ASPECTS,
    DEFAULT_MAX_ROWS_PER_GROUP,
    DEFAULT_REVIEW_LEVELS,
    Playbook,
    consistency,
    playbook,
    playbook_eval,
    rule,
    self_consistency,
)

# ============================================================================
# Mock
# ============================================================================


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002
        pass

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = json.loads(self.rfile.read(length).decode()) if length else {}
        with self.server.lock:  # type: ignore[attr-defined]
            self.server.requests_received.append(body)  # type: ignore[attr-defined]
        status, payload = self.server.responder(body)  # type: ignore[attr-defined]
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def _adheres(body):
    return 200, {
        "model": "jev-1.13.0",
        "answers": {qid: {"type": "noul", "noul": 0.9} for qid in body["questions"]},
        "usage": {"input_tokens": 100, "output_tokens": 10},
    }


@pytest.fixture
def mock_server(monkeypatch):
    server = BacklogHTTPServer(("127.0.0.1", 0), _Handler)
    server.lock = threading.Lock()
    server.requests_received = []
    server.responder = _adheres
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


@pytest.fixture
def ledger():
    # Deliberately distinct group sizes (alice 3, bob 2, carol 1) so a test can
    # identify which group a request belongs to by its record count.
    return pl.DataFrame(
        {
            "employee": ["alice", "alice", "alice", "bob", "bob", "carol"],
            "date": ["2026-03-01", "2026-03-08", "2026-03-15",
                     "2026-03-02", "2026-03-09", "2026-03-03"],
            "amount_usd": [1800, 1800, 1800, 300, 250, 500],
            "category": ["travel", "travel", "travel", "meals", "software", "meals"],
        }
    )


POLICY = playbook("expense_policy", within_limit=rule("No employee may claim over $5,000 in total."))


# ============================================================================
# Rules and playbooks (pure Python)
# ============================================================================


def test_a_rule_is_a_noul_that_asks_about_the_records_together():
    r = rule("No employee may claim over $5,000 in total.")
    assert r["type"] == "noul"
    assert r["instructions"]["rule"] == "No employee may claim over $5,000 in total."
    # The framing that makes it a *cross-row* question rather than a per-row one.
    assert "together" in r["instructions"]["evaluate"]


def test_pass_and_fail_descriptions_become_the_criteria():
    r = rule("x", passes="totals are under the cap", fails="a total exceeds the cap")
    assert r["criteria"] == {
        "true": "totals are under the cap",
        "false": "a total exceeds the cap",
    }


def test_a_structured_rule_passes_through_untouched():
    r = rule({"rule": "x", "scope": "per calendar month"})
    assert r["instructions"] == {"rule": "x", "scope": "per calendar month"}


def test_playbooks_keep_declaration_order_and_reject_empties():
    pb = playbook("p", a=rule("x"), b=rule("y"), c=rule("z"))
    assert list(pb.rules) == ["a", "b", "c"]
    assert len(pb) == 3
    with pytest.raises(ValueError, match="no rules"):
        playbook("p")
    with pytest.raises(ValueError, match="non-empty name"):
        Playbook("", {"a": rule("x")})


# ============================================================================
# Consistency checks (the rules you cannot write down in advance)
# ============================================================================


def test_consistency_is_a_score_not_a_yes_no():
    """Measured: a yes/no coherence question caught 2 of 4 planted issues; the
    graded review-priority score caught 4 of 4 with no false positives."""
    c = consistency("employee record")
    assert c["type"] == "score"
    assert c["levels"] == DEFAULT_REVIEW_LEVELS


def test_consistency_refuses_to_become_a_checklist():
    c = consistency("invoice")
    instructions = c["instructions"]
    assert "invoice" in instructions["question"]
    # The whole point: it must not narrow to a fixed set of checks.
    assert "TOGETHER" in instructions["judge"]
    assert "fixed checklist" in instructions["do_not"]


def test_hints_are_offered_as_non_exhaustive():
    c = consistency("shipment", notable=["delivery date vs dispatch date"])
    key = "worth_attention_but_not_exhaustive"
    assert c["instructions"][key] == ["delivery date vs dispatch date"]
    # Hints must not silently become the only thing looked at.
    assert "fixed checklist" in c["instructions"]["do_not"]


def test_custom_levels_are_respected():
    c = consistency("record", levels=["fine", "odd", "wrong", "impossible"])
    assert c["levels"] == ["fine", "odd", "wrong", "impossible"]


def test_self_consistency_pairs_a_priority_score_with_a_location():
    pb = self_consistency("employee record")
    assert list(pb.rules) == ["review_priority", "where"]
    assert pb.rules["review_priority"]["type"] == "score"
    assert pb.rules["where"]["type"] == "choice"
    options = [o["name"] for o in pb.rules["where"]["options"]]
    assert options == list(DEFAULT_CONSISTENCY_ASPECTS)
    assert "nothing" in options  # a coherent record needs somewhere to land


def test_aspects_are_generic_not_edge_cases():
    """Naming aspects tells a reviewer where to look; it is not the same as
    enumerating the edge cases the check exists to discover."""
    assert set(DEFAULT_CONSISTENCY_ASPECTS) == {
        "nothing", "time", "amounts", "activity", "status", "identity"
    }


def test_self_consistency_accepts_a_domain_taxonomy():
    pb = self_consistency("invoice", aspects={"nothing": "fine", "tax": "VAT looks wrong"})
    assert [o["name"] for o in pb.rules["where"]["options"]] == ["nothing", "tax"]


def test_a_consistency_playbook_runs_through_playbook_eval(mock_server):
    def responder(body):
        answers = {}
        for qid, q in body["questions"].items():
            if q["type"] == "score":
                answers[qid] = {"type": "score", "score": 1.5, "confidence": 0.8,
                                "legend": {}, "probabilities": {}}
            else:
                answers[qid] = {"type": "choice", "choice": "time", "confidence": 0.7,
                                "probabilities": {}}
        return 200, {"model": "jev-1.13.0", "answers": answers,
                     "usage": {"input_tokens": 10, "output_tokens": 2}}

    mock_server.responder = responder
    df = pl.DataFrame({"id": ["E1", "E2"], "years_employed": [3.0, 3.0],
                       "pto_days_taken_last_12_months": [18, 0]})
    out = playbook_eval(df, self_consistency("employee record"), by="id")
    assert out["review_priority"].to_list() == [1.5, 1.5]
    assert out["where"].to_list() == ["time", "time"]
    assert "where_confidence" in out.columns  # how you spot the uncertain ones
    assert out["_error"].null_count() == 2


# ============================================================================
# Evaluation
# ============================================================================


def test_one_row_and_one_request_per_group(mock_server, ledger):
    out = playbook_eval(ledger, POLICY, by="employee")
    assert out.height == 3
    assert sorted(out["employee"]) == ["alice", "bob", "carol"]
    assert len(mock_server.requests_received) == 3
    assert out["within_limit"].to_list() == [0.9, 0.9, 0.9]
    assert out["_error"].null_count() == 3


def test_each_group_is_sent_as_an_array_of_records(mock_server, ledger):
    playbook_eval(ledger, POLICY, by="employee", records=["date", "amount_usd"])
    states = [r["state"] for r in mock_server.requests_received]
    alice = next(s for s in states if len(s["records"]) == 3)
    assert alice["records"] == [
        {"date": "2026-03-01", "amount_usd": 1800},
        {"date": "2026-03-08", "amount_usd": 1800},
        {"date": "2026-03-15", "amount_usd": 1800},
    ]


def test_records_default_to_everything_but_keys_and_computed(mock_server, ledger):
    playbook_eval(
        ledger, POLICY, by="employee", compute={"total_usd": pl.col("amount_usd").sum()}
    )
    record = mock_server.requests_received[0]["state"]["records"][0]
    assert set(record) == {"date", "amount_usd", "category"}
    assert "employee" not in record  # the group key
    assert "total_usd" not in record  # computed, sent once per group


def test_computed_aggregates_ride_alongside_the_rows(mock_server, ledger):
    """Polars does the arithmetic; the model judges a small exact summary."""
    out = playbook_eval(
        ledger,
        POLICY,
        by="employee",
        compute={"total_usd": pl.col("amount_usd").sum(), "claims": pl.len()},
    )
    totals = dict(zip(out["employee"], out["total_usd"]))
    assert totals == {"alice": 5400, "bob": 550, "carol": 500}

    states = {len(s["state"]["records"]): s["state"] for s in mock_server.requests_received}
    assert states[3]["total_usd"] == 5400
    assert states[3]["claims"] == 3


def test_context_constants_reach_every_group(mock_server, ledger):
    playbook_eval(ledger, POLICY, by="employee", context={"limit_usd": 5000, "period": "2026-03"})
    for req in mock_server.requests_received:
        assert req["state"]["limit_usd"] == 5000
        assert req["state"]["period"] == "2026-03"


def test_every_rule_rides_in_the_same_request(mock_server, ledger):
    """N rules must not cost N requests per group."""
    pb = playbook("p", a=rule("x"), b=rule("y"), c=rule("z"))
    out = playbook_eval(ledger, pb, by="employee")
    assert len(mock_server.requests_received) == 3  # groups, not groups x rules
    for req in mock_server.requests_received:
        assert set(req["questions"]) == {"a", "b", "c"}
    for col in ("a", "b", "c"):
        assert col in out.columns


def test_grouping_by_several_keys(mock_server, ledger):
    out = playbook_eval(ledger, POLICY, by=["employee", "category"])
    assert out.height == 4  # alice/travel, bob/meals, bob/software, carol/meals
    assert {"employee", "category"} <= set(out.columns)


def test_no_group_key_treats_the_frame_as_one_state(mock_server, ledger):
    out = playbook_eval(ledger, POLICY)
    assert out.height == 1
    assert len(mock_server.requests_received) == 1
    assert len(mock_server.requests_received[0]["state"]["records"]) == 6


def test_usage_columns_are_opt_in(mock_server, ledger):
    out = playbook_eval(ledger, POLICY, by="employee", usage=True)
    assert out["_model"][0] == "jev-1.13.0"
    assert out["_input_tokens"][0] == 100


def test_the_internal_records_column_is_not_returned(mock_server, ledger):
    out = playbook_eval(ledger, POLICY, by="employee")
    assert "records" not in out.columns
    assert "row_count" in out.columns  # kept: it is how you spot dilution risk


# ============================================================================
# The dilution guard
# ============================================================================


def test_large_groups_warn_because_the_signal_measurably_degrades(mock_server):
    big = pl.DataFrame({"g": ["x"] * 40, "v": list(range(40))})
    with pytest.warns(UserWarning, match="measurably gets lost"):
        playbook_eval(big, POLICY, by="g")


def test_the_warning_threshold_is_tunable_and_quiet_when_respected(mock_server):
    big = pl.DataFrame({"g": ["x"] * 40, "v": list(range(40))})
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error")  # any warning fails the test
        playbook_eval(big, POLICY, by="g", max_rows_per_group=100)

    assert DEFAULT_MAX_ROWS_PER_GROUP == 25


def test_moving_the_predicate_into_compute_avoids_the_warning(mock_server):
    """The documented escape hatch: summarise in Polars, keep groups small."""
    big = pl.DataFrame({"g": ["x"] * 40, "v": list(range(40))})
    summary = big.group_by("g").agg(total=pl.col("v").sum())
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error")
        playbook_eval(summary, POLICY, by="g")


# ============================================================================
# Validation
# ============================================================================


def test_misuse_is_rejected_before_any_request(mock_server, ledger):
    with pytest.raises(TypeError, match="must be a Playbook"):
        playbook_eval(ledger, {"a": rule("x")}, by="employee")
    with pytest.raises(ValueError, match="at least one row"):
        playbook_eval(ledger.head(0), POLICY, by="employee")
    with pytest.raises(ValueError, match="not a column"):
        playbook_eval(ledger, POLICY, by="nope")
    with pytest.raises(ValueError, match="records columns not in the frame"):
        playbook_eval(ledger, POLICY, by="employee", records=["nope"])
    with pytest.raises(ValueError, match="collides with a group key"):
        playbook_eval(ledger, POLICY, by="employee", compute={"employee": pl.len()})
    assert mock_server.requests_received == []


def test_a_failing_group_is_isolated(mock_server, ledger):
    def responder(body):
        if len(body["state"]["records"]) == 3:  # alice, the only 3-row group
            return 422, {"detail": "bad"}
        return _adheres(body)

    mock_server.responder = responder
    out = playbook_eval(ledger, POLICY, by="employee")
    errors = [e for e in out["_error"] if e is not None]
    assert len(errors) == 1 and "422" in errors[0]
    assert out["within_limit"].null_count() == 1
    assert out.height == 3  # the other two groups still resolved


# ============================================================================
# Gated live test
# ============================================================================


@pytest.mark.skipif(
    not os.getenv("TYPESAFE_API_KEY"), reason="TYPESAFE_API_KEY not set"
)
def test_live_playbook_catches_a_cross_row_violation(monkeypatch):
    """A rule NO single row violates -- only the sum across rows does."""
    monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)

    rows = []
    for name, amounts in [("alice", [1800, 1800, 1800]), ("bob", [300, 250, 400]),
                          ("carol", [500, 450, 300])]:
        for i, amount in enumerate(amounts):
            rows.append({"employee": name, "date": f"2026-03-{i * 7 + 2:02d}",
                         "amount_usd": amount, "category": "travel"})
    df = pl.DataFrame(rows)

    out = playbook_eval(
        df,
        playbook("expense_policy",
                 within_limit=rule("No employee may claim more than $5,000 in total.")),
        by="employee",
        records=["date", "amount_usd", "category"],
        compute={"total_usd": pl.col("amount_usd").sum()},
        context={"limit_usd": 5000},
    )

    assert out["_error"].null_count() == 3, out["_error"].to_list()
    verdicts = dict(zip(out["employee"], out["within_limit"]))
    # alice totals 5400; nobody else is close.
    assert verdicts["alice"] < 0.5, verdicts
    assert verdicts["bob"] > 0.5 and verdicts["carol"] > 0.5, verdicts
