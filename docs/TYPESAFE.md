# TypeSafe System One

## Overview

`typesafe_eval` is a native inference layer for the
[TypeSafe System One API](https://docs.typesafe.ai/api), written in Rust
(`src/model_client/typesafe.rs` + `src/typesafe_expr.rs`) and exposed to Python
as a Polars expression (`polar_llama/typesafe.py`).

TypeSafe is deliberately **not** a chat-completions provider, so it does not
implement the `ModelClient` trait the OpenAI/Anthropic/Gemini/Groq/Bedrock
clients share. There is no prompt and no free-text completion. One request
carries a single `state` plus a map of typed `questions`, and returns one typed
`answer` per question:

| Question | Asks | Answer |
|---|---|---|
| `noul` | a yes/no question | probability in `[0, 1]` |
| `choice` | pick one of a closed set | the pick + a probability per option + confidence |
| `score` | rate against ordered levels | a probability-weighted value + a probability per level + confidence |

Because the answers are typed and calibrated rather than parsed out of prose,
they land in a DataFrame as ordinary numeric and string columns you can filter,
sort, threshold, and join on -- which is why this belongs in polar-llama at all.

```
   df: one state per row
         |
         v
   typesafe_eval(pl.col("message"), questions={...})
         |                -- ONE POST /v1/systemone per row, carrying
         |                   every question; bounded by
         |                   POLAR_LLAMA_MAX_CONCURRENCY
         v
   Struct{ is_urgent: f64, department: str, department_confidence: f64, ..., _error: str }
         |
         v
   .unnest("ts")          -- flat, typed columns
         |
         v
   df.filter(pl.col("department_confidence") > 0.9)   -- your code decides
```

## Why one request per row, not one per question

Every question in the mapping rides in the *same* request. TypeSafe's own
["speculative fan-out"](https://docs.typesafe.ai/patterns/fan-out) guidance is
to ask everything you might want up front and let your code decide afterwards
what to read -- batching is dramatically cheaper and faster than one call per
question, because the state is only paid for once. A per-question expression
would re-send the state once per column, so the API is deliberately shaped so
that you cannot accidentally do that.

## Usage

```python
import polars as pl
from polar_llama import typesafe_eval, noul, choice, score

df = pl.DataFrame({"message": [
    "Help! My payouts have been failing for 3 days.",
    "Hi, just wondering what your enterprise pricing looks like.",
]})

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
            "frustration": score(
                "How frustrated is the customer?",
                ["Calm", "Frustrated", "Very angry"],
            ),
        },
    )
).unnest("ts")
```

```
┌──────────────┬───────────┬────────────┬───────────────────────┬─────────────┬────────────────────────┬────────┐
│ message      ┆ is_urgent ┆ department ┆ department_confidence ┆ frustration ┆ frustration_confidence ┆ _error │
│ str          ┆ f64       ┆ str        ┆ f64                   ┆ f64         ┆ f64                    ┆ str    │
╞══════════════╪═══════════╪════════════╪═══════════════════════╪═════════════╪════════════════════════╪════════╡
│ Help! My …   ┆ 0.95      ┆ billing    ┆ 0.79                  ┆ 1.04        ┆ 0.93                   ┆ null   │
│ Hi, just …   ┆ 0.06      ┆ sales      ┆ 1.0                   ┆ 0.0         ┆ 1.0                    ┆ null   │
└──────────────┴───────────┴────────────┴───────────────────────┴─────────────┴────────────────────────┴────────┘
```

The fluent namespace works too:

```python
df.with_columns(ts=pl.col("message").llama.typesafe_eval(questions={...})).unnest("ts")
```

## Output schema

Columns are emitted in the order the questions were declared. The dtype is
derived from the question type *before* any request is made, so
`.collect_schema()` on a LazyFrame resolves the full output shape without
spending a token.

| Question type | Columns |
|---|---|
| `noul` | `<id>`: Float64 |
| `choice` | `<id>`: String, `<id>_confidence`: Float64 |
| `score` | `<id>`: Float64, `<id>_confidence`: Float64 |

Plus:

* `_error`: String -- always present, null on success.
* `probabilities=True` -- adds `<id>_p_<option>` (choice) and `<id>_p_<level index>` (score), Float64.
* `usage=True` -- adds `_model` (the *resolved* version that ran, e.g.
  `jev-1.13.0`, not the `jev-latest` alias), `_input_tokens`, `_output_tokens`,
  `_latency_ms` (total wall clock including any retries).

A noul question gets no `_confidence` column: TypeSafe returns none, because a
single probability already *is* the distribution.

Colliding field names (a question `a` next to a question `a_confidence`) are
rejected at schema time rather than producing a Struct with duplicate fields.

## State

A single expression is sent as that bare value. Several expressions are sent as
a JSON object keyed by column name, which is the shape TypeSafe recommends for
most requests -- each part of the state keeps a descriptive name:

```python
typesafe_eval(
    pl.col("message"), pl.col("order_id"), pl.col("charge_count"),
    questions={"duplicate": noul("Do the records show a duplicate charge?")},
)
# state = {"message": "...", "order_id": "A-104", "charge_count": 2}
```

Numbers and booleans keep their JSON type, and nested Polars dtypes convert
structurally rather than degrading to a debug repr: a `List` column becomes a
JSON array (a sequence of chat turns, say) and a `Struct` column a JSON object
(a record). `state_json=True` reinterprets string inputs as pre-encoded JSON
documents, so a column of JSON becomes structured state.

A length-1 input broadcasts across the frame, so a constant can ride along with
per-row state -- a shared policy, a schema, the document every row is scored
against:

```python
typesafe_eval(
    pl.col("message"),
    pl.lit(refund_policy).alias("policy"),
    questions={"refund_ok": noul("Does the policy support a refund here?")},
)
```

`instructions`, choice option descriptions, score level descriptions and noul
`criteria` all accept `string | object | array | null`, matching
[Advanced: structure](https://docs.typesafe.ai/primitives/advanced) -- pass a
dict or list wherever a string is accepted.

Raw TypeSafe question JSON is accepted alongside the builders, so an existing
payload works unchanged:

```python
questions={"dept": {"type": "choice", "instructions": "Who?",
                    "criteria": {"billing": "money", "tech": None}}}
```

## Contracts: a Pydantic model as the feature set

A **contract** is a Pydantic model naming the features to extract. Each field's
Python type picks the question type, so the struct you want out *is* the
specification of the work:

| Field type | Question | Answer |
|---|---|---|
| `bool` | Noul | probability in `[0, 1]` |
| `Literal[...]` / `Enum` | Choice | the pick + `_confidence` |
| numeric + `score_field(levels=...)` | Score | weighted value + `_confidence` |

```python
from typing import Literal
from pydantic import BaseModel, Field
from polar_llama import typesafe_eval, score_field

class ClauseFeatures(BaseModel):
    is_payment: bool = Field(description="Does this clause create a payment obligation?")
    category: Literal["fees", "term", "liability", "other"] = Field(
        description="What kind of clause is this?")
    severity: float = score_field(
        "How onerous is this clause for the Customer?",
        ["Benign", "Notable", "Onerous"])

df.with_columns(ts=typesafe_eval(pl.col("clause"), contract=ClauseFeatures)).unnest("ts")
```

`contract=` and `questions=` are mutually exclusive and mean the same thing —
`contract_questions(Model)` is the conversion, and you can call it directly to
see what a model produces.

Two things worth knowing:

- **A `bool` field returns a probability, not `True`/`False`.** That is the
  point rather than a lossy conversion — you threshold it where the stakes say
  you should.
- **A plain `str` field is rejected.** TypeSafe answers are typed and
  calibrated over a closed set; there is no free-text primitive. Enumerate the
  possibilities with `Literal[...]`, or use `bool`. The error says so.

Contracts are flat, and `Field(description=...)` becomes the question's
instructions — it is doing real work, so write it like a question.

## The per-line dimension

`typesafe_eval` fans out over **questions** for one state. `typesafe_eval_each`
fans out over **segments**: a `List` column of lines, clauses, passages or
chunks goes in, and every segment comes back with the same contract answered.

```python
from polar_llama import typesafe_eval_each

clauses = (
    df.with_columns(clause=pl.col("contract_text").str.split("\n"))
      .with_columns(f=typesafe_eval_each(pl.col("clause"), contract=ClauseFeatures))
      .explode("f")
      .unnest("f")
)

clauses.filter(pl.col("is_payment") > 0.8)
```

```
┌─────┬─────────┬──────────────────────────────────────────┬────────────┬──────────┬────────┐
│ doc ┆ line_id ┆ line                                     ┆ is_payment ┆ severity ┆ _error │
╞═════╪═════════╪══════════════════════════════════════════╪════════════╪══════════╪════════╡
│ msa ┆ 0       ┆ Definitions. 'Services' means the hosted… ┆ 0.03       ┆ 0.00     ┆ null   │
│ msa ┆ 1       ┆ Fees. Customer shall pay all fees within… ┆ 0.99       ┆ 0.12     ┆ null   │
│ msa ┆ 4       ┆ Limitation of Liability. In no event sha… ┆ 0.04       ┆ 1.52     ┆ null   │
└─────┴─────────┴──────────────────────────────────────────┴────────────┴──────────┴────────┘
```

### Why one request per document, not per line

All of a row's segments are evaluated **together**. Measured against the live
API on an 8-clause contract:

| | input tokens | round trips | document context |
|---|---|---|---|
| One request per line | 2,367 | 8 | ❌ lost |
| One request, 8 per-line questions | **743** | **1** | ✅ kept |

3.2x cheaper, 8x fewer round trips — and it is the only version that keeps
context. A clause reading *"The Term renews automatically unless either party
gives 60 days notice"* is unreadable on its own; batched, the model sees its
neighbours. That is also why the segments go into `state` once and each
question names its line by id: repeating the text inside every question
measured only ~11% more expensive, so the id indirection buys context, not
bytes.

Latency barely moves with question count: 8 segments 0.4s, 320 segments 1.0s,
640 segments 1.2s.

### Chunking

The ceiling is **token-based, not count-based**: 640 questions (~40k input
tokens) succeeded against the live API, 1,200 returned
`400 max_tokens_exceeded`. Long segments reach it at a lower count.

`max_questions` (default 200) caps questions per request, and segments are
chunked to respect it. Each segment costs one question *per contract field*, so
a 3-field contract at `max_questions=6` sends 2 segments per request. Chunks go
out concurrently under the usual `POLAR_LLAMA_MAX_CONCURRENCY`, and `line_id`
stays global, so a chunk boundary never renumbers a line. Raise it to batch
harder; lower it if long segments hit the ceiling.

Note that a chunk only carries its own segments as context. To give every chunk
the whole document (or a title, a policy, the query being matched against),
pass it as an extra context column — length-1 inputs broadcast:

```python
typesafe_eval_each(pl.col("clause"), pl.col("doc_title"), contract=ClauseFeatures)
```

### Shape and failure

The return is `List[Struct{line_id, line, <answers>, _error}]`. A Polars
expression cannot change row count, so `.explode()` is how you get one row per
segment. `include_segment=False` drops the `line` column.

A null segments list comes back null and is never sent; an empty list comes
back empty. If one chunk fails, only its segments carry `_error` — the rest of
the document still resolves.

## Confidence

Choice and score answers carry a `confidence` derived from the probability
distribution. It is the lever for deciding whether to act automatically, and
the threshold should scale with the stakes of the action -- a read-only lookup
and an irreversible refund do not deserve the same bar. See
[Confidence](https://docs.typesafe.ai/confidence).

```python
auto    = out.filter(pl.col("department_confidence") > 0.9)
review  = out.filter(pl.col("department_confidence").is_between(0.5, 0.9))
human   = out.filter(pl.col("department_confidence") < 0.5)
```

## Failure handling

Failures are per row, never per frame:

* A row whose state is entirely null is **never sent** -- it returns all-null
  with a null `_error`, because a missing input is not a failure.
* A row that fails (bad key, malformed question, exhausted retries) gets a
  populated `_error` while every other row still resolves.
* A question TypeSafe declines to answer leaves its columns null, not an error.
* Validation that can be done locally (no questions, a one-option choice, an
  unknown question type, colliding output names) raises in Python or at schema
  time, before any request is billed.

`429 Too Many Requests`, `529 Overloaded` and transient 5xx are retried with
exponential backoff and jitter, honouring `Retry-After`; the jitter is
subtracted so the 30s cap is a real ceiling. `401` and `422` are **not**
retried -- a bad key or a malformed question will not fix itself, and retrying
only bills for it again.

## Configuration

| Variable | Meaning |
|---|---|
| `TYPESAFE_API_KEY` | Bearer token (required) |
| `TYPESAFE_BASE_URL` | API root override; default `https://api.typesafe.ai` |
| `POLAR_LLAMA_MAX_CONCURRENCY` | Shared in-flight request cap (default 64) |
| `POLAR_LLAMA_TYPESAFE_MAX_RETRIES` | Retry budget for 429/529/5xx (default 3) |

```python
from polar_llama import typesafe_models
[m["name"] for m in typesafe_models()]   # ['jev-latest', 'jev-preview']
```

## Testing

`tests/test_typesafe.py` is CI-safe: it drives the real Rust expression against
a local stdlib mock of `POST /v1/systemone`, reached by pointing
`TYPESAFE_BASE_URL` at it. No API key and no network required. A gated live
test at the bottom runs only when `TYPESAFE_API_KEY` is set.

```bash
python -m pytest tests/test_typesafe.py -q
cargo test --lib --no-default-features typesafe
```

The `--no-default-features` flag drops pyo3's `extension-module` so the crate
can be linked into a test binary; `cargo build` and maturin are unaffected.
