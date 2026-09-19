# Playbooks

## Overview

A **playbook** is a set of business rules evaluated against *many rows at once*.

`typesafe_eval` judges one row. `typesafe_eval_each` judges each segment of a
document. A playbook judges a **group of rows together** — the shape you need
for a rule that no single row can violate:

> "Every escalated ticket must get a support reply."
> "Discounts above 20% need a stated justification."
> "Does it make sense that someone has worked here three years and never taken leave?"

None of those are answerable per row. The first needs to see whether a *later*
row exists; the second needs a justification judged against a policy; the third
is not a rule anyone would write in advance at all.

## Two kinds of rule, and only one needs a model

**If you can express it in Polars, express it in Polars.** "Total over $5,000"
is a `sum` — exact, instant, free, and it never hallucinates. Sending it to a
model is strictly worse. That is what `compute=` is for: Polars calculates, the
model judges.

The rules worth a model are the ones you **cannot write down**, either because
the judgment is semantic ("does this justification actually justify it?") or
because you would have to predict every edge case in advance and you cannot.
For that second kind — *is this record self-consistent across all its
dimensions?* — see the written playbook in
**[CONSISTENCY_PLAYBOOK.md](CONSISTENCY_PLAYBOOK.md)**, which is the shape most
people actually want:

```python
from polar_llama import self_consistency, playbook_eval

flags = playbook_eval(df, self_consistency("employee record"), by="employee_id")
flags.sort("review_priority", descending=True).head(20)   # triage queue, no rules written
```

The example below uses an expense cap because it is easy to verify against
ground truth, **not** because it is a good use of a model. It is deliberately
the case where you should reach for `compute=`.

```python
from polar_llama import playbook, rule, playbook_eval

policy = playbook(
    "expense_policy",
    within_limit = rule("No employee may claim more than $5,000 in total."),
    receipts     = rule("Every claim over $75 must reference a receipt."),
)

verdicts = playbook_eval(
    df,
    policy,
    by="employee",                                     # one verdict per employee
    records=["date", "amount_usd", "category"],        # columns -> a record per row
    compute={"total_usd": pl.col("amount_usd").sum()}, # Polars does the arithmetic
    context={"limit_usd": 5000},                       # constant in every state
)

verdicts.filter(pl.col("within_limit") < 0.5)          # the violations
```

```
┌──────────┬───────────┬───────────┬──────────────┬──────────┬────────┐
│ employee ┆ total_usd ┆ row_count ┆ within_limit ┆ receipts ┆ _error │
╞══════════╪═══════════╪═══════════╪══════════════╪══════════╪════════╡
│ alice    ┆ 5400      ┆ 3         ┆ 0.04         ┆ 0.88     ┆ null   │
│ bob      ┆ 950       ┆ 3         ┆ 0.98         ┆ 0.91     ┆ null   │
│ frank    ┆ 6100      ┆ 3         ┆ 0.03         ┆ 0.79     ┆ null   │
└──────────┴───────────┴───────────┴──────────────┴──────────┴────────┘
```

Each rule answers the **probability the rows adhere**, so you threshold for
violations with `< 0.5`, not `>`. Every rule in the playbook rides in the same
request per group, and groups go out in parallel under the usual
`POLAR_LLAMA_MAX_CONCURRENCY` bound with the TypeSafe client's retry and
per-group error isolation.

## Two findings that shape the API

Both are measured against the live API, not assumed.

### 1. Signal dilutes with group size

A rule that only a *sum* could violate, with one violating employee planted
among clean ones:

| rows in state | violating | clean | separated? |
|---|---|---|---|
| 6 | 0.33 | 0.98 | ✅ wide |
| 20 | 0.40 | 0.49 | ⚠️ marginal |
| 60 | 0.20 | 0.39 | ❌ |
| 150 | 0.18 | 0.20 | ❌ |
| 300 | 0.18 | 0.22 | ❌ |

Past ~20 rows it does not merely get the answer wrong — violating and clean
data become **indistinguishable**, and it flags everything. This is not
specific to arithmetic: a purely semantic rule (one un-answered support ticket
among many) separated cleanly at 11 rows and was **missed** at 59.

So `playbook_eval` groups, and warns when a group exceeds
`max_rows_per_group` (default 25).

### 2. Let Polars do the arithmetic

Feeding the model a **pre-computed aggregate** instead of raw rows to add up:

| rows summarised | violating | clean |
|---|---|---|
| 20 | 0.03 | 0.98 |
| 60 | 0.03 | 0.98 |
| 150 | 0.03 | 0.98 |
| 300 | 0.03 | 0.98 |
| 1000 | 0.02 | 0.98 |

Exact at every size, because the state stays small however many rows it
summarises — four totals is four totals whether they came from 20 rows or
1,000.

**This is what `compute=` is for.** Any counting or arithmetic predicate
belongs in Polars: it is exact, free, and deterministic. Reserve the model for
the judgment Polars cannot express — whether a note matches its category,
whether an explanation actually justifies an exception, whether a thread reads
as resolved.

The two compose: `compute=` for the arithmetic, `records=` for the raw rows the
semantic rules need.

## Parameters

| Parameter | Meaning |
|---|---|
| `by` | Group key column(s) — the entity each rule is about. `None` treats the whole frame as one group |
| `records` | Columns gathered into one JSON record per row. Defaults to every column that is not a group key and not computed |
| `compute` | Named Polars aggregates folded into each group's state |
| `context` | Constants added to every group's state (a policy, a threshold, the period) |
| `usage` | Also return `_model` and per-request token/latency accounting |
| `max_rows_per_group` | Warn above this group size (default 25) |

## The state a group receives

```json
{
  "records":   [{"date": "2026-03-01", "amount_usd": 1800, "category": "travel"}, ...],
  "row_count": 3,
  "total_usd": 5400,
  "limit_usd": 5000
}
```

`records` and `row_count` are always present — `row_count` also guarantees the
state is a named object rather than a bare array, so its shape does not depend
on whether `compute` or `context` happened to be passed. Both names are
reserved and rejected in `compute`/`context`.

## Output

One row per group: the group keys, `row_count`, anything from `compute`, one
Float64 column per rule, and `_error` (null on success). A group whose request
fails carries its error while every other group still resolves.

## Writing rules that work

- **Put arithmetic in `compute`.** "Total over $5,000" is a Polars `sum`, not a
  question. Ask the model to judge the total, not to find it.
- **Group at the entity the rule is about.** Per employee, per ticket, per
  order. This is what keeps groups small enough to stay reliable.
- **Say what pass and fail look like** via `passes=` / `fails=` when the
  boundary is subtle.
- **Threshold on confidence for anything consequential.** A rule's answer is a
  probability; route the uncertain ones to a human rather than acting.

## Testing

`tests/test_playbook.py` is CI-safe — it drives the real Rust expression
against a local mock of `POST /v1/systemone`, so no key and no network are
needed. A gated live test at the bottom plants a genuine cross-row violation
and asserts it is caught.

```bash
python -m pytest tests/test_playbook.py -q
```

## See also

- **[CONSISTENCY_PLAYBOOK.md](CONSISTENCY_PLAYBOOK.md)** — the written playbook
  for checking records are self-consistent across many dimensions, without
  enumerating rules. Includes the measured reason to use a *score* rather than
  a yes/no, and why record width and column naming dominate the result.
