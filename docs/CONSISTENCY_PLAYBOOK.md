# A written playbook: checking data is self-consistent

## What this is for

Some rules you can write down. "Total under $5,000" is a `sum` — Polars does it
exactly, instantly, free. Never ask a model.

This playbook is for the other kind: **the rules you cannot write down in
advance, because you would never finish writing them.**

> Someone has worked here three years and has never taken a day of leave.

Nobody has a `tenure > 2y AND pto_taken == 0` rule. And if you wrote it, you
would then need one for the contractor badged in at 3am on a public holiday,
the Senior Staff Engineer hired six weeks ago on the top pay band, the sales
rep with record commission and no customer meetings, the employee on parental
leave who closed a full year of tickets. The list has no end, and the next
anomaly is always the one you did not predict.

So you do not write rules. You ask one question —

> *Does this record hang together?*

— and let the model's knowledge of how the world normally works surface the
combination nobody thought to check.

```python
from polar_llama import self_consistency, playbook_eval

flags = playbook_eval(df, self_consistency("employee record"), by="employee_id")

flags.sort("review_priority", descending=True).head(20)   # your triage queue
```

That is the whole API. No rules written.

## Use a score, not a yes/no

This is the single most important thing in this document.

The obvious framing — *"is this record coherent? yes/no"* — **does not work
well**. Measured on four planted inconsistencies of four different kinds
against four clean controls:

| framing | caught |
|---|---|
| Noul: "is this record coherent?" | **2 of 4** |
| Score: "how strongly does this warrant review?" | **4 of 4**, zero false positives |

| | clean controls | planted issues |
|---|---|---|
| score | 0.07, 0.09, 0.13, 0.18 | 0.98, 1.44, 1.72, 1.82 |

A yes/no question hedges toward "plausible", because in an odd record *most*
fields are still perfectly fine — and the model is being asked to render a
verdict on the whole thing. A graded "how much does this deserve a human's
time?" asks the question you actually mean, and separates cleanly.

`consistency()` therefore returns a **score**. This is not a stylistic choice.

## It needs dimensions to cross-check

"Self-consistent across N dimensions" is exactly right, and **N matters more
than anything else you control.** The same four people, the same planted
issues, only the number of fields changed:

| record width | clean records | planted records | ranking |
|---|---|---|---|
| 5 fields | 0.83, 1.10 | 0.45, 1.23 | ❌ **inverted** |
| 12 fields | 0.14, 0.27 | 1.08, 0.57 | ✅ correct |

With five fields there is nothing to cross-check and the score is noise — the
clean records scored *higher* than the planted ones. Consistency is a
relationship between fields; give it few fields and there are few relationships
to be wrong.

**Send wide records.** Join in the extra columns before you check. Tenure alone
means nothing; tenure beside leave taken, status, login activity, pay band and
promotion history means a great deal.

## Name fields so their meaning is unambiguous

`pto_days` is ambiguous — taken, or remaining? The model cannot judge a
relationship it cannot interpret. Renaming that one column, changing nothing
else:

| field names | clean records | planted records |
|---|---|---|
| `pto_days` | 0.44, 0.50 | 0.53, 0.60 |
| `pto_days_TAKEN_last_12_months` | 0.26, 0.31 | 0.44, 0.67 |

Separation roughly doubled. Column names are part of the prompt. Spell out the
unit, the window and the direction: `days_taken_last_12_months`, not `days`.

## Reading the output

`self_consistency` gives you two columns and a confidence:

| column | meaning |
|---|---|
| `review_priority` | graded — how much this record deserves a human's attention |
| `where` | which *aspect* the problem lives in (time, amounts, activity, status, identity) |
| `where_confidence` | how sure it is about that localisation |

`where` was correct 7 of 8 times, and the one "miss" was defensible — a senior
title at six weeks' tenure is a tenure problem as much as a pay problem.
Confidence is informative: it ran 0.63–0.81 on the clean controls and 0.25–0.29
on the subtler planted ones, so **low confidence marks the cases most worth a
person's eyes**, not the ones to discard.

## How to actually run it

1. **Widen the frame first.** Join everything you reasonably can. Breadth is
   the dominant factor.
2. **Rename ambiguous columns.** One rename measurably doubled separation.
3. **Run it, sort by `review_priority` descending.** You now have a triage
   queue, not a verdict.
4. **Do not threshold blindly.** The score is relative to your data; look at
   the distribution and take the top slice you have capacity to review.
5. **Feed the findings back.** When a review confirms a real problem you *can*
   express in code, write it as a Polars check. The consistency pass is for
   discovering the rules, not for enforcing the ones you already know.

That last point is the loop worth building: this finds the edge cases you did
not predict, and each confirmed one becomes a cheap deterministic check.

## What this is not

- **Not a verdict.** It ranks records for attention. A high score means "a
  person should look", never "this is wrong".
- **Not an audit trail.** Use it to find things, then verify them with code or
  a human before acting on anyone.
- **Not for rules you can already express.** If Polars can compute it, Polars
  should — see `compute=` in [PLAYBOOKS.md](PLAYBOOKS.md).
- **Not reliable on narrow records.** Under roughly ten meaningful fields,
  treat the ranking as unproven on your data.

## Validate on your own data before trusting it

Everything above was measured on planted fixtures. Your data has its own
shape, so do the same thing before relying on it: plant a handful of records
you *know* are wrong among records you know are fine, run the check, and
confirm the planted ones rank above the clean ones. If they do not, widen the
records and disambiguate the names, then measure again.
