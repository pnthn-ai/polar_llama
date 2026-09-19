"""Business rules evaluated against *many rows* at once.

`typesafe_eval` judges one row. `typesafe_eval_each` judges each segment of a
document. A **playbook** judges a *set of rows together* -- the shape you need
for a rule no single row can violate:

    "No employee may claim more than $5,000 in total."
    "Every escalated ticket must get a support reply."
    "Discounts above 20% need a stated justification."

Rows are grouped, each group's rows become a JSON array of records, and every
rule in the playbook is answered for that group in one request. Groups are
evaluated in parallel under the shared concurrency bound, with the retry and
per-row error isolation the TypeSafe client already provides.

Two things about this are measured, not assumed, and they shape the API:

**Signal dilutes with group size.** On a rule only a *sum* could violate, the
answer separated cleanly at 6 rows per group (0.33 violating vs 0.98 clean),
was marginal at 20 (0.40 vs 0.49), and had collapsed by 60 (0.20 vs 0.39) --
violating and clean data became indistinguishable. A purely semantic rule (one
un-answered ticket among many) held at 11 rows and was missed at 59. So a
playbook groups, and warns when a group gets big enough to lose the signal.

**Let Polars do the arithmetic.** Feeding the model a pre-computed aggregate
instead of raw rows to add up was exact at every size tested, up to 1,000 rows
summarised per group (0.03 violating vs 0.98 clean) -- because the state stays
small however many rows it summarises. `compute=` exists for that: any counting
or arithmetic predicate belongs in Polars, which is exact and free. Reserve the
model for the judgment Polars cannot express.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Union

import polars as pl

from polar_llama.typesafe import DEFAULT_MODEL, noul, typesafe_eval

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr

__all__ = [
    "DEFAULT_MAX_ROWS_PER_GROUP",
    "Playbook",
    "rule",
    "playbook",
    "playbook_eval",
]

#: Group size past which a single violation starts getting lost in the noise.
#: Measured: clean vs violating separated at 6 rows, marginal at 20, gone by 60.
DEFAULT_MAX_ROWS_PER_GROUP = 25


# ============================================================================
# Rules
# ============================================================================


def rule(
    statement: Any,
    *,
    passes: Any = None,
    fails: Any = None,
) -> Dict[str, Any]:
    """One business rule, answered as the probability that the rows adhere.

    Parameters
    ----------
    statement
        The rule, stated as you would to a colleague -- "No employee may claim
        more than $5,000 in total". A dict or list is accepted where a rule has
        several parts worth labelling.
    passes, fails
        Optional descriptions of what adherence and violation look like. Worth
        writing when the boundary is subtle; they become the noul's criteria.

    Returns
    -------
    dict
        A question spec. The answer is a probability in ``[0, 1]`` where **1
        means the rows adhere** and 0 means they violate -- so you threshold
        for violations with ``< 0.5``, not ``>``.

    Examples
    --------
    >>> rule("No employee may claim more than $5,000 in total.")  # doctest: +ELLIPSIS
    {'type': 'noul', ...}
    """
    instructions: Any
    if isinstance(statement, str):
        instructions = {
            "rule": statement,
            "evaluate": "Do these records, taken together, satisfy the rule?",
        }
    else:
        instructions = statement

    return noul(
        instructions,
        true=passes if passes is not None else "The records satisfy the rule",
        false=fails if fails is not None else "The records violate the rule",
    )


class Playbook:
    """A named, ordered set of business rules.

    Reusable across frames and runs: define the policy once, apply it wherever
    the data lives.

    Examples
    --------
    >>> from polar_llama import playbook, rule
    >>> expenses = playbook(
    ...     "expense_policy",
    ...     within_limit=rule("No employee may claim more than $5,000 in total."),
    ...     receipts=rule("Every claim over $75 must reference a receipt."),
    ... )
    >>> list(expenses.rules)
    ['within_limit', 'receipts']
    """

    def __init__(self, name: str, rules: Mapping[str, Any]):
        if not name or not isinstance(name, str):
            raise ValueError("A playbook needs a non-empty name")
        if not rules:
            raise ValueError(f"Playbook {name!r} declares no rules")
        self.name = name
        self.rules: Dict[str, Any] = dict(rules)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Playbook({self.name!r}, rules={list(self.rules)})"

    def __len__(self) -> int:
        return len(self.rules)


def playbook(name: str, **rules: Any) -> Playbook:
    """Build a named :class:`Playbook` from keyword rules.

    Each keyword becomes a rule id and an output column. Declaration order is
    preserved.
    """
    return Playbook(name, rules)


# ============================================================================
# Evaluation
# ============================================================================


def _as_list(value: Optional[Union[str, Sequence[str]]]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def playbook_eval(
    df: pl.DataFrame,
    playbook: Playbook,
    *,
    by: Optional[Union[str, Sequence[str]]] = None,
    records: Optional[Sequence[str]] = None,
    compute: Optional[Mapping[str, "IntoExpr"]] = None,
    context: Optional[Mapping[str, Any]] = None,
    model: Optional[str] = None,
    usage: bool = False,
    max_rows_per_group: int = DEFAULT_MAX_ROWS_PER_GROUP,
) -> pl.DataFrame:
    """Evaluate a playbook against each group of rows.

    Parameters
    ----------
    df
        The rows to judge.
    playbook
        The rules to apply. Every rule is answered in the *same* request per
        group, so asking more rules costs little beyond their own answers.
    by
        Group key column(s) -- the entity each rule is about (an employee, a
        ticket, an order). ``None`` treats the whole frame as one group, which
        is rarely what you want: see the warning in Notes.
    records
        Columns gathered into one JSON record per row. Defaults to every
        column that is not a group key and not produced by ``compute``.
    compute
        Named Polars aggregate expressions folded into each group's state, e.g.
        ``{"total_usd": pl.col("amount").sum()}``. **Put every arithmetic or
        counting predicate here.** Polars computes it exactly and for free, and
        the model then judges a small, precise summary instead of adding up raw
        rows -- which is both more accurate and independent of group size.
    context
        Constant values added to every group's state: a policy document, a
        threshold, the period under review.
    model
        System One model. Defaults to ``"jev-latest"``.
    usage
        Also return ``_model`` and per-request token/latency accounting.
    max_rows_per_group
        Warn when a group carries more rows than this. Default 25, from
        measured separation (clean vs violating) falling apart past ~20 rows.

    Returns
    -------
    polars.DataFrame
        One row per group: the group keys, anything from ``compute``, one
        Float64 column per rule holding the **probability the rows adhere**,
        and ``_error`` (null on success).

    Notes
    -----
    Threshold for violations with ``< 0.5``, not ``>`` -- the answer is
    adherence, so low means broken.

    With ``by=None`` the whole frame becomes one state. A single violation
    among many rows measurably gets lost, so this warns above
    ``max_rows_per_group`` and is only sensible for genuinely small frames.

    Examples
    --------
    >>> import polars as pl
    >>> from polar_llama import playbook, rule, playbook_eval
    >>>
    >>> expenses = playbook(
    ...     "expense_policy",
    ...     within_limit=rule("No employee may claim more than $5,000 in total."),
    ... )
    >>> verdicts = playbook_eval(
    ...     df,
    ...     expenses,
    ...     by="employee",
    ...     records=["date", "amount_usd", "category"],
    ...     compute={"total_usd": pl.col("amount_usd").sum()},
    ... )  # doctest: +SKIP
    >>> verdicts.filter(pl.col("within_limit") < 0.5)  # doctest: +SKIP
    """
    if not isinstance(playbook, Playbook):
        raise TypeError(
            f"playbook must be a Playbook (see polar_llama.playbook()), got "
            f"{type(playbook).__name__}"
        )
    if df.is_empty():
        raise ValueError("playbook_eval needs at least one row")

    keys = _as_list(by)
    for k in keys:
        if k not in df.columns:
            raise ValueError(f"Group key {k!r} is not a column in the frame")

    reserved = {"records", "row_count"}
    computed = dict(compute or {})
    for name in computed:
        if name in keys:
            raise ValueError(f"compute name {name!r} collides with a group key")
        if name in reserved:
            raise ValueError(f"compute name {name!r} is reserved by playbook_eval")
    for name in (context or {}):
        if name in reserved:
            raise ValueError(f"context name {name!r} is reserved by playbook_eval")

    if records is None:
        record_cols = [c for c in df.columns if c not in keys and c not in computed]
    else:
        record_cols = list(records)
        missing = [c for c in record_cols if c not in df.columns]
        if missing:
            raise ValueError(f"records columns not in the frame: {missing}")
    if not record_cols:
        raise ValueError("No record columns to send -- every column is a group key")

    # One row per group: the rows as a List[Struct], plus any computed aggregates.
    # `row_count` is always sent, which also guarantees the state is a named
    # JSON *object* rather than a bare array -- a single state column would be
    # sent as the array alone, making the shape depend on whether `compute` or
    # `context` happened to be passed.
    agg = [pl.struct(record_cols).alias("records"), pl.len().alias("row_count")]
    agg += [expr.alias(name) for name, expr in computed.items()]

    if keys:
        grouped = df.group_by(keys, maintain_order=True).agg(agg)
    else:
        # Group by a constant rather than `select(agg)`: `pl.struct(...)` is
        # elementwise in a select context and would yield one row per input
        # row. Routing both paths through group_by keeps them identical.
        grouped = df.group_by(pl.lit(1).alias("_all")).agg(agg).drop("_all")

    biggest = int(grouped["row_count"].max() or 0)
    if biggest > max_rows_per_group:
        warnings.warn(
            f"playbook {playbook.name!r}: largest group has {biggest} rows "
            f"(max_rows_per_group={max_rows_per_group}). A single violation "
            f"measurably gets lost in groups this large -- separation between "
            f"clean and violating data collapses past ~20 rows. Group more "
            f"finely, or move the predicate into compute= so Polars does the "
            f"arithmetic and the model judges a small summary instead.",
            UserWarning,
            stacklevel=2,
        )

    state: List["IntoExpr"] = [pl.col("records"), pl.col("row_count")]
    state += [pl.col(name) for name in computed]
    for name, value in (context or {}).items():
        state.append(pl.lit(value).alias(name))

    out = grouped.with_columns(
        _verdict=typesafe_eval(*state, questions=playbook.rules, model=model, usage=usage)
    ).unnest("_verdict")

    # `records` was the payload; `row_count` stays, since it is how a caller
    # sees which groups are large enough for a violation to get lost.
    return out.drop("records")
