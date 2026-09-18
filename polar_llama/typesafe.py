"""TypeSafe System One evaluation as a Polars expression.

TypeSafe (https://docs.typesafe.ai/api) is not a chat-completions API. One
request carries a single ``state`` plus a map of typed ``questions``, and
returns one typed ``answer`` per question:

* :func:`noul` -- a yes/no question, answered with a probability in ``[0, 1]``.
* :func:`choice` -- pick one of a closed set, answered with the pick, a
  probability for every option, and a confidence.
* :func:`score` -- rate against ordered levels, answered with a
  probability-weighted value, a distribution, and a confidence.

That maps onto a DataFrame directly: one state per row, every question
answered in the *same* request, each typed answer landing as an ordinary
column. Ask all the questions you might want at once -- TypeSafe's own
"speculative fan-out" guidance is that batching is dramatically cheaper and
faster than one call per question, and your code decides afterwards which
answers it actually reads.

Configuration comes from the environment, matching the rest of polar-llama:

``TYPESAFE_API_KEY``
    Bearer token (required).
``TYPESAFE_BASE_URL``
    API root override; defaults to ``https://api.typesafe.ai``.
``POLAR_LLAMA_MAX_CONCURRENCY``
    Shared in-flight request cap (default 64).
``POLAR_LLAMA_TYPESAFE_MAX_RETRIES``
    Retry budget for 429/529/5xx, with exponential backoff (default 3).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Type, Union

import polars as pl

from polar_llama.utils import parse_into_expr, register_plugin, parse_version

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr
    from pydantic import BaseModel

if parse_version(pl.__version__) < parse_version("0.20.16"):
    from polars.utils.udfs import _get_shared_lib_location

    lib: Union[str, Path] = _get_shared_lib_location(__file__)
else:
    lib = Path(__file__).parent

#: TypeSafe's flagship System One model.
DEFAULT_MODEL = "jev-latest"

#: Re-exported from ``polar_llama`` under a name that stays unambiguous next
#: to the chat providers' own defaults.
DEFAULT_TYPESAFE_MODEL = DEFAULT_MODEL

#: Every free-form TypeSafe field accepts ``string | object | array | null``.
Entry = Any

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_TYPESAFE_MODEL",
    "DEFAULT_MAX_QUESTIONS",
    "noul",
    "choice",
    "score",
    "choice_field",
    "score_field",
    "contract_questions",
    "typesafe_eval",
    "typesafe_eval_each",
    "list_models",
]

#: Default cap on questions per request for the per-line path. The real limit
#: is token-based: measured against the live API, 640 questions (~40k input
#: tokens) succeeded and 1200 returned ``400 max_tokens_exceeded``. Long
#: segments reach that ceiling at a lower count, so the default leaves room.
DEFAULT_MAX_QUESTIONS = 200


# ============================================================================
# Question builders
# ============================================================================


def noul(
    instructions: Entry,
    *,
    true: Entry = None,
    false: Entry = None,
) -> Dict[str, Any]:
    """A yes/no question, answered with the probability that the answer is yes.

    Parameters
    ----------
    instructions
        The yes/no question. A string, or a dict/list when the question has
        several parts worth labelling.
    true, false
        Optional descriptions of what a yes (near 1) and a no (near 0) mean.
        Pin these down when the boundary is subtle.

    Returns
    -------
    dict
        A question spec for :func:`typesafe_eval`.

    Notes
    -----
    Noul answers carry no ``confidence`` -- a single probability already *is*
    the distribution -- so a noul question produces exactly one output column.

    Examples
    --------
    >>> noul("Does this convey urgency?",
    ...      true="Explicitly time-sensitive",
    ...      false="No urgency expressed")  # doctest: +ELLIPSIS
    {'type': 'noul', ...}
    """
    spec: Dict[str, Any] = {"type": "noul", "instructions": instructions}
    if true is not None or false is not None:
        spec["criteria"] = {"true": true, "false": false}
    return spec


def choice(
    instructions: Entry,
    criteria: Union[Mapping[str, Entry], Sequence[str]],
) -> Dict[str, Any]:
    """Pick one option from a set you define.

    Parameters
    ----------
    instructions
        What the model should decide.
    criteria
        Either a mapping of ``option -> description`` (use ``None`` for an
        option that needs no extra detail) or a plain sequence of option names.
        At least two options are required.

    Returns
    -------
    dict
        A question spec for :func:`typesafe_eval`.

    Notes
    -----
    Option order is preserved, and fixes the order of the ``<id>_p_<option>``
    columns emitted when ``probabilities=True``.

    Examples
    --------
    >>> choice("Which team should handle this?",
    ...        {"billing": "Payments, invoicing, refunds",
    ...         "technical": "Bugs, outages, integrations",
    ...         "sales": None})  # doctest: +ELLIPSIS
    {'type': 'choice', ...}
    """
    if isinstance(criteria, Mapping):
        options = [{"name": str(k), "description": v} for k, v in criteria.items()]
    else:
        options = [{"name": str(k), "description": None} for k in criteria]

    if len(options) < 2:
        raise ValueError("A choice question needs at least 2 options")

    names = [o["name"] for o in options]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate choice options: {names}")

    return {"type": "choice", "instructions": instructions, "options": options}


def score(instructions: Entry, criteria: Sequence[Entry]) -> Dict[str, Any]:
    """Rate the state against ordered, descriptive levels.

    Parameters
    ----------
    instructions
        What the model should rate.
    criteria
        An ordered sequence of level descriptions, lowest first. At least two
        are required.

    Returns
    -------
    dict
        A question spec for :func:`typesafe_eval`.

    Notes
    -----
    The answer is a probability-weighted value across the levels, so it can
    land *between* them -- a 0/1/2 rubric can legitimately answer 1.6. With
    ``probabilities=True`` the per-level columns are keyed by level index
    (``<id>_p_0``, ``<id>_p_1``, ...), matching TypeSafe's legend.

    Examples
    --------
    >>> score("How frustrated is the customer?",
    ...       ["Calm", "Frustrated", "Very angry"])  # doctest: +ELLIPSIS
    {'type': 'score', ...}
    """
    levels = list(criteria)
    if len(levels) < 2:
        raise ValueError("A score question needs at least 2 levels")
    return {"type": "score", "instructions": instructions, "levels": levels}



# ============================================================================
# Contracts: a Pydantic model describing the features to extract
# ============================================================================


def choice_field(description: str, criteria: Union[Mapping[str, Entry], Sequence[str]], **kwargs):
    """A Pydantic ``Field`` for a Choice question with per-option rubrics.

    Only needed when the options want descriptions; a bare ``Literal[...]``
    field already becomes a Choice on its own.

    Examples
    --------
    >>> from typing import Literal
    >>> from pydantic import BaseModel
    >>> from polar_llama import choice_field
    >>>
    >>> class Clause(BaseModel):
    ...     kind: Literal["fees", "term"] = choice_field(
    ...         "What kind of clause is this?",
    ...         {"fees": "Payment, invoicing, refunds", "term": "Duration or renewal"},
    ...     )
    """
    from pydantic import Field

    if isinstance(criteria, Mapping):
        extra = {"criteria": dict(criteria)}
    else:
        extra = {"criteria": {str(k): None for k in criteria}}
    return Field(description=description, json_schema_extra=extra, **kwargs)


def score_field(description: str, levels: Sequence[Entry], **kwargs):
    """A Pydantic ``Field`` for a Score question over ordered levels.

    A numeric field needs levels to become a Score -- there is no default
    rubric to fall back on -- so this is the only way to express one.

    Examples
    --------
    >>> from pydantic import BaseModel
    >>> from polar_llama import score_field
    >>>
    >>> class Review(BaseModel):
    ...     frustration: float = score_field(
    ...         "How frustrated is the customer?",
    ...         ["Calm", "Frustrated", "Very angry"],
    ...     )
    """
    from pydantic import Field

    levels = list(levels)
    if len(levels) < 2:
        raise ValueError("A score field needs at least 2 levels")
    return Field(description=description, json_schema_extra={"levels": levels}, **kwargs)


def _deref(schema: dict, defs: dict) -> dict:
    """Resolve ``$ref`` / ``allOf`` / ``anyOf`` while keeping outer keys.

    Deliberately does NOT reuse ``_pydantic_to_json_schema``: that helper
    strips every sibling of a ``$ref`` for OpenAI strict mode, which would
    throw away the ``description`` on an Enum-typed field -- and the
    description is exactly what becomes the question's instructions here.
    TypeSafe has no JSON-schema mode at all, so none of that applies.
    """
    schema = dict(schema)

    if "allOf" in schema and len(schema["allOf"]) == 1:
        inner = schema.pop("allOf")[0]
        schema = {**_deref(inner, defs), **schema}

    if "anyOf" in schema:
        branches = [b for b in schema.pop("anyOf") if b.get("type") != "null"]
        if len(branches) == 1:
            schema = {**_deref(branches[0], defs), **schema}

    if "$ref" in schema:
        ref = schema.pop("$ref")
        target = defs.get(ref.rsplit("/", 1)[-1], {})
        # Outer keys win: a field's own description beats the Enum class's.
        schema = {**_deref(target, defs), **schema}

    return schema


def _instructions_for(field_name: str, schema: dict) -> str:
    """A field's description, falling back to its (humanized) name."""
    description = schema.get("description")
    if description:
        return description
    return field_name.replace("_", " ").strip()


def contract_questions(contract: Type["BaseModel"]) -> Dict[str, Any]:
    """Derive TypeSafe questions from a Pydantic model.

    The model is the *contract*: each field names one feature to extract, and
    its Python type picks the question type.

    ===============================  =========  ==============================
    Field type                       Question   Answer
    ===============================  =========  ==============================
    ``bool``                         Noul       probability in ``[0, 1]``
    ``Literal[...]`` / ``Enum``      Choice     the pick + confidence
    ``float``/``int`` + levels       Score      weighted value + confidence
    ===============================  =========  ==============================

    Parameters
    ----------
    contract
        A Pydantic ``BaseModel`` subclass.

    Returns
    -------
    dict
        Question id -> question spec, in field-declaration order. Usable
        directly as ``questions=``.

    Raises
    ------
    ValueError
        For a field TypeSafe cannot answer. Notably a plain ``str`` field:
        TypeSafe returns typed, calibrated answers over a closed set, and has
        no free-text primitive -- use ``Literal[...]`` to enumerate the
        possibilities.

    Notes
    -----
    A ``bool`` field comes back as a **probability**, not a ``True``/``False``.
    That is the point rather than a lossy conversion: you threshold it where
    the stakes of the decision say you should.

    Examples
    --------
    >>> from typing import Literal
    >>> from pydantic import BaseModel, Field
    >>> from polar_llama import contract_questions, score_field
    >>>
    >>> class ClauseFeatures(BaseModel):
    ...     is_payment: bool = Field(description="Does this clause create a payment obligation?")
    ...     category: Literal["fees", "term", "liability"] = Field(description="Clause type?")
    ...     severity: float = score_field("How onerous?", ["Benign", "Notable", "Onerous"])
    >>>
    >>> sorted(contract_questions(ClauseFeatures))
    ['category', 'is_payment', 'severity']
    """
    try:
        from pydantic import BaseModel
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Pydantic is required for contracts. Install with: pip install pydantic>=2.0.0"
        ) from exc

    if not (isinstance(contract, type) and issubclass(contract, BaseModel)):
        raise TypeError(
            f"contract must be a Pydantic BaseModel subclass, got {contract!r}"
        )

    schema = contract.model_json_schema()
    defs = schema.get("$defs", {})
    questions: Dict[str, Any] = {}

    for field_name, raw in schema.get("properties", {}).items():
        field = _deref(raw, defs)
        instructions = _instructions_for(field_name, field)
        levels = field.get("levels")
        enum = field.get("enum")
        ftype = field.get("type")

        if levels is not None:
            questions[field_name] = score(instructions, levels)
        elif enum is not None:
            criteria = field.get("criteria") or {str(v): None for v in enum}
            # Keep the declared option order, and let `criteria` supply rubrics
            # only for options the enum actually has.
            ordered = {str(v): criteria.get(str(v)) for v in enum}
            questions[field_name] = choice(instructions, ordered)
        elif ftype == "boolean":
            questions[field_name] = noul(instructions)
        elif ftype in ("number", "integer"):
            raise ValueError(
                f"Contract field '{field_name}' is numeric but has no levels, so there is "
                f"no rubric to score against. Use score_field(description, levels=[...]) "
                f"to say what the levels mean."
            )
        elif ftype == "string":
            raise ValueError(
                f"Contract field '{field_name}' is a plain str. TypeSafe answers are typed "
                f"and calibrated over a closed set -- there is no free-text primitive -- so "
                f"enumerate the possibilities with Literal[...] (a Choice), or use bool for "
                f"a yes/no (a Noul)."
            )
        else:
            raise ValueError(
                f"Contract field '{field_name}' has unsupported type {ftype!r}. A contract is "
                f"flat: use bool (Noul), Literal[...]/Enum (Choice), or a numeric field with "
                f"score_field(...) levels (Score)."
            )

    if not questions:
        raise ValueError(f"Contract {contract.__name__} declares no fields")

    return questions


def _resolve_questions(
    questions: Optional[Mapping[str, Any]],
    contract: Optional[Type["BaseModel"]],
) -> Mapping[str, Any]:
    """Take exactly one of `questions=` / `contract=`."""
    if (questions is None) == (contract is None):
        raise ValueError("Pass exactly one of questions= or contract=")
    return contract_questions(contract) if contract is not None else questions


# ============================================================================
# Question encoding
# ============================================================================

_RAW_KEYS = {"type", "instructions", "criteria", "options", "levels"}


def _normalize(question_id: str, spec: Any) -> Dict[str, Any]:
    """Accept both the builder output and raw TypeSafe API question JSON.

    Passing the documented wire shape straight through (``{"type": "choice",
    "criteria": {...}}``) is supported so an existing TypeSafe payload works
    here unchanged.
    """
    if not isinstance(spec, Mapping):
        raise TypeError(
            f"Question '{question_id}' must be a dict built by noul()/choice()/"
            f"score(), or raw TypeSafe question JSON; got {type(spec).__name__}"
        )

    spec = dict(spec)
    qtype = spec.get("type")

    if qtype == "noul":
        criteria = spec.get("criteria")
        out = noul(
            spec.get("instructions"),
            true=(criteria or {}).get("true"),
            false=(criteria or {}).get("false"),
        )
    elif qtype == "choice":
        # Builder form carries `options`; the raw API form carries `criteria`.
        source = spec.get("options")
        if source is None:
            source = spec.get("criteria")
            if source is None:
                raise ValueError(f"Choice question '{question_id}' needs criteria")
            out = choice(spec.get("instructions"), source)
        else:
            out = {
                "type": "choice",
                "instructions": spec.get("instructions"),
                "options": [dict(o) for o in source],
            }
            if len(out["options"]) < 2:
                raise ValueError(
                    f"Choice question '{question_id}' needs at least 2 options"
                )
    elif qtype == "score":
        levels = spec.get("levels")
        if levels is None:
            levels = spec.get("criteria")
        if levels is None:
            raise ValueError(f"Score question '{question_id}' needs criteria")
        out = score(spec.get("instructions"), levels)
    else:
        raise ValueError(
            f"Question '{question_id}' has unknown type {qtype!r}; "
            "expected 'noul', 'choice' or 'score'"
        )

    unknown = set(spec) - _RAW_KEYS
    if unknown:
        raise ValueError(
            f"Question '{question_id}' has unexpected keys: {sorted(unknown)}"
        )

    out["id"] = question_id
    return out


def _encode_questions(questions: Mapping[str, Any]) -> str:
    """Serialize questions to the ordered JSON array the Rust side expects.

    An array rather than an object: declared order fixes output column order,
    and a JSON object round-trip through Rust would re-sort the keys.
    """
    if not isinstance(questions, Mapping):
        raise TypeError("questions must be a mapping of question id -> question spec")
    if not questions:
        raise ValueError("typesafe_eval requires at least one question")

    specs = []
    for question_id, spec in questions.items():
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f"Question ids must be non-empty strings; got {question_id!r}")
        if "#" in question_id:
            # Reserved: the per-line path keys answers "<line index>#<id>".
            raise ValueError(
                f"Question id {question_id!r} may not contain '#' (reserved separator)"
            )
        specs.append(_normalize(question_id, spec))

    return json.dumps(specs, ensure_ascii=False, sort_keys=False)


# ============================================================================
# Expression
# ============================================================================


def typesafe_eval(
    *state: "IntoExpr",
    questions: Optional[Mapping[str, Any]] = None,
    contract: Optional[Type["BaseModel"]] = None,
    model: Optional[str] = None,
    probabilities: bool = False,
    usage: bool = False,
    state_json: bool = False,
) -> pl.Expr:
    """Evaluate each row against typed TypeSafe questions.

    Every question is answered in a single request per row, and the answers
    come back as a Struct you can ``.unnest()`` into ordinary columns.

    Parameters
    ----------
    *state
        One or more expressions forming the state. A single expression is sent
        as that bare value; several are sent as a JSON object keyed by column
        name, which is the shape TypeSafe recommends for most requests because
        each part of the state keeps a descriptive name. ``List`` and
        ``Struct`` columns convert to JSON arrays and objects. A length-1
        input broadcasts, so a constant -- a shared policy, a schema, the
        document every row is scored against -- can ride along with per-row
        state.
    questions
        Mapping of question id to a spec from :func:`noul`, :func:`choice` or
        :func:`score` (raw TypeSafe question JSON is also accepted). Insertion
        order fixes output column order. Mutually exclusive with ``contract``.
    contract
        A Pydantic model describing the features to extract, converted by
        :func:`contract_questions`. Mutually exclusive with ``questions``.
    model
        System One model. Defaults to ``"jev-latest"``.
    probabilities
        Also emit the full distribution for choice/score questions, as
        ``<id>_p_<option>`` / ``<id>_p_<level index>``.
    usage
        Also emit ``_model`` (the resolved model version that actually ran),
        ``_input_tokens``, ``_output_tokens`` and ``_latency_ms``.
    state_json
        Treat string inputs as pre-encoded JSON documents rather than literal
        text, so a column of JSON becomes structured state.

    Returns
    -------
    polars.Expr
        A Struct expression. Per question: a ``noul`` contributes ``<id>``
        (Float64); a ``choice`` contributes ``<id>`` (String) and
        ``<id>_confidence`` (Float64); a ``score`` contributes ``<id>``
        (Float64) and ``<id>_confidence`` (Float64). ``_error`` is always
        present and null on success.

    Notes
    -----
    Rows whose state is entirely null are never sent, and come back all-null
    with a null ``_error`` -- a missing input is not a failure. Rows that fail
    (bad key, malformed question, exhausted retries) get a populated
    ``_error`` while the rest of the frame still resolves, so one bad row never
    takes down the job.

    Confidence is the lever for acting automatically versus routing to a
    human, and the threshold should scale with the stakes of the action.

    Examples
    --------
    >>> import polars as pl
    >>> from polar_llama import typesafe_eval, noul, choice, score
    >>>
    >>> df = pl.DataFrame({
    ...     "message": ["Help! My payouts have been failing for 3 days."]
    ... })
    >>> result = df.with_columns(
    ...     ts=typesafe_eval(
    ...         pl.col("message"),
    ...         questions={
    ...             "is_urgent": noul("Does this convey urgency?"),
    ...             "department": choice(
    ...                 "Which team should handle this?",
    ...                 {"billing": "Payments, invoicing, refunds",
    ...                  "technical": "Bugs, outages, integrations"},
    ...             ),
    ...             "frustration": score(
    ...                 "How frustrated is the customer?",
    ...                 ["Calm", "Frustrated", "Very angry"],
    ...             ),
    ...         },
    ...     )
    ... ).unnest("ts")  # doctest: +SKIP
    >>>
    >>> # Act on the confident rows, route the rest to a human.
    >>> auto = result.filter(pl.col("department_confidence") > 0.9)  # doctest: +SKIP
    """
    if not state:
        raise ValueError("typesafe_eval needs at least one state expression")

    exprs = [parse_into_expr(e) for e in state]

    kwargs = {
        "questions": _encode_questions(_resolve_questions(questions, contract)),
        "model": model,
        "probabilities": bool(probabilities),
        "usage": bool(usage),
        "state_json": bool(state_json),
    }

    return register_plugin(
        args=exprs,
        symbol="typesafe_eval",
        is_elementwise=True,
        lib=lib,
        kwargs=kwargs,
    )


def typesafe_eval_each(
    segments: "IntoExpr",
    *context: "IntoExpr",
    questions: Optional[Mapping[str, Any]] = None,
    contract: Optional[Type["BaseModel"]] = None,
    model: Optional[str] = None,
    probabilities: bool = False,
    usage: bool = False,
    include_segment: bool = True,
    max_questions: int = DEFAULT_MAX_QUESTIONS,
) -> pl.Expr:
    """Apply one contract to *every segment* of a document, per row.

    Where :func:`typesafe_eval` fans out over questions for a single state,
    this fans out over **segments**: a column of lines, clauses, passages or
    chunks goes in, and every segment comes back with the same contract
    answered for it.

    The segments of a row are evaluated **together in one request** rather than
    one request per segment. Measured against the live API on an 8-clause
    contract, that is 3.2x fewer input tokens and 8x fewer round trips -- and
    it is the only version that keeps context, because the model sees the
    neighbouring lines while judging each one. A clause like *"renews
    automatically unless either party gives notice"* is unreadable alone.

    Parameters
    ----------
    segments
        A ``List`` column of segments -- one list per row. Split however suits
        the data, e.g. ``pl.col("text").str.split("\n")``.
    *context
        Further columns folded into every request's state as shared context
        (a title, a policy, the query being matched against). Length-1 inputs
        broadcast.
    questions
        Mapping of question id to a :func:`noul` / :func:`choice` /
        :func:`score` spec. Mutually exclusive with ``contract``.
    contract
        A Pydantic model describing the features to extract from each segment.
        Mutually exclusive with ``questions``.
    model
        System One model. Defaults to ``"jev-latest"``.
    probabilities
        Also emit the full distribution for choice/score questions.
    usage
        Also emit ``_model`` and per-request token/latency accounting.
    include_segment
        Emit the segment's own text as a ``line`` column (default True).
    max_questions
        Cap on questions per request; segments are chunked to respect it. Each
        segment costs one question per contract field, so the segments per
        request is this divided by the contract size. Raise it to batch harder,
        lower it if long segments hit ``max_tokens_exceeded``.

    Returns
    -------
    polars.Expr
        ``List[Struct{line_id, line, <answers>, _error}]`` -- one entry per
        segment, in order. A Polars expression cannot change row count, so
        ``.explode()`` it to get one row per segment.

    Notes
    -----
    A null segments list comes back as a null list; an empty list as an empty
    list. If a chunk fails, every segment in that chunk carries the error in
    ``_error`` while the rest of the document still resolves.

    Examples
    --------
    >>> import polars as pl
    >>> from typing import Literal
    >>> from pydantic import BaseModel, Field
    >>> from polar_llama import typesafe_eval_each, score_field
    >>>
    >>> class ClauseFeatures(BaseModel):
    ...     is_payment: bool = Field(
    ...         description="Does this clause create a payment obligation on the Customer?"
    ...     )
    ...     category: Literal["fees", "term", "liability", "other"] = Field(
    ...         description="What kind of clause is this?"
    ...     )
    ...     severity: float = score_field(
    ...         "How onerous is this clause for the Customer?",
    ...         ["Benign", "Notable", "Onerous"],
    ...     )
    >>>
    >>> clauses = df.with_columns(
    ...     clause=pl.col("contract_text").str.split("\n")
    ... ).with_columns(
    ...     f=typesafe_eval_each(pl.col("clause"), contract=ClauseFeatures)
    ... ).explode("f").unnest("f")  # doctest: +SKIP
    >>>
    >>> # One row per clause, typed and filterable.
    >>> clauses.filter(pl.col("is_payment") > 0.8)  # doctest: +SKIP
    """
    exprs = [parse_into_expr(segments)] + [parse_into_expr(e) for e in context]

    if max_questions < 1:
        raise ValueError("max_questions must be at least 1")

    kwargs = {
        "questions": _encode_questions(_resolve_questions(questions, contract)),
        "model": model,
        "probabilities": bool(probabilities),
        "usage": bool(usage),
        "include_segment": bool(include_segment),
        "max_questions": int(max_questions),
    }

    return register_plugin(
        args=exprs,
        symbol="typesafe_eval_each",
        is_elementwise=True,
        lib=lib,
        kwargs=kwargs,
    )


def list_models() -> List[Dict[str, str]]:
    """List the System One models available to ``TYPESAFE_API_KEY``.

    Returns
    -------
    list of dict
        One entry per model, with ``name`` and (when reported) ``description``
        and ``release_date``.

    Examples
    --------
    >>> from polar_llama.typesafe import list_models
    >>> [m["name"] for m in list_models()]  # doctest: +SKIP
    ['jev-latest', 'jev-preview']
    """
    try:
        from polar_llama.polar_llama import _typesafe_list_models
    except ImportError as exc:  # pragma: no cover - build/install problem
        raise RuntimeError(
            "polar_llama native extension is unavailable; TypeSafe model "
            "listing requires the compiled module"
        ) from exc
    return _typesafe_list_models()
