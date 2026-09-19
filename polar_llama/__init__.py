from __future__ import annotations

from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Optional,
    Tuple,
    Union,
    Type,
    Dict,
    Any,
    List,
)
import json

import polars as pl

from polar_llama.utils import parse_into_expr, register_plugin, parse_version
from polar_llama.types import CacheStrategy, CacheConfig, CacheMetrics
from polar_llama.checkpoint import Checkpoint, checkpointed_expr
from polar_llama.keys import (
    config_fingerprint,
    canonicalize_messages_input,
    endpoint_fingerprint_input,
    request_fingerprint as _request_fingerprint,
)
from polar_llama.dedup import (
    DedupeStats,
    ResponseCache,
    ResponseCacheStore,
    _parse_ttl,
    deduped_expr,
)
from polar_llama import pricing
from polar_llama.pricing import register_model_price, set_price_table

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr
    from pydantic import BaseModel

if parse_version(pl.__version__) < parse_version("0.20.16"):
    from polars.utils.udfs import _get_shared_lib_location

    lib: str | Path = _get_shared_lib_location(__file__)
else:
    lib = Path(__file__).parent

# Import Provider enum directly from the extension module
try:
    # First try relative import from the extension module in current directory
    from .polar_llama import Provider
except ImportError:
    # Fallback to try absolute import
    try:
        from polar_llama.polar_llama import Provider
    except ImportError:
        # Define a basic Provider class as fallback if neither import works
        class Provider:
            OPENAI = "openai"
            ANTHROPIC = "anthropic"
            GEMINI = "gemini"
            GROQ = "groq"
            BEDROCK = "bedrock"

            def __init__(self, provider_str):
                self.value = provider_str

            def __str__(self):
                return self.value


# Import the streaming pyfunction directly from the extension module. Unlike
# the other expressions, `inference_stream` cannot go through
# `register_plugin` (kwargs cross that boundary via serde, which cannot carry
# a Python callable), so it calls this pyfunction from a `map_batches` UDF.
try:
    from .polar_llama import _stream_inference_batch
except ImportError:
    try:
        from polar_llama.polar_llama import _stream_inference_batch
    except ImportError:
        _stream_inference_batch = None


# Import and initialize the expressions helper to ensure expressions are registered
from polar_llama.expressions import ensure_expressions_registered, get_lib_path

# Ensure the expressions are registered
ensure_expressions_registered()
# Update the lib path to make sure we're using the actual library
lib = get_lib_path()


def _pydantic_to_json_schema(model: Type["BaseModel"]) -> dict:
    """Convert a Pydantic model to JSON schema."""
    try:
        from pydantic import BaseModel

        if not issubclass(model, BaseModel):
            raise ValueError("response_model must be a Pydantic BaseModel subclass")

        # Get the JSON schema from the Pydantic model
        schema = model.model_json_schema()

        # Recursively add additionalProperties: false to all objects
        # This is required by some providers like Groq.
        # We skip objects that already have additionalProperties defined
        # (e.g. a user-supplied response_model using Dict[str, str], which
        # needs dynamic keys and is therefore incompatible with OpenAI strict
        # mode -- see _validate_strict_mode_schema for a warning about that).
        # Taxonomy-generated models (see _create_taxonomy_pydantic_model) never
        # contain dynamic-key map objects, so this skip does not apply to them.
        def add_additional_properties_false(obj):
            if isinstance(obj, dict):
                if obj.get("type") == "object" and "additionalProperties" not in obj:
                    # Only add if not already present (Dict types already have it defined)
                    obj["additionalProperties"] = False
                # Defensive OpenAI-strict-mode enforcement: whenever an object
                # node declares fixed `properties` (i.e. it is not a dynamic
                # map with additionalProperties left permissive), `required`
                # must list every property key. Pydantic v2 already does this
                # for models where every field is required, but we enforce it
                # here too so a future Pydantic change (or an Optional field
                # with a default) can never silently reintroduce a strict-mode
                # rejection like the one in issue #51.
                if (
                    obj.get("type") == "object"
                    and "properties" in obj
                    and obj.get("additionalProperties") is False
                ):
                    obj["required"] = list(obj["properties"].keys())
                # Recursively process nested objects
                for key, value in obj.items():
                    if isinstance(value, dict):
                        add_additional_properties_false(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                add_additional_properties_false(item)

        add_additional_properties_false(schema)

        # OpenAI strict mode forbids a `$ref` node from carrying sibling
        # keywords: pydantic v2 emits a field that references a submodel AND
        # has a description as `{"$ref": "#/$defs/X", "description": "..."}`,
        # which OpenAI rejects with "$ref cannot have keywords {'description'}".
        # (Taxonomy fields hit this: each outer field is a $ref to a
        # <Field>Result model plus the taxonomy field's description.) The
        # description is cosmetic for structured output, so drop every sibling
        # of `$ref` so the reference stands alone.
        def strip_ref_siblings(obj):
            if isinstance(obj, dict):
                if "$ref" in obj and len(obj) > 1:
                    ref = obj["$ref"]
                    obj.clear()
                    obj["$ref"] = ref
                for value in obj.values():
                    if isinstance(value, dict):
                        strip_ref_siblings(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                strip_ref_siblings(item)

        strip_ref_siblings(schema)
        return schema
    except ImportError:
        raise ImportError(
            "Pydantic is required for structured outputs. Install with: pip install pydantic>=2.0.0"
        )


def _extract_type_from_anyof(any_of: list, root_schema: dict) -> tuple:
    """
    Extract the non-null type from an anyOf pattern (used for Optional fields).

    Returns a tuple of (type_string, type_schema) where type_schema contains
    the full schema for complex types like arrays or objects.
    """
    for option in any_of:
        # Handle $ref in anyOf options
        if "$ref" in option:
            ref_path = option["$ref"]
            if ref_path.startswith("#/"):
                ref_parts = ref_path[2:].split("/")
                ref_schema = root_schema
                for part in ref_parts:
                    ref_schema = ref_schema.get(part, {})
                return (ref_schema.get("type", "object"), ref_schema)

        option_type = option.get("type")
        if option_type and option_type != "null":
            return (option_type, option)

    return ("string", {})  # Default fallback


def _json_schema_to_polars_dtype(schema: dict, root_schema: dict = None) -> pl.DataType:
    """Convert a JSON schema to a Polars DataType."""
    # Keep reference to root schema for resolving $ref
    if root_schema is None:
        root_schema = schema

    properties = schema.get("properties", {})

    fields = []
    for field_name, field_schema in properties.items():
        pl_type = _field_schema_to_polars_dtype(field_schema, root_schema)
        fields.append(pl.Field(field_name, pl_type))

    # Add error fields for error handling (only at the root level)
    if root_schema == schema:
        fields.append(pl.Field("_error", pl.Utf8))
        fields.append(pl.Field("_details", pl.Utf8))
        fields.append(pl.Field("_raw", pl.Utf8))

    return pl.Struct(fields)


def _field_schema_to_polars_dtype(field_schema: dict, root_schema: dict) -> pl.DataType:
    """Convert a single field's JSON schema to a Polars DataType."""
    # Handle $ref references
    if "$ref" in field_schema:
        ref_path = field_schema["$ref"]
        if ref_path.startswith("#/"):
            ref_parts = ref_path[2:].split("/")
            ref_schema = root_schema
            for part in ref_parts:
                ref_schema = ref_schema.get(part, {})
            return _json_schema_to_polars_dtype(ref_schema, root_schema)
        else:
            return pl.Utf8  # Unknown reference format

    # Handle anyOf patterns (used for Optional fields in Pydantic v2)
    if "anyOf" in field_schema:
        field_type, type_schema = _extract_type_from_anyof(
            field_schema["anyOf"], root_schema
        )
        # Create a temporary field_schema with the extracted type info
        field_schema = {**type_schema, "type": field_type}

    field_type = field_schema.get("type")

    # Handle type arrays like ["string", "null"] (alternative Optional format)
    if isinstance(field_type, list):
        # Find the non-null type
        for t in field_type:
            if t != "null":
                field_type = t
                break
        else:
            field_type = "string"  # Default if only null

    # Map JSON schema types to Polars types
    if field_type == "string":
        return pl.Utf8
    elif field_type == "integer":
        return pl.Int64
    elif field_type == "number":
        return pl.Float64
    elif field_type == "boolean":
        return pl.Boolean
    elif field_type == "array":
        # Handle arrays - use List type
        items_schema = field_schema.get("items", {})
        if items_schema:
            items_dtype = _field_schema_to_polars_dtype(items_schema, root_schema)
            return pl.List(items_dtype)
        return pl.List(pl.Utf8)  # Default to string list
    elif field_type == "object":
        # Check if this is a Dict[str, str] type (has additionalProperties).
        # Taxonomy-generated models never produce this shape (see
        # _create_taxonomy_pydantic_model, which uses List[ThinkingItem]
        # instead of Dict[str, str] to stay OpenAI-strict-mode compliant --
        # issue #51). This branch remains for user-supplied response_models
        # that still use Dict[str, str] with non-strict providers.
        if (
            "additionalProperties" in field_schema
            and field_schema["additionalProperties"] != False
        ):
            # This is a dictionary type, just use Utf8 for now
            # (Polars doesn't have a good way to represent arbitrary dicts in structs)
            return pl.Utf8
        else:
            # Nested objects - recursively convert
            return _json_schema_to_polars_dtype(field_schema, root_schema)
    else:
        # Default to string for unknown types
        return pl.Utf8


def _parse_json_to_struct(json_str_series: pl.Series, dtype: pl.DataType) -> pl.Series:
    """Parse a JSON string series into a struct series.

    The dtype parameter is critical - it ensures Polars uses the schema derived
    from the Pydantic model rather than inferring from data. This fixes issues
    with Optional fields that may be null in some rows but present in others.
    """
    # Pass the dtype to json_decode to use the Pydantic-derived schema
    # instead of inferring from data. This ensures Optional fields are
    # always included even when null in early rows.
    try:
        return json_str_series.str.json_decode(dtype=dtype)
    except Exception:
        # If parsing fails with schema, try without and cast
        # This handles edge cases where the response doesn't match schema
        try:
            parsed = json_str_series.str.json_decode()
            return parsed.cast(dtype, strict=False)
        except Exception:
            # Last resort: return inferred schema
            return json_str_series.str.json_decode()


# ============================================================================
# Usage & Cost Accounting (issue #76)
# ============================================================================

#: Per-row usage/cost struct dtype returned as the `usage` field when
#: `usage=True` is passed to `inference_async` / `inference_messages`.
#: `cost_usd` is computed in Python (never emitted by the Rust plugin, which
#: only carries token counts + latency); `null` when the (provider, model)
#: pair has no known price (see `polar_llama.pricing`).
USAGE_DTYPE = pl.Struct(
    {
        "input_tokens": pl.Int64,
        "output_tokens": pl.Int64,
        "cached_tokens": pl.Int64,
        "latency_ms": pl.Int64,
        "cost_usd": pl.Float64,
    }
)

#: The envelope's `usage` sub-struct as emitted by the Rust plugin (no
#: `cost_usd` -- that's computed Python-side in `_decode_usage_envelope`).
_ENVELOPE_USAGE_DTYPE = pl.Struct(
    {
        "input_tokens": pl.Int64,
        "output_tokens": pl.Int64,
        "cached_tokens": pl.Int64,
        "latency_ms": pl.Int64,
    }
)


def _decode_usage_envelope(
    json_str_series: pl.Series,
    response_dtype: pl.DataType,
    provider: Optional[str],
    model: Optional[str],
    price_table_override: Optional[Union[Dict[str, Any], str, Path]],
) -> pl.Series:
    """Decode the `usage=True` JSON envelope into `Struct{response, usage}`.

    `usage=True` makes the Rust plugin emit, per row, a JSON envelope string
    `{"response": ..., "usage": {"input_tokens", "output_tokens",
    "cached_tokens", "latency_ms"}}` -- the same round-trip mechanism already
    used for structured output / in-band errors (`create_error_response` ->
    `_parse_json_to_struct`). This decodes that envelope with a single
    `str.json_decode(dtype=...)` call and appends a vectorized `cost_usd`
    computed from the resolved price for (provider, model) -- price
    resolution happens once per call (provider/model are per-call constants
    for a given expression), so the cost math is pure column arithmetic.

    Null input rows decode to a null envelope (`_parse_json_to_struct`
    preserves nulls); the null-ness of the row is explicitly re-propagated to
    the final output (constructing `pl.struct(...)` from all-null fields
    would otherwise produce a non-null struct with null fields, not a null
    struct row).
    """
    envelope_dtype = pl.Struct(
        {"response": response_dtype, "usage": _ENVELOPE_USAGE_DTYPE}
    )
    full_dtype = pl.Struct({"response": response_dtype, "usage": USAGE_DTYPE})

    decoded = _parse_json_to_struct(json_str_series, envelope_dtype)

    resolved_model = pricing.effective_model(provider, model)
    price = pricing.resolve_price(provider, resolved_model, price_table_override)
    if price is None:
        pricing.warn_unknown_model_once(provider, resolved_model)

    df = pl.DataFrame({"__env": decoded})

    env = pl.col("__env")
    usage = env.struct.field("usage")
    input_f = usage.struct.field("input_tokens")
    output_f = usage.struct.field("output_tokens")
    cached_f = usage.struct.field("cached_tokens")
    latency_f = usage.struct.field("latency_ms")

    if price is not None:
        cached_or_zero = cached_f.fill_null(0)
        non_cached = (input_f.fill_null(0) - cached_or_zero).clip(lower_bound=0)
        cost_expr = (
            non_cached * price.input_per_1m
            + cached_or_zero * price.effective_cached_input_per_1m()
            + output_f.fill_null(0) * price.output_per_1m
        ) / 1_000_000.0
        # No usage info at all (provider omitted the block entirely) ->
        # cost stays null rather than reporting a misleading $0.00.
        cost_expr = (
            pl.when(input_f.is_null() & output_f.is_null())
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(cost_expr)
        )
    else:
        cost_expr = pl.lit(None, dtype=pl.Float64)

    built = pl.struct(
        response=env.struct.field("response"),
        usage=pl.struct(
            input_tokens=input_f,
            output_tokens=output_f,
            cached_tokens=cached_f,
            latency_ms=latency_f,
            cost_usd=cost_expr,
        ),
    )

    result_expr = (
        pl.when(env.is_null()).then(pl.lit(None, dtype=full_dtype)).otherwise(built)
    )

    return df.select(result_expr.alias("__out"))["__out"]


# ============================================================================
# Taxonomy-based Tagging
# ============================================================================


def _create_taxonomy_pydantic_model(
    taxonomy: Dict[str, Dict[str, Any]],
) -> Type["BaseModel"]:
    """
    Create a Pydantic model from a taxonomy definition.

    The taxonomy should be a dict with the following structure:
    {
        "field_name": {
            "description": "Description of the field",
            "values": {
                "value1": "Definition of value1",
                "value2": "Definition of value2",
                ...
            }
        },
        ...
    }

    Each field in the output model will be a struct containing:
    - thinking: List[{value, reasoning}] - one reasoning entry per possible value
    - reflection: str - overall reflection on the field analysis
    - value: str - the selected value
    - confidence: float - confidence in the selection (0.0 to 1.0)

    Note: `thinking` is deliberately a List of fixed-key objects rather than a
    `Dict[str, str]` keyed by value name. A dynamic-key map has no fixed
    `properties`, so it cannot satisfy OpenAI's strict-mode JSON schema
    requirements (every object node needs `properties`, `required` covering
    every key, and `additionalProperties: false`) -- see issue #51. Using a
    List of fixed-shape `{value, reasoning}` items sidesteps this entirely,
    regardless of how taxonomy value names are spelled (including names that
    aren't valid Python identifiers, e.g. "high-priority" or "3rd party").
    """
    try:
        from pydantic import BaseModel, Field, create_model
        from typing import List

        # Shared "thinking" item model: one entry per candidate taxonomy
        # value, holding the value name being considered and the reasoning
        # for/against it. Fixed keys (no dynamic map) keep this strict-mode
        # compliant. Defined once and reused across all taxonomy fields to
        # avoid duplicate $defs entries in the generated schema.
        ThinkingItem = create_model(
            "ThinkingItem",
            value=(
                str,
                Field(..., description="The taxonomy value name being considered"),
            ),
            reasoning=(
                str,
                Field(
                    ...,
                    description="Reasoning for why this value does or does not apply",
                ),
            ),
        )

        # Create a field result model for each taxonomy field
        field_models = {}

        for field_name, field_config in taxonomy.items():
            values = field_config.get("values", {})
            value_names = list(values.keys())

            # Create the field result model. `thinking` is a list with one
            # {value, reasoning} entry per possible value -- no dynamic keys.
            field_result_model = create_model(
                f"{field_name.title()}Result",
                thinking=(
                    List[ThinkingItem],
                    Field(
                        ...,
                        description=(
                            f"One entry per possible value ({', '.join(value_names)}), "
                            "each with the candidate value name and your reasoning for/against it"
                        ),
                    ),
                ),
                reflection=(
                    str,
                    Field(
                        ...,
                        description="Overall reflection on your analysis of this field",
                    ),
                ),
                value=(
                    str,
                    Field(
                        ...,
                        description=f"Selected value from: {', '.join(value_names)}",
                    ),
                ),
                confidence=(
                    float,
                    Field(
                        ...,
                        ge=0.0,
                        le=1.0,
                        description="Confidence in the selected value (0.0 to 1.0)",
                    ),
                ),
            )

            field_models[field_name] = (
                field_result_model,
                Field(..., description=field_config.get("description", "")),
            )

        # Create the main taxonomy result model
        TaxonomyResult = create_model("TaxonomyResult", **field_models)

        return TaxonomyResult

    except ImportError:
        raise ImportError(
            "Pydantic is required for taxonomy tagging. Install with: pip install pydantic>=2.0.0"
        )


def _create_taxonomy_prompt(
    taxonomy: Dict[str, Dict[str, Any]], document_field_name: str = "document"
) -> str:
    """
    Create a system prompt for taxonomy-based tagging.

    This prompt instructs the model to analyze a document according to the
    provided taxonomy and return structured tags with reasoning.
    """
    prompt_parts = [
        f"You are an expert document analyst. Analyze the provided {document_field_name} and tag it according to the following taxonomy.",
        "",
        "# Taxonomy Fields",
        "",
    ]

    for field_name, field_config in taxonomy.items():
        description = field_config.get("description", "")
        values = field_config.get("values", {})

        prompt_parts.append(f"## {field_name}")
        if description:
            prompt_parts.append(f"{description}")
        prompt_parts.append("")
        prompt_parts.append("Possible values:")

        for value_name, value_definition in values.items():
            prompt_parts.append(f"- **{value_name}**: {value_definition}")

        prompt_parts.append("")

    prompt_parts.extend(
        [
            "# Instructions",
            "",
            "For each field in the taxonomy:",
            "",
            "1. **Thinking**: For each possible value, add one entry to the `thinking` list containing `value` (the exact value name from the taxonomy) and `reasoning` (why it does or does not apply to the document). Include exactly one entry per possible value.",
            "",
            "2. **Reflection**: After thinking through all values, reflect on your analysis. Consider which value best fits the document and why.",
            "",
            "3. **Value**: Select the single best value from the possible values for this field. Note this is distinct from the per-candidate `value` entries inside `thinking` -- this `value` is your final selection.",
            "",
            "4. **Confidence**: Provide your confidence in this selection as a number between 0.0 (not confident) and 1.0 (very confident).",
            "",
            "Return your analysis in the structured format with all required fields.",
        ]
    )

    return "\n".join(prompt_parts)


def _make_run_pending(
    symbol: str, kwargs: Dict[str, Any]
) -> Callable[[pl.Series], pl.Series]:
    """Build the `run_pending` callback `checkpointed_expr` calls per chunk.

    Runs the *unchanged* Rust plugin expression eagerly over a small,
    freshly-built one-column DataFrame holding just the pending rows for
    this chunk -- the nested `DataFrame(...).select(register_plugin(...))`
    described in the checkpointing design (`polar_llama/checkpoint.py`).
    """

    def run_pending(s: pl.Series) -> pl.Series:
        if len(s) == 0:
            return pl.Series([], dtype=pl.Utf8)
        inner_df = pl.DataFrame({"__ckpt_in": s})
        out = inner_df.select(
            register_plugin(
                args=[pl.col("__ckpt_in")],
                symbol=symbol,
                is_elementwise=True,
                lib=lib,
                kwargs=kwargs,
            ).alias("__ckpt_out")
        )
        return out["__ckpt_out"]

    return run_pending


def inference_async(
    expr: IntoExpr,
    *,
    provider: Optional[Union[str, Provider]] = None,
    model: Optional[str] = None,
    response_model: Optional[Type["BaseModel"]] = None,
    response_format: Optional[Type["BaseModel"]] = None,
    cache: Union[bool, CacheConfig] = False,
    system_prompt: Optional[str] = None,
    checkpoint: Optional[Union[str, Path, Checkpoint]] = None,
    usage: bool = False,
    price_table: Optional[Union[Dict[str, Any], str, Path]] = None,
    dedupe: bool = False,
    response_cache: Optional[Union[str, Path, ResponseCache]] = None,
    dedupe_stats: Optional[DedupeStats] = None,
) -> pl.Expr:
    """
    Asynchronously infer completions for the given text expressions using an LLM.

    Parameters
    ----------
    expr : polars.Expr
        The text expression to use for inference
    provider : str or Provider, optional
        The provider to use (OpenAI, Anthropic, Gemini, Groq, Bedrock)
    model : str, optional
        The model name to use
    response_model : Type[BaseModel], optional
        A Pydantic model class to define structured output schema.
        The LLM response will be validated against this schema.
        Returns a Struct with fields matching the Pydantic model.
    system_prompt : str, optional
        A system prompt to prepend to all messages. Required for caching to work
        effectively with text prompts. When provided with cache=True, the system
        prompt will be cached and reused across all requests, providing ~90% cost
        savings on input tokens (for Anthropic).
    cache : bool or CacheConfig, optional
        Enable cache optimization for batch processing. When True, uses
        automatic cache optimization. Pass a CacheConfig for fine-grained control.
        Default: False (caching disabled).

        When caching is enabled, Polar Llama will:
        1. Detect shared prefixes (system prompts, schemas) across rows
        2. Group rows by shared content for efficient API calls
        3. Add provider-specific cache control markers
        4. Order requests to maximize cache hits

        Example:
            >>> # Simple: enable automatic caching
            >>> df.with_columns(
            ...     response=inference_async(pl.col("messages"), cache=True)
            ... )
            >>>
            >>> # Advanced: configure caching behavior
            >>> from polar_llama import CacheConfig, CacheStrategy
            >>> config = CacheConfig(strategy=CacheStrategy.SYSTEM_PROMPT, ttl="1h")
            >>> df.with_columns(
            ...     response=inference_async(pl.col("messages"), cache=config)
            ... )
    checkpoint : str, Path, or Checkpoint, optional
        Enable resumable batch checkpointing (issue #75). A str/Path is
        shorthand for ``Checkpoint(path)``; pass a `Checkpoint` for
        fine-grained control (``flush_every``, ``retry_failed``,
        ``on_mismatch``). When set, results are persisted to the given
        sidecar directory as they complete, and re-running the same
        expression against the same directory skips rows already computed
        -- so killing a batch partway through and re-running only pays for
        the rows not yet done. A row is considered "the same request" if
        its content (plus every other request-shaping parameter: provider,
        model, system_prompt, response schema) is unchanged; changing any of
        those forces a full recompute. Failed rows are stored too (so a
        crash doesn't lose them) and are retried on resume by default
        (``retry_failed=True``); set ``retry_failed=False`` to keep stored
        errors as-is. Default: None (checkpointing disabled -- identical to
        the pre-#75 code path).

        Example:
            >>> # First run: computes all rows, persisting as it goes.
            >>> # If killed partway through and re-run against the same
            >>> # directory, only the not-yet-completed rows are recomputed.
            >>> df.with_columns(
            ...     response=inference_async(
            ...         pl.col("prompt"), checkpoint="runs/batch1.ckpt"
            ...     )
            ... )
    usage : bool, optional
        Enable per-row usage/cost accounting (issue #76). When True, the
        result is composed with a ``usage`` struct field:
        ``Struct{response: Utf8 (or the response_model struct), usage:
        USAGE_DTYPE}`` where ``USAGE_DTYPE`` is
        ``Struct{input_tokens, output_tokens, cached_tokens, latency_ms:
        Int64, cost_usd: Float64}``. Every usage field is nullable -- a
        provider that omits its usage block (or a parse failure) leaves the
        corresponding field null, never an error. ``cost_usd`` is computed
        from a packaged price table (overridable via ``price_table=``);
        unknown (provider, model) pairs get a null ``cost_usd`` and a
        one-time ``UserWarning``. Default: False (byte-identical to the
        pre-#76 output -- no envelope, no extra column).

        Not currently supported together with ``checkpoint=`` (raises
        ``ValueError``): the checkpoint store's failed-row detection
        (``polar_llama.checkpoint._is_error_row``) inspects the raw stored
        string for a top-level ``{"_error": ...}`` shape, which the usage
        envelope wraps one level deeper (``{"response": {"_error": ...},
        "usage": {...}}``) -- combining the two today would silently stop
        retrying failed rows on resume. Tracked as follow-up work (issue #76
        design doc, checkpoint interop section).

        Aggregation recipes::

            >>> # Total cost for the job
            >>> df["r"].struct.field("usage").struct.field("cost_usd").sum()
            >>> # Cost per 1k rows
            >>> total_cost / (df.height / 1000)
            >>> # Cache hit-rate
            >>> u = df["r"].struct.field("usage")
            >>> u.struct.field("cached_tokens").sum() / u.struct.field("input_tokens").sum()
    price_table : dict, str, or Path, optional
        Per-call override for cost resolution (``usage=True`` only).
        Resolution order: this override > the process-wide registry
        (``polar_llama.register_model_price`` / ``set_price_table``) > the
        packaged default table. Same shape as the packaged
        ``polar_llama/data/prices.json``: ``{"<provider>": {"<model>":
        {"input_per_1m": ..., "output_per_1m": ..., "cached_input_per_1m":
        ...}}}``.
    dedupe : bool, optional
        Enable in-run duplicate collapsing (issue #77): identical rows
        *within one batch* are sent to the backend once, and the single
        result is fanned back out to every row that asked for it. Zero
        extra I/O -- pure in-memory content-hash grouping. Default: False
        (byte-identical to the pre-#77 code path -- every row is sent as
        its own request, even exact duplicates). Ignored (silently
        subsumed, no warning) when ``checkpoint=`` is also set -- the
        checkpoint UDF already collapses duplicates per batch.
    response_cache : str, Path, or ResponseCache, optional
        Enable a persistent, cross-job response cache (issue #77): a
        str/Path is shorthand for ``ResponseCache(path)``; pass a
        ``ResponseCache`` for ``ttl``/``on_mismatch`` control. Identical
        requests -- across different runs, even different processes --
        reuse a prior result instead of re-calling the backend. Implies
        ``dedupe=True`` for free (computing content keys to consult the
        store already does the grouping). Only successful results are ever
        persisted; a failed row always recomputes on the next run. Not
        currently supported together with ``checkpoint=`` or ``usage=True``
        (raises ``ValueError`` -- see below). Default: None (disabled).

        See also ``polar_llama.ResponseCache.clear()`` /
        ``.prune()`` for manual invalidation/compaction.

        Disambiguation -- four different things are all called "cache" in
        this library:

        =================  ================================================
        Kwarg              What it caches
        =================  ================================================
        ``cache=``         Provider prompt-prefix caching (transport-level;
                            changes how a request is *sent*, not what is
                            asked for).
        ``checkpoint=``     Resume the *same* job after a crash.
        ``dedupe=``         Collapse exact-duplicate rows *within one run*.
        ``response_cache=`` Reuse results *across* runs/processes for
                            identical requests.
        =================  ================================================

        These compose freely except where noted above: ``dedupe=``/
        ``response_cache=`` alongside provider ``cache=`` simply means
        fewer unique calls reach the provider, so fewer opportunities for
        its own prompt-cache to hit.
    dedupe_stats : DedupeStats, optional
        Mutable accumulator populated (in place, at collect time) with
        ``rows_total``, ``rows_null``, ``cache_hits``, ``rows_collapsed``,
        and ``calls_made`` when ``dedupe=True`` or ``response_cache=...``
        is set. See ``polar_llama.DedupeStats``. Ignored (never populated)
        when neither is set.

        Example::

            >>> stats = DedupeStats()
            >>> out = df.with_columns(
            ...     r=inference_async(pl.col("p"), dedupe=True, dedupe_stats=stats)
            ... )
            >>> stats.hit_rate, stats.rows_collapsed, stats.calls_made  # doctest: +SKIP

    See also ``polar_llama.build_manifest`` (issue #85) for a deterministic,
    shareable audit record of a run's configuration -- reuses this
    function's own request fingerprint, so it's guaranteed consistent with
    what was actually sent.

    Returns
    -------
    polars.Expr
        Expression with inferred completions as a Struct (if response_model
        or usage=True provided) or String (otherwise)
    """
    expr = parse_into_expr(expr)

    if checkpoint is not None and usage:
        raise ValueError(
            "inference_async: usage=True is not currently supported together "
            "with checkpoint=... (the checkpoint store's failed-row detection "
            "does not look inside the usage envelope -- see the `usage` "
            "parameter docstring). Use one or the other for now."
        )

    if response_cache is not None and checkpoint is not None:
        raise ValueError(
            "inference_async: response_cache=... cannot be combined with "
            "checkpoint=... -- two persistent stores for one expression. "
            "Point checkpoint= at a directory for same-job resume, "
            "response_cache= at a directory for cross-job reuse; the "
            "checkpoint path already collapses in-batch duplicates on its "
            "own, so composing the two is not needed."
        )

    if response_cache is not None and usage:
        raise ValueError(
            "inference_async: response_cache=... is not currently supported "
            "together with usage=True (the same envelope problem as "
            "checkpoint=... + usage=True -- see the `usage` parameter "
            "docstring: a failed row's error is nested one level deeper "
            "under the usage envelope, so it would be misclassified as ok "
            "and cached; a served row's cached usage would also misreport "
            "spend). Use one or the other for now."
        )

    # Convert Provider to string to make it picklable. All keys are always
    # present (None values deserialize to Option::None) because polars
    # serializes an empty kwargs dict to empty bytes, which the plugin
    # cannot parse.
    if provider is not None and isinstance(provider, Provider):
        provider = str(provider)
    kwargs = {
        "provider": provider,
        "model": model,
        "response_schema": None,
        "response_model_name": None,
        "usage": bool(usage),
    }

    # Handle response_format alias
    if response_model is None and response_format is not None:
        response_model = response_format

    struct_dtype = None
    if response_model is not None:
        _validate_strict_mode_schema(response_model)
        schema = _pydantic_to_json_schema(response_model)
        # Pass the JSON schema as a JSON string to Rust
        kwargs["response_schema"] = json.dumps(schema)
        kwargs["response_model_name"] = response_model.__name__
        # Create the target struct dtype for later conversion
        struct_dtype = _json_schema_to_polars_dtype(schema)

    # Handle cache configuration
    if cache is True:
        kwargs["cache"] = True
        # Use defaults: strategy=AUTO, min_tokens=1024, ttl="5m"
    elif isinstance(cache, CacheConfig):
        kwargs.update(cache.to_kwargs())
    else:
        kwargs["cache"] = False

    # Pass system_prompt for caching support
    if system_prompt is not None:
        kwargs["system_prompt"] = system_prompt

    if checkpoint is not None:
        if not isinstance(checkpoint, Checkpoint):
            checkpoint = Checkpoint(checkpoint)
        fingerprint, fingerprint_inputs = _request_fingerprint(
            "inference_async", kwargs, system_prompt
        )
        result_expr = checkpointed_expr(
            expr,
            run_pending=_make_run_pending("inference_async", kwargs),
            fingerprint=fingerprint,
            checkpoint=checkpoint,
            fingerprint_inputs=fingerprint_inputs,
            has_schema=kwargs["response_schema"] is not None,
        )
    elif dedupe or response_cache is not None:
        fingerprint, fingerprint_inputs = _request_fingerprint(
            "inference_async", kwargs, system_prompt
        )
        store = None
        ttl_seconds = None
        if response_cache is not None:
            rc = (
                response_cache
                if isinstance(response_cache, ResponseCache)
                else ResponseCache(response_cache)
            )
            store = ResponseCacheStore(
                rc.path,
                fingerprint,
                rc.on_mismatch,
                fingerprint_inputs=fingerprint_inputs,
            )
            ttl_seconds = _parse_ttl(rc.ttl)
        result_expr = deduped_expr(
            expr,
            run_pending=_make_run_pending("inference_async", kwargs),
            fingerprint=fingerprint,
            store=store,
            ttl_seconds=ttl_seconds,
            stats=dedupe_stats,
            has_schema=kwargs["response_schema"] is not None,
        )
    else:
        result_expr = register_plugin(
            args=[expr],
            symbol="inference_async",
            is_elementwise=True,
            lib=lib,
            kwargs=kwargs,
        )

    if usage:
        # usage=True subsumes the response_model struct decode: a single
        # json_decode of the composed envelope dtype handles both.
        response_dtype = struct_dtype if struct_dtype is not None else pl.Utf8
        result_expr = result_expr.map_batches(
            lambda s: _decode_usage_envelope(
                s, response_dtype, provider, model, price_table
            ),
            return_dtype=pl.Struct({"response": response_dtype, "usage": USAGE_DTYPE}),
        )
    elif struct_dtype is not None:
        # If response_model was provided (and usage=False), convert JSON
        # strings to structs.
        result_expr = result_expr.map_batches(
            lambda s: _parse_json_to_struct(s, struct_dtype), return_dtype=struct_dtype
        )

    return result_expr


def inference(
    expr: IntoExpr,
    *,
    provider: Optional[Union[str, Provider]] = None,
    model: Optional[str] = None,
    response_model: Optional[Type["BaseModel"]] = None,
    response_format: Optional[Type["BaseModel"]] = None,
) -> pl.Expr:
    """
    Synchronously infer completions for the given text expressions using an LLM.

    .. deprecated::
        This function is deprecated. Use `inference_async` instead for better
        performance and caching support.

    Parameters
    ----------
    expr : polars.Expr
        The text expression to use for inference
    provider : str or Provider, optional
        The provider to use (OpenAI, Anthropic, Gemini, Groq, Bedrock)
    model : str, optional
        The model name to use
    response_model : Type[BaseModel], optional
        A Pydantic model class to define structured output schema.
        The LLM response will be validated against this schema.
        Returns a Struct with fields matching the Pydantic model.
    response_format : Type[BaseModel], optional
        Alias for response_model.

    Returns
    -------
    polars.Expr
        Expression with inferred completions as a Struct (if response_model provided)
        or String (if no response_model)
    """
    import warnings

    warnings.warn(
        "inference() is deprecated and will be removed in a future version. "
        "Use inference_async() instead for better performance and caching support.",
        DeprecationWarning,
        stacklevel=2,
    )
    expr = parse_into_expr(expr)

    # Convert Provider to string to make it picklable. All keys are always
    # present (None values deserialize to Option::None) because polars
    # serializes an empty kwargs dict to empty bytes, which the plugin
    # cannot parse.
    if provider is not None and isinstance(provider, Provider):
        provider = str(provider)
    kwargs = {
        "provider": provider,
        "model": model,
        "response_schema": None,
        "response_model_name": None,
    }

    # Handle response_format alias
    if response_model is None and response_format is not None:
        response_model = response_format

    struct_dtype = None
    if response_model is not None:
        _validate_strict_mode_schema(response_model)
        schema = _pydantic_to_json_schema(response_model)
        # Pass the JSON schema as a JSON string to Rust
        kwargs["response_schema"] = json.dumps(schema)
        kwargs["response_model_name"] = response_model.__name__
        # Create the target struct dtype for later conversion
        struct_dtype = _json_schema_to_polars_dtype(schema)

    result_expr = register_plugin(
        args=[expr],
        symbol="inference",
        is_elementwise=True,
        lib=lib,
        kwargs=kwargs,
    )

    # If response_model was provided, convert JSON strings to structs
    if struct_dtype is not None:
        # Use map_batches to convert the JSON string series to struct series
        result_expr = result_expr.map_batches(
            lambda s: _parse_json_to_struct(s, struct_dtype), return_dtype=struct_dtype
        )

    return result_expr


def inference_messages(
    expr: IntoExpr,
    *,
    provider: Optional[Union[str, Provider]] = None,
    model: Optional[str] = None,
    response_model: Optional[Type["BaseModel"]] = None,
    response_format: Optional[Type["BaseModel"]] = None,
    cache: Union[bool, CacheConfig] = False,
    checkpoint: Optional[Union[str, Path, Checkpoint]] = None,
    usage: bool = False,
    price_table: Optional[Union[Dict[str, Any], str, Path]] = None,
    dedupe: bool = False,
    response_cache: Optional[Union[str, Path, ResponseCache]] = None,
    dedupe_stats: Optional[DedupeStats] = None,
) -> pl.Expr:
    """
    Process message arrays (conversations) for inference using LLMs.

    This function accepts properly formatted JSON message arrays and sends them
    to the LLM for inference while preserving conversation context.

    Parameters
    ----------
    expr : polars.Expr
        The expression containing JSON message arrays
    provider : str or Provider, optional
        The provider to use (OpenAI, Anthropic, Gemini, Groq, Bedrock)
    model : str, optional
        The model name to use
    response_model : Type[BaseModel], optional
        A Pydantic model class to define structured output schema.
        The LLM response will be validated against this schema.
        Returns a Struct with fields matching the Pydantic model.
    response_format : Type[BaseModel], optional
        Alias for response_model.
    cache : bool or CacheConfig, optional
        Enable cache optimization for batch processing. When True, uses
        automatic cache optimization. Pass a CacheConfig for fine-grained control.
        Default: False (caching disabled).

        This is particularly effective for message arrays where many rows share
        the same system prompt. The first request warms the cache, and subsequent
        requests with the same system prompt get a 90% discount on input tokens
        (for Anthropic) or 50% (for OpenAI).

        Example:
            >>> # Enable caching for Anthropic Claude
            >>> df.with_columns(
            ...     response=inference_messages(
            ...         pl.col("messages"),
            ...         provider=Provider.ANTHROPIC,
            ...         model="claude-sonnet-4-20250514",
            ...         cache=True
            ...     )
            ... )
    checkpoint : str, Path, or Checkpoint, optional
        Enable resumable batch checkpointing (issue #75). See
        ``inference_async`` for the full description; behaves identically
        here. Both JSON-string and ``List(Struct{role, content})`` input
        rows are canonicalized to the same content key when their decoded
        conversation content is identical, so the two input shapes share
        checkpoint entries.
    usage : bool, optional
        Enable per-row usage/cost accounting (issue #76). See
        ``inference_async`` for the full description; behaves identically
        here. Not currently supported together with ``checkpoint=`` (raises
        ``ValueError`` -- see the ``inference_async`` docstring for why).
    price_table : dict, str, or Path, optional
        Per-call cost-resolution override (``usage=True`` only). See
        ``inference_async``.
    dedupe : bool, optional
        Enable in-run duplicate collapsing (issue #77). See
        ``inference_async`` for the full description; behaves identically
        here, including sharing content keys across the two accepted input
        shapes (JSON-string vs. ``List(Struct{role, content})``) exactly
        like ``checkpoint=`` does. Default: False.
    response_cache : str, Path, or ResponseCache, optional
        Enable a persistent, cross-job response cache (issue #77). See
        ``inference_async`` for the full description; behaves identically
        here. Not currently supported together with ``checkpoint=`` or
        ``usage=True`` (raises ``ValueError``). Default: None.
    dedupe_stats : DedupeStats, optional
        Mutable accumulator for ``dedupe=``/``response_cache=`` aggregates.
        See ``inference_async``.

    See also ``polar_llama.build_manifest`` (issue #85, symbol=
    ``"inference_messages"``) for a deterministic, shareable audit record
    of a run's configuration.

    Returns
    -------
    polars.Expr
        Expression with inferred completions as a Struct (if response_model
        or usage=True provided) or String (otherwise)
    """
    expr = parse_into_expr(expr)

    if checkpoint is not None and usage:
        raise ValueError(
            "inference_messages: usage=True is not currently supported "
            "together with checkpoint=... (see the `usage` parameter "
            "docstring on inference_async for why). Use one or the other "
            "for now."
        )

    if response_cache is not None and checkpoint is not None:
        raise ValueError(
            "inference_messages: response_cache=... cannot be combined with "
            "checkpoint=... (see the `response_cache` parameter docstring "
            "on inference_async for why). Use one or the other for now."
        )

    if response_cache is not None and usage:
        raise ValueError(
            "inference_messages: response_cache=... is not currently "
            "supported together with usage=True (see the `response_cache` "
            "parameter docstring on inference_async for why). Use one or "
            "the other for now."
        )

    # Both JSON-string input and List(Struct{role, content}) input are handled
    # natively by the Rust `inference_messages` expression, so no Python UDF
    # (map_batches) is needed here — the default path stays lazy/streaming
    # (unless checkpointing or usage=True is requested, both of which need a UDF).

    # Convert Provider to string to make it picklable. All keys are always
    # present (None values deserialize to Option::None) because polars
    # serializes an empty kwargs dict to empty bytes, which the plugin
    # cannot parse.
    if provider is not None and not isinstance(provider, str):
        provider = provider.as_str() if hasattr(provider, "as_str") else str(provider)
    kwargs = {
        "provider": provider,
        "model": model,
        "response_schema": None,
        "response_model_name": None,
        "usage": bool(usage),
    }

    # Handle response_format alias
    if response_model is None and response_format is not None:
        response_model = response_format

    struct_dtype = None
    if response_model is not None:
        _validate_strict_mode_schema(response_model)
        schema = _pydantic_to_json_schema(response_model)
        # Pass the JSON schema as a JSON string to Rust
        kwargs["response_schema"] = json.dumps(schema)
        kwargs["response_model_name"] = response_model.__name__
        # Create the target struct dtype for later conversion
        struct_dtype = _json_schema_to_polars_dtype(schema)

    # Handle cache configuration
    if cache is True:
        kwargs["cache"] = True
        # Use defaults: strategy=AUTO, min_tokens=1024, ttl="5m"
    elif isinstance(cache, CacheConfig):
        kwargs.update(cache.to_kwargs())
    else:
        kwargs["cache"] = False

    if checkpoint is not None:
        if not isinstance(checkpoint, Checkpoint):
            checkpoint = Checkpoint(checkpoint)
        fingerprint, fingerprint_inputs = _request_fingerprint(
            "inference_messages", kwargs
        )
        result_expr = checkpointed_expr(
            expr,
            run_pending=_make_run_pending("inference_messages", kwargs),
            fingerprint=fingerprint,
            checkpoint=checkpoint,
            canonicalize=canonicalize_messages_input,
            fingerprint_inputs=fingerprint_inputs,
            has_schema=kwargs["response_schema"] is not None,
        )
    elif dedupe or response_cache is not None:
        fingerprint, fingerprint_inputs = _request_fingerprint(
            "inference_messages", kwargs
        )
        store = None
        ttl_seconds = None
        if response_cache is not None:
            rc = (
                response_cache
                if isinstance(response_cache, ResponseCache)
                else ResponseCache(response_cache)
            )
            store = ResponseCacheStore(
                rc.path,
                fingerprint,
                rc.on_mismatch,
                fingerprint_inputs=fingerprint_inputs,
            )
            ttl_seconds = _parse_ttl(rc.ttl)
        result_expr = deduped_expr(
            expr,
            run_pending=_make_run_pending("inference_messages", kwargs),
            fingerprint=fingerprint,
            canonicalize=canonicalize_messages_input,
            store=store,
            ttl_seconds=ttl_seconds,
            stats=dedupe_stats,
            has_schema=kwargs["response_schema"] is not None,
        )
    else:
        result_expr = register_plugin(
            args=[expr],
            symbol="inference_messages",
            is_elementwise=True,
            lib=lib,
            kwargs=kwargs,
        )

    if usage:
        # usage=True subsumes the response_model struct decode (single
        # json_decode of the composed envelope dtype).
        response_dtype = struct_dtype if struct_dtype is not None else pl.Utf8
        result_expr = result_expr.map_batches(
            lambda s: _decode_usage_envelope(
                s, response_dtype, provider, model, price_table
            ),
            return_dtype=pl.Struct({"response": response_dtype, "usage": USAGE_DTYPE}),
        )
    elif struct_dtype is not None:
        # If response_model was provided (and usage=False), convert JSON
        # strings to structs.
        result_expr = result_expr.map_batches(
            lambda s: _parse_json_to_struct(s, struct_dtype), return_dtype=struct_dtype
        )

    return result_expr


# ============================================================================
# Streaming Inference
# ============================================================================

#: DataFrame-only streaming output dtype: a single self-describing struct
#: column so the DataFrame stays rectangular even after cancellation.
#: `finished=True` means a terminator was seen ("[DONE]" / `message_stop`, or
#: the buffered-fallback path used by providers with no native SSE support).
#: `finished=False` means the stream was cut off (callback raised, Ctrl-C, a
#: mid-stream provider `error` event, a transport drop, or EOF without a
#: terminator) -- `text` holds whatever partial content arrived, possibly "".
STREAM_RESPONSE_DTYPE = pl.Struct({"text": pl.Utf8, "finished": pl.Boolean})


def inference_stream(
    expr: IntoExpr,
    *,
    provider: Optional[Union[str, Provider]] = None,
    model: Optional[str] = None,
    on_token: Optional[Callable[[int, str], None]] = None,
    messages: bool = False,
    response_model: Optional[Type["BaseModel"]] = None,
    response_format: Optional[Type["BaseModel"]] = None,
) -> pl.Expr:
    """
    Stream completions token-by-token for the given text expressions.

    Unlike ``inference_async``, this is a text-only, DataFrame-oriented
    streaming API: there is no Python iterator. Instead, each row's stream is
    driven to completion (or cancellation) internally, with an optional
    ``on_token`` callback invoked for every delta as it arrives, and the final
    per-row result returned as a ``Struct{text: Utf8, finished: Boolean}``
    (``STREAM_RESPONSE_DTYPE``) column -- so the DataFrame stays rectangular
    and consistent even when a row's stream is cut off.

    Parameters
    ----------
    expr : polars.Expr
        The text expression to use for inference (or JSON message arrays,
        with ``messages=True``).
    provider : str or Provider, optional
        The provider to use (OpenAI, Anthropic, Gemini, Groq, Bedrock).
        Gemini and Bedrock do not have a native streaming transport wired up
        here; they fall back to one full-text delta followed by completion.
    model : str, optional
        The model name to use.
    on_token : callable, optional
        ``on_token(row_index, delta)``, called for every text delta as it
        arrives. ``row_index`` is the index *within the batch this pyfunction
        call executes*, which equals the column index under the default
        ``collect()`` engine, but may restart from 0 per batch under the
        streaming/new-streaming engine. If the callback raises, that row's
        stream is cancelled (see Cancellation below); no exception ever
        propagates out of the expression.
    messages : bool, optional
        When True, each row is a JSON-encoded message array (as produced by
        ``string_to_message`` / ``combine_messages``) instead of a bare user
        message string.
    response_model, response_format : optional
        Not supported by streaming. Passing either raises ``ValueError``
        immediately -- use ``inference_async`` for structured output.

    Notes
    -----
    Usage/cost accounting (issue #76, ``inference_async``'s ``usage=``
    parameter) is explicitly OUT OF SCOPE for streaming -- there is no
    ``usage`` parameter here and ``STREAM_RESPONSE_DTYPE`` is unchanged.
    Future work could extend ``StreamEvent``/``streaming.rs`` to surface
    OpenAI's ``stream_options.include_usage`` final SSE chunk and
    Anthropic's ``message_delta.usage`` event, but that is a separate change.

    Returns
    -------
    polars.Expr
        Expression of dtype ``STREAM_RESPONSE_DTYPE``
        (``Struct{text: Utf8, finished: Boolean}``). A null input row produces
        a null struct row.

    Cancellation
    ------------
    If the ``on_token`` callback raises, or the user presses Ctrl-C (only
    detected on the main Python thread), in-flight streams are aborted and the
    call returns normally with a ``RuntimeWarning``: unfinished rows come back
    with whatever partial text had arrived and ``finished=False``. The
    DataFrame's height and schema are always intact -- streaming never raises
    the callback's exception or a ``KeyboardInterrupt`` out of the expression.

    Examples
    --------
    >>> import polars as pl
    >>> from polar_llama import inference_stream
    >>>
    >>> df = pl.DataFrame({"prompt": ["Tell me a joke", "Say hi"]})
    >>> result = df.with_columns(
    ...     response=inference_stream(
    ...         pl.col("prompt"),
    ...         on_token=lambda i, d: print(d, end=""),
    ...     )
    ... )
    >>> result.select(
    ...     pl.col("response").struct.field("text"),
    ...     pl.col("response").struct.field("finished"),
    ... )
    """
    if response_model is not None or response_format is not None:
        raise ValueError(
            "inference_stream is text-only: response_model is not supported; "
            "use inference_async for structured output"
        )

    if _stream_inference_batch is None:
        raise ImportError(
            "_stream_inference_batch could not be imported from the polar_llama "
            "native extension; inference_stream is unavailable. Did the build "
            "succeed?"
        )

    expr = parse_into_expr(expr)

    if provider is not None and not isinstance(provider, str):
        provider_str = (
            provider.as_str() if hasattr(provider, "as_str") else str(provider)
        )
    else:
        provider_str = provider

    def _udf(s: pl.Series) -> pl.Series:
        texts, finished = _stream_inference_batch(
            s.to_list(), provider_str, model, on_token, messages
        )
        return pl.Series(
            s.name,
            [
                None if t is None else {"text": t, "finished": f}
                for t, f in zip(texts, finished)
            ],
            dtype=STREAM_RESPONSE_DTYPE,
        )

    return expr.map_batches(_udf, return_dtype=STREAM_RESPONSE_DTYPE)


def string_to_message(expr: IntoExpr, *, message_type: str) -> pl.Expr:
    """
    Convert a string to a message with the specified type.

    Parameters
    ----------
    expr : polars.Expr
        The text expression to convert
    message_type : str
        The type of message to create ("user", "system", "assistant")

    Returns
    -------
    polars.Expr
        Expression with formatted messages
    """
    expr = parse_into_expr(expr)
    return register_plugin(
        args=[expr],
        symbol="string_to_message",
        is_elementwise=True,
        lib=lib,
        kwargs={"message_type": message_type},
    )


def combine_messages(*exprs: IntoExpr) -> pl.Expr:
    """
    Combine multiple message expressions into a single message array.

    This function takes multiple message expressions (strings containing JSON formatted messages)
    and combines them into a single JSON array of messages, preserving the order.

    Parameters
    ----------
    *exprs : polars.Expr
        One or more expressions containing messages to combine

    Returns
    -------
    polars.Expr
        Expression with combined message arrays
    """
    args = [parse_into_expr(expr) for expr in exprs]

    return register_plugin(
        args=args,
        symbol="combine_messages",
        is_elementwise=True,
        lib=lib,
    )


def embedding_async(
    expr: IntoExpr,
    *,
    provider: Optional[Union[str, Provider]] = None,
    model: Optional[str] = None,
) -> pl.Expr:
    """
    Asynchronously generate embeddings for the given text expressions.

    This function generates vector embeddings for text using various embedding providers.
    The embeddings are computed in parallel for maximum performance and memory efficiency.

    Parameters
    ----------
    expr : polars.Expr
        The text expression to generate embeddings for
    provider : str or Provider, optional
        The provider to use (OpenAI, Gemini, Bedrock). Default: OpenAI
    model : str, optional
        The embedding model name to use. If not specified, uses the default
        model for the provider:
        - OpenAI: "text-embedding-3-small" (1536 dimensions)
        - Gemini: "text-embedding-004" (768 dimensions)
        - Bedrock: "amazon.titan-embed-text-v1" (1536 dimensions)

    Returns
    -------
    polars.Expr
        Expression with embeddings as List[Float64] (vector of floats)

    Examples
    --------
    >>> import polars as pl
    >>> from polar_llama import embedding_async, Provider
    >>>
    >>> # Create a dataframe with text
    >>> df = pl.DataFrame({
    ...     "text": ["Hello world", "Machine learning is fun"]
    ... })
    >>>
    >>> # Generate embeddings using OpenAI (default)
    >>> result = df.with_columns(
    ...     embeddings=embedding_async(pl.col("text"))
    ... )
    >>>
    >>> # Use a specific provider and model
    >>> result = df.with_columns(
    ...     embeddings=embedding_async(
    ...         pl.col("text"),
    ...         provider=Provider.OPENAI,
    ...         model="text-embedding-3-large"
    ...     )
    ... )
    >>>
    >>> # Access the embedding dimensions
    >>> result.select([
    ...     "text",
    ...     pl.col("embeddings").list.len().alias("dimensions")
    ... ])
    """
    expr = parse_into_expr(expr)

    # Convert Provider to string to make it picklable; keep all keys present
    # so the serialized kwargs are never empty.
    if provider is not None and isinstance(provider, Provider):
        provider = str(provider)
    kwargs = {"provider": provider, "model": model}

    return register_plugin(
        args=[expr],
        symbol="embedding_async",
        is_elementwise=True,
        lib=lib,
        kwargs=kwargs,
    )


def embedding_local(
    expr: IntoExpr,
    *,
    model: str = "mlx-community/bge-small-en-v1.5-bf16",
    engine: str = "in_process",
    batch_size: int = 64,
    normalize: bool = True,
) -> pl.Expr:
    """
    Generate embeddings for the given text expression using a local (offline)
    model, instead of calling a hosted provider API.

    The fully-offline counterpart of :func:`embedding_async` (issue #83):
    runs an embedding model in this process via ``mlx_embeddings``. Output
    dtype is ``List[Float64]``, identical to ``embedding_async`` -- a column
    produced by either function is a drop-in input to ``cosine_similarity``,
    ``knn_hnsw``, ``HnswIndex``, and ``cluster_embeddings``.

    Requires the optional ``[local]`` extra on Apple silicon
    (``pip install polar-llama[local]``); importing ``polar_llama`` (or even
    calling this function with ``engine="fake"`` /
    ``POLAR_LLAMA_LOCAL_ENGINE=fake``) never requires ``mlx``/``mlx_embeddings``
    to be installed. See docs/LOCAL_EMBEDDINGS.md.

    Parameters
    ----------
    expr : polars.Expr
        The text expression to generate embeddings for.
    model : str, optional
        An ``mlx-community`` (or other mlx-embeddings-compatible) model repo
        id. Defaults to a small BGE conversion (384 dims); downloads once
        from the HF Hub on first use and is cached under
        ``~/.cache/huggingface``.
    engine : str, optional
        ``"in_process"`` (default, alias ``"mlx"``) uses the real
        ``mlx_embeddings`` model. ``"fake"`` forces the dependency-free
        deterministic engine (also reachable via the
        ``POLAR_LLAMA_LOCAL_ENGINE=fake`` environment override).
    batch_size : int, optional
        Number of texts handed to the model per call; larger columns are
        processed in row-aligned chunks of this size (default 64).
    normalize : bool, optional
        L2-normalize each output vector (default ``True``).

    Returns
    -------
    polars.Expr
        Expression with embeddings as ``List[Float64]`` (vector of floats),
        in the original row order. A null input row (or a per-row engine
        failure) produces a null output row.

    Examples
    --------
    >>> import polars as pl
    >>> from polar_llama import embedding_local
    >>>
    >>> df = pl.DataFrame({
    ...     "text": ["Hello world", "Machine learning is fun"]
    ... })
    >>>
    >>> result = df.with_columns(
    ...     embeddings=embedding_local(pl.col("text"))
    ... )
    """
    # Imported lazily so that a top-level `import polar_llama` never pulls in
    # the local backend (and therefore never risks importing mlx).
    from polar_llama.local.embed import embedding_local as _embedding_local

    return _embedding_local(
        expr,
        model=model,
        engine=engine,
        batch_size=batch_size,
        normalize=normalize,
    )


def cosine_similarity(
    expr1: IntoExpr,
    expr2: IntoExpr,
) -> pl.Expr:
    """
    Calculate cosine similarity between two embedding vectors.

    Cosine similarity measures the cosine of the angle between two vectors,
    ranging from -1 (opposite) to 1 (identical). For normalized embeddings,
    this is equivalent to dot product.

    Parameters
    ----------
    expr1 : polars.Expr
        First embedding vector (List[Float64])
    expr2 : polars.Expr
        Second embedding vector (List[Float64])

    Returns
    -------
    polars.Expr
        Cosine similarity score (Float64)

    Examples
    --------
    >>> import polars as pl
    >>> from polar_llama import embedding_async, cosine_similarity
    >>>
    >>> df = pl.DataFrame({
    ...     "text1": ["Hello world"],
    ...     "text2": ["Hello there"]
    ... })
    >>> result = df.with_columns(
    ...     emb1=embedding_async(pl.col("text1")),
    ...     emb2=embedding_async(pl.col("text2"))
    ... ).with_columns(
    ...     similarity=cosine_similarity(pl.col("emb1"), pl.col("emb2"))
    ... )
    """
    expr1 = parse_into_expr(expr1)
    expr2 = parse_into_expr(expr2)
    return register_plugin(
        args=[expr1, expr2],
        symbol="cosine_similarity",
        is_elementwise=True,
        lib=lib,
    )


def dot_product(
    expr1: IntoExpr,
    expr2: IntoExpr,
) -> pl.Expr:
    """
    Calculate dot product between two embedding vectors.

    The dot product is the sum of element-wise products of two vectors.

    Parameters
    ----------
    expr1 : polars.Expr
        First embedding vector (List[Float64])
    expr2 : polars.Expr
        Second embedding vector (List[Float64])

    Returns
    -------
    polars.Expr
        Dot product value (Float64)

    Examples
    --------
    >>> import polars as pl
    >>> from polar_llama import embedding_async, dot_product
    >>>
    >>> df = pl.DataFrame({
    ...     "text1": ["Hello world"],
    ...     "text2": ["Hello there"]
    ... })
    >>> result = df.with_columns(
    ...     emb1=embedding_async(pl.col("text1")),
    ...     emb2=embedding_async(pl.col("text2"))
    ... ).with_columns(
    ...     dot_prod=dot_product(pl.col("emb1"), pl.col("emb2"))
    ... )
    """
    expr1 = parse_into_expr(expr1)
    expr2 = parse_into_expr(expr2)
    return register_plugin(
        args=[expr1, expr2],
        symbol="dot_product",
        is_elementwise=True,
        lib=lib,
    )


def euclidean_distance(
    expr1: IntoExpr,
    expr2: IntoExpr,
) -> pl.Expr:
    """
    Calculate Euclidean distance between two embedding vectors.

    The Euclidean distance is the straight-line distance between two points
    in n-dimensional space.

    Parameters
    ----------
    expr1 : polars.Expr
        First embedding vector (List[Float64])
    expr2 : polars.Expr
        Second embedding vector (List[Float64])

    Returns
    -------
    polars.Expr
        Euclidean distance (Float64)

    Examples
    --------
    >>> import polars as pl
    >>> from polar_llama import embedding_async, euclidean_distance
    >>>
    >>> df = pl.DataFrame({
    ...     "text1": ["Hello world"],
    ...     "text2": ["Hello there"]
    ... })
    >>> result = df.with_columns(
    ...     emb1=embedding_async(pl.col("text1")),
    ...     emb2=embedding_async(pl.col("text2"))
    ... ).with_columns(
    ...     distance=euclidean_distance(pl.col("emb1"), pl.col("emb2"))
    ... )
    """
    expr1 = parse_into_expr(expr1)
    expr2 = parse_into_expr(expr2)
    return register_plugin(
        args=[expr1, expr2],
        symbol="euclidean_distance",
        is_elementwise=True,
        lib=lib,
    )


def knn_hnsw(
    query_expr: IntoExpr,
    reference_expr: IntoExpr,
    *,
    k: int = 5,
) -> pl.Expr:
    """
    Find k-nearest neighbors using Hierarchical Navigable Small World (HNSW) algorithm.

    This function builds an HNSW index from the reference embeddings and searches
    for the k nearest neighbors for each query embedding. HNSW provides approximate
    nearest neighbor search that is much faster than exact search for large datasets.

    Parameters
    ----------
    query_expr : polars.Expr
        Query embedding vectors (List[Float64])
    reference_expr : polars.Expr
        Reference embedding vectors to search through (List[Float64])
    k : int, optional
        Number of nearest neighbors to return (default: 5)

    Returns
    -------
    polars.Expr
        List of indices (List[Int64]) of the k nearest neighbors

    Examples
    --------
    >>> import polars as pl
    >>> from polar_llama import embedding_async, knn_hnsw
    >>>
    >>> # Create a corpus and query
    >>> corpus_df = pl.DataFrame({
    ...     "id": [1, 2, 3, 4],
    ...     "text": [
    ...         "Machine learning is fun",
    ...         "Deep learning uses neural networks",
    ...         "Python is a programming language",
    ...         "Data science involves statistics"
    ...     ]
    ... }).with_columns(
    ...     embeddings=embedding_async(pl.col("text"))
    ... )
    >>>
    >>> # Find similar documents
    >>> query = pl.DataFrame({
    ...     "query": ["neural networks"]
    ... }).with_columns(
    ...     query_emb=embedding_async(pl.col("query"))
    ... )
    >>>
    >>> # Cross join and find neighbors
    >>> result = query.join(
    ...     corpus_df.select([pl.col("embeddings").alias("corpus_emb")]),
    ...     how="cross"
    ... ).with_columns(
    ...     neighbors=knn_hnsw(
    ...         pl.col("query_emb"),
    ...         pl.col("corpus_emb"),
    ...         k=3
    ...     )
    ... )
    """
    query_expr = parse_into_expr(query_expr)
    reference_expr = parse_into_expr(reference_expr)
    return register_plugin(
        args=[query_expr, reference_expr],
        symbol="knn_hnsw",
        is_elementwise=True,
        lib=lib,
        kwargs={"k": k},
    )


def tag_taxonomy(
    expr: IntoExpr,
    taxonomy: Dict[str, Dict[str, Any]],
    *,
    provider: Optional[Union[str, Provider]] = None,
    model: Optional[str] = None,
) -> pl.Expr:
    """
    Tag documents according to a taxonomy definition with detailed reasoning.

    This function analyzes documents and assigns tags based on a predefined taxonomy,
    providing detailed reasoning for each classification decision. The taxonomy allows
    you to define fields (categories), their possible values, and definitions for each value.

    For each taxonomy field, the model will:
    1. Think through each possible value with reasoning
    2. Reflect on the overall analysis
    3. Select the best value
    4. Provide a confidence score

    Parameters
    ----------
    expr : polars.Expr
        The document expression to analyze and tag
    taxonomy : Dict[str, Dict[str, Any]]
        A dictionary defining the taxonomy structure:
        {
            "field_name": {
                "description": "Description of what this field represents",
                "values": {
                    "value1": "Definition of value1",
                    "value2": "Definition of value2",
                    ...
                }
            },
            ...
        }
    provider : str or Provider, optional
        The LLM provider to use (OpenAI, Anthropic, Gemini, Groq, Bedrock)
    model : str, optional
        The specific model name to use

    Returns
    -------
    polars.Expr
        Expression with structured tags as a Struct. Each taxonomy field becomes
        a nested struct containing:
        - thinking: List[{value, reasoning}] - one entry per possible value
        - reflection: str - overall reflection on the field analysis
        - value: str - the selected value
        - confidence: float - confidence score (0.0 to 1.0)

    Examples
    --------
    >>> import polars as pl
    >>> from polar_llama import tag_taxonomy, Provider
    >>>
    >>> # Define a taxonomy
    >>> taxonomy = {
    ...     "sentiment": {
    ...         "description": "The emotional tone of the text",
    ...         "values": {
    ...             "positive": "Text expresses positive emotions, optimism, or favorable opinions",
    ...             "negative": "Text expresses negative emotions, pessimism, or unfavorable opinions",
    ...             "neutral": "Text is factual and objective without clear emotional content"
    ...         }
    ...     },
    ...     "urgency": {
    ...         "description": "How urgent or time-sensitive the content is",
    ...         "values": {
    ...             "high": "Requires immediate attention or action",
    ...             "medium": "Should be addressed soon but not immediately critical",
    ...             "low": "Can be addressed at any convenient time"
    ...         }
    ...     }
    ... }
    >>>
    >>> # Create a dataframe with documents
    >>> df = pl.DataFrame({
    ...     "id": [1, 2],
    ...     "document": [
    ...         "URGENT: The server is down and customers can't access the site!",
    ...         "Our quarterly results exceeded expectations. Great work team!"
    ...     ]
    ... })
    >>>
    >>> # Apply taxonomy tagging
    >>> result = df.with_columns(
    ...     tags=tag_taxonomy(
    ...         pl.col("document"),
    ...         taxonomy,
    ...         provider=Provider.OPENAI,
    ...         model="gpt-4"
    ...     )
    ... )
    >>>
    >>> # Access specific fields and values
    >>> result.select([
    ...     "document",
    ...     pl.col("tags").struct.field("sentiment").struct.field("value").alias("sentiment"),
    ...     pl.col("tags").struct.field("sentiment").struct.field("confidence").alias("sentiment_conf"),
    ...     pl.col("tags").struct.field("urgency").struct.field("value").alias("urgency")
    ... ])
    """
    # Create the Pydantic model from the taxonomy
    response_model = _create_taxonomy_pydantic_model(taxonomy)

    # Create the system prompt with taxonomy instructions
    system_prompt = _create_taxonomy_prompt(taxonomy, document_field_name="document")

    # Parse the document expression
    doc_expr = parse_into_expr(expr)

    # Create a system message for each row by mapping over the document column
    # This ensures the system message is broadcast to match the number of rows
    system_message_expr = doc_expr.map_batches(
        lambda s: pl.Series([system_prompt] * len(s)), return_dtype=pl.Utf8
    ).pipe(string_to_message, message_type="system")

    # Create a user message with the document
    user_message_expr = doc_expr.pipe(string_to_message, message_type="user")

    # Combine the messages
    messages_expr = combine_messages(system_message_expr, user_message_expr)

    # Call inference_messages with the structured output model
    return inference_messages(
        messages_expr, provider=provider, model=model, response_model=response_model
    )


# ============================================================================
# Codebook Induction (issue #78) — see docs/CODEBOOK_INDUCTION.md
# ============================================================================

from polar_llama.codebook import (
    CLUSTER_STRUCT_DTYPE,
    Codebook,
    CodebookEntry,
    CodebookInductionResult,
    apply_codebook,
    cluster_embeddings,
    codebook_to_taxonomy,
    dedupe_codebook_entries,
    induce_codebook,
)


# ============================================================================
# Inter-rater Reliability (issue #79) — see docs/RELIABILITY_METRICS.md
# ============================================================================

from polar_llama.reliability import (
    cohens_kappa,
    krippendorffs_alpha,
)


# ============================================================================
# Survey Data-Quality Flags (issue #80) — see docs/QUALITY_FLAGS.md
# ============================================================================

from polar_llama.quality import (
    AI_DETECTION_LIMITATION,
    QualityConfig,
    QualityReport,
    ai_likelihood,
    duplicate_answer_score,
    gibberish_score,
    quality_report,
    response_length_score,
    speeder_score,
    straightlining_score,
)


# ============================================================================
# Human-in-the-loop Review Loop (issue #81) — see docs/HITL_WORKFLOW.md
# ============================================================================

from polar_llama.hitl import (
    CorrectionResult,
    corrections_to_trainset,
    export_review_sample,
    import_corrections,
    retune_from_corrections,
)


# ============================================================================
# Tool Use (MCP) — see docs/design/MCP_TOOL_INTEGRATION.md
# ============================================================================

from polar_llama.tools import (
    TOOL_RESULT_DTYPE,
    execute_tool_calls,
    mcp_tools,
    tool_results_to_message,
    tools_to_response_model,
)


# ============================================================================
# Persistent HNSW Index (issue #82) — see docs/VECTOR_SIMILARITY_AND_ANN.md
# ============================================================================

from polar_llama.index import (
    NEIGHBOR_STRUCT_DTYPE,
    HnswIndex,
)


# ============================================================================
# TypeSafe System One — typed evaluation (https://docs.typesafe.ai/api)
# ============================================================================

from polar_llama.typesafe import (
    DEFAULT_TYPESAFE_MODEL,
    choice,
    choice_field,
    contract_questions,
    noul,
    score,
    score_field,
    typesafe_eval,
    typesafe_eval_each,
)
from polar_llama.typesafe import list_models as typesafe_models


# ============================================================================
# Playbooks — business rules evaluated across many rows — see docs/PLAYBOOKS.md
# ============================================================================

from polar_llama.playbook import (
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
# Deterministic Run Manifests (issue #85) — see docs/RUN_MANIFESTS.md
# ============================================================================

from polar_llama.manifest import (
    MANIFEST_FORMAT_VERSION,
    ManifestIntegrityError,
    ManifestMismatchError,
    RunManifest,
    build_manifest,
    load_manifest,
    replay,
    save_manifest,
    with_manifest_id,
)


# ============================================================================
# Polars Namespace Accessor
# ============================================================================


@pl.api.register_expr_namespace("llama")
class LlamaNamespace:
    """
    Polars namespace accessor for polar-llama functionality.
    Allows using `.llama` on expressions for a fluent API.
    """

    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def to_message(self, *, role: str = "user") -> pl.Expr:
        """
        Convert a string expression to a message with the specified role.

        Parameters
        ----------
        role : str
            The role of the message ("user", "system", "assistant")

        Returns
        -------
        polars.Expr
            Expression with formatted messages
        """
        return string_to_message(self._expr, message_type=role)

    def inference(
        self,
        *,
        provider: Optional[Union[str, Provider]] = None,
        model: Optional[str] = None,
        response_model: Optional[Type["BaseModel"]] = None,
        response_format: Optional[Type["BaseModel"]] = None,
    ) -> pl.Expr:
        """
        Synchronously infer completions for the expression using an LLM.

        Parameters
        ----------
        provider : str or Provider, optional
            The provider to use
        model : str, optional
            The model name to use
        response_model : Type[BaseModel], optional
            Pydantic model for structured output
        response_format : Type[BaseModel], optional
            Alias for response_model

        Returns
        -------
        polars.Expr
            Expression with inferred completions
        """
        return inference(
            self._expr,
            provider=provider,
            model=model,
            response_model=response_model or response_format,
        )

    def inference_async(
        self,
        *,
        provider: Optional[Union[str, Provider]] = None,
        model: Optional[str] = None,
        response_model: Optional[Type["BaseModel"]] = None,
        response_format: Optional[Type["BaseModel"]] = None,
        cache: Union[bool, CacheConfig] = False,
        system_prompt: Optional[str] = None,
        checkpoint: Optional[Union[str, Path, Checkpoint]] = None,
        usage: bool = False,
        price_table: Optional[Union[Dict[str, Any], str, Path]] = None,
        dedupe: bool = False,
        response_cache: Optional[Union[str, Path, ResponseCache]] = None,
        dedupe_stats: Optional[DedupeStats] = None,
    ) -> pl.Expr:
        """
        Asynchronously infer completions for the expression using an LLM.

        Parameters
        ----------
        provider : str or Provider, optional
            The provider to use
        model : str, optional
            The model name to use
        response_model : Type[BaseModel], optional
            Pydantic model for structured output
        response_format : Type[BaseModel], optional
            Alias for response_model
        cache : bool or CacheConfig, optional
            Enable cache optimization for batch processing
        system_prompt : str, optional
            Shared system prompt cached across all rows when cache=True
            (see the functional ``inference_async`` for details).
        checkpoint : str, Path, or Checkpoint, optional
            Enable resumable batch checkpointing (see the functional
            ``inference_async`` for details).
        usage : bool, optional
            Enable per-row usage/cost accounting (see the functional
            ``inference_async`` for details).
        price_table : dict, str, or Path, optional
            Per-call cost-resolution override (``usage=True`` only).
        dedupe : bool, optional
            Enable in-run duplicate collapsing (see the functional
            ``inference_async`` for details).
        response_cache : str, Path, or ResponseCache, optional
            Enable a persistent, cross-job response cache (see the
            functional ``inference_async`` for details).
        dedupe_stats : DedupeStats, optional
            Mutable accumulator for ``dedupe=``/``response_cache=``
            aggregates.

        Returns
        -------
        polars.Expr
            Expression with inferred completions
        """
        return inference_async(
            self._expr,
            provider=provider,
            model=model,
            response_model=response_model or response_format,
            cache=cache,
            system_prompt=system_prompt,
            checkpoint=checkpoint,
            usage=usage,
            price_table=price_table,
            dedupe=dedupe,
            response_cache=response_cache,
            dedupe_stats=dedupe_stats,
        )

    def inference_stream(
        self,
        *,
        provider: Optional[Union[str, Provider]] = None,
        model: Optional[str] = None,
        on_token: Optional[Callable[[int, str], None]] = None,
        messages: bool = False,
    ) -> pl.Expr:
        """
        Stream completions token-by-token for the expression.
        See ``polar_llama.inference_stream``.
        """
        return inference_stream(
            self._expr,
            provider=provider,
            model=model,
            on_token=on_token,
            messages=messages,
        )

    def inference_local(
        self,
        *,
        model: str,
        system: Optional[str] = None,
        engine: str = "server",
        base_url: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        usage: bool = False,
        price_table: Optional[Union[Dict[str, Any], str, Path]] = None,
    ) -> pl.Expr:
        """
        Infer completions for the expression using a local model.

        Runs completions against a model on this machine instead of a remote
        provider API. Completions are returned as String in the original row
        order.

        Parameters
        ----------
        model : str
            Model identifier. An OpenAI-compatible model name for
            ``engine="server"``, or an mlx-lm model path/repo for
            ``engine="in_process"``.
        system : str, optional
            System prompt. Kept as a separate argument so it forms an immutable
            prefix (the basis for prefix caching on the local engine).
        engine : str, optional
            ``"server"`` (default) routes through the existing async fan-out to
            a local OpenAI-compatible endpoint (no Rust changes; low risk).
            ``"in_process"`` uses the in-process mlx engine via a ``map_batches``
            UDF and requires the optional ``[local]`` extra
            (``pip install polar-llama[local]``).
        base_url : str, optional
            Base URL of the local OpenAI-compatible server
            (``engine="server"`` only).
        max_tokens : int, optional
            Maximum number of tokens to generate (default: 512).
        temperature : float, optional
            Sampling temperature (default: 0.0).
        top_p : float, optional
            Nucleus sampling probability (default: 1.0).
        stop : list of str, optional
            Stop sequences that halt generation.
        usage : bool, optional
            Enable per-row usage/cost accounting (issue #76). See
            ``polar_llama.local.expr.inference_local`` for the full
            description -- ``engine="in_process"`` always reports
            ``cost_usd=0.0``; ``engine="server"`` resolves cost from the
            price table (usually null for a local model name unless you
            register one).
        price_table : dict, str, or Path, optional
            Per-call cost-resolution override (``usage=True``,
            ``engine="server"`` only).

        Returns
        -------
        polars.Expr
            Expression with String completions, in the original row order
            (or a usage-struct column when ``usage=True``).
        """
        # Imported lazily so that a top-level ``import polar_llama`` never pulls
        # in the local backend (and therefore never risks importing mlx).
        from polar_llama.local.expr import inference_local as _inference_local

        return _inference_local(
            self._expr,
            model=model,
            system=system,
            engine=engine,
            base_url=base_url,
            usage=usage,
            price_table=price_table,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )

    def tag_taxonomy(
        self,
        taxonomy: Dict[str, Dict[str, Any]],
        *,
        provider: Optional[Union[str, Provider]] = None,
        model: Optional[str] = None,
    ) -> pl.Expr:
        """
        Tag documents according to a taxonomy definition.
        """
        return tag_taxonomy(self._expr, taxonomy, provider=provider, model=model)

    def typesafe_eval(
        self,
        *extra_state: "IntoExpr",
        questions: Dict[str, Any],
        model: Optional[str] = None,
        probabilities: bool = False,
        usage: bool = False,
        state_json: bool = False,
    ) -> pl.Expr:
        """
        Evaluate this column against typed TypeSafe System One questions.

        Answers come back as a Struct -- `.unnest()` it to get one ordinary
        column per answer. See :func:`polar_llama.typesafe_eval`.

        Parameters
        ----------
        *extra_state
            Further columns to fold into the state; with any given, the state
            becomes a JSON object keyed by column name instead of this
            column's bare value. A length-1 input broadcasts, so a constant
            can ride along with per-row state.
        questions
            Mapping of question id to a :func:`noul` / :func:`choice` /
            :func:`score` spec. Insertion order fixes output column order.
        model
            System One model. Defaults to ``"jev-latest"``.
        probabilities
            Also emit the full distribution for choice/score questions.
        usage
            Also emit `_model` and per-row token/latency accounting.
        state_json
            Treat string inputs as pre-encoded JSON documents.

        Returns
        -------
        polars.Expr
            A Struct expression, always including a null-on-success `_error`.
        """
        return typesafe_eval(
            self._expr,
            *extra_state,
            questions=questions,
            model=model,
            probabilities=probabilities,
            usage=usage,
            state_json=state_json,
        )

    def typesafe_eval_each(
        self,
        *context: "IntoExpr",
        questions: Optional[Dict[str, Any]] = None,
        contract: Optional[Type["BaseModel"]] = None,
        model: Optional[str] = None,
        probabilities: bool = False,
        usage: bool = False,
        include_segment: bool = True,
        max_questions: int = 200,
    ) -> pl.Expr:
        """
        Apply one contract to every segment of this List column, per row.

        All of a row's segments are evaluated in one request (chunked to
        respect `max_questions`), so N segments do not cost N calls and the
        model sees the neighbouring segments as context. Returns
        `List[Struct{line_id, line, <answers>, _error}]` -- `.explode()` it for
        one row per segment. See :func:`polar_llama.typesafe_eval_each`.
        """
        return typesafe_eval_each(
            self._expr,
            *context,
            questions=questions,
            contract=contract,
            model=model,
            probabilities=probabilities,
            usage=usage,
            include_segment=include_segment,
            max_questions=max_questions,
        )

    def embedding(
        self,
        *,
        provider: Optional[Union[str, Provider]] = None,
        model: Optional[str] = None,
    ) -> pl.Expr:
        """
        Generate embeddings for the expression using an embedding model.

        Parameters
        ----------
        provider : str or Provider, optional
            The provider to use
        model : str, optional
            The model name to use

        Returns
        -------
        polars.Expr
            Expression with embeddings as List[Float64]
        """
        return embedding_async(self._expr, provider=provider, model=model)

    def embedding_local(
        self,
        *,
        model: str = "mlx-community/bge-small-en-v1.5-bf16",
        engine: str = "in_process",
        batch_size: int = 64,
        normalize: bool = True,
    ) -> pl.Expr:
        """
        Generate embeddings for the expression using a local (offline) model.

        See ``polar_llama.embedding_local`` for the full parameter/return docs.

        Parameters
        ----------
        model : str, optional
            An mlx-embeddings-compatible model repo id.
        engine : str, optional
            ``"in_process"`` (default) uses the real ``mlx_embeddings``
            model; ``"fake"`` forces the dependency-free deterministic
            engine.
        batch_size : int, optional
            Number of texts handed to the model per call.
        normalize : bool, optional
            L2-normalize each output vector (default ``True``).

        Returns
        -------
        polars.Expr
            Expression with embeddings as ``List[Float64]``.
        """
        return embedding_local(
            self._expr,
            model=model,
            engine=engine,
            batch_size=batch_size,
            normalize=normalize,
        )

    def cosine_similarity(self, other: IntoExpr) -> pl.Expr:
        """
        Calculate cosine similarity with another embedding vector.

        Parameters
        ----------
        other : polars.Expr
            The other embedding vector to compare with

        Returns
        -------
        polars.Expr
            Cosine similarity score
        """
        return cosine_similarity(self._expr, other)

    def cohens_kappa(
        self,
        other: IntoExpr,
        *,
        weights: Optional[str] = None,
        n_bootstrap: Optional[int] = None,
        ci: float = 0.95,
        seed: int = 0,
    ) -> pl.Expr:
        """
        Cohen's kappa between this column and another column of ratings.

        See `polar_llama.cohens_kappa` for the full parameter/return docs.

        Parameters
        ----------
        other : polars.Expr
            The other column of categorical ratings to compare with.

        Returns
        -------
        polars.Expr
            `Float64` kappa (or `Struct{value, ci_low, ci_high}` when
            `n_bootstrap` is given).
        """
        return cohens_kappa(
            self._expr, other, weights=weights, n_bootstrap=n_bootstrap, ci=ci, seed=seed
        )

    def krippendorffs_alpha(
        self,
        *others: IntoExpr,
        level: str = "nominal",
        n_bootstrap: Optional[int] = None,
        ci: float = 0.95,
        seed: int = 0,
    ) -> pl.Expr:
        """
        Krippendorff's alpha across this column and one or more other
        rater columns.

        See `polar_llama.krippendorffs_alpha` for the full parameter/return
        docs.

        Parameters
        ----------
        *others : polars.Expr
            The other rater columns (at least one, so at least 2 raters
            total).

        Returns
        -------
        polars.Expr
            `Float64` alpha (or `Struct{value, ci_low, ci_high}` when
            `n_bootstrap` is given).
        """
        return krippendorffs_alpha(
            [self._expr, *others], level=level, n_bootstrap=n_bootstrap, ci=ci, seed=seed
        )

    def straightlining_score(
        self,
        others: List[IntoExpr],
        *,
        scale_min: Optional[float] = None,
        scale_max: Optional[float] = None,
        min_answers: int = 3,
    ) -> pl.Expr:
        """
        Straightlining suspicion score across this column and other grid
        (Likert) columns. See `polar_llama.straightlining_score`.
        """
        return straightlining_score(
            [self._expr, *others],
            scale_min=scale_min,
            scale_max=scale_max,
            min_answers=min_answers,
        )

    def gibberish_score(self, *, min_chars: int = 8) -> pl.Expr:
        """
        Keyboard-mash / gibberish suspicion score for this text column.
        See `polar_llama.gibberish_score`.
        """
        return gibberish_score(self._expr, min_chars=min_chars)

    def duplicate_answer_score(
        self, others: List[IntoExpr], *, min_answer_chars: int = 10
    ) -> pl.Expr:
        """
        Cross-answer near-duplicate suspicion score across this column and
        other open-end text columns. See `polar_llama.duplicate_answer_score`.
        """
        return duplicate_answer_score(
            [self._expr, *others], min_answer_chars=min_answer_chars
        )

    def response_length_score(self, *, rz_cap: float = 3.0) -> pl.Expr:
        """
        Response-length outlier suspicion score for this text column. See
        `polar_llama.response_length_score`.
        """
        return response_length_score(self._expr, rz_cap=rz_cap)

    def speeder_score(
        self,
        *,
        percentile: float = 0.05,
        min_duration_seconds: Optional[float] = None,
        median_fraction: Optional[float] = None,
    ) -> pl.Expr:
        """
        Speeder suspicion score for this duration column. See
        `polar_llama.speeder_score`.
        """
        return speeder_score(
            self._expr,
            percentile=percentile,
            min_duration_seconds=min_duration_seconds,
            median_fraction=median_fraction,
        )

    def ai_likelihood(
        self,
        *,
        provider: Optional[Union[str, Provider]] = None,
        model: Optional[str] = None,
    ) -> pl.Expr:
        """
        AI-generated-text likelihood score for this text column (flag, not
        verdict -- see `polar_llama.ai_likelihood` for the mandatory
        limitation discussion).
        """
        return ai_likelihood(self._expr, provider=provider, model=model)

    def dot_product(self, other: IntoExpr) -> pl.Expr:
        """
        Calculate dot product with another embedding vector.

        Parameters
        ----------
        other : polars.Expr
            The other embedding vector to compute dot product with

        Returns
        -------
        polars.Expr
            Dot product value
        """
        return dot_product(self._expr, other)

    def euclidean_distance(self, other: IntoExpr) -> pl.Expr:
        """
        Calculate Euclidean distance to another embedding vector.

        Parameters
        ----------
        other : polars.Expr
            The other embedding vector to measure distance to

        Returns
        -------
        polars.Expr
            Euclidean distance
        """
        return euclidean_distance(self._expr, other)

    def knn_hnsw(self, reference: IntoExpr, *, k: int = 5) -> pl.Expr:
        """
        Find k-nearest neighbors using HNSW algorithm.

        Parameters
        ----------
        reference : polars.Expr
            Reference embeddings to search through
        k : int, optional
            Number of nearest neighbors to return (default: 5)

        Returns
        -------
        polars.Expr
            List of indices of the k nearest neighbors
        """
        return knn_hnsw(self._expr, reference, k=k)

    def cluster_embeddings(
        self,
        *,
        k: Optional[int] = None,
        k_min: int = 2,
        k_max: int = 20,
        max_iter: int = 100,
        n_init: int = 8,
        seed: int = 0,
        silhouette_sample: int = 200,
    ) -> pl.Expr:
        """
        Cluster this embeddings column with k-means. See
        ``polar_llama.cluster_embeddings``.
        """
        return cluster_embeddings(
            self._expr,
            k=k,
            k_min=k_min,
            k_max=k_max,
            max_iter=max_iter,
            n_init=n_init,
            seed=seed,
            silhouette_sample=silhouette_sample,
        )

    def apply_codebook(
        self,
        codebook,
        *,
        provider: Optional[Union[str, Provider]] = None,
        model: Optional[str] = None,
    ) -> pl.Expr:
        """
        Multi-label-apply a codebook to this text column. See
        ``polar_llama.apply_codebook``.
        """
        return apply_codebook(self._expr, codebook, provider=provider, model=model)

    def execute_tool_calls(
        self,
        *,
        transport: Optional[str] = None,
        executor=None,
        tools=None,
        concurrency: int = 32,
        timeout_s: int = 30,
    ) -> pl.Expr:
        """
        Execute a column of emitted tool calls in parallel.
        See ``polar_llama.execute_tool_calls``.
        """
        return execute_tool_calls(
            self._expr,
            transport=transport,
            executor=executor,
            tools=tools,
            concurrency=concurrency,
            timeout_s=timeout_s,
        )

    def tool_results_to_message(self, *, role: str = "user") -> pl.Expr:
        """
        Render a tool-results column as a message for the synthesis turn.
        See ``polar_llama.tool_results_to_message``.
        """
        return tool_results_to_message(self._expr, role=role)


# ============================================================================
# Helper Functions
# ============================================================================


def template(format_string: str, *args: IntoExpr, **kwargs: IntoExpr) -> pl.Expr:
    """
    Helper function for prompt templating that abstracts away Polars version differences.

    Usage:
        polar_llama.template("Hello {}", pl.col("name"))
        polar_llama.template("Hello {name}", name=pl.col("name"))

    Parameters
    ----------
    format_string : str
        The format string
    *args : polars.Expr
        Positional arguments for formatting
    **kwargs : polars.Expr
        Keyword arguments for formatting

    Returns
    -------
    polars.Expr
        Formatted string expression
    """
    # Check if current Polars version supports kwargs in pl.format
    # pl.format with kwargs was introduced in recent versions
    # If we want to be safe, we can try to use it, and if it fails, fallback or error
    # But simpler is to rely on pl.format if available.

    # For now, we'll just wrap pl.format.
    # If the user is on an old version that doesn't support kwargs, they should use positional args
    # or we can try to implement a polyfill if needed.
    # However, the recommendation implies we should abstract it.

    try:
        return pl.format(format_string, *args, **kwargs)
    except TypeError:
        # Fallback for older Polars versions that might not support kwargs
        if kwargs:
            # If kwargs are provided but not supported, we can't easily map them to positional
            # without parsing the format string.
            # But we can try to warn or just let it fail if we can't fix it.
            # Alternatively, we can assume the user knows what they are doing or guide them.
            # But the recommendation says "Abstract Prompt Templating... ensure the library works consistently".
            pass
        return pl.format(format_string, *args)


# ============================================================================
# Prompt Optimization (DSPy-style)
# ============================================================================

# Imported late so optimize.py can lazily import inference helpers from here.
from polar_llama import optimize  # noqa: E402
from polar_llama.optimize import (  # noqa: E402
    BootstrapFewShot,
    Evaluation,
    InputField,
    InstructionOptimizer,
    OutputField,
    Predict,
    Signature,
    evaluate,
)


def _validate_strict_mode_schema(model: Type["BaseModel"]) -> None:
    """
    Validate that a Pydantic model is compatible with OpenAI Strict Mode.
    Specifically checks for default values which are not allowed.
    """
    try:
        from pydantic import BaseModel
        from pydantic.fields import FieldInfo

        if not issubclass(model, BaseModel):
            return

        for name, field in model.model_fields.items():
            # Warn about Dict-typed fields: pydantic emits these as
            # {"type": "object", "additionalProperties": {...}} with no fixed
            # `properties`, which OpenAI strict mode (always requested by the
            # Rust client, see src/model_client/openai.rs) rejects -- this is
            # the root cause of issue #51 for user-supplied response_models.
            annotation = getattr(field, "annotation", None)
            origin = getattr(annotation, "__origin__", None)
            if origin is dict or annotation is dict:
                import warnings

                warnings.warn(
                    f"Field '{name}' in model '{model.__name__}' is a Dict type. "
                    "Dict-typed fields produce JSON schemas with dynamic keys "
                    "(additionalProperties) that are incompatible with OpenAI "
                    "Structured Outputs strict mode. Use a nested model or a "
                    "List of key/value items instead (e.g. List[{value, reasoning}]).",
                    UserWarning,
                )

            # Check if field has a default value
            # In Pydantic v2, field.default is PydanticUndefined if no default
            # field.is_required() is another way to check

            if not field.is_required():
                # It has a default value (or is Optional with default None)
                # OpenAI Strict Mode doesn't support default values for required fields?
                # Actually, OpenAI Strict Mode requires all fields to be required (no optionality in the JSON schema sense of missing keys),
                # but it DOES allow nullable fields if they are part of the schema.
                # However, the issue reported is "rejects fields with default values like currency: str = 'USD'".
                # This means the field is NOT required in Pydantic, so it has a default.
                # OpenAI expects the schema to NOT have defaults if we want strict adherence?
                # Or rather, OpenAI's structured outputs require all fields to be specified in the schema and usually required.

                # Let's warn if we see a default value that is not None (Optional is usually fine if handled correctly, but defaults like "USD" are problematic)
                if (
                    field.default is not None
                    and str(field.default) != "PydanticUndefined"
                ):
                    import warnings

                    warnings.warn(
                        f"Field '{name}' in model '{model.__name__}' has a default value '{field.default}'. "
                        "OpenAI Structured Outputs (Strict Mode) may not support default values. "
                        "Consider removing the default value or handling it explicitly.",
                        UserWarning,
                    )
    except ImportError:
        pass
