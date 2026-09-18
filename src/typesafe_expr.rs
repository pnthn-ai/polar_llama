//! The `typesafe_eval` polars expression: TypeSafe System One evaluation as a
//! DataFrame operation (<https://docs.typesafe.ai/api>).
//!
//! One row in, one HTTP request out, every question answered in that same
//! request, and the typed answers landing as ordinary Polars columns inside a
//! Struct the caller can `.unnest()`. Batching every question into one call is
//! the shape TypeSafe's own "speculative fan-out" guidance recommends, and it
//! is also the only shape that makes sense here -- a per-question expression
//! would re-send the same state once per column.
//!
//! Output layout, in the order the questions were declared:
//!
//! | question type | fields                                                       |
//! |---------------|--------------------------------------------------------------|
//! | `noul`        | `<id>`: Float64 (probability of yes)                          |
//! | `choice`      | `<id>`: String, `<id>_confidence`: Float64                    |
//! | `score`       | `<id>`: Float64, `<id>_confidence`: Float64                   |
//!
//! With `probabilities=True`, each choice/score question also contributes one
//! `<id>_p_<option>` / `<id>_p_<level index>` Float64 column. `_error` is
//! always present and null on success; `usage=True` adds `_model`,
//! `_input_tokens`, `_output_tokens` and `_latency_ms`.
//!
//! Noul answers carry no `confidence` -- that is TypeSafe's design, not an
//! omission here: a single probability already is the distribution.

use polars::prelude::*;
use polars_core::chunked_array::builder::AnonymousOwnedListBuilder;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use serde_json::{Map, Value};
use std::collections::HashSet;

use crate::model_client::typesafe::{
    self, Evaluation, QuestionKind, QuestionSpec, DEFAULT_TYPESAFE_MODEL,
};
use crate::utils::RT;

/// Run a future to completion on the shared runtime, spawning it so several
/// Polars threads can have batches in flight at once (same pattern as
/// `expressions::run_async`).
fn run_async<F, T>(future: F) -> T
where
    F: std::future::Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    let handle = RT.spawn(future);
    RT.block_on(handle).expect("TypeSafe task panicked")
}

#[derive(Debug, Deserialize)]
pub struct TypeSafeKwargs {
    /// JSON *array* of question specs. An array, not an object, because the
    /// declared order fixes the output column order and a JSON object would
    /// lose it (`serde_json::Map` re-sorts keys in this build).
    questions: String,
    #[serde(default)]
    model: Option<String>,
    /// Emit the full probability distribution for choice/score questions.
    #[serde(default)]
    probabilities: Option<bool>,
    /// Emit `_model` and per-row token/latency accounting.
    #[serde(default)]
    usage: Option<bool>,
    /// Treat string inputs as pre-encoded JSON rather than literal text, so a
    /// column of JSON documents becomes structured `state`.
    #[serde(default)]
    state_json: Option<bool>,
}

// ============================================================================
// Column plan -- the single source of truth for both the declared output
// dtype and the values written into it. Deriving the schema and the data
// from one plan is what makes it impossible for them to drift apart.
// ============================================================================

#[derive(Debug, Clone)]
enum Col {
    /// `answers[q].noul`
    Noul(String),
    /// `answers[q].choice`
    Choice(String),
    /// `answers[q].score`
    Score(String),
    /// `answers[q].confidence`
    Confidence(String),
    /// `answers[q].probabilities[key]`
    Prob(String, String),
    /// Per-line mode only: the segment's 0-based position in the row's list.
    LineId,
    /// Per-line mode only: the segment's own text.
    LineText,
    Error,
    Model,
    InputTokens,
    OutputTokens,
    LatencyMs,
}

impl Col {
    fn dtype(&self) -> DataType {
        match self {
            Col::Noul(_) | Col::Score(_) | Col::Confidence(_) | Col::Prob(_, _) => {
                DataType::Float64
            }
            Col::Choice(_) | Col::Error | Col::Model | Col::LineText => DataType::String,
            Col::LineId | Col::InputTokens | Col::OutputTokens | Col::LatencyMs => DataType::Int64,
        }
    }
}

/// The flat output layout: one `(field name, extraction rule)` per column.
type ColumnPlan = Vec<(String, Col)>;

/// Expand the question specs into the flat column plan.
///
/// Rejects name collisions up front (a question called `x` next to one called
/// `x_confidence`) rather than letting Polars build a Struct with duplicate
/// field names.
fn plan_columns(
    specs: &[QuestionSpec],
    probabilities: bool,
    usage: bool,
) -> PolarsResult<ColumnPlan> {
    let mut cols: ColumnPlan = Vec::new();

    for spec in specs {
        let id = &spec.id;
        match &spec.kind {
            QuestionKind::Noul { .. } => {
                cols.push((id.clone(), Col::Noul(id.clone())));
            }
            QuestionKind::Choice { options, .. } => {
                cols.push((id.clone(), Col::Choice(id.clone())));
                cols.push((format!("{id}_confidence"), Col::Confidence(id.clone())));
                if probabilities {
                    for opt in options {
                        cols.push((
                            format!("{id}_p_{}", opt.name),
                            Col::Prob(id.clone(), opt.name.clone()),
                        ));
                    }
                }
            }
            QuestionKind::Score { levels, .. } => {
                cols.push((id.clone(), Col::Score(id.clone())));
                cols.push((format!("{id}_confidence"), Col::Confidence(id.clone())));
                if probabilities {
                    // TypeSafe keys score probabilities by level index, as
                    // strings ("0", "1", ...), matching the legend.
                    for level in 0..levels.len() {
                        cols.push((
                            format!("{id}_p_{level}"),
                            Col::Prob(id.clone(), level.to_string()),
                        ));
                    }
                }
            }
        }
    }

    cols.push(("_error".to_string(), Col::Error));
    if usage {
        cols.push(("_model".to_string(), Col::Model));
        cols.push(("_input_tokens".to_string(), Col::InputTokens));
        cols.push(("_output_tokens".to_string(), Col::OutputTokens));
        cols.push(("_latency_ms".to_string(), Col::LatencyMs));
    }

    ensure_unique_fields(&cols)?;
    Ok(cols)
}

/// A Struct cannot carry two fields of the same name, so a collision (a
/// question `a` next to a question `a_confidence`) is caught here -- at schema
/// resolution, before any request is billed.
fn ensure_unique_fields(cols: &ColumnPlan) -> PolarsResult<()> {
    let mut seen = HashSet::new();
    for (name, _) in cols {
        if !seen.insert(name.as_str()) {
            polars_bail!(ComputeError: "typesafe_eval: duplicate output field '{}' -- rename the question or option that collides with it", name);
        }
    }
    Ok(())
}

fn plan_from_kwargs(kwargs: &TypeSafeKwargs) -> PolarsResult<(Vec<QuestionSpec>, ColumnPlan)> {
    let specs = typesafe::parse_questions(&kwargs.questions)
        .map_err(|e| polars_err!(ComputeError: "typesafe_eval: {}", e))?;
    let cols = plan_columns(
        &specs,
        kwargs.probabilities.unwrap_or(false),
        kwargs.usage.unwrap_or(false),
    )?;
    Ok((specs, cols))
}

fn typesafe_output_type(_input_fields: &[Field], kwargs: TypeSafeKwargs) -> PolarsResult<Field> {
    let (_, cols) = plan_from_kwargs(&kwargs)?;
    let fields = cols
        .iter()
        .map(|(name, col)| Field::new(PlSmallStr::from_str(name), col.dtype()))
        .collect::<Vec<_>>();
    Ok(Field::new(
        PlSmallStr::from_static(""),
        DataType::Struct(fields),
    ))
}

// ============================================================================
// State assembly
// ============================================================================

/// Convert one scalar cell to JSON. Numbers and booleans keep their type so a
/// state object reads naturally on the model's side; anything exotic (dates,
/// binary) degrades to its string rendering rather than failing the row.
/// Nested dtypes never reach here -- `series_to_json` handles them
/// structurally.
fn any_value_to_json(av: &AnyValue) -> Value {
    match av {
        AnyValue::Null => Value::Null,
        AnyValue::Boolean(b) => Value::Bool(*b),
        AnyValue::String(s) => Value::String((*s).to_string()),
        AnyValue::StringOwned(s) => Value::String(s.to_string()),
        AnyValue::Int8(v) => Value::from(*v),
        AnyValue::Int16(v) => Value::from(*v),
        AnyValue::Int32(v) => Value::from(*v),
        AnyValue::Int64(v) => Value::from(*v),
        AnyValue::UInt8(v) => Value::from(*v),
        AnyValue::UInt16(v) => Value::from(*v),
        AnyValue::UInt32(v) => Value::from(*v),
        AnyValue::UInt64(v) => Value::from(*v),
        AnyValue::Float32(v) => Value::from(*v as f64),
        AnyValue::Float64(v) => Value::from(*v),
        other => Value::String(other.to_string()),
    }
}

/// Convert a whole column to one JSON value per row.
///
/// Nested dtypes are converted structurally and recursively -- a `List` column
/// becomes a JSON array and a `Struct` column a JSON object -- because
/// TypeSafe's `state` explicitly accepts arrays and objects (a list of chat
/// turns, a record). Rendering those through `AnyValue`'s `Display` instead
/// would ship a Polars debug repr to the model and bill for it.
fn series_to_json(s: &Series) -> PolarsResult<Vec<Value>> {
    let n = s.len();
    match s.dtype() {
        DataType::List(_) => {
            let ca = s.list()?;
            let mut out = Vec::with_capacity(n);
            for opt in ca.amortized_iter() {
                match opt {
                    None => out.push(Value::Null),
                    Some(inner) => out.push(Value::Array(series_to_json(inner.as_ref())?)),
                }
            }
            Ok(out)
        }
        DataType::Struct(_) => {
            let ca = s.struct_()?;
            let fields: Vec<(String, Vec<Value>)> = ca
                .fields_as_series()
                .iter()
                .map(|f| Ok((f.name().to_string(), series_to_json(f)?)))
                .collect::<PolarsResult<_>>()?;
            let is_null = s.is_null();
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                if is_null.get(i).unwrap_or(false) {
                    out.push(Value::Null);
                    continue;
                }
                let mut obj = Map::new();
                for (name, values) in &fields {
                    obj.insert(name.clone(), values[i].clone());
                }
                out.push(Value::Object(obj));
            }
            Ok(out)
        }
        _ => {
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let av = s.get(i).map_err(|e| polars_err!(ComputeError: "{}", e))?;
                out.push(any_value_to_json(&av));
            }
            Ok(out)
        }
    }
}

/// Reinterpret a JSON-bearing string cell as structured state.
fn decode_state_json(value: Value) -> Result<Value, String> {
    match value {
        Value::String(s) => serde_json::from_str(&s)
            .map_err(|e| format!("state_json=True but the value is not valid JSON: {e}")),
        other => Ok(other),
    }
}

/// Build one `state` per row.
///
/// `Ok(None)` means "every state input was null" -- that row is emitted as all
/// nulls without ever reaching the API, so a frame with gaps costs nothing for
/// the gaps.
fn build_states(
    inputs: &[Series],
    state_json: bool,
    n_rows: usize,
) -> PolarsResult<Vec<Result<Option<Value>, String>>> {
    let columns: Vec<Vec<Value>> = inputs
        .iter()
        .map(|s| {
            let values = series_to_json(s)?;
            if values.len() == n_rows {
                Ok(values)
            } else if values.len() == 1 {
                // A length-1 input broadcasts. Pairing a constant with
                // per-row state -- a shared policy, a schema, the document
                // every row is being scored against -- is a natural thing to
                // ask for, and TypeSafe's own state guidance shows exactly
                // that shape.
                Ok(vec![values[0].clone(); n_rows])
            } else {
                polars_bail!(
                    ComputeError:
                    "typesafe_eval: state column '{}' has length {}, expected {} or 1 (broadcast)",
                    s.name(), values.len(), n_rows
                )
            }
        })
        .collect::<PolarsResult<_>>()?;
    let mut states = Vec::with_capacity(n_rows);

    if inputs.len() == 1 {
        for value in columns.into_iter().next().unwrap_or_default() {
            if value.is_null() {
                states.push(Ok(None));
            } else if state_json {
                states.push(decode_state_json(value).map(Some));
            } else {
                states.push(Ok(Some(value)));
            }
        }
        return Ok(states);
    }

    // Several inputs -> one JSON object per row, keyed by column name. This is
    // the shape TypeSafe recommends for most requests: named fields keep the
    // relationship between the parts of the state explicit.
    for i in 0..n_rows {
        let mut obj = Map::new();
        let mut all_null = true;
        let mut row_err: Option<String> = None;

        for (s, column) in inputs.iter().zip(&columns) {
            let mut value = column[i].clone();
            if !value.is_null() {
                all_null = false;
                if state_json {
                    match decode_state_json(value) {
                        Ok(decoded) => value = decoded,
                        Err(e) => {
                            row_err = Some(format!("column '{}': {e}", s.name()));
                            break;
                        }
                    }
                }
            }
            obj.insert(s.name().to_string(), value);
        }

        states.push(match row_err {
            Some(e) => Err(e),
            None if all_null => Ok(None),
            None => Ok(Some(Value::Object(obj))),
        });
    }

    Ok(states)
}

// ============================================================================
// Output buffers
// ============================================================================

enum ColBuf {
    F64(Vec<Option<f64>>),
    Str(Vec<Option<String>>),
    I64(Vec<Option<i64>>),
}

impl ColBuf {
    fn new(dtype: &DataType, capacity: usize) -> Self {
        match dtype {
            DataType::Float64 => ColBuf::F64(Vec::with_capacity(capacity)),
            DataType::Int64 => ColBuf::I64(Vec::with_capacity(capacity)),
            _ => ColBuf::Str(Vec::with_capacity(capacity)),
        }
    }

    fn push_null(&mut self) {
        match self {
            ColBuf::F64(v) => v.push(None),
            ColBuf::Str(v) => v.push(None),
            ColBuf::I64(v) => v.push(None),
        }
    }

    fn into_series(self, name: &str) -> Series {
        let name = PlSmallStr::from_str(name);
        match self {
            ColBuf::F64(v) => Float64Chunked::from_iter_options(name, v.into_iter()).into_series(),
            ColBuf::I64(v) => Int64Chunked::from_iter_options(name, v.into_iter()).into_series(),
            ColBuf::Str(v) => StringChunked::from_iter_options(name, v.into_iter()).into_series(),
        }
    }
}

/// Write one output row across every column.
///
/// `qid_prefix` is empty in per-row mode. In per-line mode every question is
/// sent once per segment under the key `"<line index>#<question id>"`, so the
/// prefix selects this segment's answers out of the shared response --
/// the whole point being that one request answers the contract for every line.
/// `line` carries the segment's id and text for the `LineId`/`LineText`
/// columns, and is `None` in per-row mode.
fn push_row(
    bufs: &mut [ColBuf],
    cols: &[(String, Col)],
    ev: Option<&Evaluation>,
    err: Option<&str>,
    line: Option<(i64, &str)>,
    qid_prefix: &str,
) {
    let answer = |q: &str| -> Option<&Value> {
        let key = if qid_prefix.is_empty() {
            q.to_string()
        } else {
            format!("{qid_prefix}{q}")
        };
        ev.and_then(|e| e.answer(&key))
    };
    let num = |q: &str, field: &str| -> Option<f64> {
        answer(q).and_then(|a| a.get(field)).and_then(Value::as_f64)
    };

    for (buf, (_, col)) in bufs.iter_mut().zip(cols) {
        match (buf, col) {
            (ColBuf::F64(v), Col::Noul(q)) => v.push(num(q, "noul")),
            (ColBuf::F64(v), Col::Score(q)) => v.push(num(q, "score")),
            (ColBuf::F64(v), Col::Confidence(q)) => v.push(num(q, "confidence")),
            (ColBuf::F64(v), Col::Prob(q, key)) => v.push(
                answer(q)
                    .and_then(|a| a.get("probabilities"))
                    .and_then(|p| p.get(key))
                    .and_then(Value::as_f64),
            ),
            (ColBuf::Str(v), Col::Choice(q)) => v.push(
                answer(q)
                    .and_then(|a| a.get("choice"))
                    .and_then(Value::as_str)
                    .map(str::to_string),
            ),
            (ColBuf::I64(v), Col::LineId) => v.push(line.map(|(id, _)| id)),
            (ColBuf::Str(v), Col::LineText) => v.push(line.map(|(_, t)| t.to_string())),
            (ColBuf::Str(v), Col::Error) => v.push(err.map(str::to_string)),
            (ColBuf::Str(v), Col::Model) => v.push(ev.map(|e| e.model.clone())),
            (ColBuf::I64(v), Col::InputTokens) => v.push(ev.and_then(|e| e.usage.input_tokens)),
            (ColBuf::I64(v), Col::OutputTokens) => v.push(ev.and_then(|e| e.usage.output_tokens)),
            (ColBuf::I64(v), Col::LatencyMs) => v.push(ev.and_then(|e| e.usage.latency_ms)),
            // dtype and plan are derived from the same `Col`, so this is
            // unreachable; stay null rather than panic if that ever changes.
            (buf, _) => buf.push_null(),
        }
    }
}

// ============================================================================
// Expression
// ============================================================================

#[polars_expr(output_type_func_with_kwargs=typesafe_output_type)]
fn typesafe_eval(inputs: &[Series], kwargs: TypeSafeKwargs) -> PolarsResult<Series> {
    let (specs, cols) = plan_from_kwargs(&kwargs)?;

    if inputs.is_empty() {
        polars_bail!(ComputeError: "typesafe_eval: at least one state column is required");
    }
    // Length-1 inputs broadcast against the longest column (see
    // `build_states`), so the row count is the widest input.
    let n_rows = inputs.iter().map(|s| s.len()).max().unwrap_or(0);

    let model = kwargs
        .model
        .clone()
        .unwrap_or_else(|| DEFAULT_TYPESAFE_MODEL.to_string());
    let states = build_states(inputs, kwargs.state_json.unwrap_or(false), n_rows)?;

    // Only rows with a usable state cost a request; nulls and malformed
    // state_json rows are filled in locally.
    let mut send_indices: Vec<usize> = Vec::new();
    let mut send_states: Vec<Value> = Vec::new();
    for (i, state) in states.iter().enumerate() {
        if let Ok(Some(v)) = state {
            send_indices.push(i);
            send_states.push(v.clone());
        }
    }

    let mut evaluations: Vec<Option<Result<Evaluation, String>>> = vec![None; n_rows];
    if !send_states.is_empty() {
        let specs_owned = specs.clone();
        let model_owned = model.clone();
        let results = run_async(async move {
            typesafe::evaluate_batch(&send_states, &model_owned, &specs_owned).await
        });
        for (i, result) in send_indices.into_iter().zip(results) {
            evaluations[i] = Some(result.map_err(|e| e.to_string()));
        }
    }

    let mut bufs: Vec<ColBuf> = cols
        .iter()
        .map(|(_, col)| ColBuf::new(&col.dtype(), n_rows))
        .collect();

    for (i, state) in states.into_iter().enumerate() {
        match state {
            // Null state: every field null, including `_error` -- a missing
            // input is not a failure.
            Ok(None) => push_row(&mut bufs, &cols, None, None, None, ""),
            Err(e) => push_row(&mut bufs, &cols, None, Some(&e), None, ""),
            Ok(Some(_)) => match &evaluations[i] {
                Some(Ok(ev)) => push_row(&mut bufs, &cols, Some(ev), None, None, ""),
                Some(Err(e)) => push_row(&mut bufs, &cols, None, Some(e), None, ""),
                None => push_row(&mut bufs, &cols, None, Some("no result returned"), None, ""),
            },
        }
    }

    let series: Vec<Series> = bufs
        .into_iter()
        .zip(&cols)
        .map(|(buf, (name, _))| buf.into_series(name))
        .collect();

    let out = StructChunked::from_series(PlSmallStr::from_static(""), n_rows, series.iter())?;
    Ok(out.into_series())
}

// ============================================================================
// Per-line dimension: one contract, applied to every segment of a document
// ============================================================================

/// Default cap on questions per request.
///
/// The real limit is token-based, not count-based: measured against the live
/// API, 640 questions (~40k input tokens) succeeded while 1200 returned
/// `400 max_tokens_exceeded`. Long segments hit that ceiling at a lower count,
/// so the default leaves generous headroom -- and 200 questions per request is
/// already 200x fewer round trips than evaluating a line at a time.
const DEFAULT_MAX_QUESTIONS: usize = 200;

/// Separator between a segment's index and the contract's question id in the
/// composite key sent to the API. `#` cannot appear in a Python identifier, so
/// a contract derived from a Pydantic model can never collide with it; the
/// Python layer rejects a hand-written question id containing one.
const LINE_QID_SEP: char = '#';

#[derive(Debug, Deserialize)]
pub struct TypeSafeEachKwargs {
    questions: String,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    probabilities: Option<bool>,
    #[serde(default)]
    usage: Option<bool>,
    /// Upper bound on questions per request; segments are chunked to respect
    /// it. Each segment contributes one question per contract entry, so the
    /// segments per request is this divided by the contract size.
    #[serde(default)]
    max_questions: Option<usize>,
    /// Emit the segment's own text alongside its answers.
    #[serde(default)]
    include_segment: Option<bool>,
}

fn plan_columns_each(
    specs: &[QuestionSpec],
    probabilities: bool,
    usage: bool,
    include_segment: bool,
) -> PolarsResult<ColumnPlan> {
    let mut cols: ColumnPlan = vec![("line_id".to_string(), Col::LineId)];
    if include_segment {
        cols.push(("line".to_string(), Col::LineText));
    }
    cols.extend(plan_columns(specs, probabilities, usage)?);
    ensure_unique_fields(&cols)?;
    Ok(cols)
}

fn plan_each_from_kwargs(
    kwargs: &TypeSafeEachKwargs,
) -> PolarsResult<(Vec<QuestionSpec>, ColumnPlan)> {
    let specs = typesafe::parse_questions(&kwargs.questions)
        .map_err(|e| polars_err!(ComputeError: "typesafe_eval_each: {}", e))?;
    let cols = plan_columns_each(
        &specs,
        kwargs.probabilities.unwrap_or(false),
        kwargs.usage.unwrap_or(false),
        kwargs.include_segment.unwrap_or(true),
    )?;
    Ok((specs, cols))
}

/// The segments argument must be a `List` column. Checked here rather than in
/// the expression body so a `pl.col("text")` mistake surfaces at schema
/// resolution -- before a single request is billed.
fn ensure_segments_dtype(dtype: &DataType) -> PolarsResult<()> {
    if matches!(dtype, DataType::List(_) | DataType::Null) {
        return Ok(());
    }
    polars_bail!(
        ComputeError:
        "typesafe_eval_each: the first argument must be a List column of segments, got {} -- \
         split a document first, e.g. pl.col(\"text\").str.split(\"\\n\")",
        dtype
    )
}

fn typesafe_each_output_type(
    input_fields: &[Field],
    kwargs: TypeSafeEachKwargs,
) -> PolarsResult<Field> {
    if let Some(first) = input_fields.first() {
        ensure_segments_dtype(first.dtype())?;
    }
    let (_, cols) = plan_each_from_kwargs(&kwargs)?;
    let fields = cols
        .iter()
        .map(|(name, col)| Field::new(PlSmallStr::from_str(name), col.dtype()))
        .collect::<Vec<_>>();
    Ok(Field::new(
        PlSmallStr::from_static(""),
        DataType::List(Box::new(DataType::Struct(fields))),
    ))
}

/// The segment text as the model (and the `line` column) sees it: a JSON
/// string stays a bare string, anything else is rendered as compact JSON.
fn segment_text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// One request: a contiguous slice of one row's segments, plus the row it
/// belongs to so the answers can be put back where they came from.
struct Chunk {
    row: usize,
    /// Global segment indices covered by this chunk, in order.
    line_ids: Vec<usize>,
}

/// Build the request for one chunk: the chunk's segments go into `state` once,
/// keyed by their global index, and each question names the index it applies
/// to. Repeating the segment text inside every question instead measures only
/// ~11% more expensive -- the win here is context, not bytes: a clause like
/// "renews automatically unless either party gives notice" is unreadable
/// without its neighbours, and this way the model sees them.
fn build_chunk_state(chunk: &Chunk, segments: &[Value], context: &Map<String, Value>) -> Value {
    let mut lines = Map::new();
    for &id in &chunk.line_ids {
        lines.insert(id.to_string(), segments[id].clone());
    }
    let mut state = Map::new();
    state.insert("lines".to_string(), Value::Object(lines));
    for (k, v) in context {
        state.insert(k.clone(), v.clone());
    }
    Value::Object(state)
}

/// Expand the contract into one question per segment in the chunk, keyed
/// `"<line index>#<question id>"`.
fn build_chunk_questions(chunk: &Chunk, specs: &[QuestionSpec]) -> Vec<QuestionSpec> {
    let mut out = Vec::with_capacity(chunk.line_ids.len() * specs.len());
    for &id in &chunk.line_ids {
        for spec in specs {
            let mut per_line = spec.clone();
            per_line.id = format!("{id}{LINE_QID_SEP}{}", spec.id);
            per_line.retarget_to_line(&id.to_string());
            out.push(per_line);
        }
    }
    out
}

#[polars_expr(output_type_func_with_kwargs=typesafe_each_output_type)]
fn typesafe_eval_each(inputs: &[Series], kwargs: TypeSafeEachKwargs) -> PolarsResult<Series> {
    let (specs, cols) = plan_each_from_kwargs(&kwargs)?;

    if inputs.is_empty() {
        polars_bail!(ComputeError: "typesafe_eval_each: a segments column is required");
    }
    let segments_series = &inputs[0];
    ensure_segments_dtype(segments_series.dtype())?;

    let n_rows = inputs.iter().map(|s| s.len()).max().unwrap_or(0);
    let model = kwargs
        .model
        .clone()
        .unwrap_or_else(|| DEFAULT_TYPESAFE_MODEL.to_string());

    // Each segment costs one question per contract entry, so the segments per
    // request is the question budget divided by the contract size.
    let max_questions = kwargs.max_questions.unwrap_or(DEFAULT_MAX_QUESTIONS).max(1);
    let per_chunk = (max_questions / specs.len().max(1)).max(1);

    // Shared context: any further columns ride along in every chunk's state.
    let context_columns = build_context_columns(&inputs[1..], n_rows)?;

    // Segments per row, and the chunks they decompose into.
    let mut row_segments: Vec<Option<Vec<Value>>> = Vec::with_capacity(n_rows);
    let mut chunks: Vec<Chunk> = Vec::new();
    {
        let ca = segments_series.list()?;
        for (row, opt) in ca.amortized_iter().enumerate() {
            match opt {
                None => row_segments.push(None),
                Some(inner) => {
                    let values = series_to_json(inner.as_ref())?;
                    for start in (0..values.len()).step_by(per_chunk) {
                        let end = (start + per_chunk).min(values.len());
                        chunks.push(Chunk {
                            row,
                            line_ids: (start..end).collect(),
                        });
                    }
                    row_segments.push(Some(values));
                }
            }
        }
    }

    // One request per chunk, all in flight together under the shared cap.
    let mut results: Vec<Result<Evaluation, String>> = Vec::new();
    if !chunks.is_empty() {
        let mut states = Vec::with_capacity(chunks.len());
        let mut question_sets = Vec::with_capacity(chunks.len());
        for chunk in &chunks {
            let segments = row_segments[chunk.row]
                .as_ref()
                .expect("a chunk only exists for a row with segments");
            let empty = Map::new();
            let context = context_columns.get(chunk.row).unwrap_or(&empty);
            states.push(build_chunk_state(chunk, segments, context));
            question_sets.push(build_chunk_questions(chunk, &specs));
        }
        let model_owned = model.clone();
        results = run_async(async move {
            typesafe::evaluate_batch_varying(&states, &model_owned, &question_sets).await
        })
        .into_iter()
        .map(|r| r.map_err(|e| e.to_string()))
        .collect();
    }

    // Which chunk covers each (row, segment)?
    let mut chunk_of: Vec<Vec<usize>> = vec![Vec::new(); n_rows];
    for (i, chunk) in chunks.iter().enumerate() {
        chunk_of[chunk.row].push(i);
    }

    let inner_dtype = DataType::Struct(
        cols.iter()
            .map(|(name, col)| Field::new(PlSmallStr::from_str(name), col.dtype()))
            .collect(),
    );
    let mut builder =
        AnonymousOwnedListBuilder::new(PlSmallStr::from_static(""), n_rows, Some(inner_dtype));

    for row in 0..n_rows {
        let segments = match &row_segments[row] {
            // A null segments list is a missing input, not a failure.
            None => {
                builder.append_null();
                continue;
            }
            Some(values) => values,
        };
        if segments.is_empty() {
            builder.append_empty();
            continue;
        }

        let mut bufs: Vec<ColBuf> = cols
            .iter()
            .map(|(_, col)| ColBuf::new(&col.dtype(), segments.len()))
            .collect();

        for &chunk_idx in &chunk_of[row] {
            let chunk = &chunks[chunk_idx];
            let (ev, err) = match &results[chunk_idx] {
                Ok(ev) => (Some(ev), None),
                Err(e) => (None, Some(e.as_str())),
            };
            for &id in &chunk.line_ids {
                let text = segment_text(&segments[id]);
                push_row(
                    &mut bufs,
                    &cols,
                    ev,
                    err,
                    Some((id as i64, &text)),
                    &format!("{id}{LINE_QID_SEP}"),
                );
            }
        }

        let series: Vec<Series> = bufs
            .into_iter()
            .zip(&cols)
            .map(|(buf, (name, _))| buf.into_series(name))
            .collect();
        let struct_series =
            StructChunked::from_series(PlSmallStr::from_static(""), segments.len(), series.iter())?
                .into_series();
        builder.append_series(&struct_series)?;
    }

    Ok(builder.finish().into_series())
}

/// Per-row shared context: the columns after the segments column, as a JSON
/// object per row, broadcast where length-1.
fn build_context_columns(
    inputs: &[Series],
    n_rows: usize,
) -> PolarsResult<Vec<Map<String, Value>>> {
    if inputs.is_empty() {
        return Ok(vec![Map::new(); n_rows]);
    }
    let columns: Vec<Vec<Value>> = inputs
        .iter()
        .map(|s| {
            let values = series_to_json(s)?;
            if values.len() == n_rows {
                Ok(values)
            } else if values.len() == 1 {
                Ok(vec![values[0].clone(); n_rows])
            } else {
                polars_bail!(
                    ComputeError:
                    "typesafe_eval_each: context column '{}' has length {}, expected {} or 1 (broadcast)",
                    s.name(), values.len(), n_rows
                )
            }
        })
        .collect::<PolarsResult<_>>()?;

    Ok((0..n_rows)
        .map(|i| {
            let mut obj = Map::new();
            for (s, column) in inputs.iter().zip(&columns) {
                obj.insert(s.name().to_string(), column[i].clone());
            }
            obj
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_client::typesafe::parse_questions;
    use serde_json::json;

    const QUESTIONS: &str = r#"[
        {"id":"urgent","type":"noul","instructions":"Urgent?"},
        {"id":"dept","type":"choice","instructions":"Who?",
         "options":[{"name":"billing"},{"name":"technical"}]},
        {"id":"mood","type":"score","instructions":"How?","levels":["Calm","Angry"]}
    ]"#;

    fn names(cols: &ColumnPlan) -> Vec<String> {
        cols.iter().map(|(n, _)| n.clone()).collect()
    }

    #[test]
    fn plan_follows_declared_question_order() {
        let specs = parse_questions(QUESTIONS).unwrap();
        let cols = plan_columns(&specs, false, false).unwrap();
        assert_eq!(
            names(&cols),
            vec![
                "urgent",
                "dept",
                "dept_confidence",
                "mood",
                "mood_confidence",
                "_error"
            ]
        );
    }

    #[test]
    fn noul_gets_no_confidence_column() {
        let specs = parse_questions(QUESTIONS).unwrap();
        let cols = plan_columns(&specs, true, false).unwrap();
        assert!(!names(&cols).contains(&"urgent_confidence".to_string()));
    }

    #[test]
    fn probabilities_expand_options_and_level_indices() {
        let specs = parse_questions(QUESTIONS).unwrap();
        let cols = plan_columns(&specs, true, false).unwrap();
        let n = names(&cols);
        assert!(n.contains(&"dept_p_billing".to_string()));
        assert!(n.contains(&"dept_p_technical".to_string()));
        assert!(n.contains(&"mood_p_0".to_string()));
        assert!(n.contains(&"mood_p_1".to_string()));
    }

    #[test]
    fn usage_appends_accounting_columns() {
        let specs = parse_questions(QUESTIONS).unwrap();
        let cols = plan_columns(&specs, false, true).unwrap();
        let n = names(&cols);
        for expected in ["_model", "_input_tokens", "_output_tokens", "_latency_ms"] {
            assert!(n.contains(&expected.to_string()), "missing {expected}");
        }
    }

    #[test]
    fn dtypes_match_the_answer_types() {
        let specs = parse_questions(QUESTIONS).unwrap();
        let cols = plan_columns(&specs, true, true).unwrap();
        let by_name: std::collections::HashMap<_, _> =
            cols.iter().map(|(n, c)| (n.as_str(), c.dtype())).collect();
        assert_eq!(by_name["urgent"], DataType::Float64);
        assert_eq!(by_name["dept"], DataType::String);
        assert_eq!(by_name["dept_confidence"], DataType::Float64);
        assert_eq!(by_name["mood"], DataType::Float64);
        assert_eq!(by_name["mood_p_1"], DataType::Float64);
        assert_eq!(by_name["_error"], DataType::String);
        assert_eq!(by_name["_input_tokens"], DataType::Int64);
    }

    #[test]
    fn colliding_field_names_are_rejected() {
        let specs = parse_questions(
            r#"[{"id":"a","type":"choice","instructions":"x",
                 "options":[{"name":"p"},{"name":"q"}]},
                {"id":"a_confidence","type":"noul","instructions":"y"}]"#,
        )
        .unwrap();
        assert!(plan_columns(&specs, false, false).is_err());
    }

    #[test]
    fn single_string_column_becomes_a_bare_string_state() {
        let s = Series::new("msg".into(), &["hello", "world"]);
        let states = build_states(&[s], false, 2).unwrap();
        assert_eq!(states[0], Ok(Some(Value::String("hello".into()))));
    }

    #[test]
    fn null_rows_are_skipped_rather_than_sent() {
        let s = Series::new("msg".into(), &[Some("hi"), None]);
        let states = build_states(&[s], false, 2).unwrap();
        assert!(matches!(states[0], Ok(Some(_))));
        assert_eq!(states[1], Ok(None));
    }

    #[test]
    fn several_columns_become_one_named_object() {
        let a = Series::new("message".into(), &["charged twice"]);
        let b = Series::new("order_id".into(), &[104i64]);
        let states = build_states(&[a, b], false, 1).unwrap();
        let state = states[0].clone().unwrap().unwrap();
        assert_eq!(state["message"], "charged twice");
        assert_eq!(state["order_id"], 104);
    }

    #[test]
    fn per_line_plan_prepends_the_segment_columns() {
        let specs = parse_questions(r#"[{"id":"pays","type":"noul","instructions":"x"}]"#).unwrap();
        let cols = plan_columns_each(&specs, false, false, true).unwrap();
        assert_eq!(names(&cols), vec!["line_id", "line", "pays", "_error"]);
        assert_eq!(cols[0].1.dtype(), DataType::Int64);
        assert_eq!(cols[1].1.dtype(), DataType::String);
    }

    #[test]
    fn include_segment_false_drops_the_text_column() {
        let specs = parse_questions(r#"[{"id":"pays","type":"noul","instructions":"x"}]"#).unwrap();
        let cols = plan_columns_each(&specs, false, false, false).unwrap();
        assert_eq!(names(&cols), vec!["line_id", "pays", "_error"]);
    }

    #[test]
    fn a_contract_field_named_line_would_collide_and_is_rejected() {
        let specs = parse_questions(r#"[{"id":"line","type":"noul","instructions":"x"}]"#).unwrap();
        assert!(plan_columns_each(&specs, false, false, true).is_err());
        // ...but is fine when the segment text isn't emitted.
        assert!(plan_columns_each(&specs, false, false, false).is_ok());
    }

    #[test]
    fn chunk_questions_are_keyed_by_line_and_keep_the_contract() {
        let specs = parse_questions(
            r#"[{"id":"pays","type":"noul","instructions":"Payment?"},
                {"id":"kind","type":"choice","instructions":"Which?",
                 "options":[{"name":"a"},{"name":"b"}]}]"#,
        )
        .unwrap();
        let chunk = Chunk {
            row: 0,
            line_ids: vec![3, 4],
        };
        let qs = build_chunk_questions(&chunk, &specs);
        // Every segment costs one question per contract entry.
        assert_eq!(qs.len(), 4);
        let ids: Vec<&str> = qs.iter().map(|q| q.id.as_str()).collect();
        assert_eq!(ids, vec!["3#pays", "3#kind", "4#pays", "4#kind"]);
        let wire = qs[0].to_wire();
        assert_eq!(wire["instructions"]["question"], "Payment?");
        assert_eq!(wire["instructions"]["evaluate_line_id"], "3");
    }

    #[test]
    fn chunk_state_carries_the_segments_once_under_their_global_ids() {
        let segments = vec![json!("a"), json!("b"), json!("c")];
        let chunk = Chunk {
            row: 0,
            line_ids: vec![1, 2],
        };
        let mut context = Map::new();
        context.insert("title".to_string(), json!("MSA"));
        let state = build_chunk_state(&chunk, &segments, &context);
        // Global ids, so a chunk boundary never renumbers a line.
        assert_eq!(state["lines"], json!({"1": "b", "2": "c"}));
        assert_eq!(state["title"], "MSA");
    }

    #[test]
    fn a_list_column_becomes_a_json_array() {
        let s = Series::new(
            "turns".into(),
            &[Series::new("".into(), &["hi", "my card was charged twice"])],
        );
        let states = build_states(&[s], false, 1).unwrap();
        assert_eq!(
            states[0].clone().unwrap().unwrap(),
            json!(["hi", "my card was charged twice"])
        );
    }

    #[test]
    fn a_struct_column_becomes_a_json_object() {
        let df = df!["text" => ["charged twice"], "amount" => [49i64]].unwrap();
        let s = df.into_struct("ticket".into()).into_series();
        let states = build_states(&[s], false, 1).unwrap();
        let state = states[0].clone().unwrap().unwrap();
        assert_eq!(state["text"], "charged twice");
        assert_eq!(state["amount"], 49);
    }

    #[test]
    fn a_length_one_column_broadcasts_across_the_frame() {
        let rows = Series::new("msg".into(), &["a", "b", "c"]);
        let constant = Series::new("policy".into(), &["duplicates are refundable"]);
        let states = build_states(&[rows, constant], false, 3).unwrap();
        assert_eq!(states.len(), 3);
        for (i, expected) in ["a", "b", "c"].iter().enumerate() {
            let state = states[i].clone().unwrap().unwrap();
            assert_eq!(state["msg"], *expected);
            assert_eq!(state["policy"], "duplicates are refundable");
        }
    }

    #[test]
    fn a_column_that_neither_matches_nor_broadcasts_is_rejected() {
        let rows = Series::new("msg".into(), &["a", "b", "c"]);
        let wrong = Series::new("other".into(), &["x", "y"]);
        let err = build_states(&[rows, wrong], false, 3).unwrap_err();
        assert!(err.to_string().contains("expected 3 or 1"), "{err}");
    }

    #[test]
    fn state_json_decodes_documents_and_reports_bad_ones() {
        let good = Series::new("s".into(), &[r#"{"a":1}"#]);
        let states = build_states(&[good], true, 1).unwrap();
        assert_eq!(states[0].clone().unwrap().unwrap()["a"], 1);

        let bad = Series::new("s".into(), &["not json"]);
        let states = build_states(&[bad], true, 1).unwrap();
        assert!(states[0].is_err());
    }
}
