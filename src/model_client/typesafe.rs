//! TypeSafe "System One" inference layer (<https://docs.typesafe.ai/api>).
//!
//! TypeSafe is deliberately *not* a chat-completions provider, so it does not
//! implement [`ModelClient`](super::ModelClient): there is no prompt and no
//! free-text completion. One request carries a single `state` plus a map of
//! typed `questions`, and comes back with one typed `answer` per question --
//! a yes/no probability (`noul`), a pick from a closed set with a full
//! probability distribution (`choice`), or a probability-weighted rating over
//! ordered levels (`score`).
//!
//! That shape is exactly a DataFrame row: one state per row, N questions ->
//! N typed columns, all answered in a *single* HTTP call per row (the
//! "speculative fan-out" pattern TypeSafe's own docs recommend). The polars
//! expression built on top of this module lives in
//! `crate::expressions::typesafe_eval`.
//!
//! Configuration (env):
//!   * `TYPESAFE_API_KEY`   -- bearer token (required).
//!   * `TYPESAFE_BASE_URL`  -- API root override; default `https://api.typesafe.ai`.
//!   * `POLAR_LLAMA_TYPESAFE_MAX_RETRIES` -- retry budget for 429/529/5xx (default 3).

use futures::StreamExt;
use reqwest::{Client, StatusCode};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::{ModelClientError, Usage};

/// TypeSafe's flagship System One model.
pub const DEFAULT_TYPESAFE_MODEL: &str = "jev-latest";

/// API root, overridable for proxies/self-hosted gateways/tests.
fn api_base() -> String {
    std::env::var("TYPESAFE_BASE_URL").unwrap_or_else(|_| "https://api.typesafe.ai".to_string())
}

/// `POST` target for evaluations.
pub fn evaluation_endpoint() -> String {
    format!("{}/v1/systemone", api_base().trim_end_matches('/'))
}

/// `GET` target for the model catalogue.
pub fn models_endpoint() -> String {
    format!("{}/v1/models", api_base().trim_end_matches('/'))
}

/// Bearer token. Empty string when unset -- the API answers 401, which the
/// caller surfaces per row rather than panicking the whole frame.
pub fn api_key() -> String {
    std::env::var("TYPESAFE_API_KEY").unwrap_or_default()
}

/// Ceiling for a single backoff sleep.
const MAX_BACKOFF_MS: u64 = 30_000;

fn max_retries() -> u32 {
    std::env::var("POLAR_LLAMA_TYPESAFE_MAX_RETRIES")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(3)
}

// ============================================================================
// Question specs (the polar-llama-side description, not the wire format)
// ============================================================================

/// Every TypeSafe free-form field (`instructions`, option descriptions, level
/// descriptions, noul `criteria.true`/`criteria.false`) accepts
/// `string | object | array | null`, so all of them are held as raw
/// `serde_json::Value` and passed through untouched.
pub type Entry = Value;

/// One named question. `id` is the key the answer comes back under; TypeSafe
/// documents that the key itself never reaches the model.
#[derive(Debug, Clone, Deserialize)]
pub struct QuestionSpec {
    pub id: String,
    #[serde(flatten)]
    pub kind: QuestionKind,
}

/// A choice option. Kept as an ordered `Vec` rather than a map so the
/// generated probability columns have a stable, user-declared order --
/// `serde_json::Map` is a `BTreeMap` here and would silently re-sort them.
#[derive(Debug, Clone, Deserialize)]
pub struct ChoiceOption {
    pub name: String,
    #[serde(default)]
    pub description: Entry,
}

/// Optional descriptions of what a yes and a no mean for a noul question.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct NoulCriteria {
    #[serde(default, rename = "true")]
    pub yes: Entry,
    #[serde(default, rename = "false")]
    pub no: Entry,
}

impl NoulCriteria {
    fn is_empty(&self) -> bool {
        self.yes.is_null() && self.no.is_null()
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum QuestionKind {
    /// Yes/no. Answer is a probability in `[0, 1]`; carries no confidence.
    Noul {
        #[serde(default)]
        instructions: Entry,
        #[serde(default)]
        criteria: Option<NoulCriteria>,
    },
    /// Pick one of a closed set. Answer carries the pick, a distribution over
    /// every option, and a confidence.
    Choice {
        #[serde(default)]
        instructions: Entry,
        options: Vec<ChoiceOption>,
    },
    /// Rate against ordered levels. Answer carries a probability-weighted
    /// value that can land between levels, a distribution, and a confidence.
    Score {
        #[serde(default)]
        instructions: Entry,
        levels: Vec<Entry>,
    },
}

impl QuestionSpec {
    /// Render this question into TypeSafe's wire format.
    ///
    /// Choice `criteria` becomes a JSON object; because `serde_json::Map` is a
    /// `BTreeMap` in this build, its keys serialize in sorted order. That is
    /// deterministic (which matters for run manifests) and independent of the
    /// declared order used for output columns.
    pub fn to_wire(&self) -> Value {
        match &self.kind {
            QuestionKind::Noul {
                instructions,
                criteria,
            } => {
                let mut q = json!({ "type": "noul", "instructions": instructions });
                if let Some(c) = criteria {
                    if !c.is_empty() {
                        q["criteria"] = json!({ "true": c.yes, "false": c.no });
                    }
                }
                q
            }
            QuestionKind::Choice {
                instructions,
                options,
            } => {
                let mut criteria = Map::new();
                for opt in options {
                    criteria.insert(opt.name.clone(), opt.description.clone());
                }
                json!({
                    "type": "choice",
                    "instructions": instructions,
                    "criteria": Value::Object(criteria),
                })
            }
            QuestionKind::Score {
                instructions,
                levels,
            } => json!({
                "type": "score",
                "instructions": instructions,
                "criteria": levels,
            }),
        }
    }
}

impl QuestionKind {
    /// Mutable access to this question's `instructions`, whatever its type.
    fn instructions_mut(&mut self) -> &mut Entry {
        match self {
            QuestionKind::Noul { instructions, .. }
            | QuestionKind::Choice { instructions, .. }
            | QuestionKind::Score { instructions, .. } => instructions,
        }
    }
}

impl QuestionSpec {
    /// Point this question at one segment of a document held in `state.lines`.
    ///
    /// The original instructions are preserved verbatim under `question`, with
    /// the target line named alongside them. This is what lets a single
    /// request answer the same contract for every line while the model still
    /// sees the surrounding lines as context.
    pub fn retarget_to_line(&mut self, line_id: &str) {
        let instructions = self.kind.instructions_mut();
        let original = std::mem::replace(instructions, Value::Null);
        *instructions = json!({
            "question": original,
            "evaluate_line_id": line_id,
        });
    }
}

/// Parse the JSON array of question specs handed over from Python.
///
/// An array (not an object) because declared order drives output column
/// order, and a JSON object would lose it.
pub fn parse_questions(json_str: &str) -> Result<Vec<QuestionSpec>, String> {
    let specs: Vec<QuestionSpec> = serde_json::from_str(json_str)
        .map_err(|e| format!("Failed to parse TypeSafe questions: {e}"))?;
    if specs.is_empty() {
        return Err("TypeSafe requires at least one question".to_string());
    }
    for spec in &specs {
        match &spec.kind {
            QuestionKind::Choice { options, .. } => {
                if options.len() < 2 {
                    return Err(format!(
                        "Choice question '{}' needs at least 2 options",
                        spec.id
                    ));
                }
            }
            QuestionKind::Score { levels, .. } => {
                if levels.len() < 2 {
                    return Err(format!(
                        "Score question '{}' needs at least 2 levels",
                        spec.id
                    ));
                }
            }
            QuestionKind::Noul { .. } => {}
        }
    }
    Ok(specs)
}

/// Build the full request body for one state.
pub fn build_request(state: &Value, model: &str, questions: &[QuestionSpec]) -> Value {
    let mut map = Map::new();
    for spec in questions {
        map.insert(spec.id.clone(), spec.to_wire());
    }
    json!({
        "state": state,
        "model": model,
        "questions": Value::Object(map),
    })
}

// ============================================================================
// Responses
// ============================================================================

/// One evaluated row: the answers keyed by question id, plus accounting.
#[derive(Debug, Clone)]
pub struct Evaluation {
    /// The model that actually ran (a resolved version such as `jev-1.13.0`,
    /// not necessarily the alias that was requested).
    pub model: String,
    /// Raw answer objects, keyed by question id. Kept as `Value` rather than
    /// typed structs so a field TypeSafe adds later can never break parsing.
    pub answers: Map<String, Value>,
    pub usage: Usage,
}

impl Evaluation {
    pub fn answer(&self, question_id: &str) -> Option<&Value> {
        self.answers.get(question_id)
    }
}

/// A model listed by `GET /v1/models`.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelCard {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub release_date: Option<String>,
}

fn parse_evaluation(body: &str, latency_ms: i64) -> Result<Evaluation, ModelClientError> {
    let v: Value = serde_json::from_str(body)?;
    let answers = v
        .get("answers")
        .and_then(Value::as_object)
        .cloned()
        .ok_or_else(|| {
            ModelClientError::ParseError("TypeSafe response has no `answers` object".to_string())
        })?;
    let model = v
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    // Usage is best-effort, exactly as for the chat providers: a missing or
    // renamed block leaves the fields `None` and is never an error.
    let usage_block = v.get("usage");
    let usage = Usage {
        input_tokens: usage_block
            .and_then(|u| u.get("input_tokens"))
            .and_then(Value::as_i64),
        output_tokens: usage_block
            .and_then(|u| u.get("output_tokens"))
            .and_then(Value::as_i64),
        // TypeSafe reports no prompt-cache field.
        cached_tokens: None,
        latency_ms: Some(latency_ms),
    };
    Ok(Evaluation {
        model,
        answers,
        usage,
    })
}

// ============================================================================
// Retry policy
// ============================================================================

/// TypeSafe documents 429 (rate limited) and 529 (overloaded) as "retry with
/// exponential backoff"; transient 5xx are treated the same way. Everything
/// else (401 bad key, 422 malformed question) is a caller error that retrying
/// would only bill for again.
fn is_retryable(status: StatusCode) -> bool {
    let code = status.as_u16();
    code == 429 || code == 529 || (500..=599).contains(&code)
}

/// Exponential backoff with decorrelated jitter, capped at 30s.
///
/// Jitter is seeded from the wall clock rather than a `rand` dependency: its
/// only job is to stop a whole DataFrame's worth of rows from retrying in
/// lockstep after a shared 429.
fn backoff_delay(attempt: u32, retry_after: Option<Duration>) -> Duration {
    if let Some(d) = retry_after {
        return d.min(Duration::from_secs(60));
    }

    let base_ms = 500u64.saturating_mul(1u64 << attempt.min(6));
    let capped = base_ms.min(MAX_BACKOFF_MS);
    // Jitter is subtracted, never added, so the cap is a real ceiling: the
    // delay lands in [75%, 100%] of the backoff for this attempt.
    let jitter_span = capped / 4;
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(0);
    let jitter = if jitter_span == 0 {
        0
    } else {
        nanos % jitter_span
    };
    Duration::from_millis(capped.saturating_sub(jitter))
}

fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    headers
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .map(Duration::from_secs)
}

// ============================================================================
// Client
// ============================================================================

/// Evaluate a single state against `questions`.
///
/// `latency_ms` on the returned [`Usage`] is total wall clock across *all*
/// attempts, so a row that needed two backoffs reports the time the caller
/// actually waited.
pub async fn evaluate(
    http: &Client,
    state: &Value,
    model: &str,
    questions: &[QuestionSpec],
) -> Result<Evaluation, ModelClientError> {
    let body = build_request(state, model, questions);
    let endpoint = evaluation_endpoint();
    let key = api_key();
    let budget = max_retries();

    let t0 = Instant::now();
    let mut attempt = 0u32;
    loop {
        let sent = http
            .post(&endpoint)
            .bearer_auth(&key)
            .json(&body)
            .send()
            .await;

        match sent {
            Ok(response) => {
                let status = response.status();
                let retry_after = parse_retry_after(response.headers());
                let text = response.text().await?;

                if status.is_success() {
                    return parse_evaluation(&text, t0.elapsed().as_millis() as i64);
                }
                if is_retryable(status) && attempt < budget {
                    tokio::time::sleep(backoff_delay(attempt, retry_after)).await;
                    attempt += 1;
                    continue;
                }
                return Err(ModelClientError::Http(status.as_u16(), text));
            }
            Err(e) => {
                // Connect/timeout failures are worth one more try; a body or
                // decode error is not going to change.
                if (e.is_timeout() || e.is_connect()) && attempt < budget {
                    tokio::time::sleep(backoff_delay(attempt, None)).await;
                    attempt += 1;
                    continue;
                }
                return Err(ModelClientError::RequestError(e));
            }
        }
    }
}

/// Evaluate many states against the same questions, bounded by the shared
/// `POLAR_LLAMA_MAX_CONCURRENCY` limit and preserving input order.
pub async fn evaluate_batch(
    states: &[Value],
    model: &str,
    questions: &[QuestionSpec],
) -> Vec<Result<Evaluation, ModelClientError>> {
    let http = super::http_client();
    let requests: Vec<_> = states
        .iter()
        .map(|state| evaluate(http, state, model, questions))
        .collect();

    futures::stream::iter(requests)
        .buffered(super::max_concurrency())
        .collect()
        .await
}

/// Evaluate many states, each against its *own* question set.
///
/// The per-line path needs this because every chunk carries a different
/// expansion of the contract (one question per segment in that chunk), unlike
/// [`evaluate_batch`] where every row shares one question set.
pub async fn evaluate_batch_varying(
    states: &[Value],
    model: &str,
    question_sets: &[Vec<QuestionSpec>],
) -> Vec<Result<Evaluation, ModelClientError>> {
    let http = super::http_client();
    let requests: Vec<_> = states
        .iter()
        .zip(question_sets)
        .map(|(state, questions)| evaluate(http, state, model, questions))
        .collect();

    futures::stream::iter(requests)
        .buffered(super::max_concurrency())
        .collect()
        .await
}

/// List the models available to the configured API key.
pub async fn list_models(http: &Client) -> Result<Vec<ModelCard>, ModelClientError> {
    let response = http
        .get(models_endpoint())
        .bearer_auth(api_key())
        .send()
        .await?;
    let status = response.status();
    let text = response.text().await?;
    if !status.is_success() {
        return Err(ModelClientError::Http(status.as_u16(), text));
    }
    let v: Value = serde_json::from_str(&text)?;
    let models = v.get("models").cloned().unwrap_or(v);
    serde_json::from_value(models).map_err(ModelClientError::Serialization)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn specs(json_str: &str) -> Vec<QuestionSpec> {
        parse_questions(json_str).expect("specs parse")
    }

    #[test]
    fn noul_wire_omits_empty_criteria() {
        let s = specs(r#"[{"id":"urgent","type":"noul","instructions":"Urgent?"}]"#);
        let wire = s[0].to_wire();
        assert_eq!(wire["type"], "noul");
        assert_eq!(wire["instructions"], "Urgent?");
        assert!(wire.get("criteria").is_none());
    }

    #[test]
    fn noul_wire_keeps_true_false_keys() {
        let s = specs(
            r#"[{"id":"u","type":"noul","instructions":"Urgent?",
                 "criteria":{"true":"time-sensitive","false":"not"}}]"#,
        );
        let wire = s[0].to_wire();
        assert_eq!(wire["criteria"]["true"], "time-sensitive");
        assert_eq!(wire["criteria"]["false"], "not");
    }

    #[test]
    fn choice_options_become_a_criteria_map_with_nulls_preserved() {
        let s = specs(
            r#"[{"id":"dept","type":"choice","instructions":"Who?",
                 "options":[{"name":"billing","description":"money"},{"name":"sales"}]}]"#,
        );
        let wire = s[0].to_wire();
        assert_eq!(wire["criteria"]["billing"], "money");
        assert_eq!(wire["criteria"]["sales"], Value::Null);
    }

    #[test]
    fn score_levels_become_an_ordered_array() {
        let s =
            specs(r#"[{"id":"f","type":"score","instructions":"How?","levels":["Calm","Angry"]}]"#);
        let wire = s[0].to_wire();
        assert_eq!(wire["criteria"], json!(["Calm", "Angry"]));
    }

    #[test]
    fn structured_entries_pass_through_untouched() {
        let s = specs(
            r#"[{"id":"q","type":"noul",
                 "instructions":{"question":"ok?","focus":"tone"},
                 "criteria":{"true":{"what":"yes","examples":["a"]},"false":null}}]"#,
        );
        let wire = s[0].to_wire();
        assert_eq!(wire["instructions"]["focus"], "tone");
        assert_eq!(wire["criteria"]["true"]["examples"][0], "a");
    }

    #[test]
    fn degenerate_questions_are_rejected_before_any_request() {
        assert!(parse_questions("[]").is_err());
        assert!(parse_questions(
            r#"[{"id":"c","type":"choice","instructions":"x","options":[{"name":"only"}]}]"#
        )
        .is_err());
        assert!(parse_questions(
            r#"[{"id":"s","type":"score","instructions":"x","levels":["one"]}]"#
        )
        .is_err());
    }

    #[test]
    fn retargeting_preserves_the_original_instructions() {
        let mut s = specs(r#"[{"id":"pays","type":"noul","instructions":"Payment obligation?"}]"#)
            .pop()
            .unwrap();
        s.retarget_to_line("7");
        let wire = s.to_wire();
        assert_eq!(wire["instructions"]["question"], "Payment obligation?");
        assert_eq!(wire["instructions"]["evaluate_line_id"], "7");
        // The question's own type and criteria are untouched.
        assert_eq!(wire["type"], "noul");
    }

    #[test]
    fn retargeting_keeps_structured_instructions_intact() {
        let mut s = specs(
            r#"[{"id":"q","type":"choice","instructions":{"ask":"Which?","focus":"tone"},
                 "options":[{"name":"a"},{"name":"b"}]}]"#,
        )
        .pop()
        .unwrap();
        s.retarget_to_line("2");
        let wire = s.to_wire();
        assert_eq!(wire["instructions"]["question"]["focus"], "tone");
        assert_eq!(wire["instructions"]["evaluate_line_id"], "2");
        assert_eq!(wire["criteria"]["a"], Value::Null);
    }

    #[test]
    fn request_body_matches_the_documented_shape() {
        let s = specs(r#"[{"id":"urgent","type":"noul","instructions":"Urgent?"}]"#);
        let body = build_request(&json!("Help!"), "jev-latest", &s);
        assert_eq!(body["state"], "Help!");
        assert_eq!(body["model"], "jev-latest");
        assert_eq!(body["questions"]["urgent"]["type"], "noul");
    }

    #[test]
    fn documented_response_parses_into_answers_and_usage() {
        let body = r#"{"model":"jev-1.13.0",
            "answers":{"is_urgent":{"type":"noul","noul":0.92}},
            "usage":{"input_tokens":312,"output_tokens":48}}"#;
        let ev = parse_evaluation(body, 7).expect("parse");
        assert_eq!(ev.model, "jev-1.13.0");
        assert_eq!(ev.answer("is_urgent").unwrap()["noul"], 0.92);
        assert_eq!(ev.usage.input_tokens, Some(312));
        assert_eq!(ev.usage.output_tokens, Some(48));
        assert_eq!(ev.usage.latency_ms, Some(7));
    }

    #[test]
    fn missing_usage_block_is_not_an_error() {
        let ev = parse_evaluation(r#"{"model":"m","answers":{}}"#, 1).expect("parse");
        assert_eq!(ev.usage.input_tokens, None);
        assert_eq!(ev.usage.latency_ms, Some(1));
    }

    #[test]
    fn only_rate_limit_and_transient_statuses_retry() {
        assert!(is_retryable(StatusCode::from_u16(429).unwrap()));
        assert!(is_retryable(StatusCode::from_u16(529).unwrap()));
        assert!(is_retryable(StatusCode::from_u16(503).unwrap()));
        assert!(!is_retryable(StatusCode::from_u16(401).unwrap()));
        assert!(!is_retryable(StatusCode::from_u16(422).unwrap()));
    }

    #[test]
    fn backoff_grows_and_honors_retry_after() {
        assert!(backoff_delay(3, None) > backoff_delay(0, None));
        // The cap is a real ceiling at every attempt -- jitter must never
        // push a sleep past it.
        for attempt in 0..20 {
            let d = backoff_delay(attempt, None);
            assert!(
                d <= Duration::from_millis(MAX_BACKOFF_MS),
                "attempt {attempt} slept {d:?}, over the cap"
            );
            assert!(
                d >= Duration::from_millis(375),
                "attempt {attempt} barely slept"
            );
        }
        assert_eq!(
            backoff_delay(0, Some(Duration::from_secs(2))),
            Duration::from_secs(2)
        );
        // A server asking for an absurd wait is still bounded.
        assert_eq!(
            backoff_delay(0, Some(Duration::from_secs(3600))),
            Duration::from_secs(60)
        );
    }
}
