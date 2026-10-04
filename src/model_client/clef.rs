//! Cloudflare Clef: an alternative backend for the System One evaluation layer.
//!
//! Clef (`@cf/cloudflare/clef`, 27B) and Clef-flash (`@cf/cloudflare/clef-flash`,
//! 9B) are open-weight decision models hosted on Workers AI. They take the same
//! System One request as TypeSafe's Jev -- one `state` plus a map of typed
//! `noul` / `choice` / `score` `questions` -- and return one typed answer per
//! question. So everything above the transport (question specs, the column
//! plan, the per-line fan-out) is shared with [`super::typesafe`]; only three
//! things differ, and they all live here:
//!
//!   * **Where the request goes.** Workers AI addresses the model in the path
//!     (`/accounts/{account}/ai/run/{model}`), so the body carries no `model`.
//!   * **How it is authenticated.** A Cloudflare API token, scoped to an account.
//!   * **How the response is wrapped.** Cloudflare's v4 envelope,
//!     `{ success, errors, result: { model, answers, usage } }`, where `success`
//!     can be `false` on an HTTP 200.
//!
//! Configuration (env):
//!   * `CLOUDFLARE_ACCOUNT_ID` -- the account that owns the Workers AI usage (required).
//!   * `CLOUDFLARE_API_TOKEN`  -- API token with Workers AI permission (required);
//!     `CLOUDFLARE_API_KEY` is accepted as a fallback name.
//!   * `CLOUDFLARE_AI_GATEWAY_ID` -- route through this AI Gateway instead of
//!     calling Workers AI directly (optional).
//!   * `CLOUDFLARE_API_BASE_URL` -- API root override for proxies and tests;
//!     default `https://api.cloudflare.com/client/v4`. Ignored when a gateway is set.

use serde_json::{json, Map, Value};

use super::typesafe::{Evaluation, QuestionSpec};
use super::{ModelClientError, Usage};

/// The 27B Clef model, Cloudflare's default.
pub const DEFAULT_CLEF_MODEL: &str = "@cf/cloudflare/clef";

/// Workers AI namespace the Clef family is published under. A bare `clef` or
/// `clef-flash` is resolved into it, so callers can use the short names.
const CLEF_NAMESPACE: &str = "@cf/cloudflare/";

/// Expand a short model name (`clef`, `clef-flash`) into its Workers AI id.
/// Anything already namespaced (`@cf/...`, `@hf/...`) passes through.
pub fn resolve_model(model: &str) -> String {
    if model.starts_with('@') {
        model.to_string()
    } else {
        format!("{CLEF_NAMESPACE}{model}")
    }
}

fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

pub fn account_id() -> Option<String> {
    env_nonempty("CLOUDFLARE_ACCOUNT_ID")
}

/// Bearer token. Empty when unset -- Cloudflare answers 401/403, which the
/// caller surfaces per row rather than panicking the whole frame.
pub fn api_token() -> String {
    env_nonempty("CLOUDFLARE_API_TOKEN")
        .or_else(|| env_nonempty("CLOUDFLARE_API_KEY"))
        .unwrap_or_default()
}

fn api_base() -> String {
    env_nonempty("CLOUDFLARE_API_BASE_URL")
        .unwrap_or_else(|| "https://api.cloudflare.com/client/v4".to_string())
}

/// `POST` target for one model, direct or through AI Gateway.
pub fn evaluation_endpoint(account: &str, model: &str) -> String {
    let model = resolve_model(model);
    match env_nonempty("CLOUDFLARE_AI_GATEWAY_ID") {
        Some(gateway) => {
            format!("https://gateway.ai.cloudflare.com/v1/{account}/{gateway}/workers-ai/{model}")
        }
        None => format!(
            "{}/accounts/{account}/ai/run/{model}",
            api_base().trim_end_matches('/')
        ),
    }
}

/// Fail before any request is billed when the account is not configured: the
/// endpoint cannot even be addressed without it.
pub fn require_account() -> Result<String, String> {
    account_id().ok_or_else(|| {
        "Clef requires CLOUDFLARE_ACCOUNT_ID (and CLOUDFLARE_API_TOKEN) to be set".to_string()
    })
}

/// Request body: the System One payload minus `model`, which is in the path.
pub fn build_request(state: &Value, questions: &[QuestionSpec]) -> Value {
    let mut map = Map::new();
    for spec in questions {
        map.insert(spec.id.clone(), spec.to_wire());
    }
    json!({ "state": state, "questions": Value::Object(map) })
}

/// Pull Cloudflare's own reason out of a v4 envelope's `errors` array, so a
/// failure reads "Authentication error" rather than a raw JSON dump.
pub fn envelope_error(body: &str) -> Option<String> {
    let v: Value = serde_json::from_str(body).ok()?;
    let messages: Vec<String> = v
        .get("errors")?
        .as_array()?
        .iter()
        .filter_map(
            |e| match (e.get("code"), e.get("message").and_then(Value::as_str)) {
                (Some(code), Some(msg)) if !code.is_null() => Some(format!("{msg} (code {code})")),
                (_, Some(msg)) => Some(msg.to_string()),
                _ => None,
            },
        )
        .collect();
    (!messages.is_empty()).then(|| messages.join("; "))
}

/// Bring a Clef answer into the shape the column plan reads, which is Jev's.
///
/// Clef is documented as Jev-compatible, so on the wire this is normally a
/// no-op. It exists so that the field names that do vary between
/// implementations still land in the right column rather than silently
/// becoming null:
///   * a noul probability reported as `probability` is copied to `noul`;
///   * score probabilities reported as an array (ordered like the levels) are
///     re-keyed by level index, which is how Jev keys them.
fn normalize_answer(answer: &mut Value) {
    let Some(obj) = answer.as_object_mut() else {
        return;
    };
    if obj.get("type").and_then(Value::as_str) == Some("noul") && !obj.contains_key("noul") {
        if let Some(p) = obj.get("probability").cloned() {
            obj.insert("noul".to_string(), p);
        }
    }
    if let Some(Value::Array(probs)) = obj.get("probabilities") {
        let keyed: Map<String, Value> = probs
            .iter()
            .enumerate()
            .map(|(i, p)| (i.to_string(), p.clone()))
            .collect();
        obj.insert("probabilities".to_string(), Value::Object(keyed));
    }
}

fn token_count(usage: Option<&Value>, names: &[&str]) -> Option<i64> {
    let usage = usage?;
    names
        .iter()
        .find_map(|n| usage.get(*n).and_then(Value::as_i64))
}

/// Parse a successful HTTP response. `requested_model` stands in for `_model`
/// when the result does not name the model that ran.
pub fn parse_evaluation(
    body: &str,
    requested_model: &str,
    latency_ms: i64,
) -> Result<Evaluation, ModelClientError> {
    let v: Value = serde_json::from_str(body)?;

    // Cloudflare can report failure inside a 200.
    if v.get("success").and_then(Value::as_bool) == Some(false) {
        let reason = envelope_error(body).unwrap_or_else(|| body.to_string());
        return Err(ModelClientError::ParseError(format!(
            "Clef request failed: {reason}"
        )));
    }

    // The v4 envelope wraps the output in `result`; a gateway or proxy that
    // has already unwrapped it is accepted as-is.
    let result = match v.get("result") {
        Some(r) if r.is_object() => r,
        _ => &v,
    };

    let mut answers = result
        .get("answers")
        .and_then(Value::as_object)
        .cloned()
        .ok_or_else(|| {
            ModelClientError::ParseError("Clef response has no `answers` object".to_string())
        })?;
    for answer in answers.values_mut() {
        normalize_answer(answer);
    }

    let model = result
        .get("model")
        .and_then(Value::as_str)
        .filter(|m| !m.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| resolve_model(requested_model));

    // Usage is best-effort. Workers AI models variously report the System One
    // names or the OpenAI-style ones, so both are read.
    let usage_block = result.get("usage");
    let usage = Usage {
        input_tokens: token_count(usage_block, &["input_tokens", "prompt_tokens"]),
        output_tokens: token_count(usage_block, &["output_tokens", "completion_tokens"]),
        cached_tokens: None,
        latency_ms: Some(latency_ms),
    };

    Ok(Evaluation {
        model,
        answers,
        usage,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_client::typesafe::parse_questions;

    #[test]
    fn short_names_resolve_into_the_cloudflare_namespace() {
        assert_eq!(resolve_model("clef"), "@cf/cloudflare/clef");
        assert_eq!(resolve_model("clef-flash"), "@cf/cloudflare/clef-flash");
        assert_eq!(resolve_model("@cf/cloudflare/clef"), "@cf/cloudflare/clef");
    }

    #[test]
    fn request_body_carries_no_model() {
        let s =
            parse_questions(r#"[{"id":"urgent","type":"noul","instructions":"Urgent?"}]"#).unwrap();
        let body = build_request(&json!("Help!"), &s);
        assert_eq!(body["state"], "Help!");
        assert_eq!(body["questions"]["urgent"]["type"], "noul");
        assert!(body.get("model").is_none());
    }

    #[test]
    fn envelope_unwraps_into_answers_and_usage() {
        let body = r#"{"success":true,"errors":[],"messages":[],
            "result":{"model":"@cf/cloudflare/clef",
                      "answers":{"u":{"type":"noul","noul":0.9}},
                      "usage":{"input_tokens":222,"output_tokens":3}}}"#;
        let ev = parse_evaluation(body, "clef", 5).unwrap();
        assert_eq!(ev.model, "@cf/cloudflare/clef");
        assert_eq!(ev.answer("u").unwrap()["noul"], 0.9);
        assert_eq!(ev.usage.input_tokens, Some(222));
        assert_eq!(ev.usage.output_tokens, Some(3));
        assert_eq!(ev.usage.latency_ms, Some(5));
    }

    #[test]
    fn missing_model_falls_back_to_the_requested_one() {
        let body = r#"{"success":true,"result":{"answers":{}}}"#;
        let ev = parse_evaluation(body, "clef-flash", 1).unwrap();
        assert_eq!(ev.model, "@cf/cloudflare/clef-flash");
    }

    #[test]
    fn openai_style_usage_names_are_read() {
        let body = r#"{"success":true,"result":{"answers":{},
            "usage":{"prompt_tokens":10,"completion_tokens":2}}}"#;
        let ev = parse_evaluation(body, "clef", 1).unwrap();
        assert_eq!(ev.usage.input_tokens, Some(10));
        assert_eq!(ev.usage.output_tokens, Some(2));
    }

    #[test]
    fn success_false_is_an_error_even_on_200() {
        let body = r#"{"success":false,"errors":[{"code":10000,"message":"Authentication error"}],
            "result":null}"#;
        let err = parse_evaluation(body, "clef", 1).unwrap_err().to_string();
        assert!(err.contains("Authentication error"), "{err}");
    }

    #[test]
    fn variant_answer_fields_are_normalized_to_the_jev_shape() {
        let body = r#"{"success":true,"result":{"answers":{
            "u":{"type":"noul","probability":0.3},
            "s":{"type":"score","score":1.2,"confidence":0.8,"probabilities":[0.1,0.6,0.3]}}}}"#;
        let ev = parse_evaluation(body, "clef", 1).unwrap();
        assert_eq!(ev.answer("u").unwrap()["noul"], 0.3);
        assert_eq!(ev.answer("s").unwrap()["probabilities"]["1"], 0.6);
    }

    #[test]
    fn envelope_errors_are_readable() {
        let body = r#"{"success":false,"errors":[{"code":7003,"message":"No route"}]}"#;
        assert_eq!(envelope_error(body).unwrap(), "No route (code 7003)");
        assert!(envelope_error("not json").is_none());
    }
}
