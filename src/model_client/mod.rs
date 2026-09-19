pub mod openai;
pub mod anthropic;
pub mod gemini;
pub mod groq;
pub mod bedrock;
pub mod typesafe;
pub mod streaming;

pub use streaming::{StreamEvent, stream_batch};

use reqwest::Client;
use std::error::Error;
use std::fmt;
use std::sync::LazyLock;
use std::time::{Duration, Instant};
use serde_json::Value;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use futures::StreamExt;

/// Shared HTTP client reused across all requests so connections are pooled
/// instead of being re-established for every batch.
static HTTP_CLIENT: LazyLock<Client> = LazyLock::new(|| {
    Client::builder()
        .timeout(Duration::from_secs(600))
        .connect_timeout(Duration::from_secs(30))
        .build()
        .unwrap_or_else(|_| Client::new())
});

/// Access the shared HTTP client.
pub fn http_client() -> &'static Client {
    &HTTP_CLIENT
}

/// Maximum number of concurrent in-flight requests per batch.
/// Override with the POLAR_LLAMA_MAX_CONCURRENCY environment variable.
fn max_concurrency() -> usize {
    std::env::var("POLAR_LLAMA_MAX_CONCURRENCY")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(64)
}

use crate::cache::CacheControl;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    /// Cache control marker for Anthropic/Bedrock
    /// Only serialized when present
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

/// Per-request token/latency accounting, captured when `usage=True` is
/// requested from Python. Every field is optional: a provider that omits its
/// usage block (or a parse failure) simply leaves the corresponding field
/// `None` -- this is never treated as an error (see issue #76).
#[derive(Debug, Default, Clone, Serialize)]
pub struct Usage {
    /// Total prompt tokens, INCLUDING any cached subset (normalized so
    /// `cached_tokens <= input_tokens` uniformly across providers).
    pub input_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
    /// Cached subset of `input_tokens`. `None` when the provider doesn't
    /// report a cache field at all; `Some(0)` when it does and reports no hit.
    pub cached_tokens: Option<i64>,
    /// Wall-clock request latency in milliseconds (request send through full
    /// body read). Always present on a successful call.
    pub latency_ms: Option<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum Provider {
    OpenAI,
    Anthropic,
    Gemini,
    Groq,
    Bedrock,
}

impl Provider {
    pub fn as_str(&self) -> &'static str {
        match self {
            Provider::OpenAI => "openai",
            Provider::Anthropic => "anthropic",
            Provider::Gemini => "gemini",
            Provider::Groq => "groq",
            Provider::Bedrock => "bedrock",
        }
    }
}

// Implement FromStr trait for Provider
impl FromStr for Provider {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "openai" => Ok(Provider::OpenAI),
            "anthropic" => Ok(Provider::Anthropic),
            "gemini" => Ok(Provider::Gemini),
            "groq" => Ok(Provider::Groq),
            "bedrock" => Ok(Provider::Bedrock),
            _ => Err(format!("Unknown provider: {s}")),
        }
    }
}

#[derive(Debug)]
pub enum ModelClientError {
    Http(u16, String),
    Serialization(serde_json::Error),
    RequestError(reqwest::Error),
    ParseError(String),
}

impl fmt::Display for ModelClientError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ModelClientError::Http(code, ref message) => write!(f, "HTTP Error {code}: {message}"),
            ModelClientError::Serialization(ref err) => write!(f, "Serialization Error: {err}"),
            ModelClientError::RequestError(ref err) => write!(f, "Request Error: {err}"),
            ModelClientError::ParseError(ref err) => write!(f, "Parse Error: {err}"),
        }
    }
}

impl Error for ModelClientError {}

impl From<reqwest::Error> for ModelClientError {
    fn from(err: reqwest::Error) -> Self {
        ModelClientError::RequestError(err)
    }
}

impl From<serde_json::Error> for ModelClientError {
    fn from(err: serde_json::Error) -> Self {
        ModelClientError::Serialization(err)
    }
}

#[async_trait]
pub trait ModelClient {
    /// Get the provider enum
    fn provider(&self) -> Provider;

    /// The name of the client provider
    fn provider_name(&self) -> &str {
        self.provider().as_str()
    }

    /// The API endpoint for the model
    fn api_endpoint(&self) -> String;

    /// The model name to use
    fn model_name(&self) -> &str;

    /// Format messages for the specific provider's API
    fn format_messages(&self, messages: &[Message]) -> Value;

    /// Parse the API response to extract the completion text
    fn parse_response(&self, response_text: &str) -> Result<String, ModelClientError>;

    /// Parse the same raw response body `parse_response` receives for
    /// token-usage accounting (issue #76). Default: no usage support.
    /// Providers override this to pull their usage block out of the raw
    /// JSON body via `serde_json::Value` pointer access -- deliberately NOT
    /// via the typed response structs, so a missing/renamed usage field can
    /// never break the main parse path. A parse failure here (malformed
    /// body, missing block) returns `None`, never an error: usage is
    /// best-effort, never load-bearing for the response text itself.
    fn parse_usage(&self, _response_text: &str) -> Option<Usage> {
        None
    }

    /// Attach provider-specific authentication to a request.
    /// Default: OpenAI-style Bearer token.
    fn apply_auth(&self, request: reqwest::RequestBuilder, api_key: &str) -> reqwest::RequestBuilder {
        request.bearer_auth(api_key)
    }

    /// Send a request to the API
    async fn send_request(&self, client: &Client, messages: &[Message]) -> Result<String, ModelClientError> {
        self.send_request_structured(client, messages, None, None).await
    }

    /// Send a request with structured output support. Thin wrapper over
    /// `send_request_structured_with_usage` that discards the captured
    /// `Usage` -- the canonical send lives there (see issue #76 design:
    /// making the usage-carrying method the default body, and this the
    /// wrapper, is the only way that cannot let the two paths desync, since
    /// provider overrides -- Anthropic's cache headers, Bedrock's AWS SDK
    /// call -- only need to be written once).
    async fn send_request_structured(
        &self,
        client: &Client,
        messages: &[Message],
        schema: Option<&str>,
        model_name: Option<&str>
    ) -> Result<String, ModelClientError> {
        self.send_request_structured_with_usage(client, messages, schema, model_name)
            .await
            .map(|(text, _usage)| text)
    }

    /// Canonical structured-output send: identical request/response handling
    /// to `send_request_structured`, plus wall-clock latency timing and a
    /// best-effort `parse_usage` call on success. Providers that need custom
    /// request shaping around the send (Anthropic's cache-beta headers,
    /// Bedrock's AWS SDK `converse()` call) override this method instead of
    /// `send_request_structured` directly.
    async fn send_request_structured_with_usage(
        &self,
        client: &Client,
        messages: &[Message],
        schema: Option<&str>,
        model_name: Option<&str>,
    ) -> Result<(String, Option<Usage>), ModelClientError> {
        let api_key = self.get_api_key();
        let body = self.format_request_body(messages, schema, model_name);

        let t0 = Instant::now();
        let response = self
            .apply_auth(client.post(self.api_endpoint()), &api_key)
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        let text = response.text().await?;
        let latency_ms = t0.elapsed().as_millis() as i64;

        if status.is_success() {
            let parsed = self.parse_response(&text)?;
            let mut usage = self.parse_usage(&text).unwrap_or_default();
            usage.latency_ms = Some(latency_ms);
            Ok((parsed, Some(usage)))
        } else {
            Err(ModelClientError::Http(status.as_u16(), text))
        }
    }

    /// Format the full request body including messages and model name
    fn format_request_body(&self, messages: &[Message], schema: Option<&str>, model_name: Option<&str>) -> Value {
        let formatted_messages = self.format_messages(messages);
        let mut body = serde_json::json!({
            "model": self.model_name(),
            "messages": formatted_messages
        });

        // Add structured output support based on provider
        if let Some(schema_str) = schema {
            if let Ok(schema_value) = serde_json::from_str::<Value>(schema_str) {
                match self.provider() {
                    Provider::OpenAI | Provider::Groq => {
                        // OpenAI and Groq use response_format with json_schema
                        body["response_format"] = serde_json::json!({
                            "type": "json_schema",
                            "json_schema": {
                                "name": model_name.unwrap_or("response"),
                                "strict": true,
                                "schema": schema_value
                            }
                        });
                    },
                    Provider::Anthropic => {
                        // Anthropic uses forced tool use for structured outputs
                        body["tools"] = serde_json::json!([{
                            "name": model_name.unwrap_or("response"),
                            "description": "Extract structured data according to the schema",
                            "input_schema": schema_value
                        }]);
                        body["tool_choice"] = serde_json::json!({
                            "type": "tool",
                            "name": model_name.unwrap_or("response")
                        });
                    },
                    _ => {
                        // For other providers, we'll validate post-response
                    }
                }
            }
        }

        body
    }

    /// Get the API key for this provider
    fn get_api_key(&self) -> String {
        match self.provider() {
            Provider::OpenAI => std::env::var("OPENAI_API_KEY").unwrap_or_default(),
            Provider::Anthropic => std::env::var("ANTHROPIC_API_KEY").unwrap_or_default(),
            Provider::Gemini => std::env::var("GEMINI_API_KEY").unwrap_or_default(),
            Provider::Groq => std::env::var("GROQ_API_KEY").unwrap_or_default(),
            Provider::Bedrock => String::new(), // Bedrock uses AWS credentials
        }
    }

    /// Streaming send: emits `StreamEvent`s for this row over `tx` as they
    /// arrive. Default implementation is a buffered fallback so providers
    /// without a native SSE override (Gemini, Bedrock) still work: it awaits
    /// the full response and emits a single `Delta` followed by `Done`.
    async fn send_request_streaming(
        &self,
        client: &Client,
        messages: &[Message],
        row: usize,
        tx: &tokio::sync::mpsc::Sender<(usize, streaming::StreamEvent)>,
    ) -> Result<(), ModelClientError> {
        let text = self.send_request(client, messages).await?;
        let _ = tx.send((row, streaming::StreamEvent::Delta(text))).await;
        let _ = tx.send((row, streaming::StreamEvent::Done)).await;
        Ok(())
    }
}

/// Trait for embedding providers
#[async_trait]
pub trait EmbeddingClient {
    /// Get the provider enum
    fn provider(&self) -> Provider;

    /// The name of the client provider
    fn provider_name(&self) -> &str {
        self.provider().as_str()
    }

    /// The API endpoint for embeddings
    fn embedding_endpoint(&self) -> String;

    /// The embedding model name to use
    fn embedding_model(&self) -> &str;

    /// Get the dimensions of the embedding vectors
    fn embedding_dimensions(&self) -> usize;

    /// Generate embeddings for a batch of texts
    async fn generate_embeddings(
        &self,
        client: &Client,
        texts: &[String],
    ) -> Result<Vec<Vec<f64>>, ModelClientError>;

    /// Get the API key for this provider
    fn get_api_key(&self) -> String {
        match self.provider() {
            Provider::OpenAI => std::env::var("OPENAI_API_KEY").unwrap_or_default(),
            Provider::Anthropic => std::env::var("ANTHROPIC_API_KEY").unwrap_or_default(),
            Provider::Gemini => std::env::var("GEMINI_API_KEY").unwrap_or_default(),
            Provider::Groq => std::env::var("GROQ_API_KEY").unwrap_or_default(),
            Provider::Bedrock => String::new(), // Bedrock uses AWS credentials
        }
    }
}

/// A JSON schema compiled once per batch, instead of once per row.
enum SchemaCheck {
    None,
    Valid(jsonschema::Validator),
    Invalid(String),
}

impl SchemaCheck {
    fn compile(schema: Option<&str>) -> Self {
        match schema {
            None => SchemaCheck::None,
            Some(schema_str) => {
                let schema_value: Value = match serde_json::from_str(schema_str) {
                    Ok(v) => v,
                    Err(e) => return SchemaCheck::Invalid(format!("Failed to parse schema: {e}")),
                };
                match jsonschema::validator_for(&schema_value) {
                    Ok(validator) => SchemaCheck::Valid(validator),
                    Err(e) => SchemaCheck::Invalid(format!("Failed to compile schema: {e}")),
                }
            }
        }
    }

    /// Validate a response against the compiled schema.
    fn validate(&self, response: &str) -> Result<(), String> {
        let validator = match self {
            SchemaCheck::None => return Ok(()),
            SchemaCheck::Invalid(err) => return Err(err.clone()),
            SchemaCheck::Valid(v) => v,
        };

        let response_value: Value = serde_json::from_str(response)
            .map_err(|e| format!("Failed to parse response as JSON: {e}"))?;

        let errors: Vec<String> = validator
            .iter_errors(&response_value)
            .map(|e| format!("{} at {}", e, e.instance_path()))
            .collect();

        if errors.is_empty() {
            Ok(())
        } else {
            Err(format!("Schema validation failed: {}", errors.join("; ")))
        }
    }
}

/// Validate JSON response against a JSON schema
pub fn validate_json_schema(response: &str, schema_str: &str) -> Result<(), String> {
    SchemaCheck::compile(Some(schema_str)).validate(response)
}

/// Create an error response object
pub fn create_error_response(error_type: &str, details: &str, raw: Option<&str>) -> String {
    let error_obj = if let Some(raw_content) = raw {
        serde_json::json!({
            "_error": error_type,
            "_details": details,
            "_raw": raw_content
        })
    } else {
        serde_json::json!({
            "_error": error_type,
            "_details": details
        })
    };
    serde_json::to_string(&error_obj).unwrap_or_else(|_| format!(r#"{{"_error": "{}"}}"#, error_type))
}

/// Build the `usage=True` JSON envelope: `{"response": ..., "usage": {...}}`.
///
/// `is_json_response` selects how `response` is embedded: `true` embeds it as
/// a parsed JSON *value* (structured-output success, or any
/// `create_error_response` object -- both are already JSON strings, so a
/// second `json_decode` on the Python side would double-encode them);
/// `false` embeds it as a JSON *string* (plain-text success, no schema). A
/// `response` that fails to parse as JSON despite `is_json_response=true`
/// falls back to embedding it as a string rather than dropping the row.
///
/// `usage: None` (provider omitted usage, or `send_request_structured` isn't
/// on the usage-aware path) serializes every usage field as JSON `null` --
/// never an error; `json_decode(dtype=...)` on the Python side fills nulls
/// naturally.
pub fn wrap_usage_envelope(response: &str, is_json_response: bool, usage: Option<&Usage>) -> String {
    let response_value: Value = if is_json_response {
        serde_json::from_str(response).unwrap_or_else(|_| Value::String(response.to_string()))
    } else {
        Value::String(response.to_string())
    };
    let usage_value = match usage {
        Some(u) => serde_json::to_value(u).unwrap_or(Value::Null),
        None => serde_json::json!({
            "input_tokens": Value::Null,
            "output_tokens": Value::Null,
            "cached_tokens": Value::Null,
            "latency_ms": Value::Null,
        }),
    };
    serde_json::json!({"response": response_value, "usage": usage_value}).to_string()
}

/// Create a client for the given provider and model
pub fn create_client(provider: Provider, model: &str) -> Box<dyn ModelClient + Send + Sync> {
    match provider {
        Provider::OpenAI => Box::new(openai::OpenAIClient::new_with_model(model)),
        Provider::Anthropic => Box::new(anthropic::AnthropicClient::new_with_model(model)),
        Provider::Gemini => Box::new(gemini::GeminiClient::new_with_model(model)),
        Provider::Groq => Box::new(groq::GroqClient::new_with_model(model)),
        Provider::Bedrock => Box::new(bedrock::BedrockClient::new_with_model(model)),
    }
}

/// Core batch runner: sends all requests concurrently (bounded by
/// `max_concurrency`) over the shared HTTP client, preserving input order.
///
/// When `errors_as_json` is true, request errors are surfaced as structured
/// error JSON objects; otherwise they map to `None`.
async fn run_one<T: ModelClient + Sync + ?Sized>(
    client: &T,
    messages: &[Message],
    schema: Option<&str>,
    model_name: Option<&str>,
    schema_check: &SchemaCheck,
    errors_as_json: bool,
    with_usage: bool,
) -> Option<String> {
    // Always send via the usage-aware path -- for `with_usage=false` callers
    // the captured `Usage` is simply discarded below, so the response text
    // (and therefore output) is byte-identical to the pre-#76 code path; the
    // only added cost is one `Instant::now()`/`elapsed()` pair and a
    // best-effort `parse_usage` call, never a second network request.
    let has_schema = schema.is_some();
    match client.send_request_structured_with_usage(http_client(), messages, schema, model_name).await {
        Ok((response, usage)) => {
            match schema_check.validate(&response) {
                Ok(()) => {
                    if with_usage {
                        Some(wrap_usage_envelope(&response, has_schema, usage.as_ref()))
                    } else {
                        Some(response)
                    }
                },
                Err(validation_error) => {
                    let err = create_error_response(
                        "validation_failed",
                        &validation_error,
                        Some(&response),
                    );
                    if with_usage {
                        Some(wrap_usage_envelope(&err, true, usage.as_ref()))
                    } else {
                        Some(err)
                    }
                }
            }
        },
        Err(e) => {
            eprintln!("Error fetching from {}: {}", client.provider_name(), e);
            if errors_as_json {
                let err = create_error_response("api_error", &e.to_string(), None);
                if with_usage {
                    Some(wrap_usage_envelope(&err, true, None))
                } else {
                    Some(err)
                }
            } else {
                None
            }
        }
    }
}

async fn run_batch<T: ModelClient + Sync + ?Sized>(
    client: &T,
    message_arrays: &[Vec<Message>],
    schema: Option<&str>,
    model_name: Option<&str>,
    errors_as_json: bool,
    with_usage: bool,
) -> Vec<Option<String>> {
    let schema_check = SchemaCheck::compile(schema);

    // Collect the futures eagerly so the stream holds concrete future values;
    // requests still only run when polled, bounded by `buffered`.
    let requests: Vec<_> = message_arrays
        .iter()
        .map(|messages| run_one(client, messages, schema, model_name, &schema_check, errors_as_json, with_usage))
        .collect();

    futures::stream::iter(requests)
        .buffered(max_concurrency())
        .collect()
        .await
}

fn to_user_message_arrays(messages: &[String]) -> Vec<Vec<Message>> {
    messages
        .iter()
        .map(|content| {
            vec![Message {
                role: "user".to_string(),
                content: content.clone(),
                cache_control: None,
            }]
        })
        .collect()
}

/// The main function to fetch data from model providers
pub async fn fetch_data_generic<T: ModelClient + Sync + ?Sized>(
    client: &T,
    messages: &[String]
) -> Vec<Option<String>> {
    let message_arrays = to_user_message_arrays(messages);
    run_batch(client, &message_arrays, None, None, false, false).await
}

/// Fetch data with structured output support and validation
pub async fn fetch_data_generic_with_schema<T: ModelClient + Sync + ?Sized>(
    client: &T,
    messages: &[String],
    schema: Option<&str>,
    model_name: Option<&str>
) -> Vec<Option<String>> {
    let message_arrays = to_user_message_arrays(messages);
    run_batch(client, &message_arrays, schema, model_name, true, false).await
}

/// Enhanced function to fetch data that supports either single messages or arrays of messages
pub async fn fetch_data_generic_enhanced<T: ModelClient + Sync + ?Sized>(
    client: &T,
    message_arrays: &[Vec<Message>]
) -> Vec<Option<String>> {
    run_batch(client, message_arrays, None, None, false, false).await
}

/// Enhanced function with schema validation for message arrays
pub async fn fetch_data_generic_enhanced_with_schema<T: ModelClient + Sync + ?Sized>(
    client: &T,
    message_arrays: &[Vec<Message>],
    schema: Option<&str>,
    model_name: Option<&str>
) -> Vec<Option<String>> {
    run_batch(client, message_arrays, schema, model_name, true, false).await
}

/// Fetch data (single-message-per-row) with the full `usage=True` envelope
/// option threaded through (issue #76). `errors_as_json` mirrors the
/// existing schema/no-schema pairing (`Some(schema)` => `true`) so behavior
/// for `with_usage=false` is identical to calling the pre-#76 functions above.
pub async fn fetch_data_generic_with_options<T: ModelClient + Sync + ?Sized>(
    client: &T,
    messages: &[String],
    schema: Option<&str>,
    model_name: Option<&str>,
    with_usage: bool,
) -> Vec<Option<String>> {
    let message_arrays = to_user_message_arrays(messages);
    run_batch(client, &message_arrays, schema, model_name, schema.is_some(), with_usage).await
}

/// Enhanced (message-array) fetch with the `usage=True` envelope option
/// threaded through (issue #76). See `fetch_data_generic_with_options`.
pub async fn fetch_data_generic_enhanced_with_options<T: ModelClient + Sync + ?Sized>(
    client: &T,
    message_arrays: &[Vec<Message>],
    schema: Option<&str>,
    model_name: Option<&str>,
    with_usage: bool,
) -> Vec<Option<String>> {
    run_batch(client, message_arrays, schema, model_name, schema.is_some(), with_usage).await
}

/// Example function showing how to use the different model clients with specific models
pub async fn example_usage(messages: &[String], provider_str: &str, model: &str) -> Vec<Option<String>> {
    let provider = Provider::from_str(provider_str).unwrap_or(Provider::OpenAI);
    let client = create_client(provider, model);
    fetch_data_generic(&*client, messages).await
}

/// Enhanced example function supporting message arrays
pub async fn example_usage_enhanced(
    message_arrays: &[Vec<Message>],
    provider_str: &str,
    model: &str
) -> Vec<Option<String>> {
    let provider = Provider::from_str(provider_str).unwrap_or(Provider::OpenAI);
    let client = create_client(provider, model);
    fetch_data_generic_enhanced(&*client, message_arrays).await
}

async fn embed_one<T: EmbeddingClient + Sync + ?Sized>(
    client: &T,
    text: &String,
) -> Option<Vec<f64>> {
    match client.generate_embeddings(http_client(), std::slice::from_ref(text)).await {
        Ok(embeddings) => embeddings.into_iter().next(),
        Err(e) => {
            eprintln!("Error generating embedding from {}: {}", client.provider_name(), e);
            None
        }
    }
}

/// Parallel embedding generation function with bounded concurrency
pub async fn fetch_embeddings_generic<T: EmbeddingClient + Sync + ?Sized>(
    client: &T,
    texts: &[String]
) -> Vec<Option<Vec<f64>>> {
    let requests: Vec<_> = texts.iter().map(|text| embed_one(client, text)).collect();

    futures::stream::iter(requests)
        .buffered(max_concurrency())
        .collect()
        .await
}

/// Create an embedding client for the given provider and model
pub fn create_embedding_client(provider: Provider, model: &str) -> Box<dyn EmbeddingClient + Send + Sync> {
    match provider {
        Provider::OpenAI => Box::new(openai::OpenAIEmbeddingClient::new_with_model(model)),
        // Other providers can be added here as they're implemented
        _ => Box::new(openai::OpenAIEmbeddingClient::new_with_model(model)),
    }
}

#[cfg(test)]
mod usage_tests {
    use super::*;
    use crate::model_client::anthropic::AnthropicClient;
    use crate::model_client::gemini::GeminiClient;
    use crate::model_client::groq::GroqClient;
    use crate::model_client::openai::OpenAIClient;

    // -- parse_usage: canned real-response-shaped bodies -------------------

    #[test]
    fn openai_parse_usage_extracts_prompt_completion_and_cached() {
        let client = OpenAIClient::new_with_model("gpt-4o-mini");
        let body = r#"{"id":"x","model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1200,"completion_tokens":34,"prompt_tokens_details":{"cached_tokens":1000}}}"#;
        let usage = client.parse_usage(body).expect("usage present");
        assert_eq!(usage.input_tokens, Some(1200));
        assert_eq!(usage.output_tokens, Some(34));
        assert_eq!(usage.cached_tokens, Some(1000));
    }

    #[test]
    fn openai_parse_usage_missing_block_returns_none() {
        let client = OpenAIClient::new_with_model("gpt-4o-mini");
        let body = r#"{"id":"x","model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}"#;
        assert!(client.parse_usage(body).is_none());
    }

    #[test]
    fn groq_parse_usage_matches_openai_shape() {
        let client = GroqClient::new_with_model("llama-3.3-70b-versatile");
        let body = r#"{"id":"x","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":50,"completion_tokens":10}}"#;
        let usage = client.parse_usage(body).expect("usage present");
        assert_eq!(usage.input_tokens, Some(50));
        assert_eq!(usage.output_tokens, Some(10));
        assert_eq!(usage.cached_tokens, None); // absent prompt_tokens_details -> None, never an error
    }

    #[test]
    fn anthropic_parse_usage_normalizes_cache_into_input_tokens() {
        let client = AnthropicClient::new_with_model("claude-haiku-4-5");
        // Anthropic's own `input_tokens` EXCLUDES cache reads/writes -- verify
        // the normalization folds both into `input_tokens` so
        // `cached_tokens <= input_tokens` uniformly (see anthropic.rs::parse_usage doc).
        let body = r#"{"id":"x","model":"m","content":[{"type":"text","text":"hi"}],"usage":{"input_tokens":100,"output_tokens":20,"cache_read_input_tokens":500,"cache_creation_input_tokens":50}}"#;
        let usage = client.parse_usage(body).expect("usage present");
        assert_eq!(usage.input_tokens, Some(100 + 500 + 50));
        assert_eq!(usage.output_tokens, Some(20));
        assert_eq!(usage.cached_tokens, Some(500));
    }

    #[test]
    fn anthropic_parse_usage_no_cache_fields_defaults_to_zero() {
        let client = AnthropicClient::new_with_model("claude-haiku-4-5");
        let body = r#"{"id":"x","model":"m","content":[{"type":"text","text":"hi"}],"usage":{"input_tokens":100,"output_tokens":20}}"#;
        let usage = client.parse_usage(body).expect("usage present");
        assert_eq!(usage.input_tokens, Some(100));
        assert_eq!(usage.cached_tokens, Some(0));
    }

    #[test]
    fn gemini_parse_usage_extracts_prompt_and_candidate_counts() {
        let client = GeminiClient::new_with_model("gemini-2.5-flash");
        let body = r#"{"candidates":[{"content":{"parts":[{"text":"hi"}],"role":"model"}}],"usageMetadata":{"promptTokenCount":300,"candidatesTokenCount":40,"cachedContentTokenCount":100}}"#;
        let usage = client.parse_usage(body).expect("usage present");
        assert_eq!(usage.input_tokens, Some(300));
        assert_eq!(usage.output_tokens, Some(40));
        assert_eq!(usage.cached_tokens, Some(100));
    }

    // -- envelope shape ------------------------------------------------------

    #[test]
    fn envelope_wraps_plain_text_response_as_json_string() {
        let usage = Usage {
            input_tokens: Some(10),
            output_tokens: Some(5),
            cached_tokens: None,
            latency_ms: Some(42),
        };
        let env = wrap_usage_envelope("hello world", false, Some(&usage));
        let parsed: Value = serde_json::from_str(&env).unwrap();
        assert_eq!(parsed["response"], Value::String("hello world".to_string()));
        assert_eq!(parsed["usage"]["input_tokens"], 10);
        assert_eq!(parsed["usage"]["output_tokens"], 5);
        assert_eq!(parsed["usage"]["cached_tokens"], Value::Null);
        assert_eq!(parsed["usage"]["latency_ms"], 42);
    }

    #[test]
    fn envelope_embeds_structured_response_as_json_value_not_string() {
        let usage = Usage {
            input_tokens: Some(1),
            output_tokens: Some(1),
            cached_tokens: Some(0),
            latency_ms: Some(1),
        };
        let env = wrap_usage_envelope(r#"{"value": 42}"#, true, Some(&usage));
        let parsed: Value = serde_json::from_str(&env).unwrap();
        // A single json_decode on the Python side must see an object, not a
        // JSON-encoded string of an object.
        assert!(parsed["response"].is_object());
        assert_eq!(parsed["response"]["value"], 42);
    }

    #[test]
    fn envelope_error_object_is_embedded_as_json_value() {
        let err = create_error_response("api_error", "boom", None);
        let env = wrap_usage_envelope(&err, true, None);
        let parsed: Value = serde_json::from_str(&env).unwrap();
        assert_eq!(parsed["response"]["_error"], "api_error");
        assert_eq!(parsed["usage"]["input_tokens"], Value::Null);
        assert_eq!(parsed["usage"]["latency_ms"], Value::Null);
    }

    #[test]
    fn envelope_missing_usage_is_all_null_never_an_error() {
        let env = wrap_usage_envelope("text", false, None);
        let parsed: Value = serde_json::from_str(&env).unwrap();
        assert_eq!(parsed["usage"]["input_tokens"], Value::Null);
        assert_eq!(parsed["usage"]["output_tokens"], Value::Null);
        assert_eq!(parsed["usage"]["cached_tokens"], Value::Null);
        assert_eq!(parsed["usage"]["latency_ms"], Value::Null);
    }

    #[test]
    fn envelope_malformed_json_response_falls_back_to_string_embedding() {
        // is_json_response=true but the text isn't actually valid JSON: must
        // not panic, and must fall back to embedding it as a string.
        let env = wrap_usage_envelope("not json", true, None);
        let parsed: Value = serde_json::from_str(&env).unwrap();
        assert_eq!(parsed["response"], Value::String("not json".to_string()));
    }
}
