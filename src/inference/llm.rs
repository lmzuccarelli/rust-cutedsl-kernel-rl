use custom_logger as log;
use hyper::StatusCode;
use reqwest::Client;
use serde_derive::Deserialize;
use serde_derive::Serialize;
use serde_json::Value;
use std::error::Error as StdError;
use std::time::Duration;
use std::time::Instant;

// openapi schema

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnMaasResponse {
    pub content: Vec<Content>,
    pub id: String,
    pub model: String,
    pub role: String,
    #[serde(rename = "stop_reason")]
    pub stop_reason: String,
    #[serde(rename = "type")]
    pub type_field: String,
    pub usage: EnMaasUsage,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Content {
    pub signature: Option<String>,
    pub thinking: Option<String>,
    #[serde(rename = "type")]
    pub type_field: String,
    pub text: String,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnMaasUsage {
    #[serde(rename = "cache_creation_input_tokens")]
    pub cache_creation_input_tokens: i64,
    #[serde(rename = "cache_read_input_tokens")]
    pub cache_read_input_tokens: i64,
    #[serde(rename = "input_tokens")]
    pub input_tokens: i64,
    #[serde(rename = "output_tokens")]
    pub output_tokens: i64,
    #[serde(rename = "prompt_tokens_details")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenaiChatCompletions {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
    #[serde(rename = "service_tier")]
    pub service_tier: Option<String>,
    #[serde(rename = "system_fingerprint")]
    pub system_fingerprint: Value,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Choice {
    pub index: i64,
    pub message: Message,
    #[serde(rename = "finish_reason")]
    pub finish_reason: String,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Message {
    pub role: String,
    pub content: String,
    pub refusal: Option<Value>,
    pub annotations: Option<Vec<Value>>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
    #[serde(rename = "prompt_tokens")]
    pub prompt_tokens: Option<i64>,
    #[serde(rename = "completion_tokens")]
    pub completion_tokens: Option<i64>,
    #[serde(rename = "total_tokens")]
    pub total_tokens: Option<i64>,
    #[serde(rename = "prompt_tokens_details")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(rename = "completion_tokens_details")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptTokensDetails {
    #[serde(rename = "cached_tokens")]
    pub cached_tokens: i64,
    #[serde(rename = "audio_tokens")]
    pub audio_tokens: Option<i64>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompletionTokensDetails {
    #[serde(rename = "reasoning_tokens")]
    pub reasoning_tokens: i64,
    #[serde(rename = "audio_tokens")]
    pub audio_tokens: Option<i64>,
    #[serde(rename = "accepted_prediction_tokens")]
    pub accepted_prediction_tokens: i64,
    #[serde(rename = "rejected_prediction_tokens")]
    pub rejected_prediction_tokens: i64,
}

#[allow(unused)]
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RequestSchema {
    pub model: String,
    pub messages: Vec<RequestMessage>,
    // pub temperature: f64,
    // pub top_p: f64,
    pub stream: bool,
    pub max_tokens: i32,
    // pub chat_template_kwargs: Option<ChatTemplate>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatTemplate {
    enable_thinking: bool,
}

#[allow(unused)]
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RequestMessage {
    pub role: String,
    pub content: String,
}

/// Token budget for an OpenAPI compatible endpoint.
///
/// `max_output_tokens` is only an upper bound: the value actually sent is also
/// capped by the room left in the context window after the prompt is accounted
/// for. This avoids the hard failure that occurs when a large kernel + NCU
/// payload and a fixed `max_tokens` request cannot both fit.
#[derive(Debug, Clone)]
pub struct LlmTokenConfig {
    pub max_output_tokens: i32,
    pub context_window: i32,
    pub input_token_margin: i32,
    pub beta_header: Option<String>,
}

impl Default for LlmTokenConfig {
    fn default() -> Self {
        Self {
            max_output_tokens: 64_000,
            context_window: 200_000,
            input_token_margin: 2_000,
            beta_header: None,
        }
    }
}

// Below this there is not enough room to produce a useful kernel rewrite, so
// fail with an actionable message instead of letting the endpoint return an
// opaque 400.
const MIN_OUTPUT_TOKENS: i32 = 4_096;
// Some gateways advertise a lower output cap than the model actually supports.
const MAX_MAX_TOKENS_RETRIES: u8 = 2;
// A `status: None` reqwest failure means no HTTP response was received at all
// (DNS, TCP, TLS, connection reset, timeout). Those are worth retrying before
// giving up, unlike 4xx responses which are deterministic.
const MAX_TRANSPORT_RETRIES: u8 = 2;
// Exponential backoff base: attempt 1 waits 2s, attempt 2 waits 4s.
const TRANSPORT_RETRY_BASE_DELAY_SECS: u64 = 2;
// Streaming keeps bytes (deltas and pings) flowing, so a long silence means the
// connection is dead. A full 128k-token generation can exceed 20 minutes, so
// the overall cap is generous; the idle read timeout catches stalls early.
const STREAM_READ_TIMEOUT_SECS: u64 = 300;
const REQUEST_TOTAL_TIMEOUT_SECS: u64 = 3600;

// Measured ratio for large kernel + NCU payloads is ~2.6 chars/token (a
// 179k-char prompt was counted as ~69k input tokens by the endpoint). The old
// 4 chars/token estimate under-counted by ~55%, which can overrun a 200k
// window once a 128k output budget is requested. 2.5 chars/token deliberately
// over-counts instead, keeping the prompt + max_tokens total inside the window.
const CHARS_PER_TOKEN_NUMERATOR: i64 = 2;
const CHARS_PER_TOKEN_DENOMINATOR: i64 = 5;

/// Conservative token estimate for code, English and tabular NCU dumps.
/// Deliberately rounds up so the budget stays inside the context window.
fn estimate_tokens(text: &str) -> i64 {
    let chars = text.chars().count() as i64;
    let numerator = chars * CHARS_PER_TOKEN_NUMERATOR;
    (numerator + CHARS_PER_TOKEN_DENOMINATOR - 1) / CHARS_PER_TOKEN_DENOMINATOR
}

/// Render a reqwest error together with its full source chain and the
/// classification flags. `Display` on `reqwest::Error` often only says "error
/// sending request", hiding the useful cause (e.g. "connection reset by peer"
/// or "tls handshake eof").
fn describe_reqwest_error(err: &reqwest::Error) -> String {
    let mut message = format!(
        "{} (timeout={}, connect={}, request={}, body={})",
        err,
        err.is_timeout(),
        err.is_connect(),
        err.is_request(),
        err.is_body()
    );
    let mut source = StdError::source(err);
    let mut depth = 0;
    while let Some(cause) = source {
        message.push_str(&format!("; caused by: {}", cause));
        source = cause.source();
        depth += 1;
        if depth >= 8 {
            break;
        }
    }
    message
}

/// Incremental parser for Messages API server-sent-event stream.
///
/// Only `text_delta` content is collected, so thinking blocks (always present
/// on Opus 5.5) are skipped naturally. Bytes are fed as they arrive; partial
/// events are buffered until their terminating blank line is seen.
#[derive(Default, Debug)]
struct StreamAccumulator {
    // Raw bytes, not a String: a network chunk can end in the middle of a
    // multi-byte UTF-8 character, so decoding only happens per complete event.
    buffer: Vec<u8>,
    text: String,
    stop_reason: Option<String>,
    output_tokens: Option<i64>,
    done: bool,
}

impl StreamAccumulator {
    /// Feed a raw chunk. Returns `Err` with the API's message when the stream
    /// carries an `error` event (e.g. `overloaded_error` mid-generation).
    fn feed(&mut self, chunk: &[u8]) -> Result<(), String> {
        // CR only ever appears as part of CRLF line endings (JSON escapes it in
        // payloads), so dropping it normalises CRLF streams to LF.
        self.buffer.extend(chunk.iter().filter(|&&b| b != b'\r'));
        while let Some(pos) = self.buffer.windows(2).position(|w| w == b"\n\n") {
            let event: Vec<u8> = self.buffer.drain(..pos + 2).collect();
            self.handle_event(&String::from_utf8_lossy(&event))?;
        }
        Ok(())
    }

    fn handle_event(&mut self, event: &str) -> Result<(), String> {
        let data: String = event
            .lines()
            .filter_map(|line| line.strip_prefix("data:"))
            .map(str::trim_start)
            .collect::<Vec<_>>()
            .join("\n");
        if data.is_empty() {
            return Ok(());
        }
        let value: Value = match serde_json::from_str(&data) {
            Ok(value) => value,
            Err(e) => {
                log::warn!("[run] skipping unparsable stream event ({}): {}", e, data);
                return Ok(());
            }
        };
        match value["type"].as_str().unwrap_or_default() {
            "content_block_start" => {
                let block = &value["content_block"];
                if block["type"] == "text"
                    && let Some(text) = block["text"].as_str()
                {
                    self.text.push_str(text);
                }
            }
            "content_block_delta" => {
                let delta = &value["delta"];
                if delta["type"] == "text_delta"
                    && let Some(text) = delta["text"].as_str()
                {
                    self.text.push_str(text);
                }
            }
            "message_delta" => {
                if let Some(reason) = value["delta"]["stop_reason"].as_str() {
                    self.stop_reason = Some(reason.to_string());
                }
                if let Some(tokens) = value["usage"]["output_tokens"].as_i64() {
                    self.output_tokens = Some(tokens);
                }
            }
            "message_stop" => self.done = true,
            "error" => {
                return Err(format!(
                    "{}: {}",
                    value["error"]["type"].as_str().unwrap_or("error"),
                    value["error"]["message"].as_str().unwrap_or(&data)
                ));
            }
            // message_start, content_block_stop, ping and unknown future
            // event types carry nothing we need.
            _ => {}
        }
        Ok(())
    }
}

/// Sleep with exponential backoff before a transport retry. Returns `false`
/// once the retry budget is exhausted.
async fn transport_backoff(retries: &mut u8, detail: &str) -> bool {
    if *retries >= MAX_TRANSPORT_RETRIES {
        return false;
    }
    *retries += 1;
    let delay =
        Duration::from_secs(TRANSPORT_RETRY_BASE_DELAY_SECS * 2u64.pow(*retries as u32 - 1));
    log::warn!(
        "[run] llm openapi transport failure (retry {}/{}): {}; retrying in {:?}",
        retries,
        MAX_TRANSPORT_RETRIES,
        detail,
        delay
    );
    tokio::time::sleep(delay).await;
    true
}

/// True when the endpoint error is specifically about the requested output
/// budget, as opposed to the input prompt itself being too large.
fn is_max_tokens_error(body: &str) -> bool {
    let body = body.to_ascii_lowercase();
    body.contains("max_tokens")
        || body.contains("max output")
        || body.contains("max_completion_tokens")
        || body.contains("maximum allowed number of output tokens")
}

pub trait LlmInterfaceOpenApi {
    async fn run(
        prompt: String,
        url: String,
        token: String,
        model: String,
        response_type: String,
        token_config: LlmTokenConfig,
    ) -> Result<String, Box<dyn std::error::Error>>;
}

pub struct LlmOpenApi {}

impl LlmInterfaceOpenApi for LlmOpenApi {
    async fn run(
        prompt: String,
        url: String,
        token: String,
        model: String,
        response_type: String,
        token_config: LlmTokenConfig,
    ) -> Result<String, Box<dyn std::error::Error>> {
        log::debug!("[run] executing llm {} inference endpoint", response_type);
        log::debug!("[run] executing llm url {}", url);
        log::debug!("[run] executing llm model {}", model);

        let start = Instant::now();

        let input_tokens = estimate_tokens(&prompt);
        let context_window = token_config.context_window as i64;
        let margin = token_config.input_token_margin.max(0) as i64;
        let available = context_window - input_tokens - margin;
        log::info!(
            "[run] token budget: prompt chars {}, estimated input tokens {}, context window {}, safety margin {}, available output {}",
            prompt.len(),
            input_tokens,
            context_window,
            margin,
            available
        );

        if available < MIN_OUTPUT_TOKENS as i64 {
            return Err(Box::from(format!(
                "[run] prompt does not fit the configured context window: ~{} input tokens + {} safety margin > {} context window. Reduce the code/NCU payload, raise `llm_context_window`, or use a model/endpoint with a larger context window.",
                input_tokens, margin, context_window
            )));
        }

        // Never ask for more than the user configured, and never ask for more
        // than the space left in the context window. This is the key change vs.
        // the old hardcoded `max_tokens: 128000`.
        let requested_max_tokens = token_config.max_output_tokens.max(MIN_OUTPUT_TOKENS) as i64;
        let mut max_tokens = requested_max_tokens.min(available) as i32;
        if (max_tokens as i64) < requested_max_tokens {
            log::warn!(
                "[run] context window limits output to {} tokens (requested {}); a full kernel rewrite may be truncated. Reduce the prompt or raise `llm_context_window`.",
                max_tokens,
                requested_max_tokens
            );
        } else if max_tokens < 40_000 {
            log::warn!(
                "[run] only {} output tokens requested; a full kernel rewrite may be truncated. Raise `llm_max_output_tokens`.",
                max_tokens
            );
        }

        let system_msg = RequestMessage {
            role: "user".to_string(),
            content: "you are a cuda kernel expert, help the user with generating code, analyzing and profiling of cuda kernels".to_string(),
        };
        let user_msg = RequestMessage {
            role: "user".to_owned(),
            content: prompt,
        };
        let vec_msgs = vec![system_msg, user_msg];

        // TODO: check if kwargs is supported
        // This is used to suppress thinking in qwen3.8-27B
        let _ct = ChatTemplate {
            enable_thinking: false,
        };

        let client_res = Client::builder()
            .danger_accept_invalid_certs(true)
            .http1_title_case_headers()
            .timeout(Duration::from_secs(REQUEST_TOTAL_TIMEOUT_SECS))
            .read_timeout(Duration::from_secs(STREAM_READ_TIMEOUT_SECS))
            .build();

        // Anthropic-compatible endpoints are streamed: a non-streaming request
        // leaves the connection silent for the whole generation (7+ minutes for
        // a full kernel rewrite), which proxies drop as idle. The OpenAI-style
        // path keeps its existing non-streaming behaviour.
        let use_stream = response_type == "Enmaas";

        let client = match client_res {
            Ok(client) => {
                log::debug!("[run] llm openapi client created");
                client
            }
            Err(e) => {
                return Err(Box::from(format!("[run] llm openapi {} ", e)));
            }
        };

        let mut retries = 0u8;
        let mut transport_retries = 0u8;
        loop {
            let rs = RequestSchema {
                model: model.clone(),
                // temperature: 0.1,
                stream: use_stream,
                max_tokens,
                // top_p: 0.8,
                messages: vec_msgs.clone(),
                // chat_template_kwargs: Some(ct.clone()),
            };
            let json_request = serde_json::to_string(&rs)?;
            log::trace!(
                "[run] openapi json payload (max_tokens {}) {}",
                max_tokens,
                json_request
            );

            let mut request = client
                .post(url.as_str())
                .bearer_auth(token.trim())
                .header("Content-Type", "application/json")
                .header("X-Api-Key", token.trim());
            if let Some(ref beta) = token_config.beta_header {
                request = request.header("anthropic-beta", beta);
            }

            let mut result = match request.body(json_request).send().await {
                Ok(result) => result,
                Err(e) => {
                    let detail = describe_reqwest_error(&e);
                    let transient =
                        e.is_timeout() || e.is_connect() || e.is_request() || e.is_body();
                    if transient && transport_backoff(&mut transport_retries, &detail).await {
                        continue;
                    }
                    return Err(Box::from(format!(
                        "[run] llm openapi transport failure after {} attempt(s): {}",
                        transport_retries + 1,
                        detail,
                    )));
                }
            };

            let status = result.status();
            log::debug!("[run] llm openapi response status {}", status);

            if use_stream && status == StatusCode::OK {
                let mut acc = StreamAccumulator::default();
                let mut stream_failure: Option<String> = None;
                loop {
                    match result.chunk().await {
                        Ok(Some(chunk)) => {
                            if let Err(api_error) = acc.feed(&chunk) {
                                return Err(Box::from(format!(
                                    "[run] llm openapi stream error event: {}",
                                    api_error
                                )));
                            }
                            if acc.done {
                                break;
                            }
                        }
                        Ok(None) => break,
                        Err(e) => {
                            stream_failure = Some(describe_reqwest_error(&e));
                            break;
                        }
                    }
                }

                if !acc.done {
                    // Connection died mid-generation: nothing usable came back,
                    // so this is retried like any other transport failure.
                    let detail = stream_failure
                        .unwrap_or_else(|| "stream closed before message_stop".to_string());
                    let detail = format!("{} (after {} chars of output)", detail, acc.text.len());
                    if transport_backoff(&mut transport_retries, &detail).await {
                        continue;
                    }
                    return Err(Box::from(format!(
                        "[run] llm openapi stream interrupted after {} attempt(s): {}",
                        transport_retries + 1,
                        detail,
                    )));
                }

                let stop_reason = acc.stop_reason.as_deref().unwrap_or("unknown");
                if stop_reason == "max_tokens" {
                    log::warn!(
                        "[run] response hit max_tokens {}; generated code is likely truncated",
                        max_tokens
                    );
                } else if stop_reason == "refusal" {
                    return Err(Box::from(
                        "[run] llm openapi request was refused (stop_reason: refusal)",
                    ));
                }
                log::info!(
                    "[run] executing llm openapi completed task in {:?} (stop_reason {}, output tokens {})",
                    start.elapsed(),
                    stop_reason,
                    acc.output_tokens
                        .map_or("unknown".to_string(), |t| t.to_string())
                );
                return Ok(acc.text);
            }
            let contents = match result.bytes().await {
                Ok(contents) => contents,
                Err(e) => {
                    return Err(Box::from(format!(
                        "[run] llm openapi failed reading response body (status {}): {}",
                        status,
                        describe_reqwest_error(&e)
                    )));
                }
            };

            if status == StatusCode::OK {
                log::trace!(
                    "[run] llm openapi client response {}",
                    String::from_utf8_lossy(&contents)
                );
                let response = match response_type.as_str() {
                    "Enmaas" => {
                        let chat_response: EnMaasResponse = serde_json::from_slice(&contents)?;
                        chat_response.content[0].text.clone()
                    }
                    "Openai" => {
                        let chat_response: OpenaiChatCompletions =
                            serde_json::from_slice(&contents)?;
                        chat_response.choices[0].message.content.clone()
                    }
                    &_ => {
                        let chat_response: OpenaiChatCompletions =
                            serde_json::from_slice(&contents)?;
                        chat_response.choices[0].message.content.clone()
                    }
                };

                let elapsed = start.elapsed();
                log::info!(
                    "[run] executing llm openapi completed task in {:?}",
                    elapsed
                );
                return Ok(response);
            }

            let body = String::from_utf8_lossy(&contents).to_string();
            // Some gateways cap the output budget below the model maximum even
            // though the model itself supports it. Back off automatically.
            if retries < MAX_MAX_TOKENS_RETRIES
                && max_tokens > MIN_OUTPUT_TOKENS
                && is_max_tokens_error(&body)
            {
                let next = (max_tokens / 2).max(MIN_OUTPUT_TOKENS);
                log::warn!(
                    "[run] endpoint rejected max_tokens {}; retrying with {} ({})",
                    max_tokens,
                    next,
                    body
                );
                max_tokens = next;
                retries += 1;
                continue;
            }

            return Err(Box::from(format!("[run] llm openapi {} {}", status, body)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_STREAM: &str = "event: message_start\r\n\
data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10}}}\r\n\r\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"hidden\"}}\n\n\
event: ping\n\
data: {\"type\": \"ping\"}\n\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello \"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"kernel\"}}\n\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":42}}\n\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\n";

    #[test]
    fn stream_collects_text_and_skips_thinking() {
        let mut acc = StreamAccumulator::default();
        acc.feed(SAMPLE_STREAM.as_bytes()).unwrap();
        assert_eq!(acc.text, "Hello kernel");
        assert_eq!(acc.stop_reason.as_deref(), Some("end_turn"));
        assert_eq!(acc.output_tokens, Some(42));
        assert!(acc.done);
    }

    #[test]
    fn stream_handles_events_split_across_chunks() {
        let mut acc = StreamAccumulator::default();
        for chunk in SAMPLE_STREAM.as_bytes().chunks(7) {
            acc.feed(chunk).unwrap();
        }
        assert_eq!(acc.text, "Hello kernel");
        assert!(acc.done);
    }

    #[test]
    fn stream_keeps_multibyte_chars_split_across_chunks() {
        let event = "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"2 × 4\"}}\n\n";
        let bytes = event.as_bytes();
        let split = event.find('×').unwrap() + 1; // inside the 2-byte char
        let mut acc = StreamAccumulator::default();
        acc.feed(&bytes[..split]).unwrap();
        acc.feed(&bytes[split..]).unwrap();
        assert_eq!(acc.text, "2 × 4");
    }

    #[test]
    fn stream_reports_error_events() {
        let mut acc = StreamAccumulator::default();
        let err = acc
            .feed(b"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n")
            .unwrap_err();
        assert_eq!(err, "overloaded_error: Overloaded");
    }

    #[test]
    fn stream_without_message_stop_is_not_done() {
        let mut acc = StreamAccumulator::default();
        let cut = SAMPLE_STREAM.find("event: message_delta").unwrap();
        acc.feed(&SAMPLE_STREAM.as_bytes()[..cut]).unwrap();
        assert!(!acc.done);
    }
}
