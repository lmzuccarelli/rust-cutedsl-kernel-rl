use custom_logger as log;
use hyper::StatusCode;
use reqwest::Client;
use serde_derive::Deserialize;
use serde_derive::Serialize;
use serde_json::Value;
use std::process::Command;
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

pub trait LlmInterface {
    async fn run(prompt: String) -> Result<String, Box<dyn std::error::Error>>;
}

/// Token budget for an OpenAPI/Anthropic-compatible endpoint.
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

/// Conservative token estimate: roughly four characters per token for code and
/// English. Deliberately rounds up so the budget stays inside the window.
fn estimate_tokens(text: &str) -> i64 {
    ((text.chars().count() as i64) + 3) / 4
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

pub struct LlmClaude {}
pub struct LlmOpenCode {}
pub struct LlmOpenApi {}

impl LlmInterface for LlmClaude {
    async fn run(prompt: String) -> Result<String, Box<dyn std::error::Error>> {
        log::debug!("[run] executing llm claude inference endpoint");
        let start = Instant::now();

        // TODO: address ARG_MAX issues
        let output = Command::new("claude").args(vec!["-p", &prompt]).output()?;

        let elapsed = start.elapsed();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        // preserve output
        println!("{}", stdout);
        log::info!("[run] executing llm claude completed task in {:?}", elapsed);

        if !output.status.success() {
            // return the first failure (fail fast)
            return Err(Box::from(stderr));
        }

        Ok(stdout)
    }
}

impl LlmInterface for LlmOpenCode {
    async fn run(prompt: String) -> Result<String, Box<dyn std::error::Error>> {
        log::debug!("[run] executing llm opencode inference endpoint");
        let start = Instant::now();

        let output = Command::new("opencode")
            .args(vec!["-p", &prompt])
            .output()?;

        let elapsed = start.elapsed();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        // preserve output
        println!("{}", stdout);
        log::info!(
            "[run] executing llm opencode completed task in {:?}",
            elapsed
        );

        if !output.status.success() {
            // return the first failure (fail fast)
            return Err(Box::from(stderr));
        }

        Ok(stdout)
    }
}

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
        let mut max_tokens =
            (token_config.max_output_tokens.max(MIN_OUTPUT_TOKENS) as i64).min(available) as i32;
        if max_tokens < 40_000 {
            log::warn!(
                "[run] only {} output tokens available; a full kernel rewrite may be truncated. Reduce the prompt or raise `llm_context_window`.",
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
            .timeout(Duration::new(1200, 0))
            .build();

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
        loop {
            let rs = RequestSchema {
                model: model.clone(),
                // temperature: 0.1,
                stream: false,
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

            let result = match request.body(json_request).send().await {
                Ok(result) => result,
                Err(e) => {
                    return Err(Box::from(format!("[run] llm openapi error {}", e)));
                }
            };

            let status = result.status();
            log::debug!("[run] llm openapi response status {}", status);
            let contents = result.bytes().await?;

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
