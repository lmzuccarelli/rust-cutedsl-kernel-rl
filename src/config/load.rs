use serde_derive::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::str::FromStr;

fn default_llm_max_output_tokens() -> i32 {
    // A full rewrite of a large CuteDSL kernel needs roughly 30-40k output
    // tokens, so 64k leaves headroom without asking the endpoint for its
    // entire output allowance (which some gateways reject outright).
    64_000
}

fn default_llm_context_window() -> i32 {
    // Claude Opus 4.6+ has a 1M token window, but proxy/gateway deployments
    // often expose the older 200k window. Keep the conservative default and
    // let the operator raise it in the config.
    200_000
}

fn default_llm_input_token_margin() -> i32 {
    // Headroom for tokenizer variance and the chat/system wrapper.
    2_000
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Parameters {
    pub name: String,
    pub log_level: String,
    pub gpu_server_url: String,
    pub llm_server_url: String,
    pub workflow_batch: Vec<String>,
    // Maps a workflow_batch kernel path to the single `@cute.kernel` function
    // name that should be profiled for it. Without an entry the profiler falls
    // back to profiling every annotated kernel it can find in the file (noisy).
    #[serde(default)]
    pub kernel_names: HashMap<String, String>,
    pub working_dir: String,
    pub llm_model: String,
    pub llm_agent: LlmAgent,
    pub cutedsl_env_path: String,
    pub controller_mode: ControllerMode,
    pub token_file: Option<String>,
    pub openapi_url: Option<String>,
    pub response_type: Option<ResponseType>,
    pub gpu_arch: u8,
    pub max_rollout: u8,
    pub rollout_start: u8,
    pub flow_control: u8,
    pub create_stats: bool,
    pub use_error_vec: bool,
    pub exclude_trajectories: Vec<String>,
    pub acceptance_threshold: f64,
    pub degradation_threshold: f64,
    // Upper bound for the `max_tokens` field sent to the LLM endpoint. The
    // effective value is additionally capped by the room left in the context
    // window after the prompt has been accounted for.
    #[serde(default = "default_llm_max_output_tokens")]
    pub llm_max_output_tokens: i32,
    // Total context window exposed by the endpoint. Used to avoid sending a
    // request whose prompt + max_tokens cannot fit.
    #[serde(default = "default_llm_context_window")]
    pub llm_context_window: i32,
    // Safety margin subtracted from the context window when budgeting.
    #[serde(default = "default_llm_input_token_margin")]
    pub llm_input_token_margin: i32,
    // Optional `anthropic-beta` header (e.g. for older 1M-context betas).
    #[serde(default)]
    pub llm_beta_header: Option<String>,
    // used for debugging
    pub exit_on_fail: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct WorkItem {
    pub name: String,
    pub prompt: Option<String>,
    pub target_dir: String,
    pub kernel_file: Option<String>,
    pub code: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum LlmAgent {
    Claude,
    Opencode,
    Api,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum ControllerMode {
    Baseline,
    Agent,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum ResponseType {
    Enmaas,
    Openai,
}

impl fmt::Display for LlmAgent {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl FromStr for LlmAgent {
    type Err = ();
    fn from_str(input: &str) -> Result<LlmAgent, Self::Err> {
        match input {
            "claude" => Ok(LlmAgent::Claude),
            "opencode" => Ok(LlmAgent::Opencode),
            "api" => Ok(LlmAgent::Api),
            _ => Err(()),
        }
    }
}

impl fmt::Display for ResponseType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub trait ConfigInterface {
    fn read(&self, dir: String) -> Result<Parameters, Box<dyn std::error::Error>>;
}

#[derive(Debug, Clone)]
pub struct ImplConfigInterface {}

impl ConfigInterface for ImplConfigInterface {
    fn read(&self, name: String) -> Result<Parameters, Box<dyn std::error::Error>> {
        let json_data = File::open(&name)?;
        let params = serde_json::from_reader(json_data)?;
        Ok(params)
    }
}
