use std::fs;
use std::io;
use std::path::Path;

use crate::prompts::optimizer_artifacts_creator::{
    DEFAULT_PATH, create_available_optimizations, create_performance_state_category,
};

/// The source code to emit, held in a string buffer.
const GENERATED_PROMPT_SOURCE: &str = r##"// N.B. This code is auto generated do not change it inline !!!
//
// version: {version}
//
// Use the following workflow to re-generate the "generated_prompts.rs" file
// a. Update the artifacts/optimization_database.json (adding new techniques)
// b. Execute the 'make generate' recipe, this will re-generate the new file
pub fn get_state_prompt(code: String, profile: String) -> String {
    format!(
        r#"
You are a GPU performance analysis expert. Analyze this NVIDIA NSight Compute (NCU) profiling report (in text format) and provide a qualitative summary of the kernel's performance state.

CODE IMPLEMENTATION:
```python
{code}
```

NCU REPORT:
{profile}

Provide your analysis in this EXACT format:

PERFORMANCE_SIGNATURE: [2-3 sentence summary of what is limiting performance and the overall execution pattern]

RELATIVE_PATTERNS:
- memory_pressure: [very_low|low|moderate|high|very_high]
- compute_utilization: [very_low|low|moderate|high|very_high]
- access_patterns: [excellent|good|moderate|poor|very_poor]
- cache_efficiency: [excellent|good|moderate|poor|very_poor]
- occupancy_level: [very_low|low|moderate|high|very_high]
- parallelism_utilization: [very_low|low|moderate|high|very_high]
- specialied_hw_usage: [very_low|low|moderate|high|very_high]
- [List 3-5 key secondary performance characteristics]
- [Focus on patterns you observe in the data]
- [Include cache behavior, memory access patterns, occupancy]
- [Note any resource conflicts or inefficiencies]

PRIMARY_BOTTLENECK: [memory_bound|compute_bound|latency_bound|hybrid_bound]

//code signature: loop pattern /branches(in summary/generic)

CONTEXT_DESCRIPTION: [Brief description of the workload characteristics and optimization opportunities]

Focus on qualitative patterns and relationships rather than specific numbers. Look for the underlying performance characteristics that drive behavior.
"#
    )
}

pub fn get_optimization_plan(top_n: u8, state_analysis_response: String, code: String) -> String {
    format!(r#"
You are a world-class GPU optimisation expert.
Based on the kernel implementation and the qualitative state analysis below, choose the **{top_n}** optimisation techniques that are most likely to improve performance.
From the list of AVAILABLE OPTIMISATIONS pick only those with the highest relevance to the observed performance characteristics **and** the specific code patterns you see.

For *each* chosen technique provide a concise explanation *why* it is relevant and *how* it should be applied to the given code.
Also assign a numerical RELEVANCE_SCORE between 0.0 (not relevant) and 1.0 (perfect match).

Return your answer as **valid JSON** in the EXACT format (no extra keys, no comments, do not wrap the JSON in markdown fences):

[
  {{
    "technique": "<name>",
    "relevance_score": <float 0-1>,
    "description": "<explain what the optimisation does and *why* it applies to the current code>"
  }},
  ... (exactly {top_n} entries)
]

Besides the available optimisation techniques in Performance State Categories, you should always consider using the following techniques:

- For memory bandwidth bound, or compute bandwidth bound kernels: prioritize using **SIMD_operations**: Use packed SIMD datatypes such as half2
- For compute bandwidth bound and compute throughput bound kernels: prioritize using **tensor_core_utilization**: Use tensor core library such as wmma when there exist a tensor cores (sm_70+) in the target GPU archecture.

----------------------- CURRENT KERNEL CONTEXT -----------------------
STATE ANALYSIS RESPONSE:
{state_analysis_response}

CODE IMPLEMENTATION:
```python
{code}
```

----------------------- AVAILABLE OPTIMISATIONS ----------------------
{avalaible_optimizations}
    "#).to_string()
}

pub fn get_performance_state_category() -> String {
    r#"{performance_state_category}
  "#
    .to_string()
}

pub fn get_task_generate_code_prompt(
    technique: String,
    category: String,
    description: String,
    profile: String,
    code: String,
) -> String {
    format!(
        r#"
OPTIMIZATION TASK:
You are an expert CUDA/CuteDSL optimization agent, and you are provided an optimization plan. 
Your task is to apply the optimization plan to the current kernel. 
You will be provided with the annotated source code, the raw NCU profile (in text format), and the optimization plan.

OPTIMIZATION STRATEGY: {technique}
PREDICTED IMPROVEMENT: N/A
CATEGORY: {category}

STRATEGY DESCRIPTION:
{description}

CURRENT KERNEL ANALYSIS:
{profile}

ANNOTATED SOURCE CODE (with per-line analysis):
```python
{code}
```
{optimization_database}

CRITICAL REQUIREMENTS:
1. Reference the optimization database for detailed implementation guidance.
2. Generate COMPLETE, EXECUTABLE CuteDSL kernel code.
3. Import ALL necessary components:
   - import numpy as np
   - import cutlass
   - import cutlass.cute as cute
   - import torch
   - Ensure there is a validation harness (main()) included
   - Generate full completed CuteDSL code (no code snippets)
4. Format ALL code in a single ```python code block.
5. Focus specifically on the technique described in the database.
6. EXECUTIONAL SAFETY: Ensure all constants are properly defined.
7. Summarize the optimization technique applied and the reason for the improvement before the code. Be concise in your summarization.
8. DO NOT include any cutedsl code in the summarization.
9. DO NOT attempt to write the code to disk, as you don't have permission to do so, simply output the generated code to console only.
10. DO NOT change the detected kernel function signature and parameters, keep them intact as given in the ANNOTATED SOURCE CODE section

APPROACH:
1. Apply the requested optimization technique systematically.
2. If applying a new technique not yet attempted in the code, start with the most minimal example, focusing on correctness.
3. Please use the reference code provided in the prompt as helper functions for your optimized kernel.
4. Generate CuteDSL (python) optimized code addressing the identified performance issues.

"#).to_string()
}
"##;

/// Write the wrapped string buffer to `src/inference/generated_prompt.rs`.
///
/// Returns the number of bytes written on success.
pub fn write_generated_prompt() -> io::Result<usize> {
    write_generated_prompt_to("src/inference/generated_prompts.rs")
}

/// Write the wrapped string buffer to an arbitrary path (handy for tests).
pub fn write_generated_prompt_to<P: AsRef<Path>>(path: P) -> io::Result<usize> {
    let available_optimizations =
        create_available_optimizations(DEFAULT_PATH).map_err(io::Error::other)?;
    let performance_state_category =
        create_performance_state_category(DEFAULT_PATH).map_err(io::Error::other)?;
    // The optimization database is a single markdown document that carries a
    // `{performance_state_category}` placeholder; it must be substituted in
    // before that placeholder is expanded below.
    let optimization_database = fs::read_to_string("artifacts/optimization_database.md")?;

    let version = env!("CARGO_PKG_VERSION");

    let result = GENERATED_PROMPT_SOURCE
        .replace("{version}", version)
        .replace("{avalaible_optimizations}", &available_optimizations)
        .replace("{optimization_database}", &optimization_database)
        .replace("{performance_state_category}", &performance_state_category);

    fs::write(&path, &result)?;
    Ok(result.len())
}
