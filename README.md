# Overview

A complete cutedsl kernel reinforcement workflow written in Rust

## Usage

Clone this repo

```bash
git clone https://github.com/lmzuccarelli/rust-cutedsl-kernel-rl

cd rust-cutedsl-kernel-rl
```

### Build binary

This assumes you have installed Rust

```bash
make build
```

## Prerequisites and dependencies (cuda 13.2 specific)

- python
- libtorch
- cuda 13.2 and cuda 13.2 driver
- nvshmem-cuda-13
- cusparselt
- libnccl-2.30.3-1+cuda13.2
- libnccl-devel-2.30.3-1+cuda13.2
- libnccl-static-2.30.3-1+cuda13.2
- cudnn9-cuda-13
- nvidia nsights (ncu)
- cutlass (for CuteDSL)

The installation of these dependencies is out of scope for this project and it's left up to the user to install.

## Workflow

There are basically 2 types of endpoint servers

- llm (for inferencing)
- gpu

The llm service can be executed on any server (no gpu required). It can use an OpenAI-compatible llm service, the default is using claude with opus4.8

The gpu requires a gpu and relevant cuda and cutedsl dependencies to be installed and available, this is where the 'kernel' will be executed and profiled through ncu (nvidia nsights).

### Basic Flow

- The user initiates a workflow via the various service endpoints (via a config file) using a json payload to indicate the kernel/s to be executed and profiled.
- The gpu process is used to get an initial baseline by executing the kernel and then profiling it using **ncu** (Nvidia's profiler) for the *Elapsed Cycles*.
- The llm process is prompted with results from the ncu insights (referencing the Speed of Light results), the known database of problems and recommended optimisations with the cuda-kernel code as reference.
- This process is repeated until the set trajectory value is reached (max_rollout).
- The trajectory with the best score (lowest Elapsed Cycles as a percentage from the baseline) will be saved together with the relevant cutedsl-kernel optimized code.
- A simple workflow cli is used for the workflow control.

## Local kernel execution and profiling

The `tests/manual-testing-kernel.sh` script mirrors exactly what the Rust RL harness
does, so a kernel can be exercised and profiled by hand without running a full workflow:

- `src/kernel/execute.rs` runs `python <kernel_file>` and treats a non-zero exit (or the word "failed" in stdout) as a failure.
- `src/kernel/profile.rs` extracts the cute-kernel name and runs `ncu`, then re-imports the report with `--page details`.

The script targets `cutedsl-kernels/cuteDSL_gemm.py` and the `dense_gemm_kernel`
cute kernel by default.

### Running the script

```bash
# run everything (setup checks, execute, profile)
tests/manual-testing-kernel.sh

# just the env / dependency checks
tests/manual-testing-kernel.sh setup

# just run/validate the kernel
tests/manual-testing-kernel.sh execute

# just profile with ncu
tests/manual-testing-kernel.sh profile
```

### Configuration

The script reads the following environment variables (the defaults match
`config/application-config*.json`):

- `CUTEDSL_ENV_PATH` - path to the cutedsl python environment (default `/home/lzuccarelli/Projects/cutlass/target_cute_env`). The env `python3` and `ninja` are used from here.
- `CUDA_BIN` - path to the cuda `bin` directory (default `/usr/local/cuda/bin`), used to locate `ncu`.
- `KERNEL_NAME` - the cute-kernel function name to profile (default `dense_gemm_kernel`). This is the name `profile.rs` extracts via the regex `@cute.kernel[\sdef\s]+([a-zA-Z0-9_]*)`.

Example overriding the kernel name:

```bash
KERNEL_NAME=my_gemm_kernel tests/manual-testing-kernel.sh profile
```

### What each stage does

- **setup** - prints the python / torch / cuda versions, verifies the `flashinfer` glue imports used by `cuteDSL_gemm.py`, and checks that `ninja` and `ncu` are discoverable.
- **execute** - runs the kernel with the env `python3` from `cutedsl-kernels/`; the driver prints a `cosine_similarity` line and `PASS`, exiting non-zero on failure.
- **profile** - collects an ncu profile (requires sudo for GPU perf counters) into `profile-<KERNEL_NAME>.ncu-rep`, then imports it and writes the details page to `<KERNEL_NAME>.profile` (the same page `profile.rs` scrapes Elapsed Cycles from).

## Caveats

### Update the sudo user in the /etc/sudoers file

Example

```bash
myuser ALL=(ALL) NOPASSWD: ALL
```

This disables asking for a password (ncu needs elevated privileges) refer to [permissions](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters#AllUsersTag)

### Ensure LD_LIBRARY_PATH envar is set

Before executing the gpu or compiler services ensure the LD_LIBRARY_PATH envar is set correctly (link to cuda lib64 and CUPTI)

Example

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-<version>/lib64:/usr/local/cuda/extras/CUPTI/lib64
```

## Troubleshooting

### Manually test cuda kernels

Use the following cli in the specific "trajectory_XXX/step_XXX/build" directory on the gpu server (base directory is "out") on the gpu server

```bash
python3 <kernel-name>.py
```

### Manually profile cutedsl kernels

Look for the kernel name in file kernel_file.py (search for @cute.kernel annotation)

Use the following cli to execute a profile

```bash
# execute all metrics (this will include Speed of Light)
sudo ncu --target-processes all --kernel-name regex:<kernel-function_name> --set full -o profile-<kernel-name> -f /bin/python3 kernel_file.py

# convert to text format
sudo ncu --import profile-<kernel-name>.ncu-rep --page details > kernel-name.profile
```

## LLM token budget

The LLM prompt for a code-generation step contains the full kernel source, the
NCU profile and the optimization database. For `cuteDSL_gemm.py` that is roughly
140 KB of Python (~31k tokens), a 13 KB NCU report (~3k tokens) and ~19 KB of
optimization guidance (~5k tokens), so around **40k input tokens** before the
chat wrapper. The old code also hardcoded `max_tokens: 128000`, which many
gateways reject even when the model itself supports a 128k output budget.

The controller now budgets the request before calling the endpoint:

- `llm_max_output_tokens` (default `64000`) is the upper bound for `max_tokens`.
- `llm_context_window` (default `200000`) is the context window exposed by your
  endpoint. Raise it to `1000000` only if the gateway actually serves the
  Claude 1M window; the conservative default prevents an opaque 400.
- `llm_input_token_margin` (default `2000`) is reserved for tokenizer variance.
- `llm_beta_header` (optional) is sent as the `anthropic-beta` header, for
  gateways that still require a context beta flag.

The effective `max_tokens` is
`min(llm_max_output_tokens, llm_context_window - estimated_input_tokens - margin)`.
If the prompt alone does not fit, the call fails early with a message that says
which knob to change instead of sending an oversized payload. If a gateway
rejects the requested output budget, the call automatically retries with half
the value (twice).

For very large kernels the long-term fix is to send only the section being
edited and ask the model for a patch/diff rather than a complete file; that
keeps both the input and output well below any context limit.

### Diagnosing the endpoint error

`process_post_call` writes the response body to the `.txt`/`.json` file in the
trajectory even when the HTTP status is not 200, so inspect e.g.
`replay-buffer/<model>/<kernel>/rl-ncu/baseline/llm_state_response.txt`:

- `prompt is too long` / `context_length_exceeded` -> input is too large.
  Reduce the code/NCU payload or raise `llm_context_window`.
- an error mentioning `max_tokens` / `max output tokens` -> the endpoint caps
  the output budget. Lower `llm_max_output_tokens` (the retry logic also does
  this automatically).
- `413` / `request entity too large` -> the body size limit, not the token
  limit. Reduce the payload or split the request.

## Replay buffer

The actual trajectory and steps within each trajectory has not been pushed into this repo, we are currently looking into creating
a S3 bucket solution
