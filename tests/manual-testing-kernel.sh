#!/usr/bin/env bash
#
# Manual testing for cutedsl-kernels/cuteDSL_gemm.py
# -------------------------------------------------------------------------
# Mirrors what the Rust RL harness does so the kernel can be exercised by hand:
#   * src/kernel/execute.rs runs `python <kernel_file>` and treats a non-zero
#     exit (or the word "failed" in stdout) as failure.
#   * src/kernel/profile.rs extracts the cute-kernel name and runs ncu, then
#     re-imports the report with `--page details`.
#
# The commands below are the exact ones used to bring up and validate the merged
# execution + profiling harness appended to cuteDSL_gemm.py.
#
# Usage:
#   tests/manual-testing-kernel.sh            # run everything (setup checks, execute, profile)
#   tests/manual-testing-kernel.sh execute    # just run/validate the kernel
#   tests/manual-testing-kernel.sh profile    # just profile with ncu
#   tests/manual-testing-kernel.sh setup      # just the env / dependency checks
set -euo pipefail

# --- Configuration (matches config/application-config*.json) ----------------
CUTEDSL_ENV_PATH="${CUTEDSL_ENV_PATH:-/home/lzuccarelli/Projects/cutlass/target_cute_env}"
PY="${CUTEDSL_ENV_PATH}/bin/python3"
CUDA_BIN="${CUDA_BIN:-/usr/local/cuda/bin}"
NCU="${CUDA_BIN}/ncu"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KERNEL_DIR="${REPO_ROOT}/cutedsl-kernels"
KERNEL_FILE="cuteDSL_gemm.py"

# The cute-kernel name profile.rs extracts via the regex
#   @cute.kernel[\sdef\s]+([a-zA-Z0-9_]*)
KERNEL_NAME="${KERNEL_NAME:-dense_gemm_kernel}"

# -------------------------------------------------------------------------
# setup: dependency / environment checks
# -------------------------------------------------------------------------
setup() {
  echo "== environment checks =="
  "${PY}" --version
  "${PY}" -I -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"

  # flashinfer is required by cuteDSL_gemm.py (import at top) and by the harness
  # (nvfp4_quantize + _compile_block_scaled_gemm). Install once if missing:
  #   "${PY}" -m pip install flashinfer-python
  "${PY}" -I -c "
from flashinfer import nvfp4_quantize, SfLayout
from flashinfer.cute_dsl.utils import sm120_make_smem_layout_sfa, torch_to_cutlass_dtype
from flashinfer.gemm.gemm_base import _compile_block_scaled_gemm, _prepare_alpha_for_launch
from flashinfer.utils import get_device_sm_count
print('flashinfer glue OK')
"

  # ninja (flashinfer JIT) and nvcc must be discoverable. They live in the env
  # bin and CUDA bin; the harness driver prepends both to PATH at runtime.
  "${CUTEDSL_ENV_PATH}/bin/ninja" --version
  "${NCU}" --version | head -1
}

# -------------------------------------------------------------------------
# execute: run + validate the kernel (same as src/kernel/execute.rs)
# -------------------------------------------------------------------------
execute() {
  echo "== execute: ${KERNEL_FILE} =="
  cd "${KERNEL_DIR}"
  # execute.rs runs bare `python <kernel_file>`; we use the env python3 directly.
  # The driver prints a cosine_similarity line and PASS, exiting non-zero on fail.
  "${PY}" "${KERNEL_FILE}"
}

# -------------------------------------------------------------------------
# profile: ncu profile + report import (same as src/kernel/profile.rs)
# -------------------------------------------------------------------------
profile() {
  echo "== profile: ${KERNEL_FILE} (kernel: ${KERNEL_NAME}) =="
  cd "${KERNEL_DIR}"

  # 1) Collect the profile (needs sudo for GPU perf counters).
  sudo "${NCU}" \
    --target-processes all \
    --kernel-name "regex:${KERNEL_NAME}" \
    --set detailed \
    -o "profile-${KERNEL_NAME}" \
    -f \
    "${PY}" "${KERNEL_FILE}"

  # 2) Import and print the details page (profile.rs scrapes Elapsed Cycles from this).
  local report="profile-${KERNEL_NAME}.ncu-rep"
  [ -f "profile-${KERNEL_NAME}.ncu-repz" ] && report="profile-${KERNEL_NAME}.ncu-repz"
  sudo "${NCU}" --import "${report}" --page details | tee "${KERNEL_NAME}.profile"
}

main() {
  case "${1:-all}" in
    setup)   setup ;;
    execute) execute ;;
    profile) profile ;;
    all)     setup; execute; profile ;;
    *) echo "usage: $0 [all|setup|execute|profile]" >&2; exit 2 ;;
  esac
}

main "$@"
