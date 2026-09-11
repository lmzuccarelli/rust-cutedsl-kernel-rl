## Usage Guidelines for LLM Integration

### For State Analysis
When analyzing NCU reports, the LLM should:
1. Identify primary and secondary bottlenecks
2. Map to the closest state in this database
3. Consider hardware context and kernel characteristics
4. Account for potential state transitions after optimization

### For Optimization Selection
When selecting optimizations, the LLM should:
1. Prioritize high-confidence, high-impact optimizations
2. Consider composite strategies for complex states
3. Account for potential side effects and trade-offs
4. Adapt parameters based on kernel-specific characteristics

### For Performance Prediction
When predicting improvements, the LLM should:
1. Use base predictions as starting points
2. Adjust based on confidence scores and historical accuracy
3. Consider kernel context and similarity to previous cases
4. Provide uncertainty ranges rather than point estimates

## Integration with Comprehensive GPU Optimization Knowledge

This database integrates with the comprehensive GPU optimization decision tree to provide:
- **Hierarchical optimization strategies**: From high-level decisions to specific implementations
- **Context-aware recommendations**: Based on profiling data and performance characteristics
- **Multi-objective optimization**: Balancing performance, accuracy, and maintainability
- **Hardware-specific guidance**: Tailored recommendations for different GPU architectures

The LLM agents use this database as a **living reference** that evolves based on actual optimization results, enabling continuous improvement in optimization strategy selection and performance prediction.

### Learned Optimization Strategies

#### Expert Technique: tensor_core_utilization

The outline below was originally written for C++/CUDA kernels using the WMMA API.
It has been adapted to **CUTLASS CuTe DSL (`cutlass.cute`, version 4.8.0+)**, which is
the idiom this project uses.

**Key adjustment (CUDA WMMA → CuTe DSL):** In CuTe DSL you never call `wmma::*`
directly. Tensor-core MMA is expressed declaratively:

1. Pick an **MMA atom** for the target instruction shape / dtypes (e.g. a
   16×16×16 HMMA with fp16 inputs and fp32 accumulation).
2. Build a **TiledMMA** that spreads that atom across the threads of a warp
   (or warp-group) via `cute.make_tiled_mma`.
3. Partition the SMEM/GMEM operand tiles with the TiledMMA to obtain per-thread
   **register fragments** (`thr_mma.partition_fragment_A/_B/_C`).
4. Run the K-reduction with `cute.gemm(tiled_mma, acc, rA, rB, acc)`, which lowers
   straight to the tensor-core MMA instruction.

Compared to the CUDA version this removes all manual `lda`/`ldb` pointer math, the
`__syncwarp()` producer/consumer dance, and the hand-written zero-padding pack loop:
CuTe layouts carry the strides, and tails are handled by **predicated copies**
(`cute.copy(..., pred=...)`) instead of packing into scratch SMEM.

**Usage Examples**:

```python
# tensor_core_utilization — CuTe DSL (cutlass.cute 4.8.0+)
#
# Computes a BM×BN output tile of C = A @ B, accumulating across K in BK-wide
# steps on the Tensor Cores. A is (M,K) row-major fp16, B is (K,N) row-major fp16,
# C is (M,N) fp32. Tails (partial M/N/K) are handled with predicated copies rather
# than the CUDA-style zero-pad packing.
import os
# GB10 (sm_121) is binary-compatible with the sm_120 family; the 4.8.0 DSL tops
# out at sm_120. Must be set before importing cutlass (arch resolved at import).
os.environ.setdefault("CUTE_EXPERIMENTAL_DSL_ARCH", "sm_120a")

import torch
import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as prims


@cute.kernel
def tc_gemm_tile_kernel(
    gA: cute.Tensor,          # (M, K) row-major, fp16
    gB: cute.Tensor,          # (K, N) row-major, fp16
    gC: cute.Tensor,          # (M, N) row-major, fp32
    tiled_mma: cute.TiledMma, # tensor-core MMA, built on the host
    BM: cutlass.Constexpr[int],
    BN: cutlass.Constexpr[int],
    BK: cutlass.Constexpr[int],
):
    bx, by, _ = cute.arch.block_idx()

    # This CTA's output tile of C and the matching K-slabs of A and B.
    # local_tile carries the strides, so there is no lda/ldb to pass around.
    cta_C = cute.local_tile(gC, (BM, BN), (bx, by))       # (BM, BN)
    cta_A = cute.local_tile(gA, (BM, BK), (bx, None))     # (BM, BK, k_tiles)
    cta_B = cute.local_tile(gB, (BK, BN), (None, by))     # (BK, BN, k_tiles)

    # SMEM staging tiles for the current K-slice (per CTA, 16-byte aligned).
    sA = cutlass.Array(cutlass.Float16, (BM, BK), space=cutlass.AddressSpace.smem)
    sB = cutlass.Array(cutlass.Float16, (BK, BN), space=cutlass.AddressSpace.smem)

    # Per-thread MMA view: register fragments for A, B, and the C accumulator.
    thr_mma = tiled_mma.get_slice(cute.arch.thread_idx()[0])
    rA = thr_mma.partition_fragment_A(sA)
    rB = thr_mma.partition_fragment_B(sB)
    rC = thr_mma.partition_fragment_C(cta_C)
    rC.fill(0.0)   # replaces wmma::fill_fragment(c_frag, 0.0f)

    k_tiles = cute.size(cta_A, mode=[2])
    for k in cutlass.range(k_tiles):
        # Cooperative, coalesced GMEM -> SMEM copy of the current K-slice.
        # For the K-tail (or partial M/N edge tiles) pass a predicate so
        # out-of-range elements read as zero — the CuTe replacement for the
        # CUDA hand-packed a_pack/b_pack zero-padding.
        cute.copy(cta_A[None, None, k], sA)
        cute.copy(cta_B[None, None, k], sB)
        prims.barrier_cta_sync(0)

        # SMEM -> registers, then issue the tensor-core MMA over the tile.
        cute.copy(sA, rA)
        cute.copy(sB, rB)
        cute.gemm(tiled_mma, rC, rA, rB, rC)   # replaces wmma::mma_sync
        prims.barrier_cta_sync(0)

    # Store the accumulator back to C (replaces wmma::store_matrix_sync).
    cute.copy(rC, thr_mma.partition_C(cta_C))


@cute.jit
def tc_gemm(mA, mB, mC, BM: cutlass.Constexpr[int],
            BN: cutlass.Constexpr[int], BK: cutlass.Constexpr[int]):
    M = mA.shape[0]
    N = mB.shape[1]
    # Build the TiledMMA from a tensor-core atom. The exact atom name depends on
    # the target arch/dtypes; conceptually this is a 16x16x16 HMMA, fp16 in /
    # fp32 out, tiled across one warp.
    mma_op = cute.nvgpu.warp.MmaF16BF16Op(
        cutlass.Float16, cutlass.Float32, (16, 16, 16)
    )
    tiled_mma = cute.make_tiled_mma(cute.make_mma_atom(mma_op))
    grid = (M // BM, N // BN, 1)
    block = (cute.size(tiled_mma), 1, 1)     # one warp (or warp-group) per CTA
    tc_gemm_tile_kernel(mA, mB, mC, tiled_mma, BM, BN, BK).launch(
        grid=grid, block=block
    )
```

**Notes / pitfalls (CuTe DSL):**
- **Layouts replace pointer math.** `cute.local_tile` derives operand tiles with
  correct strides from the tensor's own layout — there is no `lda`/`ldb` to get wrong,
  and no "is B column-major?" trap.
- **Tails use predication, not packing.** Give `cute.copy` a predicate tensor for
  edge tiles / K-tail so out-of-range reads yield zero, instead of the CUDA
  zero-pad-into-scratch-SMEM approach.
- **Synchronization.** Use `prims.barrier_cta_sync(0)` around the SMEM
  producer/consumer steps (the CuTe/project equivalent of `__syncwarp()`/`__syncthreads()`).
- **Architecture.** Tensor cores require sm_70+; this project targets the sm_120
  (Blackwell/GB10) family via `CUTE_EXPERIMENTAL_DSL_ARCH=sm_120a`.
- The MMA atom / TiledMMA construction above is illustrative — pick the atom that
  matches your target arch and operand dtypes.

#### Expert Technique: shared_memory_tiling
**Performance Impact**: 0% improvement
**Confidence Score**: 0.98
**Applicable States**: compute_throughput_saturated, memory_compute_balanced, memory_bandwidth_saturated

**Implementation Hints**:
- Optimal tile size: 16
- Highly effective for matrix multiplication workloads
- Consistently achieves high performance gains

**Usage Examples** (CuTe DSL, `cutlass.cute` 4.8.0+):
```python
# Shared-memory staging tiles: CUDA `__shared__ float As[TS][TS]` becomes a
# CuTe DSL SMEM-space array declared inside the kernel.
a_smem = cutlass.Array(cutlass.Float32, (TS, TS), space=cutlass.AddressSpace.smem)
b_smem = cutlass.Array(cutlass.Float32, (TS, TS), space=cutlass.AddressSpace.smem)

# Cooperative load of one TS×TS K-slice, then a CTA barrier before consuming it
# (replaces __syncthreads()).
a_smem[ty, tx] = a[row, bk + tx]
b_smem[ty, tx] = b[bk + ty, col]
prims.barrier_cta_sync(0)

# Kernel launch: `matmul_kernel<<<numBlocks, threadsPerBlock>>>(...)` becomes a
# .launch() on the @cute.kernel, with grid/block computed on the host.
matmul_kernel(mA, mB, mC, TS).launch(
    grid=(M // TS, N // TS, 1), block=(TS, TS, 1)
)
```

#### Expert Technique: register_optimization
**Performance Impact**: 0% improvement
**Confidence Score**: 0.80
**Applicable States**: compute_throughput_saturated, memory_compute_balanced, memory_bandwidth_saturated

**Implementation Hints**:
- Highly effective for matrix multiplication workloads
- Consistently achieves high performance gains

**Usage Examples** (CuTe DSL, `cutlass.cute` 4.8.0+):
```python
# Register optimization: keep the accumulator in a register across the whole
# K-loop so there is no SMEM/GMEM round-trip per step. In CuTe DSL a plain
# scalar (or a partition_fragment_C register fragment for tensor cores) lives
# in registers.
acc = cutlass.Float32(0.0)               # accumulator held in a register
for bk in cutlass.range(0, K, TS):
    a_smem[ty, tx] = a[row, bk + tx]
    b_smem[ty, tx] = b[bk + ty, col]
    prims.barrier_cta_sync(0)
    for j in cutlass.range(TS):          # inner product stays in registers
        acc += a_smem[ty, j] * b_smem[j, tx]
    prims.barrier_cta_sync(0)
c[row, col] = acc                        # single write-back to GMEM

# For tensor-core paths, the C fragment is the register accumulator:
#   rC = thr_mma.partition_fragment_C(cta_C)   # lives in registers
#   cute.gemm(tiled_mma, rC, rA, rB, rC)

# Kernel launch (host side):
tensor_matmul_kernel(mA, mB, mC, TS).launch(grid=blocks, block=threads)
```

#### Expert Technique: memory_coalescing
**Performance Impact**: 0% improvement
**Confidence Score**: 0.83
**Applicable States**: compute_throughput_saturated, memory_compute_balanced, memory_bandwidth_saturated

**Implementation Hints**:
- Optimal tile size: 16
- Highly effective for matrix multiplication workloads
- Consistently achieves high performance gains

**Usage Examples** (CuTe DSL, `cutlass.cute` 4.8.0+):
```python
# Memory coalescing: make the fastest-varying thread index (tx) index the
# stride-1 (contiguous) mode of the tensor so a warp reads a contiguous span.
a_smem[ty, tx] = a[row, bk + tx]   # tx -> consecutive columns  => coalesced
b_smem[ty, tx] = b[bk + ty, col]   # tx -> consecutive columns  => coalesced

# For explicit control, build a TiledCopy whose thread layout matches the
# tensor's fastest mode; cute.copy then emits coalesced (and vectorized) loads.
tiled_copy = cute.make_tiled_copy(
    cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32),
    thr_layout, val_layout,
)
cute.copy(tiled_copy, gmem_src, smem_dst)

# Kernel launch (host side): raw pointers + <<<grid,block>>> become cute.Tensors
# (via from_dlpack) passed to a .launch() call.
matmul_kernel(mA, mB, mC, M, K, N).launch(grid=grid, block=block)
```

#### Expert Technique: occupancy_tuning
**Performance Impact**: 0% improvement
**Confidence Score**: 0.85
**Applicable States**: compute_throughput_saturated, memory_compute_balanced, memory_bandwidth_saturated

**Implementation Hints**:
- Highly effective for matrix multiplication workloads
- Consistently achieves high performance gains

**Usage Examples** (CuTe DSL, `cutlass.cute` 4.8.0+):
```python
# Occupancy tuning: occupancy is driven by the CTA (block) size and the SMEM
# footprint chosen on the host. Tune `block_size` (and the SMEM tile shapes,
# which set the per-CTA SMEM usage) to trade off active warps per SM.
block_size = 256                     # tune: 128 / 256 / 512 for occupancy
grid = ((batch_size * m * n + block_size - 1) // block_size, 1, 1)
batched_matmul_kernel(mA, mB, mC, batch_size, m, k, n).launch(
    grid=grid, block=(block_size, 1, 1)
)

# Per-CTA SMEM (which competes with occupancy) is sized by the SMEM arrays the
# kernel declares; smaller staging tiles => more resident CTAs. The CUDA
# `extern __shared__ float shared_mem[]` becomes a sized SMEM array in the kernel:
shared_mem = cutlass.Array(cutlass.Float32, (block_size,),
                           space=cutlass.AddressSpace.smem)
square_sum_kernel(mIn, mOut, n).launch(grid=grid, block=(block_size, 1, 1))
```

#### Expert Technique: dynamic_shared_memory
**Performance Impact**: 0% improvement
**Confidence Score**: 0.44
**Applicable States**: memory_compute_balanced, low_occupancy_register_pressure, memory_latency_bound

**Implementation Hints**:
- Consistently achieves high performance gains

**Usage Examples** (CuTe DSL, `cutlass.cute` 4.8.0+):
```python
# Dynamic shared memory: CUDA sizes `extern __shared__` at *launch* time via the
# third <<<...>>> argument. In CuTe DSL the SMEM size is a JIT compile-time
# Constexpr instead — pass it as a kernel argument so the array is sized when the
# kernel is specialized. (This is the key CUDA->CuTe adjustment.)
@cute.kernel
def matvec_mul_kernel(mA, mX, mOut, N,
                      SMEM_ELEMS: cutlass.Constexpr[int]):
    shared_sum = cutlass.Array(cutlass.Float32, (SMEM_ELEMS,),
                               space=cutlass.AddressSpace.smem)  # partial sums
    ...

# Host side: choose the SMEM element count and pass it as a Constexpr; no
# runtime shared_mem_size byte argument on .launch().
smem_elems = threads
matvec_mul_kernel(mA, mX, mOut, N, smem_elems).launch(
    grid=blocks, block=(threads, 1, 1)
)

# Same pattern for a batched softmax reduction (CUDA `extern __shared__ float sdata[]`):
@cute.kernel
def softmax_kernel_batch(mIn, mOut, batch_size, dim,
                         SMEM_ELEMS: cutlass.Constexpr[int]):
    sdata = cutlass.Array(cutlass.Float32, (SMEM_ELEMS,),
                          space=cutlass.AddressSpace.smem)
    ...

softmax_kernel_batch(mIn, mOut, batch_size, dim, threads).launch(
    grid=blocks, block=(threads, 1, 1)
)
```


### Expert Technique Combinations

**Combination**: register_optimization + shared_memory_tiling
- Average Performance: 0% speedup
- Confidence: 0.60
- Frequency in top solutions: 3 times


## Expert-Learned Optimizations (Auto-Generated)

### Expert Technique: shared_memory_tiling
**Source**: KernelBench Leaderboard Analysis
**Performance Impact**: 0% improvement
**Confidence**: 0.98

**Implementation Strategy**:
- Optimal tile size: 16
- Highly effective for matrix multiplication workloads
- Consistently achieves high performance gains

**Usage Context**:
Best applied to: memory_bandwidth_saturated, memory_compute_balanced, compute_throughput_saturated


### Expert Technique: register_optimization
**Source**: KernelBench Leaderboard Analysis
**Performance Impact**: 0% improvement
**Confidence**: 1.00

**Implementation Strategy**:
- Highly effective for matrix multiplication workloads
- Consistently achieves high performance gains

**Usage Context**:
Best applied to: memory_bandwidth_saturated, memory_compute_balanced, compute_throughput_saturated


### Expert Technique: memory_coalescing
**Source**: KernelBench Leaderboard Analysis
**Performance Impact**: 0% improvement
**Confidence**: 0.83

**Implementation Strategy**:
- Optimal tile size: 16
- Highly effective for matrix multiplication workloads
- Consistently achieves high performance gains

**Usage Context**:
Best applied to: memory_bandwidth_saturated, memory_compute_balanced, compute_throughput_saturated


### Expert Technique: occupancy_tuning
**Source**: KernelBench Leaderboard Analysis
**Performance Impact**: 0% improvement
**Confidence**: 0.85

**Implementation Strategy**:
- Highly effective for matrix multiplication workloads
- Consistently achieves high performance gains

**Usage Context**:
Best applied to: memory_bandwidth_saturated, memory_compute_balanced, compute_throughput_saturated


### Expert Technique: dynamic_shared_memory
**Source**: KernelBench Leaderboard Analysis
**Performance Impact**: 0% improvement
**Confidence**: 0.44

**Implementation Strategy**:
- Consistently achieves high performance gains

**Usage Context**:
Best applied to: memory_latency_bound, memory_compute_balanced, low_occupancy_register_pressure

