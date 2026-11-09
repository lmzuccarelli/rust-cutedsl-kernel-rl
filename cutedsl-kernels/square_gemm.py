"""
square_gemm.py – C = A @ B  (square, FP32) with CuTeDSL (cutlass.cute 4.8.0+).

Tiled shared-memory GEMM: each CTA computes a TS×TS output tile, each thread
one output element, accumulating over K in TS-wide tiles staged through SMEM.
Uses only plain FP32 FMA (no tensor cores / LDSM), so it is correct and
portable across architectures.

Pattern from:
  examples/python/CuTeDSL/experimental/primitives/tutorial/03_gemm_tiled_smem.py
"""

import os

# The GB10 (Grace Blackwell) GPU reports compute capability sm_121, which this
# CUTLASS DSL build (4.8.0) does not yet enumerate (it tops out at sm_120).
# sm_121 is binary-compatible with the sm_120 family, so target sm_120a.
# Must be set before importing cutlass, as the arch is resolved at import time.
os.environ.setdefault("CUTE_EXPERIMENTAL_DSL_ARCH", "sm_120a")

import torch
import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as prims


@cute.kernel
def square_gemm_kernel(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    TS: cutlass.Constexpr[int],
):
    """C = A @ B. One CTA per TS×TS output tile, one thread per element."""
    tx, ty, _ = cute.arch.thread_idx()   # tx: col within tile, ty: row within tile
    bx, by, _ = cute.arch.block_idx()    # bx: tile row,        by: tile col

    # Shared-memory staging tiles for the current K-slice.
    a_smem = cutlass.Array(cutlass.Float32, (TS, TS), space=cutlass.AddressSpace.smem)
    b_smem = cutlass.Array(cutlass.Float32, (TS, TS), space=cutlass.AddressSpace.smem)

    M = a.shape[0]
    K = a.shape[1]
    N = b.shape[1]

    row = bx * TS + ty
    col = by * TS + tx

    acc = cutlass.Float32(0.0)
    for bk in cutlass.range(0, K, TS):
        # Cooperatively load one TS×TS tile of A and of B into SMEM.
        a_smem[ty, tx] = a[row, bk + tx]
        b_smem[ty, tx] = b[bk + ty, col]

        prims.barrier_cta_sync(0)

        # Multiply the two staged tiles.
        for j in cutlass.range(TS):
            acc += a_smem[ty, j] * b_smem[j, tx]

        prims.barrier_cta_sync(0)

    c[row, col] = acc


@cute.jit
def square_gemm(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    TS: cutlass.Constexpr[int],
):
    M = mA.shape[0]
    N = mB.shape[1]
    block = (TS, TS, 1)
    grid = (M // TS, N // TS, 1)
    square_gemm_kernel(mA, mB, mC, TS).launch(grid=grid, block=block)


def main():
    N = 1024          # square; must be divisible by TS
    TS = 32           # tile size == block edge (TS*TS = 1024 threads/CTA)
    assert N % TS == 0, "N must be divisible by TS"

    torch.manual_seed(0)
    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)
    C = torch.zeros(N, N, device="cuda", dtype=torch.float32)

    square_gemm(
        cute.runtime.from_dlpack(A),
        cute.runtime.from_dlpack(B),
        cute.runtime.from_dlpack(C),
        TS=TS,
    )
    torch.cuda.synchronize()

    ref = A @ B
    max_diff = (C - ref).abs().max().item()
    print(f"N={N}  TS={TS}  max|C - ref| = {max_diff:.4e}")
    torch.testing.assert_close(C, ref, atol=1e-2, rtol=1e-2)
    print("✅ square_gemm OK")


if __name__ == "__main__":
    main()
