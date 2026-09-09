"""
square_gemm.py – C = A @ B  (N×N, FP16, SM80 single-warp tensor core)
Patterns from: examples/python/CuTeDSL/ampere/gemm.py
"""

import torch
import cutlass
from cutlass import cute
from cutlass.cute import experimental as cute_ext
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack


@cute.experimental.kernel
def square_gemm_kernel(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    """Compute C = A @ B. Single warp, 16×8×8 MMA atom."""

    cta_m = cute.arch.block_idx(0)
    cta_n = cute.arch.block_idx(1)
    tid_x = cute.arch.thread_idx()

    M = cute.size(mA, 0)
    K = cute.size(mA, 1)
    N = cute.size(mB, 1)

    # ── MMA atom: m16n8k8, FP16×FP16→FP32 ──────────────────────────────────
    mma_op = cutlass.cute.nvgpu.warp.MmaF16BF16Op(
        cutlass.Float16,
        cutlass.Float32,
        (16, 8, 8),
    )
    tiled_mma = cute.make_tiled_mma(mma_op, (1, 1, 1))

    # ── Tile global tensors (one MMA tile per CTA) ─────────────────────────
    gA = cute.tiled_divide(mA, (16, 8))
    gB = cute.tiled_divide(mB, (8, 8))
    gC = cute.tiled_divide(mC, (16, 8))

    gA_tile = gA[(None, None), (cta_m, None)]   # (16, K)
    gB_tile = gB[(None, None), (None, cta_n)]   # (K, 8)
    gC_tile = gC[(None, None), (cta_m, cta_n)]  # (16, 8)

    k_tile_size = K // 8

    # ── SMEM ───────────────────────────────────────────────────────────────
    bufferSA = cute_ext.allocate(
        cute.Float16,
        cutlass.AddressSpace.smem,
        cute.make_layout((16, 8), stride=(8, 1)),
        alignment=16,
    )
    bufferSB = cute_ext.allocate(
        cute.Float16,
        cutlass.AddressSpace.smem,
        cute.make_layout((8, 8), stride=(8, 1)),
        alignment=16,
    )

    # ── Register accumulator (FP32) ────────────────────────────────────────
    bufferRC = cute_ext.allocate(
        cute.Float32,
        cutlass.AddressSpace.rmem,
        cute.make_layout((16, 8), stride=(8, 1)),
    )
    bufferRC.fill(0.0)

    # ── G→S copy atom (cp.async, 32-bit = 2×FP16) ─────────────────────────
    g2s_copy_atom = cute.make_copy_atom(
        cutlass.cute.nvgpu.cpasync.CopyG2SOp(),
        cute.Float16,
        num_bits_per_copy=32,
    )

    tiled_copy_g2s_A = cute.make_tiled_copy_tv(
        g2s_copy_atom,
        cute.make_layout((16, 2), stride=(2, 1)),   # 32 threads
        cute.make_layout((1, 4)),                    # 4 elems/thread
    )
    tiled_copy_g2s_B = cute.make_tiled_copy_tv(
        g2s_copy_atom,
        cute.make_layout((8, 4), stride=(4, 1)),    # 32 threads
        cute.make_layout((1, 2)),                    # 2 elems/thread
    )

    # ── S→R copy atoms (LDSM) ──────────────────────────────────────────────
    copy_s2r_a = cute.make_tiled_copy_A(
        cute.make_copy_atom(
            cutlass.cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=1),
            cute.Float16,
        ),
        tiled_mma,
    )
    copy_s2r_b = cute.make_tiled_copy_B(
        cute.make_copy_atom(
            cutlass.cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=1),
            cute.Float16,
        ),
        tiled_mma,
    )

    # ── Per-thread register fragments ──────────────────────────────────────
    thr_mma = tiled_mma.get_slice(tid_x)
    bufferRA = thr_mma.make_fragment_A(bufferSA)
    bufferRB = thr_mma.make_fragment_B(bufferSB)

    # ── SMEM dest fragments for G→S ────────────────────────────────────────
    tCsA = tiled_copy_g2s_A.get_slice(tid_x).partition_D(bufferSA)
    tCsB = tiled_copy_g2s_B.get_slice(tid_x).partition_D(bufferSB)

    # ── Main K-loop ────────────────────────────────────────────────────────
    for k_tile in range(k_tile_size):
        gA_k = gA_tile[None, None, k_tile]
        gB_k = gB_tile[None, None, k_tile]

        # Partition gmem for G→S
        tAgA_k = cute_ext.partition(
            gA_k,
            tid_x,
            layout_tv=tiled_copy_g2s_A.layout_src_tv_tiled,
            tiler=cute.core._pack_tile(tiled_copy_g2s_A.tiler_mn),
        )
        tAgA_k = cute_ext.predicated_tensor_origin(
            cute.make_tensor(tAgA_k.iterator.align(4), tAgA_k.layout)
        )

        tBgB_k = cute_ext.partition(
            gB_k,
            tid_x,
            layout_tv=tiled_copy_g2s_B.layout_src_tv_tiled,
            tiler=cute.core._pack_tile(tiled_copy_g2s_B.tiler_mn),
        )
        tBgB_k = cute_ext.predicated_tensor_origin(
            cute.make_tensor(tBgB_k.iterator.align(4), tBgB_k.layout)
        )

        # G→S (cp.async)
        cute_ext.copy(tAgA_k, tCsA, copy_atom=g2s_copy_atom)
        cute_ext.copy(tBgB_k, tCsB, copy_atom=g2s_copy_atom)

        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        # S→R (LDSM)
        cute_ext.partition_and_copy(
            copy_s2r_a.get_slice(tid_x), bufferSA, bufferRA
        )
        cute_ext.partition_and_copy(
            copy_s2r_b.get_slice(tid_x), bufferSB, bufferRB
        )

        # MMA:  C += A @ B
        cute_ext.dot(tiled_mma, bufferRA, bufferRB, bufferRC)

        cute.arch.barrier()

    # ── Epilogue: R→G ──────────────────────────────────────────────────────
    copy_r2g_c = cute.make_tiled_copy_C(
        cute.make_copy_atom(
            cutlass.cute.nvgpu.CopyUniversalOp(),
            mC.element_type,
            num_bits_per_copy=16,
        ),
        tiled_mma,
    )
    cute_ext.partition_and_copy(
        copy_r2g_c.get_slice(tid_x), bufferRC, gC_tile
    )


@cute.experimental.jit
def square_gemm_host(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    M = cute.size(mA, 0)
    N = cute.size(mC, 1)

    # Grid: one CTA per (16×8) tile
    grid = (M // 16, N // 8, 1)

    square_gemm_kernel(mA, mB, mC).launch(
        grid=grid,
        block=(32, 1, 1),  # single warp
    )


def main():
    N = 256  # must be divisible by 16 and 8
    A = torch.randn(N, N, device="cuda", dtype=torch.float16)
    B = torch.randn(N, N, device="cuda", dtype=torch.float16)
    C = torch.zeros(N, N, device="cuda", dtype=torch.float16)

    mA = from_dlpack(A)
    mB = from_dlpack(B)
    mC = from_dlpack(C)

    square_gemm_host(mA, mB, mC)
    torch.cuda.synchronize()

    ref = (A.float() @ B.float()).half()
    diff = (C.float() - ref.float()).abs().max().item()
    print(f"N={N}  max|C - ref| = {diff:.4f}")
    assert diff < 0.5, f"FAILED – max diff {diff}"
    print("✅ square_gemm OK")


if __name__ == "__main__":
    main()
