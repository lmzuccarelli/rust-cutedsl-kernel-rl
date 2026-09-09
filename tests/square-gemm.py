import torch
import cutlass
import cutlass.cute as cute


# --- Kernel Definition ---
@cute.kernel
def sgemm_kernel(mA: cute.Tensor, mB: cute.Tensor, mD: cute.Tensor,
                 M: cutlass.Constexpr, N: cutlass.Constexpr, K: cutlass.Constexpr):
    """Compute D = A @ B.

    Grid is a single CTA of 32 threads. Each thread walks a strided subset of
    the MxN output tile and accumulates the K-reduction in fp32.
    """
    tid, _, _ = cute.arch.thread_idx()

    num_threads = 32
    # Cover every (m, n) output element with a grid-strided loop over the
    # flattened MxN index space.
    idx = tid
    while idx < M * N:
        m = idx // N
        n = idx % N

        acc = cutlass.Float32(0.0)
        for k in cutlass.range_constexpr(K):
            a = mA[m, k].to(cutlass.Float32)
            b = mB[k, n].to(cutlass.Float32)
            acc += a * b

        mD[m, n] = acc.to(cutlass.Float16)
        idx += num_threads


# --- Host-Side JIT Launcher ---
@cute.jit
def run_sgemm(mA: cute.Tensor, mB: cute.Tensor, mD: cute.Tensor,
              M: cutlass.Constexpr, N: cutlass.Constexpr, K: cutlass.Constexpr):
    cutlass.cuda.initialize_cuda_context()

    sgemm_kernel(mA, mB, mD, M, N, K).launch(
        grid=(1, 1, 1),
        block=(32, 1, 1),
    )


# --- Verification Harness ---
def main():
    # 1. Define dimensions
    M, N, K = 16, 8, 8
    device = torch.device("cuda:0")

    # 2. Initialize inputs on CUDA device as float16
    torch.manual_seed(42)

    A_torch = torch.randn((M, K), dtype=torch.float16, device=device)
    B_torch = torch.randn((K, N), dtype=torch.float16, device=device)
    D_torch = torch.zeros((M, N), dtype=torch.float16, device=device)

    print("Compiling and launching the CuTe DSL kernel...")

    # 3. Convert PyTorch tensors to CuTe tensors and compile the launcher.
    mA = cute.runtime.from_dlpack(A_torch)
    mB = cute.runtime.from_dlpack(B_torch)
    mD = cute.runtime.from_dlpack(D_torch)

    compiled = cute.compile(run_sgemm, mA, mB, mD, M, N, K)
    compiled(mA, mB, mD)
    torch.cuda.synchronize()

    # 4. Compute reference baseline using PyTorch
    D_ref = torch.matmul(A_torch, B_torch)

    # 5. Verify absolute and relative tolerances
    print("\n--- Verification Results ---")
    is_close = torch.allclose(D_torch, D_ref, rtol=1e-2, atol=1e-2)
    print(f"Arrays are close: {is_close}")

    if not is_close:
        max_diff = torch.max(torch.abs(D_torch - D_ref))
        print(f"Max absolute difference: {max_diff.item()}")
    else:
        print("Success! The kernel output matches the PyTorch reference.")


if __name__ == "__main__":
    main()
