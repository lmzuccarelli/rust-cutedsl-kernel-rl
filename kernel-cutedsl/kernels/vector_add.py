import numpy as np
import cutlass
import cutlass.cute as cute
import torch

# 1. Define the Device Kernel Execution Strategy
@cute.kernel
def vector_add_kernel(x_ptr, y_ptr, out_ptr, num_elements):
    tidx, _, _ = cute.arch.thread_idx()
    idx = tidx 
    
    if idx < num_elements:
        out_ptr[idx] = x_ptr[idx] + y_ptr[idx]

# 2. Define the Host-Side JIT Configuration Launcher
@cute.jit
def run_vector_add(x, y, out, num_elements):
    cutlass.cuda.initialize_cuda_context()
    
    threads_per_block = 256
    blocks_per_grid = 1
    
    # Bind the tensors directly to the kernel call, then supply grid configurations
    vector_add_kernel(x, y, out, num_elements).launch(
        grid=(blocks_per_grid, 1, 1), 
        block=(threads_per_block, 1, 1)
    )

if __name__ == "__main__":
    N = 256
    
    # Allocate device tensors directly inside PyTorch memory structures
    x_input = torch.ones(N, dtype=torch.float32, device="cuda") * 1.5
    y_input = torch.ones(N, dtype=torch.float32, device="cuda") * 2.5
    out_result = torch.zeros(N, dtype=torch.float32, device="cuda")
    
    print("Compiling the CuTe DSL down to PTX assembly...")
    
    # Pass the baseline arguments so the JIT engine captures the structural signature
    compiled_runner = cute.compile(run_vector_add, x_input, y_input, out_result, N)
    
    print("Executing compiled hardware loops...")
    for i in range(5):
        compiled_runner(x_input, y_input, out_result, N)
        
    print("Verification Check (Index 0 Output):", out_result[0].item())
