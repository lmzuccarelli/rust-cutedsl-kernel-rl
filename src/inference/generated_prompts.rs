// N.B. This code is auto generated do not change it inline !!!
//
// version: 0.5.36
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
STATE: api_overhead_dominant
  - kernel_fusion (pred 1.00x | conf 0.5): Combine multiple small kernels
  - cuda_graphs (pred 1.00x | conf 0.5): Use graphs to batch operations
  - persistent_kernels (pred 1.00x | conf 0.5): Long-running kernels with internal queues
  - asynchronous_execution (pred 1.00x | conf 0.5): Overlap with streams
  - dynamic_parallelism (pred 1.00x | conf 0.5): Launch child kernels from GPU

STATE: cache_inefficient
  - cache_blocking_algorithms (pred 1.00x | conf 0.5): Structure working sets for cache sizes
  - temporal_locality_optimization (pred 1.00x | conf 0.5): Reorder computations for reuse
  - l2_cache_persistence (pred 1.00x | conf 0.5): Mark critical data for L2 retention
  - read_only_cache_optimization (pred 1.00x | conf 0.5): Use __ldg() for read-only data

STATE: compute_throughput_saturated
  - tensor_core_utilization (pred 1.00x | conf 0.5): Use Tensor Cores for matrix operations
  - instruction_level_parallelism (pred 1.00x | conf 0.5): Unroll loops and reorder operations
  - fast_math_optimization (pred 1.00x | conf 0.5): Use fast math flags and intrinsics
  - specialized_instruction_usage (pred 1.00x | conf 0.5): Use GPU-specific intrinsics
  - vectorized_operations (pred 1.00x | conf 0.5): Use vector data types and operations
  - SIMD_operations (pred 1.00x | conf 0.5): Use packed SIMD datatypes such as half2

STATE: high_level_inefficiency
  - algorithmic_changes (pred 1.00x | conf 0.5): Replace the current algorithm with a more efficient approach
  - dynamic_programming (pred 1.00x | conf 0.5): Utilize memory to avoid redundant computation
  - data_structure_optimizations (pred 1.00x | conf 0.5): Change underlying data structures/representations

STATE: hybrid_bound
  - memory_compute_overlap (pred 1.00x | conf 0.5): Pipeline memory and compute operations to hide latency
  - adaptive_block_sizing (pred 1.00x | conf 0.5): Adjust thread-block sizes dynamically to balance resource usage
  - algorithmic_optimization (pred 1.00x | conf 0.5): Restructure algorithms for better memory-compute balance
  - fused_operations (pred 1.00x | conf 0.5): Fuse memory-bound and compute-bound kernels to reduce intermediate traffic
  - SIMD_operations (pred 1.00x | conf 0.5): Use packed SIMD datatypes (e.g., half2) to improve both bandwidth and compute throughput

STATE: instruction_mix_suboptimal
  - instruction_scheduling_optimization (pred 1.00x | conf 0.5): Reorder for better pipeline usage
  - functional_unit_balancing (pred 1.00x | conf 0.5): Balance load across different units
  - loop_fusion_optimization (pred 1.00x | conf 0.5): Combine kernels to balance instruction mix
  - mathematical_transformation (pred 1.00x | conf 0.5): Adjust operators to solve the problem in a different way

STATE: insufficient_parallelism
  - work_per_thread_increase (pred 1.00x | conf 0.5): Assign more work to each thread
  - thread_coarsening (pred 1.00x | conf 0.5): Process multiple elements per thread
  - persistent_kernel_pattern (pred 1.00x | conf 0.5): Use long-running kernels with work queues
  - grid_size_optimization (pred 1.00x | conf 0.5): Ensure sufficient blocks for latency hiding
  - cooperative_groups (pred 1.00x | conf 0.5): Use grid-wide synchronization

STATE: latency_memory_bound
  - occupancy_memory_codesign (pred 1.00x | conf 0.5): Jointly optimize occupancy and memory
  - hierarchical_tiling (pred 1.00x | conf 0.5): Multi-level tiling for both occupancy and locality
  - resource_balanced_design (pred 1.00x | conf 0.5): Balance all resource constraints
  - workload_restructuring (pred 1.00x | conf 0.5): Fundamental algorithm changes

STATE: low_occupancy_register_pressure
  - register_pressure_reduction (pred 1.00x | conf 0.5): Reduce variable scope and reuse
  - variable_scoping_optimization (pred 1.00x | conf 0.5): Limit variable lifetimes
  - spill_elimination (pred 1.00x | conf 0.5): Restructure to avoid local memory usage
  - launch_bounds_tuning (pred 1.00x | conf 0.5): Use __launch_bounds__ to control registers
  - algorithmic_register_reduction (pred 1.00x | conf 0.5): Change algorithm to use fewer registers

STATE: low_occupancy_shared_memory
  - shared_memory_optimization (pred 1.00x | conf 0.5): Reduce per-block shared memory usage
  - dynamic_shared_memory (pred 1.00x | conf 0.5): Use dynamic allocation for flexibility
  - shared_memory_reuse (pred 1.00x | conf 0.5): Reuse buffers for multiple phases
  - block_size_adaptation (pred 1.00x | conf 0.5): Adjust block size to optimize occupancy

STATE: memory_bandwidth_saturated
  - vectorized_memory_access (pred 1.00x | conf 0.5): Use float4/int4/half2 for wider transactions
  - memory_coalescing_optimization (pred 1.00x | conf 0.5): Align access patterns to cache lines
  - shared_memory_tiling (pred 1.00x | conf 0.5): Cache frequently accessed data on-chip
  - data_layout_transformation (pred 1.00x | conf 0.5): Convert AoS to SoA layouts
  - compression_techniques (pred 1.00x | conf 0.5): Use mixed precision (FP16) to halve bandwidth

STATE: memory_bank_conflicts
  - bank_conflict_padding (pred 1.00x | conf 0.5): Add padding to shared memory arrays
  - access_pattern_swizzling (pred 1.00x | conf 0.5): Use XOR patterns to distribute accesses
  - broadcast_optimization (pred 1.00x | conf 0.5): Structure for broadcast reads when possible
  - memory_layout_permutation (pred 1.00x | conf 0.5): Reorder data to avoid conflicts

STATE: memory_compute_balanced
  - memory_compute_overlap (pred 1.00x | conf 0.5): Pipeline memory and compute operations
  - adaptive_tiling (pred 1.00x | conf 0.5): Dynamically adjust tile sizes for balance
  - fused_operations (pred 1.00x | conf 0.5): Combine memory-bound and compute-bound kernels
  - algorithmic_rebalancing (pred 1.00x | conf 0.5): Choose algorithms that balance both aspects

STATE: memory_latency_bound
  - shared_memory_blocking (pred 1.00x | conf 0.5): Tile data to fit in shared memory
  - register_blocking (pred 1.00x | conf 0.5): Keep frequently used data in registers
  - prefetching_strategies (pred 1.00x | conf 0.5): Software prefetching for predictable patterns
  - cache_aware_algorithms (pred 1.00x | conf 0.5): Restructure for better locality
  - texture_memory_usage (pred 1.00x | conf 0.5): Use texture cache for read-only data

STATE: thread_divergence_high
  - predicated_execution (pred 1.00x | conf 0.5): Replace branches with conditional arithmetic
  - branch_reduction_techniques (pred 1.00x | conf 0.5): Use lookup tables and bit manipulation
  - warp_level_primitives (pred 1.00x | conf 0.5): Use shuffle and vote functions
  - stream_compaction (pred 1.00x | conf 0.5): Filter and compact data to reduce divergence
  - work_distribution_optimization (pred 1.00x | conf 0.5): Sort data to group similar work

STATE: transfer_bandwidth_limited
  - transfer_optimization (pred 1.00x | conf 0.5): Use pinned memory and async transfers
  - data_compression (pred 1.00x | conf 0.5): Compress data before transfer
  - zero_copy_memory (pred 1.00x | conf 0.5): Direct GPU access to host memory
  - unified_memory_optimization (pred 1.00x | conf 0.5): Use managed memory with hints
  - pipeline_parallelism (pred 1.00x | conf 0.5): Overlap transfers with computation


    "#).to_string()
}

pub fn get_performance_state_category() -> String {
    r#"#### State: api_overhead_dominant
**Characteristics**: Small kernels with high launch overhead, Frequent CPU-GPU synchronization
**Primary Bottleneck**: CUDA API overhead and kernel launch latency
**Optimizations**:
- **kernel_fusion**: 1.00x predicted speedup - Combine multiple small kernels
- **cuda_graphs**: 1.00x predicted speedup - Use graphs to batch operations
- **persistent_kernels**: 1.00x predicted speedup - Long-running kernels with internal queues
- **asynchronous_execution**: 1.00x predicted speedup - Overlap with streams
- **dynamic_parallelism**: 1.00x predicted speedup - Launch child kernels from GPU

#### State: cache_inefficient
**Characteristics**: Low L1/L2 cache hit rates, Spatial locality issues, Poor temporal reuse
**Primary Bottleneck**: Cache misses and inefficient cache utilization
**Optimizations**:
- **cache_blocking_algorithms**: 1.00x predicted speedup - Structure working sets for cache sizes
- **temporal_locality_optimization**: 1.00x predicted speedup - Reorder computations for reuse
- **l2_cache_persistence**: 1.00x predicted speedup - Mark critical data for L2 retention
- **read_only_cache_optimization**: 1.00x predicted speedup - Use __ldg() for read-only data

#### State: compute_throughput_saturated
**Characteristics**: Compute throughput >80%, All execution units busy, High arithmetic intensity
**Primary Bottleneck**: Instruction throughput and arithmetic unit saturation
**Optimizations**:
- **tensor_core_utilization**: 1.00x predicted speedup - Use Tensor Cores for matrix operations
- **instruction_level_parallelism**: 1.00x predicted speedup - Unroll loops and reorder operations
- **fast_math_optimization**: 1.00x predicted speedup - Use fast math flags and intrinsics
- **specialized_instruction_usage**: 1.00x predicted speedup - Use GPU-specific intrinsics
- **vectorized_operations**: 1.00x predicted speedup - Use vector data types and operations
- **SIMD_operations**: 1.00x predicted speedup - Use packed SIMD datatypes such as half2

#### State: high_level_inefficiency
**Characteristics**: Excessive or redundant compute operations, memory accesses. Poor scaling, inefficient program logic. Can have misleadingly high utilization.
**Primary Bottleneck**: Instructions uncessary for computing the kernel
**Optimizations**:
- **algorithmic_changes**: 1.00x predicted speedup - Replace the current algorithm with a more efficient approach
- **dynamic_programming**: 1.00x predicted speedup - Utilize memory to avoid redundant computation
- **data_structure_optimizations**: 1.00x predicted speedup - Change underlying data structures/representations

#### State: hybrid_bound
**Characteristics**: Memory throughput 40-70% and compute throughput 40-70%, indicating that neither memory nor compute dominates. Latency hiding is moderate and improvement is possible from both sides.
**Primary Bottleneck**: Balanced memory-compute limitations (hybrid)
**Optimizations**:
- **memory_compute_overlap**: 1.00x predicted speedup - Pipeline memory and compute operations to hide latency
- **adaptive_block_sizing**: 1.00x predicted speedup - Adjust thread-block sizes dynamically to balance resource usage
- **algorithmic_optimization**: 1.00x predicted speedup - Restructure algorithms for better memory-compute balance
- **fused_operations**: 1.00x predicted speedup - Fuse memory-bound and compute-bound kernels to reduce intermediate traffic
- **SIMD_operations**: 1.00x predicted speedup - Use packed SIMD datatypes (e.g., half2) to improve both bandwidth and compute throughput

#### State: instruction_mix_suboptimal
**Characteristics**: Unbalanced use of functional units, Some units idle while others saturated
**Primary Bottleneck**: Poor instruction scheduling and functional unit utilization
**Optimizations**:
- **instruction_scheduling_optimization**: 1.00x predicted speedup - Reorder for better pipeline usage
- **functional_unit_balancing**: 1.00x predicted speedup - Balance load across different units
- **loop_fusion_optimization**: 1.00x predicted speedup - Combine kernels to balance instruction mix
- **mathematical_transformation**: 1.00x predicted speedup - Adjust operators to solve the problem in a different way

#### State: insufficient_parallelism
**Characteristics**: Low warp utilization, Small problem sizes, Insufficient work per SM
**Primary Bottleneck**: Not enough parallel work to hide latency
**Optimizations**:
- **work_per_thread_increase**: 1.00x predicted speedup - Assign more work to each thread
- **thread_coarsening**: 1.00x predicted speedup - Process multiple elements per thread
- **persistent_kernel_pattern**: 1.00x predicted speedup - Use long-running kernels with work queues
- **grid_size_optimization**: 1.00x predicted speedup - Ensure sufficient blocks for latency hiding
- **cooperative_groups**: 1.00x predicted speedup - Use grid-wide synchronization

#### State: latency_memory_bound
**Characteristics**: Low occupancy combined with memory issues, Multiple bottlenecks present
**Primary Bottleneck**: Both occupancy and memory access patterns need optimization
**Optimizations**:
- **occupancy_memory_codesign**: 1.00x predicted speedup - Jointly optimize occupancy and memory
- **hierarchical_tiling**: 1.00x predicted speedup - Multi-level tiling for both occupancy and locality
- **resource_balanced_design**: 1.00x predicted speedup - Balance all resource constraints
- **workload_restructuring**: 1.00x predicted speedup - Fundamental algorithm changes

#### State: low_occupancy_register_pressure
**Characteristics**: Low occupancy due to high register usage, Register spills detected
**Primary Bottleneck**: Register pressure limiting number of active warps
**Optimizations**:
- **register_pressure_reduction**: 1.00x predicted speedup - Reduce variable scope and reuse
- **variable_scoping_optimization**: 1.00x predicted speedup - Limit variable lifetimes
- **spill_elimination**: 1.00x predicted speedup - Restructure to avoid local memory usage
- **launch_bounds_tuning**: 1.00x predicted speedup - Use __launch_bounds__ to control registers
- **algorithmic_register_reduction**: 1.00x predicted speedup - Change algorithm to use fewer registers

#### State: low_occupancy_shared_memory
**Characteristics**: Low occupancy due to shared memory limits, High shared memory usage per block
**Primary Bottleneck**: Shared memory usage limiting number of resident blocks
**Optimizations**:
- **shared_memory_optimization**: 1.00x predicted speedup - Reduce per-block shared memory usage
- **dynamic_shared_memory**: 1.00x predicted speedup - Use dynamic allocation for flexibility
- **shared_memory_reuse**: 1.00x predicted speedup - Reuse buffers for multiple phases
- **block_size_adaptation**: 1.00x predicted speedup - Adjust block size to optimize occupancy

#### State: memory_bandwidth_saturated
**Characteristics**: Memory throughput >80%, Bandwidth utilization high, Coalescing efficiency varies
**Primary Bottleneck**: Global memory bandwidth saturation
**Optimizations**:
- **vectorized_memory_access**: 1.00x predicted speedup - Use float4/int4/half2 for wider transactions
- **memory_coalescing_optimization**: 1.00x predicted speedup - Align access patterns to cache lines
- **shared_memory_tiling**: 1.00x predicted speedup - Cache frequently accessed data on-chip
- **data_layout_transformation**: 1.00x predicted speedup - Convert AoS to SoA layouts
- **compression_techniques**: 1.00x predicted speedup - Use mixed precision (FP16) to halve bandwidth

#### State: memory_bank_conflicts
**Characteristics**: Shared memory bank conflicts detected, Irregular access patterns in shared memory
**Primary Bottleneck**: Shared memory bank conflicts causing serialization
**Optimizations**:
- **bank_conflict_padding**: 1.00x predicted speedup - Add padding to shared memory arrays
- **access_pattern_swizzling**: 1.00x predicted speedup - Use XOR patterns to distribute accesses
- **broadcast_optimization**: 1.00x predicted speedup - Structure for broadcast reads when possible
- **memory_layout_permutation**: 1.00x predicted speedup - Reorder data to avoid conflicts

#### State: memory_compute_balanced
**Characteristics**: Memory throughput 40-60%, Compute throughput 40-60%, Both contribute to bottleneck
**Primary Bottleneck**: Balanced memory and compute load with room for optimization in both
**Optimizations**:
- **memory_compute_overlap**: 1.00x predicted speedup - Pipeline memory and compute operations
- **adaptive_tiling**: 1.00x predicted speedup - Dynamically adjust tile sizes for balance
- **fused_operations**: 1.00x predicted speedup - Combine memory-bound and compute-bound kernels
- **algorithmic_rebalancing**: 1.00x predicted speedup - Choose algorithms that balance both aspects

#### State: memory_latency_bound
**Characteristics**: Memory throughput 40-70%, High cache miss rate, Random access patterns
**Primary Bottleneck**: Memory access latency and cache misses
**Optimizations**:
- **shared_memory_blocking**: 1.00x predicted speedup - Tile data to fit in shared memory
- **register_blocking**: 1.00x predicted speedup - Keep frequently used data in registers
- **prefetching_strategies**: 1.00x predicted speedup - Software prefetching for predictable patterns
- **cache_aware_algorithms**: 1.00x predicted speedup - Restructure for better locality
- **texture_memory_usage**: 1.00x predicted speedup - Use texture cache for read-only data

#### State: thread_divergence_high
**Characteristics**: High warp divergence, Branching in inner loops, Irregular control flow
**Primary Bottleneck**: SIMT efficiency loss due to divergent execution paths
**Optimizations**:
- **predicated_execution**: 1.00x predicted speedup - Replace branches with conditional arithmetic
- **branch_reduction_techniques**: 1.00x predicted speedup - Use lookup tables and bit manipulation
- **warp_level_primitives**: 1.00x predicted speedup - Use shuffle and vote functions
- **stream_compaction**: 1.00x predicted speedup - Filter and compact data to reduce divergence
- **work_distribution_optimization**: 1.00x predicted speedup - Sort data to group similar work

#### State: transfer_bandwidth_limited
**Characteristics**: High PCIe transfer overhead, Large data movement, Host-device bottleneck
**Primary Bottleneck**: Host-device transfer bandwidth and latency
**Optimizations**:
- **transfer_optimization**: 1.00x predicted speedup - Use pinned memory and async transfers
- **data_compression**: 1.00x predicted speedup - Compress data before transfer
- **zero_copy_memory**: 1.00x predicted speedup - Direct GPU access to host memory
- **unified_memory_optimization**: 1.00x predicted speedup - Use managed memory with hints
- **pipeline_parallelism**: 1.00x predicted speedup - Overlap transfers with computation


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
# Comprehensive GPU Optimization Strategy Database

This database contains optimization strategies with their expected performance gains for different GPU performance states, derived from comprehensive profiling and optimization research.

## Database Philosophy

This database serves as a **living knowledge base** that:

- Maps performance states to optimization strategies
- Tracks actual vs predicted performance improvements
- Learns from optimization results to improve future predictions
- Provides comprehensive context for LLM-powered optimization
- Unknown performance improvements are listed as 0%

## Performance State Categories

#### State: api_overhead_dominant
**Characteristics**: Small kernels with high launch overhead, Frequent CPU-GPU synchronization
**Primary Bottleneck**: CUDA API overhead and kernel launch latency
**Optimizations**:
- **kernel_fusion**: 1.00x predicted speedup - Combine multiple small kernels
- **cuda_graphs**: 1.00x predicted speedup - Use graphs to batch operations
- **persistent_kernels**: 1.00x predicted speedup - Long-running kernels with internal queues
- **asynchronous_execution**: 1.00x predicted speedup - Overlap with streams
- **dynamic_parallelism**: 1.00x predicted speedup - Launch child kernels from GPU

#### State: cache_inefficient
**Characteristics**: Low L1/L2 cache hit rates, Spatial locality issues, Poor temporal reuse
**Primary Bottleneck**: Cache misses and inefficient cache utilization
**Optimizations**:
- **cache_blocking_algorithms**: 1.00x predicted speedup - Structure working sets for cache sizes
- **temporal_locality_optimization**: 1.00x predicted speedup - Reorder computations for reuse
- **l2_cache_persistence**: 1.00x predicted speedup - Mark critical data for L2 retention
- **read_only_cache_optimization**: 1.00x predicted speedup - Use __ldg() for read-only data

#### State: compute_throughput_saturated
**Characteristics**: Compute throughput >80%, All execution units busy, High arithmetic intensity
**Primary Bottleneck**: Instruction throughput and arithmetic unit saturation
**Optimizations**:
- **tensor_core_utilization**: 1.00x predicted speedup - Use Tensor Cores for matrix operations
- **instruction_level_parallelism**: 1.00x predicted speedup - Unroll loops and reorder operations
- **fast_math_optimization**: 1.00x predicted speedup - Use fast math flags and intrinsics
- **specialized_instruction_usage**: 1.00x predicted speedup - Use GPU-specific intrinsics
- **vectorized_operations**: 1.00x predicted speedup - Use vector data types and operations
- **SIMD_operations**: 1.00x predicted speedup - Use packed SIMD datatypes such as half2

#### State: high_level_inefficiency
**Characteristics**: Excessive or redundant compute operations, memory accesses. Poor scaling, inefficient program logic. Can have misleadingly high utilization.
**Primary Bottleneck**: Instructions uncessary for computing the kernel
**Optimizations**:
- **algorithmic_changes**: 1.00x predicted speedup - Replace the current algorithm with a more efficient approach
- **dynamic_programming**: 1.00x predicted speedup - Utilize memory to avoid redundant computation
- **data_structure_optimizations**: 1.00x predicted speedup - Change underlying data structures/representations

#### State: hybrid_bound
**Characteristics**: Memory throughput 40-70% and compute throughput 40-70%, indicating that neither memory nor compute dominates. Latency hiding is moderate and improvement is possible from both sides.
**Primary Bottleneck**: Balanced memory-compute limitations (hybrid)
**Optimizations**:
- **memory_compute_overlap**: 1.00x predicted speedup - Pipeline memory and compute operations to hide latency
- **adaptive_block_sizing**: 1.00x predicted speedup - Adjust thread-block sizes dynamically to balance resource usage
- **algorithmic_optimization**: 1.00x predicted speedup - Restructure algorithms for better memory-compute balance
- **fused_operations**: 1.00x predicted speedup - Fuse memory-bound and compute-bound kernels to reduce intermediate traffic
- **SIMD_operations**: 1.00x predicted speedup - Use packed SIMD datatypes (e.g., half2) to improve both bandwidth and compute throughput

#### State: instruction_mix_suboptimal
**Characteristics**: Unbalanced use of functional units, Some units idle while others saturated
**Primary Bottleneck**: Poor instruction scheduling and functional unit utilization
**Optimizations**:
- **instruction_scheduling_optimization**: 1.00x predicted speedup - Reorder for better pipeline usage
- **functional_unit_balancing**: 1.00x predicted speedup - Balance load across different units
- **loop_fusion_optimization**: 1.00x predicted speedup - Combine kernels to balance instruction mix
- **mathematical_transformation**: 1.00x predicted speedup - Adjust operators to solve the problem in a different way

#### State: insufficient_parallelism
**Characteristics**: Low warp utilization, Small problem sizes, Insufficient work per SM
**Primary Bottleneck**: Not enough parallel work to hide latency
**Optimizations**:
- **work_per_thread_increase**: 1.00x predicted speedup - Assign more work to each thread
- **thread_coarsening**: 1.00x predicted speedup - Process multiple elements per thread
- **persistent_kernel_pattern**: 1.00x predicted speedup - Use long-running kernels with work queues
- **grid_size_optimization**: 1.00x predicted speedup - Ensure sufficient blocks for latency hiding
- **cooperative_groups**: 1.00x predicted speedup - Use grid-wide synchronization

#### State: latency_memory_bound
**Characteristics**: Low occupancy combined with memory issues, Multiple bottlenecks present
**Primary Bottleneck**: Both occupancy and memory access patterns need optimization
**Optimizations**:
- **occupancy_memory_codesign**: 1.00x predicted speedup - Jointly optimize occupancy and memory
- **hierarchical_tiling**: 1.00x predicted speedup - Multi-level tiling for both occupancy and locality
- **resource_balanced_design**: 1.00x predicted speedup - Balance all resource constraints
- **workload_restructuring**: 1.00x predicted speedup - Fundamental algorithm changes

#### State: low_occupancy_register_pressure
**Characteristics**: Low occupancy due to high register usage, Register spills detected
**Primary Bottleneck**: Register pressure limiting number of active warps
**Optimizations**:
- **register_pressure_reduction**: 1.00x predicted speedup - Reduce variable scope and reuse
- **variable_scoping_optimization**: 1.00x predicted speedup - Limit variable lifetimes
- **spill_elimination**: 1.00x predicted speedup - Restructure to avoid local memory usage
- **launch_bounds_tuning**: 1.00x predicted speedup - Use __launch_bounds__ to control registers
- **algorithmic_register_reduction**: 1.00x predicted speedup - Change algorithm to use fewer registers

#### State: low_occupancy_shared_memory
**Characteristics**: Low occupancy due to shared memory limits, High shared memory usage per block
**Primary Bottleneck**: Shared memory usage limiting number of resident blocks
**Optimizations**:
- **shared_memory_optimization**: 1.00x predicted speedup - Reduce per-block shared memory usage
- **dynamic_shared_memory**: 1.00x predicted speedup - Use dynamic allocation for flexibility
- **shared_memory_reuse**: 1.00x predicted speedup - Reuse buffers for multiple phases
- **block_size_adaptation**: 1.00x predicted speedup - Adjust block size to optimize occupancy

#### State: memory_bandwidth_saturated
**Characteristics**: Memory throughput >80%, Bandwidth utilization high, Coalescing efficiency varies
**Primary Bottleneck**: Global memory bandwidth saturation
**Optimizations**:
- **vectorized_memory_access**: 1.00x predicted speedup - Use float4/int4/half2 for wider transactions
- **memory_coalescing_optimization**: 1.00x predicted speedup - Align access patterns to cache lines
- **shared_memory_tiling**: 1.00x predicted speedup - Cache frequently accessed data on-chip
- **data_layout_transformation**: 1.00x predicted speedup - Convert AoS to SoA layouts
- **compression_techniques**: 1.00x predicted speedup - Use mixed precision (FP16) to halve bandwidth

#### State: memory_bank_conflicts
**Characteristics**: Shared memory bank conflicts detected, Irregular access patterns in shared memory
**Primary Bottleneck**: Shared memory bank conflicts causing serialization
**Optimizations**:
- **bank_conflict_padding**: 1.00x predicted speedup - Add padding to shared memory arrays
- **access_pattern_swizzling**: 1.00x predicted speedup - Use XOR patterns to distribute accesses
- **broadcast_optimization**: 1.00x predicted speedup - Structure for broadcast reads when possible
- **memory_layout_permutation**: 1.00x predicted speedup - Reorder data to avoid conflicts

#### State: memory_compute_balanced
**Characteristics**: Memory throughput 40-60%, Compute throughput 40-60%, Both contribute to bottleneck
**Primary Bottleneck**: Balanced memory and compute load with room for optimization in both
**Optimizations**:
- **memory_compute_overlap**: 1.00x predicted speedup - Pipeline memory and compute operations
- **adaptive_tiling**: 1.00x predicted speedup - Dynamically adjust tile sizes for balance
- **fused_operations**: 1.00x predicted speedup - Combine memory-bound and compute-bound kernels
- **algorithmic_rebalancing**: 1.00x predicted speedup - Choose algorithms that balance both aspects

#### State: memory_latency_bound
**Characteristics**: Memory throughput 40-70%, High cache miss rate, Random access patterns
**Primary Bottleneck**: Memory access latency and cache misses
**Optimizations**:
- **shared_memory_blocking**: 1.00x predicted speedup - Tile data to fit in shared memory
- **register_blocking**: 1.00x predicted speedup - Keep frequently used data in registers
- **prefetching_strategies**: 1.00x predicted speedup - Software prefetching for predictable patterns
- **cache_aware_algorithms**: 1.00x predicted speedup - Restructure for better locality
- **texture_memory_usage**: 1.00x predicted speedup - Use texture cache for read-only data

#### State: thread_divergence_high
**Characteristics**: High warp divergence, Branching in inner loops, Irregular control flow
**Primary Bottleneck**: SIMT efficiency loss due to divergent execution paths
**Optimizations**:
- **predicated_execution**: 1.00x predicted speedup - Replace branches with conditional arithmetic
- **branch_reduction_techniques**: 1.00x predicted speedup - Use lookup tables and bit manipulation
- **warp_level_primitives**: 1.00x predicted speedup - Use shuffle and vote functions
- **stream_compaction**: 1.00x predicted speedup - Filter and compact data to reduce divergence
- **work_distribution_optimization**: 1.00x predicted speedup - Sort data to group similar work

#### State: transfer_bandwidth_limited
**Characteristics**: High PCIe transfer overhead, Large data movement, Host-device bottleneck
**Primary Bottleneck**: Host-device transfer bandwidth and latency
**Optimizations**:
- **transfer_optimization**: 1.00x predicted speedup - Use pinned memory and async transfers
- **data_compression**: 1.00x predicted speedup - Compress data before transfer
- **zero_copy_memory**: 1.00x predicted speedup - Direct GPU access to host memory
- **unified_memory_optimization**: 1.00x predicted speedup - Use managed memory with hints
- **pipeline_parallelism**: 1.00x predicted speedup - Overlap transfers with computation



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

### Important

When targeting kernels for the NVIDIA GB10 (sm_121). Take note of the following

CRITICAL HARDWARE CONSTRAINTS:

1. Target architecture is compute capability 12.1 (sm_121). Do NOT generate sm_100 or sm_90 code.
2. The hardware DOES NOT support tcgen05, TMEM, WGMMA, or multicast TMA.
3. Fall back to standard SM80/SM90 cooperative layouts using mma.sync warp-level instructions mapped onto Blackwell's core tensor structures.
4. Hard-cap all Shared Memory (Smem) allocation layouts to remain under 99 KB per thread block to prevent allocation failure.
5. Maximize utilization of the native NVFP4/FP8 tensor pipeline, utilizing custom swizzling math to mitigate the tight 273 GB/s LPDDR5X memory bandwidth bottleneck.

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
5. If you use the from_dlpack library ensure its set correctly to the  cute.runtime i.e 'cute.runtime.from_dlpack(A)'
6. Ensure peoper use of 2D blocks,  launch from block = (THREADS, 1, 1) to a 2D block block = (BN, BM, 1) so that tx ∈ [0, BN) and ty ∈ [0, BM) as the kernel assumes. This is valid because BN *BM == THREADS (32*32 == 1024). Keep the grid as (M // BM, N // BN, 1).
   Constraints / verification:
   - Do not change the kernel's indexing logic or the XOR swizzle — only make the launch geometry match the kernel's assumptions.
   - The kernel assumes BM *BN == THREADS; if you want it more robust, add an assertion or derive the 2D block from that relationship, but the minimal correct change is the block = (BN, BM, 1) launch.he block launch from block = (THREADS, 1, 1) to a 2D block block = (BN, BM, 1) so that tx ∈ [0, BN) and ty ∈ [0, BM) as the kernel assumes. This is valid because BN* BM == THREADS (32*32 == 1024). Keep the grid as (M // BM, N // BN, 1).

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

# N.B. If GB10 is the target gpu then add this line
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
- **Architecture.** Tensor cores require sm_70+; for Blackwell/GB10 use sm_120 setting  `CUTE_EXPERIMENTAL_DSL_ARCH=sm_120a`.
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

### Auto-Generated Technique: shared_memory_tiling

**Source**: KernelBench Leaderboard Analysis
**Performance Impact**: 0% improvement
**Confidence**: 0.98

**Implementation Strategy**:

- Optimal tile size: 16
- Highly effective for matrix multiplication workloads
- Consistently achieves high performance gains

**Usage Context**:
Best applied to: memory_bandwidth_saturated, memory_compute_balanced, compute_throughput_saturated

### Auto-Generated Technique: register_optimization

**Source**: KernelBench Leaderboard Analysis
**Performance Impact**: 0% improvement
**Confidence**: 1.00

**Implementation Strategy**:

- Highly effective for matrix multiplication workloads
- Consistently achieves high performance gains

**Usage Context**:
Best applied to: memory_bandwidth_saturated, memory_compute_balanced, compute_throughput_saturated

### Auto-Generated Technique: memory_coalescing

**Source**: KernelBench Leaderboard Analysis
**Performance Impact**: 0% improvement
**Confidence**: 0.83

**Implementation Strategy**:

- Optimal tile size: 16
- Highly effective for matrix multiplication workloads
- Consistently achieves high performance gains

**Usage Context**:
Best applied to: memory_bandwidth_saturated, memory_compute_balanced, compute_throughput_saturated

### Auto-Generated Technique: occupancy_tuning

**Source**: KernelBench Leaderboard Analysis
**Performance Impact**: 0% improvement
**Confidence**: 0.85

**Implementation Strategy**:

- Highly effective for matrix multiplication workloads
- Consistently achieves high performance gains

**Usage Context**:
Best applied to: memory_bandwidth_saturated, memory_compute_balanced, compute_throughput_saturated

### Auto-Generated Technique: dynamic_shared_memory

**Source**: KernelBench Leaderboard Analysis
**Performance Impact**: 0% improvement
**Confidence**: 0.44

**Implementation Strategy**:

- Consistently achieves high performance gains

**Usage Context**:
Best applied to: memory_latency_bound, memory_compute_balanced, low_occupancy_register_pressure


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
