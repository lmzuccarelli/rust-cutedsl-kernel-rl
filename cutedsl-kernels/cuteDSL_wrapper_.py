## from ~/lkesem/vllm/.venv/lib/python3.12/site-packages/flashinfer/gemm/gemm_base.py

@backend_requirement(
    {
        "cudnn": _cudnn_gemm_fp4_requirement,
        "trtllm": _trtllm_gemm_fp4_requirement,
        "cutlass": _cutlass_gemm_fp4_requirement,
        "cute-dsl": _cute_dsl_gemm_fp4_requirement,
        "cutedsl_low_latency": _cutedsl_low_latency_gemm_fp4_requirement,
        "b12x": _b12x_gemm_fp4_requirement,
    },
    common_check=_check_mm_fp4_problem_size,
    heuristic_func=_heuristic_func_mm_fp4,  # result stored in mm_fp4.suitable_auto_backends
)
@flashinfer_api(trace=mm_fp4_trace)
def mm_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
    block_size: int = 16,
    use_8x4_sf_layout: bool = False,
    backend: Literal[
        "cudnn", "trtllm", "cutlass", "cute-dsl", "cutedsl_low_latency", "b12x", "auto"
    ] = "auto",
    use_nvfp4: bool = True,
    enable_pdl: bool = True,
) -> torch.Tensor:
    r"""MM FP4

    Parameters
    ----------
    a: torch.Tensor
        Input tensor, shape (m, k), fp4 e2m1fn_x2 or uint8.

    b: torch.Tensor
        Mat2 tensor, shape (k, n), should be column major, fp4 e2m1fn_x2 or uint8.

    a_descale: torch.Tensor
        Block scale tensor for A, shape (m, k // block_size), float8_e4m3fn or uint8.

    b_descale: torch.Tensor
        Block scale tensor for B, shape (k, n // block_size), float8_e4m3fn or uint8.

    alpha: Optional[torch.Tensor]
        Global scale tensor, float scalar.

    out_dtype: torch.dtype
        Output dtype, bf16 or fp16. When ``backend="trtllm"``, only ``bf16`` is supported.

    out: Optional[torch.Tensor]
        Out tensor, shape (m, n), bf16 or fp16, defaults to ``None``.

    block_size: int
        Block size for FP4 quantization, only 16 and 32 are supported. 16 in case of nvfp4 quantization. 32 in case of mxfp4 quantization.

    use_8x4_sf_layout: bool
        Whether to use 8x4 scale factor layout or 128x4 scale factor layout, defaults to False.

    backend: Literal["cudnn", "trtllm", "cutlass", "cute-dsl", "cutedsl_low_latency", "b12x", "auto"]
        Backend to use, defaults to ``"auto"``. On SM120, ``"auto"`` prefers
        ``"b12x"`` (NVFP4 only), then ``"cutlass"``, then ``"cudnn"``. On other
        architectures, ``"auto"`` selects between ``"cudnn"`` and ``"cutlass"``
        based on the current CUDA and cuDNN versions. The ``"trtllm"`` and
        ``"cute-dsl"`` backends are never auto-selected because they require
        different weight preparation. The ``"cutedsl_low_latency"`` backend is the last
        heuristic candidate for eligible SM100/SM103 problems and requires
        ``M <= 8``, 128x4 scale factors, and K divisible by 64 for NVFP4 or 128
        for MXFP4.

    use_nvfp4: bool
        Whether to use nvfp4 quantization or mxfp4 quantization, defaults to ``True``.
        See the ``block_size`` parameter for related constraints.

    enable_pdl: bool
        Whether to enable Programmatic Dependent Launch (PDL) for the ``cute_dsl``
        and ``cutedsl_low_latency`` backends, defaults to ``True``. PDL allows overlapping
        the tail of one kernel with the start of the next for reduced launch latency.
        This parameter is ignored by other backends.

    Notes
    -----
    When cudnn/cutlass backend is used, both a and b should quantized with nvfp4_quantize using the 128x4 scale factor layout and do_shuffle=False.
    When trtllm backend is used, b must be quantized with 128x4 layout and `do_shuffle=True`. a can be quantized with either 128x4 or 8x4 layout (controlled by `use_8x4_sf_layout`) and `do_shuffle=False`.
    When cute_dsl backend is used, both a and b should be quantized with 128x4 scale factor layout:
    nvfp4_quantize(..., do_shuffle=False) for NVFP4, or mxfp4_quantize(...) for MXFP4.
    The cutedsl_low_latency backend uses the same 128x4 scale factor layouts.

    Returns
    -------
    out: torch.Tensor
        Out tensor, shape (m, n), bf16 or fp16.

    Examples
    --------
    >>> import torch
    >>> from flashinfer import nvfp4_quantize, mm_fp4, SfLayout
    >>> a = torch.randn([48, 128], device="cuda", dtype=torch.bfloat16)
    >>> b = torch.randn([256, 128], device="cuda", dtype=torch.bfloat16)
    >>> a_global_sf = (448 * 6) / a.float().abs().nan_to_num().max()
    >>> b_global_sf = (448 * 6) / b.float().abs().nan_to_num().max()
    >>> a_fp4, a_sf = nvfp4_quantize(a, a_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    >>> b_fp4, b_sf = nvfp4_quantize(b, b_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=True)
    >>> out = mm_fp4(a_fp4, b_fp4.T, a_sf, b_sf.T, 1.0/(a_global_sf * b_global_sf), torch.bfloat16, None, backend="trtllm")
    >>> out.shape
    torch.Size([48, 256])
    """

    # allocate the output tensor if not provided
    if out is None:
        out = torch.empty(
            (a.shape[0], b.shape[1]),
            device=a.device,
            dtype=out_dtype,
        )

    workspace_buffer = _get_cache_buf(
        "mm_fp4_workspace", DEFAULT_WORKSPACE_SIZE, a.device
    )

    # Auto-select the best backend
    if backend == "auto":
        backends = mm_fp4.suitable_auto_backends
    else:
        backends = [backend]

    # At this point, backends contains a supported backend if specified, or all supported backends if backend='auto'.
    # Lazy initialization of runners to avoid overhead of creating a new runner that will not be used
    major, minor = get_compute_capability(a.device)

    tuner = AutoTuner.get()
    tuning_config = (
        _MM_FP4_TUNING_CONFIG_8x4 if use_8x4_sf_layout else _MM_FP4_TUNING_CONFIG_128x4
    )

    backend_to_runner_factory = {
        "cudnn": lambda: _cudnn_gemm_fp4_runner(tuning_config),
        "trtllm": lambda: get_trtllm_gemm_module().trtllm_fp4_gemm_runner(
            use_8x4_sf_layout
        ),
        "cutlass": lambda: get_cutlass_fp4_gemm_module(
            major, minor
        ).cutlass_fp4_gemm_runner(),
        "cute-dsl": lambda: _cute_dsl_gemm_fp4_runner(
            major, minor, enable_pdl, out_dtype, use_nvfp4
        ),
        "cutedsl_low_latency": lambda: _cutedsl_low_latency_blockscaled_gemm_runner(
            major * 10 + minor, enable_pdl
        ),
        "b12x": lambda: _b12x_gemm_fp4_runner(
            major, minor, enable_pdl, out_dtype, use_nvfp4
        ),
    }
    runners = [backend_to_runner_factory[cur_backend]() for cur_backend in backends]

    inputs = [
        a,
        b,
        a_descale,
        b_descale,
        alpha,
        out_dtype,
        out,
        block_size,
        use_nvfp4,
        workspace_buffer,
    ]
    runner, tactic = tuner.choose_one(
        "fp4_gemm",
        runners,
        tuning_config,
        inputs,
    )

    runner(inputs=inputs, tactic=tactic)
    return out

#~/lkesem/vllm/.venv/lib/python3.12/site-packages/flashinfer/gemm/gemm_base.py (calling this Sm120B12xBlockScaledDenseGemmKernel)

    from .kernels.dense_blockscaled_gemm_sm120_b12x import (
        Sm120B12xBlockScaledDenseGemmKernel,
        _select_default_dense_gemm_plan,
    )
def _b12x_gemm_fp4_runner(
    sm_major: int,
    sm_minor: int,
    enable_pdl: bool,
    out_dtype: torch.dtype,
    use_nvfp4: bool,
):
    """Create a b12x FP4 GEMM runner for SM120.

    Uses the SM120 warp-level MMA kernel with underfill tile selection.
    """
    import cutlass

    from .kernels.dense_blockscaled_gemm_sm120_b12x import (
        Sm120B12xBlockScaledDenseGemmKernel,
        _select_default_dense_gemm_plan,
    )

    from ..cute_dsl.utils import torch_to_cutlass_dtype

    if out_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            f"b12x backend does not support output dtype {out_dtype}. "
            f"Supported: torch.bfloat16, torch.float16."
        )
    c_cutlass_dtype = torch_to_cutlass_dtype(out_dtype)

    def _default_dense_plan(m, n, real_k, device):
        return _select_default_dense_gemm_plan(
            m, n, real_k, get_device_sm_count(device), expected_m=m
        )

    class B12xFp4GemmRunner(TunableRunner):
        """TunableRunner for b12x block-scaled FP4 dense GEMM on SM120.

        Tactics are tuples:
            (mma_tiler_mn, cluster_shape_mn, swap_ab, use_prefetch, kernel_type, use_tma_store)
        where kernel_type is always "sm120" and use_tma_store is always None.
        """

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> list:
            (a, b, a_descale, b_descale, alpha, _, out, _, _, _) = inputs
            m = a.shape[0]
            k_packed = a.shape[1]
            n = b.shape[1]
            real_k = k_packed * 2

            sf_vec_size = 16 if use_nvfp4 else 32
            ab_dtype = cutlass.Float4E2M1FN
            sf_dtype = cutlass.Float8E4M3FN if use_nvfp4 else cutlass.Float8E8M0FNU
            batch_size = 1

            valid_tactics = []

            def _add(mma_tiler_mn, swap_ab):
                # can_implement takes no m, so validity is M-independent.
                if not Sm120B12xBlockScaledDenseGemmKernel.can_implement(
                    ab_dtype,
                    sf_dtype,
                    sf_vec_size,
                    c_cutlass_dtype,
                    mma_tiler_mn,
                    (1, 1),
                    n,
                    real_k,
                    batch_size,
                    "k",
                    "k",
                    "n",
                    swap_ab=swap_ab,
                ):
                    return
                for use_prefetch in (False, True):
                    tac = (mma_tiler_mn, (1, 1), swap_ab, use_prefetch, "sm120", None)
                    if tac not in valid_tactics:
                        valid_tactics.append(tac)

            # A few balanced swap_ab-free tiles for the tuner to profile (a larger
            # grid overfit the bucket representative and made picks noisier).
            for mma_tiler_mn in [(64, 64), (64, 128), (128, 64), (128, 128)]:
                _add(mma_tiler_mn, swap_ab=False)

            # Also include the default-path tile (may be a narrow-N swap_ab tile
            # absent from the set above) so the tuner can't pick worse than static.
            plan = _default_dense_plan(m, n, real_k, a.device)
            _add(plan.mma_tiler_mn, swap_ab=plan.swap_ab)
            return valid_tactics

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic=None,
            do_preparation: bool = False,
            **kwargs,
        ):
            (a, b, a_descale, b_descale, alpha_tensor, _, out, _, _, _) = inputs
            m = a.shape[0]
            k_packed = a.shape[1]
            n = b.shape[1]
            real_k = k_packed * 2

            sf_vec_size = 16 if use_nvfp4 else 32
            sf_dtype = cutlass.Float8E4M3FN if use_nvfp4 else cutlass.Float8E8M0FNU
            batch_size = 1

            if tactic is None or tactic == -1:
                # Default path: the m-aware plan picks the tile (and swap_ab for
                # narrow-N) for this shape; expected_m=m since m is the actual size.
                plan = _default_dense_plan(m, n, real_k, a.device)
                tactic = (
                    plan.mma_tiler_mn,
                    (1, 1),
                    plan.swap_ab,
                    False,
                    "sm120",
                    None,
                )

            (
                mma_tiler_mn,
                cluster_shape_mn,
                swap_ab,
                use_prefetch,
                kernel_type,
                use_tma_store,
            ) = tactic

            kernel_a, kernel_b = a, b.T
            kernel_a_sf, kernel_b_sf = a_descale, b_descale.T

            sf_m = (m + 127) // 128
            sf_n = (n + 127) // 128
            sf_k = (real_k // sf_vec_size + 3) // 4

            cache_key = (
                sf_vec_size,
                mma_tiler_mn,
                cluster_shape_mn,
                swap_ab,
                use_prefetch,
                kernel_type,
                use_tma_store,
                enable_pdl,
                out_dtype,
            )

            # ctor takes mma_k/tile_k/single_work_tile_per_cta before use_prefetch,
            # so pass the later args by keyword to avoid mis-binding.
            make_kernel = lambda: Sm120B12xBlockScaledDenseGemmKernel(
                sf_vec_size,
                mma_tiler_mn,
                cluster_shape_mn,
                use_prefetch=use_prefetch,
                enable_pdl=enable_pdl,
                swap_ab=swap_ab,
            )

            # swap_ab is device-internal (applied in the kernel ctor); public C
            # stays row-major (m, n). Pass swap_ab=False to the harness so it keeps
            # the (m, n) output convention, not the SM100 operand-swap one.
            compiled_gemm, _ = _compile_block_scaled_gemm(
                _B12X_MM_FP4_KERNEL_CACHE,
                cache_key,
                make_kernel,
                ab_cutlass_dtype=cutlass.Uint8,
                sf_dtype=sf_dtype,
                c_cutlass_dtype=c_cutlass_dtype,
                ab_assumed_align=32,
                cluster_shape_mn=cluster_shape_mn,
                swap_ab=False,
                sf_m=sf_m,
                sf_n=sf_n,
                sf_k=sf_k,
                batch_size=batch_size,
            )

            alpha_for_launch = _prepare_alpha_for_launch(alpha_tensor, a.device)

            # `out` passed as-is (row-major (m, n)).
            compiled_gemm(
                kernel_a,
                kernel_b,
                out,
                sf_m,
                sf_n,
                sf_k,
                kernel_a_sf.data_ptr(),
                kernel_b_sf.data_ptr(),
                alpha_for_launch,
            )
            return out

    return B12xFp4GemmRunner()


