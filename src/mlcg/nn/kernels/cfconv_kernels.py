"""
Fused Triton kernels for CFConv operations.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fused_cutoff_gather_multiply_scatter_kernel(
    # Input pointers
    x_ptr,  # [num_nodes, feature_dim]
    filter_out_ptr,  # [num_edges, feature_dim] - output from filter_network (FP32 or FP16)
    edge_weight_ptr,  # [num_edges] - distances for cutoff
    edge_src_ptr,  # [num_edges]
    edge_dst_ptr,  # [num_edges]
    output_ptr,  # [num_nodes, feature_dim]
    # Cutoff parameters
    cutoff_upper,
    # Sizes
    num_edges,
    feature_dim,
    # Block size
    BLOCK_SIZE_F: tl.constexpr,
    FILTER_FP16: tl.constexpr,  # Whether filter_out is FP16
):
    """
    Fused kernel that combines:
    1. Cutoff calculation: C = 0.5 * (cos(dist * pi / cutoff) + 1) * (dist < cutoff)
    2. Filter output scaling: W = filter_out * C
    3. Gather: x_j = x[src]
    4. Multiply: msg = x_j * W
    5. Scatter-add: output[dst] += msg

    This eliminates the intermediate W tensor allocation and fuses 5 operations
    into a single kernel pass.

    Supports both FP32 and FP16 filter_out input:
    - If FILTER_FP16=True: loads FP16, promotes to FP32 for computation
    - If FILTER_FP16=False: loads FP32 directly
    Output is always FP32.
    """
    edge_idx = tl.program_id(axis=0)

    if edge_idx >= num_edges:
        return

    # Load edge info
    src_node = tl.load(edge_src_ptr + edge_idx)
    dst_node = tl.load(edge_dst_ptr + edge_idx)
    dist = tl.load(edge_weight_ptr + edge_idx)

    # Compute cutoff inline (CosineCutoff formula)
    # C = 0.5 * (cos(dist * pi / cutoff_upper) + 1.0) * (dist < cutoff_upper)
    pi = 3.141592653589793
    cos_val = tl.cos(dist * pi / cutoff_upper)
    C = 0.5 * (cos_val + 1.0)
    # Apply distance mask
    mask_dist = dist < cutoff_upper
    C = tl.where(mask_dist, C, 0.0)

    for f_start in range(0, feature_dim, BLOCK_SIZE_F):
        f_offsets = f_start + tl.arange(0, BLOCK_SIZE_F)
        f_mask = f_offsets < feature_dim

        # load filter network output (FP16 or FP32)
        filter_val = tl.load(
            filter_out_ptr + edge_idx * feature_dim + f_offsets,
            mask=f_mask,
            other=0.0,
        )
        # Promote FP16 to FP32 for computation
        if FILTER_FP16:
            filter_val = filter_val.to(tl.float32)

        # cutoff: W = filter_out * C (broadcasting C across features)
        W = filter_val * C

        # gather: x_j = x[src]
        x_j = tl.load(
            x_ptr + src_node * feature_dim + f_offsets, mask=f_mask, other=0.0
        )

        # multiply: msg = x_j * W
        msg = x_j * W

        # scatter-add: output[dst] += msg
        tl.atomic_add(
            output_ptr + dst_node * feature_dim + f_offsets, msg, mask=f_mask
        )


def fused_cutoff_gather_multiply_scatter(
    x: torch.Tensor,
    filter_out: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    num_nodes: int,
    cutoff_upper: float,
) -> torch.Tensor:
    """
    Fused operation: cutoff + gather + multiply + scatter.

    Computes: output[dst] += x[src] * (filter_out * cutoff(edge_weight))

    This eliminates the intermediate W tensor and fuses cutoff calculation.

    Supports both FP32 and FP16 filter_out input (auto-detected).
    Output is always FP32.

    Parameters
    ----------
    x : torch.Tensor
        Node features [num_nodes, feature_dim]
    filter_out : torch.Tensor
        Filter network output [num_edges, feature_dim], can be FP32 or FP16
    edge_weight : torch.Tensor
        Edge weights (distances) [num_edges]
    edge_src : torch.Tensor
        Source node indices [num_edges]
    edge_dst : torch.Tensor
        Destination node indices [num_edges]
    num_nodes : int
        Number of nodes
    cutoff_upper : float
        Upper cutoff distance

    Returns
    -------
    torch.Tensor
        Output [num_nodes, feature_dim] in FP32
    """
    assert x.is_cuda and x.is_contiguous()
    assert filter_out.is_contiguous()
    assert edge_weight.is_contiguous()
    assert edge_src.is_contiguous()
    assert edge_dst.is_contiguous()

    feature_dim = x.shape[1]
    num_edges = edge_src.shape[0]

    output = torch.zeros(num_nodes, feature_dim, device=x.device, dtype=x.dtype)

    if num_edges == 0:
        return output

    # Choose block size
    BLOCK_SIZE_F = min(128, triton.next_power_of_2(feature_dim))

    # Auto-detect filter_out dtype
    filter_fp16 = filter_out.dtype == torch.float16

    grid = (num_edges,)

    fused_cutoff_gather_multiply_scatter_kernel[grid](
        x,
        filter_out,
        edge_weight,
        edge_src,
        edge_dst,
        output,
        cutoff_upper,
        num_edges,
        feature_dim,
        BLOCK_SIZE_F=BLOCK_SIZE_F,
        FILTER_FP16=filter_fp16,
    )

    return output


# ============================================================================
# Fused Backward Kernel for grad_filter_out
# ============================================================================


@triton.jit
def fused_grad_filter_out_kernel(
    # Input pointers
    x_ptr,  # [num_nodes, feature_dim]
    grad_output_ptr,  # [num_nodes, feature_dim]
    edge_weight_ptr,  # [num_edges]
    edge_src_ptr,  # [num_edges]
    edge_dst_ptr,  # [num_edges]
    grad_filter_out_ptr,  # [num_edges, feature_dim] - OUTPUT (FP32 or FP16)
    # Cutoff parameters
    cutoff_upper,
    # Sizes
    num_edges,
    feature_dim,
    # Block size
    BLOCK_SIZE_F: tl.constexpr,
    OUTPUT_FP16: tl.constexpr,  # Whether to output FP16
):
    """
    Fused kernel for grad_filter_out computation in CFConv backward pass.

    Computes:
        grad_filter_out[e] = x[src[e]] * grad_output[dst[e]] * cutoff(dist[e])

    Fuses:
        1. Gather x[edge_src]
        2. Gather grad_output[edge_dst]
        3. Cutoff computation
        4. Elementwise multiply

    Memory savings: Eliminates two intermediate tensors (x_gathered, grad_gathered)

    Supports FP16 output when OUTPUT_FP16=True (matches filter_out dtype).
    Computation is always done in FP32 for numerical stability.
    """
    edge_idx = tl.program_id(axis=0)

    if edge_idx >= num_edges:
        return

    # Load edge info
    src_node = tl.load(edge_src_ptr + edge_idx)
    dst_node = tl.load(edge_dst_ptr + edge_idx)
    dist = tl.load(edge_weight_ptr + edge_idx)

    # Compute cutoff inline (CosineCutoff formula)
    # C = 0.5 * (cos(dist * pi / cutoff_upper) + 1.0) * (dist < cutoff_upper)
    pi = 3.141592653589793
    cos_val = tl.cos(dist * pi / cutoff_upper)
    C = 0.5 * (cos_val + 1.0)
    mask_dist = dist < cutoff_upper
    C = tl.where(mask_dist, C, 0.0)

    # Process features in blocks
    for f_start in range(0, feature_dim, BLOCK_SIZE_F):
        f_offsets = f_start + tl.arange(0, BLOCK_SIZE_F)
        f_mask = f_offsets < feature_dim

        # Gather x[src]
        x_j = tl.load(
            x_ptr + src_node * feature_dim + f_offsets, mask=f_mask, other=0.0
        )

        # Gather grad_output[dst]
        grad_j = tl.load(
            grad_output_ptr + dst_node * feature_dim + f_offsets,
            mask=f_mask,
            other=0.0,
        )

        # Fused multiply: x * grad * C (in FP32)
        grad_filter = x_j * grad_j * C

        # Store result (convert to FP16 if needed)
        if OUTPUT_FP16:
            grad_filter = grad_filter.to(tl.float16)
        tl.store(
            grad_filter_out_ptr + edge_idx * feature_dim + f_offsets,
            grad_filter,
            mask=f_mask,
        )


def fused_grad_filter_out(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    cutoff_upper: float,
    out_dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Compute grad_filter_out in a single fused kernel.

    grad_filter_out[e] = x[src[e]] * grad_output[dst[e]] * cutoff(dist[e])

    Parameters
    ----------
    x : torch.Tensor
        Node features [num_nodes, feature_dim]
    grad_output : torch.Tensor
        Gradient of output [num_nodes, feature_dim]
    edge_weight : torch.Tensor
        Edge weights (distances) [num_edges]
    edge_src : torch.Tensor
        Source node indices [num_edges]
    edge_dst : torch.Tensor
        Destination node indices [num_edges]
    cutoff_upper : float
        Upper cutoff distance
    out_dtype : torch.dtype, optional
        Output dtype. If None, uses x.dtype. Supports FP32 or FP16.

    Returns
    -------
    torch.Tensor
        grad_filter_out [num_edges, feature_dim]
    """
    assert x.is_cuda and x.is_contiguous()
    assert grad_output.is_contiguous()
    assert edge_weight.is_contiguous()
    assert edge_src.is_contiguous()
    assert edge_dst.is_contiguous()

    feature_dim = x.shape[1]
    num_edges = edge_src.shape[0]

    # Default output dtype is x.dtype
    if out_dtype is None:
        out_dtype = x.dtype

    grad_filter_out = torch.empty(
        num_edges, feature_dim, device=x.device, dtype=out_dtype
    )

    if num_edges == 0:
        return grad_filter_out

    BLOCK_SIZE_F = min(128, triton.next_power_of_2(feature_dim))
    grid = (num_edges,)

    # Determine if output should be FP16
    output_fp16 = out_dtype == torch.float16

    fused_grad_filter_out_kernel[grid](
        x,
        grad_output,
        edge_weight,
        edge_src,
        edge_dst,
        grad_filter_out,
        cutoff_upper,
        num_edges,
        feature_dim,
        BLOCK_SIZE_F=BLOCK_SIZE_F,
        OUTPUT_FP16=output_fp16,
    )

    return grad_filter_out


class FusedCutoffGatherMultiplyScatterFunction(torch.autograd.Function):
    """Autograd function for fused cutoff + gather + multiply + scatter.

    Supports both FP32 and FP16 filter_out:
    - Forward: auto-detects filter_out dtype, always outputs FP32
    - Backward: grad_filter_out matches filter_out dtype

    Optionally uses src-CSR for atomic-free grad_x computation when
    src_ptr and src_perm are provided.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        filter_out,
        edge_weight,
        edge_src,
        edge_dst,
        num_nodes,
        cutoff_upper,
        src_ptr,
        src_perm,
    ):
        ctx.save_for_backward(
            x, filter_out, edge_weight, edge_src, edge_dst, src_ptr, src_perm
        )
        ctx.num_nodes = num_nodes
        ctx.cutoff_upper = cutoff_upper
        ctx.filter_out_dtype = filter_out.dtype  # Save dtype for backward
        return fused_cutoff_gather_multiply_scatter(
            x,
            filter_out,
            edge_weight,
            edge_src,
            edge_dst,
            num_nodes,
            cutoff_upper,
        )

    @staticmethod
    def backward(ctx, grad_output):
        x, filter_out, edge_weight, edge_src, edge_dst, src_ptr, src_perm = (
            ctx.saved_tensors
        )
        num_nodes = ctx.num_nodes
        cutoff_upper = ctx.cutoff_upper
        filter_out_dtype = ctx.filter_out_dtype

        # Ensure contiguous for kernel
        grad_output = grad_output.contiguous()

        grad_x = None
        grad_filter_out = None

        if ctx.needs_input_grad[0]:
            # grad_x[src] += grad_output[dst] * W
            if src_ptr is not None and src_perm is not None:
                # === Src-CSR Path (no atomics!) ===
                from .kernels.csr_kernels import fused_src_csr_grad_x

                grad_x = fused_src_csr_grad_x(
                    grad_output,
                    filter_out,
                    edge_weight,
                    edge_dst,
                    src_ptr,
                    src_perm,
                    num_nodes,
                    cutoff_upper,
                )
            else:
                # === Atomic scatter path ===
                grad_x = fused_cutoff_gather_multiply_scatter(
                    grad_output,
                    filter_out,
                    edge_weight,
                    edge_dst,
                    edge_src,
                    num_nodes,
                    cutoff_upper,
                )

        if ctx.needs_input_grad[1]:
            # grad_filter_out[e] = x[src[e]] * grad_output[dst[e]] * C[e]
            # Uses fused kernel to eliminate intermediate tensors
            # Output dtype matches filter_out dtype (FP32 or FP16)
            grad_filter_out = fused_grad_filter_out(
                x,
                grad_output,
                edge_weight,
                edge_src,
                edge_dst,
                cutoff_upper,
                out_dtype=filter_out_dtype,
            )

        return grad_x, grad_filter_out, None, None, None, None, None, None, None


def fused_cutoff_gather_multiply_scatter_autograd(
    x: torch.Tensor,
    filter_out: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    num_nodes: int,
    cutoff_upper: float,
    src_ptr: torch.Tensor = None,
    src_perm: torch.Tensor = None,
) -> torch.Tensor:
    """Autograd-compatible version.

    Parameters
    ----------
    src_ptr : torch.Tensor, optional
        Src-CSR row pointers [num_nodes + 1]. If provided along with src_perm,
        enables atomic-free grad_x computation in backward.
    src_perm : torch.Tensor, optional
        Src-CSR permutation [num_edges]. Required with src_ptr.
    """
    return FusedCutoffGatherMultiplyScatterFunction.apply(
        x,
        filter_out,
        edge_weight,
        edge_src,
        edge_dst,
        num_nodes,
        cutoff_upper,
        src_ptr,
        src_perm,
    )


# ============================================================================
# Fused Linear + Tanh Kernel for Filter Network MLP
# ============================================================================


@triton.jit
def _triton_tanh(x):
    """Compute tanh using exp: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)"""
    x_clamped = tl.minimum(tl.maximum(x, -10.0), 10.0)
    exp_2x = tl.exp(2.0 * x_clamped)
    return (exp_2x - 1.0) / (exp_2x + 1.0)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
            num_stages=2,
            num_warps=4,
        ),
    ],
    key=[
        "N",
        "K",
    ],  # NOT M! M varies per step, would trigger autotune ~1500ms each,
)
@triton.jit
def fused_linear_tanh_kernel(
    # Pointers
    x_ptr,  # Input [M, K]
    w_ptr,  # Weight [K, N]
    b_ptr,  # Bias [N] or None
    y_ptr,  # Output [M, N]
    # Dimensions
    M,
    N,
    K,
    # Strides
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_ym,
    stride_yn,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused kernel: Y = tanh(X @ W + b)

    This kernel computes matrix multiplication and applies tanh in a single pass,
    avoiding intermediate tensor allocation and extra memory bandwidth.

    Optimized for filter network dimensions:
    - M = num_edges (varies: 1k-100k+)
    - K = num_rbf (typically ~50)
    - N = num_filters (typically ~128)
    """
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # block start indices
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    # offset for this block
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k

        # load X block [BLOCK_M, BLOCK_K]
        x_ptrs = (
            x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        )
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x_block = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # load W block [BLOCK_K, BLOCK_N]
        w_ptrs = (
            w_ptr + k_offs[:, None] * stride_wk + offs_n[None, :] * stride_wn
        )
        w_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        w_block = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # accumulate matmul
        acc += tl.dot(x_block, w_block)

    # add bias if present
    if HAS_BIAS:
        bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]

    # apply tanh activation (fused!)
    acc = _triton_tanh(acc)

    # store output
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=y_mask)


def fused_linear_tanh(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute Y = tanh(X @ W + b) using fused Triton kernel.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor [M, K] where M = num_edges, K = num_rbf
    weight : torch.Tensor
        Weight matrix [K, N] where N = num_filters
    bias : torch.Tensor, optional
        Bias vector [N]

    Returns
    -------
    torch.Tensor
        Output [M, N]
    """
    assert x.is_cuda and x.is_contiguous()
    assert weight.is_cuda and weight.is_contiguous()

    M, K = x.shape
    K2, N = weight.shape
    assert (
        K == K2
    ), f"Dimension mismatch: x has {K} columns but weight has {K2} rows"

    # allocate output
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # grid
    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_M"]),
            triton.cdiv(N, META["BLOCK_N"]),
        )

    # launch kernel
    fused_linear_tanh_kernel[grid](
        x,
        weight,
        bias if bias is not None else x,
        y,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        y.stride(0),
        y.stride(1),
        HAS_BIAS=(bias is not None),
    )

    return y


class FusedLinearTanhFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        y = fused_linear_tanh(x, weight, bias)
        ctx.save_for_backward(x, weight, bias, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias, y = ctx.saved_tensors

        # tanh derivative: d(tanh(z))/dz = 1 - tanh(z)^2
        #    as y = tanh(z), grad_z = grad_output * (1 - y^2)
        grad_z = grad_output * (1.0 - y * y)

        grad_x = None
        grad_weight = None
        grad_bias = None

        if ctx.needs_input_grad[0]:
            # grad_x = grad_z @ weight.T
            grad_x = grad_z @ weight.t()

        if ctx.needs_input_grad[1]:
            # grad_weight = x.T @ grad_z
            grad_weight = x.t() @ grad_z

        if bias is not None and ctx.needs_input_grad[2]:
            # grad_bias = sum(grad_z, dim=0)
            grad_bias = grad_z.sum(dim=0)

        return grad_x, grad_weight, grad_bias


def fused_linear_tanh_autograd(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None
) -> torch.Tensor:
    return FusedLinearTanhFunction.apply(x, weight, bias)


# ============================================================================
# Mixed Precision Kernels for GPTQ W16A16 Quantization
# Input: FP32, Weights: FP16, Intermediate: FP16, Output: FP32
# ============================================================================


@triton.autotune(
    configs=[
        # For large M (simulation workload: M ~ 500k-1M), use large blocks
        # BLOCK_M=128 works well for tall matrices
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64},
            num_stages=2,
            num_warps=4,
        ),
    ],
    # CRITICAL: Only key on N and K, NOT M!
    # M (num_edges) varies every simulation step, keying on M causes
    # autotune to run for every unique M value (1500ms per call!)
    key=["N", "K"],
)
@triton.jit
def fused_linear_tanh_fp16_kernel(
    # Pointers
    x_ptr,  # Input [M, K] - FP32
    w_ptr,  # Weight [K, N] - FP16
    b_ptr,  # Bias [N] - FP16 or None
    y_ptr,  # Output [M, N] - FP16
    # Dimensions
    M,
    N,
    K,
    # Strides
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_ym,
    stride_yn,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Layer 0: Y_fp16 = tanh(X_fp32 @ W_fp16 + b_fp16)

    Input is FP32, weights are FP16, output is FP16.
    Matmul naturally handles FP32 x FP16 -> FP32 accumulation.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulate in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k

        # Load X block as FP32, convert to FP16 for matmul
        x_ptrs = (
            x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        )
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x_block = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float16)

        # Load W block as FP16
        w_ptrs = (
            w_ptr + k_offs[:, None] * stride_wk + offs_n[None, :] * stride_wn
        )
        w_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        w_block = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # FP16 x FP16 matmul with FP32 accumulation
        acc += tl.dot(x_block, w_block)

    # Add bias (FP16)
    if HAS_BIAS:
        bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :].to(tl.float32)

    # Apply tanh, store as FP16
    acc = _triton_tanh(acc)
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc.to(tl.float16), mask=y_mask)


def fused_linear_tanh_fp16(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None
) -> torch.Tensor:
    """
    Y_fp16 = tanh(X @ W_fp16 + b_fp16)

    Input: FP32 or FP16, Weight: FP16, Output: FP16
    The kernel handles both input types - FP32 is converted to FP16 for matmul,
    FP16 is used directly.
    """
    assert x.dtype in (
        torch.float32,
        torch.float16,
    ), f"Input must be FP32 or FP16, got {x.dtype}"
    assert (
        weight.dtype == torch.float16
    ), f"Weight must be FP16, got {weight.dtype}"
    assert x.is_cuda and x.is_contiguous()
    assert weight.is_cuda and weight.is_contiguous()

    M, K = x.shape
    K2, N = weight.shape
    assert K == K2

    y = torch.empty((M, N), device=x.device, dtype=torch.float16)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_M"]),
            triton.cdiv(N, META["BLOCK_N"]),
        )

    fused_linear_tanh_fp16_kernel[grid](
        x,
        weight,
        bias if bias is not None else x,
        y,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        y.stride(0),
        y.stride(1),
        HAS_BIAS=(bias is not None),
    )

    return y


# ============================================================================
# Layer 1: FP16 input x FP16 weight -> FP32 output
# ============================================================================


@triton.autotune(
    configs=[
        # Small matrix configs (M < 10k, K=128, N=128)
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64},
            num_stages=1,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64},
            num_stages=1,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64},
            num_stages=1,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 128},
            num_stages=1,
            num_warps=4,
        ),
        # Medium/Large matrix configs
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
            num_stages=2,
            num_warps=4,
        ),
    ],
    key=[
        "N",
        "K",
    ],  # NOT M! M varies per step, would trigger autotune ~1500ms each,
)
@triton.jit
def linear_fp16_kernel(
    # Pointers
    x_ptr,  # Input [M, K] - FP16
    w_ptr,  # Weight [K, N] - FP16
    y_ptr,  # Output [M, N] - FP32
    # Dimensions
    M,
    N,
    K,
    # Strides
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_ym,
    stride_yn,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Layer 1: Y_fp32 = X_fp16 @ W_fp16

    FP16 x FP16 matmul with FP32 output (no activation, no bias).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k

        # Load FP16 input
        x_ptrs = (
            x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        )
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x_block = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load FP16 weight
        w_ptrs = (
            w_ptr + k_offs[:, None] * stride_wk + offs_n[None, :] * stride_wn
        )
        w_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        w_block = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # FP16 x FP16 -> FP32 accumulation
        acc += tl.dot(x_block, w_block)

    # Store as FP32
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc, mask=y_mask)


def linear_fp16(
    x: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Layer 1: Y = X_fp16 @ W_fp16

    Input: FP16, Weight: FP16, Output: FP32 or FP16 (controlled by out_dtype)

    Parameters
    ----------
    x : torch.Tensor
        Input [M, K] in FP16
    weight : torch.Tensor
        Weight [K, N] in FP16
    out_dtype : torch.dtype
        Output dtype, either torch.float32 (default) or torch.float16

    Returns
    -------
    torch.Tensor
        Output [M, N] in out_dtype
    """
    assert x.dtype == torch.float16, f"Input must be FP16, got {x.dtype}"
    assert (
        weight.dtype == torch.float16
    ), f"Weight must be FP16, got {weight.dtype}"
    assert x.is_cuda and x.is_contiguous()
    assert weight.is_cuda and weight.is_contiguous()
    assert out_dtype in (
        torch.float32,
        torch.float16,
    ), f"out_dtype must be FP32 or FP16"

    M, K = x.shape
    K2, N = weight.shape
    assert K == K2

    # Output dtype controlled by parameter
    y = torch.empty((M, N), device=x.device, dtype=out_dtype)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_M"]),
            triton.cdiv(N, META["BLOCK_N"]),
        )

    if out_dtype == torch.float32:
        linear_fp16_kernel[grid](
            x,
            weight,
            y,
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            weight.stride(1),
            y.stride(0),
            y.stride(1),
        )
    else:
        linear_fp16_to_fp16_kernel[grid](
            x,
            weight,
            y,
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            weight.stride(1),
            y.stride(0),
            y.stride(1),
        )

    return y


# Kernel for FP16 output (accumulate in FP32, store as FP16)
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64},
            num_stages=1,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64},
            num_stages=1,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64},
            num_stages=1,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 128},
            num_stages=1,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
            num_stages=2,
            num_warps=4,
        ),
    ],
    key=["N", "K"],
)
@triton.jit
def linear_fp16_to_fp16_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Y_fp16 = X_fp16 @ W_fp16

    FP16 x FP16 matmul with FP32 accumulation, FP16 output.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulate in FP32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k

        x_ptrs = (
            x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        )
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x_block = tl.load(x_ptrs, mask=x_mask, other=0.0)

        w_ptrs = (
            w_ptr + k_offs[:, None] * stride_wk + offs_n[None, :] * stride_wn
        )
        w_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        w_block = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(x_block, w_block)

    # Store as FP16
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc.to(tl.float16), mask=y_mask)


# ============================================================================
# V2 Optimized Backward Kernels
# - Fused tanh backward into grad_x/grad_weight (no intermediate tensor)
# - Persistent reduction (no atomics) for grad_weight
# - Pre-transposed weights for better memory access
# ============================================================================


@triton.autotune(
    configs=[
        # Tall-skinny configs: large BLOCK_M, full K coverage
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 128},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 128},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 64},
            num_stages=3,
            num_warps=8,
        ),
    ],
    key=["N", "K"],
)
@triton.jit
def fused_tanh_backward_grad_x_v2_kernel(
    # Inputs
    grad_out_ptr,  # [M, K_out] FP16
    y_ptr,  # [M, K_out] FP16
    weight_t_ptr,  # [K_out, N] FP16 - PRE-TRANSPOSED, contiguous
    # Output
    grad_x_ptr,  # [M, N] FP32
    # Dimensions
    M,
    N,
    K_out,
    # Strides
    stride_gm,
    stride_gk,
    stride_ym,
    stride_yk,
    stride_wtk,
    stride_wtn,  # weight_t strides
    stride_xm,
    stride_xn,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    V2: grad_x = (grad_out * (1 - y^2)) @ weight_t

    Fuses tanh derivative into matmul - no intermediate grad_z tensor.
    weight_t is pre-transposed [K_out, N], so this is standard matmul.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K_out, BLOCK_K):
        k_offs = k_start + offs_k

        # Load grad_out and y
        g_ptrs = (
            grad_out_ptr
            + offs_m[:, None] * stride_gm
            + k_offs[None, :] * stride_gk
        )
        g_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K_out)
        grad_out_block = tl.load(g_ptrs, mask=g_mask, other=0.0)

        y_ptrs = (
            y_ptr + offs_m[:, None] * stride_ym + k_offs[None, :] * stride_yk
        )
        y_block = tl.load(y_ptrs, mask=g_mask, other=0.0)

        # Fused tanh derivative: grad_z = grad_out * (1 - y^2)
        y_fp32 = y_block.to(tl.float32)
        grad_out_fp32 = grad_out_block.to(tl.float32)
        grad_z_block = grad_out_fp32 * (1.0 - y_fp32 * y_fp32)
        grad_z_fp16 = grad_z_block.to(tl.float16)

        # Load weight_t[K_out, N] - already transposed, contiguous access!
        wt_ptrs = (
            weight_t_ptr
            + k_offs[:, None] * stride_wtk
            + offs_n[None, :] * stride_wtn
        )
        wt_mask = (k_offs[:, None] < K_out) & (offs_n[None, :] < N)
        wt_block = tl.load(
            wt_ptrs, mask=wt_mask, other=0.0
        )  # [BLOCK_K, BLOCK_N]

        # Standard matmul: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N]
        acc += tl.dot(grad_z_fp16, wt_block)

    x_ptrs = (
        grad_x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    )
    x_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(x_ptrs, acc, mask=x_mask)


def fused_tanh_backward_grad_x_v2(grad_out, y, weight_t):
    """
    V2: grad_x = (grad_out * (1 - y^2)) @ weight_t

    weight_t: [K_out, N] pre-transposed (= original weight.T)
    """
    M, K_out = grad_out.shape
    K_out2, N = weight_t.shape
    assert K_out == K_out2

    grad_x = torch.empty((M, N), device=grad_out.device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    fused_tanh_backward_grad_x_v2_kernel[grid](
        grad_out,
        y,
        weight_t,
        grad_x,
        M,
        N,
        K_out,
        grad_out.stride(0),
        grad_out.stride(1),
        y.stride(0),
        y.stride(1),
        weight_t.stride(0),
        weight_t.stride(1),
        grad_x.stride(0),
        grad_x.stride(1),
    )

    return grad_x


@triton.jit
def fused_tanh_backward_grad_weight_v2_kernel(
    # Inputs
    x_ptr,  # [M, K] FP32
    grad_out_ptr,  # [M, N] FP16
    y_ptr,  # [M, N] FP16
    # Output
    grad_weight_ptr,  # [K, N] FP32
    # Dimensions
    M,
    K,
    N,
    # Strides
    stride_xm,
    stride_xk,
    stride_gm,
    stride_gn,
    stride_ym,
    stride_yn,
    stride_wk,
    stride_wn,
    # Block sizes - tuned for K=64, N=128
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    V2: Persistent reduction for grad_weight = x.T @ (grad_out * (1 - y^2))

    - Fuses tanh derivative into the kernel (no grad_z tensor)
    - Each program loops over entire M dimension (persistent)
    - No atomics needed - single store at end
    """
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)

    for m_start in range(0, M, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        m_mask = offs_m < M

        # Load x, grad_out, y
        x_ptrs = (
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        )
        x_mask = m_mask[:, None] & (offs_k[None, :] < K)
        x_block = tl.load(x_ptrs, mask=x_mask, other=0.0)

        g_ptrs = (
            grad_out_ptr
            + offs_m[:, None] * stride_gm
            + offs_n[None, :] * stride_gn
        )
        g_mask = m_mask[:, None] & (offs_n[None, :] < N)
        grad_out_block = tl.load(g_ptrs, mask=g_mask, other=0.0)

        y_ptrs = (
            y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        )
        y_block = tl.load(y_ptrs, mask=g_mask, other=0.0)

        # Fused tanh derivative
        y_fp32 = y_block.to(tl.float32)
        grad_out_fp32 = grad_out_block.to(tl.float32)
        grad_z_block = grad_out_fp32 * (1.0 - y_fp32 * y_fp32)

        # x.T @ grad_z: [BLOCK_K, BLOCK_M] @ [BLOCK_M, BLOCK_N]
        x_t = tl.trans(x_block.to(tl.float16))
        grad_z_fp16 = grad_z_block.to(tl.float16)
        acc += tl.dot(x_t, grad_z_fp16)

    # Single store - no atomics!
    w_ptrs = (
        grad_weight_ptr
        + offs_k[:, None] * stride_wk
        + offs_n[None, :] * stride_wn
    )
    w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
    tl.store(w_ptrs, acc, mask=w_mask)


def fused_tanh_backward_grad_weight_v2(x, grad_out, y, K_out, N_out):
    """V2: Persistent reduction for grad_weight (no atomics)."""
    M, K = x.shape

    grad_weight = torch.empty((K, N_out), device=x.device, dtype=torch.float32)

    # BLOCK_M tuned to fit in shared memory (max ~100KB)
    # Each iteration loads: x[BLOCK_M, BLOCK_K] + grad_out[BLOCK_M, BLOCK_N] + y[BLOCK_M, BLOCK_N]
    # Total: BLOCK_M * (BLOCK_K + 2*BLOCK_N) * 2 bytes (FP16) + accumulator
    BLOCK_M = 64  # Reduced to fit shared memory
    BLOCK_K = 32 if K >= 32 else K
    BLOCK_N = 32 if N_out >= 32 else N_out

    grid = (triton.cdiv(K, BLOCK_K), triton.cdiv(N_out, BLOCK_N))

    fused_tanh_backward_grad_weight_v2_kernel[grid](
        x,
        grad_out,
        y,
        grad_weight,
        M,
        K,
        N_out,
        x.stride(0),
        x.stride(1),
        grad_out.stride(0),
        grad_out.stride(1),
        y.stride(0),
        y.stride(1),
        grad_weight.stride(0),
        grad_weight.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        BLOCK_N=BLOCK_N,
    )

    return grad_weight.half()


# V2: Layer 1 grad_x with pre-transposed weight
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128},
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},
            num_stages=3,
            num_warps=4,
        ),
    ],
    key=["N", "K"],
)
@triton.jit
def matmul_fp32_fp16_to_fp16_v2_kernel(
    # A: [M, K] FP32, B: [K, N] FP16 (pre-transposed), C: [M, N] FP16
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Standard matmul with pre-transposed B - no tl.trans needed."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k

        a_ptrs = (
            a_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        )
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        a_block = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float16)

        b_ptrs = (
            b_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
        )
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        b_block = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a_block, b_block)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def matmul_fp32_fp16_to_fp16_v2(a, b_t):
    """C = A @ B_t where B_t is pre-transposed [K, N]"""
    M, K = a.shape
    K2, N = b_t.shape
    assert K == K2

    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    matmul_fp32_fp16_to_fp16_v2_kernel[grid](
        a,
        b_t,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b_t.stride(0),
        b_t.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c


# V2: Layer 1 persistent grad_weight
@triton.jit
def grad_weight_persistent_v2_kernel(
    x_ptr,
    grad_out_ptr,
    grad_weight_ptr,
    M,
    K,
    N,
    stride_xm,
    stride_xk,
    stride_gm,
    stride_gn,
    stride_wk,
    stride_wn,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """V2: Persistent reduction for Layer 1 grad_weight (no atomics)."""
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)

    for m_start in range(0, M, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        m_mask = offs_m < M

        x_ptrs = (
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        )
        x_mask = m_mask[:, None] & (offs_k[None, :] < K)
        x_block = tl.load(x_ptrs, mask=x_mask, other=0.0)

        g_ptrs = (
            grad_out_ptr
            + offs_m[:, None] * stride_gm
            + offs_n[None, :] * stride_gn
        )
        g_mask = m_mask[:, None] & (offs_n[None, :] < N)
        g_block = tl.load(g_ptrs, mask=g_mask, other=0.0)

        x_t = tl.trans(x_block)
        g_fp16 = g_block.to(tl.float16)
        acc += tl.dot(x_t, g_fp16)

    w_ptrs = (
        grad_weight_ptr
        + offs_k[:, None] * stride_wk
        + offs_n[None, :] * stride_wn
    )
    w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
    tl.store(w_ptrs, acc, mask=w_mask)


def grad_weight_persistent_v2(x, grad_out):
    """V2: Persistent reduction for Layer 1 grad_weight (no atomics)."""
    M, K = x.shape
    M2, N = grad_out.shape
    assert M == M2

    grad_weight = torch.empty((K, N), device=x.device, dtype=torch.float32)

    # BLOCK_M tuned to fit in shared memory
    BLOCK_M = 64  # Reduced to fit shared memory
    BLOCK_K = 64 if K >= 64 else K
    BLOCK_N = 32 if N >= 32 else N

    grid = (triton.cdiv(K, BLOCK_K), triton.cdiv(N, BLOCK_N))

    grad_weight_persistent_v2_kernel[grid](
        x,
        grad_out,
        grad_weight,
        M,
        K,
        N,
        x.stride(0),
        x.stride(1),
        grad_out.stride(0),
        grad_out.stride(1),
        grad_weight.stride(0),
        grad_weight.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        BLOCK_N=BLOCK_N,
    )

    return grad_weight.half()


# ============================================================================
# Autograd Functions with Triton Backward Kernels
# ============================================================================


class FusedLinearTanhFP16Function(torch.autograd.Function):
    """FP32/FP16 input -> FP16 output with tanh, V2 optimized Triton backward.

    Supports both FP32 and FP16 input:
    - FP32 input: converted to FP16 for matmul, grad_x returned as FP32
    - FP16 input: used directly, grad_x returned as FP16

    V2 optimizations:
    - Fused tanh backward into grad_x (no intermediate grad_z tensor)
    - Persistent reduction for grad_weight (no atomics)
    - Pre-transposed weights for better memory access
    """

    @staticmethod
    def forward(ctx, x, weight, bias):
        # x: FP32 or FP16, weight/bias: FP16, output: FP16
        y_fp16 = fused_linear_tanh_fp16(x, weight, bias)
        # Pre-transpose weight for backward: weight is [K, N], weight_t is [N, K]
        weight_t = weight.t().contiguous()
        ctx.save_for_backward(x, weight_t, y_fp16)
        ctx.has_bias = bias is not None
        ctx.input_dtype = x.dtype
        return y_fp16

    @staticmethod
    def backward(ctx, grad_output):
        x, weight_t, y_fp16 = ctx.saved_tensors
        # weight_t is [N, K] = original weight.T

        grad_output = grad_output.contiguous()

        grad_x = None
        grad_weight = None
        grad_bias = None

        if ctx.needs_input_grad[0]:
            # V2: Fused tanh backward + grad_x matmul (no intermediate tensor)
            # grad_x = (grad_out * (1-y^2)) @ weight_t
            grad_x = fused_tanh_backward_grad_x_v2(
                grad_output, y_fp16, weight_t
            )
            # Match input dtype (kernel returns FP32)
            if ctx.input_dtype == torch.float16:
                grad_x = grad_x.half()

        if ctx.needs_input_grad[1]:
            # V2: Fused tanh backward + persistent reduction (no atomics)
            # grad_weight = x.T @ (grad_out * (1-y^2))
            _, N = grad_output.shape
            grad_weight = fused_tanh_backward_grad_weight_v2(
                x, grad_output, y_fp16, x.shape[1], N
            )

        if ctx.has_bias and ctx.needs_input_grad[2]:
            # Compute grad_z for bias (still need elementwise for bias)
            grad_z = grad_output.float() * (1.0 - y_fp16.float() ** 2)
            grad_bias = grad_z.sum(dim=0).half()

        return grad_x, grad_weight, grad_bias


class LinearFP16Function(torch.autograd.Function):
    """Layer 1: FP16 input -> FP32 output, with V2 optimized Triton backward.

    V2 optimizations:
    - Pre-transposed weights for better memory access
    - Persistent reduction for grad_weight (no atomics)
    """

    @staticmethod
    def forward(ctx, x, weight):
        # x: FP16, weight: FP16, output: FP32
        y = linear_fp16(x, weight)
        # Pre-transpose weight for backward: weight is [K, N], weight_t is [N, K]
        weight_t = weight.t().contiguous()
        ctx.save_for_backward(x, weight_t)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, weight_t = ctx.saved_tensors
        # weight_t is [N, K] = original weight.T

        grad_x = None
        grad_weight = None

        if ctx.needs_input_grad[0]:
            # V2: grad_x = grad_output @ weight_t (pre-transposed, no tl.trans)
            # grad_output: [M, N], weight_t: [N, K], grad_x: [M, K]
            grad_x = matmul_fp32_fp16_to_fp16_v2(grad_output, weight_t)

        if ctx.needs_input_grad[1]:
            # V2: Persistent reduction (no atomics)
            # grad_weight = x.T @ grad_output
            grad_weight = grad_weight_persistent_v2(x, grad_output)

        return grad_x, grad_weight


class LinearFP16ToFP16Function(torch.autograd.Function):
    """FP16 input -> FP16 output (FP32 accumulation internally).

    This is used when we want to keep filter_out in FP16 to save bandwidth.
    """

    @staticmethod
    def forward(ctx, x, weight):
        # x: FP16, weight: FP16, output: FP16
        y = linear_fp16(x, weight, out_dtype=torch.float16)
        # Pre-transpose weight for backward
        weight_t = weight.t().contiguous()
        ctx.save_for_backward(x, weight_t)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, weight_t = ctx.saved_tensors

        grad_x = None
        grad_weight = None

        if ctx.needs_input_grad[0]:
            # grad_output is FP16, convert to FP32 for backward
            grad_out_fp32 = grad_output.float()
            grad_x = matmul_fp32_fp16_to_fp16_v2(grad_out_fp32, weight_t)

        if ctx.needs_input_grad[1]:
            # grad_output is FP16, convert for backward
            grad_out_fp32 = grad_output.float()
            grad_weight = grad_weight_persistent_v2(x, grad_out_fp32)

        return grad_x, grad_weight


def fused_linear_tanh_fp16_autograd(x, weight, bias=None):
    """FP32/FP16 -> FP16 with tanh (Triton backward)"""
    return FusedLinearTanhFP16Function.apply(x, weight, bias)


def linear_fp16_autograd(x, weight, out_dtype=torch.float32):
    """Layer 1: FP16 -> FP32 or FP16 (with Triton backward)

    Parameters
    ----------
    x : torch.Tensor
        Input [M, K] in FP16
    weight : torch.Tensor
        Weight [K, N] in FP16
    out_dtype : torch.dtype
        Output dtype, either torch.float32 (default) or torch.float16

    Returns
    -------
    torch.Tensor
        Output [M, N] in specified dtype
    """
    if out_dtype == torch.float32:
        return LinearFP16Function.apply(x, weight)
    else:
        return LinearFP16ToFP16Function.apply(x, weight)


# ============================================================================
# Fused Distance + Gaussian RBF + Cosine Cutoff Kernel
# ============================================================================


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_EDGES": 32, "BLOCK_RBF": 64}, num_warps=4),
        triton.Config({"BLOCK_EDGES": 64, "BLOCK_RBF": 64}, num_warps=4),
        triton.Config({"BLOCK_EDGES": 128, "BLOCK_RBF": 64}, num_warps=4),
        triton.Config({"BLOCK_EDGES": 64, "BLOCK_RBF": 32}, num_warps=4),
        triton.Config({"BLOCK_EDGES": 128, "BLOCK_RBF": 32}, num_warps=4),
    ],
    key=["num_rbf"],
)
@triton.jit
def fused_distance_gaussian_rbf_cutoff_kernel(
    # Input pointers
    pos_ptr,  # [num_nodes, 3]
    edge_src_ptr,  # [num_edges]
    edge_dst_ptr,  # [num_edges]
    centers_ptr,  # [num_rbf]
    gamma_ptr,  # scalar stored in a tensor [1]
    # Output pointers
    dist_output_ptr,  # [num_edges]
    rbf_output_ptr,  # [num_edges, num_rbf]
    # Cutoff parameters
    cutoff_upper,
    # Sizes
    num_edges,
    num_rbf,
    # Block sizes
    BLOCK_EDGES: tl.constexpr,
    BLOCK_RBF: tl.constexpr,
):
    """
    Fused kernel that combines:
    1. Distance computation: d = ||pos[dst] - pos[src]||
    2. Gaussian RBF expansion: exp(gamma * (d - centers)^2)
    3. Cosine cutoff: 0.5 * (cos(d * pi / cutoff_upper) + 1) * (d < cutoff_upper)

    Output: distances and RBF expansion with cutoff applied.

    This eliminates intermediate memory traffic by computing distance once
    and reusing it for all RBF centers and cutoff calculation.
    """
    pid_edge = tl.program_id(axis=0)
    pid_rbf = tl.program_id(axis=1)

    # Compute edge and RBF offsets for this block
    edge_start = pid_edge * BLOCK_EDGES
    rbf_start = pid_rbf * BLOCK_RBF

    edge_offsets = edge_start + tl.arange(0, BLOCK_EDGES)
    rbf_offsets = rbf_start + tl.arange(0, BLOCK_RBF)

    edge_mask = edge_offsets < num_edges
    rbf_mask = rbf_offsets < num_rbf

    # Load gamma scalar (once per block)
    gamma = tl.load(gamma_ptr)

    # Load RBF centers for this block (shared across all edges)
    centers = tl.load(centers_ptr + rbf_offsets, mask=rbf_mask, other=0.0)

    # Load edge indices
    src_nodes = tl.load(edge_src_ptr + edge_offsets, mask=edge_mask, other=0)
    dst_nodes = tl.load(edge_dst_ptr + edge_offsets, mask=edge_mask, other=0)

    # Load source positions (strided by 3)
    pos_src_x = tl.load(pos_ptr + src_nodes * 3 + 0, mask=edge_mask, other=0.0)
    pos_src_y = tl.load(pos_ptr + src_nodes * 3 + 1, mask=edge_mask, other=0.0)
    pos_src_z = tl.load(pos_ptr + src_nodes * 3 + 2, mask=edge_mask, other=0.0)

    # Load destination positions
    pos_dst_x = tl.load(pos_ptr + dst_nodes * 3 + 0, mask=edge_mask, other=0.0)
    pos_dst_y = tl.load(pos_ptr + dst_nodes * 3 + 1, mask=edge_mask, other=0.0)
    pos_dst_z = tl.load(pos_ptr + dst_nodes * 3 + 2, mask=edge_mask, other=0.0)

    # Compute distances [BLOCK_EDGES]
    dx = pos_dst_x - pos_src_x
    dy = pos_dst_y - pos_src_y
    dz = pos_dst_z - pos_src_z
    dist = tl.sqrt(dx * dx + dy * dy + dz * dz)

    # Store distances (only on first RBF block to avoid duplicate writes)
    if pid_rbf == 0:
        tl.store(dist_output_ptr + edge_offsets, dist, mask=edge_mask)

    # Compute cosine cutoff: 0.5 * (cos(d * pi / cutoff) + 1) * (d < cutoff)
    pi = 3.141592653589793
    cos_val = tl.cos(dist * pi / cutoff_upper)
    cutoff_val = 0.5 * (cos_val + 1.0)
    dist_in_range = dist < cutoff_upper
    cutoff_val = tl.where(dist_in_range, cutoff_val, 0.0)

    # Broadcast for 2D computation:
    # dist: [BLOCK_EDGES, 1]
    # centers: [1, BLOCK_RBF]
    dist_2d = dist[:, None]
    centers_2d = centers[None, :]
    cutoff_2d = cutoff_val[:, None]

    # Compute Gaussian RBF: exp(gamma * (dist - center)^2) * cutoff
    diff = dist_2d - centers_2d
    rbf_values = tl.exp(gamma * diff * diff) * cutoff_2d

    # Store output with 2D mask
    output_mask = edge_mask[:, None] & rbf_mask[None, :]
    output_offsets = edge_offsets[:, None] * num_rbf + rbf_offsets[None, :]
    tl.store(rbf_output_ptr + output_offsets, rbf_values, mask=output_mask)


def fused_distance_gaussian_rbf_cutoff(
    pos: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    centers: torch.Tensor,
    gamma: float,
    cutoff_upper: float,
) -> tuple:
    """
    Compute fused distance + Gaussian RBF expansion + cosine cutoff.

    This kernel fuses three operations that are typically separate:
    1. Distance computation: d = ||pos[dst] - pos[src]||
    2. Gaussian RBF: exp(gamma * (d - centers)^2)
    3. Cosine cutoff: 0.5 * (cos(d * pi / cutoff) + 1)

    Parameters
    ----------
    pos : torch.Tensor
        Node positions [num_nodes, 3]
    edge_src : torch.Tensor
        Source node indices [num_edges]
    edge_dst : torch.Tensor
        Destination node indices [num_edges]
    centers : torch.Tensor
        RBF centers [num_rbf]
    gamma : float
        Gaussian width parameter (typically negative: -0.5 / spacing^2)
    cutoff_upper : float
        Upper cutoff distance for cosine envelope

    Returns
    -------
    distances : torch.Tensor
        Pairwise distances [num_edges]
    rbf_expansion : torch.Tensor
        RBF expansion with cutoff applied [num_edges, num_rbf]
    """
    assert pos.is_cuda and pos.is_contiguous()
    assert edge_src.is_contiguous()
    assert edge_dst.is_contiguous()
    assert centers.is_contiguous()

    num_edges = edge_src.shape[0]
    num_rbf = centers.shape[0]

    # Allocate outputs
    distances = torch.empty(num_edges, device=pos.device, dtype=pos.dtype)
    rbf_output = torch.empty(
        (num_edges, num_rbf), device=pos.device, dtype=pos.dtype
    )

    if num_edges == 0:
        return distances, rbf_output

    # Convert gamma to tensor for Triton
    gamma_tensor = torch.tensor([gamma], device=pos.device, dtype=pos.dtype)

    # 2D grid for the optimized kernel
    def grid(META):
        return (
            triton.cdiv(num_edges, META["BLOCK_EDGES"]),
            triton.cdiv(num_rbf, META["BLOCK_RBF"]),
        )

    fused_distance_gaussian_rbf_cutoff_kernel[grid](
        pos,
        edge_src,
        edge_dst,
        centers,
        gamma_tensor,
        distances,
        rbf_output,
        cutoff_upper,
        num_edges,
        num_rbf,
    )

    return distances, rbf_output


class FusedDistanceGaussianRBFCutoffFunction(torch.autograd.Function):
    """Autograd function for fused distance + Gaussian RBF + cutoff.

    This function computes:
    - distances = ||pos[dst] - pos[src]||
    - rbf = exp(gamma * (distances - centers)^2) * cutoff(distances)

    Backward pass computes gradients with respect to positions.
    """

    @staticmethod
    def forward(ctx, pos, edge_src, edge_dst, centers, gamma, cutoff_upper):
        distances, rbf_expansion = fused_distance_gaussian_rbf_cutoff(
            pos, edge_src, edge_dst, centers, gamma, cutoff_upper
        )
        ctx.save_for_backward(pos, edge_src, edge_dst, centers, distances)
        ctx.gamma = gamma
        ctx.cutoff_upper = cutoff_upper
        return distances, rbf_expansion

    @staticmethod
    def backward(ctx, grad_distances, grad_rbf):
        pos, edge_src, edge_dst, centers, distances = ctx.saved_tensors
        gamma = ctx.gamma
        cutoff_upper = ctx.cutoff_upper

        grad_pos = None

        if ctx.needs_input_grad[0]:
            # Compute position gradients
            # d(distance)/d(pos_dst) = (pos_dst - pos_src) / distance
            # d(distance)/d(pos_src) = -(pos_dst - pos_src) / distance

            pi = 3.141592653589793
            num_edges = edge_src.shape[0]

            # Direction vectors (normalized)
            dr = pos[edge_dst] - pos[edge_src]  # [num_edges, 3]
            dist_safe = distances.clamp(min=1e-8)
            direction = dr / dist_safe.unsqueeze(-1)  # [num_edges, 3]

            # Compute cutoff and its derivative
            dist_in_range = (distances < cutoff_upper).float()
            cos_val = torch.cos(distances * pi / cutoff_upper)
            cutoff_val = 0.5 * (cos_val + 1.0) * dist_in_range

            # d(cutoff)/d(dist) = -0.5 * pi/cutoff_upper * sin(dist * pi / cutoff_upper)
            sin_val = torch.sin(distances * pi / cutoff_upper)
            d_cutoff_d_dist = (
                -0.5 * (pi / cutoff_upper) * sin_val * dist_in_range
            )

            # Compute RBF and its derivative w.r.t. distance
            # rbf = exp(gamma * (dist - center)^2) * cutoff
            # d(rbf)/d(dist) = [2*gamma*(dist-center)*exp(...)*cutoff + exp(...)*d_cutoff]
            diff = distances.unsqueeze(-1) - centers.unsqueeze(
                0
            )  # [num_edges, num_rbf]
            exp_term = torch.exp(gamma * diff**2)

            # Chain rule for rbf gradient
            d_rbf_d_dist = 2 * gamma * diff * exp_term * cutoff_val.unsqueeze(
                -1
            ) + exp_term * d_cutoff_d_dist.unsqueeze(-1)

            # Aggregate gradient from all RBF channels
            grad_dist_from_rbf = (grad_rbf * d_rbf_d_dist).sum(
                dim=-1
            )  # [num_edges]

            # Total gradient w.r.t. distance
            total_grad_dist = grad_distances + grad_dist_from_rbf

            # Convert to position gradients
            grad_dr = (
                total_grad_dist.unsqueeze(-1) * direction
            )  # [num_edges, 3]

            # Scatter gradients to positions
            grad_pos = torch.zeros_like(pos)
            grad_pos.index_add_(0, edge_dst, grad_dr)
            grad_pos.index_add_(0, edge_src, -grad_dr)

        return grad_pos, None, None, None, None, None


def fused_distance_gaussian_rbf_cutoff_autograd(
    pos: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    centers: torch.Tensor,
    gamma: float,
    cutoff_upper: float,
) -> tuple:
    """Autograd-compatible version of fused distance + Gaussian RBF + cutoff."""
    return FusedDistanceGaussianRBFCutoffFunction.apply(
        pos, edge_src, edge_dst, centers, gamma, cutoff_upper
    )


# ============================================================================
# Fused Tanh + Linear Kernel for InteractionBlock
# Computes: Y = (tanh(X) @ W) + b  (tanh applied FIRST, then matmul)
# ============================================================================


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
            num_stages=2,
            num_warps=4,
        ),
    ],
    key=[
        "N",
        "K",
    ],  # NOT M! M varies per step, would trigger autotune ~1500ms each
)
@triton.jit
def fused_tanh_linear_kernel(
    # Pointers
    x_ptr,  # Input [M, K]
    w_ptr,  # Weight [K, N]
    b_ptr,  # Bias [N] or None
    y_ptr,  # Output [M, N]
    # Dimensions
    M,
    N,
    K,
    # Strides
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_ym,
    stride_yn,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused kernel: Y = tanh(X) @ W + b

    This kernel applies tanh to the input FIRST, then computes matrix
    multiplication, avoiding intermediate tensor allocation.

    Used in InteractionBlock where the order is: tanh(x) followed by linear(x).

    Optimized for InteractionBlock dimensions:
    - M = num_nodes (varies: 1k-100k+)
    - K = hidden_channels (typically ~128)
    - N = hidden_channels (typically ~128)
    """
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # block start indices
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    # offset for this block
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k

        # load X block [BLOCK_M, BLOCK_K]
        x_ptrs = (
            x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        )
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x_block = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # apply tanh to input (fused!)
        x_block = _triton_tanh(x_block)

        # load W block [BLOCK_K, BLOCK_N]
        w_ptrs = (
            w_ptr + k_offs[:, None] * stride_wk + offs_n[None, :] * stride_wn
        )
        w_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        w_block = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # accumulate matmul
        acc += tl.dot(x_block, w_block)

    # add bias if present
    if HAS_BIAS:
        bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]

    # store output
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=y_mask)


def fused_tanh_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute Y = tanh(X) @ W + b using fused Triton kernel.

    This applies tanh to input FIRST, then matrix multiplication.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor [M, K] where M = num_nodes, K = hidden_channels
    weight : torch.Tensor
        Weight matrix [K, N] where N = hidden_channels
    bias : torch.Tensor, optional
        Bias vector [N]

    Returns
    -------
    torch.Tensor
        Output [M, N]
    """
    assert x.is_cuda and x.is_contiguous()
    assert weight.is_cuda and weight.is_contiguous()

    M, K = x.shape
    K2, N = weight.shape
    assert (
        K == K2
    ), f"Dimension mismatch: x has {K} columns but weight has {K2} rows"

    # allocate output
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # grid
    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_M"]),
            triton.cdiv(N, META["BLOCK_N"]),
        )

    # launch kernel
    fused_tanh_linear_kernel[grid](
        x,
        weight,
        bias if bias is not None else x,
        y,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        y.stride(0),
        y.stride(1),
        HAS_BIAS=(bias is not None),
    )

    return y


class FusedTanhLinearFunction(torch.autograd.Function):
    """Autograd function for fused tanh + linear.

    Forward: y = tanh(x) @ W + b
    Backward:
        - grad_x = (1 - tanh(x)^2) * (grad_y @ W.T)
        - grad_W = tanh(x).T @ grad_y
        - grad_b = grad_y.sum(dim=0)
    """

    @staticmethod
    def forward(ctx, x, weight, bias):
        y = fused_tanh_linear(x, weight, bias)
        # Save tanh(x) for backward - we need it for both grad_x and grad_W
        tanh_x = torch.tanh(x)
        ctx.save_for_backward(tanh_x, weight, bias)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        tanh_x, weight, bias = ctx.saved_tensors

        grad_x = None
        grad_weight = None
        grad_bias = None

        if ctx.needs_input_grad[0]:
            # grad_x = (1 - tanh(x)^2) * (grad_y @ W.T)
            # tanh derivative: d(tanh(x))/dx = 1 - tanh(x)^2
            grad_linear = grad_output @ weight.t()
            grad_x = (1.0 - tanh_x * tanh_x) * grad_linear

        if ctx.needs_input_grad[1]:
            # grad_W = tanh(x).T @ grad_y
            grad_weight = tanh_x.t() @ grad_output

        if bias is not None and ctx.needs_input_grad[2]:
            # grad_b = sum(grad_y, dim=0)
            grad_bias = grad_output.sum(dim=0)

        return grad_x, grad_weight, grad_bias


def fused_tanh_linear_autograd(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None
) -> torch.Tensor:
    """Autograd-compatible fused tanh + linear.

    Computes: y = tanh(x) @ W + b

    Parameters
    ----------
    x : torch.Tensor
        Input tensor [M, K]
    weight : torch.Tensor
        Weight matrix [K, N]
    bias : torch.Tensor, optional
        Bias vector [N]

    Returns
    -------
    torch.Tensor
        Output [M, N]
    """
    return FusedTanhLinearFunction.apply(x, weight, bias)
