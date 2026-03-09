"""
Fused Triton kernels for CFConv operations.
"""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from .tanh_linear import _triton_tanh


@triton.jit
def matmul_fp32_fp16_to_fp16_kernel(
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


@triton_op("mlcg_kernels::matmul_fp32_fp16_to_fp16", mutates_args={})
def matmul_fp32_fp16_to_fp16(
    a: torch.Tensor, b_t: torch.Tensor
) -> torch.Tensor:
    """C = A @ B_t where B_t is pre-transposed [K, N]"""
    M, K = a.shape
    K2, N = b_t.shape
    assert K == K2

    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    wrap_triton(matmul_fp32_fp16_to_fp16_kernel)[grid](
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
def grad_weight_persistent_kernel(
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


@triton_op("mlcg_kernels::grad_weight_persistent", mutates_args={})
def grad_weight_persistent(
    x: torch.Tensor, grad_out: torch.Tensor
) -> torch.Tensor:
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

    wrap_triton(grad_weight_persistent_kernel)[grid](
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


@triton_op("mlcg_kernels::linear_fp16", mutates_args={})
def linear_fp16(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """
    Layer 1: Y = X_fp16 @ W_fp16

    Input: FP16, Weight: FP16, Output: FP32

    Parameters
    ----------
    x : torch.Tensor
        Input [M, K] in FP16
    weight : torch.Tensor
        Weight [K, N] in FP16

    Returns
    -------
    torch.Tensor
        Output [M, N] in torch.float32
    """
    assert x.dtype == torch.float16, f"Input must be FP16, got {x.dtype}"
    assert (
        weight.dtype == torch.float16
    ), f"Weight must be FP16, got {weight.dtype}"
    assert x.is_cuda and x.is_contiguous()
    assert weight.is_cuda and weight.is_contiguous()

    M, K = x.shape
    K2, N = weight.shape
    assert K == K2

    # Output dtype controlled by parameter
    y = torch.empty((M, N), device=x.device, dtype=torch.float32)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_M"]),
            triton.cdiv(N, META["BLOCK_N"]),
        )

    wrap_triton(linear_fp16_kernel)[grid](
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


def setup_context_linear_fp16(ctx, inputs, output):
    # x: FP16, weight: FP16, output: FP32
    x, weight = inputs
    # y = linear_fp16(x, weight)
    # Pre-transpose weight for backward: weight is [K, N], weight_t is [N, K]
    weight_t = weight.t().contiguous()
    ctx.save_for_backward(x, weight_t)


def backward_linear_fp16(ctx, grad_output):
    x, weight_t = ctx.saved_tensors
    # weight_t is [N, K] = original weight.T

    grad_x = None
    grad_weight = None

    if ctx.needs_input_grad[0]:
        # V2: grad_x = grad_output @ weight_t (pre-transposed, no tl.trans)
        # grad_output: [M, N], weight_t: [N, K], grad_x: [M, K]
        grad_x = matmul_fp32_fp16_to_fp16(grad_output, weight_t)

    if ctx.needs_input_grad[1]:
        # V2: Persistent reduction (no atomics)
        # grad_weight = x.T @ grad_output
        grad_weight = grad_weight_persistent(x, grad_output)

    return grad_x, grad_weight


linear_fp16.register_autograd(
    backward_linear_fp16, setup_context=setup_context_linear_fp16
)

@linear_fp16.register_kernel("cpu")
def cpu_linear_to_fp16(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:

    out = x @ weight  # FIXME: check if matrix has to be transpose
    return out

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


@triton_op("mlcg_kernels::linear_fp16_to_fp16", mutates_args={})
def linear_fp16_to_fp16(
    x: torch.Tensor,
    weight: torch.Tensor,
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

    M, K = x.shape
    K2, N = weight.shape
    assert K == K2

    # Output dtype controlled by parameter
    y = torch.empty((M, N), device=x.device, dtype=torch.float16)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_M"]),
            triton.cdiv(N, META["BLOCK_N"]),
        )

    wrap_triton(linear_fp16_to_fp16_kernel)[grid](
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


def setup_context_linear_fp16_to_linear_fp16(ctx, inputs, output):
    x, weight = inputs
    # x: FP16, weight: FP16, output: FP16
    # Pre-transpose weight for backward
    weight_t = weight.t().contiguous()
    ctx.save_for_backward(x, weight_t)


def backward_linear_fp16_to_linear_fp16(ctx, grad_output):
    x, weight_t = ctx.saved_tensors
    grad_x = None
    grad_weight = None

    if ctx.needs_input_grad[0]:
        # grad_output is FP16, convert to FP32 for backward
        grad_out_fp32 = grad_output.float()
        grad_x = matmul_fp32_fp16_to_fp16(grad_out_fp32, weight_t)

    if ctx.needs_input_grad[1]:
        # grad_output is FP16, convert for backward
        grad_out_fp32 = grad_output.float()
        grad_weight = grad_weight_persistent(x, grad_out_fp32)

    return grad_x, grad_weight


linear_fp16_to_fp16.register_autograd(
    backward_linear_fp16_to_linear_fp16,
    setup_context=setup_context_linear_fp16_to_linear_fp16,
)

@linear_fp16_to_fp16.register_kernel("cpu")
def cpu_linear_fp16_to_fp16(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:

    out = x @ weight  # FIXME: check if matrix has to be transpose
    return out


def wrapper_linear_fp16(x, weight, out_dtype=torch.float32):
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
        return linear_fp16(x, weight)
    else:
        return linear_fp16_to_fp16(x, weight)
