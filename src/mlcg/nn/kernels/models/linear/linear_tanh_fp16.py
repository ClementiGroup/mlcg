"""
Fused Triton kernels for CFConv operations.
"""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from .tanh_linear import _triton_tanh

# ============================================================================
# Mixed Precision Kernels for GPTQ W16A16 Quantization
# Input: FP32, Weights: FP16, Intermediate: FP16, Output: FP32
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
def fused_tanh_backward_grad_x_kernel(
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


@triton_op("mlcg_kernels::fused_tanh_backward_grad_x", mutates_args={})
def fused_tanh_backward_grad_x(
    grad_out: torch.Tensor, y: torch.Tensor, weight_t: torch.Tensor
) -> torch.Tensor:
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

    wrap_triton(fused_tanh_backward_grad_x_kernel)[grid](
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
def fused_tanh_backward_grad_weight_kernel(
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


@triton_op("mlcg_kernels::fused_tanh_backward_grad_weight", mutates_args={})
def fused_tanh_backward_grad_weight(
    x: torch.Tensor,
    grad_out: torch.Tensor,
    y: torch.Tensor,
    K_out: int,
    N_out: int,
) -> torch.Tensor:
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

    wrap_triton(fused_tanh_backward_grad_weight_kernel)[grid](
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


@triton_op("mlcg_kernels::fused_tanh_linear_fp16", mutates_args={})
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

    wrap_triton(fused_linear_tanh_fp16_kernel)[grid](
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


def setup_context(ctx, inputs, output):
    x, weight, bias = inputs
    y_fp16 = fused_linear_tanh_fp16(x, weight, bias)
    # Pre-transpose weight for backward: weight is [K, N], weight_t is [N, K]
    weight_t = weight.t().contiguous()
    ctx.save_for_backward(x, weight_t, y_fp16)
    ctx.has_bias = bias is not None
    ctx.input_dtype = x.dtype


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
        grad_x = fused_tanh_backward_grad_x(grad_output, y_fp16, weight_t)
        # Match input dtype (kernel returns FP32)
        if ctx.input_dtype == torch.float16:
            grad_x = grad_x.half()

    if ctx.needs_input_grad[1]:
        # V2: Fused tanh backward + persistent reduction (no atomics)
        # grad_weight = x.T @ (grad_out * (1-y^2))
        _, N = grad_output.shape
        grad_weight = fused_tanh_backward_grad_weight(
            x, grad_output, y_fp16, x.shape[1], N
        )

    if ctx.has_bias and ctx.needs_input_grad[2]:
        # Compute grad_z for bias (still need elementwise for bias)
        grad_z = grad_output.float() * (1.0 - y_fp16.float() ** 2)
        grad_bias = grad_z.sum(dim=0).half()

    return grad_x, grad_weight, grad_bias


fused_linear_tanh_fp16.register_autograd(backward, setup_context=setup_context)


@fused_linear_tanh_fp16.register_kernel("cpu")
def cpu_fused_linear_tanh_fp16(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None
) -> torch.Tensor:
    linear_layer = x.half() @ weight
    if bias is not None:
        linear_layer += bias
    out = torch.tanh(linear_layer)
    return out
