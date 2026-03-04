"""
Fused Tanh + Linear Kernel
Computes: Y = (tanh(X) @ W) + b  (tanh applied FIRST, then matmul)
"""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


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


@triton_op("mlcg_kernels::fused_tanh_linear", mutates_args={})
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
    if not x.is_contiguous():
        x = x.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()

    M, K = x.shape
    K2, N = weight.shape
    assert (  # FIXME: change this assert to something compiler friendly
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


def setup_context(ctx, inputs, output):
    x, weight, bias = inputs
    tanh_x = torch.tanh(x)
    ctx.save_for_backward(tanh_x, weight, bias)


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


fused_tanh_linear.register_autograd(backward, setup_context=setup_context)


@fused_tanh_linear.register_kernel("cpu")
def cpu_fused_tanh_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None
) -> torch.Tensor:
    out = torch.tanh(x) @ weight  # FIXME: check if matrix has to be transpose
    if bias is not None:
        out = out + bias
    return out
