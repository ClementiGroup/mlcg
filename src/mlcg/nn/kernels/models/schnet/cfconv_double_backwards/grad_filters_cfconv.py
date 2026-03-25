import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from ....utils import ensure_contiguous

from ...cutoffs import (
    _cosine_cutoff,
    _d_cosine_cutoff_dd,
    _d2_cosine_cutoff_dd2,
    _torch_cosine_cutoff,
    _torch_d_cosine_cutoff_dd,
    _torch_d2_cosine_cutoff_dd2,
)

triton_pi = tl.constexpr(3.141592653589793)


# ============================================================================
# grad_x_grad_filters_fused_cfconv
# ============================================================================


@triton.jit
def grad_x_grad_filters_fused_cfconv_kernel(
    # Input pointers
    grad_output_ptr,  # [num_nodes, feature_dim]
    edge_weight_ptr,  # [num_edges]
    src_perm_ptr,  # [num_edges] - src-CSR permutation
    src_ptr_ptr,  # [num_nodes + 1] - src-CSR row pointers
    edge_dst_ptr,  # [num_edges]
    grad_grad_filters_ptr,  # [num_edges, feature_dim]
    # Output pointer
    grad_x_grad_filters_ptr,  # [num_edges, feature_dim] - OUTPUT (FP32 or FP16)
    # Cutoff parameters
    cutoff_upper,
    # Sizes
    num_nodes,
    feature_dim,
    # Block size
    BLOCK_F: tl.constexpr,
    OUTPUT_FP16: tl.constexpr,  # Whether to output FP16
):
    """
    Fused kernel for grad_filters computation in CFConv backward pass.

    Computes:
        grad_x_grad_filters[e] = grad_grad_filters[e] * grad_output[dst[e]] * cutoff(dist[e])

    Fuses:
        1. Gather grad_output[edge_dst]
        2. Cutoff computation
        3. Elementwise multiply

    Supports FP16 output when OUTPUT_FP16=True (matches filters dtype).
    Computation is always done in FP32 for numerical stability.
    """
    target_node = tl.program_id(axis=0)

    if target_node >= num_nodes:
        return

    seg_start_src = tl.load(src_ptr_ptr + target_node)
    seg_end_src = tl.load(src_ptr_ptr + target_node + 1)

    # Compute cutoff inline (CosineCutoff formula)
    # C = _cosine_cutoff(dist, cutoff_upper)

    for f_start in range(0, feature_dim, BLOCK_F):
        f_offsets = f_start + tl.arange(0, BLOCK_F)
        f_mask = f_offsets < feature_dim

        acc = tl.zeros([BLOCK_F], dtype=tl.float32)
        for e_csr in range(seg_start_src, seg_end_src):
            edge_idx = tl.load(src_perm_ptr + e_csr)

            dst_node = tl.load(edge_dst_ptr + edge_idx)

            distances = tl.load(edge_weight_ptr + edge_idx)

            # Get cutoff

            C = _cosine_cutoff(distances, cutoff_upper)

            # Gather grad_output[dst]
            grad_j = tl.load(
                grad_output_ptr + dst_node * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )

            grad_grad_filters = tl.load(
                grad_grad_filters_ptr + edge_idx * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )

            # Fused multiply: grad * C (in FP32)
            grad_filters = grad_grad_filters * grad_j * C

            # Store result (convert to FP16 if needed)
            if OUTPUT_FP16:
                grad_filters = grad_filters.to(tl.float16)

            acc += grad_filters

        tl.store(
            grad_x_grad_filters_ptr + target_node * feature_dim + f_offsets,
            acc,
            mask=f_mask,
        )


@triton_op("mlcg_kernels::grad_x_grad_filters_fused_cfconv", mutates_args={})
@ensure_contiguous
def grad_x_grad_filters_fused_cfconv(
    grad_output: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    grad_grad_filters: torch.Tensor,
    cutoff_upper: float,
    src_ptr: torch.Tensor,
    src_perm: torch.Tensor,
    out_dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Compute grad_filters in a single fused kernel.

    grad_x_grad_filters[e] = grad_filters[e] * grad_output[dst[e]] * cutoff(dist[e])

    Parameters
    ----------
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
        grad_filters [num_edges, feature_dim]
    """
    num_nodes = grad_output.shape[0]
    feature_dim = grad_output.shape[1]
    num_edges = edge_dst.shape[0]

    # Default output dtype is x.dtype
    if out_dtype is None:
        out_dtype = grad_output.dtype

    grad_x_grad_filters = torch.empty(
        num_edges, feature_dim, device=grad_output.device, dtype=out_dtype
    ).contiguous()

    if num_edges == 0:
        return grad_x_grad_filters

    BLOCK_F = min(128, triton.next_power_of_2(feature_dim))
    grid = (num_nodes,)

    # Determine if output should be FP16
    output_fp16 = out_dtype == torch.float16

    wrap_triton(grad_x_grad_filters_fused_cfconv_kernel)[grid](
        grad_output,
        edge_weight,
        src_perm,
        src_ptr,
        edge_dst,
        grad_grad_filters,
        grad_x_grad_filters,
        cutoff_upper,
        num_nodes,
        feature_dim,
        BLOCK_F=BLOCK_F,
        OUTPUT_FP16=output_fp16,
    )

    return grad_x_grad_filters


def setup_context_grad_x_grad_filters_fused_cfconv(ctx, inputs, output):
    (
        grad_output,
        edge_weight,
        edge_src,
        edge_dst,
        grad_grad_filters,  # FIXME: add this in backward
        cutoff_upper,
        _,
        _,
    ) = inputs

    ctx.save_for_backward(
        grad_output,
        edge_weight,
        edge_src,
        edge_dst,
    )

    ctx.cutoff_upper = cutoff_upper


def backward_grad_x_grad_filters_fused_cfconv(ctx, grad_grad_x_grad_filters):
    (
        grad_output,
        edge_weight,
        edge_src,
        edge_dst,
    ) = ctx.saved_tensors
    grad_grad_out = None
    grad_edge_weight = None
    dC_dd = _torch_d_cosine_cutoff_dd(edge_weight, ctx.cutoff_upper)
    if ctx.needs_input_grad[0]:
        C = _torch_cosine_cutoff(edge_weight, ctx.cutoff_upper)
        grad_grad_out = grad_grad_x_grad_filters * C.unsqueeze(1)
    if ctx.needs_input_grad[1]:
        dC_dd = _torch_d_cosine_cutoff_dd(edge_weight, ctx.cutoff_upper)
        expanded = (
            grad_grad_x_grad_filters
            * grad_output[edge_dst]
            * dC_dd.unsqueeze(1)
        )
        grad_edge_weight = torch.zeros_like(edge_weight).index_add(0, edge_src, expanded)

    return (grad_grad_out, grad_edge_weight, None, None, None, None, None)


grad_x_grad_filters_fused_cfconv.register_autograd(
    backward_grad_x_grad_filters_fused_cfconv,
    setup_context=setup_context_grad_x_grad_filters_fused_cfconv,
)


@grad_x_grad_filters_fused_cfconv.register_kernel("cpu")
def cpu_grad_x_grad_filters_fused_cfconv(
    grad_output: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    grad_grad_filters: torch.Tensor,
    cutoff_upper: float,
    src_ptr: torch.Tensor,
    src_perm: torch.Tensor,
    out_dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    CPU fallback for fused_grad_filters
    """
    C = _torch_cosine_cutoff(edge_weight, cutoff_upper)
    expanded = grad_grad_filters * grad_output[edge_dst] * C.unsqueeze(-1)
    return torch.zeros_like(grad_output).index_add(0, edge_src, expanded)


# ============================================================================
# grad_grad_out_grad_filters_fused_cfconv
# ============================================================================


@triton.jit
def grad_grad_out_grad_filters_fused_cfconv_kernel(
    # Input pointers
    x_ptr,  # [num_nodes, feature_dim]
    edge_weight_ptr,  # [num_edges]
    edge_src_ptr,  # [num_edges]
    grad_grad_filters_ptr,  # [num_edges, feature_dim]
    # Output ptr
    grad_grad_out_grad_filters_ptr,  # [num_edges, feature_dim] - OUTPUT (FP32 or FP16)
    # Cutoff parameters
    dst_perm_ptr,  # [num_edges] - dst-CSR permutation
    dst_ptr_ptr,  # [num_nodes + 1] - dst-CSR row pointers
    cutoff_upper,
    # Sizes
    num_nodes,
    feature_dim,
    # Block size
    BLOCK_F: tl.constexpr,
    OUTPUT_FP16: tl.constexpr,  # Whether to output FP16
):
    """
    Fused kernel for grad_filters computation in CFConv backward pass.

    Computes:
        grad_filters[e] = grad_grad_filters[e] * x[src[e]] * cutoff(dist[e])

    Fuses:
        1. Gather x[edge_src]
        2. Cutoff computation
        3. Elementwise multiply

    Supports FP16 output when OUTPUT_FP16=True (matches filters dtype).
    Computation is always done in FP32 for numerical stability.
    """
    target_node = tl.program_id(axis=0)

    if target_node >= num_nodes:
        return

    seg_start_dst = tl.load(dst_ptr_ptr + target_node)
    seg_end_dst = tl.load(dst_ptr_ptr + target_node + 1)

    for f_start in range(0, feature_dim, BLOCK_F):
        f_offsets = f_start + tl.arange(0, BLOCK_F)
        f_mask = f_offsets < feature_dim

        acc = tl.zeros([BLOCK_F], dtype=tl.float32)
        for e_csr in range(seg_start_dst, seg_end_dst):
            edge_idx = tl.load(dst_perm_ptr + e_csr)

            src_node = tl.load(edge_src_ptr + edge_idx)

            distances = tl.load(edge_weight_ptr + edge_idx)

            C = _cosine_cutoff(distances, cutoff_upper)
            x = tl.load(
                x_ptr + src_node * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )
            grad_grad_filters = tl.load(
                grad_grad_filters_ptr + edge_idx * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,  # FIXME: here
            )
            # Fused multiply: x * grad * C (in FP32)
            acc += grad_grad_filters * x * C[:, None]

        # Store result (convert to FP16 if needed)

        tl.store(
            grad_grad_out_grad_filters_ptr + target_node * feature_dim + f_offsets,
            acc,
            mask=f_mask,
        )


@triton_op(
    "mlcg_kernels::grad_grad_out_grad_filters_fused_cfconv", mutates_args={}
)
@ensure_contiguous
def grad_grad_out_grad_filters_fused_cfconv(
    x: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    grad_grad_filters: torch.Tensor,
    dst_perm: torch.Tensor,
    dst_ptr: torch.Tensor,
    cutoff_upper: float,
    out_dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Compute grad_filters in a single fused kernel.

    grad_filters[e] = grad_filters[e] x[src[e]] * grad_output[dst[e]] * cutoff(dist[e])

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
        grad_filters [num_edges, feature_dim]
    """
    num_nodes = x.shape[0]
    feature_dim = x.shape[1]
    num_edges = edge_src.shape[0]

    # Default output dtype is x.dtype
    if out_dtype is None:
        out_dtype = x.dtype

    grad_grad_out_grad_filters = torch.empty(
        num_edges, feature_dim, device=x.device, dtype=out_dtype
    ).contiguous()

    if num_edges == 0:
        return grad_grad_out_grad_filters

    BLOCK_F = 128
    grid = (num_nodes,)

    # Determine if output should be FP16
    output_fp16 = out_dtype == torch.float16

    wrap_triton(grad_grad_out_grad_filters_fused_cfconv_kernel)[grid](
        x,
        edge_weight,
        edge_src,
        grad_grad_filters,
        grad_grad_out_grad_filters,
        dst_perm,
        dst_ptr,
        cutoff_upper,
        num_nodes,
        feature_dim,
        BLOCK_F=BLOCK_F,
        OUTPUT_FP16=output_fp16,
    )

    return grad_grad_out_grad_filters


def setup_context_grad_grad_out_grad_filters_fused_cfconv(ctx, inputs, output):
    (
        x,
        edge_weight,
        edge_src,
        edge_dst,
        _,
        _,
        cutoff_upper,
    ) = inputs
    ctx.save_for_backward(
        x,
        edge_weight,
        edge_src,
        edge_dst,
    )
    ctx.cutoff_upper = cutoff_upper


def backward_grad_grad_out_grad_filters_fused_cfconv(
    ctx, grad_grad_out_grad_filters
):
    (
        x,
        edge_weight,
        edge_src,
        edge_dst,
    ) = ctx.saved_tensors
    grad_x = None
    grad_edge_weight = None
    if ctx.needs_input_grad[0]:
        C = _torch_cosine_cutoff(edge_weight, ctx.cutoff_upper)
        grad_x = grad_grad_out_grad_filters * C.unsqueeze(1)
    if ctx.needs_input_grad[1]:
        dC_dd = _torch_d_cosine_cutoff_dd(edge_weight, ctx.cutoff_upper)
        expanded = grad_grad_out_grad_filters * x[edge_src] * dC_dd.unsqueeze(1)
        grad_edge_weight = torch.zeros_like(edge_weight).index_add(0, edge_dst, expanded)
    return (
        grad_x,
        grad_edge_weight,
        None,
        None,
        None,
        None,
    )


grad_grad_out_grad_filters_fused_cfconv.register_autograd(
    backward_grad_grad_out_grad_filters_fused_cfconv,
    setup_context=setup_context_grad_grad_out_grad_filters_fused_cfconv,
)


@grad_grad_out_grad_filters_fused_cfconv.register_kernel("cpu")
def cpu_grad_grad_out_grad_filters_fused_cfconv(
    x: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    grad_grad_filters: torch.Tensor,
    dst_perm: torch.Tensor,
    dst_ptr: torch.Tensor,
    cutoff_upper: float,
    out_dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    CPU fallback for fused_grad_filters
    """
    C = _torch_cosine_cutoff(edge_weight, cutoff_upper)
    expanded = grad_grad_filters * x[edge_src] * C.unsqueeze(-1)
    return torch.zeros_like(x).index_add(0, edge_dst, expanded)


# ============================================================================
# grad_edge_weight_grad_filters_fused_cfconv
# ============================================================================


@triton.jit
def grad_edge_weight_grad_filters_fused_cfconv_kernel(
    # Input pointers
    x_ptr,  # [num_nodes, feature_dim]
    grad_output_ptr,  # [num_nodes, feature_dim]
    edge_weight_ptr,  # [num_edges]
    edge_src_ptr,  # [num_edges]
    edge_dst_ptr,  # [num_edges]
    grad_grad_filters_ptr,  # [num_edges, feature_dim]
    # Output ptr
    grad_filters_ptr,  # [num_edges, feature_dim] - OUTPUT (FP32 or FP16)
    # Cutoff parameters
    cutoff_upper,
    # Sizes
    num_edges,
    feature_dim,
    # Block size
    BLOCK_F: tl.constexpr,
    OUTPUT_FP16: tl.constexpr,  # Whether to output FP16
):
    """
    Fused kernel for grad_filters computation in CFConv backward pass.

    Computes:
        grad_filters[e] = grad_grad_filters[e] x[src[e]] * grad_output[dst[e]] * d_cutoff_dd(dist[e])

    Fuses:
        1. Gather x[edge_src]
        2. Gather grad_output[edge_dst]
        3. Cutoff computation
        4. Elementwise multiply

    Memory savings: Eliminates two intermediate tensors (x_gathered, grad_gathered)

    Supports FP16 output when OUTPUT_FP16=True (matches filters dtype).
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
    dC = _d_cosine_cutoff_dd(dist, cutoff_upper)

    # Process features in blocks
    for f_start in range(0, feature_dim, BLOCK_F):
        f_offsets = f_start + tl.arange(0, BLOCK_F)
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
        grad_grad_filters = tl.load(
            grad_grad_filters_ptr + edge_idx * feature_dim + f_offsets,
            mask=f_mask,
            other=0.0,
        )

        # Fused multiply: x * grad * C (in FP32)
        grad_filters = grad_grad_filters * x_j * grad_j * dC

        # Store result (convert to FP16 if needed)
        if OUTPUT_FP16:
            grad_filters = grad_filters.to(tl.float16)
        tl.store(
            grad_filters_ptr + edge_idx * feature_dim + f_offsets,
            grad_filters,
            mask=f_mask,
        )


@triton_op(
    "mlcg_kernels::grad_edge_weight_grad_filters_fused_cfconv", mutates_args={}
)
@ensure_contiguous
def grad_edge_weight_grad_filters_fused_cfconv(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    grad_grad_filters: torch.Tensor,
    cutoff_upper: float,
    out_dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Compute grad_edge_grad_filters in a single fused kernel.

    grad_edge_grad_filters[e] = grad_filters[e] x[src[e]] * grad_output[dst[e]] * d_cutoff_dd(dist[e])

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
        grad_filters [num_edges, feature_dim]
    """
    feature_dim = x.shape[1]
    num_edges = edge_src.shape[0]

    # Default output dtype is x.dtype
    if out_dtype is None:
        out_dtype = x.dtype

    grad_filters = torch.empty(
        num_edges, feature_dim, device=x.device, dtype=out_dtype
    ).contiguous()

    if num_edges == 0:
        return grad_filters

    BLOCK_F = min(128, triton.next_power_of_2(feature_dim))
    grid = (num_edges,)

    # Determine if output should be FP16
    output_fp16 = out_dtype == torch.float16

    wrap_triton(grad_edge_weight_grad_filters_fused_cfconv_kernel)[grid](
        x,
        grad_output,
        edge_weight,
        edge_src,
        edge_dst,
        grad_grad_filters,
        grad_filters,
        cutoff_upper,
        num_edges,
        feature_dim,
        BLOCK_F=BLOCK_F,
        OUTPUT_FP16=output_fp16,
    )

    return grad_filters


def setup_context_grad_edge_weight_grad_filters_fused_cfconv(
    ctx, inputs, output
):
    x, grad_output, edge_weight, edge_src, edge_dst, cutoff_upper = inputs
    ctx.save_for_backward(x, grad_output, edge_weight, edge_src, edge_dst)
    ctx.cutoff_upper = cutoff_upper


def backward_grad_edge_weight_grad_filters_fused_cfconv(
    ctx, grad_grad_edge_weight_grad_filters
):
    x, grad_output, edge_weight, edge_src, edge_dst = ctx.saved_tensors
    dC_dd = _torch_d_cosine_cutoff_dd(edge_weight, ctx.cutoff_upper)
    grad_grad_x = None
    grad_grad_out = None
    grad_grad_edge_weights = None
    if ctx.needs_input_grad[0]:
        # gradient respect to x
        expanded = (
            grad_grad_edge_weight_grad_filters
            * grad_output[edge_dst]
            * dC_dd.unsqueeze(1)
        )
        grad_grad_x = torch.zeros_like(x).index_add(0, edge_src, expanded)
    if ctx.needs_input_grad[1]:
        # gradient respect to grad_out
        expanded = (
            grad_grad_edge_weight_grad_filters
            * x[edge_src]
            * dC_dd.unsqueeze(1)
        )
        grad_grad_out = torch.zeros_like(x).index_add(0, edge_dst, expanded)
    if ctx.needs_input_grad[2]:
        # gradient respect to edge weights
        d2C_dd2 = _torch_d2_cosine_cutoff_dd2(edge_weight, ctx.cutoff_upper)
        grad_grad_edge_weights = (
            grad_grad_edge_weight_grad_filters
            * x[edge_src]
            * grad_output[edge_dst]
        ).sum(dim=1) * d2C_dd2
    return (
        grad_grad_x,
        grad_grad_out,
        grad_grad_edge_weights,
        None,
        None,
        None,
        None,
    )


grad_edge_weight_grad_filters_fused_cfconv.register_autograd(
    backward_grad_edge_weight_grad_filters_fused_cfconv,
    setup_context=setup_context_grad_edge_weight_grad_filters_fused_cfconv,
)


@grad_edge_weight_grad_filters_fused_cfconv.register_kernel("cpu")
def cpu_grad_edge_weight_grad_filters_fused_cfconv(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    grad_grad_filters: torch.Tensor,
    cutoff_upper: float,
    out_dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    CPU fallback for fused_grad_filters
    """
    dC = _torch_d_cosine_cutoff_dd(edge_weight, cutoff_upper)
    grad_edge_grad_filters = (
        grad_grad_filters
        * grad_output[edge_dst]
        * x[edge_src]
        * dC.unsqueeze(-1)
    )
    return grad_edge_grad_filters
