"""
Fused Triton kernels for second-order gradient computations in SchNet CFConv (Continuous Filter Convolution).

This module implements optimized backward pass operations for SchNet's continuous filter convolution
layer. It provides fused GPU kernels (Triton) and CPU fallback implementations for computing gradients
with respect to node features (x), output features (grad_output), filter outputs (filters), and edge
weights (distances).

The kernels use sparse CSR format for efficient edge aggregation and support mixed precision
computation (FP16/FP32).

Main functions:
    - grad_x_grad_edge_weight_fused_cfconv: Compute gradients w.r.t input node features
    - grad_grad_out_grad_edge_weight_fused_cfconv: Compute gradients w.r.t output features
    - grad_filters_grad_edge_weight_fused_cfconv: Compute gradients w.r.t filter outputs
    - grad_edge_weight_grad_edge_weight_fused_cfconv: Compute gradients w.r.t edge distances

Each operation includes:
    - Triton kernel for GPU computation
    - PyTorch wrapper
    - Autograd backward pass
    - CPU fallback implementation
"""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from ....utils import ensure_contiguous

from ...cutoffs import (
    _d_cosine_cutoff_dd,
    _d2_cosine_cutoff_dd2,
    _torch_d_cosine_cutoff_dd,
    _torch_d2_cosine_cutoff_dd2,
    _torch_d3_cosine_cutoff_dd3,
)

triton_pi = tl.constexpr(3.141592653589793)


# ============================================================================
# grad_x_grad_edge_weight_fused_cfconv
# ============================================================================


@triton.jit
def grad_x_grad_edge_weight_fused_cfconv_kernel(
    # Input pointers
    grad_output_ptr,  # [num_nodes, feature_dim]
    filters_ptr,  # [num_edges, feature_dim] - filters outputs (FP32 or FP16)
    edge_weight_ptr,  # [num_edges]
    edge_dst_ptr,  # [num_edges]
    grad_edge_out_ptr,  # [num_edges, feature_dim] - OUTPUT (FP32 or FP16)
    src_perm_ptr,  # [num_edges] - src-CSR permutation
    src_ptr_ptr,  # [num_nodes + 1] - src-CSR row pointers
    # Output pointers
    grad_x_ptr,  # [num_nodes, feature_dim]
    # Cutoff parameters
    cutoff_upper,
    # Sizes
    num_nodes,
    feature_dim,
    # Block size #FIXME: check if other casting are needed
    BLOCK_F: tl.constexpr,
    filters_FP16: tl.constexpr,
    GRAD_EDGE_FP16: tl.constexpr,  # Whether to output FP16
):
    """
    Triton kernel: Compute grad_x by aggregating over edges using source-CSR format.

    Grid layout: One thread block per source node
    Computation: For each node, iterate through all its outgoing edges and accumulate
    gradients by multiplying edge gradients, filter outputs, downstream gradients, and cutoff derivatives.

    Computes:
        Computes: grad_x[src] = sum_{e: src[e]=src} grad_edge[e] * grad_output[dst[e]] * filters[e] * d_cutoff_dd[e]
    """
    target_node = tl.program_id(axis=0)

    if target_node >= num_nodes:
        return

    seg_start_src = tl.load(src_ptr_ptr + target_node)
    seg_end_src = tl.load(src_ptr_ptr + target_node + 1)

    for f_start in range(0, feature_dim, BLOCK_F):
        f_offsets = f_start + tl.arange(0, BLOCK_F)
        f_mask = f_offsets < feature_dim

        acc = tl.zeros([BLOCK_F], dtype=tl.float32)
        for e_csr in range(seg_start_src, seg_end_src):
            edge_idx = tl.load(src_perm_ptr + e_csr)

            dst_node = tl.load(edge_dst_ptr + edge_idx)

            distances = tl.load(edge_weight_ptr + edge_idx)
            C = _d_cosine_cutoff_dd(distances, cutoff_upper)
            grad_output = tl.load(
                grad_output_ptr + dst_node * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )
            filters = tl.load(
                filters_ptr + edge_idx * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )
            if filters_FP16:
                filters = filters.to(tl.float32)
            grad_edge = tl.load(
                grad_edge_out_ptr + edge_idx * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )
            if GRAD_EDGE_FP16:
                grad_edge = grad_edge.to(tl.float32)

            acc += (
                grad_edge * filters * grad_output * C[:, None]
            )  # FIXME: check this broadcast, maybe just C

        tl.store(
            grad_x_ptr + target_node * feature_dim + f_offsets, acc, mask=f_mask
        )


@triton_op(
    "mlcg_kernels::grad_x_grad_edge_weight_fused_cfconv", mutates_args={}
)
@ensure_contiguous
def grad_x_grad_edge_weight_fused_cfconv(
    grad_output: torch.Tensor,
    filters: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    grad_edge_out: torch.Tensor,
    src_perm: torch.Tensor,
    src_ptr: torch.Tensor,
    cutoff_upper: float,
    grad_edge_dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Compute gradients w.r.t. source node features in CFConv second-order backward pass.

    Computes: grad_x[src] = sum_{e: src[e]=src} grad_edge[e] * grad_output[dst[e]] * filters[e] * d_cutoff_dd[e]

    This operation aggregates gradients from all edges appearing in the source nodes' neighborhoods,
    multiplying by the filter outputs and the gradient of the cosine cutoff function.

    Args:
        grad_output: Gradients of output features [num_nodes, feature_dim]
        filters: Filter outputs from forward pass [num_edges, feature_dim], FP32 or FP16
        edge_weight: Edge distances/weights [num_edges]
        edge_src: Source node indices [num_edges]
        edge_dst: Destination node indices [num_edges]
        grad_edge_out: Gradients w.r.t edge representations [num_edges, feature_dim]
        src_perm: Permutation array for source CSR format [num_edges]
        src_ptr: Row pointers for source CSR format [num_nodes + 1]
        cutoff_upper: Upper bound for cosine cutoff function
        grad_edge_dtype: Data type of grad_edge_out (FP32 or FP16)

    Returns:
        Gradients w.r.t. source node features [num_nodes, feature_dim], dtype=float32
    """

    num_nodes = grad_output.shape[0]
    feature_dim = grad_output.shape[1]
    num_edges = edge_dst.shape[0]

    # Allocate output (zeros for nodes with no outgoing edges)
    grad_x = torch.zeros(
        num_nodes, feature_dim, device=grad_output.device, dtype=torch.float32
    ).contiguous()

    if num_edges == 0:
        return grad_x

    filters_fp16 = filters.dtype == torch.float16
    grad_edge_fp16 = grad_edge_dtype == torch.float16

    BLOCK_F = 128
    grid = (num_nodes,)

    wrap_triton(grad_x_grad_edge_weight_fused_cfconv_kernel)[grid](
        grad_output,
        filters,
        edge_weight,
        edge_dst,
        grad_edge_out,
        src_perm,
        src_ptr,
        grad_x,
        cutoff_upper,
        num_nodes,
        feature_dim,
        BLOCK_F=BLOCK_F,
        filters_FP16=filters_fp16,
        GRAD_EDGE_FP16=grad_edge_fp16,
    )

    return grad_x


def setup_context_grad_x_grad_edge_weight_fused_cfconv(ctx, inputs, output):
    (
        grad_output,
        filters,
        edge_weight,
        edge_src,
        edge_dst,
        grad_edge_out,
        src_perm,
        src_ptr,
        cutoff_upper,
        grad_edge_dtype,
    ) = inputs

    ctx.save_for_backward(
        grad_output,
        filters,
        edge_weight,
        edge_src,
        edge_dst,
        grad_edge_out,
    )

    ctx.cutoff_upper = cutoff_upper


def backward_grad_x_grad_edge_weight_fused_cfconv(ctx, grad_grad_x):
    (
        grad_output,
        filters,
        edge_weight,
        edge_src,
        edge_dst,
        grad_edge_out,
    ) = ctx.saved_tensors

    cutoff_upper = ctx.cutoff_upper

    grad_grad_output = grad_filters = grad_edge_weight = grad_grad_edge_out = (
        None
    )

    dC_dd = _torch_d_cosine_cutoff_dd(edge_weight, cutoff_upper)

    if ctx.needs_input_grad[0]:
        grad_grad_output = torch.zeros_like(grad_output)
        expanded = (
            grad_grad_x[edge_src] * grad_edge_out * filters * dC_dd.unsqueeze(1)
        )
        grad_grad_output.index_add_(0, edge_dst, expanded)

    if ctx.needs_input_grad[1]:
        grad_filters = (
            grad_grad_x[edge_src]
            * grad_edge_out
            * grad_output[edge_dst]
            * dC_dd.unsqueeze(1)
        )

    if ctx.needs_input_grad[2]:
        d2C_dd2 = _torch_d2_cosine_cutoff_dd2(edge_weight, cutoff_upper)
        grad_edge_weight = (
            grad_grad_x[edge_src]
            * grad_edge_out
            * grad_output[edge_dst]
            * filters
        ).sum(dim=1) * d2C_dd2

    if ctx.needs_input_grad[5]:
        grad_grad_edge_out = (
            grad_grad_x[edge_src]
            * grad_output[edge_dst]
            * filters
            * dC_dd.unsqueeze(1)
        )

    return (
        grad_grad_output,
        grad_filters,
        grad_edge_weight,
        None,
        None,
        grad_grad_edge_out,
        None,
        None,
        None,
        None,
    )


grad_x_grad_edge_weight_fused_cfconv.register_autograd(
    backward_grad_x_grad_edge_weight_fused_cfconv,
    setup_context=setup_context_grad_x_grad_edge_weight_fused_cfconv,
)


@grad_x_grad_edge_weight_fused_cfconv.register_kernel("cpu")
def cpu_grad_x_grad_edge_weight_fused_cfconv(
    grad_output: torch.Tensor,
    filters: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    grad_edge_out: torch.Tensor,
    src_perm: torch.Tensor,
    src_ptr: torch.Tensor,
    cutoff_upper: float,
    grad_edge_dtype: torch.dtype = None,
) -> torch.Tensor:

    dC_dd = _torch_d_cosine_cutoff_dd(edge_weight, cutoff_upper)
    grad_x = torch.zeros(grad_output.shape[0]).contiguous()
    expanded = (
        grad_edge_out * grad_output[edge_dst] * filters * dC_dd.unsqueeze(1)
    )
    grad_x.index_add_(0, edge_src, expanded)

    return grad_x  # FIXME: check dtype


# ============================================================================
# grad_grad_out_grad_edge_weight_fused_cfconv
# ============================================================================


@triton.jit
def grad_grad_out_grad_edge_weight_fused_cfconv_kernel(
    # Input pointers
    x_ptr,  # [num_nodes, feature_dim]
    filters_ptr,  # [num_edges, feature_dim] - filters outputs (FP32 or FP16)
    edge_weight_ptr,  # [num_edges]
    edge_src_ptr,  # [num_edges]
    grad_edge_out_ptr,  # [num_edges, feature_dim] - OUTPUT (FP32 or FP16)
    dst_perm_ptr,  # [num_edges] - dst-CSR permutation
    dst_ptr_ptr,  # [num_nodes + 1] - dst-CSR row pointers
    # Output pointers
    grad_grad_out_ptr,  # [num_nodes, feature_dim]
    # Cutoff parameters
    cutoff_upper,
    # Sizes
    num_nodes,
    feature_dim,
    # Block size #FIXME: check if other casting are needed
    BLOCK_F: tl.constexpr,
    filters_FP16: tl.constexpr,
    GRAD_EDGE_FP16: tl.constexpr,  # Whether to output FP16
):
    """
    Triton kernel: Compute grad_grad_out by aggregating over edges using destination-CSR format.

    Grid layout: One thread block per destination node
    Computation: For each node, iterate through all its incoming edges and accumulate
    gradients by multiplying edge gradients, source features, filter outputs, and cutoff derivatives.

    Computes:
        Computes: grad_grad_out[dst] = sum_{e: dst[e]=dst} grad_edge[e] * x[src[e]] * filters[e] * d_cutoff_dd[e]

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
            C = _d_cosine_cutoff_dd(distances, cutoff_upper)
            x = tl.load(
                x_ptr + src_node * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )
            filters = tl.load(
                filters_ptr + edge_idx * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )
            if filters_FP16:
                filters = filters.to(tl.float32)
            grad_edge = tl.load(
                grad_edge_out_ptr + edge_idx * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )
            if GRAD_EDGE_FP16:
                grad_edge = grad_edge.to(tl.float32)

            acc += (
                grad_edge * x * filters * C[:, None]
            )  # FIXME: check this broadcast, maybe just C

        tl.store(
            grad_grad_out_ptr + target_node * feature_dim + f_offsets,
            acc,
            mask=f_mask,
        )


@triton_op(
    "mlcg_kernels::grad_grad_out_grad_edge_weight_fused_cfconv", mutates_args={}
)
@ensure_contiguous
def grad_grad_out_grad_edge_weight_fused_cfconv(
    x: torch.Tensor,
    filters: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    grad_edge_out: torch.Tensor,
    dst_perm: torch.Tensor,
    dst_ptr: torch.Tensor,
    cutoff_upper: float,
    grad_edge_dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Compute gradients w.r.t. destination node outputs in CFConv second-order backward pass.

    Computes: grad_grad_out[dst] = sum_{e: dst[e]=dst} grad_edge[e] * x[src[e]] * filters[e] * d_cutoff_dd[e]

    This operation aggregates gradients from edges with a common destination node, multiplying by
    source node features, filter outputs, and the gradient of the cosine cutoff function.

    Args:
        x: Input node features [num_nodes, feature_dim]
        filters: Filter outputs from forward pass [num_edges, feature_dim], FP32 or FP16
        edge_weight: Edge distances/weights [num_edges]
        edge_src: Source node indices [num_edges]
        edge_dst: Destination node indices [num_edges]
        grad_edge_out: Gradients w.r.t edge representations [num_edges, feature_dim]
        dst_perm: Permutation array for destination CSR format [num_edges]
        dst_ptr: Row pointers for destination CSR format [num_nodes + 1]
        cutoff_upper: Upper bound for cosine cutoff function
        grad_edge_dtype: Data type of grad_edge_out (FP32 or FP16)

    Returns:
        Gradients w.r.t. destination node outputs [num_nodes, feature_dim], dtype=float32
    """

    num_nodes = x.shape[0]
    num_edges = edge_src.shape[0]
    feature_dim = x.shape[1]

    # Allocate output (zeros for nodes with no outgoing edges)
    grad_grad_out = torch.zeros(
        num_nodes, feature_dim, device=x.device, dtype=torch.float32
    ).contiguous()

    if num_edges == 0:
        return grad_grad_out

    filters_fp16 = filters.dtype == torch.float16
    grad_edge_fp16 = grad_edge_dtype == torch.float16

    BLOCK_F = 128
    grid = (num_nodes,)

    wrap_triton(grad_grad_out_grad_edge_weight_fused_cfconv_kernel)[grid](
        x,
        filters,
        edge_weight,
        edge_src,
        grad_edge_out,
        dst_perm,
        dst_ptr,
        grad_grad_out,
        cutoff_upper,
        num_nodes,
        feature_dim,
        BLOCK_F=BLOCK_F,
        filters_FP16=filters_fp16,
        GRAD_EDGE_FP16=grad_edge_fp16,
    )

    return grad_grad_out


def setup_context_grad_grad_out_grad_edge_weight_fused_cfconv(
    ctx, inputs, output
):
    (
        x,
        filters,
        edge_weight,
        edge_src,
        edge_dst,
        grad_edge_out,
        dst_perm,
        dst_ptr,
        cutoff_upper,
        grad_edge_dtype,
    ) = inputs

    ctx.save_for_backward(
        x, filters, edge_weight, edge_src, edge_dst, grad_edge_out
    )

    ctx.cutoff_upper = cutoff_upper


def backward_grad_grad_out_grad_edge_weight_fused_cfconv(
    ctx, grad_grad_grad_out
):
    (
        x,
        filters,
        edge_weight,
        edge_src,
        edge_dst,
        grad_edge_out,
    ) = ctx.saved_tensors

    cutoff_upper = ctx.cutoff_upper

    grad_x = grad_filters = grad_edge_weight = grad_grad_edge_out = None

    dC_dd = _torch_d_cosine_cutoff_dd(edge_weight, cutoff_upper)

    if ctx.needs_input_grad[0]:
        grad_x = torch.zeros_like(x)
        expanded = (
            grad_grad_grad_out[edge_dst]
            * grad_edge_out
            * filters
            * dC_dd.unsqueeze(1)
        )
        grad_x.index_add_(0, edge_src, expanded)

    if ctx.needs_input_grad[1]:
        grad_filters = (
            grad_grad_grad_out[edge_dst]
            * grad_edge_out
            * x[edge_src]
            * dC_dd.unsqueeze(1)
        )

    if ctx.needs_input_grad[2]:
        d2C_dd2 = _torch_d2_cosine_cutoff_dd2(edge_weight, cutoff_upper)
        grad_edge_weight = (
            grad_grad_grad_out[edge_dst] * grad_edge_out * x[edge_src] * filters
        ).sum(dim=1) * d2C_dd2

    if ctx.needs_input_grad[5]:
        grad_grad_edge_out = (
            grad_grad_grad_out[edge_dst]
            * x[edge_src]
            * filters
            * dC_dd.unsqueeze(1)
        )

    return (
        grad_x,
        grad_filters,
        grad_edge_weight,
        None,
        None,
        grad_grad_edge_out,
        None,
        None,
        None,
        None,
    )


grad_grad_out_grad_edge_weight_fused_cfconv.register_autograd(
    backward_grad_grad_out_grad_edge_weight_fused_cfconv,
    setup_context=setup_context_grad_grad_out_grad_edge_weight_fused_cfconv,
)


@grad_grad_out_grad_edge_weight_fused_cfconv.register_kernel("cpu")
def cpu_grad_grad_out_grad_edge_weight_fused_cfconv(
    x: torch.Tensor,
    filters: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    grad_edge_out: torch.Tensor,
    dst_perm: torch.Tensor,
    dst_ptr: torch.Tensor,
    cutoff_upper: float,
    grad_edge_dtype: torch.dtype = None,
) -> torch.Tensor:

    dC_dd = _torch_d_cosine_cutoff_dd(edge_weight, cutoff_upper)
    grad_grad_out = torch.zeros_like(x)
    expanded = grad_edge_out * x[edge_src] * filters * dC_dd.unsqueeze(1)
    grad_grad_out.index_add_(0, edge_dst, expanded)

    return grad_grad_out  # FIXME: check dtype


# ============================================================================
# grad_filters_grad_edge_weight_fused_cfconv
# ============================================================================


@triton.jit
def grad_filters_grad_edge_weight_fused_cfconv_kernel(
    # Input pointers
    x_ptr,  # [num_nodes, feature_dim]
    grad_output_ptr,  # [num_nodes, feature_dim]
    edge_weight_ptr,  # [num_edges]
    edge_src_ptr,  # [num_edges]
    edge_dst_ptr,  # [num_edges]
    grad_edge_out_ptr,  # [num_edges, feature_dim] - OUTPUT (FP32 or FP16)
    # Output pointers
    grad_filters_ptr,  # [num_nodes, feature_dim]
    # Cutoff parameters
    cutoff_upper,
    # Sizes
    num_edges,
    feature_dim,
    # Block size #FIXME: check if other casting are needed
    BLOCK_F: tl.constexpr,
    GRAD_EDGE_FP16: tl.constexpr,  # Whether to output FP16
):
    """
    Triton kernel: Compute grad_filters for each edge independently.

    Grid layout: One thread block per edge
    Computation: For each edge, compute the product of gradients, cutoff derivative, and input features.
    No aggregation needed - one output per edge.

    Computes:
        Computes: grad_filters[e] =  grad_edge[e] * grad_output[dst[e]] * x[src[e]] * d_cutoff_dd[e]

    """
    edge_idx = tl.program_id(axis=0)

    if edge_idx >= num_edges:
        return

    src_node = tl.load(edge_src_ptr + edge_idx)
    dst_node = tl.load(edge_dst_ptr + edge_idx)
    dist = tl.load(edge_weight_ptr + edge_idx)
    C = _d_cosine_cutoff_dd(dist, cutoff_upper)

    for f_start in range(0, feature_dim, BLOCK_F):
        f_offset = f_start + tl.arange(0, BLOCK_F)
        f_mask = f_offset < feature_dim

        grad_out = tl.load(
            grad_output_ptr + dst_node * feature_dim + f_offset,
            mask=f_mask,
            other=0.0,
        )
        x = tl.load(
            x_ptr + src_node * feature_dim + f_offset, mask=f_mask, other=0.0
        )
        grad_edge = tl.load(
            grad_edge_out_ptr + edge_idx * feature_dim + f_offset,
            mask=f_mask,
            other=0.0,
        )
        if GRAD_EDGE_FP16:
            grad_edge = grad_edge.to(tl.float32)
        tl.store(
            grad_filters_ptr + edge_idx * feature_dim + f_offset,
            grad_edge
            * grad_out
            * x
            * C[:, None],  # FIXME: check this broadcast, maybe just C
            mask=f_mask,
        )


@triton_op(
    "mlcg_kernels::grad_filters_grad_edge_weight_fused_cfconv", mutates_args={}
)
@ensure_contiguous
def grad_filters_grad_edge_weight_fused_cfconv(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    grad_edge_out: torch.Tensor,
    cutoff_upper: float,
    grad_edge_dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Compute gradients w.r.t. filter outputs in CFConv second-order backward pass.

    Computes: grad_filters[e] = grad_edge[e] * grad_output[dst[e]] * x[src[e]] * d_cutoff_dd[e]

    This operation computes gradients for each edge directly, without any aggregation, by multiplying
    the edge gradients with the corresponding source and destination features and the cutoff gradient.

    Args:
        x: Input node features [num_nodes, feature_dim]
        grad_output: Gradients of output features [num_nodes, feature_dim]
        edge_weight: Edge distances/weights [num_edges]
        edge_src: Source node indices [num_edges]
        edge_dst: Destination node indices [num_edges]
        grad_edge_out: Gradients w.r.t edge representations [num_edges, feature_dim]
        cutoff_upper: Upper bound for cosine cutoff function
        grad_edge_dtype: Data type of grad_edge_out (FP32 or FP16)

    Returns:
        Gradients w.r.t. filter outputs [num_edges, feature_dim], dtype=float32
    """

    feature_dim = x.shape[1]
    num_edges = edge_src.shape[0]

    # Allocate output (zeros for nodes with no outgoing edges)
    grad_filters = torch.zeros(
        num_edges, feature_dim, device=x.device, dtype=torch.float32
    ).contiguous()  # FIXME: check dtype is correct

    if num_edges == 0:
        return grad_filters

    grad_edge_fp16 = grad_edge_dtype == torch.float16

    BLOCK_F = 128
    grid = (num_edges,)

    wrap_triton(grad_filters_grad_edge_weight_fused_cfconv_kernel)[grid](
        x,
        grad_output,
        edge_weight,
        edge_src,
        edge_dst,
        grad_edge_out,
        grad_filters,
        cutoff_upper,
        num_edges,
        feature_dim,
        BLOCK_F=BLOCK_F,
        GRAD_EDGE_FP16=grad_edge_fp16,
    )

    return grad_filters


def setup_context_grad_filters_grad_edge_weight_fused_cfconv(
    ctx, inputs, output
):
    (
        x,
        grad_output,
        edge_weight,
        edge_src,
        edge_dst,
        grad_edge_out,
        cutoff_upper,
        grad_edge_dtype,
    ) = inputs

    ctx.save_for_backward(
        x,
        grad_output,
        edge_weight,
        edge_src,
        edge_dst,
        grad_edge_out,
    )

    ctx.cutoff_upper = cutoff_upper


def backward_grad_filters_grad_edge_weight_fused_cfconv(ctx, grad_grad_filters):
    (
        x,
        grad_output,
        edge_weight,
        edge_src,
        edge_dst,
        grad_edge_out,
    ) = ctx.saved_tensors

    cutoff_upper = ctx.cutoff_upper

    grad_x = grad_grad_output = grad_edge_weight = grad_grad_edge_out = None

    dC_dd = _torch_d_cosine_cutoff_dd(edge_weight, cutoff_upper)

    if ctx.needs_input_grad[0]:
        grad_x = torch.zeros_like(x)
        expanded = (
            grad_grad_filters
            * grad_edge_out
            * grad_output[edge_dst]
            * dC_dd.unsqueeze(1)
        )
        grad_x.index_add_(0, edge_src, expanded)

    if ctx.needs_input_grad[1]:
        grad_grad_output = torch.zeros_like(grad_output)
        expanded = (
            grad_grad_filters * grad_edge_out * x[edge_src] * dC_dd.unsqueeze(1)
        )
        grad_grad_output.index_add_(0, edge_dst, expanded)

    if ctx.needs_input_grad[2]:
        d2C_dd2 = _torch_d2_cosine_cutoff_dd2(edge_weight, cutoff_upper)
        grad_edge_weight = (
            grad_grad_filters
            * grad_edge_out
            * grad_output[edge_dst]
            * x[edge_src]
        ).sum(dim=1) * d2C_dd2

    if ctx.needs_input_grad[5]:
        grad_grad_edge_out = (
            grad_grad_filters
            * grad_output[edge_dst]
            * x[edge_src]
            * dC_dd.unsqueeze(1)
        )

    return (
        grad_x,
        grad_grad_output,
        grad_edge_weight,
        None,
        None,
        grad_grad_edge_out,
        None,
        None,
    )


grad_filters_grad_edge_weight_fused_cfconv.register_autograd(
    backward_grad_filters_grad_edge_weight_fused_cfconv,
    setup_context=setup_context_grad_filters_grad_edge_weight_fused_cfconv,
)


@grad_filters_grad_edge_weight_fused_cfconv.register_kernel("cpu")
def cpu_grad_filters_grad_edge_weight_fused_cfconv(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    grad_edge_out: torch.Tensor,
    cutoff_upper: float,
    grad_edge_dtype: torch.dtype = None,
) -> torch.Tensor:

    dC_dd = _torch_d_cosine_cutoff_dd(edge_weight, cutoff_upper)
    grad_filters = (
        grad_edge_out * grad_output[edge_dst] * x[edge_src]
    ) * dC_dd.unsqueeze(1)

    return grad_filters  # FIXME: check dtype


# ============================================================================
# grad_edge_weight_grad_edge_weight_fused_cfconv
# ============================================================================


@triton.jit
def grad_edge_weight_grad_edge_weight_fused_cfconv_kernel(
    # Input pointers
    x_ptr,  # [num_nodes, feature_dim]
    grad_output_ptr,  # [num_nodes, feature_dim]
    filters_ptr,  # [num_edges, feature_dim]
    edge_weight_ptr,  # [num_edges]
    edge_src_ptr,  # [num_edges]
    edge_dst_ptr,  # [num_edges]
    grad_edge_out_ptr,  # [num_edges, feature_dim] - OUTPUT (FP32 or FP16)
    # Output pointers
    grad_edge_weight_ptr,  # [num_edges]
    # Cutoff parameters
    cutoff_upper,
    # Sizes
    num_edges,
    feature_dim,
    # Block size #FIXME: check if other casting are needed
    BLOCK_F: tl.constexpr,
    filters_FP16: tl.constexpr,
    GRAD_EDGE_FP16: tl.constexpr,  # Whether to output FP16
):
    """
    Triton kernel: Compute grad_edge_weight for each edge independently.

    Grid layout: One thread block per edge
    Computation: For each edge, compute the product of gradients and features across all dimensions,
    then multiply by the second derivative of the cosine cutoff function.

    Computes:
        Computes: grad_edge_weight[e] =  (grad_edge[e] * grad_output[dst[e]] * x[src[e]] * filters[e]).sum(axis=-1) * d2_cutoff_dd2[e]

    """
    edge_idx = tl.program_id(axis=0)

    if edge_idx >= num_edges:
        return

    src_node = tl.load(edge_src_ptr + edge_idx)
    dst_node = tl.load(edge_dst_ptr + edge_idx)
    dist = tl.load(edge_weight_ptr + edge_idx)
    C = _d2_cosine_cutoff_dd2(dist, cutoff_upper)

    acc = tl.zeros_like(C)

    for f_start in range(0, feature_dim, BLOCK_F):
        f_offset = f_start + tl.arange(0, BLOCK_F)
        f_mask = f_offset < feature_dim

        grad_out = tl.load(
            grad_output_ptr + dst_node * feature_dim + f_offset,
            mask=f_mask,
            other=0.0,
        )
        x = tl.load(
            x_ptr + src_node * feature_dim + f_offset, mask=f_mask, other=0.0
        )
        filters = tl.load(
            filters_ptr + edge_idx * feature_dim + f_offset,
            mask=f_mask,
            other=0.0,
        )
        grad_edge = tl.load(
            grad_edge_out_ptr + edge_idx * feature_dim + f_offset,
            mask=f_mask,
            other=0.0,
        )
        if filters_FP16:
            filters = filters.to(tl.float32)
        if GRAD_EDGE_FP16:
            grad_edge = grad_edge.to(tl.float32)

        acc += tl.sum(grad_edge * grad_out * x * filters, axis=-1)

    tl.store(grad_edge_weight_ptr + edge_idx, acc * C)


@triton_op(
    "mlcg_kernels::grad_edge_weight_grad_edge_weight_fused_cfconv",
    mutates_args={},
)
@ensure_contiguous
def grad_edge_weight_grad_edge_weight_fused_cfconv(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    filters: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    grad_edge_out: torch.Tensor,
    cutoff_upper: float,
    grad_edge_dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Compute gradients w.r.t. edge distances/weights in CFConv second-order backward pass.

    Computes: grad_edge_weight[e] = (grad_edge[e] * grad_output[dst[e]] * x[src[e]] * filters[e]).sum(axis=-1) * d2_cutoff_dd2[e]

    This operation computes scalar gradients for each edge by summing products across features
    and multiplying by the second derivative of the cosine cutoff function. This is critical for
    optimizing edge-based operations.

    Args:
        x: Input node features [num_nodes, feature_dim]
        grad_output: Gradients of output features [num_nodes, feature_dim]
        filters: Filter outputs from forward pass [num_edges, feature_dim], FP32 or FP16
        edge_weight: Edge distances/weights [num_edges]
        edge_src: Source node indices [num_edges]
        edge_dst: Destination node indices [num_edges]
        grad_edge_out: Gradients w.r.t edge representations [num_edges, feature_dim]
        cutoff_upper: Upper bound for cosine cutoff function
        grad_edge_dtype: Data type of grad_edge_out (FP32 or FP16)

    Returns:
        Gradients w.r.t. edge distances [num_edges], dtype=float32
    """

    feature_dim = x.shape[1]
    num_edges = edge_src.shape[0]

    grad_edge_weight = torch.zeros(
        num_edges, device=x.device, dtype=torch.float32
    ).contiguous()

    if num_edges == 0:
        return grad_edge_weight

    filters_fp16 = filters.dtype == torch.float16
    grad_edge_fp16 = grad_edge_dtype == torch.float16

    BLOCK_F = 128
    grid = (num_edges,)

    wrap_triton(grad_edge_weight_grad_edge_weight_fused_cfconv_kernel)[grid](
        x,
        grad_output,
        filters,
        edge_weight,
        edge_src,
        edge_dst,
        grad_edge_out,
        grad_edge_weight,
        cutoff_upper,
        num_edges,
        feature_dim,
        BLOCK_F=BLOCK_F,
        filters_FP16=filters_fp16,
        GRAD_EDGE_FP16=grad_edge_fp16,
    )

    return grad_edge_weight


def setup_context_grad_edge_weight_grad_edge_weight_fused_cfconv(
    ctx, inputs, output
):
    (
        x,
        grad_output,
        filters,
        edge_weight,
        edge_src,
        edge_dst,
        grad_edge_out,
        cutoff_upper,
        grad_edge_dtype,
    ) = inputs

    ctx.save_for_backward(
        x,
        grad_output,
        filters,
        edge_weight,
        edge_src,
        edge_dst,
        grad_edge_out,
    )

    ctx.cutoff_upper = cutoff_upper


def backward_grad_edge_weight_grad_edge_weight_fused_cfconv(
    ctx, grad_grad_edge_weight
):
    (
        x,
        grad_output,
        filters,
        edge_weight,
        edge_src,
        edge_dst,
        grad_edge_out,
    ) = ctx.saved_tensors

    cutoff_upper = ctx.cutoff_upper

    d2C_dd2 = _torch_d2_cosine_cutoff_dd2(edge_weight, cutoff_upper)

    grad_x = grad_grad_output = grad_filters = grad_edge_weight = (
        grad_grad_edge_out
    ) = None
    if ctx.needs_input_grad[0]:
        grad_x = torch.zeros_like(x)
        expanded = (
            grad_grad_edge_weight.unsqueeze(1)
            * grad_edge_out
            * grad_output[edge_dst]
            * filters
            * d2C_dd2.unsqueeze(1)
        )
        grad_x.index_add_(0, edge_src, expanded)

    if ctx.needs_input_grad[1]:
        grad_grad_output = torch.zeros_like(grad_output)
        expanded = (
            grad_grad_edge_weight.unsqueeze(1)
            * grad_edge_out
            * x[edge_src]
            * filters
            * d2C_dd2.unsqueeze(1)
        )
        grad_grad_output.index_add_(0, edge_dst, expanded)

    if ctx.needs_input_grad[2]:
        grad_filters = (
            grad_grad_edge_weight.unsqueeze(1)
            * grad_edge_out
            * grad_output[edge_dst]
            * x[edge_src]
            * d2C_dd2.unsqueeze(1)
        )

    if ctx.needs_input_grad[3]:
        d3C_dd3 = _torch_d3_cosine_cutoff_dd3(edge_weight, cutoff_upper)
        grad_edge_weight = (
            grad_grad_edge_weight
            * (
                grad_edge_out * grad_output[edge_dst] * x[edge_src] * filters
            ).sum(dim=1)
            * d3C_dd3
        )

    if ctx.needs_input_grad[6]:
        grad_grad_edge_out = (
            grad_grad_edge_weight.unsqueeze(1)
            * grad_output[edge_dst]
            * x[edge_src]
            * filters
            * d2C_dd2.unsqueeze(1)
        )

    return (
        grad_x,
        grad_grad_output,
        grad_filters,
        grad_edge_weight,
        None,
        None,
        grad_grad_edge_out,
        None,
        None,
    )


grad_edge_weight_grad_edge_weight_fused_cfconv.register_autograd(
    backward_grad_edge_weight_grad_edge_weight_fused_cfconv,
    setup_context=setup_context_grad_edge_weight_grad_edge_weight_fused_cfconv,
)


@grad_edge_weight_grad_edge_weight_fused_cfconv.register_kernel("cpu")
def cpu_grad_edge_weight_grad_edge_weight_fused_cfconv(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    filters: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    grad_edge_out: torch.Tensor,
    cutoff_upper: float,
    grad_edge_dtype: torch.dtype = None,
) -> torch.Tensor:

    d2C_dd2 = _torch_d2_cosine_cutoff_dd2(edge_weight, cutoff_upper)
    grad_edge_weight = (
        grad_edge_out * grad_output[edge_dst] * x[edge_src] * filters
    ).sum(dim=-1) * d2C_dd2

    return grad_edge_weight  # FIXME: check dtype
