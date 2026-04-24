"""
Fused Triton kernels for double backward pass gradients in SchNet CFConv (Continuous Filter Convolution).

This module implements optimized second-order backward operations for SchNet's continuous filter convolution
layer, enabling double gradient computation needed for force matching in neural network training. It provides
fused GPU kernels (Triton) and CPU fallback implementations for computing gradients with respect to filter
outputs, edge weights, and node features.

The kernels use sparse CSR format for efficient edge aggregation and support mixed precision computation (FP16/FP32).

Main functions:
    - grad_grad_out_grad_x_fused_cfconv: Compute second derivatives w.r.t. output features
    - grad_filters_grad_x_fused_cfconv: Compute second derivatives w.r.t. filter outputs
    - grad_edge_weight_grad_x_fused_cfconv: Compute second derivatives w.r.t. edge distances

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
    _cosine_cutoff,
    _d_cosine_cutoff_dd,
    _torch_cosine_cutoff,
    _torch_d_cosine_cutoff_dd,
    _torch_d2_cosine_cutoff_dd2,
)

triton_pi = tl.constexpr(3.141592653589793)

def filter_configs(configs, named_args, **kwargs):
    feature_dim = named_args["feature_dim"]
    filtered = [
        cfg for cfg in configs
        if cfg.kwargs["BLOCK_F"] <= feature_dim
    ]
    return filtered if filtered else [min(configs, key=lambda c: c.kwargs["BLOCK_F"])]

# ============================================================================
# grad_grad_out_grad_x_fused_cfconv
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_F": 8},   num_warps=2, num_stages=2),
        triton.Config({"BLOCK_F": 16},  num_warps=2, num_stages=2),
        triton.Config({"BLOCK_F": 32},  num_warps=2, num_stages=2),
        triton.Config({"BLOCK_F": 32},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_F": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_F": 64},  num_warps=8, num_stages=3),
        triton.Config({"BLOCK_F": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_F": 128}, num_warps=8, num_stages=3),
    ],
    key=["feature_dim"],
    prune_configs_by={"early_config_prune": filter_configs},
)
@triton.jit
def grad_grad_out_grad_x_fused_cfconv_kernel(
    # Input pointers
    grad_grad_x_ptr,  # [num_nodes, feature_dim]
    filters_ptr,  # [num_edges, feature_dim] - filters outputs (FP32 or FP16)
    edge_weight_ptr,  # [num_edges]
    edge_src_ptr,  # [num_edges]
    dst_perm_ptr,  # [num_edges] - dst-CSR permutation
    dst_ptr_ptr,  # [num_nodes + 1] - dst-CSR row pointers
    # Output pointer
    grad_grad_out_ptr,
    # Cutoff parameters
    cutoff_upper,
    # Sizes
    num_nodes,
    feature_dim: tl.constexpr,
    # Block size #FIXME: check if other casting are needed
    filters_FP16: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    """
    Triton kernel: Compute grad_grad_out by aggregating over edges using destination-CSR format.

    Grid layout: One thread block per destination node
    Computation: For each node, iterate through all its incoming edges and accumulate
    second-order gradients by multiplying edge gradients, filter outputs, and cutoff values.

    Computes:
        Computes: grad_grad_out[dst] =  sum_{e: dst[e]=dst} grad_grad_x[src[e]] * filters[e] * cutoff[e]

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

            grad_grad_x = tl.load(
                grad_grad_x_ptr + src_node * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )
            filters = tl.load(
                filters_ptr + edge_idx * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )
            distances = tl.load(edge_weight_ptr + edge_idx)
            C = _cosine_cutoff(distances, cutoff_upper)

            if filters_FP16:
                filters = filters.to(tl.float32)

            acc += (
                grad_grad_x * filters * C  # [:, None]
            )  # FIXME: check this broadcast, maybe just C

        tl.store(
            grad_grad_out_ptr + target_node * feature_dim + f_offsets,
            acc,
            mask=f_mask,
        )


@triton_op(
    "mlcg_kernels::grad_grad_out_grad_x_fused_cfconv",
    mutates_args={},
)
@ensure_contiguous
def grad_grad_out_grad_x_fused_cfconv(
    filters: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    grad_grad_x: torch.Tensor,
    dst_perm: torch.Tensor,
    dst_ptr: torch.Tensor,
    cutoff_upper: float,
) -> torch.Tensor:
    """
    Compute second derivatives w.r.t. output features in CFConv double backward pass.

    Computes: grad_grad_out[dst] = sum_{e: dst[e]=dst} grad_grad_x[src[e]] * filters[e] * cutoff[e]

    This operation aggregates second-order gradients from all edges contributing to each destination node,
    weighted by filter outputs and the cosine cutoff function.

    Args:
        filters: Filter outputs from forward pass [num_edges, feature_dim], FP32 or FP16
        edge_weight: Edge distances/weights [num_edges]
        edge_src: Source node indices [num_edges]
        edge_dst: Destination node indices [num_edges]
        grad_grad_x: Second-order gradients w.r.t. source features [num_nodes, feature_dim]
        dst_perm: Permutation array for destination CSR format [num_edges]
        dst_ptr: Row pointers for destination CSR format [num_nodes + 1]
        cutoff_upper: Upper bound for cosine cutoff function

    Returns:
        Second-order gradients w.r.t. output features [num_nodes, feature_dim]
    """
    num_nodes = grad_grad_x.shape[0]
    num_edges = edge_src.shape[0]
    feature_dim = grad_grad_x.shape[1]

    grad_grad_out = torch.zeros(
        num_nodes,
        feature_dim,
        device=grad_grad_x.device,
        dtype=grad_grad_x.dtype,
    ).contiguous()  # FIXME: check dtype

    if num_edges == 0:
        return grad_grad_out

    filters_fp16 = filters.dtype == torch.float16
    grid = (num_nodes,)

    wrap_triton(grad_grad_out_grad_x_fused_cfconv_kernel)[grid](
        grad_grad_x,
        filters,
        edge_weight,
        edge_src,
        dst_perm,
        dst_ptr,
        grad_grad_out,
        cutoff_upper,
        num_nodes,
        feature_dim,
        filters_FP16=filters_fp16,
    )

    return grad_grad_out


def setup_context_grad_grad_out_grad_x_fused_cfconv(ctx, inputs, output):
    (
        filters,
        edge_weight,
        edge_src,
        edge_dst,
        grad_grad_x,
        dst_perm,
        dst_ptr,
        cutoff_upper,
    ) = inputs

    ctx.save_for_backward(
        filters,
        edge_weight,
        edge_src,
        edge_dst,
        grad_grad_x,
    )
    ctx.cutoff_upper = cutoff_upper


def backward_grad_grad_out_grad_x_fused_cfconv(ctx, grad_grad_grad_out):
    (
        filters,
        edge_weight,
        edge_src,
        edge_dst,
        grad_grad_x,
    ) = ctx.saved_tensors

    cutoff_upper = ctx.cutoff_upper

    grad_filters = grad_edge_weight = grad_grad_grad_x = None

    C = _torch_cosine_cutoff(edge_weight, cutoff_upper)

    if ctx.needs_input_grad[0]:
        grad_filters = (
            grad_grad_grad_out[edge_dst]
            * grad_grad_x[edge_src]
            * C.unsqueeze(1)
        )

    if ctx.needs_input_grad[1]:
        dC_dd = _torch_d_cosine_cutoff_dd(edge_weight, cutoff_upper)
        grad_edge_weight = (
            grad_grad_grad_out[edge_dst] * grad_grad_x[edge_src] * filters
        ).sum(dim=1) * dC_dd

    if ctx.needs_input_grad[4]:
        grad_grad_grad_x = torch.zeros_like(grad_grad_x)
        expanded = grad_grad_grad_out[edge_src] * filters * C.unsqueeze(1)
        grad_grad_grad_x.index_add_(0, edge_src, expanded)

    return (
        grad_filters,
        grad_edge_weight,
        None,
        None,
        grad_grad_grad_x,
        None,
        None,
        None,
    )


grad_grad_out_grad_x_fused_cfconv.register_autograd(
    backward_grad_grad_out_grad_x_fused_cfconv,
    setup_context=setup_context_grad_grad_out_grad_x_fused_cfconv,
)


@grad_grad_out_grad_x_fused_cfconv.register_kernel("cpu")
def cpu_grad_grad_out_grad_x_fused_cfconv(
    filters: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    grad_grad_x: torch.Tensor,
    dst_perm: torch.Tensor,
    dst_ptr: torch.Tensor,
    cutoff_upper: float,
) -> torch.Tensor:

    C = _torch_cosine_cutoff(edge_weight, cutoff_upper)
    grad_grad_out = torch.zeros_like(grad_grad_x)
    expanded = grad_grad_x[edge_src] * filters * C.unsqueeze(1)
    grad_grad_out.index_add_(0, edge_dst, expanded)

    return grad_grad_out  # FIXME: check dtype


# ============================================================================
# grad_filters_grad_x_fused_cfconv
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_F": 8},   num_warps=2, num_stages=2),
        triton.Config({"BLOCK_F": 16},  num_warps=2, num_stages=2),
        triton.Config({"BLOCK_F": 32},  num_warps=2, num_stages=2),
        triton.Config({"BLOCK_F": 32},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_F": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_F": 64},  num_warps=8, num_stages=3),
        triton.Config({"BLOCK_F": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_F": 128}, num_warps=8, num_stages=3),
    ],
    key=["feature_dim"],
    prune_configs_by={"early_config_prune": filter_configs},
)
@triton.jit
def grad_filters_grad_x_fused_cfconv_kernel(
    # Input pointers
    grad_grad_x_ptr,  # [num_nodes, feature_dim]
    grad_out_ptr,  # [num_nodes, feature_dim]
    edge_weight_ptr,  # [num_edges]
    edge_src_ptr,  # [num_edges]
    edge_dst_ptr,  # [num_edges]
    # Output pointer
    grad_filters_ptr,
    # Cutoff parameters
    cutoff_upper,
    # Sizes
    num_edges,
    feature_dim: tl.constexpr,
    # Block size
    BLOCK_F: tl.constexpr,
):
    """
    Triton kernel: Compute grad_filters for each edge independently.

    Grid layout: One thread block per edge
    Computation: For each edge, multiply second-order source gradients with first-order destination gradients
    weighted by the cosine cutoff function. No aggregation needed.

    Computes:
        Computes: grad_filters[e] =  grad_grad_x[src[e]] * grad_out[dst[e]] * cutoff[e]

    """
    edge_idx = tl.program_id(axis=0)

    if edge_idx >= num_edges:
        return

    src_node = tl.load(edge_src_ptr + edge_idx)
    dst_node = tl.load(edge_dst_ptr + edge_idx)

    dist = tl.load(edge_weight_ptr + edge_idx)
    C = _cosine_cutoff(dist, cutoff_upper)

    for f_start in range(0, feature_dim, BLOCK_F):
        f_offset = f_start + tl.arange(0, BLOCK_F)
        f_mask = f_offset < feature_dim

        grad_grad_x = tl.load(
            grad_grad_x_ptr + src_node * feature_dim + f_offset,
            mask=f_mask,
            other=0.0,
        )
        grad_out = tl.load(
            grad_out_ptr + dst_node * feature_dim + f_offset,
            mask=f_mask,
            other=0.0,
        )

        tl.store(
            grad_filters_ptr + edge_idx * feature_dim + f_offset,
            grad_grad_x
            * grad_out
            * C,  # [:, None],  # FIXME: check this broadcast, maybe just C
            mask=f_mask,
        )


@triton_op("mlcg_kernels::grad_filters_grad_x_fused_cfconv", mutates_args={})
@ensure_contiguous
def grad_filters_grad_x_fused_cfconv(
    grad_out: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    grad_grad_x: torch.Tensor,
    cutoff_upper: float,
) -> torch.Tensor:
    """
    Compute second derivatives w.r.t. filter outputs in CFConv double backward pass.

    Computes: grad_filters[e] = grad_grad_x[src[e]] * grad_out[dst[e]] * cutoff[e]

    This operation computes second-order gradients for each edge directly by multiplying
    the second-order gradients from the source with first-order gradients from the destination,
    weighted by the cosine cutoff function.

    Args:
        grad_out: First-order output gradients [num_nodes, feature_dim]
        edge_weight: Edge distances/weights [num_edges]
        edge_src: Source node indices [num_edges]
        edge_dst: Destination node indices [num_edges]
        grad_grad_x: Second-order gradients w.r.t. source features [num_nodes, feature_dim]
        cutoff_upper: Upper bound for cosine cutoff function

    Returns:
        Second-order gradients w.r.t. filter outputs [num_edges, feature_dim]
    """
    num_edges = edge_src.shape[0]
    feature_dim = grad_out.shape[1]

    grad_filters = torch.zeros(
        num_edges, feature_dim, device=grad_out.device, dtype=grad_out.dtype
    ).contiguous()  # FIXME: check if dtype is correct

    if num_edges == 0:
        return grad_filters

    grid = (num_edges,)

    wrap_triton(grad_filters_grad_x_fused_cfconv_kernel)[grid](
        grad_grad_x,
        grad_out,
        edge_weight,
        edge_src,
        edge_dst,
        grad_filters,
        cutoff_upper,
        num_edges,
        feature_dim,
    )

    return grad_filters


def setup_context_grad_filters_grad_x_fused_cfconv(ctx, inputs, output):
    (
        grad_out,
        edge_weight,
        edge_src,
        edge_dst,
        grad_grad_x,
        cutoff_upper,
    ) = inputs

    ctx.save_for_backward(
        grad_out,
        edge_weight,
        edge_src,
        edge_dst,
        grad_grad_x,
    )
    ctx.cutoff_upper = cutoff_upper


def backward_grad_filters_grad_x_fused_cfconv(ctx, grad_grad_filters):
    (
        grad_out,
        edge_weight,
        edge_src,
        edge_dst,
        grad_grad_x,
    ) = ctx.saved_tensors

    cutoff_upper = ctx.cutoff_upper

    grad_grad_out = grad_edge_weight = grad_grad_grad_x = None

    C = _torch_cosine_cutoff(edge_weight, cutoff_upper)

    if ctx.needs_input_grad[0]:
        grad_grad_out = torch.zeros_like(grad_out)
        expanded = grad_grad_filters * grad_grad_x[edge_src] * C.unsqueeze(1)
        grad_grad_out.index_add_(0, edge_dst, expanded)

    if ctx.needs_input_grad[1]:
        dC_dd = _torch_d_cosine_cutoff_dd(edge_weight, cutoff_upper)
        grad_edge_weight = (
            grad_grad_filters * grad_grad_x[edge_src] * grad_out[edge_dst]
        ).sum(dim=1) * dC_dd

    if ctx.needs_input_grad[4]:
        grad_grad_grad_x = torch.zeros_like(grad_grad_x)
        expanded = grad_grad_filters * grad_out[edge_dst] * C.unsqueeze(1)
        grad_grad_grad_x.index_add_(0, edge_src, expanded)

    return (
        grad_grad_out,
        grad_edge_weight,
        None,
        None,
        grad_grad_grad_x,
        None,
    )


grad_filters_grad_x_fused_cfconv.register_autograd(
    backward_grad_filters_grad_x_fused_cfconv,
    setup_context=setup_context_grad_filters_grad_x_fused_cfconv,
)


@grad_filters_grad_x_fused_cfconv.register_kernel("cpu")
def cpu_grad_filters_grad_x_fused_cfconv(
    grad_out: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    grad_grad_x: torch.Tensor,
    cutoff_upper: float,
) -> torch.Tensor:

    C = _torch_cosine_cutoff(edge_weight, cutoff_upper)
    grad_filters = grad_grad_x[edge_src] * grad_out[edge_dst] * C.unsqueeze(1)

    return grad_filters  # FIXME: check dtype


# ============================================================================
# grad_edge_weight_grad_x_fused_cfconv
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_F": 8},   num_warps=2, num_stages=2),
        triton.Config({"BLOCK_F": 16},  num_warps=2, num_stages=2),
        triton.Config({"BLOCK_F": 32},  num_warps=2, num_stages=2),
        triton.Config({"BLOCK_F": 32},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_F": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_F": 64},  num_warps=8, num_stages=3),
        triton.Config({"BLOCK_F": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_F": 128}, num_warps=8, num_stages=3),
    ],
    key=["feature_dim"],
    prune_configs_by={"early_config_prune": filter_configs},
)
@triton.jit
def grad_edge_weight_grad_x_fused_cfconv_kernel(
    # Input pointers
    grad_grad_x_ptr,  # [num_nodes, feature_dim]
    grad_out_ptr,  # [num_nodes, feature_dim]
    filters_ptr,  # [num_edges, feature_dim]
    edge_weight_ptr,  # [num_edges]
    edge_src_ptr,  # [num_edges]
    edge_dst_ptr,  # [num_edges]
    # Output pointer
    grad_edge_weight_ptr,
    # Cutoff parameters
    cutoff_upper,
    # Sizes
    num_edges,
    feature_dim: tl.constexpr,
    # Block size
    filters_FP16: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    """
    Triton kernel: Compute grad_edge_weight for each edge independently.

    Grid layout: One thread block per edge
    Computation: For each edge, accumulate products of gradients and features across all dimensions,
    then multiply by the first derivative of the cosine cutoff function.

    Computes:
        Computes: grad_edge_weight[e] =  (grad_grad_x[src[e]] * grad_out[dst[e]] * filters[e]).sum(axs=-1) * d_cutoff_dd[e]

    """
    edge_idx = tl.program_id(axis=0)
    if edge_idx >= num_edges:
        return

    src_node = tl.load(edge_src_ptr + edge_idx)
    dst_node = tl.load(edge_dst_ptr + edge_idx)

    dist = tl.load(edge_weight_ptr + edge_idx)
    dC = _d_cosine_cutoff_dd(dist, cutoff_upper)

    acc = tl.zeros_like(dC)

    for f_start in range(0, feature_dim, BLOCK_F):
        f_offset = f_start + tl.arange(0, BLOCK_F)
        f_mask = f_offset < feature_dim

        grad_grad_x = tl.load(
            grad_grad_x_ptr + src_node * feature_dim + f_offset,
            mask=f_mask,
            other=0.0,
        )
        grad_out = tl.load(
            grad_out_ptr + dst_node * feature_dim + f_offset,
            mask=f_mask,
            other=0.0,
        )
        filters = tl.load(
            filters_ptr + edge_idx * feature_dim + f_offset,
            mask=f_mask,
            other=0.0,
        )

        if filters_FP16:
            filters = filters.to(tl.float32)

        acc += tl.sum(grad_grad_x * grad_out * filters, axis=-1)

    tl.store(grad_edge_weight_ptr + edge_idx, acc * dC)


@triton_op(
    "mlcg_kernels::grad_edge_weight_grad_x_fused_cfconv", mutates_args={}
)
@ensure_contiguous
def grad_edge_weight_grad_x_fused_cfconv(
    grad_out: torch.Tensor,
    filters: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    grad_grad_x: torch.Tensor,
    cutoff_upper: float,
) -> torch.Tensor:
    """
    Compute second derivatives w.r.t. edge distances in CFConv double backward pass.

    Computes: grad_edge_weight[e] = (grad_grad_x[src[e]] * grad_out[dst[e]] * filters[e]).sum(axis=-1) * d_cutoff_dd[e]

    This operation computes scalar second-order gradients for each edge by summing products across features
    and multiplying by the first derivative of the cosine cutoff function. This enables gradient flow
    for optimizing edge-specific properties in the graph.

    Args:
        grad_out: First-order output gradients [num_nodes, feature_dim]
        filters: Filter outputs from forward pass [num_edges, feature_dim], FP32 or FP16
        edge_weight: Edge distances/weights [num_edges]
        edge_src: Source node indices [num_edges]
        edge_dst: Destination node indices [num_edges]
        grad_grad_x: Second-order gradients w.r.t. source features [num_nodes, feature_dim]
        cutoff_upper: Upper bound for cosine cutoff function

    Returns:
        Second-order gradients w.r.t. edge distances [num_edges]
    """

    num_edges = edge_src.shape[0]
    feature_dim = grad_out.shape[1]

    grad_edge_weight = torch.zeros(
        num_edges, device=grad_out.device, dtype=grad_out.dtype
    ).contiguous()

    if num_edges == 0:
        return grad_edge_weight

    filters_fp16 = filters.dtype == torch.float16

    grid = (num_edges,)

    wrap_triton(grad_edge_weight_grad_x_fused_cfconv_kernel)[grid](
        grad_grad_x,
        grad_out,
        filters,
        edge_weight,
        edge_src,
        edge_dst,
        grad_edge_weight,
        cutoff_upper,
        num_edges,
        feature_dim,
        filters_FP16=filters_fp16,
    )

    return grad_edge_weight


def setup_context_grad_edge_weight_grad_x_fused_cfconv(ctx, inputs, output):
    (
        grad_out,
        filters,
        edge_weight,
        edge_src,
        edge_dst,
        grad_grad_x,
        cutoff_upper,
    ) = inputs

    ctx.save_for_backward(
        grad_out, filters, edge_weight, edge_src, edge_dst, grad_grad_x
    )

    ctx.cutoff_upper = cutoff_upper


def backward_grad_edge_weight_grad_x_fused_cfconv(ctx, grad_grad_edge_weight):
    (
        grad_out,
        filters,
        edge_weight,
        edge_src,
        edge_dst,
        grad_grad_x,
    ) = ctx.saved_tensors

    cutoff_upper = ctx.cutoff_upper

    grad_grad_out = grad_filters = grad_edge_weight = grad_grad_grad_x = None

    dC_dd = _torch_d_cosine_cutoff_dd(edge_weight, cutoff_upper)

    if ctx.needs_input_grad[0]:
        grad_grad_out = torch.zeros_like(grad_out)
        expanded = (
            grad_grad_edge_weight.unsqueeze(1)
            * grad_grad_x[edge_src]
            * filters
            * dC_dd.unsqueeze(1)
        )
        grad_grad_out.index_add_(0, edge_dst, expanded)

    if ctx.needs_input_grad[1]:
        grad_filters = (
            grad_grad_edge_weight.unsqueeze(1)
            * grad_grad_x[edge_src]
            * grad_out[edge_dst]
            * dC_dd.unsqueeze(1)
        )

    if ctx.needs_input_grad[2]:
        d2C_dd2 = _torch_d2_cosine_cutoff_dd2(edge_weight, cutoff_upper)
        grad_edge_weight = (
            grad_grad_edge_weight
            * (grad_grad_x[edge_src] * grad_out[edge_dst] * filters).sum(dim=1)
            * d2C_dd2
        )

    if ctx.needs_input_grad[5]:
        grad_grad_grad_x = torch.zeros_like(grad_grad_x)
        expanded = (
            grad_edge_weight.unsqueeze(1)
            * grad_out[edge_dst]
            * filters
            * dC_dd.unsqueeze(1)
        )
        grad_grad_grad_x.index_add_(0, edge_src, expanded)

    return (
        grad_grad_out,
        grad_filters,
        grad_edge_weight,
        None,
        None,
        grad_grad_grad_x,
        None,
    )


grad_edge_weight_grad_x_fused_cfconv.register_autograd(
    backward_grad_edge_weight_grad_x_fused_cfconv,
    setup_context=setup_context_grad_edge_weight_grad_x_fused_cfconv,
)


@grad_edge_weight_grad_x_fused_cfconv.register_kernel("cpu")
def cpu_grad_edge_weight_grad_x_fused_cfconv(
    grad_out: torch.Tensor,
    filters: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    grad_grad_x: torch.Tensor,
    cutoff_upper: float,
) -> torch.Tensor:

    dC_dd = _torch_d_cosine_cutoff_dd(edge_weight, cutoff_upper)
    grad_edge_weight = (
        grad_grad_x[edge_src] * grad_out[edge_dst] * filters
    ).sum(dim=1) * dC_dd

    return grad_edge_weight  # FIXME: check dtype
