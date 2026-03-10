"""
Backward kernels meant to be used for kernels in cfconv
"""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from ...utils import ensure_contiguous
from ..cutoffs import _cosine_cutoff, _d_cosine_cutoff_dd
from .cfconv_double_backwards import (
    grad_x_grad_edge_weight_fused_cfconv,
    grad_grad_out_grad_edge_weight_fused_cfconv,
    grad_filters_grad_edge_weight_fused_cfconv,
    grad_edge_weight_grad_edge_weight_fused_cfconv,
)

triton_pi = tl.constexpr(3.141592653589793)

# ============================================================================
# Fused Backward Kernel for grad_filters
# ============================================================================


@triton.jit
def grad_filters_fused_cfconv_kernel(
    # Input pointers
    x_ptr,  # [num_nodes, feature_dim]
    grad_output_ptr,  # [num_nodes, feature_dim]
    edge_weight_ptr,  # [num_edges]
    edge_src_ptr,  # [num_edges]
    edge_dst_ptr,  # [num_edges]
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
        grad_filters[e] = x[src[e]] * grad_output[dst[e]] * cutoff(dist[e])

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
    C = _cosine_cutoff(dist, cutoff_upper)

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

        # Fused multiply: x * grad * C (in FP32)
        grad_filters = x_j * grad_j * C

        # Store result (convert to FP16 if needed)
        if OUTPUT_FP16:
            grad_filters = grad_filters.to(tl.float16)
        tl.store(
            grad_filters_ptr + edge_idx * feature_dim + f_offsets,
            grad_filters,
            mask=f_mask,
        )


@triton_op("mlcg_kernels::grad_filters_fused_cfconv", mutates_args={})
@ensure_contiguous
def grad_filters_fused_cfconv(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    cutoff_upper: float,
    out_dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Compute grad_filters in a single fused kernel.

    grad_filters[e] = x[src[e]] * grad_output[dst[e]] * cutoff(dist[e])

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
    )

    if num_edges == 0:
        return grad_filters

    BLOCK_F = min(128, triton.next_power_of_2(feature_dim))
    grid = (num_edges,)

    # Determine if output should be FP16
    output_fp16 = out_dtype == torch.float16

    wrap_triton(grad_filters_fused_cfconv_kernel)[grid](
        x,
        grad_output,
        edge_weight,
        edge_src,
        edge_dst,
        grad_filters,
        cutoff_upper,
        num_edges,
        feature_dim,
        BLOCK_F=BLOCK_F,
        OUTPUT_FP16=output_fp16,
    )

    return grad_filters


def setup_context_grad_filters_fused_cfconv(ctx, inputs, output):
    raise NotImplementedError

def backward_grad_filters_fused_cfconv(ctx, grad_output):
    raise NotImplementedError

grad_filters_fused_cfconv.register_autograd(
    backward_grad_filters_fused_cfconv, setup_context=setup_context_grad_filters_fused_cfconv
)


@grad_filters_fused_cfconv.register_kernel("cpu")
def grad_filters_fused_cfconv(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    cutoff_upper: float,
    out_dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    CPU fallback for fused_grad_filters
    """
    C = 0.5 * (torch.cos(edge_weight * torch.pi / cutoff_upper) + 1)
    C = C * (edge_weight < cutoff_upper).float()
    grad_filters = grad_output[edge_dst] * x[edge_src] * C.unsqueeze(-1)

    return grad_filters


# ============================================================================
# Fused Backward Kernel for src_csr_grad_x
# ============================================================================


@triton.jit
def grad_x_fused_cfconv_kernel(
    # Input pointers
    grad_output_ptr,  # [num_nodes, feature_dim] - gradient from output (FP32)
    filters_ptr,  # [num_edges, feature_dim] - filters outputs (FP32 or FP16)
    edge_weight_ptr,  # [num_edges] - distances for cutoff
    edge_dst_ptr,  # [num_edges] - destination indices (original order)
    src_perm_ptr,  # [num_edges] - src-CSR permutation
    src_ptr_ptr,  # [num_nodes + 1] - src-CSR row pointers
    # Output pointer
    grad_x_ptr,  # [num_nodes, feature_dim] - gradient w.r.t. x (FP32)
    # Parameters
    cutoff_upper,
    num_nodes,
    feature_dim,
    # Block sizes
    BLOCK_F: tl.constexpr,
    filters_FP16: tl.constexpr,  # Whether filters is FP16
):
    """
    Fused src-CSR grad_x kernel for CFConv backward pass.

    Computes: grad_x[src] = sum_{e: src[e]=src} grad_output[dst[e]] * filters[e] * cutoff[e]

    Key features:
    - One block per SOURCE node (no atomics!)
    - 4 warps covering 128 features (BLOCK_F=128)
    - FP32 accumulation in registers
    - Single store to grad_x per source node
    - Supports FP16 filters (loads FP16, promotes to FP32)

    Grid: (num_nodes,) - one block per source node
    """
    src_node = tl.program_id(0)

    if src_node >= num_nodes:
        return

    # Get segment bounds from src-CSR row pointers
    seg_start = tl.load(src_ptr_ptr + src_node)
    seg_end = tl.load(src_ptr_ptr + src_node + 1)

    # Process features in blocks
    for f_start in range(0, feature_dim, BLOCK_F):
        f_offsets = f_start + tl.arange(0, BLOCK_F)
        f_mask = f_offsets < feature_dim

        # Accumulate in FP32 registers (no atomics!)
        acc = tl.zeros([BLOCK_F], dtype=tl.float32)

        # Loop over all edges originating from this source node
        for e_csr in range(seg_start, seg_end):
            # Get original edge index via src-CSR permutation
            edge_idx = tl.load(src_perm_ptr + e_csr)

            # Load destination node index
            dst_node = tl.load(edge_dst_ptr + edge_idx)

            # Load distance and compute cutoff
            dist = tl.load(edge_weight_ptr + edge_idx)
            C = _cosine_cutoff(dist, cutoff_upper)

            # Load filters output (FP16 or FP32)
            filters_val = tl.load(
                filters_ptr + edge_idx * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )
            # Promote FP16 to FP32 for computation
            if filters_FP16:
                filters_val = filters_val.to(tl.float32)

            # Apply cutoff: W = filters * cutoff
            W = filters_val * C

            # Gather grad_output[dst]
            grad_dst = tl.load(
                grad_output_ptr + dst_node * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )

            # Accumulate: grad_x[src] += grad_output[dst] * W
            acc += grad_dst * W

        # Single store per source node - no atomic needed!
        tl.store(
            grad_x_ptr + src_node * feature_dim + f_offsets,
            acc,
            mask=f_mask,
        )


@triton_op("mlcg_kernels::grad_x_fused_cfconv", mutates_args={})
@ensure_contiguous
def grad_x_fused_cfconv(
    grad_output: torch.Tensor,
    filters: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    src_ptr: torch.Tensor,
    src_perm: torch.Tensor,
    num_nodes: int,
    cutoff_upper: float,
) -> torch.Tensor:
    """
    Compute grad_x using src-CSR segment reduce (no atomics).

    grad_x[src] = sum_{e: src[e]=src} grad_output[dst[e]] * filters[e] * cutoff[e]

    This replaces the atomic scatter used in the backward pass for grad_x
    with a more efficient segment-reduce that has no atomics.

    Parameters
    ----------
    grad_output : torch.Tensor
        Gradient from output [num_nodes, feature_dim], FP32
    filters : torch.Tensor
        filters network output [num_edges, feature_dim], FP32 or FP16
    edge_weight : torch.Tensor
        Edge weights (distances) [num_edges]
    edge_src: torch.Tensor
        Source node indices [num_edges]
    edge_dst : torch.Tensor
        Destination node indices [num_edges]
    src_ptr : torch.Tensor
        Src-CSR row pointers [num_nodes + 1]
    src_perm : torch.Tensor
        Src-CSR permutation [num_edges]
    num_nodes : int
        Number of nodes
    cutoff_upper : float
        Upper cutoff distance

    Returns
    -------
    torch.Tensor
        grad_x [num_nodes, feature_dim] in FP32
    """
    feature_dim = grad_output.shape[1]

    # Allocate output (zeros for nodes with no outgoing edges)
    grad_x = torch.zeros(
        num_nodes, feature_dim, device=grad_output.device, dtype=torch.float32
    ).contiguous()

    num_edges = edge_dst.shape[0]
    if num_edges == 0:
        return grad_x

    # Block size covers all 128 features (4 warps x 32 threads)
    BLOCK_F = 128

    # Auto-detect filters dtype
    filters_fp16 = filters.dtype == torch.float16

    # One block per source node
    grid = (num_nodes,)

    wrap_triton(grad_x_fused_cfconv_kernel)[grid](
        grad_output,
        filters,
        edge_weight,
        edge_dst,
        src_perm,
        src_ptr,
        grad_x,
        cutoff_upper,
        num_nodes,
        feature_dim,
        BLOCK_F=BLOCK_F,
        filters_FP16=filters_fp16,
        num_warps=4,
    )

    return grad_x


def setup_context_grad_x_fused_cfconv(ctx, inputs, output):
    raise NotImplementedError

def backward_grad_x_fused_cfconv(ctx, grad_output):
    raise NotImplementedError

grad_x_fused_cfconv.register_autograd(
    backward_grad_x_fused_cfconv, setup_context=setup_context_grad_x_fused_cfconv
)


@grad_x_fused_cfconv.register_kernel("cpu")
def cpu_grad_x_fused_cfconv(
    grad_output: torch.Tensor,
    filters: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    src_ptr: torch.Tensor,
    src_perm: torch.Tensor,
    num_nodes: int,
    cutoff_upper: float,
) -> torch.Tensor:
    """
    CPU fallback for fused_src_csr_grad_x
    """
    C = 0.5 * (torch.cos(edge_weight * torch.pi / cutoff_upper) + 1)
    C = C * (edge_weight < cutoff_upper).float()
    grad_x = torch.zeros_like(grad_output)
    grad_x.index_add_(
        0, edge_src, grad_output[edge_dst] * filters * C.unsqueeze(-1)
    )
    return grad_x


@triton.jit
def grad_edge_weight_fused_cfconv_kernel(
    # Input pointers
    x_ptr,  # [num_nodes, feature_dim]
    grad_output_ptr,  # [num_nodes, feature_dim]
    filters_ptr,  # [num_edges, feature_dim] - filters outputs (FP32 or FP16)
    edge_weight_ptr,  # [num_edges]
    edge_src_ptr,  # [num_edges]
    edge_dst_ptr,  # [num_edges]
    grad_edge_out_ptr,  # [num_edges, feature_dim] - OUTPUT (FP32 or FP16)
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
        grad_filters[e] = grad_output[dst[e]] * x[src[e]] * filters[e] * dcutoff_ddist(dist[e])

    Fuses:
        1. Gather x[edge_src]
        2. Gather grad_output[edge_dst]
        3. Gather filters
        4. Cutoff derivative computation
        5. Elementwise multiply

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
    distances = tl.load(edge_weight_ptr + edge_idx)

    # d(cutoff)/d(dist) = -0.5 * pi/cutoff_upper * sin(dist * pi / cutoff_upper)
    d_cutoff_d_dist = _d_cosine_cutoff_dd(distances, cutoff_upper)

    # Process features in blocks
    acc = tl.zeros_like(d_cutoff_d_dist)

    for f_start in range(0, feature_dim, BLOCK_F):
        f_offsets = f_start + tl.arange(0, BLOCK_F)
        f_mask = f_offsets < feature_dim

        # Gather x[src]
        x_j = tl.load(
            x_ptr + src_node * feature_dim + f_offsets, mask=f_mask, other=0.0
        )

        # Gather filters output
        filters_val = tl.load(
            filters_ptr + edge_idx * feature_dim + f_offsets,
            mask=f_mask,
            other=0.0,
        )

        # Gather grad_output[dst]
        grad_j = tl.load(
            grad_output_ptr + dst_node * feature_dim + f_offsets,
            mask=f_mask,
            other=0.0,
        )

        # Fused multiply: x * grad * C (in FP32)
        # grad_edge = x_j * filters_val * grad_j * d_cutoff_d_dist
        acc += tl.sum(x_j * filters_val * grad_j, axis=-1)

    grad_edge = acc * d_cutoff_d_dist
    # Store result (convert to FP16 if needed)
    if OUTPUT_FP16:
        grad_edge = grad_edge.to(tl.float16)

    tl.store(
        grad_edge_out_ptr + edge_idx,
        grad_edge,
        # mask=f_mask,
    )


@triton_op("mlcg_kernels::grad_edge_weight_fused_cfconv", mutates_args={})
@ensure_contiguous
def grad_edge_weight_fused_cfconv(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    filters: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    cutoff_upper: float,
    src_ptr: torch.Tensor,
    src_perm: torch.Tensor,
    dst_ptr: torch.Tensor,
    dst_perm: torch.Tensor,
    out_dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Compute grad_filters in a single fused kernel.

    fused_grad_edge_weight[e] = x[src[e]] * grad_output[dst[e]] * cutoff(dist[e])

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

    grad_edge_out = torch.empty(num_edges, device=x.device, dtype=out_dtype)

    if num_edges == 0:
        return grad_edge_out

    BLOCK_F = min(128, triton.next_power_of_2(feature_dim))
    grid = (num_edges,)

    # Determine if output should be FP16
    output_fp16 = out_dtype == torch.float16
    wrap_triton(grad_edge_weight_fused_cfconv)[grid](
        x,
        grad_output,
        filters,
        edge_weight,
        edge_src,
        edge_dst,
        grad_edge_out,
        cutoff_upper,
        num_edges,
        feature_dim,
        BLOCK_F=BLOCK_F,
        OUTPUT_FP16=output_fp16,
    )

    return grad_edge_out


def setup_context_grad_edge_weight_fused_cfconv(ctx, inputs, output):
    (
        x,
        grad_output,
        filters,
        edge_weight,
        edge_src,
        edge_dst,
        cutoff_upper,
        src_ptr,
        src_perm,
        dst_ptr,
        dst_perm,
        out_dtype,
    ) = inputs

    ctx.save_for_backward(
        x,
        grad_output,
        filters,
        edge_weight,
        edge_src,
        edge_dst,
        src_ptr,
        src_perm,
        dst_ptr,
        dst_perm,
    )
    ctx.cutoff_upper = cutoff_upper
    ctx.out_dtype = out_dtype

def backward_grad_edge_weight_fused_cfconv(ctx, grad_grad_edge_out):

    (
        x,
        grad_output,
        filters,
        edge_weight,
        edge_src,
        edge_dst,
        src_ptr,
        src_perm,
        dst_ptr,
        dst_perm,
    ) = ctx.saved_tensors

    cutoff_upper = ctx.cutoff_upper
    grad_edge_dtype = ctx.out_dtype
    grad_x = grad_grad_output = grad_filters = grad_edge_weight = None
    
    if ctx.needs_input_grad[0]:
        grad_x = grad_x_grad_edge_weight_fused_cfconv(
            grad_output,
            filters,
            edge_weight,
            edge_dst,
            grad_grad_edge_out,
            src_perm,
            src_ptr,
            cutoff_upper,
            grad_edge_dtype,
        )
    
    if ctx.needs_input_grad[1]:
        grad_grad_out = grad_grad_out_grad_edge_weight_fused_cfconv(
            x,
            filters,
            edge_weight,
            edge_src,
            grad_grad_edge_out,
            dst_perm,
            dst_ptr,
            cutoff_upper,
            grad_edge_dtype,
        )

    if ctx.needs_input_grad[2]:
        grad_filters = grad_filters_grad_edge_weight_fused_cfconv(
            x,
            grad_output,
            edge_weight,
            edge_src,
            edge_dst,
            grad_grad_edge_out,
            cutoff_upper,
            grad_edge_dtype,
        )

    if ctx.needs_input_grad[3]:
        grad_edge_weight = grad_edge_weight_grad_edge_weight_fused_cfconv(
            x,
            grad_output,
            filters,
            edge_weight,
            edge_src,
            edge_dst,
            grad_grad_edge_out,
            cutoff_upper,
            grad_edge_dtype,
        )

    return grad_x, grad_grad_output, grad_filters, grad_edge_weight, None, None, None, None

grad_edge_weight_fused_cfconv.register_autograd(
    backward_grad_edge_weight_fused_cfconv, setup_context=setup_context_grad_edge_weight_fused_cfconv
)


@grad_edge_weight_fused_cfconv.register_kernel("cpu")
def cpu_fused_grad_edge_weight(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    filters: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    cutoff_upper: float,
    src_ptr: torch.Tensor,
    src_perm: torch.Tensor,
    dst_ptr: torch.Tensor,
    dst_perm: torch.Tensor,
    out_dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    CPU fallback for fused_grad_filters
    """
    dC_dd = (
        -0.5
        * torch.sin(edge_weight * torch.pi / cutoff_upper)
        * torch.pi
        / cutoff_upper
    )
    dC_dd = dC_dd * (edge_weight < cutoff_upper).float()

    grad_edge_weight = (grad_output[edge_dst] * x[edge_src] * filters).sum(
        -1
    ) * dC_dd

    return grad_edge_weight
