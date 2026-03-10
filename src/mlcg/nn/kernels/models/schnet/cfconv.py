"""
Fused Triton kernels for CFConv operations adopting csr representation
for more efficient scatter operations.
"""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton
from .cfconv_backwards import (
    grad_x_fused_cfconv,
    grad_filters_fused_cfconv,
    grad_edge_weight_fused_cfconv,
)

from ..cutoffs import _cosine_cutoff
from ...utils import ensure_contiguous

triton_pi = tl.constexpr(3.141592653589793)


@triton.jit
def fused_cfconv_kernel(
    # Input pointers
    x_ptr,  # [num_nodes, feature_dim] - node features
    filters_ptr,  # [num_edges, feature_dim] - filters outputs (original order, FP32 or FP16)
    edge_weight_ptr,  # [num_edges] - distances (original order)
    edge_src_ptr,  # [num_edges] - source indices (original order)
    csr_perm_ptr,  # [num_edges] - CSR permutation
    dst_ptr_ptr,  # [num_nodes + 1] - CSR row pointers
    # Output pointer
    output_ptr,  # [num_nodes, feature_dim]
    # Parameters
    cutoff_upper,
    num_nodes,
    feature_dim,
    # Block size
    BLOCK_F: tl.constexpr,
    filters_FP16: tl.constexpr,  # Whether filters is FP16
):
    """
    Fused CSR-based CFConv: cutoff + gather + multiply + segment-reduce.

    This kernel fuses:
    1. Cutoff calculation: C = 0.5 * (cos(dist * pi / cutoff) + 1)
    2. filters scaling: W = filters * C
    3. Gather: x_j = x[src]
    4. Multiply: msg = x_j * W
    5. Segment reduce: output[dst] = sum(msg) for all edges to dst

    Key difference from atomic scatter: NO ATOMICS!
    Each block processes one destination node and has exclusive write access.

    Supports both FP32 and FP16 filters input:
    - If filters_FP16=True: loads FP16, promotes to FP32 for computation
    - If filters_FP16=False: loads FP32 directly
    Output is always FP32.

    Grid: (num_nodes,) - one block per destination
    """
    node_idx = tl.program_id(0)

    if node_idx >= num_nodes:
        return

    # Get segment bounds from CSR row pointers
    seg_start = tl.load(dst_ptr_ptr + node_idx)
    seg_end = tl.load(dst_ptr_ptr + node_idx + 1)

    # Process features in blocks
    for f_start in range(0, feature_dim, BLOCK_F):
        f_offsets = f_start + tl.arange(0, BLOCK_F)
        f_mask = f_offsets < feature_dim

        # Accumulate in registers (no atomics!)
        acc = tl.zeros([BLOCK_F], dtype=tl.float32)

        # Loop over all edges targeting this node
        for e_csr in range(seg_start, seg_end):
            # Get original edge index via CSR permutation
            edge_idx = tl.load(csr_perm_ptr + e_csr)

            # Load source node index
            src_node = tl.load(edge_src_ptr + edge_idx)

            # Load distance and compute cutoff
            dist = tl.load(edge_weight_ptr + edge_idx)
            C = _cosine_cutoff(dist, cutoff_upper)

            # Load filters output and apply cutoff (FP16 or FP32)
            filters_val = tl.load(
                filters_ptr + edge_idx * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )
            # Promote FP16 to FP32 for computation
            if filters_FP16:
                filters_val = filters_val.to(tl.float32)
            W = filters_val * C

            # Gather source features
            x_j = tl.load(
                x_ptr + src_node * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )

            # Accumulate message
            acc += x_j * W

        # Single store per destination - no atomic needed!
        tl.store(
            output_ptr + node_idx * feature_dim + f_offsets,
            acc,
            mask=f_mask,
        )


@triton_op("mlcg_kernels::fused_cfconv", mutates_args={})
@ensure_contiguous
def fused_cfconv(
    x: torch.Tensor,
    filters: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    dst_ptr: torch.Tensor,
    csr_perm: torch.Tensor,
    num_nodes: int,
    cutoff_upper: float,
    src_ptr: torch.Tensor,
    src_perm: torch.Tensor,
) -> torch.Tensor:
    """
    Fused CSR-based CFConv operation.

    This is a drop-in replacement for fused_cutoff_gather_multiply_scatter
    that uses CSR format for efficient aggregation without atomics.

    Supports both FP32 and FP16 filters input (auto-detected).
    Output is always FP32.

    Parameters
    ----------
    x : torch.Tensor
        Node features [num_nodes, feature_dim]
    filters : torch.Tensor
        filters outputs [num_edges, feature_dim], can be FP32 or FP16
    edge_weight : torch.Tensor
        Edge weights (distances) [num_edges]
    edge_src : torch.Tensor
        Source node indices [num_edges]
    edge_dst : torch.Tensor
        Destination node indices [num_edges]
    dst_ptr : torch.Tensor
        CSR row pointers [num_nodes + 1]
    csr_perm : torch.Tensor
        CSR permutation [num_edges]
    num_nodes : int
        Number of nodes
    cutoff_upper : float
        Upper cutoff distance
    src_ptr : torch.Tensor
        Src-CSR row pointers [num_nodes + 1]. Along with src_perm,
        enables atomic-free grad_x computation in backward.
    src_perm : torch.Tensor
        Src-CSR permutation [num_edges].

    Returns
    -------
    torch.Tensor
        Output [num_nodes, feature_dim] in FP32
    """
    feature_dim = x.shape[1]

    # Allocate output (zeros needed for nodes with no incoming edges)
    output = torch.zeros(num_nodes, feature_dim, device=x.device, dtype=x.dtype).contiguous()

    num_edges = edge_src.shape[0]
    if num_edges == 0:
        return output

    # Choose block size
    BLOCK_F = min(128, triton.next_power_of_2(feature_dim))

    # Auto-detect filters dtype
    filters_fp16 = filters.dtype == torch.float16

    # One block per destination node
    grid = (num_nodes,)

    wrap_triton(fused_cfconv_kernel)[grid](
        x,
        filters,
        edge_weight,
        edge_src,
        csr_perm,
        dst_ptr,
        output,
        cutoff_upper,
        num_nodes,
        feature_dim,
        BLOCK_F=BLOCK_F,
        filters_FP16=filters_fp16,
    )

    return output


def setup_context(ctx, inputs, output):
    (
        x,
        filters,
        edge_weight,
        edge_src,
        edge_dst,
        dst_ptr,
        csr_perm,
        num_nodes,
        cutoff_upper,
        src_ptr,
        src_perm,
    ) = inputs

    ctx.save_for_backward(
        x,
        filters,
        edge_weight,
        edge_src,
        edge_dst,
        src_ptr,
        src_perm,
        dst_ptr,
        csr_perm,
    )

    ctx.num_nodes = num_nodes
    ctx.cutoff_upper = cutoff_upper
    ctx.filters_dtype = filters.dtype


def backward(ctx, grad_output):
    (
        x,
        filters,
        edge_weight,
        edge_src,
        edge_dst,
        src_ptr,
        src_perm,
        dst_ptr,
        dst_perm,
    ) = ctx.saved_tensors

    num_nodes = ctx.num_nodes
    cutoff_upper = ctx.cutoff_upper
    filters_dtype = ctx.filters_dtype

    grad_output = grad_output.contiguous()

    grad_x = None
    grad_filters = None
    grad_edge_weight = None

    if ctx.needs_input_grad[0]:
        # grad_x[src] += grad_output[dst] * W
        grad_x = grad_x_fused_cfconv(
            grad_output,
            filters,
            edge_weight,
            edge_src,
            edge_dst,
            src_ptr,
            src_perm,
            num_nodes,
            cutoff_upper,
        )

    if ctx.needs_input_grad[1]:
        # grad_filters[e] = x[src[e]] * grad_output[dst[e]] * C[e]
        # Output dtype matches filters dtype (FP32 or FP16)
        grad_filters = grad_filters_fused_cfconv(
            x,
            grad_output,
            edge_weight,
            edge_src,
            edge_dst,
            cutoff_upper,
            out_dtype=filters_dtype,
        )

    if ctx.needs_input_grad[2]:
        grad_edge_weight = grad_edge_weight_fused_cfconv(
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
            out_dtype=filters_dtype,
        )

    return (
        grad_x,
        grad_filters,
        grad_edge_weight,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


fused_cfconv.register_autograd(backward, setup_context=setup_context)


@fused_cfconv.register_kernel("cpu")
def cpu_fused_cfconv(
    x: torch.Tensor,
    filters: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    dst_ptr: torch.Tensor,
    csr_perm: torch.Tensor,
    num_nodes: int,
    cutoff_upper: float,
    src_ptr: torch.Tensor,
    src_perm: torch.Tensor,
) -> torch.Tensor:
    """
    CPU fallback for fused_cfconv
    """

    C = 0.5 * (torch.cos(edge_weight * torch.pi / cutoff_upper) + 1)
    C = C * (edge_weight < cutoff_upper).float()
    messages = x[edge_src] * filters * C.unsqueeze(-1)
    out = torch.zeros_like(x)
    out.index_add_(0, edge_dst, messages)
    # out = scatter(messages, edge_dst, dim=0, reduce='sum')
    return out
