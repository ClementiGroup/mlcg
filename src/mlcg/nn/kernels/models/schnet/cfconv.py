"""
Fused Triton kernels for CFConv operations adopting csr representation
for more efficient scatter operations.
"""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from ...utils import ensure_contiguous
from .cfconv_backward import backward_fused_cfconv


def filter_configs(configs, named_args, **kwargs):
    feature_dim = named_args["feature_dim"]
    filtered = [cfg for cfg in configs if cfg.kwargs["BLOCK_F"] <= feature_dim]
    return (
        filtered
        if filtered
        else [min(configs, key=lambda c: c.kwargs["BLOCK_F"])]
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_F": 8}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_F": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_F": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_F": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_F": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_F": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_F": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_F": 128}, num_warps=8, num_stages=3),
    ],
    key=["feature_dim"],
    prune_configs_by={"early_config_prune": filter_configs},
)
@triton.jit
def fused_cfconv_kernel(
    # Input pointers
    x_ptr,  # [num_nodes, feature_dim] - node features
    filters_ptr,  # [num_edges, feature_dim] - filters outputs (original order, FP32 or FP16)
    edge_weight_ptr,  # [num_edges] - Cutoff(distances) (original order)
    edge_src_ptr,  # [num_edges] - source indices (original order)
    dst_perm_ptr,  # [num_edges] - CSR permutation
    dst_ptr_ptr,  # [num_nodes + 1] - CSR row pointers
    # Output pointer
    output_ptr,  # [num_nodes, feature_dim]
    # Parameters
    num_nodes,
    feature_dim: tl.constexpr,
    # Block size
    BLOCK_F: tl.constexpr,
):
    """
    Fused CSR-based CFConv kernel.
    Performs

    output[dst] = sum_{src} (x_{src} * filters_{e} * cutoff_{e})

    Grid: (num_nodes,) - one block per destination
    """
    node_idx = tl.program_id(0)

    if node_idx >= num_nodes:
        return

    # Get segment bounds from CSR row pointers
    seg_start = tl.load(dst_ptr_ptr + node_idx)
    seg_end = tl.load(dst_ptr_ptr + node_idx + 1)

    for f_start in range(0, feature_dim, BLOCK_F):
        f_offsets = f_start + tl.arange(0, BLOCK_F)
        f_mask = f_offsets < feature_dim

        acc = tl.zeros([BLOCK_F], dtype=tl.float32)
        for e_csr in range(seg_start, seg_end):
            edge_idx = tl.load(dst_perm_ptr + e_csr)
            src_node = tl.load(edge_src_ptr + edge_idx)

            C = tl.load(edge_weight_ptr + edge_idx)
            filters_val = tl.load(
                filters_ptr + edge_idx * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )

            W = filters_val * C
            x_j = tl.load(
                x_ptr + src_node * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )

            acc += x_j * W

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
    dst_perm: torch.Tensor,
    num_nodes: int,
    src_ptr: torch.Tensor,
    src_perm: torch.Tensor,
) -> torch.Tensor:
    """
    Fused CSR-based CFConv operation.

    This is a drop-in replacement for fused_cutoff_gather_multiply_scatter
    that uses CSR format for efficient aggregation without atomics.

    Parameters
    ----------
    x : torch.Tensor
        Node features [num_nodes, feature_dim]
    filters : torch.Tensor
        filters outputs [num_edges, feature_dim], can be FP32 or FP16
    edge_weight : torch.Tensor
        Edge weights (Cutoff(distances)) [num_edges]
    edge_src : torch.Tensor
        Source node indices [num_edges]
    edge_dst : torch.Tensor
        Destination node indices [num_edges]
    dst_ptr : torch.Tensor
        CSR row pointers [num_nodes + 1]
    dst_perm : torch.Tensor
        CSR permutation [num_edges]
    num_nodes : int
        Number of nodes
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

    output = torch.zeros(
        num_nodes, feature_dim, device=x.device, dtype=x.dtype
    ).contiguous()

    num_edges = edge_src.shape[0]
    if num_edges == 0:
        return output

    grid = (num_nodes,)
    wrap_triton(fused_cfconv_kernel)[grid](
        x,
        filters,
        edge_weight,
        edge_src,
        dst_perm,
        dst_ptr,
        output,
        num_nodes,
        feature_dim,
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
        dst_perm,
        num_nodes,
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
        dst_perm,
    )
    ctx.num_nodes = num_nodes


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

    need_grad_x = ctx.needs_input_grad[0]
    need_grad_filters = ctx.needs_input_grad[1]
    need_grad_edge_weight = ctx.needs_input_grad[2]

    grad_x, grad_filters, grad_edge_weight = backward_fused_cfconv(
        grad_output,
        x,
        filters,
        edge_weight,
        edge_src,
        edge_dst,
        src_ptr,
        src_perm,
        dst_ptr,
        dst_perm,
        num_nodes,
        need_grad_x,
        need_grad_filters,
        need_grad_edge_weight,
    )

    if not need_grad_x:
        grad_x = None
    if not need_grad_filters:
        grad_filters = None
    if not need_grad_edge_weight:
        grad_edge_weight = None

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
    )


fused_cfconv.register_autograd(backward, setup_context=setup_context)


@fused_cfconv.register_kernel("cpu")
def _(
    x: torch.Tensor,
    filters: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    dst_ptr: torch.Tensor,
    csr_perm: torch.Tensor,
    num_nodes: int,
    src_ptr: torch.Tensor,
    src_perm: torch.Tensor,
) -> torch.Tensor:
    """
    CPU fallback for fused_cfconv
    """

    messages = x[edge_src] * filters * edge_weight.unsqueeze(-1)
    out = torch.zeros_like(x)
    out = out.index_add(0, edge_dst, messages)
    return out
