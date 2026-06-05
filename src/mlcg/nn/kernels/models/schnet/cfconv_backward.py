import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton
from typing import List

from ...utils import ensure_contiguous
from .cfconv_double_backward import double_backward_fused_cfconv


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
def grad_filters_grad_weight_cfconv_kernel(
    grad_output_ptr,  # [num_nodes, feature_dim]
    x_ptr,  # [num_nodes, feature_dim]
    filters_ptr,  # [num_edges, feature_dim]
    edge_weight_ptr,  # [num_edges]
    edge_src_ptr,  # [num_edges]
    edge_dst_ptr,  # [num_edges]
    grad_filters_ptr,  # [num_edges, feature_dim]
    grad_edge_weight_ptr,  # [num_edges]
    num_edges,
    feature_dim: tl.constexpr,
    NEED_GRAD_FILTERS: tl.constexpr,
    NEED_GRAD_EDGE_WEIGHT: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    """
    Kernel for computation of grad_filters and grad_weight for CFConv backward pass.

    Computes:
        grad_filters[e] = grad_output[dst[e]] * x[src[e]] * edge_weight[e]
        grad_edge_weight[e] = (grad_output[dst[e]] * x[src[e]] * filters[e]).sum()

    """
    edge_idx = tl.program_id(axis=0)

    if edge_idx >= num_edges:
        return

    # Load edge info
    src_node = tl.load(edge_src_ptr + edge_idx)
    dst_node = tl.load(edge_dst_ptr + edge_idx)

    if NEED_GRAD_FILTERS:
        C = tl.load(edge_weight_ptr + edge_idx)

    if NEED_GRAD_EDGE_WEIGHT:
        acc_grad_edge_weight = tl.zeros([BLOCK_F], dtype=tl.float32)

    # Process features in blocks
    for f_start in range(0, feature_dim, BLOCK_F):
        f_offsets = f_start + tl.arange(0, BLOCK_F)
        f_mask = f_offsets < feature_dim

        x_src = tl.load(
            x_ptr + src_node * feature_dim + f_offsets, mask=f_mask, other=0.0
        )

        grad_output_dst = tl.load(
            grad_output_ptr + dst_node * feature_dim + f_offsets,
            mask=f_mask,
            other=0.0,
        )

        if NEED_GRAD_EDGE_WEIGHT:
            filters = tl.load(
                filters_ptr + edge_idx * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )
            acc_grad_edge_weight += grad_output_dst * x_src * filters

        if NEED_GRAD_FILTERS:
            grad_filters = x_src * grad_output_dst * C
            tl.store(
                grad_filters_ptr + edge_idx * feature_dim + f_offsets,
                grad_filters,
                mask=f_mask,
            )

    if NEED_GRAD_EDGE_WEIGHT:
        tl.store(grad_edge_weight_ptr + edge_idx, tl.sum(acc_grad_edge_weight))


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
def grad_x_cfconv_kernel(
    grad_output_ptr,  # [num_nodes, feature_dim] - gradient from output (FP32)
    filters_ptr,  # [num_edges, feature_dim] - filters outputs (FP32 or FP16)
    edge_weight_ptr,  # [num_edges] - distances for cutoff
    edge_dst_ptr,  # [num_edges] - destination indices (original order)
    src_perm_ptr,  # [num_edges] - src-CSR permutation
    src_ptr_ptr,  # [num_nodes + 1] - src-CSR row pointers
    grad_x_ptr,  # [num_nodes, feature_dim]
    num_nodes,
    feature_dim: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    """
    Kernel for computation of grad_x for CFConv backward pass using
    CSR format for efficient aggregation without atomics.

    Computes:
        grad_x[src] = sum_{e: src[e]=src} grad_output[dst[e]] * filters[e] * cutoff[e]

    """
    src_node = tl.program_id(0)

    if src_node >= num_nodes:
        return

    seg_start = tl.load(src_ptr_ptr + src_node)
    seg_end = tl.load(src_ptr_ptr + src_node + 1)

    for f_start in range(0, feature_dim, BLOCK_F):
        f_offsets = f_start + tl.arange(0, BLOCK_F)
        f_mask = f_offsets < feature_dim

        acc = tl.zeros([BLOCK_F], dtype=tl.float32)

        for e_csr in range(seg_start, seg_end):
            edge_idx = tl.load(src_perm_ptr + e_csr)
            dst_node = tl.load(edge_dst_ptr + edge_idx)
            C = tl.load(edge_weight_ptr + edge_idx)
            filters_val = tl.load(
                filters_ptr + edge_idx * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )
            grad_output_dst = tl.load(
                grad_output_ptr + dst_node * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )

            acc += grad_output_dst * filters_val * C

        tl.store(
            grad_x_ptr + src_node * feature_dim + f_offsets,
            acc,
            mask=f_mask,
        )


@triton_op("mlcg_kernels::backward_fused_cfconv", mutates_args={})
@ensure_contiguous
def backward_fused_cfconv(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    filters: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    src_ptr: torch.Tensor,
    src_perm: torch.Tensor,
    dst_ptr: torch.Tensor,
    dst_perm: torch.Tensor,
    num_nodes: int,
    need_grad_x: bool,
    need_grad_filters: bool,
    need_grad_edge_weight: bool,
) -> List[torch.Tensor]:
    """
    ADD DOCSTRING
    """
    num_edges = edge_src.shape[0]
    feature_dim = x.shape[1]

    grad_filters = (
        torch.zeros_like(filters).contiguous()
        if need_grad_filters
        else torch.empty(0)
    )
    grad_edge_weight = (
        torch.zeros_like(edge_weight).contiguous()
        if need_grad_edge_weight
        else torch.empty(0)
    )
    grad_x = torch.zeros_like(x).contiguous() if need_grad_x else torch.empty(0)

    if need_grad_edge_weight or need_grad_filters:
        grid = (num_edges,)
        wrap_triton(grad_filters_grad_weight_cfconv_kernel)[grid](
            grad_output,
            x,
            filters,
            edge_weight,
            edge_src,
            edge_dst,
            grad_filters,
            grad_edge_weight,
            num_edges,
            feature_dim,
            NEED_GRAD_FILTERS=need_grad_filters,
            NEED_GRAD_EDGE_WEIGHT=need_grad_edge_weight,
        )

    if need_grad_x:
        grid = (num_nodes,)
        wrap_triton(grad_x_cfconv_kernel)[grid](
            grad_output,
            filters,
            edge_weight,
            edge_dst,
            src_perm,
            src_ptr,
            grad_x,
            num_nodes,
            feature_dim,
        )

    return [grad_x, grad_filters, grad_edge_weight]


def setup_context(ctx, inputs, output):
    (
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
    ) = inputs
    ctx.save_for_backward(
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
    )
    ctx.num_nodes = num_nodes


def backward(ctx, grad_output):

    (
        grad_out,
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
    grad_grad_x, grad_grad_filters, grad_grad_edge_weight = grad_output

    need_grad_grad_output = ctx.needs_input_grad[0]
    need_grad_x = ctx.needs_input_grad[1]
    need_grad_filters = ctx.needs_input_grad[2]
    need_grad_edge_weight = ctx.needs_input_grad[3]

    grad_grad_output, grad_x, grad_filters, grad_edge_weight = (
        double_backward_fused_cfconv(
            grad_grad_x,
            grad_grad_filters,
            grad_grad_edge_weight,
            grad_out,
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
            need_grad_grad_output,
        )
    )

    if not need_grad_grad_output:
        grad_grad_output = None
    if not need_grad_x:
        grad_x = None
    if not need_grad_filters:
        grad_filters = None
    if not need_grad_edge_weight:
        grad_edge_weight = None

    return (
        grad_grad_output,
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
        None,
        None,
    )


backward_fused_cfconv.register_autograd(backward, setup_context=setup_context)


@backward_fused_cfconv.register_kernel("cpu")
def _(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    filters: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    src_ptr: torch.Tensor,
    src_perm: torch.Tensor,
    dst_ptr: torch.Tensor,
    dst_perm: torch.Tensor,
    num_nodes: int,
    need_grad_x: bool,
    need_grad_filters: bool,
    need_grad_edge_weight: bool,
) -> List[torch.Tensor]:
    """
    CPU fallback for backward_fused_cfconv
    """

    grad_filters = torch.zeros_like(filters)
    grad_edge_weight = torch.zeros_like(edge_weight)
    grad_x = torch.zeros_like(x)

    if need_grad_filters:
        grad_filters = (
            grad_output[edge_dst] * x[edge_src] * edge_weight.unsqueeze(-1)
        )
    if need_grad_edge_weight:
        grad_edge_weight = (grad_output[edge_dst] * x[edge_src] * filters).sum(
            -1
        )
    if need_grad_x:
        gx = grad_output[edge_dst] * filters * edge_weight.unsqueeze(-1)
        grad_x.index_add_(0, edge_src, gx)

    return [grad_x, grad_filters, grad_edge_weight]
