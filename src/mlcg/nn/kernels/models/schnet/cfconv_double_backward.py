import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton
from typing import List

from ...utils import ensure_contiguous


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
def double_grad_filters_grad_weight_cfconv_kernel(
    grad_grad_x_ptr,  # [num_nodes, feature_dim]
    grad_grad_filters_ptr,  # [num_edges, feature_dim]
    grad_grad_edge_weight_ptr,  # [num_edges]
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
    Kernel for computation of double backward of `fused_cfconv`.
    This kernel computes the double grad component grad_filters and grad_weights

    Computes:
        grad_filters[e] = grad_grad_edge_weight[e] * grad_output[dst[e]] * x[src[e]]
                          + grad_grad_x[src[e]] * grad_output[dst[e]] * edge_weight[e]
        grad_edge_weight[e] = (
            grad_grad_filters[e] * grad_output[dst[e]] * x[src[e]]
            + grad_grad_x[src[e]] * grad_output[dst[e]] * filters[e]
        ).sum()

    """
    edge_idx = tl.program_id(axis=0)

    if edge_idx >= num_edges:
        return

    # Load edge info
    src_node = tl.load(edge_src_ptr + edge_idx)
    dst_node = tl.load(edge_dst_ptr + edge_idx)

    if NEED_GRAD_FILTERS:
        C = tl.load(edge_weight_ptr + edge_idx)
        ggC = tl.load(grad_grad_edge_weight_ptr + edge_idx)

    if NEED_GRAD_EDGE_WEIGHT:
        acc_grad_edge_weight = tl.zeros([BLOCK_F], dtype=tl.float32)

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

        grad_grad_x_src = tl.load(
            grad_grad_x_ptr + src_node * feature_dim + f_offsets,
            mask=f_mask,
            other=0.0,
        )

        if NEED_GRAD_EDGE_WEIGHT:
            filters = tl.load(
                filters_ptr + edge_idx * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )
            grad_grad_filters = tl.load(
                grad_grad_filters_ptr + edge_idx * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )
            acc_grad_edge_weight += (
                grad_grad_filters * x_src + grad_grad_x_src * filters
            ) * grad_output_dst

        if NEED_GRAD_FILTERS:
            grad_filters = (ggC * x_src + grad_grad_x_src * C) * grad_output_dst
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
def double_grad_x_cfconv_kernel(
    grad_grad_filters_ptr,  # [num_edges, feature_dim]
    grad_grad_edge_weight_ptr,  # [num_edges]
    grad_output_ptr,  # [num_nodes, feature_dim]
    filters_ptr,  # [num_edges, feature_dim]
    edge_weight_ptr,  # [num_edges]
    edge_dst_ptr,  # [num_edges]
    src_perm_ptr,  # [num_edges]
    src_ptr_ptr,  # [num_nodes + 1]
    grad_x_ptr,  # [num_nodes, feature_dim]
    num_nodes,
    feature_dim: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    """
    Kernel for computation of double backward of `fused_cfconv`.
    This kernel computes the double grad component grad_x

    CSR format for efficient aggregation without atomics.

    Computes:
        grad_x[src] = (
            sum_{e: src[e]=src} grad_grad_filters[e] * grad_output[dst[e]] * edge_weight[e]
            + sum_{e: src[e]=src} grad_grad_edge_weight[e] * grad_output[dst[e]] * filters[e]
        )

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
            ggC = tl.load(grad_grad_edge_weight_ptr + edge_idx)
            filters = tl.load(
                filters_ptr + edge_idx * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )
            grad_grad_filters = tl.load(
                grad_grad_filters_ptr + edge_idx * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )
            grad_output_dst = tl.load(
                grad_output_ptr + dst_node * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )

            acc += (grad_grad_filters * C + ggC * filters) * grad_output_dst

        tl.store(
            grad_x_ptr + src_node * feature_dim + f_offsets,
            acc,
            mask=f_mask,
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
def double_grad_grad_out_cfconv_kernel(
    grad_grad_x_ptr,  # [num_nodes, feature_dim]
    grad_grad_filters_ptr,  # [num_edges, feature_dim]
    grad_grad_edge_weight_ptr,  # [num_edges]
    x_ptr,  # [num_nodes, feature_dim]
    filters_ptr,  # [num_edges, feature_dim]
    edge_weight_ptr,  # [num_edges]
    edge_src_ptr,  # [num_edges]
    dst_perm_ptr,  # [num_edges]
    dst_ptr_ptr,  # [num_nodes + 1]
    grad_grad_output_ptr,  # [num_nodes, feature_dim]
    num_nodes,
    feature_dim: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    """
    Kernel for computation of double backward of `fused_cfconv`.
    This kernel computes the double grad component grad_x and grad_grad_out

    CSR format for efficient aggregation without atomics.

    Computes:
        grad_grad_out[dst] = (
            sum_{e: dst[e]=dst} grad_grad_filters[e] * x[src[e]] * edge_weight[e]
            + sum_{e: dst[e]=dst} grad_grad_edge_weight[e] * x[src[e]] * filters[e]
            + sum_{e: dst[e]=dst} grad_grad_x[src[e]] * filters[e] * edge_weight[e]
        )

    """
    dst_node = tl.program_id(0)

    if dst_node >= num_nodes:
        return

    seg_start = tl.load(dst_ptr_ptr + dst_node)
    seg_end = tl.load(dst_ptr_ptr + dst_node + 1)

    for f_start in range(0, feature_dim, BLOCK_F):
        f_offsets = f_start + tl.arange(0, BLOCK_F)
        f_mask = f_offsets < feature_dim

        acc = tl.zeros([BLOCK_F], dtype=tl.float32)

        for e_csr in range(seg_start, seg_end):
            edge_idx = tl.load(dst_perm_ptr + e_csr)
            src_node = tl.load(edge_src_ptr + edge_idx)

            C = tl.load(edge_weight_ptr + edge_idx)
            ggC = tl.load(grad_grad_edge_weight_ptr + edge_idx)
            filters = tl.load(
                filters_ptr + edge_idx * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )
            grad_grad_filters = tl.load(
                grad_grad_filters_ptr + edge_idx * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )
            x_src = tl.load(
                x_ptr + src_node * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )
            grad_grad_x_src = tl.load(
                grad_grad_x_ptr + src_node * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )

            acc += (
                grad_grad_filters * x_src * C
                + (ggC * x_src + grad_grad_x_src * C) * filters
            )

        tl.store(
            grad_grad_output_ptr + dst_node * feature_dim + f_offsets,
            acc,
            mask=f_mask,
        )


@triton_op("mlcg_kernels::double_backward_fused_cfconv", mutates_args={})
@ensure_contiguous
def double_backward_fused_cfconv(
    grad_grad_x: torch.Tensor,
    grad_grad_filters: torch.Tensor,
    grad_grad_edge_weight: torch.Tensor,
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
    need_grad_grad_output: bool,
) -> List[torch.Tensor]:
    """
    ADD DOCSTRING
    """
    num_edges = edge_src.shape[0]
    feature_dim = x.shape[1]

    grad_grad_output = (
        torch.zeros_like(grad_output)
        if need_grad_grad_output
        else torch.empty(0)
    )
    grad_filters = (
        torch.zeros_like(filters) if need_grad_filters else torch.empty(0)
    )
    grad_edge_weight = (
        torch.zeros_like(edge_weight)
        if need_grad_edge_weight
        else torch.empty(0)
    )
    grad_x = torch.zeros_like(x) if need_grad_x else torch.empty(0)

    if need_grad_filters or need_grad_edge_weight:
        grid = (num_edges,)
        wrap_triton(double_grad_filters_grad_weight_cfconv_kernel)[grid](
            grad_grad_x,
            grad_grad_filters,
            grad_grad_edge_weight,
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
            need_grad_filters,
            need_grad_edge_weight,
        )

    if need_grad_x:
        grid = (num_nodes,)
        wrap_triton(double_grad_x_cfconv_kernel)[grid](
            grad_grad_filters,
            grad_grad_edge_weight,
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

    if need_grad_grad_output:
        grid = (num_nodes,)
        wrap_triton(double_grad_grad_out_cfconv_kernel)[grid](
            grad_grad_x,
            grad_grad_filters,
            grad_grad_edge_weight,
            x,
            filters,
            edge_weight,
            edge_src,
            dst_perm,
            dst_ptr,
            grad_grad_output,
            num_nodes,
            feature_dim,
        )

    return [grad_grad_output, grad_x, grad_filters, grad_edge_weight]


@double_backward_fused_cfconv.register_kernel("cpu")
def _(
    grad_grad_x: torch.Tensor,
    grad_grad_filters: torch.Tensor,
    grad_grad_edge_weight: torch.Tensor,
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
    need_grad_grad_output: bool,
) -> List[torch.Tensor]:
    """
    CPU fallback for double_backward_fused_cfconv
    """
    grad_grad_output = torch.zeros_like(grad_output)
    grad_x = torch.zeros_like(x)
    grad_filters = torch.zeros_like(filters)
    grad_edge_weight = torch.zeros_like(edge_weight)

    if need_grad_filters:
        grad_filters = (
            grad_grad_edge_weight.unsqueeze(-1) * x[edge_src]
            + grad_grad_x[edge_src] * edge_weight.unsqueeze(-1)
        ) * grad_output[edge_dst]

    if need_grad_edge_weight:
        grad_edge_weight = (
            (grad_grad_filters * x[edge_src] + grad_grad_x[edge_src] * filters)
            * grad_output[edge_dst]
        ).sum(-1)

    if need_grad_x:
        g_x = (
            grad_grad_filters * edge_weight.unsqueeze(-1)
            + grad_grad_edge_weight.unsqueeze(-1) * filters
        ) * grad_output[edge_dst]
        grad_x.index_add_(0, edge_src, g_x)

    if need_grad_grad_output:
        gg_o = (
            grad_grad_filters * x[edge_src] * edge_weight.unsqueeze(-1)
            + grad_grad_edge_weight.unsqueeze(-1) * x[edge_src] * filters
            + grad_grad_x[edge_src] * filters * edge_weight.unsqueeze(-1)
        )
        grad_grad_output.index_add_(0, edge_dst, gg_o)

    return [grad_grad_output, grad_x, grad_filters, grad_edge_weight]
