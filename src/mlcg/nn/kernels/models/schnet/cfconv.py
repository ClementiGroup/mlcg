"""
CFConv message passing on the destination CSR of
a symmetric neighbour list
(:func:`csr_from_edge_index`, or ``RadiusCSR``):
``out[i] = sum_{e in row i} x[col[e]] * filters[e] * edge_weight[e]``, with
every per-edge tensor in row order. The backward and double backward are ops
too (force training).

The neighbour list is symmetric, with the same filter on both directions of
an edge (a radius graph with filters of the distance). The backward sums
over the edges *leaving* node ``i``, ``grad_x[i] = sum_j g[j] * W_(i->j)``;
by symmetry these are the edges of row ``i`` with ``W_(i->j) = W_(j->i)``,
so every kernel walks rows. The gradients of positions and parameters are
exact; a per-edge gradient may be attributed to the reverse edge.

Two kernels do all the work, one program per node:

- ``segment_kernel``: a sum over a row of neighbour features times edge
  weights (forward, ``grad_x``, double-backward ``grad_x`` and
  ``grad_grad_output``);
- ``edge_kernel``: the per-edge gradients of the filters and edge weights.
"""

from typing import List

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from ...utils import ensure_contiguous, tile_autotune


def _prune(configs, named_args, **kwargs):
    fits = [c for c in configs if c.kwargs["BLOCK_F"] <= named_args["F"]]
    return fits or [configs[0]]


def _autotune():
    return triton.autotune(
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
        key=["F"],
        prune_configs_by={"early_config_prune": _prune},
    )


@_autotune()
@triton.jit
def segment_kernel(
    node_ptr,
    node2_ptr,
    filters_ptr,
    weight_ptr,
    d_filters_ptr,
    d_weight_ptr,
    row_ptr,
    col_ptr,
    out_ptr,
    F: tl.constexpr,
    TANGENT: tl.constexpr,
    NODE2: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    """``out[i] = sum_e node[n] * w_t + node2[n] * w`` over the edges
    ``e = (n -> i)`` of row ``i``, with ``w = filters[e] * weight[e]`` and
    ``w_t = w``, or with ``TANGENT``
    ``d_filters[e] * weight[e] + filters[e] * d_weight[e]``."""
    i = tl.program_id(0).to(tl.int64)
    start = tl.load(row_ptr + i).to(tl.int64)
    end = tl.load(row_ptr + i + 1).to(tl.int64)
    for f0 in range(0, F, BLOCK_F):
        f = f0 + tl.arange(0, BLOCK_F)
        f_ok = f < F
        acc = tl.zeros([BLOCK_F], dtype=tl.float32)
        for e in range(start, end):
            n = tl.load(col_ptr + e).to(tl.int64)
            c = tl.load(weight_ptr + e)
            phi = tl.load(filters_ptr + e * F + f, mask=f_ok, other=0.0)
            w = phi * c
            w_t = w
            if TANGENT:
                d_phi = tl.load(d_filters_ptr + e * F + f, mask=f_ok, other=0.0)
                w_t = d_phi * c + phi * tl.load(d_weight_ptr + e)
            acc += tl.load(node_ptr + n * F + f, mask=f_ok, other=0.0) * w_t
            if NODE2:
                acc += tl.load(node2_ptr + n * F + f, mask=f_ok, other=0.0) * w
        tl.store(out_ptr + i * F + f, acc, mask=f_ok)


@tile_autotune(
    [
        {"BLOCK_E": e, "BLOCK_F": f}
        for e in (4, 8, 16, 32)
        for f in (32, 64, 128)
    ]
)
@triton.jit
def edge_kernel(
    grad_ptr,
    node_ptr,
    node2_ptr,
    a_filters_ptr,
    a_weight_ptr,
    filters_ptr,
    weight_ptr,
    row_ptr,
    col_ptr,
    grad_filters_ptr,
    grad_weight_ptr,
    F: tl.constexpr,
    NODE2: tl.constexpr,
    NEED_FILTERS: tl.constexpr,
    NEED_WEIGHT: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    """For the edges ``e = (n -> i)`` of row ``i``, ``BLOCK_E`` at a time:
    ``grad_filters[e] = grad[i] * (node[n] * a_weight[e] + node2[n] * weight[e])``,
    ``grad_weight[e] = sum_f grad[i] * (node[n] * a_filters[e] + node2[n] * filters[e])``
    (the ``node2`` terms only with ``NODE2``)."""
    i = tl.program_id(0).to(tl.int64)
    start = tl.load(row_ptr + i).to(tl.int64)
    end = tl.load(row_ptr + i + 1).to(tl.int64)
    for e0 in range(start, end, BLOCK_E):
        e = e0 + tl.arange(0, BLOCK_E)
        e_ok = e < end
        n = tl.load(col_ptr + e, mask=e_ok, other=0).to(tl.int64)[:, None]
        a_weight = tl.load(a_weight_ptr + e, mask=e_ok, other=0.0)[:, None]
        weight = tl.load(weight_ptr + e, mask=e_ok, other=0.0)[:, None]
        acc = tl.zeros([BLOCK_E], dtype=tl.float32)
        for f0 in range(0, F, BLOCK_F):
            f = f0 + tl.arange(0, BLOCK_F)
            mask = e_ok[:, None] & (f < F)[None, :]
            edge_f = e[:, None] * F + f[None, :]
            g = tl.load(grad_ptr + i * F + f, mask=f < F, other=0.0)[None, :]
            node = tl.load(node_ptr + n * F + f[None, :], mask=mask, other=0.0)
            if NODE2:
                node2 = tl.load(
                    node2_ptr + n * F + f[None, :], mask=mask, other=0.0
                )
            if NEED_FILTERS:
                grad_phi = node * a_weight
                if NODE2:
                    grad_phi += node2 * weight
                tl.store(grad_filters_ptr + edge_f, g * grad_phi, mask=mask)
            if NEED_WEIGHT:
                term = node * tl.load(
                    a_filters_ptr + edge_f, mask=mask, other=0.0
                )
                if NODE2:
                    term += node2 * tl.load(
                        filters_ptr + edge_f, mask=mask, other=0.0
                    )
                acc += tl.sum(g * term, axis=1)
        if NEED_WEIGHT:
            tl.store(grad_weight_ptr + e, acc, mask=e_ok)


def _segment(node, filters, weight, row_ptr, col, node2=None, tangent=None):
    out = torch.zeros_like(node)
    if node.shape[0]:
        d_filters, d_weight = (filters, weight) if tangent is None else tangent
        wrap_triton(segment_kernel)[(node.shape[0],)](
            node,
            node if node2 is None else node2,
            filters,
            weight,
            d_filters,
            d_weight,
            row_ptr,
            col,
            out,
            node.shape[1],
            TANGENT=tangent is not None,
            NODE2=node2 is not None,
        )
    return out


def _edge_grads(
    grad,
    node,
    a_filters,
    a_weight,
    filters,
    weight,
    row_ptr,
    col,
    need_filters,
    need_weight,
    node2=None,
):
    grad_filters = (
        torch.zeros_like(filters) if need_filters else filters.new_empty(0)
    )
    grad_weight = (
        torch.zeros_like(weight) if need_weight else weight.new_empty(0)
    )
    if (need_filters or need_weight) and grad.shape[0]:
        wrap_triton(edge_kernel)[(grad.shape[0],)](
            grad,
            node,
            node if node2 is None else node2,
            a_filters,
            a_weight,
            filters,
            weight,
            row_ptr,
            col,
            grad_filters,
            grad_weight,
            node.shape[1],
            NODE2=node2 is not None,
            NEED_FILTERS=need_filters,
            NEED_WEIGHT=need_weight,
        )
    return grad_filters, grad_weight


def csr_from_edge_index(edge_index: torch.Tensor, num_nodes: int):
    """``(row_ptr, col, order)`` of a symmetric ``[2, E]`` list (sources,
    destinations) in any order: edges sorted by destination, then source;
    per-edge tensors in row order are ``t[order]``."""
    src, dst = edge_index
    order = torch.sort(dst * num_nodes + src).indices
    nodes = torch.arange(num_nodes + 1, device=dst.device)
    row_ptr = torch.searchsorted(dst[order], nodes, out_int32=True)
    return row_ptr, src[order].int(), order


@triton_op("mlcg_kernels::fused_cfconv", mutates_args={})
@ensure_contiguous
def fused_cfconv(
    x: torch.Tensor,
    filters: torch.Tensor,
    edge_weight: torch.Tensor,
    row_ptr: torch.Tensor,
    col: torch.Tensor,
) -> torch.Tensor:
    """
    Parameters
    ----------
    x : [num_nodes, F] node features
    filters : [num_edges, F] filter values, in row order (FP32 or FP16)
    edge_weight : [num_edges] cutoff of each edge, in row order
    row_ptr, col : destination CSR of a symmetric neighbour list
        (:func:`csr_from_edge_index`)

    Returns
    -------
    [num_nodes, F]
    """
    return _segment(x, filters, edge_weight, row_ptr, col)


@triton_op("mlcg_kernels::backward_fused_cfconv", mutates_args={})
@ensure_contiguous
def backward_fused_cfconv(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    filters: torch.Tensor,
    edge_weight: torch.Tensor,
    row_ptr: torch.Tensor,
    col: torch.Tensor,
    need_grad_x: bool,
    need_grad_filters: bool,
    need_grad_edge_weight: bool,
) -> List[torch.Tensor]:
    """Gradients of ``<grad_output, fused_cfconv(x, filters, edge_weight)>``
    wrt ``x``, ``filters`` and ``edge_weight`` (empty if not needed)."""
    grad_filters, grad_weight = _edge_grads(
        grad_output,
        x,
        filters,
        edge_weight,
        filters,
        edge_weight,
        row_ptr,
        col,
        need_grad_filters,
        need_grad_edge_weight,
    )
    grad_x = (
        _segment(grad_output, filters, edge_weight, row_ptr, col)
        if need_grad_x
        else x.new_empty(0)
    )
    return [grad_x, grad_filters, grad_weight]


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
    row_ptr: torch.Tensor,
    col: torch.Tensor,
    need_grad_grad_output: bool,
    need_grad_x: bool,
    need_grad_filters: bool,
    need_grad_edge_weight: bool,
) -> List[torch.Tensor]:
    """Gradients of ``<grad_grad_*, backward_fused_cfconv(...)>`` wrt
    ``grad_output``, ``x``, ``filters`` and ``edge_weight``."""
    tangent = (grad_grad_filters, grad_grad_edge_weight)
    grad_filters, grad_weight = _edge_grads(
        grad_output,
        x,
        *tangent,
        filters,
        edge_weight,
        row_ptr,
        col,
        need_grad_filters,
        need_grad_edge_weight,
        node2=grad_grad_x,
    )
    grad_x = (
        _segment(
            grad_output, filters, edge_weight, row_ptr, col, tangent=tangent
        )
        if need_grad_x
        else x.new_empty(0)
    )
    grad_grad_output = (
        _segment(
            x,
            filters,
            edge_weight,
            row_ptr,
            col,
            node2=grad_grad_x,
            tangent=tangent,
        )
        if need_grad_grad_output
        else x.new_empty(0)
    )
    return [grad_grad_output, grad_x, grad_filters, grad_weight]


def _setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _backward(ctx, grad_output):
    x, filters, edge_weight, *graph = ctx.saved_tensors
    grads = backward_fused_cfconv(
        grad_output, x, filters, edge_weight, *graph, *ctx.needs_input_grad[:3]
    )
    grads = [
        g if need else None for g, need in zip(grads, ctx.needs_input_grad)
    ]
    return (*grads, None, None)


def _setup_backward_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:6])


def _double_backward(ctx, grad_grads):
    grad_output, x, filters, edge_weight, *graph = ctx.saved_tensors
    # a gradient the backward didn't compute was returned empty
    grad_grads = [
        g if g.numel() else torch.zeros_like(p)
        for g, p in zip(grad_grads, (x, filters, edge_weight))
    ]
    needs = ctx.needs_input_grad[:4]
    grads = double_backward_fused_cfconv(
        *grad_grads, grad_output, x, filters, edge_weight, *graph, *needs
    )
    grads = [g if need else None for g, need in zip(grads, needs)]
    return (*grads, *([None] * 5))


fused_cfconv.register_autograd(_backward, setup_context=_setup_context)
backward_fused_cfconv.register_autograd(
    _double_backward, setup_context=_setup_backward_context
)


def _edge_nodes(row_ptr, col, num_edges):
    """Source and destination of every edge, in row order."""
    nodes = torch.arange(row_ptr.numel() - 1, device=col.device)
    dst = torch.repeat_interleave(nodes, row_ptr.diff(), output_size=num_edges)
    return col[:num_edges].long(), dst


@fused_cfconv.register_kernel("cpu")
def _(x, filters, edge_weight, row_ptr, col):
    src, dst = _edge_nodes(row_ptr, col, filters.shape[0])
    messages = x[src] * filters * edge_weight[:, None]
    return torch.zeros_like(x).index_add(0, dst, messages)


@backward_fused_cfconv.register_kernel("cpu")
def _(
    grad_output,
    x,
    filters,
    edge_weight,
    row_ptr,
    col,
    need_grad_x,
    need_grad_filters,
    need_grad_edge_weight,
):
    src, dst = _edge_nodes(row_ptr, col, filters.shape[0])
    g, x_src = grad_output[dst], x[src]
    grad_x = torch.zeros_like(x).index_add(
        0, src, g * filters * edge_weight[:, None]
    )
    grad_filters = g * x_src * edge_weight[:, None]
    grad_weight = (g * x_src * filters).sum(-1)
    return [grad_x, grad_filters, grad_weight]


@double_backward_fused_cfconv.register_kernel("cpu")
def _(
    grad_grad_x,
    grad_grad_filters,
    grad_grad_edge_weight,
    grad_output,
    x,
    filters,
    edge_weight,
    row_ptr,
    col,
    need_grad_grad_output,
    need_grad_x,
    need_grad_filters,
    need_grad_edge_weight,
):
    src, dst = _edge_nodes(row_ptr, col, filters.shape[0])
    g, x_src, ggx_src = grad_output[dst], x[src], grad_grad_x[src]
    C, ggC = edge_weight[:, None], grad_grad_edge_weight[:, None]
    grad_filters = (ggC * x_src + ggx_src * C) * g
    grad_weight = ((grad_grad_filters * x_src + ggx_src * filters) * g).sum(-1)
    grad_x = torch.zeros_like(x).index_add(
        0, src, (grad_grad_filters * C + ggC * filters) * g
    )
    grad_grad_output = torch.zeros_like(x).index_add(
        0,
        dst,
        grad_grad_filters * x_src * C + (ggC * x_src + ggx_src * C) * filters,
    )
    return [grad_grad_output, grad_x, grad_filters, grad_weight]
