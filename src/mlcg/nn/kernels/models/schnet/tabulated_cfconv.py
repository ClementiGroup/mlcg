"""
CFConv with a tabulated filter: ``out[i] = sum_{j in N(i)} x[j] * phi(d_ij)``
where ``phi(d) = cutoff(d) * filter_network(rbf(d))`` is read from a cubic
Hermite table (see :mod:`hermite`) and ``d_ij`` is recomputed from ``pos``.

The graph is a destination CSR ``(row_ptr, col)`` of a *symmetric* neighbor
list (``j in N(i)`` iff ``i in N(j)``), which lets the backward pass gather
``grad_x`` and ``grad_pos`` per node without atomics, a source CSR or a
per-edge filter. Edges with ``d >= cutoff`` contribute nothing. The table
freezes the filter weights: inference only, no double backward. With
periodic boundary conditions, ``d_ij`` is the minimum-image distance
(:mod:`~mlcg.nn.kernels.pbc`).
"""

import copy
from functools import partial
from typing import List, Tuple

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from ...hermite import (
    aligned_num_knots,
    hermite_relative_error,
    hermite_slope,
    hermite_table,
    hermite_value,
    load_hermite_segment,
    locate_knot,
)
from ...pbc import minimum_image
from ...utils import ensure_contiguous


def _prune(configs, named_args, tile_per_thread, **kwargs):
    """Drop configs wider than the features or whose [BLOCK_E, BLOCK_F]
    tiles exceed ``tile_per_thread`` elements per thread (they spill)."""
    width = triton.next_power_of_2(named_args["F"])

    def fits(c):
        block_f = c.kwargs.get("BLOCK_F", width)
        tile = c.kwargs["BLOCK_E"] * block_f / (32 * c.num_warps)
        return block_f <= width and tile <= tile_per_thread

    return [c for c in configs if fits(c)] or [
        min(configs, key=lambda c: c.kwargs["BLOCK_E"])
    ]


def _configs(blocks, warps):
    return [
        triton.Config(dict(block), num_warps=w)
        for block in blocks
        for w in warps
    ]


@triton.jit
def _load_pos(pos_ptr, i):
    return (
        tl.load(pos_ptr + i * 3),
        tl.load(pos_ptr + i * 3 + 1),
        tl.load(pos_ptr + i * 3 + 2),
    )


@triton.jit
def _graph_index(batch_ptr, i, PBC: tl.constexpr):
    """Graph of node ``i``, read only for the minimum image."""
    mol = 0
    if PBC:
        mol = tl.load(batch_ptr + i)
    return mol


@triton.jit
def _neighbor_tile(
    e, end, col_ptr, pos_ptr, px, py, pz, cutoff, cell_ptr, pbc_ptr, mol, PBC
):
    """Neighbors ``j`` of edges ``e``, ``p_i - p_j`` (minimum image if
    ``PBC``), distance, in-range."""
    in_row = e < end
    j = tl.load(col_ptr + e, mask=in_row, other=0).to(tl.int64)
    dx = px - tl.load(pos_ptr + j * 3, mask=in_row, other=0.0)
    dy = py - tl.load(pos_ptr + j * 3 + 1, mask=in_row, other=0.0)
    dz = pz - tl.load(pos_ptr + j * 3 + 2, mask=in_row, other=0.0)
    if PBC:
        dx, dy, dz = minimum_image(dx, dy, dz, cell_ptr, pbc_ptr, mol)
    d = tl.sqrt(dx * dx + dy * dy + dz * dz)
    return j, dx, dy, dz, d, in_row & (d < cutoff)


@triton.autotune(
    configs=_configs(
        [
            {"BLOCK_E": e, "BLOCK_F": f}
            for e in (8, 16, 32, 64)
            for f in (32, 64, 128)
        ],
        (2, 4, 8),
    ),
    key=["F"],
    prune_configs_by={
        "early_config_prune": partial(_prune, tile_per_thread=16)
    },
)
@triton.jit
def tabulated_cfconv_kernel(
    x_ptr,
    pos_ptr,
    row_ptr,
    col_ptr,
    table_ptr,
    out_ptr,
    batch_ptr,
    cell_ptr,
    pbc_ptr,
    cutoff,
    num_knots,
    F: tl.constexpr,
    PBC: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64)
    mol = _graph_index(batch_ptr, i, PBC)
    f = tl.program_id(1) * BLOCK_F + tl.arange(0, BLOCK_F)
    f_ok = f < F
    start = tl.load(row_ptr + i)
    end = tl.load(row_ptr + i + 1)
    px, py, pz = _load_pos(pos_ptr, i)
    cutoff = tl.cast(cutoff, tl.float32)

    acc = tl.zeros([BLOCK_E, BLOCK_F], dtype=tl.float32)
    for e0 in range(start, end, BLOCK_E):
        j, _, _, _, d, valid = _neighbor_tile(
            e0 + tl.arange(0, BLOCK_E),
            end,
            col_ptr,
            pos_ptr,
            px,
            py,
            pz,
            cutoff,
            cell_ptr,
            pbc_ptr,
            mol,
            PBC,
        )
        k, t = locate_knot(d, cutoff, num_knots)
        mask = valid[:, None] & f_ok[None, :]
        v0, m0, dv, m1 = load_hermite_segment(table_ptr, k, f, mask, F)
        x_j = tl.load(x_ptr + j[:, None] * F + f[None, :], mask=mask, other=0.0)
        acc += x_j * hermite_value(t[:, None], v0, m0, dv, m1)

    tl.store(out_ptr + i * F + f, tl.sum(acc, axis=0), mask=f_ok)


@triton.autotune(
    configs=_configs([{"BLOCK_E": e} for e in (2, 4, 8, 16)], (2, 4, 8)),
    key=["F"],
    prune_configs_by={
        "early_config_prune": partial(_prune, tile_per_thread=16)
    },
)
@triton.jit
def tabulated_cfconv_backward_kernel(
    grad_out_ptr,
    x_ptr,
    pos_ptr,
    row_ptr,
    col_ptr,
    table_ptr,
    grad_x_ptr,
    grad_pos_ptr,
    batch_ptr,
    cell_ptr,
    pbc_ptr,
    cutoff,
    num_knots,
    F: tl.constexpr,
    PBC: tl.constexpr,
    BLOCK_F: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """Per node ``i`` (symmetric graph):
    ``grad_x[i] = sum_j g[j] * phi(d_ij)``,
    ``grad_pos[i] = sum_j (p_i - p_j) / d_ij
    * sum_f phi'_f(d_ij) (g[i, f] x[j, f] + g[j, f] x[i, f])``."""
    i = tl.program_id(0).to(tl.int64)
    mol = _graph_index(batch_ptr, i, PBC)
    f = tl.arange(0, BLOCK_F)
    f_ok = f < F
    start = tl.load(row_ptr + i)
    end = tl.load(row_ptr + i + 1)
    px, py, pz = _load_pos(pos_ptr, i)
    cutoff = tl.cast(cutoff, tl.float32)
    g_i = tl.load(grad_out_ptr + i * F + f, mask=f_ok, other=0.0)[None, :]
    x_i = tl.load(x_ptr + i * F + f, mask=f_ok, other=0.0)[None, :]

    acc_x = tl.zeros([BLOCK_E, BLOCK_F], dtype=tl.float32)
    acc_px = tl.zeros([BLOCK_E], dtype=tl.float32)
    acc_py = tl.zeros([BLOCK_E], dtype=tl.float32)
    acc_pz = tl.zeros([BLOCK_E], dtype=tl.float32)
    for e0 in range(start, end, BLOCK_E):
        j, dx, dy, dz, d, valid = _neighbor_tile(
            e0 + tl.arange(0, BLOCK_E),
            end,
            col_ptr,
            pos_ptr,
            px,
            py,
            pz,
            cutoff,
            cell_ptr,
            pbc_ptr,
            mol,
            PBC,
        )
        k, t = locate_knot(d, cutoff, num_knots)
        mask = valid[:, None] & f_ok[None, :]
        v0, m0, dv, m1 = load_hermite_segment(table_ptr, k, f, mask, F)
        node_j = j[:, None] * F + f[None, :]
        g_j = tl.load(grad_out_ptr + node_j, mask=mask, other=0.0)
        x_j = tl.load(x_ptr + node_j, mask=mask, other=0.0)

        t = t[:, None]
        acc_x += g_j * hermite_value(t, v0, m0, dv, m1)
        slope = hermite_slope(t, v0, m0, dv, m1)
        grad_d = tl.sum(slope * (g_i * x_j + g_j * x_i), axis=1)
        scale = grad_d * (num_knots - 1) / (d * cutoff)
        scale = tl.where(valid & (d > 0), scale, 0.0)
        acc_px += scale * dx
        acc_py += scale * dy
        acc_pz += scale * dz

    tl.store(grad_x_ptr + i * F + f, tl.sum(acc_x, axis=0), mask=f_ok)
    tl.store(grad_pos_ptr + i * 3, tl.sum(acc_px))
    tl.store(grad_pos_ptr + i * 3 + 1, tl.sum(acc_py))
    tl.store(grad_pos_ptr + i * 3 + 2, tl.sum(acc_pz))


@triton_op("mlcg_kernels::tabulated_cfconv", mutates_args={})
@ensure_contiguous
def tabulated_cfconv(
    x: torch.Tensor,
    pos: torch.Tensor,
    row_ptr: torch.Tensor,
    col: torch.Tensor,
    table: torch.Tensor,
    cutoff: float,
    batch: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
) -> torch.Tensor:
    """
    Parameters
    ----------
    x : [num_nodes, F] float32 node features
    pos : [num_nodes, 3] float32 positions
    row_ptr : [num_nodes + 1] destination-CSR row pointers (int32/int64)
    col : [num_edges] source node of each CSR edge (int32/int64)
    table : [num_knots, 3, F] float32 Hermite table of the filter over
        ``[0, cutoff]`` (:func:`build_filter_table`)
    cutoff : upper end of the table; longer edges are ignored
    batch : [num_nodes] graph index of each node
    cell, pbc : [B, 9] reduced cells and [B, 3] periodic flags of
        :func:`~mlcg.nn.kernels.pbc.periodic_cells`, or empty tensors
        (no periodic boundary conditions)

    Returns
    -------
    [num_nodes, F] float32
    """
    num_nodes, F = x.shape
    out = torch.empty_like(x)
    if num_nodes == 0:
        return out
    wrap_triton(tabulated_cfconv_kernel)[
        lambda meta: (num_nodes, triton.cdiv(F, meta["BLOCK_F"]))
    ](
        x,
        pos,
        row_ptr,
        col,
        table,
        out,
        batch,
        cell,
        pbc,
        cutoff,
        table.shape[0],
        F,
        PBC=cell.numel() > 0,
    )
    return out


@triton_op("mlcg_kernels::tabulated_cfconv_backward", mutates_args={})
@ensure_contiguous
def tabulated_cfconv_backward(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    pos: torch.Tensor,
    row_ptr: torch.Tensor,
    col: torch.Tensor,
    table: torch.Tensor,
    cutoff: float,
    batch: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
) -> List[torch.Tensor]:
    """Gradients of ``<grad_out, tabulated_cfconv(x, pos, ...)>`` wrt
    ``x`` and ``pos``. Requires a symmetric neighbor list."""
    num_nodes, F = x.shape
    grad_x = torch.empty_like(x)
    grad_pos = torch.empty_like(pos)
    if num_nodes == 0:
        return [grad_x, grad_pos]
    wrap_triton(tabulated_cfconv_backward_kernel)[(num_nodes,)](
        grad_out,
        x,
        pos,
        row_ptr,
        col,
        table,
        grad_x,
        grad_pos,
        batch,
        cell,
        pbc,
        cutoff,
        table.shape[0],
        F,
        PBC=cell.numel() > 0,
        BLOCK_F=triton.next_power_of_2(F),
    )
    return [grad_x, grad_pos]


@tabulated_cfconv.register_fake
def _(x, pos, row_ptr, col, table, cutoff, batch, cell, pbc):
    return torch.empty_like(x)


@tabulated_cfconv_backward.register_fake
def _(grad_out, x, pos, row_ptr, col, table, cutoff, batch, cell, pbc):
    return [torch.empty_like(x), torch.empty_like(pos)]


def _setup_context(ctx, inputs, output):
    x, pos, row_ptr, col, table, cutoff, batch, cell, pbc = inputs
    ctx.save_for_backward(x, pos, row_ptr, col, table, batch, cell, pbc)
    ctx.cutoff = cutoff


def _backward(ctx, grad_out):
    x, pos, row_ptr, col, table, batch, cell, pbc = ctx.saved_tensors
    grad_x, grad_pos = tabulated_cfconv_backward(
        grad_out, x, pos, row_ptr, col, table, ctx.cutoff, batch, cell, pbc
    )
    return grad_x, grad_pos, None, None, None, None, None, None, None


def _no_double_backward(ctx, *grads):
    raise NotImplementedError(
        "tabulated_cfconv has no double backward: the filter table is an "
        "inference-only (MD) path. Use the standard CFConv for force training."
    )


tabulated_cfconv.register_autograd(_backward, setup_context=_setup_context)
tabulated_cfconv_backward.register_autograd(
    _no_double_backward, setup_context=lambda ctx, inputs, output: None
)


def build_filter_table(
    filter_network: torch.nn.Module,
    rbf_layer: torch.nn.Module,
    cutoff: torch.nn.Module,
    cutoff_upper: float,
    num_knots: int,
) -> Tuple[torch.Tensor, float]:
    """``[num_knots, 3, F]`` table of ``cutoff(d) * filter_network(rbf(d))``
    over ``[0, cutoff_upper]``, built in float64 from the modules themselves
    (any radial basis, cutoff, activation or filter depth), and its relative
    interpolation error (:func:`hermite_relative_error`). Knots are placed on
    the cutoffs' breakpoints (:func:`aligned_num_knots`), so ``num_knots``
    may grow by up to 2x for a cutoff that declares any."""
    device = next(filter_network.parameters()).device
    filter_network, rbf_layer, cutoff = (
        copy.deepcopy(m).double() for m in (filter_network, rbf_layer, cutoff)
    )

    def phi(d):
        return cutoff(d).unsqueeze(-1) * filter_network(rbf_layer(d))

    breakpoints = [
        b
        for m in (cutoff, getattr(rbf_layer, "cutoff", None))
        if hasattr(m, "breakpoints")
        for b in m.breakpoints()
    ]
    num_knots = aligned_num_knots(cutoff_upper, num_knots, breakpoints)
    with torch.no_grad():
        table = hermite_table(phi, cutoff_upper, num_knots, device)
        return table, hermite_relative_error(phi, table, cutoff_upper)
