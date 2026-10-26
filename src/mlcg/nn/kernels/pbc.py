"""
Periodic boundary conditions for the kernels that recompute displacements
from positions (``radius_csr``, ``tabulated_cfconv``).

Cells follow OpenMM's reduced triclinic form, lattice vectors as rows:
``a = (ax, 0, 0)``, ``b = (bx, by, 0)``, ``c = (cx, cy, cz)`` with
``|bx|, |cx| <= ax / 2`` and ``|cy| <= by / 2``. The minimum image is applied
as in OpenMM (``c``, then ``b``, then ``a``), which is exact when every
periodic width (``ax``, ``by``, ``cz``) is at least twice the cutoff.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

_TOL = 1e-6


def periodic_cells(
    cell: Optional[torch.Tensor],
    pbc: Optional[torch.Tensor],
    num_graphs: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(cell [B, 9], pbc [B, 3])`` float32 per graph for the kernels, or
    two empty tensors without a cell. A single cell is shared by all graphs;
    ``pbc`` defaults to periodic on every axis."""
    if cell is None:
        empty = torch.empty(0, device=device)
        return empty, empty
    cell = cell.reshape(-1, 9).float().expand(num_graphs, 9).contiguous()
    if pbc is None:
        pbc = torch.ones(num_graphs, 3, device=cell.device)
    pbc = pbc.reshape(-1, 3).float().expand(num_graphs, 3).contiguous()
    return cell, pbc


def cell_validity(
    cell: torch.Tensor, pbc: torch.Tensor, cutoff: float
) -> torch.Tensor:
    """Per graph: True if the cell is in reduced form and every periodic
    width is at least ``2 * cutoff`` (``cell [B, 9]``, ``pbc [B, 3]``)."""
    ax, bx, by, cx, cy, cz = (cell[:, k] for k in (0, 3, 4, 6, 7, 8))
    upper = torch.cat([cell[:, 1:3], cell[:, 5:6]], dim=1)
    upper_zero = upper.abs().amax(dim=1) <= _TOL * ax.abs()
    reduced = (
        upper_zero
        & (bx.abs() <= ax / 2 + _TOL)
        & (cx.abs() <= ax / 2 + _TOL)
        & (cy.abs() <= by / 2 + _TOL)
    )
    wide = (torch.stack([ax, by, cz], dim=1) >= 2 * cutoff) | (pbc == 0)
    return reduced & wide.all(dim=1)


def check_cells(cell, pbc, cutoff: float) -> None:
    """Host-side check, raises ``ValueError`` for an unsupported cell."""
    if cell is None:
        return
    num_graphs = cell.reshape(-1, 9).shape[0]
    cell, pbc = periodic_cells(cell, pbc, num_graphs, cell.device)
    if cell.numel() and not bool(cell_validity(cell, pbc, cutoff).all()):
        raise ValueError(
            "Periodic cells must be in OpenMM's reduced form with every "
            f"periodic width >= 2 * cutoff = {2 * cutoff}"
        )


def minimum_image_vectors(
    d: torch.Tensor, cell: torch.Tensor, pbc: torch.Tensor, graph: torch.Tensor
) -> torch.Tensor:
    """Differentiable torch twin of :func:`minimum_image` for displacements
    ``d [E, 3]`` of edges in graphs ``graph [E]`` (cells as returned by
    :func:`periodic_cells`; no-op without a cell). The lattice shifts are
    constants, so gradients of any order flow through ``d`` unchanged."""
    if cell.numel() == 0:
        return d
    cell, pbc = cell[graph], pbc[graph]
    for axis, vector in ((2, slice(6, 9)), (1, slice(3, 6)), (0, slice(0, 3))):
        length = cell[:, 4 * axis]
        shift = torch.floor(d[:, axis] / length + 0.5).detach()
        shift = torch.where(pbc[:, axis] > 0, shift, 0.0)
        d = d - shift[:, None] * cell[:, vector]
    return d


@triton.jit
def _wrap(dx, dy, dz, length, coord, vx, vy, vz, periodic):
    n = tl.where(periodic > 0, tl.floor(coord / length + 0.5), 0.0)
    return dx - n * vx, dy - n * vy, dz - n * vz


@triton.jit
def minimum_image(dx, dy, dz, cell_ptr, pbc_ptr, mol):
    """Minimum-image displacement in graph ``mol``'s reduced cell."""
    c = cell_ptr + mol * 9
    p = pbc_ptr + mol * 3
    cx, cy, cz = tl.load(c + 6), tl.load(c + 7), tl.load(c + 8)
    dx, dy, dz = _wrap(dx, dy, dz, cz, dz, cx, cy, cz, tl.load(p + 2))
    bx, by = tl.load(c + 3), tl.load(c + 4)
    dx, dy, dz = _wrap(dx, dy, dz, by, dy, bx, by, 0.0, tl.load(p + 1))
    ax = tl.load(c + 0)
    return _wrap(dx, dy, dz, ax, dx, ax, 0.0, 0.0, tl.load(p + 0))
