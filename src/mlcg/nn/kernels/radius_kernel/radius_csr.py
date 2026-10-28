import math
from typing import List, Tuple

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from ..pbc import cell_validity, minimum_image, minimum_image_vectors
from ..utils import ensure_contiguous

BLOCK_I = 16
BLOCK_J = 64
NUM_WARPS = 4


@triton.jit
def radius_csr_kernel(
    pos_ptr,
    batch_ptr,
    graph_ptr,
    row_ptr,
    col_ptr,
    num_edges_ptr,
    cell_ptr,
    pbc_ptr,
    cutoff_sq,
    N,
    capacity,
    WRITE: tl.constexpr,
    PBC: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    """One program per block of rows, looping over the rows' graphs.

    ``WRITE=False`` stores the row degrees in ``row_ptr[1:]``.
    ``WRITE=True`` reads the row offsets, writes each row's neighbours in
    increasing ``j`` and clamps ``row_ptr`` to ``capacity``.
    """
    pid = tl.program_id(0)
    first = pid * BLOCK_I
    i = first + tl.arange(0, BLOCK_I)
    i_ok = i < N
    last = tl.minimum(first + BLOCK_I, N) - 1
    j_start = tl.load(graph_ptr + tl.load(batch_ptr + first)).to(tl.int32)
    j_end = tl.load(graph_ptr + tl.load(batch_ptr + last) + 1).to(tl.int32)

    mol_i = tl.load(batch_ptr + i, mask=i_ok, other=-1)
    xi = tl.load(pos_ptr + i * 3 + 0, mask=i_ok, other=0.0)
    yi = tl.load(pos_ptr + i * 3 + 1, mask=i_ok, other=0.0)
    zi = tl.load(pos_ptr + i * 3 + 2, mask=i_ok, other=0.0)

    if WRITE:
        start = tl.load(row_ptr + i, mask=i_ok, other=0)
        cursor = start
    else:
        cursor = tl.zeros([BLOCK_I], dtype=tl.int32)

    for j0 in range(j_start, j_end, BLOCK_J):
        j = j0 + tl.arange(0, BLOCK_J)
        j_ok = j < j_end
        mol_j = tl.load(batch_ptr + j, mask=j_ok, other=-2)
        dx = tl.load(pos_ptr + j * 3 + 0, mask=j_ok)[None, :] - xi[:, None]
        dy = tl.load(pos_ptr + j * 3 + 1, mask=j_ok)[None, :] - yi[:, None]
        dz = tl.load(pos_ptr + j * 3 + 2, mask=j_ok)[None, :] - zi[:, None]
        if PBC:
            mol = tl.maximum(mol_i, 0)[:, None]
            dx, dy, dz = minimum_image(dx, dy, dz, cell_ptr, pbc_ptr, mol)
        is_edge = (
            (mol_i[:, None] == mol_j[None, :])
            & (i[:, None] != j[None, :])
            & (dx * dx + dy * dy + dz * dz < cutoff_sq)
        )
        hits = is_edge.to(tl.int32)
        if WRITE:
            slot = cursor[:, None] + tl.cumsum(hits, axis=1) - hits
            cols = tl.broadcast_to(j[None, :], [BLOCK_I, BLOCK_J])
            tl.store(col_ptr + slot, cols, mask=is_edge & (slot < capacity))
        cursor += tl.sum(hits, axis=1)

    if WRITE:
        tl.store(row_ptr + i, tl.minimum(start, capacity), mask=i_ok)
        if last == N - 1:
            total = tl.load(row_ptr + N)
            tl.store(num_edges_ptr, total)
            tl.store(row_ptr + N, tl.minimum(total, capacity))
    else:
        tl.store(row_ptr + 1 + i, cursor, mask=i_ok)
        if pid == 0:
            tl.store(row_ptr, 0)


@triton_op("mlcg_kernels::radius_csr", mutates_args={})
@ensure_contiguous
def radius_csr(
    pos: torch.Tensor,
    batch: torch.Tensor,
    ptr: torch.Tensor,
    cutoff: float,
    capacity: int,
    cell: torch.Tensor,
    pbc: torch.Tensor,
) -> List[torch.Tensor]:
    """Radius graph as a destination CSR, without host synchronization.

    Row ``i`` holds every ``j != i`` of the same graph with
    ``|pos[j] - pos[i]| < cutoff``, sorted by ``j``. Output shapes depend
    only on ``N`` and ``capacity``, so the op is CUDA-graph capturable.

    Parameters
    ----------
    pos : torch.Tensor
        ``(N, 3)`` float32 positions.
    batch : torch.Tensor
        ``(N,)`` sorted graph index of each node.
    ptr : torch.Tensor
        ``(B + 1,)`` graph offsets, ``ptr[b] = first node of graph b``.
    cutoff : float
        Distance cutoff (strict).
    capacity : int
        Number of ``col`` slots. Edges beyond it are dropped.
    cell, pbc : torch.Tensor
        ``(B, 9)`` reduced cells and ``(B, 3)`` periodic flags (float) of
        :func:`~mlcg.nn.kernels.pbc.periodic_cells`, or empty tensors for no
        periodic boundary conditions (minimum image, see that module).

    Returns
    -------
    row_ptr : torch.Tensor
        ``(N + 1,)`` int32 row offsets, clamped to ``capacity``.
    col : torch.Tensor
        ``(capacity,)`` int32 neighbour indices; ``col[row_ptr[N]:]`` is
        uninitialized.
    num_edges : torch.Tensor
        ``(1,)`` int32 true number of directed edges. The lists are
        complete iff ``num_edges <= capacity``.
    """
    N = pos.shape[0]
    row_ptr = torch.empty(N + 1, dtype=torch.int32, device=pos.device)
    col = torch.empty(capacity, dtype=torch.int32, device=pos.device)
    num_edges = torch.empty(1, dtype=torch.int32, device=pos.device)
    grid = (triton.cdiv(N, BLOCK_I),)
    for write in (False, True):
        if write:
            torch.cumsum(row_ptr, 0, out=row_ptr)
        wrap_triton(radius_csr_kernel)[grid](
            pos,
            batch,
            ptr,
            row_ptr,
            col,
            num_edges,
            cell,
            pbc,
            cutoff * cutoff,
            N,
            capacity,
            WRITE=write,
            PBC=cell.numel() > 0,
            BLOCK_I=BLOCK_I,
            BLOCK_J=BLOCK_J,
            num_warps=NUM_WARPS,
        )
    return [row_ptr, col, num_edges]


@radius_csr.register_kernel("cpu")
def _(pos, batch, ptr, cutoff, capacity, cell, pbc):
    N = pos.shape[0]
    graph = batch[:, None].expand(N, N).reshape(-1)
    d = (pos[None, :, :] - pos[:, None, :]).reshape(-1, 3)
    d = minimum_image_vectors(d, cell, pbc, graph).reshape(N, N, 3)
    is_edge = (
        (batch[:, None] == batch[None, :])
        & ~torch.eye(N, dtype=torch.bool)
        & ((d * d).sum(-1) < cutoff * cutoff)
    )
    cols = is_edge.nonzero()[:, 1].to(torch.int32)
    row_ptr = torch.zeros(N + 1, dtype=torch.int32)
    row_ptr[1:] = is_edge.sum(1).cumsum(0)
    col = torch.zeros(capacity, dtype=torch.int32)
    kept = min(capacity, cols.numel())
    col[:kept] = cols[:kept]
    num_edges = torch.tensor([cols.numel()], dtype=torch.int32)
    return [row_ptr.clamp(max=capacity), col, num_edges]


@radius_csr.register_fake
def _(pos, batch, ptr, cutoff, capacity, cell, pbc):
    N = pos.shape[0]
    return [
        pos.new_empty(N + 1, dtype=torch.int32),
        pos.new_empty(capacity, dtype=torch.int32),
        pos.new_empty(1, dtype=torch.int32),
    ]


def max_edges(ptr: torch.Tensor) -> int:
    """Number of directed pairs inside the graphs, the largest edge count."""
    sizes = ptr.diff()
    return int((sizes * (sizes - 1)).sum())


class RadiusCSR(torch.nn.Module):
    """Radius graph as a destination CSR, with a capacity that never drops
    edges silently.

    The capacity is reserved (one host sync) whenever the number of nodes
    changes. When all intra-graph pairs fit ``all_pairs_budget`` edges the
    capacity is all pairs and no edge can be dropped; otherwise it is the
    exact count plus ``margin``, and every call checks the count (one host
    sync) and regrows. The returned ``valid`` factor is 1, or NaN if the
    lists were truncated (e.g. a new batch with the same number of nodes
    but larger graphs), to multiply into the energy so that it and the
    forces can never be wrong unnoticed.
    """

    def __init__(
        self,
        cutoff: float,
        margin: float = 0.25,
        align: int = 1024,
        all_pairs_budget: int = 1 << 25,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.margin = margin
        self.align = align
        self.all_pairs_budget = all_pairs_budget
        self.num_nodes = -1
        self.capacity = 0
        self.check_every_call = False

    @torch.compiler.disable
    def reserve(self, pos, batch, ptr, cell, pbc, num_edges=None):
        limit = max_edges(ptr)
        self.num_nodes = pos.shape[0]
        self.check_every_call = limit > self.all_pairs_budget
        if not self.check_every_call:
            self.capacity = max(1, limit)
            return
        if num_edges is None:
            num_edges = radius_csr(pos, batch, ptr, self.cutoff, 0, cell, pbc)[
                2
            ]
        padded = math.ceil(int(num_edges) * (1 + self.margin) / self.align)
        self.capacity = max(1, min(padded * self.align, limit))

    @torch.compiler.disable
    def _grow_if_needed(self, pos, batch, ptr, cell, pbc, num_edges) -> bool:
        if int(num_edges) <= self.capacity:
            return False
        self.reserve(pos, batch, ptr, cell, pbc, num_edges)
        return True

    def forward(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor,
        ptr: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``cell``, ``pbc``: from :func:`~mlcg.nn.kernels.pbc.periodic_cells`
        (empty without periodic boundary conditions). An unsupported cell
        also makes ``valid`` NaN."""
        if pos.shape[0] != self.num_nodes:
            self.reserve(pos, batch, ptr, cell, pbc)
        row_ptr, col, num_edges = radius_csr(
            pos, batch, ptr, self.cutoff, self.capacity, cell, pbc
        )
        if self.check_every_call and self._grow_if_needed(
            pos, batch, ptr, cell, pbc, num_edges
        ):
            return self.forward(pos, batch, ptr, cell, pbc)
        ok = num_edges <= self.capacity
        if cell.numel():
            ok = ok & cell_validity(cell, pbc, self.cutoff).all()
        valid = torch.where(ok, 1.0, float("nan"))
        return row_ptr, col, valid.to(pos.dtype)
