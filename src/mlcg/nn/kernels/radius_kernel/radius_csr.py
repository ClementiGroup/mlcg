import math
from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from ..models.schnet.cfconv import csr_from_edge_index
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
    excl_ptr,
    excl_col_ptr,
    cutoff_sq,
    N,
    capacity,
    WRITE: tl.constexpr,
    PBC: tl.constexpr,
    EXCLUDE: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    """One program per block of rows, looping over the rows' graphs.

    ``WRITE=False`` stores the row degrees in ``row_ptr[1:]``.
    ``WRITE=True`` reads the row offsets, writes each row's neighbours in
    increasing ``j`` and clamps ``row_ptr`` to ``capacity``.
    ``EXCLUDE`` drops the ``j`` listed in row ``i`` of the exclusion CSR.
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
    if EXCLUDE:
        excl_start = tl.load(excl_ptr + i, mask=i_ok, other=0)
        num_excl = tl.load(excl_ptr + i + 1, mask=i_ok, other=0) - excl_start
        max_excl = tl.max(num_excl)

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
        if EXCLUDE:
            for k in range(0, max_excl):
                excluded = tl.load(
                    excl_col_ptr + excl_start + k,
                    mask=i_ok & (k < num_excl),
                    other=-1,
                )
                is_edge &= excluded[:, None] != j[None, :]
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
    excl_ptr: torch.Tensor,
    excl_col: torch.Tensor,
) -> List[torch.Tensor]:
    """Radius graph as a destination CSR, without host synchronization.

    Row ``i`` holds every ``j != i`` of the same graph with
    ``|pos[j] - pos[i]| < cutoff`` that row ``i`` of the exclusions doesn't
    list, sorted by ``j``. Output shapes depend
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
    excl_ptr, excl_col : torch.Tensor
        ``(N + 1,)`` and ``(X,)`` int32 CSR of the excluded neighbours of each
        node (:func:`exclusion_csr`), or empty tensors for no exclusions.

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
            excl_ptr,
            excl_col,
            cutoff * cutoff,
            N,
            capacity,
            WRITE=write,
            PBC=cell.numel() > 0,
            EXCLUDE=excl_ptr.numel() > 0,
            BLOCK_I=BLOCK_I,
            BLOCK_J=BLOCK_J,
            num_warps=NUM_WARPS,
        )
    return [row_ptr, col, num_edges]


@radius_csr.register_kernel("cpu")
def _(pos, batch, ptr, cutoff, capacity, cell, pbc, excl_ptr, excl_col):
    N = pos.shape[0]
    graph = batch[:, None].expand(N, N).reshape(-1)
    d = (pos[None, :, :] - pos[:, None, :]).reshape(-1, 3)
    d = minimum_image_vectors(d, cell, pbc, graph).reshape(N, N, 3)
    is_edge = (
        (batch[:, None] == batch[None, :])
        & ~torch.eye(N, dtype=torch.bool)
        & ((d * d).sum(-1) < cutoff * cutoff)
    )
    if excl_ptr.numel():
        rows = torch.repeat_interleave(torch.arange(N), excl_ptr.diff())
        is_edge[rows, excl_col.long()] = False
    cols = is_edge.nonzero()[:, 1].to(torch.int32)
    row_ptr = torch.zeros(N + 1, dtype=torch.int32)
    row_ptr[1:] = is_edge.sum(1).cumsum(0)
    col = torch.zeros(capacity, dtype=torch.int32)
    kept = min(capacity, cols.numel())
    col[:kept] = cols[:kept]
    num_edges = torch.tensor([cols.numel()], dtype=torch.int32)
    return [row_ptr.clamp(max=capacity), col, num_edges]


def exclusion_csr(
    pairs: Optional[torch.Tensor], num_nodes: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(excl_ptr, excl_col)`` of :func:`radius_csr` from ``[2, X]``
    unordered pairs (each excludes both directions), or empty tensors for
    ``None``."""
    if pairs is None:
        empty = torch.empty(0, dtype=torch.int32, device=device)
        return empty, empty
    row_ptr, col, _ = csr_from_edge_index(
        torch.cat([pairs, pairs.flip(0)], dim=1), num_nodes
    )
    return row_ptr, col


def max_edges(ptr: torch.Tensor) -> int:
    """Number of directed pairs inside the graphs, the largest edge count."""
    sizes = ptr.diff()
    return int((sizes * (sizes - 1)).sum())


class RadiusCSR(torch.nn.Module):
    """Radius graph as a destination CSR, with a capacity that never drops
    edges silently.

    The capacity is the length of the ``col`` buffer. It is fixed between
    reservations, so the output shapes don't depend on the positions and a
    call needs no host sync (CUDA graphs, ``torch.compile``). It is
    reserved (one host sync) on the first call and whenever the number of
    nodes changes, in one of two modes:

    - **all pairs**, if the intra-graph pairs ``sum_b n_b (n_b - 1)`` fit
      in ``all_pairs_budget``: the capacity is all pairs. Every possible
      edge fits, so any configuration with the same graphs (e.g. a protein
      that folds during MD and gains edges) is exact, and no call checks
      the count.
    - **checked**, otherwise: the capacity is the current edge count plus
      ``margin``, rounded up to a multiple of ``align``, capped at all
      pairs. Every call reads the edge count (one host sync) and, if the
      buffer overflowed, reserves a larger capacity and searches again. The
      capacity never shrinks. The host sync rules out a CUDA graph, so
      :func:`~mlcg.nn.flash_models.capture_cuda_graph` falls back to the
      plain model.

    The returned ``valid`` factor is 1, or NaN if the result can't be
    trusted. It is multiplied into the energy, so that the energy and the
    forces become NaN instead of silently wrong. Nothing is repaired: a
    call reads no edge count or cell on the host. ``valid`` is NaN in two
    cases:

    - **Truncated lists.** Impossible in the checked mode, and in the
      all-pairs mode as long as the graphs stay the same (any configuration
      fits, e.g. a protein that folds). It happens only if a batch has the
      same number of nodes as the previous call but larger graphs. Call
      :meth:`reserve` with the new batch, or use a new instance.
    - **Unsupported cell**: not in reduced form, or a periodic width below
      ``2 * cutoff`` (:func:`~mlcg.nn.kernels.pbc.check_cells` raises with
      the reason). Use a box at least ``2 * cutoff`` wide.

    Neither case occurs in an mlcg simulation: the composition and the cell
    are fixed during a run, and an unsupported cell raises when the model
    is prepared. In ``mlcg_to_openmm``, ``set_box`` raises before a box
    below ``2 * cutoff`` is used.

    Parameters
    ----------
    cutoff : float
        Distance cutoff (strict).
    margin : float
        Checked mode: relative headroom over the edge count at reservation,
        so that a slowly growing edge count (e.g. folding) does not reserve
        again at every step.
    align : int
        Checked mode: the capacity is rounded up to a multiple of ``align``
        edges, so that nearby edge counts give the same buffer size.
    all_pairs_budget : int
        Largest number of intra-graph pairs, in edges, for which the
        all-pairs mode is used (``1 << 25``: 128 MB of int32 ``col``).
        ``0`` forces the checked mode.
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
        self.exclusions = None

    def store_exclusions(
        self, pairs: Optional[torch.Tensor], num_nodes: int = 0
    ):
        """Builds the exclusion CSR of ``pairs`` once, for the following
        calls with these pairs (fixed during a simulation); ``None`` drops
        it."""
        self.exclusions = (
            None
            if pairs is None
            else (
                pairs.clone(),
                num_nodes,
                *exclusion_csr(pairs, num_nodes, pairs.device),
            )
        )

    def _exclusions(self, pairs, num_nodes, device):
        """``(excl_ptr, excl_col)`` of ``pairs`` and, if the stored CSR is
        used, whether ``pairs`` differ from the stored ones (on the device).
        The stored CSR is used for pairs of its shape and number of nodes,
        otherwise the CSR is built now."""
        stored = self.exclusions
        if (
            pairs is None
            or stored is None
            or stored[0].shape != pairs.shape
            or stored[1] != num_nodes
        ):
            return exclusion_csr(pairs, num_nodes, device), None
        return stored[2:], (stored[0] != pairs).any()

    @torch.compiler.disable
    def reserve(self, pos, batch, ptr, cell, pbc, num_edges=None):
        limit = max_edges(ptr)
        self.num_nodes = pos.shape[0]
        self.check_every_call = limit > self.all_pairs_budget
        if not self.check_every_call:
            self.capacity = max(1, limit)
            return
        if num_edges is None:
            no_exclusions = exclusion_csr(None, pos.shape[0], pos.device)
            num_edges = radius_csr(
                pos, batch, ptr, self.cutoff, 0, cell, pbc, *no_exclusions
            )[2]
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
        excluded_pairs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``cell``, ``pbc``: from :func:`~mlcg.nn.kernels.pbc.periodic_cells`
        (empty without periodic boundary conditions). An unsupported cell
        also makes ``valid`` NaN. ``excluded_pairs``: ``[2, X]`` unordered
        pairs left out of the graph (the capacity is reserved without
        them); pairs of the shape of the stored ones (:meth:`store_exclusions`)
        but another content also make ``valid`` NaN."""
        if pos.shape[0] != self.num_nodes:
            self.reserve(pos, batch, ptr, cell, pbc)
        exclusions, changed = self._exclusions(
            excluded_pairs, pos.shape[0], pos.device
        )
        row_ptr, col, num_edges = radius_csr(
            pos, batch, ptr, self.cutoff, self.capacity, cell, pbc, *exclusions
        )
        if self.check_every_call and self._grow_if_needed(
            pos, batch, ptr, cell, pbc, num_edges
        ):
            return self.forward(pos, batch, ptr, cell, pbc, excluded_pairs)
        ok = num_edges <= self.capacity
        if changed is not None:
            ok = ok & ~changed
        if cell.numel():
            ok = ok & cell_validity(cell, pbc, self.cutoff).all()
        valid = torch.where(ok, 1.0, float("nan"))
        return row_ptr, col, valid.to(pos.dtype)
