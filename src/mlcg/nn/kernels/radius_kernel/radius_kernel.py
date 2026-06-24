import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton
from triton.language.extra import libdevice
from typing import List

from ..utils import ensure_contiguous


def _reset_count(args):
    """Reset global_count_ptr to zero before each autotune trial."""
    args["global_count_ptr"].zero_()


def _make_config(block_i, block_j, num_warps, num_stages):
    return triton.Config(
        {"BLOCK_I": block_i, "BLOCK_J": block_j},
        num_warps=num_warps,
        num_stages=num_stages,
        pre_hook=_reset_count,
    )


@triton.autotune(
    configs=[
        _make_config(16, 16, 2, 1),
        _make_config(32, 32, 4, 1),
        _make_config(16, 128, 4, 1),
        _make_config(16, 256, 8, 1),
        _make_config(32, 128, 4, 1),
        _make_config(32, 64, 4, 1),
        _make_config(64, 32, 4, 1),
        _make_config(64, 64, 8, 1),
        _make_config(64, 128, 4, 1),
        _make_config(128, 64, 4, 1),
        _make_config(128, 32, 4, 1),
        _make_config(128, 128, 8, 1),
        _make_config(256, 64, 8, 1),
        _make_config(64, 256, 8, 1),
    ],
    key=[],
)
@triton.jit
def radius_kernel(
    pos_ptr,  # (N, 3) float32
    batch_ptr,  # (N,)   int32
    cell_ptr,  # (M, 3, 3) float32
    cell_inv_ptr,  # (M, 3, 3) float32
    need_pbc_ptr,  # (M, 3) bool
    out_src_ptr,  # (max_out,) int64
    out_dst_ptr,  # (max_out,) int64
    out_dist_ptr,  # (max_out,) float32
    out_disp_ptr,  # (max_out, 3) float32 — only used when return_displacements
    global_count_ptr,  # (1,) int64 — atomic write counter
    cutoff_sq: tl.constexpr,
    max_out: int,  # Size of the allocated output
    N: int,  # Number of atoms (positions.shape[0])
    compute_self_loops: tl.constexpr,
    apply_pbc: tl.constexpr,
    return_displacements: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)

    i_off = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    j_off = pid_i * BLOCK_I + pid_j * BLOCK_J + tl.arange(0, BLOCK_J)

    i_ok = i_off < N
    j_ok = j_off < N

    mol_i = tl.load(batch_ptr + i_off, mask=i_ok, other=-1)
    mol_j = tl.load(batch_ptr + j_off, mask=j_ok, other=-2)
    active = (mol_i[:, None] == mol_j[None, :]) & i_ok[:, None] & j_ok[None, :]

    if compute_self_loops:
        active = active & (j_off[None, :] >= i_off[:, None])
    else:
        active = active & (j_off[None, :] > i_off[:, None])

    px_i = tl.load(
        pos_ptr + i_off * 3 + 0,
        mask=i_ok,
        other=0.0,
        eviction_policy="evict_last",
    )
    py_i = tl.load(
        pos_ptr + i_off * 3 + 1,
        mask=i_ok,
        other=0.0,
        eviction_policy="evict_last",
    )
    pz_i = tl.load(
        pos_ptr + i_off * 3 + 2,
        mask=i_ok,
        other=0.0,
        eviction_policy="evict_last",
    )
    px_j = tl.load(
        pos_ptr + j_off * 3 + 0,
        mask=j_ok,
        other=0.0,
        eviction_policy="evict_first",
    )
    py_j = tl.load(
        pos_ptr + j_off * 3 + 1,
        mask=j_ok,
        other=0.0,
        eviction_policy="evict_first",
    )
    pz_j = tl.load(
        pos_ptr + j_off * 3 + 2,
        mask=j_ok,
        other=0.0,
        eviction_policy="evict_first",
    )

    dx = px_j[None, :] - px_i[:, None]
    dy = py_j[None, :] - py_i[:, None]
    dz = pz_j[None, :] - pz_i[:, None]

    if apply_pbc:
        H_inv00 = tl.load(cell_inv_ptr + mol_i * 9 + 0, mask=i_ok, other=0.0)
        H_inv01 = tl.load(cell_inv_ptr + mol_i * 9 + 1, mask=i_ok, other=0.0)
        H_inv02 = tl.load(cell_inv_ptr + mol_i * 9 + 2, mask=i_ok, other=0.0)
        H_inv10 = tl.load(cell_inv_ptr + mol_i * 9 + 3, mask=i_ok, other=0.0)
        H_inv11 = tl.load(cell_inv_ptr + mol_i * 9 + 4, mask=i_ok, other=0.0)
        H_inv12 = tl.load(cell_inv_ptr + mol_i * 9 + 5, mask=i_ok, other=0.0)
        H_inv20 = tl.load(cell_inv_ptr + mol_i * 9 + 6, mask=i_ok, other=0.0)
        H_inv21 = tl.load(cell_inv_ptr + mol_i * 9 + 7, mask=i_ok, other=0.0)
        H_inv22 = tl.load(cell_inv_ptr + mol_i * 9 + 8, mask=i_ok, other=0.0)
        H00 = tl.load(cell_ptr + mol_i * 9 + 0, mask=i_ok, other=0.0)
        H01 = tl.load(cell_ptr + mol_i * 9 + 1, mask=i_ok, other=0.0)
        H02 = tl.load(cell_ptr + mol_i * 9 + 2, mask=i_ok, other=0.0)
        H10 = tl.load(cell_ptr + mol_i * 9 + 3, mask=i_ok, other=0.0)
        H11 = tl.load(cell_ptr + mol_i * 9 + 4, mask=i_ok, other=0.0)
        H12 = tl.load(cell_ptr + mol_i * 9 + 5, mask=i_ok, other=0.0)
        H20 = tl.load(cell_ptr + mol_i * 9 + 6, mask=i_ok, other=0.0)
        H21 = tl.load(cell_ptr + mol_i * 9 + 7, mask=i_ok, other=0.0)
        H22 = tl.load(cell_ptr + mol_i * 9 + 8, mask=i_ok, other=0.0)
        pbc_x = tl.load(need_pbc_ptr + mol_i * 3 + 0, mask=i_ok, other=0).to(
            tl.int1
        )
        pbc_y = tl.load(need_pbc_ptr + mol_i * 3 + 1, mask=i_ok, other=0).to(
            tl.int1
        )
        pbc_z = tl.load(need_pbc_ptr + mol_i * 3 + 2, mask=i_ok, other=0).to(
            tl.int1
        )

        # Cartesian to fractional
        fx = (
            H_inv00[:, None] * dx
            + H_inv01[:, None] * dy
            + H_inv02[:, None] * dz
        )
        fy = (
            H_inv10[:, None] * dx
            + H_inv11[:, None] * dy
            + H_inv12[:, None] * dz
        )
        fz = (
            H_inv20[:, None] * dx
            + H_inv21[:, None] * dy
            + H_inv22[:, None] * dz
        )

        # Minimum image in fractional space
        fx = tl.where(pbc_x[:, None], fx - libdevice.round(fx), fx)
        fy = tl.where(pbc_y[:, None], fy - libdevice.round(fy), fy)
        fz = tl.where(pbc_z[:, None], fz - libdevice.round(fz), fz)

        # Fractional to cartesian
        dx = H00[:, None] * fx + H01[:, None] * fy + H02[:, None] * fz
        dy = H10[:, None] * fx + H11[:, None] * fy + H12[:, None] * fz
        dz = H20[:, None] * fx + H21[:, None] * fy + H22[:, None] * fz

    dist_sq = dx * dx + dy * dy + dz * dz
    is_edge = active & (dist_sq < cutoff_sq)

    is_flat = tl.reshape(is_edge, [BLOCK_I * BLOCK_J])
    n_out = tl.sum(is_flat.to(tl.int32))
    if n_out == 0:
        return

    base = tl.atomic_add(global_count_ptr, n_out.to(tl.int64))
    rank = tl.cumsum(is_flat.to(tl.int32), axis=0) - is_flat.to(tl.int32)
    slots = base + rank.to(tl.int64)
    write = is_flat & (slots < max_out)

    src_flat = tl.reshape(
        tl.broadcast_to(j_off[None, :], [BLOCK_I, BLOCK_J]), [BLOCK_I * BLOCK_J]
    )
    dst_flat = tl.reshape(
        tl.broadcast_to(i_off[:, None], [BLOCK_I, BLOCK_J]), [BLOCK_I * BLOCK_J]
    )
    dsq_flat = tl.reshape(dist_sq, [BLOCK_I * BLOCK_J])

    # FIXME: check if here cast to int64 is ok or if we need to keep it also internally
    tl.store(out_src_ptr + slots, src_flat.to(tl.int64), mask=write)
    tl.store(out_dst_ptr + slots, dst_flat.to(tl.int64), mask=write)
    tl.store(out_dist_ptr + slots, tl.sqrt(dsq_flat), mask=write)
    if return_displacements:
        tl.store(
            out_disp_ptr + slots * 3 + 0,
            tl.reshape(dx, [BLOCK_I * BLOCK_J]),
            mask=write,
        )
        tl.store(
            out_disp_ptr + slots * 3 + 1,
            tl.reshape(dy, [BLOCK_I * BLOCK_J]),
            mask=write,
        )
        tl.store(
            out_disp_ptr + slots * 3 + 2,
            tl.reshape(dz, [BLOCK_I * BLOCK_J]),
            mask=write,
        )


@triton_op("mlcg_kernels::radius", mutates_args={})
@ensure_contiguous
def radius(
    pos: torch.Tensor,
    batch: torch.Tensor,
    cutoff: float,
    pbc: torch.Tensor = None,
    cell: torch.Tensor = None,
    avg_max_num_neigh: int = 32,
    self_loops: bool = False,
    return_displacements: bool = False,
) -> List[torch.Tensor]:
    """Build a radius graph using a Triton kernel, returning full bidirectional edges.

    For each pair of atoms (i, j) in the same batch that are within ``cutoff``
    distance, both directed edges (i→j) and (j→i) are included in the output.
    The kernel operates on upper-triangular atom pairs and then mirrors them to
    produce a symmetric edge list.

    Parameters
    ----------
    pos : torch.Tensor
        Atom positions of shape ``(N, 3)``, where N is the total number of atoms
        across all systems in the batch. Must be ``float32`` and contiguous.
    batch : torch.Tensor
        Batch assignment vector of shape ``(N,)`` mapping each atom to its
        system index. Values must be non-negative ordered integers in ``[0, B-1]``,
        where B is the number of systems in the batch.
    cutoff : float
        Distance cutoff in the same units as ``pos``. Pairs with distance
        strictly less than ``cutoff`` are included.
    pbc : torch.Tensor, optional
        Boolean tensor of shape ``(3,)`` indicating which Cartesian directions
        have periodic boundary conditions (True = periodic). Must be provided
        together with ``cell``; ignored if ``cell`` is None.
    cell : torch.Tensor, optional
        Unit cell matrix of shape ``(3, 3)`` whose rows are the lattice
        vectors. Must be provided together with ``pbc``; ignored if ``pbc``
        is None. When both are given, periodic minimum-image distances are
        computed via the fractional-coordinate convention.
    avg_max_num_neigh : int, optional
        Estimated upper bound on the number of neighbours per atom, used
        exclusively to size the output buffer: ``max_out = N * avg_max_num_neigh``
        edge slots are pre-allocated. It is not a hard per-atom cap — the
        kernel raises a ``RuntimeError`` only if the *total* edge count
        exceeds ``max_out``. Set it to a value slightly above the expected
        maximum neighbour count for your system to avoid reallocation.
        Default: ``32``.
    self_loops : bool, optional
        If True, include edges from each atom to itself (distance 0). These
        are not duplicated when mirroring. Default: ``False``.
    return_displacements : bool, optional
        If True, also return the displacement vectors ``r_j - r_i`` (wrapped
        into the minimum image when PBC is active) for each edge.
        Default: ``False``.

    Returns
    -------
    edge_index : torch.Tensor
        Long tensor of shape ``(2, E)`` containing source and destination atom
        indices for each of the E directed edges. Row 0 is the source, row 1
        is the destination.
    distances : torch.Tensor
        Float32 tensor of shape ``(E,)`` with the Euclidean distance (minimum
        image distance under PBC) for each edge.
    displacements : torch.Tensor
        Float32 tensor of shape ``(E, 3)`` with the displacement vector
        ``pos[dst] - pos[src]`` for each edge (minimum-image corrected when
        PBC is active). Only present in the return list when
        ``return_displacements=True``.

    Raises
    ------
    RuntimeError
        If the total number of discovered edges exceeds
        ``N * avg_max_num_neigh``. Increase ``avg_max_num_neigh`` to resolve.
    AssertionError
        If ``pos`` is not a 2-D tensor with exactly 3 columns.

    Notes
    -----
    - The kernel autotuned over ``BLOCK_I`` × ``BLOCK_J`` tile sizes; the
      best config is cached per (N, cutoff, device) configuration.
    - All atoms in a batch element are assumed to be contiguous in ``pos``
      (i.e. ``batch`` is sorted). Cross-system pairs are never considered.
    - Gradients flow through ``pos`` via a registered backward pass; all
      other arguments are treated as non-differentiable.

    Examples
    --------
    Simple non-periodic graph for a single system:

    >>> pos = torch.randn(100, 3, device="cuda")
    >>> batch = torch.zeros(100, dtype=torch.long, device="cuda")
    >>> edge_index, distances = radius(pos, batch, cutoff=5.0)
    >>> edge_index.shape  # (2, E)

    With PBC and displacement vectors:

    >>> cell = torch.eye(3, device="cuda") * 10.0   # 10 Å cubic box
    >>> pbc = torch.tensor([True, True, True], device="cuda")
    >>> edge_index, distances, displacements = radius(
    ...     pos, batch, cutoff=5.0, cell=cell, pbc=pbc,
    ...     return_displacements=True,
    ... )

    Batched systems:

    >>> pos = torch.randn(300, 3, device="cuda")
    >>> batch = torch.repeat_interleave(torch.arange(3, device="cuda"), 100)
    >>> edge_index, distances = radius(pos, batch, cutoff=3.0, avg_max_num_neigh=64)
    """
    assert pos.ndim == 2 and pos.shape[1] == 3
    N = pos.shape[0]
    _batch = batch.to(torch.int32).contiguous()
    M_max = int(torch.bincount(batch).max().item())
    max_out = N * avg_max_num_neigh

    if (cell is not None) and (pbc is not None):
        apply_pbc = True
        _pbc = pbc.to(torch.bool).contiguous()
        _cell = cell
        inv_cell = torch.linalg.inv(cell)
    else:
        apply_pbc = False
        _cell = torch.empty(0, device=pos.device)
        _pbc = torch.empty(0, device=pos.device)
        inv_cell = torch.empty(0, device=pos.device)

    out_src = torch.empty(max_out, dtype=torch.int64, device=pos.device)
    out_dst = torch.empty(max_out, dtype=torch.int64, device=pos.device)
    out_dist = torch.empty(max_out, dtype=torch.float32, device=pos.device)
    out_disp = (
        torch.empty((max_out, 3), dtype=torch.float32, device=pos.device)
        if return_displacements
        else torch.empty(0, device=pos.device)
    )
    global_count = torch.zeros(1, dtype=torch.int64, device=pos.device)

    def grid(META):
        return (
            triton.cdiv(N, META["BLOCK_I"]),
            triton.cdiv(M_max + META["BLOCK_I"] - 1, META["BLOCK_J"]),
        )

    wrap_triton(radius_kernel)[grid](
        pos,
        _batch,
        _cell,
        inv_cell,
        _pbc,
        out_src,
        out_dst,
        out_dist,
        out_disp,
        global_count,
        cutoff_sq=cutoff * cutoff,
        max_out=max_out,
        N=N,
        compute_self_loops=self_loops,
        apply_pbc=apply_pbc,
        return_displacements=return_displacements,
    )

    E_half = int(global_count.item())
    if E_half > max_out:
        raise RuntimeError(
            f"radius_edges: output buffer overflow: {E_half} edges found but "
            f"available are N_atoms*avg_max_num_neigh={max_out}. Increase avg_max_num_neigh."
        )
    src = out_src[:E_half]
    dst = out_dst[:E_half]
    dist_half = out_dist[:E_half]

    # Mirror the upper-triangular half to get full bidirectional edges.
    # Self-loops (diagonal tile, src == dst) must not be duplicated.
    if self_loops:
        is_self = src == dst
        edge_index = torch.stack(
            [torch.cat([src, dst[~is_self]]), torch.cat([dst, src[~is_self]])],
            dim=0,
        )
        distances = torch.cat([dist_half, dist_half[~is_self]])
        if return_displacements:
            d = out_disp[:E_half]
            return [edge_index, distances, torch.cat([d, -d[~is_self]])]
    else:
        edge_index = torch.stack(
            [torch.cat([src, dst]), torch.cat([dst, src])], dim=0
        )
        distances = dist_half.repeat(2)
        if return_displacements:
            d = out_disp[:E_half]
            return [edge_index, distances, torch.cat([d, -d])]

    return [edge_index, distances]


def setup_context(ctx, inputs, output):
    (
        pos,
        batch,
        cutoff,
        pbc,
        cell,
        avg_max_num_neigh,
        self_loops,
        return_displacements,
    ) = inputs
    edge_index = output[0]
    distances = output[1]
    ctx.had_pbc = pbc is not None and pbc.numel() > 0
    ctx.had_cell = cell is not None and cell.numel() > 0
    ctx.return_displacements = return_displacements
    displacements = (
        output[2] if return_displacements else torch.empty(0)
    )
    ctx.save_for_backward(pos, edge_index, distances, displacements)


def backward(ctx, grads):
    grad_distances = grads[1]
    grad_displacements = grads[2] if ctx.return_displacements else 0
    grad_pos = None

    if ctx.needs_input_grad[0]:
        pos, edge_index, distances, displacements = ctx.saved_tensors
        if not ctx.return_displacements:
            displacements = pos[edge_index[0]] - pos[edge_index[1]]

        unit_vec = torch.div(displacements, distances.unsqueeze(1))
        expanded_pos_grad = (
            grad_distances.unsqueeze(1) * unit_vec + grad_displacements
        )

        grad_pos = torch.zeros_like(pos).index_add(0, edge_index[0], expanded_pos_grad)
        grad_pos = grad_pos.index_add(0, edge_index[1], -expanded_pos_grad)

        # grad_pos = torch.zeros_like(pos)
        # grad_pos.index_add_(0, edge_index[0], expanded_pos_grad)
        # grad_pos.index_add_(0, edge_index[1], -expanded_pos_grad)

    return (grad_pos, None, None, None, None, None)
    # grads_out = [
    #     grad_pos,
    #     None,   # batch
    #     None,   # cutoff
    # ]
    # if ctx.had_pbc:
    #     grads_out.append(None)  # pbc
    # if ctx.had_cell:
    #     grads_out.append(None)  # cell
    # grads_out.extend([
    #     None,   # avg_max_num_neigh
    #     None,   # self_loops
    #     None,   # return_displacements
    # ])
    # return tuple(grads_out)


radius.register_autograd(backward, setup_context=setup_context)


@radius.register_kernel("cpu")
def _(
    pos: torch.Tensor,
    batch: torch.Tensor,
    cutoff: float,
    pbc: torch.Tensor = None,
    cell: torch.Tensor = None,
    avg_max_num_neigh: int = 32,
    self_loops: bool = False,
    return_displacements: bool = False,
) -> List[torch.Tensor]:
    N = pos.shape[0]
    cutoff_sq = cutoff * cutoff
    apply_pbc = (cell is not None) and (pbc is not None)
    if apply_pbc:
        pbc = pbc.to(torch.bool)
        inv_cell = torch.linalg.inv(cell)

    d = pos.unsqueeze(0) - pos.unsqueeze(1)
    if apply_pbc:
        H = cell[batch].unsqueeze(0)
        H_inv = inv_cell[batch].unsqueeze(0)
        pbc_n = pbc[batch].unsqueeze(0)
        f = torch.einsum("ijc,ijcd->ijd", d, H_inv.expand(N, N, 3, 3))
        f = torch.where(pbc_n.expand(N, N, 3), f - torch.round(f), f)
        d = torch.einsum("ijc,ijcd->ijd", f, H.expand(N, N, 3, 3))

    dist_sq = (d * d).sum(dim=-1)
    same_mol = batch.unsqueeze(0) == batch.unsqueeze(1)
    if not self_loops:
        same_mol = same_mol & ~torch.eye(N, dtype=torch.bool, device=pos.device)

    valid = same_mol & (dist_sq < cutoff_sq)
    max_out = pos.shape[0] * avg_max_num_neigh
    E_half = int(valid.triu().sum().item())
    if E_half > max_out:
        raise RuntimeError(
            f"radius_edges: output buffer overflow: {E_half} edges found but "
            f"available are N_atoms*avg_max_num_neigh={max_out}. Increase avg_max_num_neigh."
        )

    src, dst = valid.nonzero(as_tuple=True)
    distances = torch.sqrt(dist_sq[src, dst])
    edge_index = torch.stack([src, dst], dim=0)

    if return_displacements:
        return [edge_index, distances, d[dst, src]]
    return [edge_index, distances]
