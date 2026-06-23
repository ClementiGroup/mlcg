"""
Triton kernel: fixed-radius neighbor list construction.

Given:
    pos        : (N, 3)  float32  – atomic positions
    batch      : (N,)    int32    – molecule index per atom (contiguous/sorted)
    cutoff     : float            – distance threshold
    max_neigh  : int              – max neighbors stored per atom

Returns:
    edge_index : (2, E)  int64   – [src, dst] pairs (dst is the query atom)
    rel_pos    : (E, 3)  float32 – pos[src] - pos[dst]  (displacement vectors)
    num_neigh  : (N,)    int32   – actual neighbor count per atom
                                   (capped at max_neigh; use to detect overflow)
"""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton
from triton.language.extra import libdevice
from typing import List

from ..utils import ensure_contiguous


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 16}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK": 32}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK": 256}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK": 256}, num_warps=16, num_stages=4),
    ],
    key=[],
)
@triton.jit
def neighbor_list_kernel(
    pos_ptr,  # (N, 3) float32
    batch_ptr,  # (N,) int32
    mol_idxs_ptr,  # (M+1,) int32
    cell_ptr,  # (M, 3, 3) float32
    cell_inv_ptr,  # (M, 3, 3) float32
    need_pbc_ptr,  # (M, 3) bool
    neigh_idx_ptr,  # (N, max_neigh) int32  – neighbor atom indices (-1 = empty)
    num_neigh_ptr,  # (N,) int32  – count of valid neighbors
    squared_distances_ptr,  # (N, max_neigh) float32
    displacement_ptr,  # (N, max_neigh, 3) float32
    cutoff_sq: tl.constexpr,
    max_neigh: tl.constexpr,
    compute_self_loops: tl.constexpr,
    apply_pbc: tl.constexpr,
    return_displacements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Each program handles exactly one query atom."""
    atom_idx = tl.program_id(0)

    p1x = tl.load(pos_ptr + atom_idx * 3 + 0)
    p1y = tl.load(pos_ptr + atom_idx * 3 + 1)
    p1z = tl.load(pos_ptr + atom_idx * 3 + 2)

    mol = tl.load(batch_ptr + atom_idx)
    mol_start = tl.load(mol_idxs_ptr + mol)
    mol_end = tl.load(mol_idxs_ptr + mol + 1)

    if apply_pbc:
        cell_base = cell_ptr + mol * 9
        cell_inv_base = cell_inv_ptr + mol * 9

        H00 = tl.load(cell_base + 0)
        H01 = tl.load(cell_base + 1)
        H02 = tl.load(cell_base + 2)
        H10 = tl.load(cell_base + 3)
        H11 = tl.load(cell_base + 4)
        H12 = tl.load(cell_base + 5)
        H20 = tl.load(cell_base + 6)
        H21 = tl.load(cell_base + 7)
        H22 = tl.load(cell_base + 8)

        H_inv00 = tl.load(cell_inv_base + 0)
        H_inv01 = tl.load(cell_inv_base + 1)
        H_inv02 = tl.load(cell_inv_base + 2)
        H_inv10 = tl.load(cell_inv_base + 3)
        H_inv11 = tl.load(cell_inv_base + 4)
        H_inv12 = tl.load(cell_inv_base + 5)
        H_inv20 = tl.load(cell_inv_base + 6)
        H_inv21 = tl.load(cell_inv_base + 7)
        H_inv22 = tl.load(cell_inv_base + 8)

        pbc_x = tl.load(need_pbc_ptr + mol * 3 + 0).to(tl.int1)
        pbc_y = tl.load(need_pbc_ptr + mol * 3 + 1).to(tl.int1)
        pbc_z = tl.load(need_pbc_ptr + mol * 3 + 2).to(tl.int1)

    count = 0
    cand_base = mol_start

    for block_start in range(0, mol_end - mol_start, BLOCK):
        offsets = cand_base + block_start + tl.arange(0, BLOCK)
        mask = offsets < mol_end
        if not compute_self_loops:
            mask = mask & (offsets != atom_idx)

        p2x = tl.load(pos_ptr + offsets * 3 + 0, mask=mask, other=0.0)
        p2y = tl.load(pos_ptr + offsets * 3 + 1, mask=mask, other=0.0)
        p2z = tl.load(pos_ptr + offsets * 3 + 2, mask=mask, other=0.0)

        dx = p2x - p1x
        dy = p2y - p1y
        dz = p2z - p1z

        if apply_pbc:
            # cartesian to fractional: f = H_inv @ d
            fx = H_inv00 * dx + H_inv01 * dy + H_inv02 * dz
            fy = H_inv10 * dx + H_inv11 * dy + H_inv12 * dz
            fz = H_inv20 * dx + H_inv21 * dy + H_inv22 * dz

            # minimum image in fractional space
            fx = tl.where(pbc_x, fx - libdevice.round(fx), fx)
            fy = tl.where(pbc_y, fy - libdevice.round(fy), fy)
            fz = tl.where(pbc_z, fz - libdevice.round(fz), fz)

            # fractional to cartesian: d = H @ f
            dx = H00 * fx + H01 * fy + H02 * fz
            dy = H10 * fx + H11 * fy + H12 * fz
            dz = H20 * fx + H21 * fy + H22 * fz

        dist_sq = dx * dx + dy * dy + dz * dz
        is_neigh = mask & (dist_sq < cutoff_sq)

        # ── scatter valid neighbors into output ──────────────────────────────
        # Compute exclusive rank of each valid lane within this block via
        # prefix sum, then store all valid neighbors in one vectorized pass.
        is_neigh_i32 = is_neigh.to(tl.int32)
        rank = tl.cumsum(is_neigh_i32, axis=0) - is_neigh_i32  # 0-based rank
        within_budget = is_neigh & (rank + count < max_neigh)
        slots = atom_idx * max_neigh + count + rank
        tl.store(neigh_idx_ptr + slots, offsets, mask=within_budget)
        tl.store(squared_distances_ptr + slots, dist_sq, mask=within_budget)
        if return_displacements:
            tl.store(displacement_ptr + slots * 3 + 0, dx, mask=within_budget)
            tl.store(displacement_ptr + slots * 3 + 1, dy, mask=within_budget)
            tl.store(displacement_ptr + slots * 3 + 2, dz, mask=within_budget)
        count += tl.sum(within_budget.to(tl.int32))

    tl.store(num_neigh_ptr + atom_idx, count)


@triton_op("mlcg_kernels::radius", mutates_args={})
@ensure_contiguous
def radius(
    pos: torch.Tensor,  # (N, 3)  float32
    batch: torch.Tensor,  # (N,)    int32 / int64
    cutoff: float,
    pbc: torch.Tensor = None,
    cell: torch.Tensor = None,
    max_neigh: int = 32,
    self_loops: bool = False,
    return_displacements: bool = False,
) -> List[torch.Tensor]:
    """
    Add Docs
    """
    assert pos.ndim == 2 and pos.shape[1] == 3, "pos must be (N, 3)"

    N = pos.shape[0]
    M = batch.max() + 1

    _, counts = torch.unique(batch, sorted=True, return_counts=True)
    mol_idxs = torch.zeros(M + 1, device=pos.device, dtype=batch.dtype)
    mol_idxs[1:] = counts.cumsum(0)

    neigh_idx = torch.full(
        (N, max_neigh), -1, dtype=torch.int32, device=pos.device
    ).contiguous()
    num_neigh = torch.zeros(
        N, dtype=torch.int32, device=pos.device
    ).contiguous()
    squared_distances = torch.full(
        (N, max_neigh), -1, dtype=torch.float32, device=pos.device
    ).contiguous()

    if return_displacements:
        displacements = torch.zeros(
            (N, max_neigh, 3), dtype=torch.float32, device=pos.device
        ).contiguous()
    else:
        displacements = torch.empty(0)

    apply_pbc = False
    inv_cell = torch.empty(0)
    if (cell is not None) and (pbc is not None):
        apply_pbc = True
        pbc = pbc.to(torch.bool).contiguous()
        batch = batch.to(torch.int32).contiguous()
        inv_cell = torch.linalg.inv(cell)
    else:
        apply_pbc = False
        cell = torch.empty(0)
        pbc = torch.empty(0)
        inv_cell = torch.empty(0)

    grid = (N,)
    wrap_triton(neighbor_list_kernel)[grid](
        pos,
        batch,
        mol_idxs,
        cell,
        inv_cell,
        pbc,
        neigh_idx,
        num_neigh,
        squared_distances,
        displacements,
        cutoff_sq=cutoff * cutoff,
        max_neigh=max_neigh,
        compute_self_loops=self_loops,
        apply_pbc=apply_pbc,
        return_displacements=return_displacements,
    )

    valid_mask = neigh_idx >= 0
    dst_idx = (
        torch.arange(N, device=pos.device).unsqueeze(1).expand_as(neigh_idx)
    )

    src = neigh_idx[valid_mask].to(torch.int64)
    dst = dst_idx[valid_mask].to(torch.int64)
    edge_index = torch.stack([src, dst], dim=0)
    distances = torch.sqrt(squared_distances[valid_mask])

    if return_displacements:
        displacements = displacements[valid_mask]
        return [edge_index, distances, displacements]

    return [edge_index, distances]


def setup_context(ctx, inputs, output):
    (
        pos,
        batch,
        cutoff,
        pbc,
        cell,
        max_neigh,
        self_loops,
        return_displacements,
    ) = inputs
    edge_index = output[0]
    distances = output[1]
    ctx.return_displacements = return_displacements
    if ctx.return_displacements:
        displacements = output[2]
    else:
        displacements = torch.empty(0)
    ctx.apply_pbc = cell is not None and pbc is not None
    if ctx.apply_pbc:
        ctx.save_for_backward(
            pos, batch, cell, pbc, edge_index, distances, displacements
        )
    else:
        ctx.save_for_backward(pos, edge_index, distances, displacements)


def backward(ctx, grads):
    grad_distances = grads[1]
    grad_displacements = grads[2] if ctx.return_displacements else 0
    grad_pos = None

    if ctx.needs_input_grad[0]:
        if ctx.apply_pbc:
            pos, batch, cell, pbc, edge_index, distances, displacements = (
                ctx.saved_tensors
            )
        else:
            pos, edge_index, distances, displacements = ctx.saved_tensors
            cell = pbc = None

        src = edge_index[0]
        dst = edge_index[1]
        if not ctx.return_displacements:
            displacements = pos[src] - pos[dst]  # (E, 3), not yet PBC-corrected

            if ctx.apply_pbc:
                inv_cell = torch.linalg.inv(cell)
                batch_dst = batch[dst]
                inv_cell_e = inv_cell[batch_dst]  # (E, 3, 3)
                cell_e = cell[batch_dst]  # (E, 3, 3)
                pbc_e = pbc.bool()[batch_dst]  # (E, 3)
                f = torch.einsum(
                    "eij,ej->ei", inv_cell_e, displacements
                )  # fractional
                f = torch.where(pbc_e, f - torch.round(f), f)  # minimum image
                displacements = torch.einsum(
                    "eij,ej->ei", cell_e, f
                )  # back to cartesian

        unit_vec = displacements / distances.unsqueeze(1)  # (E, 3)
        weighted = (
            grad_distances.unsqueeze(1) * unit_vec + grad_displacements
        )  # (E, 3)

        grad_pos = torch.zeros_like(pos)
        grad_pos.index_add_(0, src, weighted)
        grad_pos.index_add_(0, dst, -weighted)

    return grad_pos, None, None, None, None, None, None, None


radius.register_autograd(backward, setup_context=setup_context)


@radius.register_kernel("cpu")
def _(
    pos: torch.Tensor,
    batch: torch.Tensor,
    cutoff: float,
    pbc: torch.Tensor = None,
    cell: torch.Tensor = None,
    max_neigh: int = 32,
    self_loops: bool = False,
    return_displacements: bool = False,
) -> List[torch.Tensor]:
    N = pos.shape[0]
    cutoff_sq = cutoff * cutoff

    apply_pbc = (cell is not None) and (pbc is not None)
    if apply_pbc:
        pbc = pbc.to(torch.bool)
        inv_cell = torch.linalg.inv(cell)  # (M, 3, 3)

    d = pos.unsqueeze(0) - pos.unsqueeze(1)

    if apply_pbc:
        H = cell[batch].unsqueeze(0)
        H_inv = inv_cell[batch].unsqueeze(0)
        pbc_n = pbc[batch].unsqueeze(0)

        # cartesian to fractional: f[i,j] = H_inv_dst[j] @ d[i,j]
        f = torch.einsum("ijc,ijcd->ijd", d, H_inv.expand(N, N, 3, 3))
        # minimum image
        f = torch.where(pbc_n.expand(N, N, 3), f - torch.round(f), f)
        # fractional to cartesian: d = H @ f
        d = torch.einsum("ijc,ijcd->ijd", f, H.expand(N, N, 3, 3))

    dist_sq = (d * d).sum(dim=-1)
    same_mol = batch.unsqueeze(0) == batch.unsqueeze(1)

    if not self_loops:
        not_self = ~torch.eye(N, dtype=torch.bool, device=pos.device)
        same_mol = same_mol & not_self

    valid = same_mol & (dist_sq < cutoff_sq)

    neigh_count = valid.sum(dim=0)
    if (neigh_count > max_neigh).any():
        rank = valid.cumsum(dim=0)
        valid = valid & (rank <= max_neigh)

    src, dst = valid.nonzero(as_tuple=True)
    distances = torch.sqrt(dist_sq[src, dst])
    edge_index = torch.stack([src, dst], dim=0)

    if return_displacements:
        return [edge_index, distances, d[dst, src]]

    return [edge_index, distances]
