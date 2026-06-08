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
    cutoff_sq: tl.constexpr,
    max_neigh: tl.constexpr,
    compute_self_loops: tl.constexpr,
    apply_pbc: tl.constexpr,
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
        # We do this sequentially inside the block (Triton doesn't support
        # dynamic scatter natively, so we loop over the BLOCK lanes).
        for lane in tl.static_range(BLOCK):
            if is_neigh[lane] and count < max_neigh:
                slot = atom_idx * max_neigh + count
                tl.store(neigh_idx_ptr + slot, offsets[lane])
                tl.store(squared_distances_ptr + slot, dist_sq[lane])
                count += 1

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
) -> dict:
    """
    Add Docs
    """
    assert pos.ndim == 2 and pos.shape[1] == 3, "pos must be (N, 3)"

    pbc = pbc.to(torch.bool).contiguous()
    batch = batch.to(torch.int32).contiguous()
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

    apply_pbc = False
    inv_cell = torch.empty(0)
    if (cell is not None) and (pbc is not None):
        apply_pbc = True
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
        cutoff_sq=cutoff * cutoff,
        max_neigh=max_neigh,
        compute_self_loops=self_loops,
        apply_pbc=apply_pbc,
    )

    valid_mask = neigh_idx >= 0
    dst_idx = (
        torch.arange(N, device=pos.device).unsqueeze(1).expand_as(neigh_idx)
    )

    src = neigh_idx[valid_mask].to(torch.int64)
    dst = dst_idx[valid_mask].to(torch.int64)
    edge_index = torch.stack([src, dst], dim=0)
    distances = torch.sqrt(squared_distances[valid_mask])

    return edge_index, distances
