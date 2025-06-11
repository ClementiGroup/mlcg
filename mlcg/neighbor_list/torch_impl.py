from typing import Tuple, Optional
import torch
from torch_geometric.data import Data
from torch_cluster import radius, radius_graph
from ..data.atomic_data import AtomicData


def torch_neighbor_list(
    data: Data,
    rcut: float,
    self_interaction: bool = True,
    num_workers: int = 1,
    max_num_neighbors: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Function for computing neighbor lists from pytorch geometric data
    instances that may or may not use periodic boundary conditions

    Parameters
    ----------
    data:
        Pytorch geometric data instance
    rcut:
        upper distance cutoff, in which neighbors with distances larger
        than this cutoff will be excluded.
    self_interaction:
        If True, each atom will be considered it's own neighbor. This can
        be convenient for certain bases and representations
    num_workers:
        Number of threads to spawn and use for computing the neighbor list
    max_number_neighbors
        kwarg for radius_graph function from torch_cluster package,
        specifying the maximum number of neighbors for each atom

    Returns
    -------
    torch.Tensor:
        The atom indices of the first atoms in each neighbor pair
    torch.Tensor:
        The atom indices of the second atoms in each neighbor pair
    torch.Tensor:
        The cell shifts associated with minimum image distances
        in the presence of periodic boundary conditions
    torch.Tensor:
        Mask for excluding self interactions
    """

    if "pbc" in data:
        pbc = data.pbc
    else:
        pbc = torch.zeros(3, dtype=bool, device=data.pos.device)

    if torch.any(pbc):
        if "cell" not in data:
            raise ValueError(
                f"Periodic systems need to have a unit cell defined"
            )
        (
            idx_i,
            idx_j,
            cell_shifts_2d,
            self_interaction_mask,
        ) = torch_neighbor_list_pbc(
            data,
            rcut,
            self_interaction=self_interaction,
            num_workers=num_workers,
            max_num_neighbors=max_num_neighbors,
        )
        # Convert cell_shift to match the shape: (n_edges, 3, 2)
        cell_shifts = torch.zeros(
            (idx_i.shape[0], 3, 2), 
            dtype=data.pos.dtype, 
            device=data.pos.device
        )
        cell_shifts[:, :, 1] = cell_shifts_2d
    else:
        idx_i, idx_j, self_interaction_mask = torch_neighbor_list_no_pbc(
            data,
            rcut,
            self_interaction=self_interaction,
            num_workers=num_workers,
            max_num_neighbors=max_num_neighbors,
        )
        # Cell shift must assume the shape of (n_edges, 3, 2) zeros
        cell_shifts = torch.zeros(
            (idx_i.shape[0], 3, 2), dtype=data.pos.dtype, device=data.pos.device
        )

    return idx_i, idx_j, cell_shifts, self_interaction_mask


@torch.jit.script
def compute_images(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    batch: torch.Tensor,
    n_atoms: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """TODO: add doc"""

    cell = cell.view((-1, 3, 3))
    pbc = pbc.view((-1, 3))
    reciprocal_cell = torch.linalg.inv(cell).transpose(2, 1)
    # print('reciprocal_cell: ', reciprocal_cell.device)
    inv_distances = reciprocal_cell.norm(2, dim=-1)
    # print('inv_distances: ', inv_distances.device)
    num_repeats = torch.ceil(cutoff * inv_distances).to(torch.long)
    num_repeats_ = torch.where(pbc, num_repeats, torch.zeros_like(num_repeats))
    # print('num_repeats_: ', num_repeats_.device)
    images, batch_images, shifts_expanded, shifts_idx_ = [], [], [], []
    for i_structure in range(num_repeats_.shape[0]):
        num_repeats = num_repeats_[i_structure]
        r1 = torch.arange(
            -num_repeats[0],
            num_repeats[0] + 1,
            device=cell.device,
            dtype=torch.long,
        )
        r2 = torch.arange(
            -num_repeats[1],
            num_repeats[1] + 1,
            device=cell.device,
            dtype=torch.long,
        )
        r3 = torch.arange(
            -num_repeats[2],
            num_repeats[2] + 1,
            device=cell.device,
            dtype=torch.long,
        )
        shifts_idx = torch.cartesian_prod(r1, r2, r3)
        shifts = torch.matmul(shifts_idx.to(cell.dtype), cell[i_structure])
        pos = positions[batch == i_structure]
        shift_expanded = shifts.repeat(1, n_atoms[i_structure]).view((-1, 3))
        pos_expanded = pos.repeat(shifts.shape[0], 1)
        images.append(pos_expanded + shift_expanded)

        batch_images.append(
            i_structure
            * torch.ones(
                images[-1].shape[0], dtype=torch.int64, device=cell.device
            )
        )
        shifts_expanded.append(shift_expanded)
        shifts_idx_.append(
            shifts_idx.repeat(1, n_atoms[i_structure]).view((-1, 3))
        )
    return (
        torch.cat(images, dim=0),
        torch.cat(batch_images, dim=0),
        torch.cat(shifts_expanded, dim=0),
        torch.cat(shifts_idx_, dim=0),
    )


@torch.jit.script
def strides_of(v: torch.Tensor) -> torch.Tensor:
    """TODO: add docs"""
    strides = torch.zeros(v.shape[0] + 1, dtype=torch.int64, device=v.device)
    strides[1:] = v
    strides = torch.cumsum(strides, dim=0)
    return strides


def torch_neighbor_list_no_pbc(
    data: AtomicData,
    rcut: float,
    self_interaction: Optional[bool] = True,
    num_workers: Optional[int] = 1,
    max_num_neighbors: Optional[int] = 1000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Function for producing torch neighborlists without periodic boundary
    conditions

    Parameters
    ----------
    data:
        AtomicData instance
    rcut:
        Upper distance cutoff for determining neighbor edges
    self_interaction:
        If True, self edges will added for each atom
    num_workers:
        Number of threads to use for neighbor enumeration
    max_num_neighbors:
        The maximum number of neighbors to return for each atom. For larger
        systems, it is important to make sure that this number is sufficiently
        large.

    Returns
    -------
    torch.Tensor:
        The first atoms in each edge
    torch.Tensor:
        The second atoms in each edge
    torch.Tensor:
        Boolean tensor identifying self_interacting edges
    """

    if "batch" not in data:
        batch = torch.zeros(
            data.pos.shape[0], dtype=torch.long, device=data.pos.device
        )
    else:
        batch = data.batch
    edge_index = radius_graph(
        data.pos,
        rcut,
        batch=batch,
        max_num_neighbors=max_num_neighbors,
        num_workers=num_workers,
        flow="target_to_source",
        loop=self_interaction,
    )
    self_interaction_mask = edge_index[0] != edge_index[1]
    return edge_index[0], edge_index[1], self_interaction_mask


@torch.jit.script
def get_j_idx(
    edge_index: torch.Tensor, batch_images: torch.Tensor, n_atoms: torch.Tensor
) -> torch.Tensor:
    """TODO: add docs"""
    # get neighbor index reffering to the list of original positions
    n_neighbors = torch.bincount(edge_index[0])
    strides = strides_of(n_atoms)
    n_reapeats = torch.zeros_like(n_atoms)
    for i_structure, (st, nd) in enumerate(zip(strides[:-1], strides[1:])):
        n_reapeats[i_structure] = torch.sum(n_neighbors[st:nd])
    n_atoms = torch.repeat_interleave(n_atoms, n_reapeats, dim=0)

    batch_i = torch.repeat_interleave(strides[:-1], n_reapeats, dim=0)

    n_images = torch.bincount(batch_images)
    strides_images = strides_of(n_images[:-1])
    images_shift = torch.repeat_interleave(strides_images, n_reapeats, dim=0)

    j_idx = torch.remainder(edge_index[1] - images_shift, n_atoms) + batch_i
    return j_idx


def torch_neighbor_list_pbc(
    data: AtomicData,
    rcut: float,
    self_interaction: Optional[bool] = True,
    num_workers: Optional[int] = 1,
    max_num_neighbors: Optional[int] = 1000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Function for returning torch neighborlists from AtomicData instances
    with periodic boundary conditions. The minimum image convention is used
    for resolving periodic shifts.

    Parameters
    ----------
    data:
        AtomicData instance
    rcut:
        Upper distance cutoff for determining neighbor edges
    self_interaction:
        If True, self edges will added for each atom
    num_workers:
        Number of threads to use for neighbor enumeration
    max_num_neighbors:
        The maximum number of neighbors to return for each atom. For larger
        systems, it is important to make sure that this number is sufficiently
        large.

    Returns
    -------
    torch.Tensor:
        The first atoms in each edge
    torch.Tensor:
        The second atoms in each edge
    torch.Tensor:
        Boolean tensor identifying self_interacting edges
    """

    if "batch" not in data:
        batch_y = torch.zeros(
            data.pos.shape[0], dtype=torch.long, device=data.pos.device
        )
    else:
        batch_y = data.batch

    images, batch_images, shifts_expanded, shifts_idx = compute_images(
        data.pos, data.cell, data.pbc, rcut, batch_y, data.n_atoms
    )
    edge_index = radius(
        x=images,
        y=data.pos,
        r=rcut,
        batch_x=batch_images,
        batch_y=batch_y,
        max_num_neighbors=max_num_neighbors,
        num_workers=num_workers,
    )

    j_idx = get_j_idx(edge_index, batch_images, data.n_atoms)

    # find self interactions
    is_central_cell = (shifts_idx[edge_index[1]] == 0).all(dim=1)
    mask = torch.cat(
        [is_central_cell.view(-1, 1), (edge_index[0] == j_idx).view(-1, 1)],
        dim=1,
    )
    self_interaction_mask = torch.logical_not(torch.all(mask, dim=1))

    if self_interaction:
        idx_i, idx_j = edge_index[0], j_idx
        cell_shifts = shifts_expanded[edge_index[1]]
    else:
        # remove self interaction
        idx_i, idx_j = (
            edge_index[0][self_interaction_mask],
            j_idx[self_interaction_mask],
        )
        cell_shifts = shifts_expanded[edge_index[1][self_interaction_mask]]

    return idx_i, idx_j, cell_shifts, self_interaction_mask


#def wrap_positions(data: AtomicData, device:str, eps: float = 1e-7) -> None:
#    """Wrap positions to unit cell.
#
#    Returns positions changed by a multiple of the unit cell vectors to
#    fit inside the space spanned by these vectors.
#
#    Parameters
#    ----------
#    data:
#        torch_geometric.Data instance
#    eps: float
#        Small number to prevent slightly negative coordinates from being
#        wrapped.
#    """
#    pos = data.pos
#    cell = data.cell.to(pos.dtype)
#    pbc = data.pbc
#    batch_ids = data.batch
#    print(len(batch_ids))
#
#    unique_batches = torch.unique(batch_ids)
#    n_batches = len(unique_batches)
#
#    center = torch.tensor((0.5, 0.5, 0.5)).view(1, 3).repeat(n_batches, 1).to(pos.dtype).to(device)
#    pbc = data.pbc.view(-1, 3)
#    shift = center - 0.5 - eps
#
#    # Don't change coordinates when pbc is False
#    shift[torch.logical_not(pbc)] = 0.0
#
#    fractional = torch.linalg.solve(cell[batch_ids], pos) - shift
#
#    for i, periodic in enumerate(pbc.detach()[batch_ids].T):
#        if torch.any(periodic):
#            fractional[periodic, i] %= 1.0
#            fractional[periodic, i] += shift[batch_ids][:, i]
#    data.pos = torch.einsum("bi,bii->bi", fractional, cell[batch_ids])
#
#    return data


def wrap_positions(data: AtomicData, device:str) -> None:
    """Function to wrap positions back to unit cell when
    simulations are performed.

    Parameters
    ----------
    data: AtomicData
        Contains positions, cell, pbc, and batch information.
    
    Returns
    -------
    data: AtomicData
        Contains positions updated by a multiple of the unit cell vectors to wrap
        back in the original cell.
    """
    pos = data.pos
    cell = data.cell
    pbc = data.pbc
    batch_ids = data.batch

    if not pbc.any():
        return

    # Compute fractional coordinates
    # cell_batch: [n_atoms, 3, 3] = cell[batch_ids]
    cell_batch = cell[batch_ids]
    fractional = torch.linalg.solve(cell_batch, pos.unsqueeze(-1)).squeeze(-1)

    mask = pbc[batch_ids]

    # Wrap each periodic dimension
    for dim in range(3):
        dim_mask = mask[:, dim]
        if dim_mask.any():
            fractional[dim_mask, dim] = fractional[dim_mask, dim] % 1.0

    # Convert back to cartesian coordinates
    data.pos = torch.einsum('bi,bij->bj', fractional, cell_batch)

    return data
