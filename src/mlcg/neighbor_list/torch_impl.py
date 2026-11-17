from typing import Tuple, Optional
import torch
from torch_geometric.data import Data
from pyg_lib.ops import radius
from ..data.atomic_data import AtomicData


def safe_radius(
    x: torch.Tensor,
    y: torch.Tensor,
    r: float,
    batch_x: Optional[torch.Tensor] = None,
    batch_y: Optional[torch.Tensor] = None,
    max_num_neighbors: int = 32,
    num_workers: int = 1,
    batch_size: Optional[int] = None,
    ignore_same_index: bool = False,
):
    """A wrapper around pyg_lib.ops that keeps same API of
    torch_cluster now deprecated.
    """
    if x.numel() == 0 or y.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long, device=x.device)

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y
    x, y = x.contiguous(), y.contiguous()

    if batch_size is None:
        batch_size = 1
        if batch_x is not None:
            assert x.size(0) == batch_x.numel()
            batch_size = int(batch_x.max()) + 1
        if batch_y is not None:
            assert y.size(0) == batch_y.numel()
            batch_size = max(batch_size, int(batch_y.max()) + 1)
    assert batch_size > 0

    ptr_x: Optional[torch.Tensor] = None
    ptr_y: Optional[torch.Tensor] = None

    if batch_size > 1:
        assert batch_x is not None
        assert batch_y is not None
        arange = torch.arange(batch_size + 1, device=x.device)
        ptr_x = torch.bucketize(arange, batch_x)
        ptr_y = torch.bucketize(arange, batch_y)
    return radius(
        x, y, r, ptr_x, ptr_y, max_num_neighbors, num_workers, ignore_same_index
    )


def safe_radius_graph(
    x: torch.Tensor,
    r: float,
    batch: Optional[torch.Tensor] = None,
    loop: bool = False,
    max_num_neighbors: int = 32,
    flow: str = "source_to_target",
    num_workers: int = 1,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    r"""Computes graph edges to all points within a given distance.
    adapted to maintain same API of torch_cluster `radius_graph` now deprecated.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        r (float): The radius.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. :obj:`batch` needs to be sorted.
            (default: :obj:`None`)
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element.
            If the number of actual neighbors is greater than
            :obj:`max_num_neighbors`, returned neighbors are picked randomly.
            (default: :obj:`32`)
        flow (string, optional): The flow direction when used in combination
            with message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
        num_workers (int): Number of workers to use for computation. Has no
            effect in case :obj:`batch` is not :obj:`None`, or the input lies
            on the GPU. (default: :obj:`1`)
        batch_size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)
    """

    assert flow in ["source_to_target", "target_to_source"]
    edge_index = safe_radius(
        x,
        x,
        r,
        batch,
        batch,
        max_num_neighbors,
        num_workers,
        batch_size,
        not loop,
    )
    if flow == "source_to_target":
        row, col = edge_index[1], edge_index[0]
    else:
        row, col = edge_index[0], edge_index[1]

    return torch.stack([row, col], dim=0)


def torch_neighbor_list(
    data: Data,
    rcut: float,
    self_interaction: bool = True,
    num_workers: int = 1,
    max_num_neighbors: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Function for computing neighbor lists from pytorch geometric data
    instances that may or may not use periodic boundary conditions. A batch
    can mix periodic and non-periodic structures, and the positions do not
    need to be wrapped into the unit cell.

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
        kwarg for the radius search of pyg_lib, specifying the maximum
        number of neighbors for each atom

    Returns
    -------
    torch.Tensor:
        The atom indices of the first atoms in each neighbor pair
    torch.Tensor:
        The atom indices of the second atoms in each neighbor pair
    torch.Tensor:
        The cell shifts associated with minimum image distances
        in the presence of periodic boundary conditions: the distance
        vector of a pair is ``pos[j] - pos[i] + cell_shift``
    torch.Tensor:
        Boolean mask over the returned pairs, False for self interactions
    """
    pos = data.pos
    if "batch" in data:
        batch = data.batch
    else:
        batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
    pbc = data.pbc.view(-1, 3) if "pbc" in data else None
    if pbc is None or not torch.any(pbc):
        edge_index = safe_radius_graph(
            pos,
            rcut,
            batch=batch,
            max_num_neighbors=max_num_neighbors,
            num_workers=num_workers,
            flow="target_to_source",
            loop=self_interaction,
        )
        idx_i, idx_j = edge_index[0], edge_index[1]
        cell_shifts = torch.zeros(
            (idx_i.shape[0], 3), dtype=pos.dtype, device=pos.device
        )
        return idx_i, idx_j, cell_shifts, idx_i != idx_j
    if "cell" not in data:
        raise ValueError(f"Periodic systems need to have a unit cell defined")

    cell = data.cell.view(-1, 3, 3).to(pos.dtype)
    cell = torch.where(
        pbc.any(dim=1)[:, None, None],
        cell,
        torch.eye(3, dtype=pos.dtype, device=pos.device),
    )
    offsets = _cell_offsets(pos, cell[batch], pbc[batch])
    inside = pos - _lattice_vectors(offsets, cell[batch])
    image_atom, image_shift = _images(cell, pbc, rcut, batch)
    edge_index = safe_radius(
        x=inside[image_atom]
        + _lattice_vectors(image_shift, cell[batch[image_atom]]),
        y=inside,
        r=rcut,
        batch_x=batch[image_atom],
        batch_y=batch,
        max_num_neighbors=max_num_neighbors,
        num_workers=num_workers,
    )
    idx_i = edge_index[0]
    idx_j = image_atom[edge_index[1]]
    shifts = image_shift[edge_index[1]] + offsets[idx_i] - offsets[idx_j]
    not_self = (idx_i != idx_j) | torch.any(shifts != 0, dim=1)
    if not self_interaction:
        idx_i, idx_j, shifts = (
            idx_i[not_self],
            idx_j[not_self],
            shifts[not_self],
        )
        not_self = not_self[not_self]
    cell_shifts = _lattice_vectors(shifts, cell[batch[idx_i]])
    return idx_i, idx_j, cell_shifts, not_self


def _lattice_vectors(shifts: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """``shifts @ cell`` per row, elementwise: cuBLAS may round the cell to
    TF32 in a matmul."""
    return (shifts.to(cell.dtype).unsqueeze(-1) * cell).sum(dim=1)


def _cell_offsets(
    pos: torch.Tensor, cell: torch.Tensor, pbc: torch.Tensor
) -> torch.Tensor:
    """Integer lattice coordinates of the cell holding each atom, 0 along
    non-periodic axes. The rows of ``cell`` are the lattice vectors."""
    fractional = torch.linalg.solve(
        cell.transpose(-1, -2), pos.unsqueeze(-1)
    ).squeeze(-1)
    offsets = torch.floor(fractional).to(torch.long)
    return torch.where(pbc, offsets, torch.zeros_like(offsets))


def _images(
    cell: torch.Tensor, pbc: torch.Tensor, rcut: float, batch: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Atom index and integer lattice shift of every periodic image that
    can hold a neighbor of an atom of the same structure inside its cell:
    up to ``ceil(rcut / plane spacing)`` cells away along periodic axes."""
    plane_spacing = 1.0 / torch.linalg.inv(cell).norm(dim=-2)
    reach = torch.ceil(rcut / plane_spacing).to(torch.long)
    reach = torch.where(pbc, reach, torch.zeros_like(reach))
    shifts = torch.cartesian_prod(
        *(
            torch.arange(-r, r + 1, device=cell.device)
            for r in reach.max(dim=0).values.tolist()
        )
    )
    in_reach = torch.all(shifts.abs()[None] <= reach[:, None], dim=-1)
    image_atom, shift_index = torch.nonzero(in_reach[batch], as_tuple=True)
    return image_atom, shifts[shift_index]


def wrap_positions(data: AtomicData, device: str, eps: float = 1e-7) -> None:
    """Vectorized version of position wrapping to unit cell.
    Returns positions changed by a multiple of the unit cell vectors to
    fit inside the space spanned by these vectors.
    Parameters
    ----------
    data: AtomicData
        torch_geometric.Data instance with pos, cell, pbc, batch attributes
    device: str
        Device to perform computations on
    eps: float
        Small number to prevent slightly negative coordinates from being wrapped

    Returns
    -------
    AtomicData:
        torch_geometric.Data instance with positions wrapped in unit cell
    """
    pos = data.pos
    cell = data.cell.to(pos.dtype)
    pbc = data.pbc
    batch_ids = data.batch

    cell_batch = cell[batch_ids]  # [n_atoms, 3, 3]
    # lattice vectors are the rows of the cell: pos = f @ cell
    fractional = torch.linalg.solve(
        cell_batch.transpose(-1, -2), pos.unsqueeze(-1)
    ).squeeze(-1)

    pbc_mask = pbc[batch_ids]  # [n_atoms, 3]

    shift = -eps

    fractional = torch.where(
        pbc_mask, (fractional - shift) % 1.0 + shift, fractional
    )
    # Convert back to cartesian coordinates
    data.pos = (fractional.unsqueeze(-1) * cell_batch).sum(dim=1)

    return data
