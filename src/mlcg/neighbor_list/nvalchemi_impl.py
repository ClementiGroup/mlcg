from typing import Tuple, Optional
import torch
from torch_geometric.data import Data
from nvalchemiops.neighborlist import (
    batch_cell_list,
    batch_naive_neighbor_list,
)
import warnings


RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def no_pbc_warning(box_size):
    return f"""
{RED}{BOLD}
============================================================
!!!  WARNING: No PCB information  !!!
============================================================{RESET}

`nvachemi_cell` can only be used with PBC information. Currently 
your data doesn't have pbc or cell entries, and a generic  
square box of length {box_size} will be used to compute PBCs.
{YELLOW}{BOLD}Neighbor list results are not guaranteed to be correct
{RED}{BOLD}============================================================{RESET}
"""


def nvalchemi_naive_neighbor_list(
    data: Data,
    rcut: float,
    self_interaction: bool = True,
    num_workers: int = 1,
    max_num_neighbors: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Function for computing neighbor lists from pytorch geometric data
    instances that may or may not use periodic boundary conditions using
    nvalchemi, an nvidia-provided kernel.

    Note that this function is unable to take into account self interactions.

    Parameters
    ----------
    data:
        Pytorch geometric data instance
    rcut:
        upper distance cutoff, in which neighbors with distances larger
        than this cutoff will be excluded.
    self_interaction:
        Not used, leaved for compatiblity reasons.
    num_workers:
        Not used, leaved for compatiblity issues.
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
    with_pbc = False
    if "pbc" in data:
        pbc = data.pbc
        # the type casting has to be done otherwise the library complains
        cell = data.cell.to(torch.float32)
        with_pbc = True
    else:
        pbc = None
        cell = None

    if with_pbc and torch.any(pbc):
        if "cell" not in data:
            raise ValueError(
                f"Periodic systems need to have a unit cell defined"
            )

    result = batch_naive_neighbor_list(
        data.pos,
        rcut,
        cell=cell,
        pbc=pbc,
        # same problem as in the above call for the hardcoded type cast
        batch_idx=data.batch.to(torch.int32),
        batch_ptr=data.ptr.to(torch.int32),
        max_neighbors=max_num_neighbors,
        return_neighbor_list=True,
    )

    if with_pbc:
        (idx_i, idx_j), _, idx_S = result
        cell_shifts = torch.matmul(idx_S.to(cell.dtype), cell)
        return idx_i, idx_j, cell_shifts, None
    else:
        (idx_i, idx_j), _ = result
        cell_shifts = torch.zeros(
            (idx_i.shape[0], 3), dtype=data.pos.dtype, device=data.pos.device
        )
        return idx_i, idx_j, cell_shifts, None


def nvalchemi_cell_neighbor_list(
    data: Data,
    rcut: float,
    self_interaction: bool = True,
    num_workers: int = 1,
    max_num_neighbors: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Function for computing neighbor lists from pytorch geometric data
    instances with periodic boundary conditions using nvalchemi and a cell
    list structure, an nvidia-provided kernel.

    Note that this function is unable to take into account self interactions.

    Parameters
    ----------
    data:
        Pytorch geometric data instance
    rcut:
        upper distance cutoff, in which neighbors with distances larger
        than this cutoff will be excluded.
    self_interaction:
        Not used, leaved for compatiblity reasons.
    num_workers:
        Not used, leaved for compatiblity issues.
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
    with_pbc = False
    if "pbc" in data:
        pbc = data.pbc
        # the type casting has to be done otherwise the library complains
        cell = data.cell.to(torch.float32)
        with_pbc = True

    else:
        box_size = 70
        warnings.warn(no_pbc_warning(box_size), UserWarning)
        # this is required as the method needs
        cell = (
            torch.zeros(
                data.batch[-1] + 1,
                3,
                3,
                dtype=torch.float32,
                device=data.pos.device,
            )
            + 50
        )
        cell.diagonal(dim1=-2, dim2=-1).fill_(box_size)
        pbc = torch.zeros(
            data.batch[-1] + 1, 3, dtype=torch.bool, device=data.pos.device
        )
    if with_pbc and torch.any(pbc):
        if "cell" not in data:
            raise ValueError(
                f"Periodic systems need to have a unit cell defined"
            )
    result = batch_cell_list(
        data.pos,
        rcut,
        cell=cell,
        pbc=pbc,
        # same problem as in the above call for the hardcoded type cast
        batch_idx=data.batch,
        max_neighbors=max_num_neighbors,
        return_neighbor_list=True,
    )

    (idx_i, idx_j), _, idx_S = result

    if with_pbc:
        cell_shifts = torch.matmul(idx_S.to(cell.dtype), cell)
    else:
        cell_shifts = torch.zeros(
            (idx_i.shape[0], 3), dtype=data.pos.dtype, device=data.pos.device
        )

    return idx_i, idx_j, cell_shifts, None
