from typing import Tuple
import torch
import numpy as np

from ase.neighborlist import neighbor_list
from ase import Atoms
from ..data.atomic_data import AtomicData


def ase_neighbor_list(
    data: AtomicData, rcut: float, self_interaction: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Function for converting a list of neighbor edges from
    an input AtomicData instance, to an ASE neighborlist as
    output by ase.neighborlist.neighbor_list. Return documenation
    taken from https://wiki.fysik.dtu.dk/ase/_modules/ase/neighborlist.html#neighbor_list.

    Parameters
    ----------
    data:
        Input AtomicData instance. Must contain only one structure.
    rcut:
        Upper distance cover for determining neighbors
    self_interaction:
        If True, self edges will be added.

    Returns
    -------
    torch.Tensor:
        first atom indices, of shape (n_atoms)
    torch.Tensor:
        second atom index, of shape (n_atoms)
    torch.Tensor:
        Dot product of the periodic shift vectors with the system unit cell vectors
    """

    assert data.n_atoms.shape[0] == 1, "data should contain only one structure"

    frame = Atoms(
        positions=data.pos.numpy(),
        cell=data.cell.numpy(),
        pbc=data.pbc.numpy(),
        numbers=data.z.numpy(),
    )

    idx_i, idx_j, idx_S = neighbor_list(
        "ijS", frame, cutoff=rcut, self_interaction=self_interaction
    )
    return (
        torch.from_numpy(idx_i),
        torch.from_numpy(idx_j),
        torch.from_numpy(np.dot(idx_S, frame.cell)),
    )
