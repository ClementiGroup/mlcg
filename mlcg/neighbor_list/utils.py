import torch
from ase.atoms import Atoms
from torch_geometric.data import Data


def ase2data(
    frame: Atoms, energy_tag: str = None, force_tag: str = None
) -> Data:
    r"""Helper function to convert and ASE Atoms instance into a
    pytorch geometric Data format.

    Prameters
    ---------
    frame:
        ASE atoms object
    energy_tag:
        If specified, this energy from frame.info will be stored in
        the output data.energy field.
    force_energy:
        If specified, this force from frame.info will be stroed in
        the output data.forces field.

    Returns
    -------
    Data:
        Pytorch geometric Data instance containing the following fields:

    .. code-block:python

    data = Data(
        z: atomic numbers, shape (n_atoms)
        pos: atomic positions, shape (n_atoms, 3)
        pbc: periodic boundary condition flags
        cell: unit cell vectors of shape (3,3),
        n_atoms: length of Atoms frame,
    )

    """

    z = torch.from_numpy(frame.get_atomic_numbers())
    pos = torch.from_numpy(frame.get_positions())
    pbc = torch.from_numpy(frame.get_pbc())
    cell = torch.tensor(frame.get_cell().tolist())
    n_atoms = torch.tensor([len(frame)])
    data = Data(z=z, pos=pos, pbc=pbc, cell=cell, n_atoms=n_atoms)

    if energy_tag is not None:
        E = torch.tensor(frame.info[energy_tag])
        data.energy = E
    if force_tag is not None:
        forces = torch.from_numpy(frame.arrays[force_tag])
        data.forces = forces

    return data
