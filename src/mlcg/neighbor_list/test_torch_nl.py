import pytest
import ase
from ase.build import bulk, molecule
from torch_geometric.loader import DataLoader
import numpy as np

from .ase_impl import ase_neighbor_list
from .torch_impl import torch_neighbor_list
from .utils import ase2data


def bulk_metal():
    a = 4.0
    b = a / 2
    frames = [
        ase.Atoms(
            "Ag",
            cell=[(0, b, b), (b, 0, b), (b, b, 0)],
            pbc=True,
        ),
        bulk("Cu", "fcc", a=3.6),
    ]
    return frames


def atomic_structures():
    frames = [
        molecule("CH3CH2NH2"),
        molecule("H2O"),
        molecule("methylenecyclopropane"),
    ] + bulk_metal()
    for frame in frames:
        yield (frame.get_chemical_symbols(), frame)


@pytest.mark.parametrize(
    "name, frame, cutoff, self_interaction",
    [
        (name, frame, rc, self_interaction)
        for (name, frame) in atomic_structures()
        for rc in range(2, 7, 2)
        for self_interaction in [True, False]
    ],
)
def test_neighborlist(name, frame, cutoff, self_interaction):
    """Check that torch_neighbor_list gives the same NL as ASE by comparing
    the resulting sorted list of distances between neighbors."""
    data_list = [ase2data(frame)]
    dataloader = DataLoader(data_list, batch_size=1)

    dds = []
    for data in dataloader:
        idx_i, idx_j, cell_shifts, _ = torch_neighbor_list(
            data, cutoff, self_interaction=self_interaction
        )
        dd = (data.pos[idx_j] - data.pos[idx_i] + cell_shifts).norm(dim=1)
        dds.extend(dd.numpy())
    dds = np.sort(dds)

    dd_ref = []
    for data in dataloader:
        idx_i, idx_j, cell_shifts = ase_neighbor_list(
            data, cutoff, self_interaction=self_interaction
        )
        dd = (data.pos[idx_j] - data.pos[idx_i] + cell_shifts).norm(dim=1)
        dd_ref.extend(dd.numpy())
    dd_ref = np.sort(dd_ref)

    assert np.allclose(dd_ref, dds)
