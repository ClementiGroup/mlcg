import pytest
import ase
from ase.build import bulk, molecule
from torch_geometric.loader import DataLoader
import numpy as np
import torch

from mlcg.neighbor_list.ase_impl import ase_neighbor_list
from mlcg.neighbor_list.torch_impl import torch_neighbor_list
from mlcg.neighbor_list.nvalchemi_impl import nvalchemi_neighbor_list
from mlcg.neighbor_list.utils import ase2data


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
    "nls_method, name, frame, cutoff, self_interaction",
    [
        (nls_met, name, frame, rc, self_interaction)
        for nls_met in [torch_neighbor_list, nvalchemi_neighbor_list]
        for (name, frame) in atomic_structures()
        for rc in range(2, 7, 2)
        for self_interaction in [False]
    ],
)
def test_neighborlist(nls_method, name, frame, cutoff, self_interaction):
    """Check that torch_neighbor_list gives the same NL as ASE by comparing
    the resulting sorted list of distances between neighbors."""
    data_list = [ase2data(frame)]
    dataloader = DataLoader(data_list, batch_size=1)
    distance_results = {}
    neighs_results = {}
    method_list = ["current_nls_method", "ase_ref"]
    for met_name in method_list:
        dds = []
        for data in dataloader:
            if met_name == "ase_ref":
                met = ase_neighbor_list
            else:
                met = nls_method
            idx_i, idx_j, cell_shifts, _ = met(
                data, cutoff, self_interaction=self_interaction
            )
            dd = (data.pos[idx_j] - data.pos[idx_i] + cell_shifts).norm(dim=1)
            dds.extend(dd.numpy())
        dds = np.sort(dds)
        distance_results[met_name] = dds
        neighs_results[met_name] = torch.stack([idx_i, idx_j], dim=0).sort(
            dim=1
        )
    assert np.allclose(
        distance_results["current_nls_method"], distance_results["ase_ref"]
    )
    assert np.allclose(
        neighs_results["current_nls_method"], neighs_results["ase_ref"]
    )
