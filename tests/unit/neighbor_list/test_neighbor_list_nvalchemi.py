import pytest
import ase
from ase.build import bulk, molecule
from torch_geometric.loader import DataLoader
import numpy as np
import torch

from mlcg.neighbor_list.utils import ase2data
from mlcg.neighbor_list.ase_impl import ase_neighbor_list
from mlcg.geometry.internal_coordinates import compute_distances

try:
    from mlcg.neighbor_list.nvalchemi_impl import (
        nvalchemi_naive_neighbor_list,
        nvalchemi_cell_neighbor_list,
        nvalchemi_cell_neighbor_list_raw,
    )

    NVALCH_AVAILABLE = True
except ImportError:
    print(
        "nalchemi is not installed. Please install with "
        + "pip install nvalchemi-toolkit-ops"
    )
    NVALCH_AVAILABLE = False


def sort_edges(edge_index, *tensors):
    if edge_index.numel() == 0:
        return (edge_index,) + tuple(t for t in tensors)
    stride = edge_index.max().item() + 1
    key = edge_index[0] * stride + edge_index[1]
    order = torch.argsort(key)
    return (edge_index[:, order],) + tuple(t[order] for t in tensors)


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


nvalchemi_test_set = [
    (name, frame, rc, self_interaction)
    for (name, frame) in atomic_structures()
    for rc in range(2, 7, 2)
    for self_interaction in [False]
]

# resolved lazily by name so that collection works without nvalchemi installed
NVALCHEMI_CELL_METHODS = (
    {
        "cell": nvalchemi_cell_neighbor_list,
        "raw": nvalchemi_cell_neighbor_list_raw,
    }
    if NVALCH_AVAILABLE
    else {}
)

nvalchemi_cell_method_names = ["cell","raw"]


@pytest.mark.skipif(
    not NVALCH_AVAILABLE,
    reason="nvalchemi is not available (install nvalchemi-toolkit-ops)",
)
@pytest.mark.parametrize(
    "name, frame, cutoff, self_interaction",
    nvalchemi_test_set,
)
def test_neighborlist_nvalchemi(name, frame, cutoff, self_interaction):
    """Check that nvalchemi_neighbor_list gives the same NL as ASE by comparing
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
                met = nvalchemi_naive_neighbor_list
            idx_i, idx_j, cell_shifts, _ = met(
                data, cutoff, self_interaction=self_interaction
            )
            
            dd = (data.pos[idx_j] - data.pos[idx_i] + cell_shifts).norm(dim=1)
            dds.extend(dd.numpy())
        dds = np.sort(dds)
        edge_index = torch.stack([idx_i, idx_j], dim=0)
        edge_index = sort_edges(edge_index)
        distance_results[met_name] = dds
        neighs_results[met_name] = edge_index
    assert np.allclose(
        distance_results["current_nls_method"], distance_results["ase_ref"]
    )
    assert np.allclose(
        neighs_results["current_nls_method"], neighs_results["ase_ref"]
    )


@pytest.mark.skipif(
    not NVALCH_AVAILABLE,
    reason="nvalchemi is not available (install nvalchemi-toolkit-ops)",
)
@pytest.mark.parametrize("nls_name", nvalchemi_cell_method_names)
def test_neighborlist_pbc_nvalchemi(nls_name):
    """Test that neighbor list with PBC correctly handles periodic images
    and produces the same results as ASE reference implementation."""
    nls_method = NVALCHEMI_CELL_METHODS[nls_name]

    # Create test structures with PBC
    structures = [
        bulk("Cu", "fcc", a=3.6),
    ]

    cutoffs = [
        3.0,
    ]

    for structure in structures:
        for cutoff in cutoffs:
            for self_interaction in [False]:
                # Convert to data format
                data_list = [ase2data(structure)]
                dataloader = DataLoader(data_list, batch_size=1)

                # Get nvalchemi neighbor list distances
                nvalchemi_distances = []
                for data in dataloader:
                    if "cell" in data:
                        print("Cell:\n", data.cell)
                    idx_i, idx_j, cell_shifts, _ = nls_method(
                        data, cutoff, self_interaction=self_interaction
                    )
                    if nls_name == "raw":
                        cell = data.cell.reshape(-1,3,3)
                        cell_shifts = (
                            cell_shifts.to(cell.dtype).to(cell.dtype).unsqueeze(-1)
                            * cell[data.batch[idx_i]]
                        ).sum(dim=1)
                    mapping = torch.stack([idx_i, idx_j], dim=0)
                    dd = compute_distances(data.pos, mapping, cell_shifts)
                    nvalchemi_distances.extend(dd.numpy())

                nvalchemi_distances = np.sort(nvalchemi_distances)

                # Get ASE reference distances
                ase_distances = []
                for data in dataloader:
                    idx_i, idx_j, ase_cell_shifts, _ = ase_neighbor_list(
                        data, cutoff, self_interaction=self_interaction
                    )
                    dd = (
                        data.pos[idx_j] - data.pos[idx_i] + ase_cell_shifts
                    ).norm(dim=1)
                    ase_distances.extend(dd.numpy())

                ase_distances = np.sort(ase_distances)

                assert np.allclose(
                    ase_distances, nvalchemi_distances, rtol=1e-5, atol=1e-6
                )

                assert np.all(nvalchemi_distances <= cutoff + 1e-6)


@pytest.mark.skipif(
    not NVALCH_AVAILABLE,
    reason="nvalchemi is not available (install nvalchemi-toolkit-ops)",
)
@pytest.mark.parametrize("nls_name", nvalchemi_cell_method_names)
def test_pbc_minimum_image_convention_nvalchemi(nls_name):
    """Test that PBC neighbor list correctly applies minimum image convention.
    Neighbors should be found across periodic boundaries at the shortest distance.
    """
    nls_method = NVALCHEMI_CELL_METHODS[nls_name]

    # Create a simple cubic cell with one atom
    atoms = ase.Atoms(
        "Ar",
        positions=[[0.1, 0.1, 0.1]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )

    cutoff = 3.0
    data_list = [ase2data(atoms)]
    dataloader = DataLoader(data_list, batch_size=1)

    for data in dataloader:
        idx_i, idx_j, cell_shifts, _ = nls_method(
            data, cutoff, self_interaction=False
        )

        assert len(idx_i) == 0, "Single isolated atom should have no neighbors"

    atoms = ase.Atoms(
        "Ar2",
        positions=[[0.1, 0.1, 0.1], [9.9, 0.1, 0.1]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )

    data_list = [ase2data(atoms)]
    dataloader = DataLoader(data_list, batch_size=1)

    distances = []
    for data in dataloader:
        idx_i, idx_j, cell_shifts, _ = nls_method(
            data, cutoff, self_interaction=False
        )
        if nls_name == "raw":
            cell = data.cell.reshape(-1,3,3)
            cell_shifts = (
                cell_shifts.to(cell.dtype).to(cell.dtype).unsqueeze(-1)
                * cell[data.batch[idx_i]]
            ).sum(dim=1)

        mapping = torch.stack([idx_i, idx_j], dim=0)
        dd = compute_distances(data.pos, mapping, cell_shifts)
        distances.extend(dd.numpy())

        distances = np.sort(distances)
        assert np.all(distances < 1.0), (
            f"Minimum image convention not applied correctly. "
            f"Distances: {distances}"
        )


@pytest.mark.skipif(
    not NVALCH_AVAILABLE,
    reason="nvalchemi is not available (install nvalchemi-toolkit-ops)",
)
@pytest.mark.parametrize("nls_name", nvalchemi_cell_method_names)
def test_mixed_pbc_nvalchemi(nls_name):
    """Test neighbor list with partial periodic boundary conditions."""
    nls_method = NVALCHEMI_CELL_METHODS[nls_name]

    atoms = ase.Atoms(
        "C4",
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 5.0],
        ],
        cell=[5.0, 5.0, 10.0],
        pbc=[True, True, False],  # Periodic in x,y but not z
    )

    cutoff = 2.0
    data_list = [ase2data(atoms)]
    dataloader = DataLoader(data_list, batch_size=1)
    distances = []
    for data in dataloader:
        idx_i, idx_j, cell_shifts, _ = nls_method(
            data, cutoff, self_interaction=False
        )

        cell = data.cell.reshape(-1,3,3)
        if nls_name == "raw":
            cell = data.cell.reshape(-1,3,3)
            cell_shifts = (
                cell_shifts.to(cell.dtype).unsqueeze(-1)
                * cell[data.batch[idx_i]]
            ).sum(dim=1)

        mapping = torch.stack([idx_i, idx_j], dim=0)
        dd = compute_distances(data.pos, mapping, cell_shifts)
        distances.extend(dd.numpy())
        distances = np.sort(distances)

        assert np.all(distances <= cutoff + 1e-6)

        atoms_involved = torch.cat([idx_i, idx_j]).unique()

        assert len(atoms_involved) >= 3