import pytest
import ase
from ase.build import bulk, molecule
from torch_geometric.loader import DataLoader
import numpy as np
import torch

from mlcg.neighbor_list.utils import ase2data
from mlcg.neighbor_list.ase_impl import ase_neighbor_list
from mlcg.neighbor_list.torch_impl import torch_neighbor_list
from mlcg.geometry.internal_coordinates import compute_distances

try:
    from mlcg.neighbor_list.nvalchemi_impl import nvalchemi_neighbor_list

    NVALCH_AVAILABLE = True
except ImportError:
    print(
        "nalchemiis not installed. Please install with "
        + "pip install nvalchemi-toolkit-ops"
    )
    NVALCH_AVAILABLE = False


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


test_set = [
    (torch_neighbor_list, name, frame, rc, self_interaction)
    for (name, frame) in atomic_structures()
    for rc in range(2, 7, 2)
    for self_interaction in [False, True]
]

if NVALCH_AVAILABLE:
    test_set += [
        (nvalchemi_neighbor_list, name, frame, rc, self_interaction)
        for (name, frame) in atomic_structures()
        for rc in range(2, 7, 2)
        for self_interaction in [False]
    ]


@pytest.mark.parametrize(
    "nls_method, name, frame, cutoff, self_interaction",
    test_set,
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

def test_neighborlist_pbc():
    """Test that neighbor list with PBC correctly handles periodic images
    and produces the same results as ASE reference implementation."""
    
    # Create test structures with PBC
    structures = [
        bulk("Cu", "fcc", a=3.6),
    ]
    
    cutoffs = [3.0,]
    
    for structure in structures:
        for cutoff in cutoffs:
            for self_interaction in [True, False]:
                # Convert to data format
                data_list = [ase2data(structure)]
                dataloader = DataLoader(data_list, batch_size=1)
                
                # Get torch neighbor list distances
                torch_distances = []
                for data in dataloader:
                    if "cell" in data:
                        print("Cell:\n", data.cell)
                    idx_i, idx_j, cell_shifts, _ = torch_neighbor_list(
                        data, cutoff, self_interaction=self_interaction
                    )
                    mapping = torch.stack([idx_i, idx_j], dim=0)
                    dd = compute_distances(data.pos, mapping, cell_shifts)
                    torch_distances.extend(dd.numpy())
                
                torch_distances = np.sort(torch_distances)
                
                # Get ASE reference distances
                ase_distances = []
                for data in dataloader:
                    idx_i, idx_j, ase_cell_shifts, _ = ase_neighbor_list(
                        data, cutoff, self_interaction=self_interaction
                    )
                    dd = (data.pos[idx_j] - data.pos[idx_i] + ase_cell_shifts).norm(
                        dim=1
                    )
                    ase_distances.extend(dd.numpy())
                
                ase_distances = np.sort(ase_distances)
                
                assert np.allclose(
                    ase_distances, torch_distances, rtol=1e-5, atol=1e-6
                )
                
                assert np.all(torch_distances <= cutoff + 1e-6)

def test_pbc_minimum_image_convention():
    """Test that PBC neighbor list correctly applies minimum image convention.
    Neighbors should be found across periodic boundaries at the shortest distance."""
    
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
        idx_i, idx_j, cell_shifts, _ = torch_neighbor_list(
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
        idx_i, idx_j, cell_shifts, _ = torch_neighbor_list(
            data, cutoff, self_interaction=False
        )
        
        mapping = torch.stack([idx_i, idx_j], dim=0)
        dd = compute_distances(data.pos, mapping, cell_shifts)
        distances.extend(dd.numpy())

        distances = np.sort(distances)
        assert np.all(distances < 1.0), (
            f"Minimum image convention not applied correctly. "
            f"Distances: {distances.numpy()}"
        )

def test_mixed_pbc():
    """Test neighbor list with partial periodic boundary conditions."""
    
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
        idx_i, idx_j, cell_shifts, _ = torch_neighbor_list(
            data, cutoff, self_interaction=False
        )
        
        mapping = torch.stack([idx_i, idx_j], dim=0)
        dd = compute_distances(data.pos, mapping, cell_shifts)
        distances.extend(dd.numpy())
        distances = np.sort(distances)

        assert np.all(distances <= cutoff + 1e-6)
        
        atoms_involved = torch.cat([idx_i, idx_j]).unique()
        
        assert len(atoms_involved) >= 3
