import torch
import pytest
import numpy as np

from mlcg.geometry.topology import (
    Topology,
    get_connectivity_matrix,
    get_n_paths,
    get_n_pairs,
)

# make a simple molecule defined by the following bonded
# topology, with dummy atom types and names.

atom_types = [1, 6, 2, 5, 4, 9, 8, 2, 6, 4, 7]
atom_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
bond_edges = torch.tensor(
    [[0, 1, 1, 3, 4, 3, 6, 6, 8, 8], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
)

# All of the 1-5 (unique) distance pairs
edges_1_5 = torch.tensor(
    [
        [0, 0, 0, 1, 1, 2, 2, 2, 4, 4, 5, 5],
        [5, 7, 8, 9, 10, 5, 7, 8, 9, 10, 7, 8],
    ]
)

# All of the (unique) 1-3 bonded angles
bonded_angles = torch.tensor(
    [
        [0, 0, 1, 1, 2, 3, 3, 3, 4, 6, 6, 7, 9],
        [1, 1, 3, 3, 1, 4, 6, 6, 3, 8, 8, 6, 8],
        [2, 3, 4, 6, 3, 5, 7, 8, 6, 9, 10, 8, 10],
    ]
)

# The undirected connectivity matrix associated with the bonded graph
cmat_undirected = torch.tensor(
    [
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    ]
)

# The directed connectivity matrix associated with the bonded graph
cmat_directed = torch.tensor(
    [
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)

# Topology object for the above molecule
test_topo = Topology()

# Adding atoms
for atom_type, name in zip(atom_types, atom_names):
    test_topo.add_atom(atom_type, name)

# Adding bonds
test_topo.bonds_from_edge_index(bond_edges)

# Adding angles
test_topo.angles_from_edge_index(bonded_angles)

# Bonds to remove
bonds_to_remove = [
    edge.numpy().tolist() for i, edge in enumerate(bond_edges.t()) if i % 2 == 0
]
bonds_left = [
    edge.numpy().tolist() for i, edge in enumerate(bond_edges.t()) if i % 2 != 0
]

# Angles to remove
angles_to_remove = [
    edge.numpy().tolist()
    for i, edge in enumerate(bonded_angles.t())
    if i % 2 == 0
]
angles_left = [
    edge.numpy().tolist()
    for i, edge in enumerate(bonded_angles.t())
    if i % 2 != 0
]


@pytest.mark.parametrize(
    "topo_input,cmat_expected,directed",
    [(test_topo, cmat_undirected, False), (test_topo, cmat_directed, True)],
)
def test_connectivity_matrix(topo_input, cmat_expected, directed):
    """Test to make sure that get_connectivity_matrix returns the
    expected connectivity matrix for directed and undirected cases
    """
    constructed_cmat = get_connectivity_matrix(topo_input, directed).numpy()
    np.testing.assert_array_equal(constructed_cmat, cmat_expected.numpy())


@pytest.mark.parametrize("test_topo, pairs_expected", [(test_topo, edges_1_5)])
def test_n_pairs(test_topo, pairs_expected):
    """Test to make sure that get_n_pairs returns the expected set of pairs"""
    cmat = get_connectivity_matrix(test_topo)
    recovered_pairs = get_n_pairs(cmat, n=5).numpy()
    np.testing.assert_array_equal(recovered_pairs, pairs_expected.numpy())


@pytest.mark.parametrize(
    "test_topo, edges_expected, n, unique",
    [
        (test_topo, bonded_angles, 3, True),
    ],
)
def test_n_paths(test_topo, edges_expected, n, unique):
    """Test to make sure that the 1-5 pairs are correctly returned by
    get_n_paths
    """
    cmat = get_connectivity_matrix(test_topo)
    recovered_paths = get_n_paths(cmat, n=n, unique=unique).numpy()
    np.testing.assert_array_equal(recovered_paths, edges_expected.numpy())


# <<< !!! Leave these here! the removal methods modify the topology bond/angle lists in place !!! >>> #


@pytest.mark.parametrize(
    "test_topo, bonds_to_remove, bonds_left",
    [
        (test_topo, bonds_to_remove, bonds_left),
    ],
)
def test_remove_bonds(test_topo, bonds_to_remove, bonds_left):
    """Tests to make sure bonds are removed properly"""
    num_bonds = len(test_topo.bonds[0])
    test_topo.remove_bond(bonds_to_remove)
    assert len(test_topo.bonds[0]) == num_bonds - len(bonds_to_remove)
    assert len(test_topo.bonds[1]) == num_bonds - len(bonds_to_remove)
    for i in range(len(test_topo.bonds[0])):
        atom1, atom2 = test_topo.bonds[0][i], test_topo.bonds[1][i]
        expected_atom1, expected_atom2 = bonds_left[i][0], bonds_left[i][1]
        assert atom1 == expected_atom1
        assert atom2 == expected_atom2


@pytest.mark.parametrize(
    "test_topo, angles_to_remove, angles_left",
    [
        (test_topo, angles_to_remove, angles_left),
    ],
)
def test_remove_angles(test_topo, angles_to_remove, angles_left):
    """Tests to make sure angles are removed properly"""
    num_angles = len(test_topo.angles[0])
    test_topo.remove_angle(angles_to_remove)
    assert len(test_topo.angles[0]) == num_angles - len(angles_to_remove)
    assert len(test_topo.angles[1]) == num_angles - len(angles_to_remove)
    assert len(test_topo.angles[2]) == num_angles - len(angles_to_remove)
    for i in range(len(test_topo.angles[0])):
        atom1, atom2, atom3 = (
            test_topo.angles[0][i],
            test_topo.angles[1][i],
            test_topo.angles[2][i],
        )
        expected_atom1, expected_atom2, expected_atom3 = (
            angles_left[i][0],
            angles_left[i][1],
            angles_left[i][2],
        )
        assert atom1 == expected_atom1
        assert atom2 == expected_atom2
        assert atom3 == expected_atom3
