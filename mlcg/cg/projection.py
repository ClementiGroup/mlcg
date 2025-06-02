from typing import Optional, List, Tuple, Dict, Callable, OrderedDict
from copy import deepcopy
from collections import OrderedDict
import numpy as np
import torch
import networkx as nx
from networkx.algorithms.shortest_paths.unweighted import (
    bidirectional_shortest_path,
)


from ._mappings import CA_MAP, OPEPS_MAP
from ..geometry.topology import (
    Topology,
    add_chain_bonds,
    add_chain_angles,
    get_n_pairs,
    get_connectivity_matrix,
    add_chain_dihedrals,
)
from ..geometry._symmetrize import (
    _symmetrise_distance_interaction,
)


def build_cg_matrix(
    topology: Topology,
    cg_mapping: Dict[Tuple[str, str], Tuple[str, int, int]] = None,
    special_terminal: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, OrderedDict]:
    r"""Function for producing coarse grain types, masses, and
    mapping matrices using a slicing strategy and a predetermined
    set of atoms to retain at the coarse grain resolution.

    Parameters
    ----------
    topology:
        System topology instance
    cg_mapping:
        Mapping dictionary with the following structure:

        .. code-block:: python

            {
                (residue name, atom name) : (compound name, type, mass)
                ...
            }

        Eg, a row for an alanine carbon alpha atom would be:

        .. code-block:: python

            {
                ("ALA", "CA") : ("CA_A", 1, 12)
                ...
            }

    special_termini:
        If True, special types will be reserved for the first and
        last CG atoms

    Returns
    -------
    np.ndarray:
        Array of CG atom types
    np.ndarray:
        Array of CG masses
    np.ndarray:
        One-hot transformation matrix of shape (n_high_res_atoms, n_cg_atoms)
        that maps atoms for the high resolution repesentation to the coarse
        grain representation
    OrderedDict:
        Ordered dictionary mapping each CG atom index (with respect to the
        CG topology) to a list containing the CG atom name, CG atom type
        and the CG atom mass
    """

    if cg_mapping == None:
        cg_mapping = CA_MAP

    cg_mapping_ = OrderedDict()
    n_atoms = topology.n_atoms
    for i_at, at in enumerate(topology.atoms):
        (cg_name, cg_type, cg_mass) = cg_mapping.get(
            (at.resname, at.name), (None, None, None)
        )
        if cg_name is None:
            continue
        else:
            cg_mapping_[i_at] = [cg_name, cg_type, cg_mass]
    general_mapping = OrderedDict()
    for i, item in enumerate(cg_mapping_.items()):
        general_mapping[i] = tuple([item[0], *item[1]])
    if special_terminal:
        keys = list(cg_mapping_)
        cg_mapping_[keys[0]][0] += "-terminal"
        cg_mapping_[keys[0]][1] += len(cg_mapping)
        cg_mapping_[keys[-1]][0] += "-terminal"
        cg_mapping_[keys[-1]][1] += len(cg_mapping)

    n_beads = len(cg_mapping_)

    cg_types = np.array([cg_type for (_, cg_type, _) in cg_mapping_.values()])
    cg_masses = np.array([cg_mass for (_, _, cg_mass) in cg_mapping_.values()])

    cg_matrix = np.zeros((n_beads, n_atoms))
    for i_cg, i_at in enumerate(cg_mapping_.keys()):
        cg_matrix[i_cg, i_at] = 1
    return cg_types, cg_masses, cg_matrix, general_mapping


def build_cg_topology(
    topology: Topology,
    cg_mapping: Dict[Tuple[str, str], Tuple[str, int, int]] = None,
    special_terminal: bool = True,
    bonds: Optional[Callable] = add_chain_bonds,
    angles: Optional[Callable] = add_chain_angles,
    dihedrals: Optional[Callable] = add_chain_dihedrals,
) -> Topology:
    r"""Takes an `mlcg.geometry.topology.Topology` instance and returns another
    `mlcg.geometry.topology.Topology` instance conditioned on the supplied
    CG mapping

    Parameters
    ----------
    topology:
       Original MLCG topology before coarse graining
    cg_mapping:
       A suitable CG mapping. See mclg.cg._mapping.py for examples.
    special_termini:
       If True, the first and last CG atoms recieve their own special
       types
    bonds:
       Function to enumerate and define bonds in the final CG topology
    angles:
       Function to enumerate and define angles in the final CG topology
    dihedrals:
       Function to enumerate and define dihedrals in the final CG topology

    Returns
    -------
    Topology:
        CG topoology
    """
    cg_topo = Topology()
    for at in topology.atoms:
        (cg_name, cg_type, _) = cg_mapping.get(
            (at.resname, at.name), (None, None, None)
        )
        if cg_name is None:
            continue
        cg_topo.add_atom(cg_type, cg_name, at.resname, at.resid)

    if special_terminal:
        cg_topo.names[0] += "-terminal"
        cg_topo.types[0] += len(cg_mapping)
        cg_topo.names[-1] += "-terminal"
        cg_topo.types[-1] += len(cg_mapping)

    if bonds is not None:
        bonds(cg_topo)
    if angles is not None:
        angles(cg_topo)
    if dihedrals is not None:
        dihedrals(cg_topo)

    return cg_topo


def swap_mapping_rows(
    topology: Topology,
    residue: str,
    target_atoms: List[str],
    types: np.ndarray,
    masses: np.ndarray,
    matrix: np.ndarray,
    mapping: OrderedDict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Helper function to swap atom rows systematically in
    the specified residue. The last four arguments are the
    outputs of mlcg.cg.projection.build_cg_matrix().

    Parameters
    ----------
    topology:
        Topology of the all-atom (non-CG system).
    residue:
        Three letter amino acid string specifying the
        residue in which swaps must be made
    target_atoms:
        List of atom types to swap places
    types:
        The types array that will by swapped at the
        specified atoms for the specified residue
    masses:
        The mass array that will be swapped at the
        specified atoms for the specified residue
    matrix:
        The CG mapping matrix for which rows will be
        swapped at the specified atoms for the specifed
        residue
    mapping:
        The mapping dictionary from the original topology
        to the CG atoms

    Returns
    -------
    swapped_types
    swapped_masses
    swapped_matrix
    generalized_mapping
    """

    rows_to_swap = []  # buffer for row sets to swap
    cg_list = []  # buffer for cg atoms to swap
    aa_list = []  # buffer for aa atoms to swap
    generalized_mapping = {}
    if not isinstance(mapping, OrderedDict):
        raise TypeError("Mapping must be ordered dictionary.")

    for cg_idx, item in enumerate(mapping.items()):
        cg_idx, cg_props = item
        aa_idx = cg_props[0]
        if topology.resnames[aa_idx] == residue:
            if topology.names[aa_idx] in target_atoms:
                cg_list.append(cg_idx)
                aa_list.append(aa_idx)
                assert len(cg_list) == len(aa_list)
        if len(cg_list) > 1 or len(aa_list) > 1:
            rows_to_swap.append(cg_list)
            cg_list = []
            aa_list = []

        generalized_mapping[cg_idx] = cg_props

    swapped_types = deepcopy(types)
    swapped_masses = deepcopy(masses)
    swapped_matrix = deepcopy(matrix)

    for row_set in rows_to_swap:
        swapped_types[[row_set[0], row_set[1]]] = types[
            [row_set[1], row_set[0]]
        ]
        swapped_masses[[row_set[0], row_set[1]]] = masses[
            [row_set[1], row_set[0]]
        ]
        swapped_matrix[[row_set[0], row_set[1]]] = matrix[
            [row_set[1], row_set[0]]
        ]
        map_element_1 = generalized_mapping[row_set[0]]
        map_element_2 = generalized_mapping[row_set[1]]
        generalized_mapping[row_set[0]] = map_element_2
        generalized_mapping[row_set[1]] = map_element_1

    generalized_mapping = OrderedDict(generalized_mapping)

    return swapped_types, swapped_masses, swapped_matrix, generalized_mapping


def make_opep_residue_indices(resnames: List[str]) -> List[int]:
    """Helper method to make residue index lists, which specify
    the residue index that a given atom belongs to in an
    octapeptide topology using the octapeptide CG mapping, wherein
    non-glycine amino acids are given a 5 atom mapping (N,CA,CB,C,O)
    while glycines are given a 4 atom mapping (N,CA,C,O).

    Parameters
    ----------
    resnames:
        List of three letter amino acid codes that together specify
        the molecular sequence

    Returns
    -------
    residue_indices:
        List of residue indices that specify which specify to which residue
        an atom belongs
    """

    assert all(len(res) == 3 for res in resnames)
    gly_count = resnames.count("GLY")
    num_atoms = 5 * len(resnames) - gly_count

    residue_indices = []
    for res_idx, name in enumerate(resnames):
        if name == "GLY":
            for _ in range(4):
                residue_indices.append(res_idx)
        if name != "GLY":
            for _ in range(5):
                residue_indices.append(res_idx)

    assert len(residue_indices) == num_atoms
    return residue_indices


def build_opeps_connectivity_matrix(resnames: List[str]) -> torch.Tensor:
    """Builds bonded connectivity matrix for (N,CA,CB,C,O)
    octapeptide CG mapping scheme, based on amino acid sequence.
    GLY residues are constructed as (N,CA,C,O).

    Parameters
    ----------
    resnames:
        List of three letter amino acid strings that specify
        the protein/peptide sequence.

    Returns
    -------
    connectivity_matrix:
        Undirected graph connectivity matrix for the specified sequence
        of residues.
    """
    assert all(len(res) == 3 for res in resnames)

    gly_count = resnames.count("GLY")
    num_atoms = 5 * len(resnames) - gly_count
    connectivity_matrix = np.zeros((num_atoms, num_atoms))
    gly_passed = 0
    for i, name in enumerate(resnames):
        print(name)
        idx = (5 * i) - gly_passed
        if name == "GLY":
            a1, a2, a3, a4, a5 = idx, idx + 1, idx + 2, idx + 3, idx + 4
            connectivity_matrix[a1, a2] = 1
            connectivity_matrix[a2, a3] = 1
            connectivity_matrix[a3, a4] = 1
            if i < (len(resnames) - 1):
                connectivity_matrix[a3, a5] = 1
            gly_passed += 1
        if name != "GLY":
            a1, a2, a3, a4, a5, a6 = (
                idx,
                idx + 1,
                idx + 2,
                idx + 3,
                idx + 4,
                idx + 5,
            )
            connectivity_matrix[a1, a2] = 1
            connectivity_matrix[a2, a3] = 1
            connectivity_matrix[a2, a4] = 1
            connectivity_matrix[a4, a5] = 1
            if i < (len(resnames) - 1):
                connectivity_matrix[a4, a6] = 1
    connectivity_matrix += connectivity_matrix.T
    return torch.tensor(connectivity_matrix)


def isolate_features(
    target_atoms_list: List[List[int]], full_feature_set: torch.Tensor
) -> List[torch.Tensor]:
    """Helper function for isolating molecular features from
    specified lists of beads.


    Parameters
    ----------
    target_atoms_list:
        if a feature atom from full_feature_set is in this list, that corresponding feature
        is moved to the isolated feature list. If no feature atom is in this list,
        then the corresponding feature remains in the bulk feature list
    full_feature_set:
        The set of full features that is checked against the target_atoms

    Returns
    -------
    feature_sets:
        List of N feature sets, where the length of target_atoms_list is N-1 in length,
        the first N-1 features sets correspond to those isolated features, and the final
        feauture set is the list of bulk/non-isolated features.
    """

    full_feature_set = full_feature_set.numpy()
    order = full_feature_set.shape[1]
    isolated_features = [
        tuple([[] for _ in range(order)]) for _ in range(len(target_atoms_list))
    ]
    bulk_features = tuple([[] for _ in range(order)])

    for i in range(full_feature_set.shape[1]):
        bulk_feature = True
        atoms = full_feature_set[:, i]
        for j, target_list in enumerate(target_atoms_list):
            if any([atom in target_list for atom in atoms]):
                for k, atom in enumerate(atoms):
                    isolated_features[j][k].append(atom)
                bulk_feature = False
        if bulk_feature == True:
            for k, atom in enumerate(atoms):
                bulk_features[j][k].append(atom)

    feature_sets = [
        torch.tensor(feature_set) for feature_set in isolated_features
    ]
    bulk_features = torch.tensor(bulk_bonds)
    feature_sets += bulk_features
    return feature_sets


def get_pseudobond_edges(
    bond_types, atom_types, connectivity_matrix, separation
) -> np.ndarray:
    pairs = get_n_pairs(connectivity_matrix, n=separation)
    pbonds = ([], [])
    for j in range(len(bond_types)):
        t1, t2 = bond_types[j][0], bond_types[j][1]
        for i in range(pairs.shape[1]):
            a1, a2 = pairs[0, i], pairs[1, i]
            if atom_types[a1] == t1 and atom_types[a2] == t2:
                pbonds[0].append(a1)
                pbonds[1].append(a2)
            elif atom_types[a1] == t2 and atom_types[a2] == t1:
                pbonds[0].append(a1)
                pbonds[1].append(a2)
            else:
                continue
    # sort based on the first bond indices
    pbonds = np.array(pbonds).T
    pbonds = sorted(pbonds, key=lambda x: x[0])
    return np.array(pbonds).T


def make_non_bonded_set(
    topology: Topology,
    minimum_separation: int,
    residue_indices: List[int],
    residue_exclusion=True,
    edge_exclusion: Optional[List[torch.Tensor]] = None,
):
    """Helper function for constructing non-bonded sets from a topology

    Parameters
    ----------
    topology;
        input topology
    minimum_separation:
        minimum edge separation between pairs of atoms for the non-bonded set
    residue_indices:
        list of indices mapping atoms to residue indices
    residue_exclusion:
        if True, an extra exclusion rule is applied such that if any two atoms are within
        the same residue they are automatically removed from the non-bonded set.
    edge_exclusion:
        if None, this list of pairwise edge indices will be excluded from the non-bonded
        set.

    Returns
    -------
    non_bonded_edges:
        The set of non-bonded edges
    """

    if residue_indices is None and residue_exclusion == True:
        raise ValueError(
            "If residue exclusion rule is used, residue indices must not be None."
        )
    if residue_exclusion == True:
        assert len(residue_indices) == len(topology.types)

    fully_connected_edges = _symmetrise_distance_interaction(
        topology.fully_connected2torch()
    )
    fully_connected_edges = torch.unique(fully_connected_edges, dim=1)
    fully_connected_edges = fully_connected_edges.numpy()

    connectivity_matrix = get_connectivity_matrix(topology)
    graph = nx.Graph(connectivity_matrix.numpy())
    non_bonded_edges = ([], [])
    for i in range(fully_connected_edges.shape[1]):
        bonded = False
        edge = tuple(fully_connected_edges[:, i])

        # apply residue exclusion
        if residue_exclusion:
            if np.abs(residue_indices[edge[0]] - residue_indices[edge[1]]) == 0:
                bonded = True

        # apply edge exclusion
        if edge_exclusion != None:
            for edge_list in edge_exclusion:
                for excluded_edge in edge_list:
                    if (
                        edge[0] == excluded_edge[0]
                        and edge[1] == excluded_edge[1]
                    ):
                        bonded = True
                    if (
                        edge[0] == excluded_edge[1]
                        and edge[1] == excluded_edge[0]
                    ):
                        bonded = True

        # apply minimum exclusion
        if len(bidirectional_shortest_path(graph, edge[0], edge[1])) < 4:
            bonded = True
        if bonded == False:
            non_bonded_edges[0].append(edge[0])
            non_bonded_edges[1].append(edge[1])

    non_bonded_edges = torch.tensor(non_bonded_edges)
    non_bonded_edges = torch.unique(
        _symmetrise_distance_interaction(non_bonded_edges), dim=1
    )
    return non_bonded_edges
