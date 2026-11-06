import torch
from torch_geometric.data.collate import collate
import pytest
from typing import List
import warnings

from mlcg.nn.schnet import StandardSchNet
from mlcg.nn.radial_basis import GaussianBasis
from mlcg.nn.gradients import GradientsOut
from mlcg.nn.cutoff import IdentityCutoff, CosineCutoff
from mlcg.geometry import Topology
from mlcg.data.atomic_data import AtomicData
from mlcg.data._keys import ENERGY_KEY, FORCE_KEY
from mlcg.neighbor_list.neighbor_list import atomic_data2neighbor_list
from ase.build import molecule


class MolDatabase(object):
    """Container for ASE molecules for testing"""

    def __init__(
        self,
        mol_names: List[str] = [
            "AlF3",
            "C2H3",
            "ClF",
            "PF3",
            "PH2",
            "CH3CN",
            "cyclobutene",
            "CH3ONO",
            "SiH3",
            "C3H6_D3h",
            "CO2",
            "NO",
            "trans-butane",
            "H2CCHCl",
            "LiH",
            "NH2",
            "CH",
            "CH2OCH2",
            "C6H6",
            "CH3CONH2",
            "cyclobutane",
            "H2CCHCN",
            "butadiene",
            "C",
            "H2CO",
            "CH3COOH",
            "HCF3",
            "CH3S",
            "CS2",
        ],
    ):
        self.mol_names = mol_names
        self.molecules = [molecule(name) for name in self.mol_names]
        self.mol_topos = [Topology.from_ase(mol) for mol in self.molecules]
        self.data_list = []
        data_list = []
        for mol, topo in zip(self.molecules, self.mol_topos):
            neighbor_list = topo.neighbor_list("fully connected")
            data = AtomicData.from_points(
                pos=torch.tensor(mol.get_positions()).float(),
                atom_types=torch.tensor(mol.get_atomic_numbers()),
                neighbor_list=neighbor_list,
            )
            data_list.append(data)

        self.collated_data, _, _ = collate(
            data_list[0].__class__,
            data_list=data_list,
            increment=True,
            add_batch=True,
        )
        self.force_shape = self.collated_data.pos.shape
        self.energy_shape = torch.Size([len(self.molecules)])
        self.atomic_numbers = sorted(
            torch.unique(self.collated_data.atom_types).numpy().tolist()
        )


standard_basis = GaussianBasis(cutoff=5)
standard_cutoff = IdentityCutoff(cutoff_lower=0, cutoff_upper=5)

# prepare some data with ASE

database = MolDatabase()


@pytest.mark.parametrize(
    "basis, cutoff, expected_warning",
    [
        (GaussianBasis(IdentityCutoff(1, 5)), CosineCutoff(0, 5), UserWarning),
    ],
)
def test_cutoff_warning(basis, cutoff, expected_warning):
    with pytest.warns(expected_warning):
        StandardSchNet(basis, cutoff, [128, 128])


@pytest.mark.parametrize(
    "basis, cutoff, expected_warning",
    [
        (GaussianBasis(IdentityCutoff(0, 5)), CosineCutoff(0, 5), None),
    ],
)
def test_cutoff_warning_None(basis, cutoff, expected_warning):
    with warnings.catch_warnings(record=True):
        StandardSchNet(basis, cutoff, [128, 128])


def test_minimum_interaction_block():
    with pytest.raises(ValueError):
        StandardSchNet(
            standard_basis, standard_cutoff, [128, 128], num_interactions=-1
        )


@pytest.mark.parametrize(
    "collated_data, out_keys, expected_shapes",
    [
        (
            database.collated_data,
            [ENERGY_KEY, FORCE_KEY],
            [database.energy_shape, database.force_shape],
        )
    ],
)
def test_prediction(collated_data, out_keys, expected_shapes):
    """Test to make sure that the output dictionary is properly populated
    and that the correspdonding shapes of the outputs are correct given the
    requested gradient targets.
    """
    test_schnet = StandardSchNet(standard_basis, standard_cutoff, [128, 128])
    model = GradientsOut(test_schnet, targets=FORCE_KEY).double()
    collated_data = model(collated_data)
    assert len(collated_data.out) != 0
    assert "SchNet" in collated_data.out.keys()
    for key, shape in zip(out_keys, expected_shapes):
        assert key in collated_data.out[model.name].keys()
        assert collated_data.out[model.name][key].shape == shape


def _make_excluded_data(
    mol_names: List[str],
    excluded_order: List[int] = None,
    both_directions: bool = True,
) -> AtomicData:
    data_list = []
    for i, name in enumerate(mol_names):
        mol = molecule(name)
        kwargs = {}
        if excluded_order is not None:
            bonds = Topology.from_ase(mol).bonds2torch()
            if both_directions:
                bonds = torch.cat([bonds, bonds.flip(0)], dim=1)
            kwargs["excluded_pairs"] = {
                "excluded_order": excluded_order[i],
                "index_mapping": bonds,
            }
        data_list.append(
            AtomicData.from_points(
                pos=torch.tensor(mol.get_positions()).float(),
                atom_types=torch.tensor(mol.get_atomic_numbers()),
                **kwargs,
            )
        )
    collated_data, _, _ = collate(
        data_list[0].__class__,
        data_list=data_list,
        increment=True,
        add_batch=True,
    )
    return collated_data


excl_mol_names = ["CH3CN", "C6H6", "trans-butane"]


def test_excluded_pairs_removed_from_neighbor_list():
    data_no_excl = _make_excluded_data(excl_mol_names)
    data_excl = _make_excluded_data(excl_mol_names, [2] * len(excl_mol_names))

    nl_full = atomic_data2neighbor_list(data_no_excl, rcut=5.0)
    nl_excl = atomic_data2neighbor_list(data_excl, rcut=5.0)

    num_nodes = data_excl.pos.shape[0]
    excluded = data_excl.excluded_pairs["index_mapping"]
    excluded_key = excluded[0] * num_nodes + excluded[1]
    edge_key_full = (
        nl_full["index_mapping"][0] * num_nodes + nl_full["index_mapping"][1]
    )
    edge_key_excl = (
        nl_excl["index_mapping"][0] * num_nodes + nl_excl["index_mapping"][1]
    )

    assert nl_full["excluded_order"] == 1
    assert nl_excl["excluded_order"] == 2
    assert torch.isin(excluded_key, edge_key_full).all()
    assert not torch.isin(excluded_key, edge_key_excl).any()
    assert edge_key_excl.numel() == edge_key_full.numel() - excluded_key.numel()
    assert nl_excl["cell_shifts"].shape[0] == edge_key_excl.numel()


def test_excluded_pairs_are_unordered():
    """Each pair stored once excludes both directions, as both stored."""
    order = [2] * len(excl_mol_names)
    nl_both = atomic_data2neighbor_list(
        _make_excluded_data(excl_mol_names, order), rcut=5.0
    )
    nl_once = atomic_data2neighbor_list(
        _make_excluded_data(excl_mol_names, order, both_directions=False),
        rcut=5.0,
    )
    assert torch.equal(nl_once["index_mapping"], nl_both["index_mapping"])
    assert torch.equal(nl_once["cell_shifts"], nl_both["cell_shifts"])


@pytest.mark.parametrize(
    "model_excluded_order, data_excluded_order, expect_error",
    [
        (2, 2, False),
        (2, None, True),
        (1, 2, True),
        (1, None, False),
    ],
)
def test_excluded_order_consistency(
    model_excluded_order, data_excluded_order, expect_error
):
    data_excluded_order = (
        None
        if data_excluded_order is None
        else [data_excluded_order] * len(excl_mol_names)
    )
    data = _make_excluded_data(excl_mol_names, data_excluded_order)
    model = StandardSchNet(
        GaussianBasis(cutoff=5),
        IdentityCutoff(cutoff_lower=0, cutoff_upper=5),
        [128, 128],
        excluded_order=model_excluded_order,
    )
    if expect_error:
        with pytest.raises(ValueError):
            model(data)
    else:
        model(data)


@pytest.mark.parametrize(
    "data_excluded_order, expect_error", [(None, False), (2, True)]
)
def test_excluded_order_old_model(data_excluded_order, expect_error):
    """Models pickled before `excluded_order` existed lack the attribute
    and must behave as `excluded_order=1`."""
    data_excluded_order = (
        None
        if data_excluded_order is None
        else [data_excluded_order] * len(excl_mol_names)
    )
    data = _make_excluded_data(excl_mol_names, data_excluded_order)
    model = StandardSchNet(
        GaussianBasis(cutoff=5),
        IdentityCutoff(cutoff_lower=0, cutoff_upper=5),
        [128, 128],
    )
    del model.excluded_order
    if expect_error:
        with pytest.raises(ValueError):
            model(data)
    else:
        model(data)


def test_inconsistent_excluded_order_in_batch():
    data = _make_excluded_data(excl_mol_names, [2, 3, 2])
    model = StandardSchNet(
        GaussianBasis(cutoff=5),
        IdentityCutoff(cutoff_lower=0, cutoff_upper=5),
        [128, 128],
        excluded_order=2,
    )
    with pytest.raises(ValueError):
        model(data)


def test_invalid_excluded_order():
    with pytest.raises(ValueError):
        StandardSchNet(
            GaussianBasis(cutoff=5),
            IdentityCutoff(cutoff_lower=0, cutoff_upper=5),
            [128, 128],
            excluded_order=0,
        )
