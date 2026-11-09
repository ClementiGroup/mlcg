import copy
import pytest
import torch
from typing import List
from ase.build import molecule
from torch_geometric.data.collate import collate

from mlcg.geometry import Topology
from mlcg.nn.mace import StandardMACE, OEQ_AVAILABLE
from mlcg.nn import load_and_adapt_old_checkpoint
from mlcg.data.atomic_data import AtomicData
from mlcg.nn.gradients import GradientsOut
from mlcg.data._keys import ENERGY_KEY, FORCE_KEY
from mlcg.neighbor_list.neighbor_list import atomic_data2neighbor_list
from mlcg.mol_utils import MolDatabase

database = MolDatabase()
mace_config = {
    "r_max": 10,
    "num_bessel": 10,
    "num_polynomial_cutoff": 5,
    "max_ell": 1,
    "interaction_cls": "mace.modules.blocks.RealAgnosticResidualInteractionBlock",
    "interaction_cls_first": "mace.modules.blocks.RealAgnosticResidualInteractionBlock",
    "num_interactions": 1,
    "hidden_irreps": "32x0e",
    "MLP_irreps": "16x0e",
    "avg_num_neighbors": 9,
    "correlation": 2,
    "gate": torch.nn.Tanh(),
    "max_num_neighbors": 1000,
    "pair_repulsion": False,
    "distance_transform": None,
    "radial_MLP": [32, 32],
    "radial_type": "bessel",
    "atomic_numbers": database.atomic_numbers,
}
mace_model = StandardMACE(**mace_config)


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
    print(collated_data.pos.dtype)

    test_mace = StandardMACE(**mace_config)
    model = GradientsOut(test_mace, targets=FORCE_KEY).float()
    collated_data = model(collated_data)
    assert len(collated_data.out) != 0
    assert "mace" in collated_data.out.keys()
    for key, shape in zip(out_keys, expected_shapes):
        assert key in collated_data.out[model.name].keys()
        assert collated_data.out[model.name][key].shape == shape


def _conv_tps(model):
    return [
        module.conv_tp
        for module in model.modules()
        if hasattr(module, "conv_tp")
    ]


@pytest.mark.skipif(
    not (OEQ_AVAILABLE and torch.cuda.is_available()),
    reason="openequivariance and a GPU are required",
)
@pytest.mark.parametrize(
    "interaction_cls",
    [
        "mlcg.nn.mace.CustomRealAgnosticResidualInteractionBlock",
        "mace.modules.blocks.RealAgnosticResidualInteractionBlock",
    ],
)
@pytest.mark.parametrize("copy_method", ["deepcopy", "save_load"])
def test_oeq_survives_copy(interaction_cls, copy_method, tmp_path):
    """Test that openequivariance models still predict the same energies and
    forces after being deep-copied (as done by `PLModel.get_model`) or saved
    and loaded again (as done by `load_and_adapt_old_checkpoint`), since
    openequivariance drops the MACE adaptation of `conv_tp.forward` when
    copied or unpickled.
    """
    config = {
        **mace_config,
        "hidden_irreps": "16x0e + 16x1o",
        "num_interactions": 2,
        "interaction_cls": interaction_cls,
        "interaction_cls_first": interaction_cls,
        "use_oeq": True,
    }
    model = GradientsOut(StandardMACE(**config), targets=FORCE_KEY)
    model = model.float().to("cuda")
    assert len(_conv_tps(model)) == 2

    if copy_method == "deepcopy":
        copied_model = copy.deepcopy(model)
    else:
        torch.save(model, tmp_path / "model.pt")
        copied_model = load_and_adapt_old_checkpoint(tmp_path / "model.pt")

    for conv_tp in _conv_tps(copied_model):
        assert "forward" in conv_tp.__dict__

    data = database.collated_data.to("cuda")
    out = model(data.clone()).out[model.name]
    copied_out = copied_model(data.clone()).out[model.name]
    for key in [ENERGY_KEY, FORCE_KEY]:
        torch.testing.assert_close(out[key], copied_out[key])


def _make_excluded_data(
    mol_names: List[str], excluded_order: List[int] = None
) -> AtomicData:
    data_list = []
    for i, name in enumerate(mol_names):
        mol = molecule(name)
        kwargs = {}
        if excluded_order is not None:
            bonds = Topology.from_ase(mol).bonds2torch()
            kwargs["excluded_pairs"] = {
                "excluded_order": excluded_order[i],
                "index_mapping": torch.cat([bonds, bonds.flip(0)], dim=1),
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
    model = StandardMACE(
        **mace_config,
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
    model = StandardMACE(**mace_config)
    del model.excluded_order
    if expect_error:
        with pytest.raises(ValueError):
            model(data)
    else:
        model(data)


def test_inconsistent_excluded_order_in_batch():
    data = _make_excluded_data(excl_mol_names, [2, 3, 2])
    model = StandardMACE(**mace_config, excluded_order=2)
    with pytest.raises(ValueError):
        model(data)


def test_invalid_excluded_order():
    with pytest.raises(ValueError):
        StandardMACE(**mace_config, excluded_order=0)
