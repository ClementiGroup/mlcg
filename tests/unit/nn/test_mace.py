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
