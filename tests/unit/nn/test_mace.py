import pytest

try:
    from typing import List
    from ase.build import molecule
    from mlcg.geometry import Topology
    from torch_geometric.data.collate import collate

    import mace
    from mlcg.nn.mace import StandardMACE
    from mlcg.data.atomic_data import AtomicData

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
    test_mace = StandardMACE(**mace_config)

except:
    pytest.skip("MACE installation not found...", allow_module_level=True)

import torch

from mlcg.nn.gradients import GradientsOut
from mlcg.data._keys import ENERGY_KEY, FORCE_KEY


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
