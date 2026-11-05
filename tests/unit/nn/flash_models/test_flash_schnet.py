import torch
from torch_geometric.data.collate import collate
import pytest
from typing import List

from mlcg.nn.schnet import StandardSchNet
from mlcg.nn.radial_basis import ExpNormalBasis
from mlcg.nn.gradients import GradientsOut, SumOut
from mlcg.nn.prior import HarmonicBonds
from mlcg.nn.flash_models import enable_flash_kernels
from mlcg.nn.cutoff import IdentityCutoff
from mlcg.geometry import Topology
from mlcg.data.atomic_data import AtomicData
from mlcg.data._keys import ENERGY_KEY, FORCE_KEY
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


standard_basis = ExpNormalBasis(cutoff=5)
standard_cutoff = IdentityCutoff(cutoff_lower=0, cutoff_upper=5)

# prepare some data with ASE

database = MolDatabase()


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
    test_flash_schnet = StandardSchNet(
        standard_basis, standard_cutoff, [128, 128]
    )
    test_flash_schnet.enable_kernels()
    model = GradientsOut(test_flash_schnet, targets=FORCE_KEY).float()
    collated_data = model(collated_data)
    assert len(collated_data.out) != 0
    assert "SchNet" in collated_data.out.keys()
    for key, shape in zip(out_keys, expected_shapes):
        assert key in collated_data.out[model.name].keys()
        assert collated_data.out[model.name][key].shape == shape


@pytest.mark.parametrize(
    "collated_data, out_keys",
    [
        (
            database.collated_data,
            [ENERGY_KEY, FORCE_KEY],
        )
    ],
)
def test_flash_vs_normal(collated_data, out_keys):
    """Test to make sure that the standard schnet and the flash version
    output the same values up to a tolerance
    """
    test_flash_schnet = StandardSchNet(
        standard_basis, standard_cutoff, [128, 128]
    )
    test_flash_schnet.enable_kernels()
    flash_model = GradientsOut(test_flash_schnet, targets=FORCE_KEY).float()
    test_schnet = StandardSchNet(standard_basis, standard_cutoff, [128, 128])
    # make them have the same parameters
    test_schnet.load_state_dict(flash_model.model.state_dict())
    normal_model = GradientsOut(test_schnet, targets=FORCE_KEY).float()
    out_flash = flash_model(collated_data)
    out_normal = normal_model(collated_data)
    for key in out_keys:
        torch.testing.assert_close(
            out_flash.out[flash_model.name][key],
            out_normal.out[normal_model.name][key],
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels")
def test_flash_force_training_gradients():
    """Parameter gradients of a force loss (the double backward, through the
    radius graph used as its own transpose) are as close to the standard
    model in float64 as the standard model in float32 is."""
    models = [
        StandardSchNet(ExpNormalBasis(cutoff=5), standard_cutoff, [32, 32])
        for _ in range(3)
    ]
    for model in models[1:]:
        model.load_state_dict(models[0].state_dict())
    models[0].double()
    models[2].enable_kernels()

    def grads(schnet):
        model = GradientsOut(schnet.cuda(), targets=FORCE_KEY)
        data = database.collated_data.clone().to("cuda")
        data.pos = data.pos.to(next(schnet.parameters()).dtype)
        forces = model(data).out[model.name][FORCE_KEY]
        forces.pow(2).sum().backward()
        return torch.cat(
            [p.grad.flatten().double() for p in schnet.parameters()]
        )

    reference, standard, flash = (grads(model) for model in models)
    # the standard float32 error varies by ~1.4x between runs (atomic sums;
    # tests_benchmarks/30_edges/grad_check.log)
    standard_error = (standard - reference).abs().max()
    assert (flash - reference).abs().max() <= 2 * standard_error


def test_flash_reaches_nested_modules():
    """Every module with a flash path gets its kernels, at any depth and
    inside containers without one (a SumOut of a plain dict in a
    Sequential)."""
    schnet = StandardSchNet(standard_basis, standard_cutoff, [128, 128])
    bonds = HarmonicBonds({(1, 6): {"k": 1.0, "x_0": 1.0}})
    model = torch.nn.Sequential(
        SumOut(
            {
                "SchNet": GradientsOut(schnet, targets=FORCE_KEY),
                "bonds": GradientsOut(bonds, targets=FORCE_KEY),
            }
        )
    )
    enable_flash_kernels(model)
    assert schnet._kernels and bonds._kernels
    assert all(block.conv._kernels for block in schnet.interaction_blocks)
