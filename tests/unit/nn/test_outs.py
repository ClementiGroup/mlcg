from copy import deepcopy
from typing import Dict, Union, List
import torch
import torch._functorch.config
import pytest
import numpy as np
from torch_geometric.data.collate import collate

from ase.build import molecule
from mlcg.geometry import Topology
from mlcg.geometry.topology import get_connectivity_matrix, get_n_paths
from mlcg.geometry.statistics import fit_baseline_models
from mlcg.neighbor_list.neighbor_list import make_neighbor_list
from mlcg.nn.prior import HarmonicBonds, HarmonicAngles, Dihedral
from mlcg.nn.gradients import SumOut, GradientsOut
from mlcg.nn.schnet import StandardSchNet
from mlcg.nn.cutoff import CosineCutoff
from mlcg.nn.radial_basis import GaussianBasis
from mlcg.data._keys import ENERGY_KEY, FORCE_KEY
from mlcg.data.atomic_data import AtomicData
from mlcg.mol_utils import _ASE_prior_model


@pytest.fixture
def ASE_prior_model():
    return _ASE_prior_model


test_mol = molecule("CH3CH2NH2")
test_topo = Topology.from_ase(test_mol)
unique_test_types = sorted(np.unique(test_topo.types).tolist())


class DummyGradientModel(object):
    """Minimal object for model checking"""

    def __init__(self, model_name):
        self.name = model_name


HAS_MACE = True
try:
    import mace
    from mlcg.nn.mace import StandardMACE

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
        "radial_MLP": [64, 64],
        "radial_type": "bessel",
        "atomic_numbers": unique_test_types,
    }
    mace_model = StandardMACE(**mace_config)
    mace_force_model = GradientsOut(mace_model, targets=[FORCE_KEY])#.float()
except Exception as e:
    print(e)
    mace_force_model = DummyGradientModel("mace")
    print("MACE installation not found")
    HAS_MACE = False


standard_cutoff = CosineCutoff(cutoff_lower=0, cutoff_upper=5)
standard_basis = GaussianBasis(cutoff=standard_cutoff)

schnet = StandardSchNet(
    standard_basis,
    standard_cutoff,
    [10],
    hidden_channels=10,
    embedding_size=10,
    num_filters=10,
    num_interactions=1,
    max_num_neighbors=1000,
)
schnet_force_model = GradientsOut(schnet, targets=[FORCE_KEY])#.double()


@pytest.mark.parametrize(
    "ASE_prior_model, out_targets",
    [
        (
            ASE_prior_model,
            [ENERGY_KEY, FORCE_KEY],
        )
    ],
    indirect=["ASE_prior_model"],
)
def test_outs(ASE_prior_model, out_targets):
    """Test to make sure that the output dictionary is properly populated
    and that the correspdonding shapes of the outputs are correct given the
    requested gradient targets.
    """
    data_dictionary = ASE_prior_model()

    mol = data_dictionary["molecule"]
    model = data_dictionary["model"]
    collated_data = data_dictionary["collated_prior_data"]
    atom_types = torch.tensor(mol.get_atomic_numbers())
    force_shape = collated_data.pos.shape
    energy_shape = torch.Size([data_dictionary["num_examples"]])
    expected_shapes = [energy_shape, force_shape]

    collated_data = model(collated_data)

    assert len(collated_data.out) != 0
    for name in model.models.keys():
        assert name in collated_data.out.keys()

    for target, shape in zip(model.targets, expected_shapes):
        assert target in collated_data.out.keys()
        assert shape == collated_data.out[target].shape


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "ASE_prior_model, network_model, out_targets",
    [
        (ASE_prior_model, schnet_force_model, [ENERGY_KEY, FORCE_KEY]),
        (ASE_prior_model, mace_force_model, [ENERGY_KEY, FORCE_KEY]),
    ],
    indirect=["ASE_prior_model"],
)
def test_sum_outs(ASE_prior_model, network_model, out_targets, device):
    """Tests property aggregating with SumOut"""
    if network_model.name == "mace" and HAS_MACE == False:
        pytest.skip("Skipping test, MACE installation not found...")

    torch._functorch.config.donated_buffer = False

    data_dictionary = ASE_prior_model(sum_out=False, device=device)

    prior_model = data_dictionary["model"]
    network_model = network_model.to(device)
    collated_data = data_dictionary["collated_prior_data"].to(device)
    collated_data_2 = deepcopy(collated_data)
    collated_data_3 = deepcopy(collated_data)

    for prior in prior_model.keys():
        collated_data = prior_model[prior](collated_data)
    collated_data = network_model(collated_data)
    target_totals = {target: 0.00 for target in out_targets}
    for target in out_targets:
        for key in collated_data.out.keys():
            print(key)
            target_totals[target] += collated_data.out[key][target]

    module_collection = torch.nn.ModuleDict()
    for key in prior_model.keys():
        module_collection[key] = prior_model[key]
    module_collection[network_model.name] = network_model
    aggregate_model = SumOut(module_collection, out_targets)
    collated_data_2 = aggregate_model(collated_data_2)

    # Test to make sure the the aggregate data matches the target totals
    for target in out_targets:
        torch.testing.assert_close(
            target_totals[target].detach(),
            collated_data_2.out[target].detach(),
            atol=1e-5,
            rtol=1e-5,
        )

    # Compiled: the same equivalence should hold once the aggregate model
    # is compiled. A looser tolerance is used here since torch.compile's
    # Inductor backend fuses and reorders reductions, so it will not match
    # eager execution bit-for-bit.
    compiled_model = torch.compile(aggregate_model, dynamic=True)
    collated_data_3 = compiled_model(collated_data_3)

    for target in out_targets:
        torch.testing.assert_close(
            target_totals[target].detach(),
            collated_data_3.out[target].detach(),
            atol=1e-4,
            rtol=5e-4,
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "ASE_prior_model, network_model, out_targets",
    [
        (ASE_prior_model, schnet_force_model, [ENERGY_KEY, FORCE_KEY]),
        (ASE_prior_model, mace_force_model, [ENERGY_KEY, FORCE_KEY]),
    ],
    indirect=["ASE_prior_model"],
)
def test_sum_outs_own_nl_models(
    ASE_prior_model, network_model, out_targets, device
):
    """Tests that withholding the shared ``data.neighbor_list`` from a
    network model via ``SumOut(..., own_nl_models=[...])`` leaves the
    aggregated predictions unchanged, since the withheld model builds its
    own coordinate-dependent neighbor list on the fly. This is checked
    against a reference model that shares the neighbor list with every
    model, both in eager mode and after compiling the aggregate model with
    ``torch.compile``, on CPU and, when available, on GPU.
    """
    if network_model.name == "mace" and HAS_MACE == False:
        pytest.skip("Skipping test, MACE installation not found...")

    torch._functorch.config.donated_buffer = False

    data_dictionary = ASE_prior_model(sum_out=False, device=device)
    prior_model = data_dictionary["model"]
    collated_data = data_dictionary["collated_prior_data"].to(device)

    def build_module_collection():
        module_collection = torch.nn.ModuleDict()
        for key in prior_model.keys():
            module_collection[key] = deepcopy(prior_model[key])
        module_collection[network_model.name] = deepcopy(network_model).to(
            device
        )
        return module_collection

    reference_model = SumOut(build_module_collection(), out_targets)
    reference_data = reference_model(deepcopy(collated_data))

    # Eager: withholding the shared neighbor list from the network model
    # should not change the aggregated predictions.
    eager_model = SumOut(
        build_module_collection(),
        out_targets,
        own_nl_models=[network_model.name],
    )
    eager_data = eager_model(deepcopy(collated_data))

    for target in out_targets:
        torch.testing.assert_close(
            reference_data.out[target].detach(),
            eager_data.out[target].detach(),
            atol=1e-5,
            rtol=1e-5,
        )

    # Compiled: the same equivalence should hold once the aggregate model
    # (including the own_nl_models bookkeeping) is compiled. A looser
    # tolerance is used here since torch.compile's Inductor backend fuses
    # and reorders reductions and does not preserve float64 precision in
    # all kernels, so it will not match eager execution bit-for-bit.
    compiled_model = torch.compile(
        SumOut(
            build_module_collection(),
            out_targets,
            own_nl_models=[network_model.name],
        ),
        dynamic=True,
    )
    compiled_data = compiled_model(deepcopy(collated_data))

    for target in out_targets:
        torch.testing.assert_close(
            reference_data.out[target].detach(),
            compiled_data.out[target].detach(),
            atol=1e-4,
            rtol=5e-4,
        )
