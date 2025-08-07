from copy import deepcopy
from typing import Dict, Union, List
import torch
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
    mace_force_model = GradientsOut(mace_model, targets=[FORCE_KEY]).float()
except ImportError or NameError as e:
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
schnet_force_model = GradientsOut(schnet, targets=[FORCE_KEY]).double()


@pytest.fixture
def ASE_prior_model():
    def _model_builder(
        mol: str = "CH3CH2NH2", sum_out: bool = True
    ) -> Union[torch.nn.Module, torch.nn.ModuleDict]:
        """Fixture that returns a simple prior-only model of
        an ASE molecule with HarmonicBonds and HarmonicAngles
        priors whose parameters are estimated from coordinates
        artifically perturbed by some small Gaussian noise.

        Parameters
        ----------
        mol:
            Molecule specifying string found in ase.build.molecule
            associated with the g2 organic molecule database
        sum_out:
            If True, the model constituents are wrapped within
            a SumOut instance

        Returns
        -------
        model_with_data:
            Dictionary that contains the fitted prior-only model
            under the key "model" and the noisey data, as a collated
            AtomicData instance, used for fitting under the key
            "collated_prior_data"
        """

        # Seeding
        rng = np.random.default_rng(94834)

        # Physical units
        temperature = 350  # K
        #:Boltzmann constan in kcal/mol/K
        kB = 0.0019872041
        beta = 1 / (temperature * kB)

        # Here we make a simple prior-only model of aluminum-fluoride
        # mol = molecule("AlF3")
        # Implement molecule with dihedrals
        mol = molecule(mol)
        test_topo = Topology.from_ase(mol)

        # Add in molecule with dihedral and compute edges
        conn_mat = get_connectivity_matrix(test_topo)
        dihedral_paths = get_n_paths(conn_mat, n=4)
        test_topo.dihedrals_from_edge_index(dihedral_paths)

        n_atoms = len(test_topo.types)
        initial_coords = np.array(mol.get_positions())

        prior_data_frames = []
        for i in range(1000):
            perturbed_coords = initial_coords + 0.2 * rng.standard_normal(
                initial_coords.shape
            )
            prior_data_frames.append(torch.tensor(perturbed_coords))
        prior_data_frames = torch.stack(prior_data_frames, dim=0)

        # Set up some data with bond/angle neighborlists:
        bond_edges = test_topo.bonds2torch()
        angle_edges = test_topo.angles2torch()
        dihedral_edges = test_topo.dihedrals2torch()

        # Generete some noisy data for the priors
        nls_tags = ["bonds", "angles", "dihedrals"]
        nls_orders = [2, 3, 4]
        nls_edges = [bond_edges, angle_edges, dihedral_edges]

        neighbor_lists = {
            tag: make_neighbor_list(tag, order, edge_list)
            for (tag, order, edge_list) in zip(nls_tags, nls_orders, nls_edges)
        }

        prior_data_list = []
        for frame in range(prior_data_frames.shape[0]):
            data_point = AtomicData(
                pos=prior_data_frames[frame].float(),
                atom_types=torch.tensor(test_topo.types),
                masses=torch.tensor(mol.get_masses()).float(),
                cell=None,
                neighbor_list=neighbor_lists,
            )
            prior_data_list.append(data_point)

        collated_prior_data, _, _ = collate(
            prior_data_list[0].__class__,
            data_list=prior_data_list,
            increment=True,
            add_batch=True,
        )
        # Fit the priors
        prior_cls = [HarmonicBonds, HarmonicAngles, Dihedral]
        priors, stats = fit_baseline_models(
            collated_prior_data, beta, prior_cls
        )

        # Construct the model
        priors = {
            name: GradientsOut(priors[name], targets=[FORCE_KEY])
            for name in priors.keys()
        }

        if sum_out:
            full_model = SumOut(priors)
        else:
            full_model = priors

        model_with_data = {
            "model": full_model,
            "collated_prior_data": collated_prior_data,
            "molecule": mol,
            "num_examples": len(prior_data_list),
            "neighbor_lists": neighbor_lists,
        }
        return model_with_data

    return _model_builder


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


@pytest.mark.parametrize(
    "ASE_prior_model, network_model, out_targets",
    [
        (ASE_prior_model, schnet_force_model, [ENERGY_KEY, FORCE_KEY]),
        (ASE_prior_model, mace_force_model, [ENERGY_KEY, FORCE_KEY]),
    ],
    indirect=["ASE_prior_model"],
)
def test_sum_outs(ASE_prior_model, network_model, out_targets):
    """Tests property aggregating with SumOut"""
    if network_model.name == "mace" and HAS_MACE == False:
        pytest.skip("Skipping test, MACE installation not found...")
    data_dictionary = ASE_prior_model(sum_out=False)

    prior_model = data_dictionary["model"]
    collated_data = data_dictionary["collated_prior_data"]
    collated_data_2 = deepcopy(data_dictionary["collated_prior_data"])

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
        np.testing.assert_allclose(
            target_totals[target].detach().numpy(),
            collated_data_2.out[target].detach().numpy(),
            atol=1e-5,
        )
