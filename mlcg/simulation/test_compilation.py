from typing import Dict, Union
import torch
import numpy as np
import pytest
from copy import deepcopy

from mlcg.simulation.langevin import (
    LangevinSimulation,
    OverdampedSimulation,
)
from mlcg.simulation.parallel_tempering import PTSimulation
from mlcg.data.atomic_data import AtomicData
from mlcg.data._keys import FORCE_KEY
from mlcg.nn import (
    StandardSchNet,
    StandardPaiNN,
    StandardSo3krates,
    GradientsOut,
    SumOut,
    CosineCutoff,
    GaussianBasis,
)

from mlcg.simulation.test_simulation import get_initial_data
from mlcg.geometry.topology import get_connectivity_matrix, get_n_paths
from ase.build import molecule
from mlcg.geometry import Topology
from mlcg.neighbor_list.neighbor_list import make_neighbor_list
from mlcg.nn.prior import HarmonicBonds, HarmonicAngles, Dihedral
from torch_geometric.data.collate import collate
from mlcg.geometry.statistics import fit_baseline_models


def ASE_prior_model(
    mol: str = "CH3CH2NH2", sum_out: bool = True, device="cpu"
) -> Dict:
    """Wrapper that returns a simple prior-only model of
    an ASE molecule with HarmonicBonds and HarmonicAngles
    priors whose parameters are estimated from coordinates
    artifically perturbed by some small Gaussian noise.

    Parameters
    ----------
    mol:
        Molecule specifying string found in ase.build.molecule
    sum_out:
        If True, the model constituents are wrapped within
        a SumOut instance
    device:
        Device on which to place the model

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
            pos=prior_data_frames[frame],
            atom_types=torch.tensor(test_topo.types),
            masses=torch.tensor(mol.get_masses()),
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
    priors, stats = fit_baseline_models(collated_prior_data, beta, prior_cls)

    # Construct the model
    full_model = {
        name: GradientsOut(priors[name], targets=[FORCE_KEY]).to(device)
        for name in priors.keys()
    }
    if sum_out:
        full_model = SumOut(full_model)

    model_with_data = {
        "model": full_model,
        "collated_prior_data": collated_prior_data,
        "molecule": mol,
        "num_examples": len(prior_data_list),
        "neighbor_lists": neighbor_lists,
    }
    return model_with_data


def ASE_SchNet_model(
    mol: str = "CH3CH2NH2", sum_out: bool = True, device="cpu"
) -> Union[torch.nn.Module, torch.nn.ModuleDict]:
    """Wrapper that returns a simple SchNet model of
    an ASE molecule.

    Parameters
    ----------
    mol:
        Molecule specifying string found in ase.build.molecule
        associated with the g2 organic molecule database
    sum_out:
        If True, the model constituents are wrapped within
        a SumOut instance
    device:
        Device on which to place the model

    Returns
    -------
    model_with_data:
        Dictionary that contains the fitted prior-only model
        under the key "model" and the noisey data, as a collated
        AtomicData instance, used for fitting under the key
        "collated_prior_data"
    """
    base_builder = ASE_prior_model(mol=mol, sum_out=False, device=device)

    nn_model = GradientsOut(
        StandardSchNet(
            rbf_layer=GaussianBasis(cutoff=CosineCutoff(0, 5), num_rbf=8),
            cutoff=CosineCutoff(0, 5),
            output_hidden_layer_widths=[8, 4],
            hidden_channels=8,
            embedding_size=10,
            num_filters=8,
            num_interactions=1,
        ),
        targets=[FORCE_KEY],
    )

    base_builder["model"][nn_model.name] = nn_model.to(device)
    if sum_out:
        base_builder["model"] = SumOut(base_builder["model"])

    return base_builder


def ASE_PaiNN_model(
    mol: str = "CH3CH2NH2", sum_out: bool = True, device="cpu"
) -> Union[torch.nn.Module, torch.nn.ModuleDict]:
    """Wrapper that returns a simple PaiNN model of
    an ASE molecule.

    Parameters
    ----------
    mol:
        Molecule specifying string found in ase.build.molecule
        associated with the g2 organic molecule database
    sum_out:
        If True, the model constituents are wrapped within
        a SumOut instance
    device:
        Device on which to place the model

    Returns
    -------
    model_with_data:
        Dictionary that contains the fitted prior-only model
        under the key "model" and the noisey data, as a collated
        AtomicData instance, used for fitting under the key
        "collated_prior_data"
    """
    base_builder = ASE_prior_model(mol=mol, sum_out=False, device=device)

    nn_model = GradientsOut(
        StandardPaiNN(
            rbf_layer=GaussianBasis(cutoff=CosineCutoff(0, 5), num_rbf=8),
            cutoff=CosineCutoff(0, 5),
            output_hidden_layer_widths=[8, 4],
            hidden_channels=8,
            embedding_size=10,
            num_interactions=1,
        ),
        targets=[FORCE_KEY],
    )

    base_builder["model"][nn_model.name] = nn_model.to(device)
    if sum_out:
        base_builder["model"] = SumOut(base_builder["model"])

    return base_builder


def ASE_SO3krates_model(
    mol: str = "CH3CH2NH2", sum_out: bool = True, device="cpu"
) -> Union[torch.nn.Module, torch.nn.ModuleDict]:
    """Wrapper that returns a simple SO3krates model of
    an ASE molecule.

    Parameters
    ----------
    mol:
        Molecule specifying string found in ase.build.molecule
        associated with the g2 organic molecule database
    sum_out:
        If True, the model constituents are wrapped within
        a SumOut instance
    device:
        Device on which to place the model

    Returns
    -------
    model_with_data:
        Dictionary that contains the fitted prior-only model
        under the key "model" and the noisey data, as a collated
        AtomicData instance, used for fitting under the key
        "collated_prior_data"
    """
    base_builder = ASE_prior_model(mol=mol, sum_out=False, device=device)

    nn_model = GradientsOut(
        StandardSo3krates(
            rbf_layer=GaussianBasis(cutoff=CosineCutoff(0, 5), num_rbf=8),
            cutoff=CosineCutoff(0, 5),
            output_hidden_layer_widths=[8, 4],
            hidden_channels=8,
            embedding_size=10,
            degrees=[1, 2],
            n_heads=2,
            num_interactions=1,
        ),
        targets=[FORCE_KEY],
    )

    base_builder["model"][nn_model.name] = nn_model.to(device)
    if sum_out:
        base_builder["model"] = SumOut(base_builder["model"])

    return base_builder


@pytest.mark.parametrize(
    "model, get_initial_data, add_masses, sim_class, sim_args, betas",
    [
        (
            ASE_prior_model,
            get_initial_data,
            True,
            OverdampedSimulation,
            [],
            1.0,
        ),
        (
            ASE_prior_model,
            get_initial_data,
            True,
            LangevinSimulation,
            [1.0],
            1.0,
        ),
        (
            ASE_prior_model,
            get_initial_data,
            True,
            PTSimulation,
            [1.0, 1],
            [1.67, 1.42, 1.28, 1.00, 0.8, 0.5],
        ),
        (
            ASE_SchNet_model,
            get_initial_data,
            True,
            LangevinSimulation,
            [1.0],
            1.0,
        ),
        (
            ASE_PaiNN_model,
            get_initial_data,
            True,
            LangevinSimulation,
            [1.0],
            1.0,
        ),
        (
            ASE_SO3krates_model,
            get_initial_data,
            True,
            LangevinSimulation,
            [1.0],
            1.0,
        ),
    ],
    indirect=["get_initial_data"],
)
def test_simulation_run(
    model, get_initial_data, add_masses, sim_class, sim_args, betas, tmp_path
):
    """Compare eager vs compiled simulation outputs."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dict = model(device=device)
    torch._functorch.config.donated_buffer = False

    def run_simulation(compile_flag):
        full_model = data_dict["model"]
        mol = data_dict["molecule"]
        neighbor_lists = data_dict["neighbor_lists"]
        initial_data_list = get_initial_data(
            mol, neighbor_lists, corruptor=None, add_masses=add_masses
        )

        sim_kwargs = {
            "filename": tmp_path
            / f"{sim_class.__name__}_{'compiled' if compile_flag else 'eager'}",
            "device": device,
            "compile": compile_flag,
        }

        sim = sim_class(*sim_args, **sim_kwargs)
        sim.attach_model_and_configurations(
            full_model, initial_data_list, betas
        )
        sim._set_up_simulation(overwrite=False)
        data = deepcopy(sim.initial_data).to(sim.device)
        sim.compile_model()

        return sim.calculate_potential_and_forces(data)

    energy_compiled, forces_compiled = run_simulation(compile_flag=True)
    energy_eager, forces_eager = run_simulation(compile_flag=False)

    torch.testing.assert_close(
        energy_eager, energy_compiled, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        forces_eager, forces_compiled, atol=1e-5, rtol=1e-5
    )
