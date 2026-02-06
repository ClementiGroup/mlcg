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
from ase.build import molecule

from mlcg.geometry.topology import get_connectivity_matrix, get_n_paths
from mlcg.geometry import Topology
from mlcg.neighbor_list.neighbor_list import make_neighbor_list
from mlcg.nn.prior import HarmonicBonds, HarmonicAngles, Dihedral
from torch_geometric.data.collate import collate
from mlcg.geometry.statistics import fit_baseline_models

from mlcg.mol_utils import _get_initial_data, _ASE_prior_model


@pytest.fixture
def get_initial_data():
    return _get_initial_data()


def ASE_prior_model(**kwargs):
    return _ASE_prior_model(**kwargs)


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
        sim.compile_model(data)
        return sim.calculate_potential_and_forces(data)

    energy_compiled, forces_compiled = run_simulation(compile_flag=True)
    energy_eager, forces_eager = run_simulation(compile_flag=False)

    torch.testing.assert_close(
        energy_eager, energy_compiled, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        forces_eager, forces_compiled, atol=1e-5, rtol=1e-5
    )
