import tempfile
import torch
import numpy as np
import pytest

from mlcg.simulation.base import _Simulation
from mlcg.simulation.langevin import (
    LangevinSimulation,
    OverdampedSimulation,
)
from mlcg.simulation.parallel_tempering import PTSimulation
from mlcg.data._keys import MASS_KEY, POSITIONS_KEY, ATOM_TYPE_KEY, FORCE_KEY
from mlcg.mol_utils import _get_initial_data, _ASE_prior_model

from mlcg.nn.flash_schnet import StandardFlashSchNet
from mlcg.nn.radial_basis import ExpNormalBasis
from mlcg.nn.cutoff import CosineCutoff
from mlcg.nn import GradientsOut, SumOut

torch_pi = torch.tensor(np.pi)


@pytest.fixture
def get_initial_data():
    return _get_initial_data()


@pytest.fixture
def ASE_prior_model():
    return _ASE_prior_model


### corruptors - lambdas that introduce a problem in the data list ###

# Puts the wrong mass on the fourth frame
wrong_mass_fn = lambda frame, mol: (
    (2 * torch.tensor(mol.get_masses()), MASS_KEY)
    if frame == 3
    else (torch.tensor(mol.get_masses()), MASS_KEY)
)

# Gives a structure with the wrong shape on the third frame
wrong_pos_fn = lambda frame, mol: (
    (torch.randn(7, 3), POSITIONS_KEY)
    if frame == 2
    else (torch.tensor(mol.get_positions()), POSITIONS_KEY)
)
# Gives the wrong atomic types on the second frame
wrong_atom_type_fn = lambda frame, mol: (
    (
        7 * torch.tensor(mol.get_atomic_numbers()),
        ATOM_TYPE_KEY,
    )
    if frame == 1
    else (torch.tensor(mol.get_atomic_numbers()), ATOM_TYPE_KEY)
)


@pytest.mark.parametrize(
    "ASE_prior_model, get_initial_data, corruptor, add_masses, expected_raise",
    [
        (
            # Should raise error: one frame has different masses
            ASE_prior_model,
            get_initial_data,
            wrong_mass_fn,
            True,
            ValueError,
        ),
        (
            # Should raise error: one frame has a different structure
            ASE_prior_model,
            get_initial_data,
            wrong_pos_fn,
            True,
            ValueError,
        ),
        (
            # Should raise error: one frame has a different atom types
            ASE_prior_model,
            get_initial_data,
            wrong_atom_type_fn,
            True,
            ValueError,
        ),
    ],
    indirect=["ASE_prior_model", "get_initial_data"],
)
def test_data_list_raises(
    ASE_prior_model, get_initial_data, corruptor, add_masses, expected_raise
):
    """Test to make sure certain warnings/errors are raised regarding the data list"""
    data_dictionary = ASE_prior_model()
    full_model = data_dictionary["model"]
    mol = data_dictionary["molecule"]
    neighbor_lists = data_dictionary["neighbor_lists"]
    beta = 1

    initial_data_list = get_initial_data(
        mol, neighbor_lists, corruptor, add_masses=add_masses
    )

    if isinstance(expected_raise, Exception):
        with pytest.raises(expected_raise):
            simulation = _Simulation()
            simulation._attach_configurations(initial_data_list, beta)
    if isinstance(expected_raise, UserWarning):
        with pytest.warns(expected_raise):
            simulation = _Simulation()
            simulation._attach_configurations(initial_data_list, beta)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="No CUDA devices available."
)
@pytest.mark.parametrize(
    "seed, device", [(None, "cuda"), (None, "cpu"), (1, "cuda"), (1, "cpu")]
)
def test_simulation_device(seed, device, tmp_path):
    """Test to meke sure generator devices are set correctly"""
    fn = tmp_path / _Simulation.__name__
    simulation = _Simulation(random_seed=seed, device=device, filename=fn)
    current_device = torch.device(device)
    assert simulation.device == current_device
    if seed == None:
        assert simulation.rng == None
    else:
        assert simulation.rng.device == current_device


@pytest.mark.parametrize(
    "ASE_prior_model, get_initial_data, add_masses, sim_class, sim_args, betas, sim_kwargs",
    [
        (
            ASE_prior_model,
            get_initial_data,
            True,
            OverdampedSimulation,
            [],
            1.0,
            {},
        ),
        (
            ASE_prior_model,
            get_initial_data,
            True,
            LangevinSimulation,
            [1.0],
            1.0,
            {},
        ),
        (
            ASE_prior_model,
            get_initial_data,
            True,
            PTSimulation,
            [1.0, 1],
            [1.67, 1.42, 1.28, 1.00, 0.8, 0.5],
            {},
        ),
    ],
    indirect=["ASE_prior_model", "get_initial_data"],
)
def test_simulation_run(
    ASE_prior_model,
    get_initial_data,
    add_masses,
    sim_class,
    sim_args,
    betas,
    sim_kwargs,
    tmp_path,
):
    """Test to make sure the simulation runs"""
    data_dictionary = ASE_prior_model()
    full_model = data_dictionary["model"]
    mol = data_dictionary["molecule"]
    neighbor_lists = data_dictionary["neighbor_lists"]
    initial_data_list = get_initial_data(
        mol, neighbor_lists, corruptor=None, add_masses=add_masses
    )
    sim_kwargs["filename"] = tmp_path / sim_class.__name__
    simulation = sim_class(*sim_args, **sim_kwargs)
    simulation.attach_model_and_configurations(
        full_model, initial_data_list, betas
    )
    simulation.simulate()


@pytest.mark.parametrize(
    "ASE_prior_model, get_initial_data, add_masses, sim_class, sim_args, betas, sim_kwargs",
    [
        (
            ASE_prior_model,
            get_initial_data,
            True,
            OverdampedSimulation,
            [],
            1.0,
            {},
        ),
        (
            ASE_prior_model,
            get_initial_data,
            True,
            LangevinSimulation,
            [1.0],
            1.0,
            {},
        ),
        (
            ASE_prior_model,
            get_initial_data,
            True,
            PTSimulation,
            [1.0, 1],
            [1.67, 1.42, 1.28, 1.00, 0.8, 0.5],
            {},
        ),
    ],
    indirect=["ASE_prior_model", "get_initial_data"],
)
def test_overwrite_protection(
    ASE_prior_model,
    get_initial_data,
    add_masses,
    sim_class,
    sim_args,
    betas,
    sim_kwargs,
):
    """Test to make sure that overwrite protection works"""
    data_dictionary = ASE_prior_model()
    full_model = data_dictionary["model"]
    mol = data_dictionary["molecule"]
    neighbor_lists = data_dictionary["neighbor_lists"]
    initial_data_list = get_initial_data(
        mol, neighbor_lists, corruptor=None, add_masses=add_masses
    )

    with tempfile.TemporaryDirectory() as tmp:
        filename = tmp + "/my_sim_coords_000.npy"
        open(filename, "w").close()
        sim_kwargs["filename"] = filename
        simulation = sim_class(*sim_args, **sim_kwargs)
        simulation.attach_model_and_configurations(
            full_model, initial_data_list, betas
        )
        simulation.simulate()

        with pytest.raises(RuntimeError):
            simulation.simulate()


def test_maxwell_boltzmann_stats():
    """Tests to make sure MB distribution produces the correct
    velocity moments"""
    betas = 1.67 * torch.ones((10000))
    masses = 12.0 * torch.ones((10000))
    sampled_velocities = LangevinSimulation.sample_maxwell_boltzmann(
        betas, masses
    )
    empirical_expectation = torch.sqrt(
        (sampled_velocities**2).sum(dim=1)
    ).mean()
    empirical_expectation_2 = (sampled_velocities**2).sum(dim=1).mean()
    theory_expectation = torch.sqrt(8 / (torch_pi * betas[0] * masses[0]))
    theory_expectation_2 = 3 / (betas[0] * masses[0])

    torch.testing.assert_close(
        empirical_expectation, theory_expectation, atol=1e-2, rtol=1e-5
    )
    torch.testing.assert_close(
        empirical_expectation_2, theory_expectation_2, atol=1e-2, rtol=1e-5
    )


def flash_cgschnet_model():
    standard_basis = ExpNormalBasis(cutoff=5)
    standard_cutoff = CosineCutoff(cutoff_upper=5)
    test_flash_schnet = StandardFlashSchNet(
        standard_basis, standard_cutoff, [128, 128]
    )
    return GradientsOut(test_flash_schnet, targets=[FORCE_KEY])

@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Quantization kernel tests require CUDA"
)
@pytest.mark.parametrize(
    "ASE_prior_model, get_initial_data, add_masses, sim_class, sim_args, betas, sim_kwargs",
    [
        (
            ASE_prior_model,
            get_initial_data,
            True,
            OverdampedSimulation,
            [],
            1.0,
            {},
        ),
        (
            ASE_prior_model,
            get_initial_data,
            True,
            LangevinSimulation,
            [1.0],
            1.0,
            {},
        ),
        (
            ASE_prior_model,
            get_initial_data,
            True,
            PTSimulation,
            [1.0, 1],
            [1.67, 1.42, 1.28, 1.00, 0.8, 0.5],
            {},
        ),
    ],
    indirect=["ASE_prior_model", "get_initial_data"],
)
def test_simulation_quantization(
    ASE_prior_model,
    get_initial_data,
    add_masses,
    sim_class,
    sim_args,
    betas,
    sim_kwargs,
    tmp_path,
):
    """Test to make sure the simulation runs"""
    data_dictionary = ASE_prior_model(device="cuda")
    single_model = flash_cgschnet_model()
    full_model = SumOut({single_model.name: single_model}).to("cuda")
    mol = data_dictionary["molecule"]
    neighbor_lists = data_dictionary["neighbor_lists"]
    initial_data_list = get_initial_data(
        mol, neighbor_lists, corruptor=None, add_masses=add_masses
    )
    sim_kwargs["filename"] = tmp_path / sim_class.__name__
    sim_kwargs["gptq"] = "w16a16"
    sim_kwargs["device"] = "cuda"
    simulation = sim_class(*sim_args, **sim_kwargs)
    simulation.attach_model_and_configurations(
        full_model, initial_data_list, betas
    )
    simulation.simulate()
