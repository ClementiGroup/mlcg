from typing import List, Callable, Dict
import tempfile
import torch
import numpy as np
import pytest
from copy import deepcopy
from torch_geometric.data.collate import collate
from ase.atoms import Atoms

from mlcg.simulation.base import _Simulation
from mlcg.simulation.langevin import (
    LangevinSimulation,
    OverdampedSimulation,
)
from mlcg.simulation.parallel_tempering import PTSimulation
from mlcg.data.atomic_data import AtomicData
from mlcg.data._keys import MASS_KEY, POSITIONS_KEY, ATOM_TYPE_KEY
from mlcg.mol_utils import _get_initial_data, _ASE_prior_model

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


def test_exchange_detection(tmp_path):
    """Test to make sure that the Monte-Carlo exchange criterion
    works appropriately for two ideal fermions in the same spatial state
    with different energetic spin states (to the dumbest order).
    """
    coords = torch.zeros(1, 3)  # place both particles at origin
    low_energy = 1.00  # low energy state
    high_energy = 2.00  # high energy state
    betas = [1.67, 1.42]

    test_data = [
        AtomicData.from_points(
            pos=coords,
            atom_types=torch.tensor([1]),  # dummy types
            masses=torch.tensor([1]),
        ),
        AtomicData.from_points(
            pos=coords,
            atom_types=torch.tensor([1]),  # dummy types
            masses=torch.tensor([1]),
        ),
    ]
    fn = tmp_path / PTSimulation.__name__
    expected_rate = np.exp((low_energy - high_energy) * (betas[0] - betas[1]))
    empirical_rate_arr = []
    # run 50 independent iterations of 2x1000 exchanges times and
    # assert the average acceptance rate. the acceptances are
    # tracked internally by the `attempts/approved` attributes
    for _ in range(50):
        simulation = PTSimulation(filename=fn)
        simulation._attach_configurations(
            test_data, betas
        )  # necessary to populate some attributes

        simulation.initial_data["out"]["energy"] = torch.tensor(
            [low_energy, low_energy, high_energy, high_energy]
        )
        for _ in range(4000):
            simulation._detect_exchange(simulation.initial_data)
        empirical_rate = (
            simulation._replica_exchange_approved
            / simulation._replica_exchange_attempts
        ).item()
        empirical_rate_arr.append(empirical_rate)

    np.testing.assert_almost_equal(
        expected_rate,
        np.mean(empirical_rate_arr),
        decimal=3,
    )


def test_exchange_and_rescale(tmp_path):
    # Test to make sure that replica swaps are accurate and scaled
    # appropriately.

    # mock data of 2 betas, 10 sims each, 7 atoms per molecule
    betas = [1.67, 1.42]  # only used to instantiate the simulation
    n_replicas = len(betas)
    n_indep = 10
    configurations = []
    for _ in range(n_indep):
        configurations.append(
            AtomicData.from_points(
                pos=torch.zeros(7, 3),
                atom_types=torch.ones(7),
                masses=torch.ones(7),
                velocities=torch.zeros(7, 3),
            )
        )

    # Pairs for swapping
    pairs_for_exchange = {
        "a": torch.tensor([0, 2, 3, 6, 9]),
        "b": torch.tensor([10, 12, 13, 16, 19]),
    }
    fn = tmp_path / PTSimulation.__name__
    simulation = PTSimulation(filename=fn)
    simulation._attach_configurations(configurations, betas)
    simulation._set_up_simulation()
    # randomize coordinates and velocites - as if we had run some simulation and the replicas
    # evolved independently in time.

    simulation.initial_data.pos = torch.randn(
        *simulation.initial_data.pos.shape
    ).double()
    simulation.initial_data.velocities = torch.randn(
        *simulation.initial_data.velocities.shape
    ).double()

    manual_coords = (
        simulation.initial_data.pos.numpy()
        .reshape(n_replicas * n_indep, 7, 3)
        .astype("float64")
    )
    manual_betas = np.repeat(betas, n_indep)[:, None, None]
    manual_velocities = (
        simulation.initial_data.velocities.numpy()
        .reshape(n_replicas * n_indep, 7, 3)
        .astype("float64")
    )

    hot_to_cold_vscale = np.sqrt(
        manual_betas[pairs_for_exchange["b"]]
        / manual_betas[pairs_for_exchange["a"]]
    ).astype("float64")
    cold_to_hot_vscale = np.sqrt(
        manual_betas[pairs_for_exchange["a"]]
        / manual_betas[pairs_for_exchange["b"]]
    ).astype("float64")

    swapped_coords = deepcopy(manual_coords)
    swapped_velocities = deepcopy(manual_velocities)
    swapped_coords[pairs_for_exchange["a"].numpy()] = manual_coords[
        pairs_for_exchange["b"]
    ]
    swapped_coords[pairs_for_exchange["b"].numpy()] = manual_coords[
        pairs_for_exchange["a"]
    ]
    swapped_velocities[pairs_for_exchange["a"].numpy()] = (
        manual_velocities[pairs_for_exchange["b"]] * cold_to_hot_vscale
    )
    swapped_velocities[pairs_for_exchange["b"].numpy()] = (
        manual_velocities[pairs_for_exchange["a"]] * hot_to_cold_vscale
    )

    # Perform exchange
    exchanged_data = simulation._perform_exchange(
        simulation.initial_data, pairs_for_exchange
    )

    exchanged_coords = exchanged_data.pos.numpy().reshape(
        n_replicas * n_indep, 7, 3
    )
    exchanged_and_scaled_velocities = exchanged_data.velocities.numpy().reshape(
        n_replicas * n_indep, 7, 3
    )

    np.testing.assert_almost_equal(swapped_coords, exchanged_coords, decimal=5)
    np.testing.assert_almost_equal(
        swapped_velocities, exchanged_and_scaled_velocities, decimal=5
    )


def test_maxwell_boltzmann_stats():
    """Tests to make sure MB distribution produces the correct
    velocity moments"""
    betas = 1.67 * torch.ones((10000))
    masses = 12.0 * torch.ones((10000))
    sampled_velocities = LangevinSimulation.sample_maxwell_boltzmann(
        betas, masses
    )
    emperical_expectation = torch.sqrt(
        (sampled_velocities**2).sum(dim=1)
    ).mean()
    emperical_expectation_2 = (sampled_velocities**2).sum(dim=1).mean()
    theory_expectation = torch.sqrt(8 / (torch_pi * betas[0] * masses[0]))
    theory_expectation_2 = 3 / (betas[0] * masses[0])

    np.testing.assert_almost_equal(
        emperical_expectation.numpy(), theory_expectation.numpy(), decimal=2
    )
    np.testing.assert_almost_equal(
        emperical_expectation_2.numpy(), theory_expectation_2.numpy(), decimal=2
    )


def test_pt_velocity_init(tmp_path):
    """Tests to make sure that PTSimulation.attach_configurations
    correctly for each temperature level"""
    betas = [1.67, 1.42, 1.22, 1.0]  # only used to instantiate the simulation

    # Mock data of 1000 frames of a 7 atom molecule, each atom having unit mass
    n_indep = 10000
    n_atoms = 7
    configurations = []
    for _ in range(n_indep):
        configurations.append(
            AtomicData.from_points(
                pos=torch.zeros(n_atoms, 3),
                atom_types=torch.ones(n_atoms),
                masses=torch.ones(n_atoms),
                velocities=torch.zeros(n_atoms, 3),
            )
        )
    fn = tmp_path / PTSimulation.__name__
    simulation = PTSimulation(filename=fn)
    simulation._attach_configurations(configurations, betas)
    print(simulation.initial_data.velocities)

    mass = 1.00
    for i, beta in enumerate(betas):
        theory_expectation = torch.sqrt(8 / (beta * torch_pi * mass))
        theory_expectation_2 = torch.tensor(3 / (beta * mass))

        velocities = simulation.initial_data.velocities[
            (n_indep * n_atoms) * i : (n_indep * n_atoms) * (i + 1)
        ]
        emperical_expectation = torch.sqrt((velocities**2).sum(dim=1)).mean()
        emperical_expectation_2 = (velocities**2).sum(dim=1).mean()

        np.testing.assert_almost_equal(
            emperical_expectation.numpy(), theory_expectation.numpy(), decimal=1
        )
        np.testing.assert_almost_equal(
            emperical_expectation_2.numpy(),
            theory_expectation_2.numpy(),
            decimal=1,
        )
