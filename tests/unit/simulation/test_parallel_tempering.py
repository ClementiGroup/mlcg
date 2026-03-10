import torch
import numpy as np
import pytest
from copy import deepcopy

from mlcg.simulation.parallel_tempering import PTSimulation
from mlcg.data.atomic_data import AtomicData
from mlcg.mol_utils import _get_initial_data, _ASE_prior_model

torch_pi = torch.tensor(np.pi)


@pytest.fixture
def get_initial_data():
    return _get_initial_data()


@pytest.fixture
def ASE_prior_model():
    return _ASE_prior_model


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

        torch.testing.assert_close(
            emperical_expectation, theory_expectation, atol=1e-1, rtol=1e-5
        )
        torch.testing.assert_close(
            emperical_expectation_2, theory_expectation_2, atol=1e-1, rtol=1e-5
        )
