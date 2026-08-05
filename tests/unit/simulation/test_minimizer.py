import numpy as np
import pytest
import torch
from torch_geometric.data.collate import collate

from mlcg.data.atomic_data import AtomicData
from mlcg.data._keys import ENERGY_KEY
from mlcg.mol_utils import _ASE_prior_model
from mlcg.simulation.minimizer import (
    minimize_energy,
    minimize_energy_ase,
)


@pytest.fixture
def ASE_prior_model():
    return _ASE_prior_model


def _build_perturbed_configurations(
    data_dictionary, n_configs=3, scale=0.2, seed=0
):
    """Builds a list of un-collated AtomicData instances by perturbing the
    equilibrium ASE geometry with small Gaussian noise."""
    mol = data_dictionary["molecule"]
    neighbor_lists = data_dictionary["neighbor_lists"]
    rng = np.random.default_rng(seed)
    equilibrium_coords = np.array(mol.get_positions())

    configurations = []
    for _ in range(n_configs):
        perturbed_coords = equilibrium_coords + scale * rng.standard_normal(
            equilibrium_coords.shape
        )
        configurations.append(
            AtomicData(
                pos=torch.tensor(perturbed_coords).float(),
                atom_types=torch.tensor(mol.get_atomic_numbers()),
                masses=torch.tensor(mol.get_masses()).float(),
                cell=None,
                neighbor_list=neighbor_lists,
            )
        )
    return configurations


def _energy(model, data):
    batch, _, _ = collate(
        AtomicData, data_list=[data], increment=True, add_batch=True
    )
    batch = model(batch)
    return batch.out[ENERGY_KEY].detach().item()


def test_minimize_energy_decreases_energy(ASE_prior_model):
    # Batched torch minimizer.
    data_dictionary = ASE_prior_model()
    model = data_dictionary["model"]
    configurations = _build_perturbed_configurations(data_dictionary)

    initial_energies = [_energy(model, data) for data in configurations]
    minimized = minimize_energy(model, configurations, fmax=1e-3, steps=200)
    final_energies = [_energy(model, data) for data in minimized]

    for initial, final in zip(initial_energies, final_energies):
        assert final < initial


def test_minimize_energy_ase_decreases_energy(ASE_prior_model):
    # ASE-driven minimizer.
    data_dictionary = ASE_prior_model()
    model = data_dictionary["model"]
    configurations = _build_perturbed_configurations(data_dictionary)

    initial_energies = [_energy(model, data) for data in configurations]
    minimized = minimize_energy_ase(model, configurations, fmax=1e-3, steps=200)
    final_energies = [_energy(model, data) for data in minimized]

    for initial, final in zip(initial_energies, final_energies):
        assert final < initial


@pytest.mark.parametrize("minimizer", [minimize_energy, minimize_energy_ase])
def test_converges_near_equilibrium(ASE_prior_model, minimizer):
    # The bonded prior is invariant to rigid-body motion, so relaxation can
    # introduce a small whole-molecule drift relative to the original ASE
    # frame. Rather than asserting an absolute RMSD to that frame, check that
    # minimization moves each configuration closer to it than it started.
    data_dictionary = ASE_prior_model()
    model = data_dictionary["model"]
    mol = data_dictionary["molecule"]
    configurations = _build_perturbed_configurations(data_dictionary)

    equilibrium_coords = torch.tensor(mol.get_positions()).float()
    initial_rmsds = [
        torch.sqrt(torch.mean((data.pos - equilibrium_coords) ** 2))
        for data in configurations
    ]

    minimized = minimizer(model, configurations, fmax=1e-3, steps=200)

    for initial_rmsd, data in zip(initial_rmsds, minimized):
        final_rmsd = torch.sqrt(
            torch.mean((data.pos - equilibrium_coords) ** 2)
        )
        assert final_rmsd < initial_rmsd


@pytest.mark.parametrize("minimizer", [minimize_energy, minimize_energy_ase])
def test_fixed_atoms_do_not_move(ASE_prior_model, minimizer):
    data_dictionary = ASE_prior_model()
    model = data_dictionary["model"]
    configurations = _build_perturbed_configurations(data_dictionary)
    original_positions = [data.pos.clone() for data in configurations]

    fixed_atoms = [[0] for _ in configurations]
    minimized = minimizer(
        model, configurations, fixed_atoms=fixed_atoms, fmax=1e-3, steps=200
    )

    for original, data in zip(original_positions, minimized):
        assert torch.allclose(data.pos[0], original[0], atol=1e-6)
        assert not torch.allclose(data.pos[1:], original[1:], atol=1e-3)


def test_both_minimizers_agree(ASE_prior_model):
    # From the same starting geometries and a tight force tolerance, the
    # batched torch minimizer and the ASE minimizer must relax to the same
    # local minimum. The bonded prior is invariant to rigid-body motion, so
    # the two optimizers can leave the molecule at slightly different overall
    # positions/orientations; compare the (invariant) minimized energies.
    data_dictionary = ASE_prior_model()
    model = data_dictionary["model"]
    configurations = _build_perturbed_configurations(data_dictionary)

    minimized = minimize_energy(model, configurations, fmax=1e-5, steps=500)
    minimized_ase = minimize_energy_ase(
        model, configurations, fmax=1e-5, steps=500
    )

    for data, data_ase in zip(minimized, minimized_ase):
        np.testing.assert_allclose(
            _energy(model, data), _energy(model, data_ase), rtol=1e-4
        )
