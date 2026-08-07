import warnings

import numpy as np
import pytest
import torch
from torch_geometric.data.collate import collate

from mlcg.data.atomic_data import AtomicData
from mlcg.data._keys import ENERGY_KEY, FORCE_KEY
from mlcg.mol_utils import _ASE_prior_model
from mlcg.simulation.minimizer import minimize_energy


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


def _forward(model, data):
    batch, _, _ = collate(
        AtomicData, data_list=[data], increment=True, add_batch=True
    )
    return model(batch).out


def _energy(model, data):
    return _forward(model, data)[ENERGY_KEY].detach().item()


def _max_force(model, data, atom_indices=None):
    """Largest per-atom force magnitude, optionally restricted to a subset of
    atoms. Fixed atoms are expected to carry a large force -- that is what
    holding them away from their unconstrained equilibrium means -- so they
    must be excluded when checking convergence."""
    forces = _forward(model, data)[FORCE_KEY].detach().norm(dim=1)
    if atom_indices is not None:
        forces = forces[torch.as_tensor(atom_indices, dtype=torch.long)]
    return forces.max().item()


def test_minimize_energy_decreases_energy(ASE_prior_model):
    data_dictionary = ASE_prior_model()
    model = data_dictionary["model"]
    configurations = _build_perturbed_configurations(data_dictionary)

    initial_energies = [_energy(model, data) for data in configurations]
    minimized = minimize_energy(model, configurations, fmax=1e-3, steps=200)
    final_energies = [_energy(model, data) for data in minimized]

    for initial, final in zip(initial_energies, final_energies):
        assert final < initial


def test_reaches_fmax(ASE_prior_model):
    # The documented contract: on return, every structure's largest per-atom
    # force is at or below fmax.
    data_dictionary = ASE_prior_model()
    model = data_dictionary["model"]
    configurations = _build_perturbed_configurations(data_dictionary)

    fmax = 1e-3
    minimized = minimize_energy(model, configurations, fmax=fmax, steps=500)

    for data in minimized:
        assert _max_force(model, data) <= fmax


def test_converges_near_equilibrium(ASE_prior_model):
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

    minimized = minimize_energy(model, configurations, fmax=1e-3, steps=200)

    for initial_rmsd, data in zip(initial_rmsds, minimized):
        final_rmsd = torch.sqrt(
            torch.mean((data.pos - equilibrium_coords) ** 2)
        )
        assert final_rmsd < initial_rmsd


def test_atoms_outside_free_atoms_do_not_move(ASE_prior_model):
    data_dictionary = ASE_prior_model()
    model = data_dictionary["model"]
    configurations = _build_perturbed_configurations(data_dictionary)
    original_positions = [data.pos.clone() for data in configurations]
    n_atoms = configurations[0].pos.shape[0]

    free = [i for i in range(n_atoms) if i != 0]
    minimized = minimize_energy(
        model,
        configurations,
        free_atoms=[free for _ in configurations],
        fmax=1e-3,
        steps=200,
    )

    for original, data in zip(original_positions, minimized):
        assert torch.allclose(data.pos[0], original[0], atol=1e-6)
        assert not torch.allclose(data.pos[1:], original[1:], atol=1e-3)


def test_non_free_atoms_excluded_from_convergence(ASE_prior_model):
    # A fixed atom keeps a large force, so convergence must be judged on the
    # free atoms alone -- otherwise no constrained run could ever converge.
    data_dictionary = ASE_prior_model()
    model = data_dictionary["model"]
    configurations = _build_perturbed_configurations(data_dictionary)
    n_atoms = configurations[0].pos.shape[0]

    free = [i for i in range(n_atoms) if i != 0]
    fmax = 1e-3
    minimized = minimize_energy(
        model,
        configurations,
        free_atoms=[free for _ in configurations],
        fmax=fmax,
        steps=500,
    )

    for data in minimized:
        assert _max_force(model, data, atom_indices=free) <= fmax


def test_inputs_are_not_mutated(ASE_prior_model):
    data_dictionary = ASE_prior_model()
    model = data_dictionary["model"]
    configurations = _build_perturbed_configurations(data_dictionary)
    original_positions = [data.pos.clone() for data in configurations]

    minimize_energy(model, configurations, fmax=1e-3, steps=50)

    for original, data in zip(original_positions, configurations):
        assert torch.equal(data.pos, original)


def test_lbfgs_max_iter_guard(ASE_prior_model):
    # torch.optim.LBFGS's closure here does not re-run the model, so more than
    # one inner iteration would reuse a stale gradient.
    data_dictionary = ASE_prior_model()
    model = data_dictionary["model"]
    configurations = _build_perturbed_configurations(data_dictionary)

    with pytest.raises(ValueError, match="max_iter"):
        minimize_energy(
            model,
            configurations,
            fmax=1e-3,
            steps=10,
            optimizer_cls=torch.optim.LBFGS,
            optimizer_kwargs={"max_iter": 5},
        )


@pytest.mark.parametrize(
    "optimizer_cls, optimizer_kwargs",
    [
        (torch.optim.SGD, {"lr": 1e-3}),
        (torch.optim.Adam, {"lr": 1e-3}),
    ],
)
def test_other_optimizers_supported(
    ASE_prior_model, optimizer_cls, optimizer_kwargs
):
    data_dictionary = ASE_prior_model()
    model = data_dictionary["model"]
    configurations = _build_perturbed_configurations(data_dictionary)

    initial_energies = [_energy(model, data) for data in configurations]
    minimized = minimize_energy(
        model,
        configurations,
        fmax=1e-3,
        steps=200,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs=optimizer_kwargs,
    )
    final_energies = [_energy(model, data) for data in minimized]

    for initial, final in zip(initial_energies, final_energies):
        assert final < initial


def test_converged_structures_stop_being_stepped(ASE_prior_model):
    # The design relies on skipping a structure's optimizer once it reaches
    # fmax, while the rest of the batch keeps going. Verify that directly by
    # counting each structure's own optimizer.step() calls, rather than
    # inferring it indirectly from the final positions.
    data_dictionary = ASE_prior_model()
    model = data_dictionary["model"]
    # scale=0: exactly at equilibrium, converges within very few steps (or
    # zero). A heavily perturbed sibling needs many more.
    easy = _build_perturbed_configurations(
        data_dictionary, n_configs=1, scale=0.0
    )
    hard = _build_perturbed_configurations(
        data_dictionary, n_configs=1, scale=0.5, seed=1
    )
    configurations = easy + hard

    step_counts = []

    class CountingLBFGS(torch.optim.LBFGS):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            step_counts.append(0)
            self._slot = len(step_counts) - 1

        def step(self, closure=None):
            step_counts[self._slot] += 1
            return super().step(closure)

    minimize_energy(
        model,
        configurations,
        fmax=0.05,
        steps=300,
        optimizer_cls=CountingLBFGS,
        optimizer_kwargs={"max_iter": 1},
    )

    assert len(step_counts) == 2
    easy_steps, hard_steps = step_counts
    assert (
        hard_steps > 0
    ), "the perturbed structure should need at least one step"
    assert easy_steps < hard_steps, (
        f"expected the near-equilibrium structure ({easy_steps} steps) to stop "
        f"well before the perturbed one ({hard_steps} steps)"
    )


def test_structures_converge_independently_within_a_mixed_batch(
    ASE_prior_model,
):
    # A structure's convergence must not depend on how far the others in the
    # same batch are from their own fmax.
    data_dictionary = ASE_prior_model()
    model = data_dictionary["model"]
    easy = _build_perturbed_configurations(
        data_dictionary, n_configs=1, scale=0.05, seed=2
    )
    hard = _build_perturbed_configurations(
        data_dictionary, n_configs=1, scale=0.5, seed=1
    )
    configurations = easy + hard

    fmax = 1e-3
    minimized = minimize_energy(model, configurations, fmax=fmax, steps=1000)

    for data in minimized:
        assert _max_force(model, data) <= fmax


def test_warns_when_steps_exhausted_before_convergence(ASE_prior_model):
    data_dictionary = ASE_prior_model()
    model = data_dictionary["model"]
    configurations = _build_perturbed_configurations(data_dictionary)

    # An unreachable fmax with almost no step budget guarantees the run ends
    # unconverged regardless of starting geometry, so this warning is
    # deterministic rather than borderline.
    with pytest.warns(UserWarning, match="did not reach fmax"):
        minimize_energy(model, configurations, fmax=1e-8, steps=3)

    # The same (fmax, steps) as test_reaches_fmax, which is proven to
    # converge -- so this warning would be a regression, not noise. Record
    # rather than escalate every warning to an error: unrelated code (e.g.
    # a torch indexing deprecation warning inside the harmonic prior) fires
    # on every forward pass and must not fail this test.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        minimize_energy(model, configurations, fmax=1e-3, steps=500)
    assert not any("did not reach fmax" in str(w.message) for w in caught)


def test_rejects_batched_configuration(ASE_prior_model):
    data_dictionary = ASE_prior_model()
    model = data_dictionary["model"]
    configurations = _build_perturbed_configurations(data_dictionary)
    batched, _, _ = collate(
        AtomicData, data_list=configurations, increment=True, add_batch=True
    )

    with pytest.raises(ValueError, match="batched"):
        minimize_energy(model, [batched], fmax=1e-3, steps=10)


def test_rejects_out_of_range_free_atoms(ASE_prior_model):
    data_dictionary = ASE_prior_model()
    model = data_dictionary["model"]
    configurations = _build_perturbed_configurations(data_dictionary)
    n_atoms = configurations[0].pos.shape[0]

    with pytest.raises(ValueError, match="out-of-range"):
        minimize_energy(
            model,
            configurations,
            free_atoms=[[n_atoms] for _ in configurations],
            fmax=1e-3,
            steps=10,
        )
