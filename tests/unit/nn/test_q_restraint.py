import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from mlcg.data.atomic_data import AtomicData
from mlcg.nn import GradientsOut
from mlcg.nn.prior import NativeContactQRestraint

BETA_Q, LAMBDA_Q = 1.0, 1.5
N_ATOMS, N_STRUCTURES = 12, 4
PAIRS = torch.tensor([[0, 1, 2, 3, 0, 4], [5, 6, 7, 8, 9, 11]])


def _reference_q(pos, r0):
    """Q exactly as traj_rmsd_q_analysis.py defines it."""
    d = np.linalg.norm(pos[PAIRS[0]] - pos[PAIRS[1]], axis=1)
    return np.mean(1.0 / (1.0 + np.exp(BETA_Q * (d - LAMBDA_Q * r0))))


def _batch(positions, name="native_contacts_q"):
    nl = NativeContactQRestraint.neighbor_list(PAIRS, name=name)
    data_list = []
    for pos in positions:
        data = AtomicData.from_points(
            pos=pos.clone(),
            atom_types=torch.zeros(pos.shape[0], dtype=torch.long),
        )
        data.neighbor_list = {k: dict(v) for k, v in nl.items()}
        data_list.append(data)
    batch = Batch.from_data_list(data_list)
    batch.out = {}
    return batch


@pytest.fixture
def setup():
    torch.manual_seed(0)
    reference = torch.randn(N_ATOMS, 3, dtype=torch.float64) * 0.5
    r0 = NativeContactQRestraint.compute_native_distances(reference, PAIRS)
    positions = [
        reference + 0.15 * torch.randn(N_ATOMS, 3, dtype=torch.float64)
        for _ in range(N_STRUCTURES)
    ]
    return reference, r0, positions


def test_energy_matches_reference_definition(setup):
    """Each structure is biased against its own Q0, with the analysis' Q."""
    _, r0, positions = setup
    Q0 = torch.tensor([0.1, 0.4, 0.7, 1.0], dtype=torch.float64)
    k = 200.0
    prior = NativeContactQRestraint(
        r0, Q0, torch.tensor(k), beta_q=BETA_Q, lambda_q=LAMBDA_Q
    )
    energy = prior(_batch(positions)).out[prior.name]["energy"]

    expected = [
        k * (_reference_q(p.numpy(), r0.numpy()) - Q0[i].item()) ** 2
        for i, p in enumerate(positions)
    ]
    np.testing.assert_allclose(energy.detach().numpy(), expected, rtol=1e-10)


def test_forces_match_finite_differences(setup):
    _, r0, positions = setup
    prior = NativeContactQRestraint(
        r0,
        torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float64),
        torch.tensor(150.0),
        beta_q=BETA_Q,
        lambda_q=LAMBDA_Q,
    )
    forces = (
        GradientsOut(prior, targets="forces")
        .eval()(_batch(positions))
        .out[prior.name]["forces"]
        .detach()
    )

    flat = torch.cat(positions)
    eps = 1e-6
    expected = torch.zeros_like(flat)
    for i in range(flat.shape[0]):
        for j in range(3):
            shifted = []
            for sign in (1.0, -1.0):
                moved = flat.clone()
                moved[i, j] += sign * eps
                chunks = [
                    moved[s * N_ATOMS : (s + 1) * N_ATOMS]
                    for s in range(N_STRUCTURES)
                ]
                shifted.append(
                    prior(_batch(chunks)).out[prior.name]["energy"].sum()
                )
            expected[i, j] = -(shifted[0] - shifted[1]) / (2 * eps)
    torch.testing.assert_close(forces, expected, atol=1e-6, rtol=1e-5)


def test_restraint_pulls_cv_to_its_target(setup):
    """Descending the bias alone should reach each window's Q0."""
    reference, r0, _ = setup
    # kept clear of the Q -> 1 ceiling, where the sigmoid saturates
    Q0 = torch.tensor([0.35, 0.50, 0.65, 0.80], dtype=torch.float64)
    prior = NativeContactQRestraint(
        r0, Q0, torch.tensor(500.0), beta_q=BETA_Q, lambda_q=LAMBDA_Q
    )
    model = GradientsOut(prior, targets="forces").eval()
    positions = [
        reference + 0.05 * torch.randn(N_ATOMS, 3, dtype=torch.float64)
        for _ in range(N_STRUCTURES)
    ]
    for _ in range(4000):
        forces = model(_batch(positions)).out[prior.name]["forces"].detach()
        for s in range(N_STRUCTURES):
            positions[s] += 5e-4 * forces[s * N_ATOMS : (s + 1) * N_ATOMS]

    reached = [_reference_q(p.numpy(), r0.numpy()) for p in positions]
    np.testing.assert_allclose(reached, Q0.numpy(), atol=5e-3)


def test_energy_isolates_each_structures_target(setup):
    """Identical structures, different Q0: any cross-structure mixing of Q, r0
    or the scatter shows up here, where every window sees the exact same Q."""
    reference, r0, _ = setup
    Q0 = torch.tensor([0.1, 0.4, 0.7, 0.9], dtype=torch.float64)
    prior = NativeContactQRestraint(
        r0, Q0, torch.tensor(200.0), beta_q=BETA_Q, lambda_q=LAMBDA_Q
    )
    energy = prior(_batch([reference] * N_STRUCTURES)).out[prior.name]["energy"]

    q = _reference_q(reference.numpy(), r0.numpy())
    np.testing.assert_allclose(
        energy.detach().numpy(), 200.0 * (q - Q0.numpy()) ** 2, rtol=1e-10
    )


def test_q0_of_wrong_length_is_rejected(setup):
    """A single Q0 would otherwise broadcast the whole ladder onto one window."""
    _, r0, positions = setup
    prior = NativeContactQRestraint(
        r0, torch.tensor([0.5], dtype=torch.float64), torch.tensor(100.0)
    )
    with pytest.raises(ValueError, match="one value per structure"):
        prior(_batch(positions))


def test_k_of_wrong_length_is_rejected(setup):
    """k gets the same guard as Q0: a per-window ladder must not be broadcast."""
    _, r0, positions = setup
    prior = NativeContactQRestraint(
        r0,
        torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float64),
        torch.tensor([100.0, 200.0, 300.0], dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="one value per structure"):
        prior(_batch(positions))


@pytest.mark.parametrize("k", [torch.tensor(150.0), torch.tensor([150.0])])
def test_single_k_is_shared_by_every_window(setup, k):
    """A lone k means one spring for the whole ladder, written either way."""
    _, r0, positions = setup
    Q0 = torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float64)
    prior = NativeContactQRestraint(r0, Q0, k, beta_q=BETA_Q, lambda_q=LAMBDA_Q)
    assert prior.k.shape == Q0.shape

    energy = prior(_batch(positions)).out[prior.name]["energy"]
    expected = [
        150.0 * (_reference_q(p.numpy(), r0.numpy()) - Q0[i].item()) ** 2
        for i, p in enumerate(positions)
    ]
    np.testing.assert_allclose(energy.detach().numpy(), expected, rtol=1e-10)


def test_uncollated_single_structure_is_one_window(setup):
    """Without a ptr there is no batch to infer, so Q0 must hold exactly one
    value -- otherwise Q broadcasts against the ladder and silently returns
    one energy per window for a single structure."""
    _, r0, positions = setup
    nl = NativeContactQRestraint.neighbor_list(PAIRS)
    data = AtomicData.from_points(
        pos=positions[0].clone(),
        atom_types=torch.zeros(N_ATOMS, dtype=torch.long),
    )
    data.neighbor_list = {k: dict(v) for k, v in nl.items()}
    data.out = {}
    assert not hasattr(data, "ptr")

    ladder = NativeContactQRestraint(
        r0,
        torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float64),
        torch.tensor(100.0),
    )
    with pytest.raises(ValueError, match="one value per structure"):
        ladder(data)

    single = NativeContactQRestraint(
        r0,
        torch.tensor([0.4], dtype=torch.float64),
        torch.tensor(100.0),
        beta_q=BETA_Q,
        lambda_q=LAMBDA_Q,
    )
    energy = single(data).out[single.name]["energy"]
    assert energy.shape == (1,)
    expected = 100.0 * (_reference_q(positions[0].numpy(), r0.numpy()) - 0.4) ** 2
    np.testing.assert_allclose(energy.item(), expected, rtol=1e-10)


def test_mismatched_native_contact_lists_are_rejected(setup):
    """r0 is tiled across the batch, which is only meaningful if every
    structure carries the same contact list; a shorter one would misalign r0
    against the distances instead of erroring."""
    _, r0, positions = setup
    prior = NativeContactQRestraint(
        r0, torch.tensor([0.3, 0.6], dtype=torch.float64), torch.tensor(100.0)
    )
    data_list = []
    for pos, pairs in zip(positions[:2], (PAIRS, PAIRS[:, :4])):
        data = AtomicData.from_points(
            pos=pos.clone(),
            atom_types=torch.zeros(N_ATOMS, dtype=torch.long),
        )
        nl = NativeContactQRestraint.neighbor_list(pairs)
        data.neighbor_list = {k: dict(v) for k, v in nl.items()}
        data_list.append(data)
    batch = Batch.from_data_list(data_list)
    batch.out = {}

    with pytest.raises(ValueError, match="same native contact list"):
        prior(batch)


def test_compute_q_matches_the_restrained_cv(setup):
    """The analysis-facing helper must return exactly the Q that forward biases."""
    _, r0, positions = setup
    Q0 = torch.zeros(N_STRUCTURES, dtype=torch.float64)
    k = 1.0
    prior = NativeContactQRestraint(
        r0, Q0, torch.tensor(k), beta_q=BETA_Q, lambda_q=LAMBDA_Q
    )
    # with Q0 = 0 and k = 1 the reported energy is Q^2
    energy = prior(_batch(positions)).out[prior.name]["energy"].detach()
    from_forward = torch.sqrt(energy)

    from_helper = prior.compute_q(torch.stack(positions), PAIRS)
    torch.testing.assert_close(from_helper, from_forward)
    np.testing.assert_allclose(
        from_helper.numpy(),
        [_reference_q(p.numpy(), r0.numpy()) for p in positions],
        rtol=1e-10,
    )


def test_switching_parameters_survive_a_state_dict_round_trip():
    """beta_q/lambda_q define the CV, so a reload must not fall back to the
    constructor defaults and silently analyse a different Q."""
    r0 = torch.tensor([3.0, 4.0], dtype=torch.float64)
    prior = NativeContactQRestraint(
        r0,
        torch.tensor([0.5], dtype=torch.float64),
        torch.tensor(10.0),
        beta_q=2.5,
        lambda_q=1.25,
    )
    assert {"beta_q", "lambda_q"} <= set(prior.state_dict())

    reloaded = NativeContactQRestraint(
        torch.zeros_like(r0), torch.tensor([0.0], dtype=torch.float64), torch.tensor(1.0)
    )
    reloaded.load_state_dict(prior.state_dict())
    assert float(reloaded.beta_q) == 2.5
    assert float(reloaded.lambda_q) == 1.25


def test_double_precision_buffers_are_not_downcast():
    r0 = torch.tensor([1.2345678901234567], dtype=torch.float64)
    prior = NativeContactQRestraint(
        r0, torch.tensor([0.5], dtype=torch.float64), torch.tensor(1.0)
    )
    assert prior.r0.dtype == torch.float64
    assert prior.r0.item() == r0.item()
