import numpy as np
import pytest
import torch

from mlcg.data.atomic_data import AtomicData
from mlcg.nn import GradientsOut, SumOut
from mlcg.nn.prior import NativeContactQRestraint
from mlcg.simulation import REUSSimulation

N_ATOMS = 12
PAIRS = torch.tensor([[0, 1, 2, 3, 0, 4], [5, 6, 7, 8, 9, 11]])
BETA = 1.6


def _setup(Q0, k):
    torch.manual_seed(0)
    reference = torch.randn(N_ATOMS, 3) * 0.5
    r0 = NativeContactQRestraint.compute_native_distances(reference, PAIRS)
    prior = NativeContactQRestraint(r0, torch.tensor(Q0), torch.tensor(k))
    model = SumOut(
        torch.nn.ModuleDict(
            {prior.name: GradientsOut(prior, targets="forces")}
        ),
        targets=["energy", "forces"],
    )
    nl = NativeContactQRestraint.neighbor_list(PAIRS, name=prior.name)
    configs = []
    for s in range(len(Q0)):
        configs.append(
            AtomicData.from_points(
                pos=reference + 0.3 * s * torch.randn(N_ATOMS, 3),
                atom_types=torch.arange(N_ATOMS),
                masses=torch.ones(N_ATOMS),
                neighborlist={kk: dict(v) for kk, v in nl.items()},
            )
        )
    return model, configs, prior


def _attached(tmp_path, Q0, k, **kwargs):
    model, configs, prior = _setup(Q0, k)
    sim = REUSSimulation(
        friction=1.0, filename=str(tmp_path / "sim"), **kwargs
    )
    sim.attach_model_and_configurations(model, configs, beta=BETA)
    return sim, prior


def test_acceptance_matches_metropolis(tmp_path):
    """Two windows, frozen configurations: the swap rate is
    min(1, exp(-beta [U_0(Q_1) + U_1(Q_0) - U_0(Q_0) - U_1(Q_1)]))."""
    Q0, k = [0.65, 0.5], [40.0, 60.0]  # configurations at Q 0.63, 0.55
    sim, prior = _attached(tmp_path, Q0, k)
    sim._set_up_simulation()
    data = sim.initial_data
    pos, vel = data.pos.clone(), data.velocities.clone()
    q = prior.batch_q(data).double().numpy()
    u = lambda w, x: k[w] * (q[x] - Q0[w]) ** 2
    expected = min(1.0, np.exp(-BETA * (u(0, 1) + u(1, 0) - u(0, 0) - u(1, 1))))
    assert 0.05 < expected < 0.95, expected
    for _ in range(20000):
        data.pos, data.velocities = pos.clone(), vel.clone()
        sim.replica_of_window = torch.arange(2)
        sim.detect_and_exchange_windows(data)
    rate = (sim._accepted.sum() / sim._attempted.sum()).item()
    assert sim._attempted.sum().item() == 10000  # odd parity: no pair
    np.testing.assert_allclose(rate, expected, atol=0.015)


def test_swap_exchanges_neighbours_in_q0_order(tmp_path):
    """k = 0 accepts every swap; neighbours are taken in Q0 order, not in
    batch order, and positions, velocities and replica labels move together."""
    Q0 = [0.2, 0.8, 0.4, 0.6]  # Q0 order: windows 0, 2, 3, 1
    sim, _ = _attached(tmp_path, Q0, [0.0] * 4)
    sim._set_up_simulation()
    data = sim.initial_data
    pos = data.pos.view(4, N_ATOMS, 3).clone()
    vel = data.velocities.view(4, N_ATOMS, 3).clone()
    sim.detect_and_exchange_windows(data)  # even pairs: (0, 2), (3, 1)
    perm = [2, 3, 0, 1]
    assert sim.replica_of_window.tolist() == perm
    assert torch.equal(data.pos.view(4, N_ATOMS, 3), pos[perm])
    assert torch.equal(data.velocities.view(4, N_ATOMS, 3), vel[perm])
    assert sim._forces_stale
    sim.detect_and_exchange_windows(data)  # odd pair: (2, 3)
    assert sim.replica_of_window.tolist() == [2, 3, 1, 0]


def test_simulate_exports_and_restarts(tmp_path):
    Q0, k = [0.5, 0.3, 0.4, 0.6], [50.0] * 4
    common = dict(
        n_timesteps=200,
        dt=0.004,
        save_interval=10,
        export_interval=100,
        log_interval=100,
        log_type="write",
        exchange_interval=5,
        create_checkpoints=True,
    )
    sim, _ = _attached(tmp_path, Q0, k, **common)
    sim.simulate()
    # attempts every 5 steps, alternating 2 even pairs and 1 odd pair; the
    # attempt at step 100 comes after that step's export, so it is counted
    # in the second interval (and the one at step 200 in no exported one)
    n_attempted = {"0000": 10 * 2 + 9 * 1, "0001": 10 * 1 + 10 * 2}
    for key in ("0000", "0001"):
        rep = np.load(tmp_path / f"sim_replicas_{key}.npy")
        assert rep.shape == (4, 10)
        assert all(sorted(col) == [0, 1, 2, 3] for col in rep.T)
        ex = np.load(tmp_path / f"sim_exchanges_{key}.npz")
        assert ex["window_a"].tolist() == [1, 2, 0]
        assert ex["window_b"].tolist() == [2, 0, 3]
        assert ex["attempted"].sum() == n_attempted[key]
        assert np.load(tmp_path / f"sim_coords_{key}.npy").shape[:2] == (4, 10)
    total = sum(
        np.load(tmp_path / f"sim_exchanges_{key}.npz")["accepted"].sum()
        for key in ("0000", "0001")
    )
    assert total > 0
    assert "Window exchange rate" in (tmp_path / "sim_log.txt").read_text()

    # restart: the replica labels and parity continue from the checkpoint
    ckpt = torch.load(tmp_path / "sim_checkpoint.pt", weights_only=False)
    sim2, _ = _attached(
        tmp_path,
        Q0,
        k,
        **dict(common, n_timesteps=300, read_checkpoint_file=True),
    )
    assert torch.equal(sim2.replica_of_window, ckpt["replica_of_window"])
    assert sim2._propose_even_pairs == ckpt["propose_even_pairs"]
    sim2.simulate()
    assert (tmp_path / "sim_replicas_0002.npy").exists()
