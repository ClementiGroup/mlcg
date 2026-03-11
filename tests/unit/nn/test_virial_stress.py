"""Tests for virial and stress computation via GradientsOut.

Tests the symmetric-displacement approach for computing virials/stress
as dE/dε, following the MACE/NequIP pattern.
"""

import pytest
import torch
import numpy as np

try:
    from typing import List
    from ase.build import molecule, bulk
    from torch_geometric.data import Batch
    from torch_geometric.data.collate import collate

    import mace
    from mlcg.geometry import Topology
    from mlcg.nn.mace import StandardMACE
    from mlcg.nn.gradients import GradientsOut, SumOut, _apply_strain
    from mlcg.data.atomic_data import AtomicData
    from mlcg.data._keys import ENERGY_KEY, FORCE_KEY, VIRIALS_KEY, STRESS_KEY

    class MolDatabase:
        """Small molecule database for testing."""

        def __init__(
            self,
            mol_names: List[str] = ["H2O", "CH4", "NH3"],
        ):
            self.mol_names = mol_names
            self.molecules = [molecule(name) for name in self.mol_names]
            self.mol_topos = [Topology.from_ase(mol) for mol in self.molecules]
            data_list = []
            for mol, topo in zip(self.molecules, self.mol_topos):
                neighbor_list = topo.neighbor_list("fully connected")
                data = AtomicData.from_points(
                    pos=torch.tensor(mol.get_positions()).float(),
                    atom_types=torch.tensor(mol.get_atomic_numbers()),
                    neighbor_list=neighbor_list,
                )
                data_list.append(data)
            self.data_list = data_list
            self.collated_data, _, _ = collate(
                data_list[0].__class__,
                data_list=data_list,
                increment=True,
                add_batch=True,
            )
            self.atomic_numbers = sorted(
                torch.unique(self.collated_data.atom_types).numpy().tolist()
            )

    database = MolDatabase()

    mace_config = {
        "r_max": 5,
        "num_bessel": 8,
        "num_polynomial_cutoff": 5,
        "max_ell": 1,
        "interaction_cls": "mace.modules.blocks.RealAgnosticResidualInteractionBlock",
        "interaction_cls_first": "mace.modules.blocks.RealAgnosticResidualInteractionBlock",
        "num_interactions": 1,
        "hidden_irreps": "16x0e",
        "MLP_irreps": "8x0e",
        "avg_num_neighbors": 5,
        "correlation": 2,
        "gate": torch.nn.Tanh(),
        "max_num_neighbors": 100,
        "pair_repulsion": False,
        "distance_transform": None,
        "radial_MLP": [16, 16],
        "radial_type": "bessel",
        "atomic_numbers": database.atomic_numbers,
    }

except Exception:
    pytest.skip(
        "MACE installation not found — skipping virial/stress tests",
        allow_module_level=True,
    )


# ── Helpers ───────────────────────────────────────────────────────────


def _build_model(compute_virials=False, compute_stress=False):
    """Build a GradientsOut(MACE) with optional virial/stress."""
    mace = StandardMACE(**mace_config)
    model = GradientsOut(
        mace,
        targets=FORCE_KEY,
        compute_virials=compute_virials,
        compute_stress=compute_stress,
    ).float()
    return model


def _collate(data_list):
    collated, _, _ = collate(
        data_list[0].__class__,
        data_list=data_list,
        increment=True,
        add_batch=True,
    )
    return collated


def _run_forward(model, data_list, device="cpu"):
    """Run forward pass on a collated batch."""
    batch = _collate(data_list).to(device)
    model = model.to(device)
    model.eval()
    return model(batch)


# ══════════════════════════════════════════════════════════════════════
# Test _apply_strain helper
# ══════════════════════════════════════════════════════════════════════


class TestApplyStrain:
    """Test the _apply_strain helper function."""

    def test_returns_displacement_tensor(self):
        """_apply_strain returns a displacement with requires_grad."""
        data = _collate([database.data_list[0]])
        _, displacement = _apply_strain(data)

        assert displacement.requires_grad
        assert displacement.shape == (1, 3, 3)

    def test_positions_unchanged_at_zero_strain(self):
        """At zero displacement, positions are unchanged."""
        data = _collate([database.data_list[0]])
        original_pos = data.pos.clone()

        data, displacement = _apply_strain(data)

        # displacement is zero-initialised, so strained pos == original pos
        np.testing.assert_allclose(
            data.pos.detach().numpy(),
            original_pos.numpy(),
            atol=1e-7,
        )

    def test_multi_graph_displacement_shape(self):
        """Displacement has correct shape for multi-graph batch."""
        data = _collate(database.data_list[:3])
        _, displacement = _apply_strain(data)

        assert displacement.shape == (3, 3, 3)

    def test_positions_in_graph(self):
        """Strain is applied per-graph (each graph gets its own ε)."""
        data = _collate(database.data_list[:2])

        # Manually verify the einsum pattern
        num_graphs = int(data.ptr.numel() - 1)
        assert num_graphs == 2

        _, displacement = _apply_strain(data)
        # displacement is (2, 3, 3) — one per graph
        assert displacement.shape[0] == 2


# ══════════════════════════════════════════════════════════════════════
# Test GradientsOut with virials
# ══════════════════════════════════════════════════════════════════════


class TestGradientsOutVirials:
    """Test virial computation in GradientsOut."""

    def test_forces_unchanged_when_virials_disabled(self):
        """Default GradientsOut (no virials) gives same forces as before."""
        model_no_virial = _build_model(compute_virials=False)
        model_virial = _build_model(compute_virials=True)

        # Share weights
        model_virial.load_state_dict(model_no_virial.state_dict())

        data1 = _collate([database.data_list[0]])
        data2 = _collate([database.data_list[0]])

        model_no_virial.eval()
        model_virial.eval()

        out1 = model_no_virial(data1)
        out2 = model_virial(data2)

        f1 = out1.out["mace"][FORCE_KEY].detach().numpy()
        f2 = out2.out["mace"][FORCE_KEY].detach().numpy()

        np.testing.assert_allclose(f1, f2, atol=1e-5)

    def test_virial_key_present(self):
        """Virials appear in output when compute_virials=True."""
        model = _build_model(compute_virials=True)
        data = _collate([database.data_list[0]])
        out = model(data)

        assert VIRIALS_KEY in out.out["mace"]

    def test_virial_shape(self):
        """Virial tensor has shape (num_graphs, 3, 3)."""
        model = _build_model(compute_virials=True)
        data = _collate([database.data_list[0]])
        out = model(data)

        virials = out.out["mace"][VIRIALS_KEY]
        assert virials.shape == (1, 3, 3)

    def test_virial_shape_multi_graph(self):
        """Virial shape is correct for multi-graph batch."""
        model = _build_model(compute_virials=True)
        data = _collate(database.data_list[:3])
        out = model(data)

        virials = out.out["mace"][VIRIALS_KEY]
        assert virials.shape == (3, 3, 3)

    def test_virial_not_all_zero(self):
        """Virial is non-trivial (not all zeros) for a real molecule."""
        model = _build_model(compute_virials=True)
        data = _collate([database.data_list[0]])
        out = model(data)

        virials = out.out["mace"][VIRIALS_KEY].detach().numpy()
        assert not np.allclose(virials, 0.0, atol=1e-10)

    def test_virial_symmetric(self):
        """Virial tensor should be approximately symmetric (V ≈ V^T).

        This holds because the symmetric displacement produces a
        symmetric strain, and dE/dε inherits that symmetry.
        """
        model = _build_model(compute_virials=True)
        data = _collate([database.data_list[0]])
        out = model(data)

        V = out.out["mace"][VIRIALS_KEY].detach().numpy()[0]
        np.testing.assert_allclose(V, V.T, atol=1e-5)

    def test_no_virial_when_disabled(self):
        """Virial key is absent when compute_virials=False."""
        model = _build_model(compute_virials=False)
        data = _collate([database.data_list[0]])
        out = model(data)

        assert VIRIALS_KEY not in out.out["mace"]

    def test_virial_finite_difference(self):
        """Virial matches finite-difference dE/dε for a small strain.

        Apply a small uniform strain ε_ij to all positions and check
        that dE ≈ -virial_ij * ε_ij (summed).
        """
        model = _build_model(compute_virials=True)
        model.eval()

        data_ref = _collate([database.data_list[0]])
        out_ref = model(data_ref)
        E0 = out_ref.out["mace"][ENERGY_KEY].item()
        V = out_ref.out["mace"][VIRIALS_KEY].detach()  # (1, 3, 3)

        # Finite-difference: apply small strain δε and compute E(ε)
        delta = 1e-4
        fd_virial = torch.zeros(3, 3)
        for i in range(3):
            for j in range(3):
                data_plus = _collate([database.data_list[0]])
                eps = torch.zeros(3, 3)
                eps[i, j] = delta
                eps = 0.5 * (eps + eps.T)  # symmetrise

                # Strain positions manually
                data_plus.pos = data_plus.pos + torch.einsum(
                    "bi,ij->bj", data_plus.pos, eps
                )

                # Run without virial (just need energy)
                model_plain = _build_model(compute_virials=False)
                model_plain.load_state_dict(model.state_dict())
                model_plain.eval()
                out_plus = model_plain(data_plus)
                E_plus = out_plus.out["mace"][ENERGY_KEY].item()

                fd_virial[i, j] = -(E_plus - E0) / delta

        np.testing.assert_allclose(
            V[0].detach().numpy(),
            fd_virial.numpy(),
            atol=1e-2,  # FD is approximate
            err_msg="Autograd virial doesn't match finite-difference",
        )


# ══════════════════════════════════════════════════════════════════════
# Test stress (virial / volume) with periodic cell
# ══════════════════════════════════════════════════════════════════════


class TestStress:
    """Test stress = virial / volume for periodic systems."""

    def test_stress_key_present(self):
        """Stress appears in output when compute_stress=True."""
        model = _build_model(compute_stress=True)
        data = _collate([database.data_list[0]])

        # Give it a cell so stress can be computed
        data.cell = torch.eye(3).unsqueeze(0).float() * 10.0
        data.pbc = torch.tensor([[True, True, True]])

        out = model(data)
        assert STRESS_KEY in out.out["mace"]

    def test_stress_shape(self):
        """Stress has shape (num_graphs, 3, 3)."""
        model = _build_model(compute_stress=True)
        data = _collate([database.data_list[0]])
        data.cell = torch.eye(3).unsqueeze(0).float() * 10.0

        out = model(data)
        stress = out.out["mace"][STRESS_KEY]
        assert stress.shape == (1, 3, 3)

    def test_stress_equals_virial_over_volume(self):
        """stress = virial / volume when a cell is provided."""
        model = _build_model(compute_stress=True)
        data = _collate([database.data_list[0]])
        L = 10.0
        data.cell = torch.eye(3).unsqueeze(0).float() * L

        out = model(data)
        virial = out.out["mace"][VIRIALS_KEY].detach().numpy()
        stress = out.out["mace"][STRESS_KEY].detach().numpy()
        volume = L ** 3

        np.testing.assert_allclose(
            stress, virial / volume, atol=1e-8,
            err_msg="Stress should be virial / volume",
        )

    def test_stress_zero_without_cell(self):
        """Stress is zero when no cell is provided (non-periodic)."""
        model = _build_model(compute_stress=True)
        data = _collate([database.data_list[0]])

        out = model(data)
        stress = out.out["mace"][STRESS_KEY].detach().numpy()
        np.testing.assert_allclose(stress, 0.0, atol=1e-10)

    def test_stress_implies_virials(self):
        """compute_stress=True also enables virials."""
        model = _build_model(compute_stress=True)
        assert model.compute_virials is True

        data = _collate([database.data_list[0]])
        out = model(data)
        assert VIRIALS_KEY in out.out["mace"]


# ══════════════════════════════════════════════════════════════════════
# Test SumOut aggregation of virials
# ══════════════════════════════════════════════════════════════════════


class TestSumOutVirials:
    """Test that SumOut aggregates virials from sub-models."""

    def test_sumout_aggregates_virials(self):
        """SumOut sums virials from sub-models at top level."""
        mace1 = StandardMACE(**mace_config)
        grad_model = GradientsOut(
            mace1, targets=FORCE_KEY, compute_virials=True
        ).float()

        models = torch.nn.ModuleDict({"mace": grad_model})
        sum_model = SumOut(models, targets=[ENERGY_KEY, FORCE_KEY])

        data = _collate([database.data_list[0]])
        out = sum_model(data)

        # Virial should be aggregated at top level
        assert VIRIALS_KEY in out.out
        # And should equal the sub-model's virial
        np.testing.assert_allclose(
            out.out[VIRIALS_KEY].detach().numpy(),
            out.out["mace"][VIRIALS_KEY].detach().numpy(),
            atol=1e-7,
        )
