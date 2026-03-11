"""Tests for StandardMACE cuEquivariance conversion methods.

Tests convert_to_cueq(), convert_to_e3nn(), _has_cueq_modules(),
and the full e3nn → cueq → e3nn round-trip.

These tests require cuEquivariance to be installed and a CUDA device.
"""

import pytest
import torch
import numpy as np

try:
    from typing import List
    from ase.build import molecule
    from torch_geometric.data import Batch
    from torch_geometric.data.collate import collate

    import mace
    from mlcg.geometry import Topology
    from mlcg.nn.mace import StandardMACE, CUET_AVAILABLE
    from mlcg.nn.gradients import GradientsOut
    from mlcg.data.atomic_data import AtomicData
    from mlcg.data._keys import ENERGY_KEY, FORCE_KEY

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
        "MACE installation not found — skipping cueq tests",
        allow_module_level=True,
    )


# ── Helpers ───────────────────────────────────────────────────────────


def _build_mace(use_cueq=False, device="cpu"):
    """Build a small StandardMACE."""
    model = StandardMACE(**mace_config, use_cueq=use_cueq)
    return model.to(device).float()


def _forward(model, data, device="cpu"):
    """Run a forward pass and return (energy, forces) as numpy."""
    batch = Batch.from_data_list([data]).to(device)
    model_dtype = next(model.parameters()).dtype
    batch.pos = batch.pos.to(model_dtype)
    model.eval()
    with torch.no_grad():
        out = model(batch)
    energy = out.out["mace"][ENERGY_KEY].detach().cpu().numpy()
    return energy


# ══════════════════════════════════════════════════════════════════════
# Tests that work without cuEquivariance
# ══════════════════════════════════════════════════════════════════════


class TestInitKwargs:
    """_init_kwargs are stored correctly."""

    def test_init_kwargs_present(self):
        model = _build_mace()
        assert hasattr(model, "_init_kwargs")
        assert model._init_kwargs["r_max"] == mace_config["r_max"]
        assert model._init_kwargs["hidden_irreps"] == mace_config["hidden_irreps"]

    def test_use_cueq_flag(self):
        model = _build_mace(use_cueq=False)
        assert model._use_cueq is False

    def test_correlation_stored(self):
        model = _build_mace()
        assert model._correlation == [2]  # 1 interaction, correlation=2

    def test_max_ell_stored(self):
        model = _build_mace()
        assert model._max_ell == mace_config["max_ell"]


class TestHasCueqModules:
    """Test _has_cueq_modules() detection."""

    def test_e3nn_model_no_cueq(self):
        """Plain e3nn model returns False."""
        model = _build_mace(use_cueq=False)
        assert model._has_cueq_modules() is False

    def test_state_dict_key_detection(self):
        """Detection uses 'symmetric_contractions.weight' in state dict keys."""
        model = _build_mace(use_cueq=False)
        keys = list(model.state_dict().keys())
        # e3nn uses 'symmetric_contractions.contractions...' not '.weight'
        has_cueq_key = any(
            "symmetric_contractions.weight" in k for k in keys
        )
        assert has_cueq_key is False


class TestConvertWithoutCueq:
    """Test conversion methods when cuEquivariance is NOT available."""

    def test_convert_to_e3nn_noop(self):
        """convert_to_e3nn is a no-op on an already e3nn model."""
        model = _build_mace(use_cueq=False)
        result = model.convert_to_e3nn()
        assert result is model
        assert model._use_cueq is False

    def test_convert_to_cueq_without_install_raises(self):
        """convert_to_cueq raises RuntimeError when cueq not installed."""
        if CUET_AVAILABLE:
            pytest.skip("Test only relevant when cueq is NOT installed")
        model = _build_mace(use_cueq=False)
        with pytest.raises(RuntimeError, match="not installed"):
            model.convert_to_cueq()


# ══════════════════════════════════════════════════════════════════════
# Tests that require cuEquivariance + CUDA
# ══════════════════════════════════════════════════════════════════════

requires_cueq = pytest.mark.skipif(
    not CUET_AVAILABLE, reason="cuEquivariance not installed"
)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


@requires_cueq
@requires_cuda
class TestConvertToCueq:
    """Test e3nn → cueq conversion."""

    def test_convert_sets_flag(self):
        """convert_to_cueq sets _use_cueq = True."""
        model = _build_mace(use_cueq=False, device="cuda")
        model.convert_to_cueq()
        assert model._use_cueq is True

    def test_has_cueq_modules_after_convert(self):
        """_has_cueq_modules returns True after conversion."""
        model = _build_mace(use_cueq=False, device="cuda")
        model.convert_to_cueq()
        assert model._has_cueq_modules() is True

    def test_double_convert_is_noop(self):
        """Calling convert_to_cueq twice doesn't crash."""
        model = _build_mace(use_cueq=False, device="cuda")
        model.convert_to_cueq()
        model.convert_to_cueq()  # should be a no-op
        assert model._use_cueq is True

    def test_numerical_equivalence(self):
        """e3nn and cueq give (nearly) identical outputs."""
        model = _build_mace(use_cueq=False, device="cuda")
        data = database.data_list[0]  # H2O

        # e3nn reference
        e3nn_energy = _forward(model, data, device="cuda")

        # Convert to cueq
        model.convert_to_cueq()
        cueq_energy = _forward(model, data, device="cuda")

        # ir_mul layout should give essentially zero diff
        np.testing.assert_allclose(
            cueq_energy, e3nn_energy, atol=1e-4,
            err_msg="cueq output differs from e3nn beyond tolerance",
        )

    def test_use_cueq_init_flag(self):
        """use_cueq=True in __init__ auto-converts if cueq available."""
        model = StandardMACE(**mace_config, use_cueq=True).float().cuda()
        assert model._use_cueq is True
        assert model._has_cueq_modules() is True


@requires_cueq
@requires_cuda
class TestConvertToE3nn:
    """Test cueq → e3nn conversion."""

    def test_convert_back_clears_flag(self):
        """convert_to_e3nn sets _use_cueq = False."""
        model = _build_mace(use_cueq=False, device="cuda")
        model.convert_to_cueq()
        assert model._use_cueq is True

        model.convert_to_e3nn()
        assert model._use_cueq is False

    def test_no_cueq_modules_after_convert_back(self):
        """_has_cueq_modules returns False after converting back."""
        model = _build_mace(use_cueq=False, device="cuda")
        model.convert_to_cueq()
        model.convert_to_e3nn()
        assert model._has_cueq_modules() is False


@requires_cueq
@requires_cuda
class TestRoundTrip:
    """Test full e3nn → cueq → e3nn round-trip."""

    def test_energy_roundtrip(self):
        """Energy matches after e3nn → cueq → e3nn round-trip."""
        model = _build_mace(use_cueq=False, device="cuda")
        data = database.data_list[0]

        original_energy = _forward(model, data, device="cuda")

        model.convert_to_cueq()
        model.convert_to_e3nn()

        roundtrip_energy = _forward(model, data, device="cuda")

        np.testing.assert_allclose(
            roundtrip_energy,
            original_energy,
            atol=1e-4,
            err_msg="Energy changed after e3nn → cueq → e3nn round-trip",
        )

    def test_save_load_e3nn_after_roundtrip(self, tmp_path):
        """Model can be saved as e3nn after round-trip (no cueq dependency)."""
        model = _build_mace(use_cueq=False, device="cuda")
        model.convert_to_cueq()
        model.convert_to_e3nn()

        path = str(tmp_path / "model_e3nn.pt")
        torch.save(model.cpu(), path)

        # Reload on CPU — should work without cueq
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        assert isinstance(loaded, StandardMACE)
        assert not loaded._has_cueq_modules()


@requires_cueq
@requires_cuda
class TestReplaceModules:
    """Test the _replace_modules helper."""

    def test_modules_replaced(self):
        """_replace_modules swaps all child modules from source."""
        model = _build_mace(use_cueq=False, device="cuda")

        original_interaction_id = id(model.interactions[0])
        model.convert_to_cueq()

        # After conversion, the interaction module should be different
        assert id(model.interactions[0]) != original_interaction_id

    def test_buffers_preserved(self):
        """Buffers (atomic_numbers, types_mapping) are preserved."""
        model = _build_mace(use_cueq=False, device="cuda")
        original_numbers = model.atomic_numbers.clone()

        model.convert_to_cueq()

        torch.testing.assert_close(model.atomic_numbers, original_numbers)
