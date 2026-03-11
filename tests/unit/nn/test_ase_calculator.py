"""Tests for MLCGCalculator — the ASE Calculator adapter for mlcg models."""

import pytest
import os
import tempfile
import torch
import numpy as np

try:
    from typing import List
    from ase import Atoms
    from ase.build import molecule
    from torch_geometric.data import Batch
    from torch_geometric.data.collate import collate

    import mace
    from mlcg.geometry import Topology
    from mlcg.nn.mace import StandardMACE
    from mlcg.nn.gradients import GradientsOut, SumOut
    from mlcg.nn.ase_calculator import MLCGCalculator
    from mlcg.data.atomic_data import AtomicData
    from mlcg.data._keys import ENERGY_KEY, FORCE_KEY

    # ── Shared test fixtures ─────────────────────────────────────────

    class MolDatabase:
        """Small molecule database for testing (same pattern as test_mace.py)."""

        def __init__(
            self,
            mol_names: List[str] = ["H2O", "CH4", "NH3", "C2H6"],
        ):
            self.mol_names = mol_names
            self.molecules = [molecule(name) for name in self.mol_names]
            self.mol_topos = [Topology.from_ase(mol) for mol in self.molecules]
            self.data_list = []
            data_list = []
            for mol, topo in zip(self.molecules, self.mol_topos):
                neighbor_list = topo.neighbor_list("fully connected")
                data = AtomicData.from_points(
                    pos=torch.tensor(mol.get_positions()).float(),
                    atom_types=torch.tensor(mol.get_atomic_numbers()),
                    masses=torch.tensor(mol.get_masses()).float(),
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
        "MACE installation not found — skipping MLCGCalculator tests",
        allow_module_level=True,
    )


# ── Helper: build a fresh GradientsOut(StandardMACE) ─────────────────


def _build_model(device="cpu", dtype=torch.float32):
    """Build a small GradientsOut(MACE) model for testing."""
    mace = StandardMACE(**mace_config)
    model = GradientsOut(mace, targets=FORCE_KEY)
    model = model.to(device=device, dtype=dtype)
    return model


def _save_model(model, path):
    """Save model via torch.save."""
    torch.save(model, path)
    return path


# ══════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════


class TestMLCGCalculatorInit:
    """Test constructor and validation."""

    def test_from_model_object(self):
        """Construct calculator from an in-memory model."""
        model = _build_model()
        calc = MLCGCalculator(model=model, device="cpu")
        assert calc.model is model
        assert calc.device == "cpu"

    def test_from_model_path(self, tmp_path):
        """Construct calculator from a saved model file."""
        model = _build_model()
        path = _save_model(model, str(tmp_path / "model.pt"))

        calc = MLCGCalculator(model_path=path, device="cpu")
        assert isinstance(calc.model, torch.nn.Module)

    def test_no_model_raises(self):
        """Omitting both model and model_path raises ValueError."""
        with pytest.raises(ValueError, match="Either model_path or model"):
            MLCGCalculator(device="cpu")

    def test_missing_file_raises(self):
        """Non-existent model_path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            MLCGCalculator(model_path="/nonexistent/model.pt", device="cpu")

    def test_dict_checkpoint_raises(self, tmp_path):
        """A state_dict (plain dict) raises TypeError with helpful message."""
        path = str(tmp_path / "state_dict.pt")
        torch.save({"key": torch.tensor(1.0)}, path)
        with pytest.raises(TypeError, match="dictionary"):
            MLCGCalculator(model_path=path, device="cpu")

    def test_custom_keys(self):
        """Custom energy/force key names are stored."""
        model = _build_model()
        calc = MLCGCalculator(
            model=model, device="cpu", energy_key="E", force_key="F"
        )
        assert calc.energy_key == "E"
        assert calc.force_key == "F"

    def test_masses_stored(self):
        """CG masses are stored as float64 numpy array."""
        model = _build_model()
        masses = [72.0, 56.0, 28.0]
        calc = MLCGCalculator(model=model, device="cpu", masses=masses)
        assert calc.cg_masses is not None
        assert calc.cg_masses.dtype == np.float64
        np.testing.assert_allclose(calc.cg_masses, masses)

    def test_neighbor_list_stored(self):
        """Prior neighbor list dict is stored."""
        model = _build_model()
        nl = {"bonds": {"index_mapping": torch.tensor([[0, 1], [1, 0]])}}
        calc = MLCGCalculator(model=model, device="cpu", neighbor_list=nl)
        assert "bonds" in calc.prior_neighbor_list


class TestHasCueq:
    """Test the static cueq detection method."""

    def test_plain_model_no_cueq(self):
        """A standard e3nn model is not detected as cueq."""
        model = _build_model()
        assert not MLCGCalculator._has_cueq(model)

    def test_mock_cueq_module(self):
        """A model with a cueq_config attribute is detected."""

        class FakeCueqConfig:
            enabled = True

        model = _build_model()
        # Inject a fake cueq_config onto first child
        first_child = next(model.modules())
        first_child.cueq_config = FakeCueqConfig()
        assert MLCGCalculator._has_cueq(model)


class TestCalculateSingle:
    """Test single-structure ASE calculate() interface."""

    def test_energy_and_forces_returned(self):
        """calculate() populates results['energy'] and results['forces']."""
        model = _build_model()
        calc = MLCGCalculator(model=model, device="cpu")

        atoms = molecule("H2O")
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        assert isinstance(energy, float)
        assert forces.shape == (3, 3)

    def test_matches_direct_forward(self):
        """ASE calculator output matches direct model forward pass."""
        model = _build_model()
        calc = MLCGCalculator(model=model, device="cpu")

        atoms = molecule("H2O")
        atoms.calc = calc
        ase_energy = atoms.get_potential_energy()
        ase_forces = atoms.get_forces()

        # Direct forward pass
        data = AtomicData.from_ase(atoms)
        batch = Batch.from_data_list([data]).to("cpu")
        model_dtype = next(model.parameters()).dtype
        batch.pos = batch.pos.to(model_dtype)

        model.eval()
        out = model(batch)
        direct_energy = out.out[model.name][ENERGY_KEY].item()
        direct_forces = out.out[model.name][FORCE_KEY].detach().numpy()

        np.testing.assert_allclose(ase_energy, direct_energy, atol=1e-5)
        np.testing.assert_allclose(ase_forces, direct_forces, atol=1e-5)

    def test_cg_masses_override(self):
        """CG masses override ASE's periodic-table masses inside calculate()."""
        model = _build_model()
        cg_masses = np.array([72.0, 56.0, 28.0])
        calc = MLCGCalculator(model=model, device="cpu", masses=cg_masses)

        atoms = molecule("H2O")
        atoms.calc = calc
        _ = atoms.get_potential_energy()  # triggers calculate()

        # CG masses are applied on the calculator's internal copy (self.atoms)
        np.testing.assert_allclose(calc.atoms.get_masses(), cg_masses, atol=1e-5)

    def test_mass_length_mismatch_raises(self):
        """Mismatched mass length raises ValueError."""
        model = _build_model()
        calc = MLCGCalculator(model=model, device="cpu", masses=[1.0, 2.0])

        atoms = molecule("H2O")  # 3 atoms, 2 masses
        atoms.calc = calc
        with pytest.raises(ValueError, match="CG masses length"):
            atoms.get_potential_energy()

    @pytest.mark.parametrize("mol_name", ["H2O", "CH4", "NH3"])
    def test_multiple_molecules(self, mol_name):
        """Calculator works for various small molecules."""
        model = _build_model()
        calc = MLCGCalculator(model=model, device="cpu")

        atoms = molecule(mol_name)
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        assert np.isfinite(energy)
        assert np.all(np.isfinite(forces))
        assert forces.shape == (len(atoms), 3)


class TestCalculateBatch:
    """Test batched evaluation."""

    def test_batch_shapes(self):
        """calculate_batch returns correct shapes."""
        model = _build_model()
        calc = MLCGCalculator(model=model, device="cpu")

        structures = [molecule("H2O") for _ in range(4)]
        results = calc.calculate_batch(structures)

        assert "energies" in results
        assert "forces" in results
        assert results["energies"].shape == (4,)
        assert results["forces"].shape == (4, 3, 3)

    def test_batch_matches_single(self):
        """Batched results match single-structure results."""
        model = _build_model()
        calc = MLCGCalculator(model=model, device="cpu")

        mol = molecule("CH4")
        single_energies = []
        single_forces = []
        for _ in range(3):
            mol_copy = mol.copy()
            mol_copy.calc = calc
            single_energies.append(mol_copy.get_potential_energy())
            single_forces.append(mol_copy.get_forces())

        # Same structures in batch
        batch_results = calc.calculate_batch([mol.copy() for _ in range(3)])

        for i in range(3):
            np.testing.assert_allclose(
                batch_results["energies"][i], single_energies[i], atol=1e-5
            )
            np.testing.assert_allclose(
                batch_results["forces"][i], single_forces[i], atol=1e-5
            )

    def test_batch_with_atomic_data(self):
        """calculate_batch accepts AtomicData inputs."""
        model = _build_model()
        calc = MLCGCalculator(model=model, device="cpu")

        data_list = [
            AtomicData.from_ase(molecule("H2O")) for _ in range(3)
        ]
        results = calc.calculate_batch(data_list)

        assert results["energies"].shape == (3,)
        assert results["forces"].shape == (3, 3, 3)

    def test_batch_caching(self):
        """Second call reuses cached batch template."""
        model = _build_model()
        calc = MLCGCalculator(model=model, device="cpu")

        structures = [molecule("H2O") for _ in range(2)]
        _ = calc.calculate_batch(structures)

        assert hasattr(calc, "_batch_cache_key")
        assert calc._batch_cache_key == (2, 3)

        # Second call should hit cache
        _ = calc.calculate_batch(structures)
        assert calc._batch_cache_key == (2, 3)

    def test_batch_cache_invalidation(self):
        """Cache is rebuilt when batch size changes."""
        model = _build_model()
        calc = MLCGCalculator(model=model, device="cpu")

        _ = calc.calculate_batch([molecule("H2O") for _ in range(2)])
        assert calc._batch_cache_key == (2, 3)

        _ = calc.calculate_batch([molecule("H2O") for _ in range(4)])
        assert calc._batch_cache_key == (4, 3)

    def test_invalid_type_raises(self):
        """Passing unsupported type raises TypeError."""
        model = _build_model()
        calc = MLCGCalculator(model=model, device="cpu")

        with pytest.raises(TypeError, match="Expected"):
            calc.calculate_batch(["not_a_structure"])


class TestExtractResults:
    """Test the _extract_results method for both SumOut and GradientsOut."""

    def test_top_level_keys(self):
        """SumOut-style top-level keys are extracted."""
        model = _build_model()
        calc = MLCGCalculator(model=model, device="cpu")

        results_dict = {
            ENERGY_KEY: torch.tensor([1.5]),
            FORCE_KEY: torch.randn(3, 3),
        }
        energy, forces, stress = calc._extract_results(results_dict)
        assert energy is not None
        assert forces is not None
        assert stress is None  # no stress in input
        np.testing.assert_allclose(energy, 1.5, atol=1e-5)

    def test_nested_keys(self):
        """GradientsOut-style nested keys are extracted."""
        model = _build_model()
        calc = MLCGCalculator(model=model, device="cpu")

        results_dict = {
            "mace": {
                ENERGY_KEY: torch.tensor([2.5]),
                FORCE_KEY: torch.randn(3, 3),
            }
        }
        energy, forces, stress = calc._extract_results(results_dict)
        assert energy is not None
        assert forces is not None
        assert stress is None  # no stress in input
        np.testing.assert_allclose(energy, 2.5, atol=1e-5)


class TestModelSaveLoad:
    """Test save/load round-trip with the calculator."""

    def test_save_load_roundtrip(self, tmp_path):
        """Model saved and reloaded gives same results."""
        model = _build_model()
        path = str(tmp_path / "model.pt")
        torch.save(model, path)

        calc1 = MLCGCalculator(model=model, device="cpu")
        calc2 = MLCGCalculator(model_path=path, device="cpu")

        atoms = molecule("H2O")
        atoms.calc = calc1
        e1 = atoms.get_potential_energy()
        f1 = atoms.get_forces()

        atoms.calc = calc2
        e2 = atoms.get_potential_energy()
        f2 = atoms.get_forces()

        np.testing.assert_allclose(e1, e2, atol=1e-5)
        np.testing.assert_allclose(f1, f2, atol=1e-5)
