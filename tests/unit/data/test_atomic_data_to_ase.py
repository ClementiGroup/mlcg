"""Tests for AtomicData.to_ase() and the from_ase ↔ to_ase round-trip."""

import warnings

import pytest
import torch
import numpy as np
from ase import Atoms
from ase.build import molecule

from mlcg.data.atomic_data import AtomicData, MLCG_MASS_TO_AMU
from mlcg.data._keys import ENERGY_KEY, FORCE_KEY


# ── Helpers ───────────────────────────────────────────────────────────


def _make_atomic_data(
    n_atoms=5,
    atom_types=None,
    energy=None,
    forces=None,
    masses=None,
    pbc=False,
    cell=None,
    tag=None,
):
    """Create a minimal AtomicData for testing using from_points."""
    if atom_types is None:
        atom_types = torch.tensor([6, 6, 7, 1, 1])[:n_atoms]
    pos = torch.randn(len(atom_types), 3, dtype=torch.float32)
    if masses is None:
        masses = torch.tensor([12.0, 12.0, 14.0, 1.0, 1.0], dtype=torch.float32)[
            : len(atom_types)
        ]

    kwargs = dict(
        pos=pos,
        atom_types=atom_types,
        masses=masses,
        tag=tag,
    )
    if energy is not None:
        # energy must be shape (1,) to match n_atoms shape
        kwargs["energy"] = torch.tensor([energy], dtype=torch.float32)
    if forces is not None:
        kwargs["forces"] = torch.as_tensor(forces, dtype=torch.float32)
    if pbc is not False:
        kwargs["pbc"] = torch.tensor(pbc).view(-1, 3)
    if cell is not None:
        kwargs["cell"] = torch.tensor(cell, dtype=torch.float32).view(-1, 3, 3)

    return AtomicData.from_points(**kwargs)


# ── Tests ─────────────────────────────────────────────────────────────


class TestToAse:
    """Tests for AtomicData.to_ase()."""

    def test_basic_conversion(self):
        """to_ase returns an ase.Atoms with correct positions and numbers."""
        data = _make_atomic_data()
        atoms = data.to_ase()

        assert isinstance(atoms, Atoms)
        assert len(atoms) == len(data.atom_types)
        np.testing.assert_allclose(
            atoms.get_positions(), data.pos.numpy(), atol=1e-6
        )

    def test_atomic_numbers_standard(self):
        """Standard atom types (>= 1) are preserved."""
        data = _make_atomic_data(atom_types=torch.tensor([6, 7, 8]))
        atoms = data.to_ase()
        np.testing.assert_array_equal(atoms.get_atomic_numbers(), [6, 7, 8])

    def test_atomic_numbers_cg_shift(self):
        """CG bead types starting at 0 raise NotImplementedError."""
        data = _make_atomic_data(atom_types=torch.tensor([0, 1, 2]))
        with pytest.raises(NotImplementedError, match="atom types < 1"):
            data.to_ase()

    def test_masses_preserved(self):
        """CG masses survive the conversion (valid atom types)."""
        masses = torch.tensor([72.0, 56.0, 28.0], dtype=torch.float32)
        data = _make_atomic_data(
            atom_types=torch.tensor([1, 2, 3]), masses=masses
        )
        atoms = data.to_ase()
        np.testing.assert_allclose(atoms.get_masses(), masses.numpy(), atol=1e-5)

    def test_energy_in_info(self):
        """Scalar energy is stored in atoms.info['energy']."""
        data = _make_atomic_data(
            n_atoms=3,
            atom_types=torch.tensor([6, 7, 8]),
            energy=42.5,
        )
        atoms = data.to_ase()
        assert ENERGY_KEY in atoms.info
        np.testing.assert_allclose(atoms.info[ENERGY_KEY], 42.5, atol=1e-5)

    def test_forces_in_arrays(self):
        """Forces are stored in atoms.arrays['forces']."""
        forces = torch.randn(3, 3, dtype=torch.float32)
        data = _make_atomic_data(
            n_atoms=3,
            atom_types=torch.tensor([6, 7, 8]),
            forces=forces,
        )
        atoms = data.to_ase()
        assert FORCE_KEY in atoms.arrays
        np.testing.assert_allclose(
            atoms.arrays[FORCE_KEY], forces.numpy(), atol=1e-6
        )

    def test_pbc_and_cell(self):
        """PBC flags and cell vectors survive conversion."""
        cell = [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]]
        data = _make_atomic_data(
            n_atoms=3,
            atom_types=torch.tensor([6, 7, 8]),
            pbc=[[True, True, True]],
            cell=cell,
        )
        atoms = data.to_ase()
        np.testing.assert_array_equal(atoms.get_pbc(), [True, True, True])
        np.testing.assert_allclose(atoms.get_cell()[:], cell, atol=1e-6)

    def test_non_periodic(self):
        """Non-periodic structures get pbc=False."""
        data = _make_atomic_data(
            n_atoms=3,
            atom_types=torch.tensor([6, 7, 8]),
        )
        atoms = data.to_ase()
        assert not any(atoms.get_pbc())

    def test_tag_preserved(self):
        """String tag is stored in atoms.info."""
        data = _make_atomic_data(
            n_atoms=3,
            atom_types=torch.tensor([6, 7, 8]),
            tag="test_tag",
        )
        atoms = data.to_ase()
        assert atoms.info.get("tag") == "test_tag"

    def test_no_energy_no_forces(self):
        """Conversion works when energy/forces are absent."""
        data = _make_atomic_data(
            n_atoms=3,
            atom_types=torch.tensor([6, 7, 8]),
        )
        atoms = data.to_ase()
        assert ENERGY_KEY not in atoms.info
        assert FORCE_KEY not in atoms.arrays

    # ── CG mass auto-conversion tests ────────────────────────────────

    def test_cg_masses_auto_detected(self):
        """Masses all < 1.0 are auto-converted to amu with a warning."""
        cg_masses = torch.tensor([0.15, 0.13, 0.17], dtype=torch.float32)
        data = _make_atomic_data(
            atom_types=torch.tensor([1, 2, 3]), masses=cg_masses
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            atoms = data.to_ase()
            # Should have emitted a UserWarning about auto-conversion
            assert any("mlcg CG units" in str(x.message) for x in w)

        expected = cg_masses.numpy() * MLCG_MASS_TO_AMU
        np.testing.assert_allclose(atoms.get_masses(), expected, atol=1e-3)

    def test_cg_masses_explicit_true(self):
        """convert_cg_masses=True always converts, no warning."""
        cg_masses = torch.tensor([0.15, 0.13, 0.17], dtype=torch.float32)
        data = _make_atomic_data(
            atom_types=torch.tensor([1, 2, 3]), masses=cg_masses
        )
        atoms = data.to_ase(convert_cg_masses=True)
        expected = cg_masses.numpy() * MLCG_MASS_TO_AMU
        np.testing.assert_allclose(atoms.get_masses(), expected, atol=1e-3)

    def test_cg_masses_explicit_false(self):
        """convert_cg_masses=False suppresses conversion."""
        cg_masses = torch.tensor([0.15, 0.13, 0.17], dtype=torch.float32)
        data = _make_atomic_data(
            atom_types=torch.tensor([1, 2, 3]), masses=cg_masses
        )
        atoms = data.to_ase(convert_cg_masses=False)
        np.testing.assert_allclose(
            atoms.get_masses(), cg_masses.numpy(), atol=1e-6
        )

    def test_amu_masses_not_converted(self):
        """Masses >= 1.0 (amu) are never auto-converted."""
        amu_masses = torch.tensor([12.0, 14.0, 16.0], dtype=torch.float32)
        data = _make_atomic_data(
            atom_types=torch.tensor([6, 7, 8]), masses=amu_masses
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            atoms = data.to_ase()
            # No conversion warning should fire
            assert not any("mlcg CG units" in str(x.message) for x in w)

        np.testing.assert_allclose(
            atoms.get_masses(), amu_masses.numpy(), atol=1e-6
        )


class TestFromAseToAseRoundTrip:
    """Test that from_ase → to_ase is a faithful round-trip."""

    @pytest.mark.parametrize(
        "mol_name",
        ["H2O", "CH4", "C2H6", "NH3"],
    )
    def test_positions_roundtrip(self, mol_name):
        """Positions survive an ASE → AtomicData → ASE round-trip."""
        original = molecule(mol_name)
        data = AtomicData.from_ase(original)
        recovered = data.to_ase()

        np.testing.assert_allclose(
            recovered.get_positions(),
            original.get_positions(),
            atol=1e-6,
        )

    @pytest.mark.parametrize(
        "mol_name",
        ["H2O", "CH4", "C2H6", "NH3"],
    )
    def test_atomic_numbers_roundtrip(self, mol_name):
        """Atomic numbers survive the round-trip (standard elements)."""
        original = molecule(mol_name)
        data = AtomicData.from_ase(original)
        recovered = data.to_ase()

        np.testing.assert_array_equal(
            recovered.get_atomic_numbers(),
            original.get_atomic_numbers(),
        )

    def test_energy_forces_roundtrip(self):
        """Energy and forces survive the round-trip via info/arrays."""
        original = molecule("H2O")
        # from_ase passes info[energy_tag] → torch.as_tensor, which must
        # produce shape (1,) to match n_atoms.shape.  Use a 1-element array.
        original.info[ENERGY_KEY] = np.array([-76.5], dtype=np.float32)
        original.arrays[FORCE_KEY] = np.random.randn(3, 3).astype(np.float32)

        data = AtomicData.from_ase(original)
        recovered = data.to_ase()

        np.testing.assert_allclose(
            recovered.info[ENERGY_KEY],
            original.info[ENERGY_KEY],
            atol=1e-5,
        )
        np.testing.assert_allclose(
            recovered.arrays[FORCE_KEY],
            original.arrays[FORCE_KEY],
            atol=1e-5,
        )

    def test_masses_roundtrip(self):
        """Masses survive from_ase → to_ase."""
        original = molecule("H2O")
        data = AtomicData.from_ase(original)
        recovered = data.to_ase()

        np.testing.assert_allclose(
            recovered.get_masses(),
            original.get_masses(),
            atol=1e-5,
        )
