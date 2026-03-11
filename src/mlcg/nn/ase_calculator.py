## Written by Zak El-Machachi (@zakmachachi) for mlcg-MACE with assistance
## from Claude Opus 4.6.

import os
import logging

import numpy as np
import torch
from ase import Atoms, units as ase_units
from ase.calculators.calculator import Calculator, all_changes
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from torch_geometric.data import Batch
from torch_geometric.data.collate import collate

from ..data.atomic_data import AtomicData, MLCG_MASS_TO_AMU
from ..data._keys import ENERGY_KEY, FORCE_KEY, STRESS_KEY

logger = logging.getLogger(__name__)

# Supported model energy units.  Conversion factor to eV.
# ASE's base energy unit is eV, so "eV" maps to 1.0.
UNIT_FACTORS = {
    "eV": 1.0,
    "kcal/mol": ase_units.kcal / ase_units.mol,
    "kJ/mol": ase_units.kJ / ase_units.mol,
    "Ha": ase_units.Ha,
    "Ry": ase_units.Ry,
}


def _stress_3x3_to_voigt(stress_3x3):
    """Convert a ``(*, 3, 3)`` stress tensor to ASE Voigt 6-vector form.

    ASE order: ``xx, yy, zz, yz, xz, xy``.

    Parameters
    ----------
    stress_3x3 : ndarray
        Stress tensor with shape ``(3, 3)`` or ``(N, 3, 3)``.

    Returns
    -------
    ndarray
        Voigt representation with shape ``(6,)`` or ``(N, 6)``.
    """
    if stress_3x3.ndim == 2:
        out = np.zeros(6)
        out[0] = stress_3x3[0, 0]
        out[1] = stress_3x3[1, 1]
        out[2] = stress_3x3[2, 2]
        out[3] = stress_3x3[1, 2]
        out[4] = stress_3x3[0, 2]
        out[5] = stress_3x3[0, 1]
    else:
        n = stress_3x3.shape[0]
        out = np.zeros((n, 6))
        out[:, 0] = stress_3x3[:, 0, 0]
        out[:, 1] = stress_3x3[:, 1, 1]
        out[:, 2] = stress_3x3[:, 2, 2]
        out[:, 3] = stress_3x3[:, 1, 2]
        out[:, 4] = stress_3x3[:, 0, 2]
        out[:, 5] = stress_3x3[:, 0, 1]
    return out


class MLCGCalculator(Calculator):
    """ASE Calculator for mlcg models (MACE, SchNet, Allegro, PaiNN, etc.).

    Handles two model types:

    - **CG with priors** (``SumOut``): contains prior terms
      (HarmonicBonds, HarmonicAngles, Repulsion, etc.) that require
      fixed topology-based neighbor lists.  Pass these via *neighbor_list*.
    - **Fully reactive** (``GradientsOut``, no priors): the NN computes
      its own neighbor list on-the-fly from a radial cutoff.

    Notes
    -----
    **Masses** -- ASE derives masses from atomic numbers, which are
    meaningless for CG beads (whose "atomic numbers" are bead-type
    indices).  Pass *masses* to override, or use :meth:`from_dataset`.

    **PBC / Cell** -- mlcg proteins are non-periodic.  Create ASE ``Atoms``
    with ``pbc=False`` (default); ``AtomicData.from_ase`` handles it.

    **Units** -- mlcg uses **kcal/mol** and **Angstrom**.  Beta is a
    simulation parameter; pass it to your dynamics driver (e.g.
    ``ase.md.Langevin(temperature_K=...)``).

    **cuEquivariance** -- cueq modules are preserved by ``torch.save`` /
    ``torch.load`` and detected automatically.

    **Batched evaluation** -- :meth:`calculate_batch` evaluates multiple
    structures in one forward pass, caching the collated batch template
    for subsequent calls (only positions are updated).

    **Unit conversion** -- By default the calculator returns energies and
    forces in whatever units the model uses (``energy_units="eV"``
    means no conversion).  Set ``energy_units="kcal/mol"`` (or
    ``"kJ/mol"``, ``"Ha"``, ``"Ry"``) to automatically convert the
    model output to ASE's native eV / eV/Angstrom so that ASE dynamics
    drivers (Langevin, VelocityVerlet, ...) work out of the box.

    Parameters
    ----------
    model_path : str, optional
        Path to a ``torch.save``'d model file.
    model : torch.nn.Module, optional
        Pre-loaded model (alternative to *model_path*).
    neighbor_list : dict, optional
        Fixed topology-based neighbor list dict for prior models.
        Required for ``SumOut`` models containing priors.
    masses : array-like, optional
        CG bead masses ``(n_atoms,)``.  Overrides ASE's
        periodic-table masses.
    device : str
        ``'cpu'`` or ``'cuda'``.
    energy_key : str
        Key for energy in model output dict.
    force_key : str
        Key for forces in model output dict.
    stress_key : str
        Key for stress in model output dict.
    energy_units : str
        Energy units of the **model output**.
        Supported: ``'eV'``, ``'kcal/mol'``, ``'kJ/mol'``,
        ``'Ha'``, ``'Ry'``.
        The calculator converts to ASE's native eV automatically.
    mlcg_units : bool
        If ``True``, treat *masses* as mlcg CG masses in
        kcal ps^2/(mol Angstrom^2) and convert to amu (x 418.4).
        Set ``False`` (default) for atomistic systems where
        masses are already in amu (g/mol).
    compute_stress : bool
        If ``True``, enable virial/stress computation on all
        ``GradientsOut`` layers inside the model.
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        model_path=None,
        model=None,
        neighbor_list=None,
        masses=None,
        device="cuda",
        energy_key=ENERGY_KEY,
        force_key=FORCE_KEY,
        stress_key=STRESS_KEY,
        energy_units="eV",
        mlcg_units=False,
        compute_stress=False,
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)
        self.device = device
        self.energy_key = energy_key
        self.force_key = force_key
        self.stress_key = stress_key
        self.prior_neighbor_list = neighbor_list or {}
        self._mlcg_units = mlcg_units
        if masses is not None:
            masses = np.asarray(masses, dtype=np.float64)
            if mlcg_units:
                masses = masses * MLCG_MASS_TO_AMU
                logger.info(
                    f"mlcg_units=True: masses x {MLCG_MASS_TO_AMU} -> amu "
                    f"(range {masses.min():.2f}-{masses.max():.2f} g/mol)"
                )
        self.cg_masses = masses

        if energy_units not in UNIT_FACTORS:
            raise ValueError(
                f"Unknown energy_units={energy_units!r}. "
                f"Supported: {list(UNIT_FACTORS.keys())}"
            )
        self._energy_units = energy_units
        self._to_eV = UNIT_FACTORS[energy_units]
        if self._to_eV != 1.0:
            logger.info(
                f"Model energy units: {energy_units} -> converting to eV "
                f"(factor={self._to_eV:.6g})"
            )

        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = self._load_model(model_path, device)
        else:
            raise ValueError("Either model_path or model must be provided.")

        # Enable virial/stress on GradientsOut layers if requested
        if compute_stress:
            self._enable_stress(self.model)

        uses_cueq = self._has_cueq(self.model)
        if uses_cueq:
            logger.info("Model uses cuEquivariance acceleration.")

        self.model.to(self.device)
        self.model.eval()

    # ── Constructors ──────────────────────────────────────────────────

    @classmethod
    def from_dataset(
        cls, model_path=None, model=None, dataset=None, device="cuda",
        energy_units="eV", mlcg_units=True, **kwargs,
    ):
        """Build a calculator, extracting topology NLs and masses from *dataset*.

        Parameters
        ----------
        model_path : str, optional
            Path to a ``torch.save``'d model file.
        model : torch.nn.Module, optional
            Pre-loaded model (alternative to *model_path*).
        dataset : InMemoryDataset, optional
            An mlcg ``InMemoryDataset`` (e.g. ``ChignolinDataset``)
            that provides ``priors_cls``, ``topologies``, and CG masses.
        device : str
            ``'cpu'`` or ``'cuda'``.
        energy_units : str
            Energy units of the model output (see constructor).
        mlcg_units : bool
            Convert mlcg CG masses to amu (default ``True``
            because dataset masses are in mlcg units).

        Example
        -------
        ::

            from mlcg.datasets import ChignolinDataset
            dataset = ChignolinDataset(root="./data", terminal_embeds=True)
            calc = MLCGCalculator.from_dataset(
                model_path="model.pt", dataset=dataset, device="cuda",
                energy_units="kcal/mol",
            )
        """
        prior_nls = {}
        masses = None
        if dataset is not None:
            topo_name = next(iter(dataset.topologies))
            topology = dataset.topologies[topo_name]
            for prior_cls in dataset.priors_cls:
                prior_nls.update(**prior_cls.neighbor_list(topology))

            sample = dataset.get(0)
            if hasattr(sample, "masses") and sample.masses is not None:
                masses = sample.masses.numpy()

        return cls(
            model_path=model_path,
            model=model,
            neighbor_list=prior_nls,
            masses=masses,
            device=device,
            energy_units=energy_units,
            mlcg_units=mlcg_units,
            **kwargs,
        )

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _load_model(model_path, device="cuda"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = torch.load(model_path, map_location=device, weights_only=False)

        if isinstance(model, dict):
            raise TypeError(
                f"Loaded object is a dictionary (keys: {list(model.keys())[:5]}). "
                "Expected a torch.nn.Module saved via torch.save(model, path). "
                "Lightning .ckpt files are not supported directly -- use "
                "mlcg.pl.merge_priors_and_checkpoint() first, then torch.save()."
            )
        return model

    @staticmethod
    def _has_cueq(model):
        """Detect cuEquivariance modules (works for MACE, Allegro, etc.)."""
        for module in model.modules():
            if hasattr(module, "cueq_config") and module.cueq_config is not None:
                if getattr(module.cueq_config, "enabled", False):
                    return True
            mod_pkg = getattr(type(module), "__module__", "") or ""
            if "cuequivariance" in mod_pkg:
                return True
        return False

    @staticmethod
    def _enable_stress(model):
        """Set ``compute_virials`` and ``compute_stress`` on all GradientsOut layers.

        Works for both new models and old checkpoints that were saved
        before virial support was added (the attributes simply get
        monkey-patched onto the deserialized objects).
        """
        from .gradients import GradientsOut

        count = 0
        for module in model.modules():
            if isinstance(module, GradientsOut):
                module.compute_virials = True
                module.compute_stress = True
                count += 1
        if count:
            logger.info(
                f"Enabled virial/stress on {count} GradientsOut layer(s)."
            )
        else:
            logger.warning(
                "compute_stress=True but no GradientsOut layers found "
                "in the model -- stress will not be computed."
            )

    def _detect_architectures(self):
        """Return a list of architecture names in the model.

        Works for SumOut (which wraps GradientsOut/EnergyOut sub-models)
        and for single-architecture models.
        """
        names = []
        model = self.model
        # SumOut wraps models in a ModuleDict
        if hasattr(model, "models") and hasattr(model.models, "items"):
            for name, submod in model.models.items():
                # GradientsOut / EnergyOut wrap the actual architecture
                inner = getattr(submod, "model", submod)
                names.append(type(inner).__name__)
        else:
            names.append(type(model).__name__)
        return names

    def prepare_atoms(self, atoms):
        """Ensure *atoms* has correct CG masses from this calculator.

        Call this after creating an ``Atoms`` object via
        ``config.to_ase()`` and **before** initialising velocities
        (e.g. ``MaxwellBoltzmannDistribution``).  ``to_ase()`` now
        auto-detects and converts mlcg masses, so this method is only
        needed when the calculator's ``cg_masses`` differ from those
        stored on the ``AtomicData`` (rare).
        """
        if self.cg_masses is not None:
            if len(self.cg_masses) != len(atoms):
                raise ValueError(
                    f"CG masses length ({len(self.cg_masses)}) != number of "
                    f"atoms ({len(atoms)})."
                )
            atoms.set_masses(self.cg_masses)

    # ── Single-structure evaluation ───────────────────────────────────

    def calculate(
        self, atoms=None, properties=("energy", "forces", "stress"),
        system_changes=all_changes,
    ):
        Calculator.calculate(self, atoms, properties, system_changes)

        if self.cg_masses is not None:
            if len(self.cg_masses) != len(self.atoms):
                raise ValueError(
                    f"CG masses length ({len(self.cg_masses)}) != number of "
                    f"atoms ({len(self.atoms)})."
                )
            self.atoms.set_masses(self.cg_masses)

        data = AtomicData.from_ase(
            self.atoms,
            energy_tag=self.energy_key,
            force_tag=self.force_key,
            stress_tag=self.stress_key,
        )
        if self.prior_neighbor_list:
            data.neighbor_list = self.prior_neighbor_list

        batch = Batch.from_data_list([data]).to(self.device)

        model_dtype = next(self.model.parameters()).dtype
        if batch.pos.dtype != model_dtype:
            batch.pos = batch.pos.to(model_dtype)
            if hasattr(batch, "cell") and batch.cell is not None:
                batch.cell = batch.cell.to(model_dtype)
            if hasattr(batch, "masses") and batch.masses is not None:
                batch.masses = batch.masses.to(model_dtype)

        out = self.model(batch)
        energy, forces, stress = self._extract_results(out.out)

        if energy is not None:
            self.results["energy"] = energy * self._to_eV
        if forces is not None:
            self.results["forces"] = forces.reshape((len(self.atoms), 3)) * self._to_eV
        if stress is not None:
            stress_3x3 = stress.reshape((3, 3)) * self._to_eV
            self.results["stress"] = _stress_3x3_to_voigt(stress_3x3)

    # ── Batched evaluation ────────────────────────────────────────────

    def calculate_batch(self, structures):
        """Evaluate multiple structures in one batched forward pass.

        On the first call the structures are collated into a single
        ``AtomicData`` batch that is cached on the device.  Subsequent
        calls only update positions (and cells for periodic systems)
        in-place -- the same pattern used by mlcg's Langevin integrator.

        Architecture-agnostic: works with MACE, Allegro, SchNet, PaiNN,
        So3krates, TorchMD-Net, and any other mlcg model that follows
        the standard ``forward(data) -> data.out`` protocol.

        Supports both non-periodic (CG proteins) and periodic (bulk)
        systems.  For periodic systems the cell is updated every call so
        that NPT dynamics (variable cell) work correctly.

        Parameters
        ----------
        structures : list
            List of ``AtomicData`` **or** list of ASE ``Atoms``.

        Returns
        -------
        dict
            ``'energies'`` ``(N,)``, ``'forces'`` ``(N, n_atoms, 3)``,
            and ``'stresses'`` ``(N, 3, 3)`` (when stress is available).
        """
        n_structures = len(structures)
        model_dtype = next(self.model.parameters()).dtype
        first = structures[0]

        if isinstance(first, AtomicData):
            n_atoms = len(first.atom_types)
        elif isinstance(first, Atoms):
            n_atoms = len(first)
        else:
            raise TypeError(
                f"Expected AtomicData or ase.Atoms, got {type(first).__name__}"
            )

        cache_key = (n_structures, n_atoms)
        if not hasattr(self, "_batch_cache_key") or self._batch_cache_key != cache_key:
            data_list = [self._to_atomic_data(s) for s in structures]
            self._batch_template = self._collate(data_list).to(self.device)
            if hasattr(self._batch_template, "cell") and self._batch_template.cell is not None:
                self._batch_template.cell = self._batch_template.cell.to(model_dtype)
            if hasattr(self._batch_template, "masses") and self._batch_template.masses is not None:
                self._batch_template.masses = self._batch_template.masses.to(model_dtype)
            self._batch_cache_key = cache_key
            # Detect periodicity for cell updates
            self._is_periodic = (
                isinstance(first, Atoms) and np.any(first.get_pbc())
            )
            # Detect architecture(s) for logging
            arch_names = self._detect_architectures()
            logger.info(
                f"Built batch template: {n_structures} structures x {n_atoms} atoms"
                f" (pbc={self._is_periodic}, arch={arch_names})."
            )
        # Update positions
        if isinstance(first, AtomicData):
            positions = torch.cat([s.pos for s in structures], dim=0)
        else:
            positions = torch.from_numpy(
                np.concatenate([s.get_positions() for s in structures], axis=0)
            )
        self._batch_template.pos = positions.to(
            dtype=model_dtype, device=self.device
        )

        # Update cells for periodic systems (needed for NPT / variable-cell MD)
        if getattr(self, "_is_periodic", False) and isinstance(first, Atoms):
            cells = torch.from_numpy(
                np.stack([s.get_cell().array for s in structures], axis=0)
            ).reshape(-1, 3)   # collated shape: (n_structures * 3, 3)
            self._batch_template.cell = cells.to(
                dtype=model_dtype, device=self.device
            )

        # Clear stale outputs from previous forward pass (mirrors mlcg's
        # ``data.out = {}`` pattern).
        self._batch_template.out = {}

        # Forward pass through the model
        out = self.model(self._batch_template)
        # Extract energy, forces, stress from the nested output dict
        energy, forces, stress = self._extract_results(out.out, batched=True)

        results = {}
        if energy is not None:
            results["energies"] = energy.reshape(n_structures) * self._to_eV
        if forces is not None:
            results["forces"] = forces.reshape(n_structures, n_atoms, 3) * self._to_eV
        if stress is not None:
            stress = stress.reshape(n_structures, 3, 3) * self._to_eV
            results["stress"] = _stress_3x3_to_voigt(stress)
        return results


    # ── Result extraction (shared) ────────────────────────────────────

    def _extract_results(self, results_dict, batched=False):
        """Pull energy/forces/stress from mlcg's nested output dict.

        ``SumOut`` puts aggregated keys at the top level; ``GradientsOut``
        nests them under the model name.
        """
        energy, forces, stress = None, None, None

        # Top-level keys (SumOut / single-arch aggregate)
        if self.energy_key in results_dict:
            e = results_dict[self.energy_key]
            energy = e.detach().cpu().numpy() if batched else (
                e.item() if e.numel() == 1 else e.sum().item()
            )
        if self.force_key in results_dict:
            forces = results_dict[self.force_key].detach().cpu().numpy()
        if self.stress_key in results_dict:
            stress = results_dict[self.stress_key].detach().cpu().numpy()

        # Nested keys (GradientsOut wraps under model name)
        if energy is None or forces is None or stress is None:
            for value in results_dict.values():
                if isinstance(value, dict):
                    if energy is None and self.energy_key in value:
                        e = value[self.energy_key]
                        energy = e.detach().cpu().numpy() if batched else (
                            e.item() if e.numel() == 1 else e.sum().item()
                        )
                    if forces is None and self.force_key in value:
                        forces = value[self.force_key].detach().cpu().numpy()
                    if stress is None and self.stress_key in value:
                        stress = value[self.stress_key].detach().cpu().numpy()

        return energy, forces, stress

    # ── Data helpers ──────────────────────────────────────────────────

    def _to_atomic_data(self, structure):
        """Convert a single structure to ``AtomicData``."""
        if isinstance(structure, AtomicData):
            data = structure
        elif isinstance(structure, Atoms):
            if self.cg_masses is not None:
                structure.set_masses(self.cg_masses)
            data = AtomicData.from_ase(
                structure,
                energy_tag=self.energy_key,
                force_tag=self.force_key,
                stress_tag=self.stress_key,
            )
        else:
            raise TypeError(
                f"Expected AtomicData or ase.Atoms, got {type(structure).__name__}"
            )
        if self.prior_neighbor_list and not data.neighbor_list:
            data.neighbor_list = self.prior_neighbor_list
        return data

    @staticmethod
    def _collate(data_list):
        """Collate a list of ``AtomicData`` into a single batch."""
        collated, _, _ = collate(
            data_list[0].__class__,
            data_list=data_list,
            increment=True,
            add_batch=True,
        )
        return collated


class _CachedCalculator(Calculator):
    """Dummy calculator that stores results from a batched evaluation.

    Used internally by :class:`BatchedMD` so that ASE observers
    (loggers, trajectory writers) can call
    ``atoms.get_forces()`` / ``atoms.get_potential_energy()`` after
    each step.

    When ``_frozen`` is ``True``, :meth:`calculate` is a no-op:
    existing results are preserved even if positions have changed.
    This is used by the predict-correct loop in :class:`BatchedMD`.
    """

    implemented_properties = ["energy", "forces", "stress"]
    _frozen = False

    def check_state(self, atoms, tol=1e-15):
        if self._frozen:
            return []  # Pretend nothing changed -> keep cached results
        return Calculator.check_state(self, atoms, tol)

    def calculate(self, atoms=None, properties=("energy", "forces", "stress"),
                  system_changes=all_changes):
        if self._frozen:
            return  # keep existing results -- will be corrected later
        # Results are injected externally; if we get here with no results
        # it means someone asked before the first batched eval.
        Calculator.calculate(self, atoms, properties, system_changes)
        if "energy" not in self.results:
            raise RuntimeError(
                "_CachedCalculator: no results available. "
                "Run at least one batched evaluation first."
            )

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Thermostat-agnostic batched MD -- works with ANY ASE dynamics driver #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class BatchedMD:
    """Run N independent ASE MD simulations with batched GPU force evaluation.

    Works with **any** ASE dynamics driver (``Langevin``,
    ``Bussi``, ``NPT``, ``Velocity Verlet``, ...).
    Each replica has its own thermostat / barostat with independent
    noise and state.

    **How it works (predict-correct)**

    All Velocity-Verlet-family integrators follow the same pattern
    per step:

    1. Half-kick using forces at *current* positions (cache hit).
    2. Position update (drift).
    3. Force evaluation at *new* positions -> second half-kick.

    ``BatchedMD`` exploits this:

    - **Predict**: freeze each replica's calculator so that *all*
      ``get_forces()`` calls (including the one after position
      update) return the previously cached forces.  Run each
      ``dyn.step()`` sequentially -- positions are computed
      correctly, but momenta are approximate.
    - **Batch evaluate**: one batched GPU call for all replicas'
      new positions.
    - **Correct**: adjust each replica's momenta by the exact
      difference ``correction_factor x (F_new - F_old)``.

    The correction factor is ``dt/2`` for plain VelocityVerlet and
    ``c1 = dt/2 - dt^2*gamma/8`` for ASE's Langevin (BAOAB).  The class
    reads the appropriate coefficient from the dynamics object
    automatically.

    Result: **1 GPU call per step, single-threaded, zero overhead**.

    Parameters
    ----------
    calculator : MLCGCalculator
        Base calculator with :meth:`calculate_batch`.
    atoms_list : list[Atoms]
        One ``Atoms`` per replica.
    DynClass : type
        ASE dynamics class (e.g. ``ase.md.Langevin``).
    dyn_kwargs : dict
        Keyword arguments for ``DynClass(atoms, **dyn_kwargs)``.
    init_temperature_K : float, optional
        If set, initialise velocities from a Maxwell-Boltzmann
        distribution at this temperature before the first step.
        Each replica gets an **independent** random draw.
    remove_com : bool
        Zero centre-of-mass momentum after velocity init
        (``Stationary``).  Default ``True``.  Removes net
        translational drift -- harmless for free-energy sampling
        (only 3 DoF removed from 3N).
    remove_rotation : bool
        Zero angular momentum after velocity init
        (``ZeroRotation``).  Default ``True`` for non-periodic
        systems (CG proteins), ``False`` for periodic/bulk.
        Set explicitly to override auto-detection.

    Example
    -------
    ::

        from ase.md.langevin import Langevin
        from ase import units
        from mlcg.nn import MLCGCalculator, BatchedMD

        calc = MLCGCalculator(model=model, neighbor_list=nls,
                              masses=masses, device='cuda',
                              energy_units='kcal/mol', mlcg_units=True)
        # to_ase() auto-detects CG masses and converts to amu
        atoms_list = [config.to_ase() for config in configs]

        runner = BatchedMD(calc, atoms_list, Langevin,
                           dict(timestep=2*units.fs, temperature_K=300,
                                friction=0.01/units.fs),
                           init_temperature_K=300)
        runner.run(10_000, log_interval=100)
    """

    def __init__(self, calculator, atoms_list, DynClass, dyn_kwargs,
                 init_temperature_K=None, remove_com=True,
                 remove_rotation=None):
        self._calc = calculator
        self.n_replicas = len(atoms_list)
        self.atoms_list = atoms_list
        self.nsteps = 0
        self._n_atoms = len(atoms_list[0])

        # ── Velocity initialisation ─────────────────────────────────
        if init_temperature_K is not None:
            self._init_velocities(
                init_temperature_K, remove_com, remove_rotation
            )

        # Disable gradient tracking on model parameters (inference only)
        for p in self._calc.model.parameters():
            p.requires_grad = False

        # Each replica gets a _CachedCalculator with freeze support.
        self._caches = []
        for atoms in atoms_list:
            cache = _CachedCalculator()
            atoms.calc = cache
            self._caches.append(cache)

        # Pre-allocate numpy buffer for F_old  (n_replicas, n_atoms, 3)
        self._F_old = np.zeros(
            (self.n_replicas, self._n_atoms, 3), dtype=np.float64
        )

        # Pre-allocate numpy buffer for Stress_old (n_replicas, 6)
        # ASE's get_stress() and our results["stress"] use the 6-component Voigt format
        self._Stress_old = np.zeros(
            (self.n_replicas, 6), dtype=np.float64
        )

        # Initial batch evaluation -- fills caches BEFORE constructing
        # dynamics objects.  Some integrators (e.g. LangevinBAOAB) call
        # atoms.get_forces() inside __init__, so the caches must already
        # contain valid results at construction time.
        self._inject_batch(self._batch_eval())

        # Now construct dynamics objects (safe -- caches are populated).
        self.dynamics = []
        for atoms in atoms_list:
            dyn = DynClass(atoms, **dyn_kwargs)
            self.dynamics.append(dyn)

        # Pre-compute momentum-correction factor (constant across steps).
        self._corr_c = self._correction_factor(self.dynamics[0])

        # Some dynamics (MelchionnaNPT) require explicit initialization
        # that is normally done by dyn.run()/irun() before the first
        # step().  Since BatchedMD calls step() directly, we trigger
        # initialization here -- *after* the caches have been filled so
        # that get_forces()/get_stress() work inside initialize().
        for dyn in self.dynamics:
            if hasattr(dyn, "initialized") and not dyn.initialized:
                dyn.initialize()

        logger.info(
            f"BatchedMD: {self.n_replicas} replicas x "
            f"{self._n_atoms} atoms  (predict-correct)"
        )

    # ── Velocity initialisation ───────────────────────────────────────

    def _init_velocities(self, temperature_K, remove_com, remove_rotation):
        """Maxwell-Boltzmann velocity initialisation for all replicas.

        Each replica gets an **independent** random draw.
        Optionally removes centre-of-mass translation (``Stationary``)
        and/or overall rotation (``ZeroRotation``).

        Notes
        -----
        * ``Stationary`` removes 3 translational DoF -> no effect on
          conformational free energy.
        * ``ZeroRotation`` removes 3 rotational DoF -> appropriate for
          non-periodic molecules (CG proteins), not for periodic/bulk.
        * **Langevin** is immune to the flying ice cube effect because
          each atom gets independent stochastic kicks + friction, so
          these projections are for clean initial conditions, not
          thermostat stability.
        * **Berendsen / Bussi / Nosé-Hoover**: if using one of these,
          ``Stationary`` is important -- otherwise COM translation
          accumulates energy (flying ice cube with Berendsen/Bussi) or
          persistent drift (Nosé-Hoover).
        """
        # Auto-detect: remove rotation for non-periodic systems
        if remove_rotation is None:
            is_periodic = np.any(self.atoms_list[0].get_pbc())
            remove_rotation = not is_periodic

        for atoms in self.atoms_list:
            MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
            if remove_com:
                Stationary(atoms)
            if remove_rotation:
                ZeroRotation(atoms)

        temps = self.get_temperatures()
        logger.info(
            f"Velocities initialised: T_target={temperature_K:.1f} K, "
            f"<T_init>={temps.mean():.1f} K "
            f"(range {temps.min():.1f}-{temps.max():.1f} K), "
            f"COM_removed={remove_com}, rot_removed={remove_rotation}"
        )

    # ── Core predict-correct step ─────────────────────────────────────

    def _batch_eval(self):
        """One batched GPU call for all replicas at current positions."""
        return self._calc.calculate_batch(self.atoms_list)

    def _inject_batch(self, results):
        """Push batch results into every replica's cached calculator."""
        for i, cache in enumerate(self._caches):
            cache.results = {
                "energy": float(results["energies"][i]),
                "forces": results["forces"][i].copy(),
                "stress": results["stress"][i].copy() if "stress" in results else None,
            }
            # Store a *reference* (not a copy) to the replica's Atoms.
            # Since cache.atoms IS atoms, ASE's compare_atoms() always
            # sees identical positions -> guaranteed cache hit.  This
            # avoids 100x deep-copy overhead per step.
            cache.atoms = self.atoms_list[i]

    @staticmethod
    def _correction_factor(dyn):
        """Momentum correction prefactor for the second half-kick.

        For Langevin (legacy):  ``c1 = dt/2 - dt^2*friction/8``
        (stored as ``dyn.c1`` by ASE).  For all other VV-family
        integrators: ``dt/2``.
        """
        if hasattr(dyn, "c1"):
            return dyn.c1             # Langevin BAOAB half-step coefficient
        return 0.5 * dyn.dt          # Standard Velocity Verlet
    
    @staticmethod
    def _cell_correction_factor(dyn, atoms):
        """Cell momentum correction prefactor for NPT barostats.

        Works with barostats that use ``dyn.eta`` (strain-rate tensor),
        e.g. MelchionnaNPT.

        Returns coefficient ``k`` such that ``delta_eta = k * delta_stress``.

        LangevinBAOAB is handled separately by
        :meth:`_correct_cell_momentum` because it stores cell momentum
        in ``p_eps`` (scalar or 3x3), not ``eta``.
        """
        dyn_name = type(dyn).__name__

        if dyn_name == "NPT" or dyn_name == "MelchionnaNPT":
            if not hasattr(dyn, "pfactor_given") or dyn.pfactor_given is None:
                return 0.0
            return -2.0 * dyn.dt / dyn.pfactor_given

        elif dyn_name == "NPTBerendsen":
            if not getattr(dyn, "_berendsen_warned", False):
                logger.warning(
                    "NPTBerendsen gives weak coupling to an external "
                    "pressure bath rather than a true barostat -- no cell "
                    "correction applied.  Beware!"
                )
                dyn._berendsen_warned = True
            return 0.0

        else:
            return 0.0

    @staticmethod
    def _correct_cell_momentum(dyn, atoms, dStress):
        """Apply predict-correct cell-momentum fix for NPT dynamics.

        Parameters
        ----------
        dyn : object
            ASE dynamics object.
        atoms : Atoms
            ASE Atoms.
        dStress : ndarray, shape ``(6,)``
            ``stress_new - stress_old`` in Voigt form (eV/Angstrom^3).

        Notes
        -----
        **LangevinBAOAB** -- The BAOAB B-step updates the cell momentum
        ``p_eps`` by ``0.5 * dt * force_eps``.  During the frozen predict
        phase, ``force_eps`` was computed from stale stress.  The
        dominant correction from the stress change is:

            ``delta_force_eps = -V * delta(tr sigma)``  (hydrostatic)
            ``delta_force_eps = -V * delta_sigma``       (anisotropic)

        so  ``delta_p_eps = 0.5 * dt * delta_force_eps``.

        **MelchionnaNPT / NPT** -- Uses the existing ``eta`` correction
        with ``k = -2 * dt / pfactor``.
        """
        dyn_name = type(dyn).__name__

        if dyn_name == "LangevinBAOAB" and hasattr(dyn, "p_eps"):
            # force_eps depends on stress: delta_force_eps = -V * delta_stress
            V = atoms.get_volume()
            k = -0.5 * dyn.dt * V      # coefficient for B-half step

            if getattr(dyn, "hydrostatic", False):
                # p_eps is scalar; depends on trace of stress
                dp_eps = k * (dStress[0] + dStress[1] + dStress[2])
            else:
                # p_eps is (3,3); convert Voigt -> symmetric 3x3
                dp_eps = k * np.array([
                    [dStress[0], dStress[5], dStress[4]],
                    [dStress[5], dStress[1], dStress[3]],
                    [dStress[4], dStress[3], dStress[2]],
                ])
            dyn.p_eps = dyn.p_eps + dp_eps

        elif hasattr(dyn, "eta"):
            # MelchionnaNPT / NPT -- existing code path
            k_cell = BatchedMD._cell_correction_factor(dyn, atoms)
            if k_cell == 0.0:
                return
            dEta_v = dStress * k_cell

            if hasattr(dyn, "_maketriangular"):
                dEta = dyn._maketriangular(dEta_v)
            elif np.ndim(dyn.eta) == 2:
                # Generic Voigt -> symmetric 3x3
                dEta = np.array([
                    [dEta_v[0], dEta_v[5], dEta_v[4]],
                    [dEta_v[5], dEta_v[1], dEta_v[3]],
                    [dEta_v[4], dEta_v[3], dEta_v[2]],
                ])
            else:
                dEta = dEta_v

            # Direct attribute access -- avoids side-effects from
            # set_strain_rate() which re-initialises h_past/eta_past.
            dyn.eta = dyn.eta + dEta

    def _step(self):
        """Single global MD step -- predict, batch eval, correct."""
        # ── 1. Save forces at current (pre-step) positions ────────────
        for i, c in enumerate(self._caches):
            self._F_old[i] = c.results["forces"]

            # Save old stress if available (for NPT / variable-cell dynamics)
            # Extracted strictly in its raw Voigt (6,) form
            if "stress" in c.results and c.results["stress"] is not None:
                self._Stress_old[i] = c.results["stress"]

        # ── 2. Freeze calculators -> all get_forces() return F_old ─────
        for c in self._caches:
            c._frozen = True

        # ── 3. Step each replica (single-threaded, sequential) ────────
        #    Positions are correct; momenta use stale forces for the
        #    second half-kick -- will be fixed in step 6.
        for dyn in self.dynamics:
            dyn.step()

        # ── 4. Unfreeze ───────────────────────────────────────────────
        for c in self._caches:
            c._frozen = False

        # ── 5. Batch evaluate at new positions (1 GPU call) ──────────
        results = self._batch_eval()
        self._inject_batch(results)

        # ── 6. Correct momenta: delta_p = correction x (F_new - F_old) ────────────
        c = self._corr_c
        for i in range(self.n_replicas):
            dF = results["forces"][i] - self._F_old[i]
            self.atoms_list[i].set_momenta(
                self.atoms_list[i].get_momenta() + c * dF
            )

            # ── 7. Correct cell momenta (barostat) if NPT ────────────
            if "stress" not in results:
                continue
            dyn = self.dynamics[i]
            dStress = results["stress"][i] - self._Stress_old[i]
            self._correct_cell_momentum(dyn, self.atoms_list[i], dStress)

    # ── Observer helpers ──────────────────────────────────────────────

    def attach(self, function, interval=1, *args, **kwargs):
        """Attach a **global** observer called once per global step.

        Use this for ensemble-level writers (e.g. ``TrajectoryWriter``)
        that read all replicas in a single call.  The function is called
        every *interval* global MD steps.  For per-replica observers,
        use :meth:`attach_replica`.
        """
        if not hasattr(self, '_global_observers'):
            self._global_observers = []
        self._global_observers.append((function, interval, args, kwargs))

    def attach_replica(self, idx, function, interval=1, *args, **kwargs):
        """Attach an observer to a **single** replica."""
        self.dynamics[idx].attach(function, interval, *args, **kwargs)

    # ── Run ───────────────────────────────────────────────────────────

    def run(self, steps, log_interval=100):
        """Run all replicas for *steps* MD steps.

        Single-threaded loop -- no GIL overhead, no threading.
        One batched GPU forward pass per step.

        Parameters
        ----------
        steps : int
            Number of MD steps to run.
        log_interval : int
            Print LAMMPS-style ensemble stats every this many steps.
            ``0`` or ``None`` disables logging.
        """
        logger.info(
            f"BatchedMD: running {steps} steps "
            f"({self.n_replicas} replicas, predict-correct batching)"
        )
        for _ in range(steps):
            self._step()
            self.nsteps += 1

            # Fire per-replica ASE observers
            for dyn in self.dynamics:
                dyn.nsteps += 1
                if hasattr(dyn, "call_observers"):
                    dyn.call_observers()
                else:
                    for fn, interval, args, kwargs in dyn.observers:
                        if interval > 0 and dyn.nsteps % interval == 0:
                            fn(*args, **kwargs)

            # Fire global observers (ensemble-level, once per step)
            for fn, interval, args, kwargs in getattr(self, '_global_observers', []):
                if interval > 0 and self.nsteps % interval == 0:
                    fn(*args, **kwargs)

            if log_interval and self.nsteps % log_interval == 0:
                self._log_summary()

        logger.info(f"BatchedMD: completed {steps} steps.")

    # ── Convenience properties ────────────────────────────────────────

    def get_temperatures(self):
        """Instantaneous temperature (K) for each replica, shape ``(N,)``."""
        temps = np.empty(self.n_replicas)
        for i, atoms in enumerate(self.atoms_list):
            ekin = atoms.get_kinetic_energy()
            n_dof = 3 * len(atoms)
            temps[i] = (2.0 * ekin / (n_dof * ase_units.kB)
                        if ekin > 0 else 0.0)
        return temps

    def get_potential_energies(self):
        """Potential energy (eV) for each replica, shape ``(N,)``."""
        return np.array([a.get_potential_energy() for a in self.atoms_list])

    def get_kinetic_energies(self):
        """Kinetic energy (eV) for each replica, shape ``(N,)``."""
        return np.array([a.get_kinetic_energy() for a in self.atoms_list])

    def get_max_forces(self):
        """Max force magnitude (eV/Angstrom) per replica, shape ``(N,)``."""
        max_f = np.empty(self.n_replicas)
        for i, atoms in enumerate(self.atoms_list):
            f = atoms.get_forces()
            max_f[i] = np.sqrt((f ** 2).sum(axis=1)).max()
        return max_f

    # keep old name as alias
    get_energies = get_potential_energies

    # ── LAMMPS-style ensemble logger ──────────────────────────────────

    def _log_summary(self):
        """Print a LAMMPS thermo-style line with ensemble averages.

        Columns: Step, <E_pot>, <E_kin>, <E_tot>, <Temp>, <|F|_max>.
        When stress is available (NPT), also prints <P> (GPa) and
        <V> (Angstrom^3).  Potential energy and max force are reported
        in **model units** (before eV conversion) for easier comparison
        with mlcg native.  Temperature is always in Kelvin.
        """
        to_eV = getattr(self._calc, '_to_eV', 1.0)

        e_pot_eV = self.get_potential_energies()
        e_kin_eV = self.get_kinetic_energies()
        temps = self.get_temperatures()
        max_f_eV = self.get_max_forces()

        # Convert to model units for display
        e_pot = e_pot_eV / to_eV
        e_kin = e_kin_eV / to_eV
        e_tot = e_pot + e_kin
        max_f = max_f_eV / to_eV

        # Pressure (GPa), volume and cell lengths when stress is available.
        EV_A3_TO_GPA = 160.21766208
        has_stress = any("stress" in c.results for c in self._caches)
        if has_stress:
            pressures = np.empty(self.n_replicas)
            volumes = np.empty(self.n_replicas)
            cell_lengths = np.empty((self.n_replicas, 3))
            for i, atoms in enumerate(self.atoms_list):
                try:
                    s = atoms.get_stress()  # Voigt 6-vector
                    pressures[i] = (
                        -(s[0] + s[1] + s[2]) / 3.0 * EV_A3_TO_GPA
                    )
                except Exception:
                    pressures[i] = 0.0
                volumes[i] = atoms.get_volume()
                cell = np.asarray(atoms.get_cell())
                for j in range(3):
                    cell_lengths[i, j] = np.linalg.norm(cell[j])

        # Header every 20 blocks
        if not hasattr(self, '_log_count'):
            self._log_count = 0
        if self._log_count % 20 == 0:
            units_label = getattr(
                self._calc, '_energy_units', 'eV'
            )
            hdr1 = (
                f"{'Step':>8s}  {'<E_pot>':>12s}  {'<E_kin>':>12s}  "
                f"{'<E_tot>':>12s}  {'<Temp>':>10s}  "
                f"{'<|F|_max>':>12s}"
            )
            hdr2 = (
                f"{'':>8s}  {'(' + units_label + ')':>12s}  "
                f"{'(' + units_label + ')':>12s}  "
                f"{'(' + units_label + ')':>12s}  {'(K)':>10s}  "
                f"{'(' + units_label + '/A)':>12s}"
            )
            if has_stress:
                hdr1 += (
                    f"  {'<P>':>10s}  {'<V>':>12s}  "
                    f"{'<Lx>':>8s}  {'<Ly>':>8s}  {'<Lz>':>8s}"
                )
                hdr2 += (
                    f"  {'(GPa)':>10s}  {'(A^3)':>12s}  "
                    f"{'(A)':>8s}  {'(A)':>8s}  {'(A)':>8s}"
                )
            logger.info(hdr1)
            logger.info(hdr2)
        self._log_count += 1

        line = (
            f"{self.nsteps:>8d}  "
            f"{e_pot.mean():>12.4f}  {e_kin.mean():>12.4f}  "
            f"{e_tot.mean():>12.4f}  "
            f"{temps.mean():>10.2f}  {max_f.mean():>12.4f}"
        )
        if has_stress:
            line += (
                f"  {pressures.mean():>10.6f}  "
                f"{volumes.mean():>12.2f}  "
                f"{cell_lengths[:, 0].mean():>8.3f}  "
                f"{cell_lengths[:, 1].mean():>8.3f}  "
                f"{cell_lengths[:, 2].mean():>8.3f}"
            )
        logger.info(line)


# ══════════════════════════════════════════════════════════════════════
# Trajectory writer -- mlcg-compatible streaming .npy format
# ══════════════════════════════════════════════════════════════════════


class TrajectoryWriter:
    """Streaming trajectory writer in mlcg's native ``.npy`` format.

    Mirrors the output layout of ``mlcg.simulation._Simulation.write()``:
    one file per chunk, positions stored as
    ``{filename}_coords_{NNNN}.npy`` with shape
    ``(n_replicas, chunk_frames, n_atoms, 3)`` (float32).

    Optionally also writes ``_potential_`` and ``_kineticenergy_``
    files to match mlcg's ``save_energies=True`` output.

    Designed for **production** runs (millions of steps): frames are
    flushed to disk every *export_interval* frames, keeping memory
    usage constant.

    Parameters
    ----------
    filename : str
        Base path (no extension).  Files are written as
        ``{filename}_coords_0000.npy``, ``_0001.npy``, etc.
    runner : BatchedMD
        The batched MD runner -- used to read positions, energies,
        and temperatures from all replicas.
    export_interval : int
        Number of **saved** frames to buffer before flushing to a
        ``.npy`` file.  Each flush writes one chunk.  Default: 10000.
    save_energies : bool
        Also write ``_potential_`` and ``_kineticenergy_`` files.
    energy_units : str
        Label stored in the metadata file.  Default: inferred from
        the calculator.

    Example
    -------
    ::

        writer = TrajectoryWriter("production", runner, export_interval=5000)
        runner.attach(writer.save, interval=10)  # save every 10 steps
        runner.run(1_000_000, log_interval=10000)
        writer.close()   # flush remaining frames
    """

    def __init__(self, filename, runner, export_interval=10000,
                 save_energies=True, energy_units=None):
        self._filename = filename
        self._runner = runner
        self._export_interval = export_interval
        self._save_energies = save_energies
        self._chunk_idx = 0
        self._n_saved = 0   # total frames saved across all chunks

        n_rep = runner.n_replicas
        n_atoms = runner._n_atoms

        # Pre-allocate buffers: (n_frames, n_replicas, ...)
        self._coords_buf = np.empty(
            (export_interval, n_rep, n_atoms, 3), dtype=np.float32
        )
        self._buf_pos = 0  # next write position in buffer

        if save_energies:
            self._pot_buf = np.empty(
                (export_interval, n_rep), dtype=np.float32
            )
            self._kin_buf = np.empty(
                (export_interval, n_rep), dtype=np.float32
            )
        else:
            self._pot_buf = None
            self._kin_buf = None

        # Energy unit label for metadata
        if energy_units is None and hasattr(runner._calc, '_energy_units'):
            energy_units = runner._calc._energy_units
        self._energy_units = energy_units or "eV"

        # Write metadata file once
        to_eV = getattr(runner._calc, '_to_eV', 1.0)
        meta = {
            'n_replicas': n_rep,
            'n_atoms': n_atoms,
            'export_interval': export_interval,
            'save_energies': save_energies,
            'energy_units': self._energy_units,
            'to_eV': float(to_eV),
            'coord_units': 'angstrom',
        }
        np.savez(f"{filename}_metadata.npz", **meta)

        logger.info(
            f"TrajectoryWriter: {filename}_coords_NNNN.npy, "
            f"chunk={export_interval} frames, "
            f"energies={'yes' if save_energies else 'no'}"
        )

    def save(self):
        """Buffer one frame from all replicas.  Called as an observer."""
        runner = self._runner
        idx = self._buf_pos

        # Positions (Å) -- copy from each replica's Atoms
        for i, atoms in enumerate(runner.atoms_list):
            self._coords_buf[idx, i] = atoms.get_positions()

        # Energies (model units)
        if self._save_energies:
            to_eV = getattr(runner._calc, '_to_eV', 1.0)
            self._pot_buf[idx] = runner.get_potential_energies() / to_eV
            self._kin_buf[idx] = runner.get_kinetic_energies() / to_eV

        self._buf_pos += 1
        self._n_saved += 1

        if self._buf_pos >= self._export_interval:
            self._flush()

    def _flush(self):
        """Write current buffer to disk and reset."""
        if self._buf_pos == 0:
            return

        key = f"{self._chunk_idx:04d}"
        n = self._buf_pos  # may be < export_interval for final chunk

        # Swap axes: (n_frames, n_rep, ...) -> (n_rep, n_frames, ...)
        # to match mlcg native format
        coords = self._coords_buf[:n].transpose(1, 0, 2, 3).copy()
        np.save(f"{self._filename}_coords_{key}.npy", coords)

        if self._save_energies:
            pot = self._pot_buf[:n].transpose(1, 0).copy()
            np.save(f"{self._filename}_potential_{key}.npy", pot)
            kin = self._kin_buf[:n].transpose(1, 0).copy()
            np.save(f"{self._filename}_kineticenergy_{key}.npy", kin)

        logger.info(
            f"TrajectoryWriter: wrote chunk {key} "
            f"({n} frames, {self._n_saved} total)"
        )
        self._chunk_idx += 1
        self._buf_pos = 0

    def close(self):
        """Flush any remaining buffered frames."""
        self._flush()
        logger.info(
            f"TrajectoryWriter: saved {self._n_saved} frames total "
            f"in {self._chunk_idx} chunks to {self._filename}_coords_*.npy"
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        if self._buf_pos > 0:
            self._flush()
