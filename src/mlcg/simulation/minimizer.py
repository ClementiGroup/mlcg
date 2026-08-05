from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Type, Union

import numpy as np
import torch
from torch_geometric.data.collate import collate
from torch_geometric.utils import scatter

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms
from ase.optimize import FIRE
from ase.optimize.optimize import Optimizer

from ..data.atomic_data import AtomicData
from ..data._keys import (
    ATOM_TYPE_KEY,
    MASS_KEY,
    CELL_KEY,
    PBC_KEY,
    ENERGY_KEY,
    FORCE_KEY,
)


def _num_structures(data: AtomicData) -> int:
    r"""Number of individual structures represented by an AtomicData
    instance. ``n_atoms`` is only populated by
    :py:meth:`AtomicData.from_points`, not by the raw ``AtomicData(...)``
    constructor used throughout the codebase (e.g. in
    :py:func:`mlcg.mol_utils._ASE_prior_model`), so batching is instead
    detected the same way :py:class:`~mlcg.simulation.base._Simulation` does:
    via the ``batch`` index added by collation.
    """
    if "batch" in data and data.batch is not None:
        return int(data.batch.max().item()) + 1
    return 1


def _validate_configurations_and_fixed_atoms(
    configurations: List[AtomicData],
    fixed_atoms: Optional[List[Optional[Sequence[int]]]],
) -> None:
    r"""Shared input validation for :py:func:`minimize_energy` and
    :py:func:`minimize_energy_ase`."""
    if len(configurations) == 0:
        raise ValueError(
            "configurations must be a non-empty list of AtomicData."
        )

    for i, data in enumerate(configurations):
        n_structures_i = _num_structures(data)
        if n_structures_i != 1:
            raise ValueError(
                f"configurations[{i}] is batched (spans {n_structures_i} "
                "structures); this utility expects one un-collated "
                "AtomicData structure per list element."
            )

    if fixed_atoms is not None:
        if len(fixed_atoms) != len(configurations):
            raise ValueError(
                f"len(fixed_atoms)={len(fixed_atoms)} must equal "
                f"len(configurations)={len(configurations)}."
            )
        for i, (data, idx) in enumerate(zip(configurations, fixed_atoms)):
            if idx is None:
                continue
            idx_arr = np.asarray(idx, dtype=int)
            n_atoms_i = data.pos.shape[0]
            if idx_arr.size and (
                idx_arr.min() < 0 or idx_arr.max() >= n_atoms_i
            ):
                raise ValueError(
                    f"fixed_atoms[{i}] contains an out-of-range index for a "
                    f"structure with {n_atoms_i} atoms: {idx_arr.tolist()}."
                )


class MLCGCalculator(Calculator):
    r"""ASE calculator wrapper around an mlcg model.

    On every :py:meth:`calculate` call, a fresh single-structure
    :py:class:`~mlcg.data.atomic_data.AtomicData` instance is built from the
    current :py:class:`ase.Atoms` positions and forwarded through the model to
    obtain the energy and forces.

    Parameters
    ----------
    model:
        Trained mlcg model. Must be wrapped so that its output contains both
        :py:data:`~mlcg.data._keys.ENERGY_KEY` and
        :py:data:`~mlcg.data._keys.FORCE_KEY` (e.g. via
        :py:class:`mlcg.nn.gradients.GradientsOut` or
        :py:class:`mlcg.nn.gradients.SumOut`).
    data_template:
        A single, un-collated :py:class:`AtomicData` instance describing the
        structure. ``atom_types``, ``masses``, ``cell``, ``pbc`` and
        ``neighbor_list`` are captured from it once and reused, unmodified,
        on every subsequent call; only positions change between calls.

        The ``neighbor_list`` is reused verbatim rather than cleared: bonded
        topology entries (e.g. from harmonic priors) do a direct dictionary
        lookup with no fallback and would raise if missing, while
        cutoff-based entries left absent from the dictionary (the standard
        convention) are automatically rebuilt from the current positions by
        the model itself.
    device:
        Device used to run the model.
    dtype:
        Floating point precision used to run the model.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        model: torch.nn.Module,
        data_template: AtomicData,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if _num_structures(data_template) != 1:
            raise ValueError(
                "MLCGCalculator expects a single, un-collated AtomicData "
                f"structure, but got a template spanning "
                f"{_num_structures(data_template)} structures."
            )

        self.model = model
        self.device = torch.device(device)
        self.dtype = dtype

        self.atom_types = data_template.atom_types.detach().clone()
        self.masses = (
            data_template.masses.detach().clone()
            if MASS_KEY in data_template and data_template.masses is not None
            else None
        )
        self.cell = (
            data_template.cell.detach().clone()
            if CELL_KEY in data_template and data_template.cell is not None
            else None
        )
        self.pbc = (
            data_template.pbc.detach().clone()
            if PBC_KEY in data_template and data_template.pbc is not None
            else None
        )
        self.template_neighbor_list = deepcopy(data_template.neighbor_list)

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: Sequence[str] = ("energy",),
        system_changes: Sequence[str] = all_changes,
    ):
        super().calculate(atoms, properties, system_changes)

        pos = torch.tensor(
            self.atoms.get_positions(), dtype=self.dtype, device=self.device
        )
        data = AtomicData.from_points(
            pos=pos,
            atom_types=self.atom_types,
            masses=self.masses,
            cell=self.cell,
            pbc=self.pbc,
            neighborlist=deepcopy(self.template_neighbor_list),
        )
        batch, _, _ = collate(
            AtomicData, data_list=[data], increment=True, add_batch=True
        )
        batch = batch.to(self.device)
        batch = self.model(batch)

        if FORCE_KEY not in batch.out:
            raise KeyError(
                f"Model output does not contain '{FORCE_KEY}'. The model "
                "passed to MLCGCalculator/minimize_energy must be wrapped "
                "with mlcg.nn.gradients.GradientsOut (or SumOut over "
                "GradientsOut-wrapped terms) with forces among its targets "
                "-- an energy-only model cannot be used for minimization."
            )

        self.results["energy"] = float(
            batch.out[ENERGY_KEY].detach().cpu().item()
        )
        self.results["forces"] = (
            batch.out[FORCE_KEY].detach().cpu().numpy().astype(np.float64)
        )


def minimize_energy_ase(
    model: torch.nn.Module,
    configurations: List[AtomicData],
    fixed_atoms: Optional[List[Optional[Sequence[int]]]] = None,
    optimizer_cls: Type[Optimizer] = FIRE,
    fmax: float = 0.05,
    steps: int = 500,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
) -> List[AtomicData]:
    r"""Relax each of ``configurations`` to a nearby local minimum of
    ``model``, optionally holding a subset of atoms fixed.

    Each configuration is minimized independently via an ASE optimizer
    (:py:class:`ase.optimize.FIRE` by default) acting on an
    :py:class:`MLCGCalculator`-backed :py:class:`ase.Atoms` instance. Atoms
    marked as fixed for a given configuration are held in place via
    :py:class:`ase.constraints.FixAtoms`.

    Parameters
    ----------
    model:
        Trained mlcg model producing both energies and forces (see
        :py:class:`MLCGCalculator`). Note that this model is mutated in
        place: it is switched to evaluation mode, moved to ``device``/``dtype``,
        and has ``requires_grad`` disabled on all of its parameters, mirroring
        :py:meth:`mlcg.simulation.base._Simulation._attach_model`.
    configurations:
        List of single, un-collated :py:class:`AtomicData` structures to
        relax.
    fixed_atoms:
        Optional list, parallel to ``configurations``, of atom index
        sequences that should remain fixed during minimization. Use ``None``
        for a given entry (or pass ``fixed_atoms=None`` altogether) to leave
        that configuration fully unconstrained.
    optimizer_cls:
        ASE :py:class:`~ase.optimize.optimize.Optimizer` subclass used to
        perform the minimization.
    fmax:
        Convergence threshold on the maximum force component. This is
        compared directly against the model's own force units (e.g.
        kcal/mol/Angstrom for typical CG models), unlike ASE's own optimizers
        whose default ``fmax`` assumes eV/Angstrom -- pick a value
        appropriate for the model being used.
    steps:
        Maximum number of optimizer steps per configuration.
    device:
        Device used to run the model.
    dtype:
        Floating point precision used to run the model.
    optimizer_kwargs:
        Additional keyword arguments forwarded to ``optimizer_cls``.

    Returns
    -------
    List[AtomicData]:
        New list of :py:class:`AtomicData` instances, one per input
        configuration, identical to the inputs except for relaxed positions
        (cast back to each input's own dtype/device).
    """
    _validate_configurations_and_fixed_atoms(configurations, fixed_atoms)

    model = model.eval().to(device=device, dtype=dtype)
    for param in model.parameters():
        param.requires_grad = False

    optimizer_kwargs = dict(optimizer_kwargs or {})
    optimizer_kwargs.setdefault("logfile", None)

    minimized: List[AtomicData] = []
    for i, data in enumerate(configurations):
        calc = MLCGCalculator(model, data, device=device, dtype=dtype)

        numbers = data[ATOM_TYPE_KEY].detach().cpu().numpy()
        positions = data.pos.detach().cpu().numpy().astype(np.float64)
        masses = (
            data.masses.detach().cpu().numpy()
            if MASS_KEY in data and data.masses is not None
            else None
        )
        cell = (
            data.cell.detach().cpu().numpy().reshape(3, 3)
            if CELL_KEY in data and data.cell is not None
            else None
        )
        pbc = (
            data.pbc.detach().cpu().numpy().reshape(3)
            if PBC_KEY in data and data.pbc is not None
            else False
        )
        atoms = Atoms(
            numbers=numbers,
            positions=positions,
            masses=masses,
            cell=cell,
            pbc=pbc,
        )
        atoms.calc = calc

        idx = None if fixed_atoms is None else fixed_atoms[i]
        if idx is not None and len(idx) > 0:
            atoms.set_constraint(
                FixAtoms(indices=np.asarray(idx, dtype=int))
            )

        optimizer = optimizer_cls(atoms, **optimizer_kwargs)
        optimizer.run(fmax=fmax, steps=steps)

        new_data = deepcopy(data)
        new_data.pos = torch.tensor(
            atoms.get_positions(),
            dtype=data.pos.dtype,
            device=data.pos.device,
        )
        new_data.out = {}
        minimized.append(new_data)

    return minimized



def minimize_energy(
    model: torch.nn.Module,
    configurations: List[AtomicData],
    fixed_atoms: Optional[List[Optional[Sequence[int]]]] = None,
    fmax: float = 0.05,
    steps: int = 500,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
    optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.LBFGS,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
) -> List[AtomicData]:
    r"""Relax all of ``configurations`` at once via a batched gradient-based
    minimization, exploiting the model's own batching: every configuration is
    collated into a single :py:class:`AtomicData` and all structures share
    one model forward call per step, unlike :py:func:`minimize_energy_ase`
    which relaxes one structure at a time through ASE.

    Positions are treated as the optimizable parameters and a PyTorch
    optimizer (``optimizer_cls``) drives the relaxation. The model forces are
    used directly as the negative gradient of the energy with respect to
    positions, so no second-order differentiation through the model is
    required.

    Parameters
    ----------
    model, configurations, fixed_atoms, fmax, steps, device, dtype:
        See :py:func:`minimize_energy_ase`.
    optimizer_cls:
        Any :py:class:`torch.optim.Optimizer` subclass. Defaults to
        :py:class:`torch.optim.LBFGS`, which is well-suited to energy
        minimization. First-order optimizers such as
        :py:class:`torch.optim.SGD` or :py:class:`torch.optim.Adam` are also
        supported; pass optimizer-specific hyperparameters via
        ``optimizer_kwargs``.
    optimizer_kwargs:
        Additional keyword arguments forwarded to ``optimizer_cls``.

    Returns
    -------
    List[AtomicData]:
        New list of :py:class:`AtomicData` instances, one per input
        configuration, identical to the inputs except for relaxed positions
        (cast back to each input's own dtype/device).
    """
    _validate_configurations_and_fixed_atoms(configurations, fixed_atoms)

    model = model.eval().to(device=device, dtype=dtype)
    for param in model.parameters():
        param.requires_grad = False

    n_structures = len(configurations)
    batch, _, _ = collate(
        AtomicData, data_list=configurations, increment=True, add_batch=True
    )
    batch = batch.to(device)

    batch_index = batch.batch
    n_atoms_total = batch.pos.shape[0]

    fixed_mask = torch.zeros(n_atoms_total, dtype=torch.bool, device=device)
    if fixed_atoms is not None:
        for i, idx in enumerate(fixed_atoms):
            if idx is None or len(idx) == 0:
                continue
            idx_t = (
                torch.as_tensor(idx, dtype=torch.long, device=device)
                + batch.ptr[i]
            )
            fixed_mask[idx_t] = True

    # Positions are the only optimizable parameter; model weights stay frozen.
    # GradientsOut internally calls torch.autograd.grad(energy, batch.pos),
    # which does NOT set .grad; we assign pos.grad manually from the returned
    # forces so that any torch.optim optimizer can be used without requiring
    # a second backward pass through the model.
    pos = batch.pos.detach().requires_grad_(True)
    batch.pos = pos

    optimizer_kwargs = dict(optimizer_kwargs or {})
    optimizer = optimizer_cls([pos], **optimizer_kwargs)

    # Save ptr before the loop; model calls may rebind batch attributes.
    ptr = batch.ptr

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        batch.out = {}
        batch.pos = pos
        out = model(batch)
        if FORCE_KEY not in out.out:
            raise KeyError(
                f"Model output does not contain \'{FORCE_KEY}\'. The model "
                "passed to minimize_energy must be wrapped with "
                "mlcg.nn.gradients.GradientsOut (or SumOut over "
                "GradientsOut-wrapped terms) with forces among its targets "
                "-- an energy-only model cannot be used for minimization."
            )
        f = out.out[FORCE_KEY].detach()
        g = -f
        if fixed_mask.any():
            g = g.masked_fill(fixed_mask.unsqueeze(1), 0.0)
        pos.grad = g
        if ENERGY_KEY in out.out:
            return out.out[ENERGY_KEY].sum().detach()
        return torch.tensor(0.0, dtype=dtype, device=device)

    for _ in range(steps):
        optimizer.step(closure)
        if pos.grad is None:
            continue
        # pos.grad == -forces with fixed atoms already zeroed (see closure),
        # so its per-atom norm is the force magnitude used for convergence.
        # Note: for line-search optimizers (e.g. LBFGS) this reflects the last
        # closure evaluation, which may be a trial point rather than the
        # accepted step, so the convergence estimate can be marginally noisy.
        per_structure_fmax = scatter(
            pos.grad.detach().norm(dim=1),
            batch_index,
            dim=0,
            dim_size=n_structures,
            reduce="max",
        )
        if bool((per_structure_fmax <= fmax).all()):
            break

    final_pos = pos.detach().cpu()
    minimized: List[AtomicData] = []
    for i, data in enumerate(configurations):
        start, end = int(ptr[i]), int(ptr[i + 1])
        new_data = deepcopy(data)
        new_data.pos = final_pos[start:end].to(
            dtype=data.pos.dtype, device=data.pos.device
        )
        new_data.out = {}
        minimized.append(new_data)

    return minimized
