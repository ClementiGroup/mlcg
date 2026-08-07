import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Type, Union

import numpy as np
import torch
from torch_geometric.data.collate import collate
from torch_geometric.utils import scatter

from ..data.atomic_data import AtomicData
from ..data._keys import FORCE_KEY


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


def _validate_configurations_and_free_atoms(
    configurations: List[AtomicData],
    free_atoms: Optional[List[Optional[Sequence[int]]]],
) -> None:
    r"""Input validation for :py:func:`minimize_energy`."""
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

    if free_atoms is not None:
        if len(free_atoms) != len(configurations):
            raise ValueError(
                f"len(free_atoms)={len(free_atoms)} must equal "
                f"len(configurations)={len(configurations)}."
            )
        for i, (data, idx) in enumerate(zip(configurations, free_atoms)):
            if idx is None:
                continue
            idx_arr = np.asarray(idx, dtype=int)
            n_atoms_i = data.pos.shape[0]
            if idx_arr.size and (
                idx_arr.min() < 0 or idx_arr.max() >= n_atoms_i
            ):
                raise ValueError(
                    f"free_atoms[{i}] contains an out-of-range index for a "
                    f"structure with {n_atoms_i} atoms: {idx_arr.tolist()}."
                )


def minimize_energy(
    model: torch.nn.Module,
    configurations: List[AtomicData],
    free_atoms: Optional[List[Optional[Sequence[int]]]] = None,
    fmax: float = 0.05,
    steps: int = 500,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
    optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.LBFGS,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
) -> List[AtomicData]:
    r"""Relax all of ``configurations`` to nearby local minima of ``model`` at
    once, optionally restricting relaxation to a subset of atoms.

    The relaxation is batched: every configuration is collated into a single
    :py:class:`AtomicData` so that all structures share **one model forward
    call per step**. Each configuration nevertheless gets its own, independent
    ``optimizer_cls`` instance -- and therefore its own optimizer state (e.g.
    its own L-BFGS history). That separation matters because the structures
    are physically independent (each one's energy depends only on its own
    coordinates), so the true Hessian is block-diagonal; a single optimizer
    state shared across all of them would let curvature information from one
    structure pollute the search direction chosen for the others, and would
    need substantially more steps to reach the same ``fmax``.

    The model forces are used directly as the negative gradient of the energy
    with respect to positions, so no second-order differentiation through the
    model is required.

    Parameters
    ----------
    model:
        Trained mlcg model producing forces, i.e. wrapped so that its output
        contains :py:data:`~mlcg.data._keys.FORCE_KEY` (via
        :py:class:`mlcg.nn.gradients.GradientsOut`, or
        :py:class:`mlcg.nn.gradients.SumOut` over ``GradientsOut``-wrapped
        terms). Note that this model is mutated in place: it is switched to
        evaluation mode, moved to ``device``/``dtype``, and has
        ``requires_grad`` disabled on all of its parameters, mirroring
        :py:meth:`mlcg.simulation.base._Simulation._attach_model`.
    configurations:
        List of single, un-collated :py:class:`AtomicData` structures to
        relax. They are not modified.
    free_atoms:
        Optional list, parallel to ``configurations``, of atom index
        sequences that are allowed to move during minimization; every other
        atom in that structure is held fixed at its input position. Use
        ``None`` for a given entry (or pass ``free_atoms=None`` altogether,
        the default) to leave that configuration fully unconstrained. Fixed
        atoms are held in place by zeroing their gradient, so they never
        move.
    fmax:
        Convergence threshold on the largest per-atom force magnitude (atoms
        held fixed are excluded, since they are expected to carry a nonzero
        force precisely because they are being constrained). A structure is
        considered converged once its own value drops to ``fmax`` and is then
        left alone while the rest of the batch keeps going. This is compared
        directly against the model's own force units (e.g. kcal/mol/Angstrom
        for typical CG models), unlike ASE optimizers, whose default ``fmax``
        assumes eV/Angstrom -- pick a value appropriate for the model in use.
    steps:
        Maximum number of optimizer steps. A warning is issued if the budget
        runs out before every structure has converged.
    device:
        Device used to run the model.
    dtype:
        Floating point precision used to run the model. Positions are cast to
        it for the relaxation and cast back to each input's own dtype on the
        way out.
    optimizer_cls:
        Any :py:class:`torch.optim.Optimizer` subclass. Defaults to
        :py:class:`torch.optim.LBFGS`, which is well-suited to energy
        minimization. First-order optimizers such as
        :py:class:`torch.optim.SGD` or :py:class:`torch.optim.Adam` are also
        supported; pass optimizer-specific hyperparameters via
        ``optimizer_kwargs``. Each ``.step()`` call is given the gradient
        from the one forward pass already taken this iteration, via a closure
        that does *not* re-run the model (re-running it per structure would
        defeat the point of batching); for :py:class:`torch.optim.LBFGS` this
        means ``max_iter`` must stay 1 (its default here), since a larger
        value would spend extra sub-iterations applying quasi-Newton updates
        from that same, now-stale gradient instead of a freshly evaluated
        one. The same caveat applies to any other optimizer whose ``step()``
        may invoke its closure more than once.
    optimizer_kwargs:
        Additional keyword arguments forwarded to ``optimizer_cls``.

    Returns
    -------
    List[AtomicData]:
        New list of :py:class:`AtomicData` instances, one per input
        configuration, identical to the inputs except for relaxed positions
        (cast back to each input's own dtype/device).
    """
    _validate_configurations_and_free_atoms(configurations, free_atoms)

    model = model.eval().to(device=device, dtype=dtype)
    for param in model.parameters():
        param.requires_grad = False

    n_structures = len(configurations)
    batch, _, _ = collate(
        AtomicData, data_list=configurations, increment=True, add_batch=True
    )
    batch = batch.to(device)

    batch_index = batch.batch
    # ptr lives on `device`; pulling it into python once avoids a GPU->CPU
    # sync per structure per step when it is used as a slice bound below.
    # Also save it before the loop, since model calls may rebind batch
    # attributes.
    ptr = batch.ptr.tolist()
    n_atoms_total = batch.pos.shape[0]

    fixed_mask = torch.zeros(n_atoms_total, dtype=torch.bool, device=device)
    if free_atoms is not None:
        for i, idx in enumerate(free_atoms):
            if idx is None:
                continue
            # Every atom in this structure starts fixed; only the listed
            # indices (if any) are then let free again.
            fixed_mask[ptr[i] : ptr[i + 1]] = True
            if len(idx):
                idx_t = (
                    torch.as_tensor(idx, dtype=torch.long, device=device)
                    + ptr[i]
                )
                fixed_mask[idx_t] = False
    # fixed_mask is static across steps; checking it every iteration would
    # cost an extra GPU->CPU sync per step for no reason.
    has_fixed = bool(fixed_mask.any())

    # Positions are the only optimizable parameters; model weights stay
    # frozen. GradientsOut internally calls torch.autograd.grad(energy,
    # batch.pos), which does NOT set .grad; we assign each structure's slice
    # of .grad manually from the returned forces so that any torch.optim
    # optimizer can be used without requiring a second backward pass through
    # the model. One independent leaf tensor per structure gives each one its
    # own optimizer state (see the docstring) while the forward pass stays
    # batched.
    sizes = [ptr[i + 1] - ptr[i] for i in range(n_structures)]
    param_list = [
        p.clone().requires_grad_(True)
        for p in batch.pos.detach().to(dtype).split(sizes)
    ]

    optimizer_kwargs = dict(optimizer_kwargs or {})
    # `isinstance(..., type)` first: optimizer_cls may legitimately be a
    # partial or other factory rather than a class, and issubclass() would
    # raise on those.
    if isinstance(optimizer_cls, type) and issubclass(
        optimizer_cls, torch.optim.LBFGS
    ):
        if optimizer_kwargs.get("max_iter", 1) != 1:
            raise ValueError(
                "minimize_energy's closure does not re-run the model (the "
                "gradient for this step was already computed by the one "
                "shared forward pass), so torch.optim.LBFGS's max_iter must "
                "stay 1 -- see the optimizer_cls docstring entry."
            )
        optimizer_kwargs["max_iter"] = 1
    optimizers = [optimizer_cls([p], **optimizer_kwargs) for p in param_list]

    # Cached once: torch.optim's step() API requires a closure, but the
    # gradient is already assigned manually from the shared forward pass
    # below, so the closure has nothing to compute. Reusing one tensor avoids
    # a fresh host->device scalar transfer on every one of the n_structures
    # closure calls per step.
    _dummy_loss = torch.tensor(0.0, dtype=dtype, device=device)

    def dummy_closure() -> torch.Tensor:
        return _dummy_loss

    # Structures that reach fmax are skipped for the rest of the loop: a
    # skipped structure stops moving, so its force -- and therefore its
    # converged status -- cannot change afterwards.
    done = np.zeros(n_structures, dtype=bool)

    for _ in range(steps):
        batch.pos = torch.cat(param_list, dim=0)
        batch.out = {}
        out = model(batch)
        if FORCE_KEY not in out.out:
            raise KeyError(
                f"Model output does not contain '{FORCE_KEY}'. The model "
                "passed to minimize_energy must be wrapped with "
                "mlcg.nn.gradients.GradientsOut (or SumOut over "
                "GradientsOut-wrapped terms) with forces among its targets "
                "-- an energy-only model cannot be used for minimization."
            )
        # Fresh tensor every iteration (the unary minus allocates), so it is
        # never aliased to the model output and is safe to mask in place.
        g = -out.out[FORCE_KEY].detach()
        if has_fixed:
            g.masked_fill_(fixed_mask.unsqueeze(1), 0.0)

        per_structure_fmax = scatter(
            g.norm(dim=1),
            batch_index,
            dim=0,
            dim_size=n_structures,
            reduce="max",
        )
        # One sync for the whole per-structure array, reused for both the
        # overall break check and the per-structure skip below, instead of a
        # separate .all() sync plus a GPU-side compare per structure.
        done |= (per_structure_fmax <= fmax).cpu().numpy()
        if done.all():
            break

        for i, opt in enumerate(optimizers):
            if done[i]:
                continue
            # g is rebuilt every iteration, so this slice is never aliased or
            # mutated across iterations -- no clone needed.
            param_list[i].grad = g[ptr[i] : ptr[i + 1]]
            opt.step(dummy_closure)
    else:
        n_unconverged = int((~done).sum())
        if n_unconverged:
            warnings.warn(
                f"minimize_energy: {n_unconverged} of {n_structures} "
                f"structure(s) did not reach fmax={fmax} within steps="
                f"{steps}; returning their last positions."
            )

    final_pos = torch.cat(param_list, dim=0).detach().cpu()
    minimized: List[AtomicData] = []
    for i, data in enumerate(configurations):
        new_data = deepcopy(data)
        new_data.pos = final_pos[ptr[i] : ptr[i + 1]].to(
            dtype=data.pos.dtype, device=data.pos.device
        )
        new_data.out = {}
        minimized.append(new_data)

    return minimized
