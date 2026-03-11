import torch
from typing import Optional, Sequence, Any, List, Tuple
from ..data.atomic_data import AtomicData
from ..data._keys import (
    ENERGY_KEY,
    FORCE_KEY,
    VIRIALS_KEY,
    STRESS_KEY,
    CELL_KEY,
    NEIGHBOR_LIST_KEY,
)


class SumOut(torch.nn.Module):
    r"""Property pooling wrapper for models

    Parameters
    ----------
    models:
        Dictionary of predictors models keyed by their name attribute
    targets:
        List of prediction targets that will be pooled

    Example
    -------
    To combine SchNet force predictions with prior interactions:

    .. code-block:: python

        import torch
        from mlcg.nn import (StandardSchNet, HarmonicBonds, HarmonicAngles,
                             GradientsOut, SumOut, CosineCutoff,
                             GaussianBasis)
        from mlcg.data._keys import FORCE_KEY, ENERGY_KEY

        bond_terms = GradientsOut(HarmonicBonds(bond_stats), FORCE_KEY)
        angle_terms = GradientsOut(HarmonicAngles(angle_stats), FORCE_KEY)
        cutoff = CosineCutoff()
        rbf = GaussianBasis(cutoff)
        energy_network = StandardSchNet(cutoff, rbf, [128])
        force_network = GradientsOut(energy_model, FORCE_KEY)

        models = torch.nn.ModuleDict{
                     "bonds": bond_terms,
                     "angles": angle_terms,
                     "SchNet": force_network
                 }
        full_model = SumOut(models, targets=[ENERGY_KEY, FORCE_KEY])


    """

    name: str = "SumOut"

    def __init__(
        self,
        models: torch.nn.ModuleDict,
        targets: List[str] = None,
    ):
        super(SumOut, self).__init__()
        if targets is None:
            targets = [ENERGY_KEY, FORCE_KEY]
        self.targets = targets
        self.models = models

    def forward(self, data: AtomicData) -> AtomicData:
        r"""Sums output properties from individual models into global
        property predictions

        Parameters
        ----------
        data:
            AtomicData instance whose 'out' field has been populated
            for each predictor in the model. For example:

        .. code-block::python

            AtomicData(
                out: {
                    SchNet: {
                        ENERGY_KEY: ...,
                        FORCE_KEY: ...,
                    },
                    bonds: {
                        ENERGY_KEY: ...,
                        FORCE_KEY: ...,
                    },

            ...
            )

        Returns
        -------
        data:
            AtomicData instance with updated 'out' field that now contains
            prediction target keys that map to tensors that have summed
            up the respective contributions from each predictor in the model.
            For example:

        .. code-block::python

            AtomicData(
                out: {
                    SchNet: {
                        ENERGY_KEY: ...,
                        FORCE_KEY: ...,
                    },
                    bonds: {
                        ENERGY_KEY: ...,
                        FORCE_KEY: ...,
                    },
                    ENERGY_KEY: ...,
                    FORCE_KEY: ...,
            ...
            )

        """
        for target in self.targets:
            data.out[target] = 0.00
        for name in self.models.keys():
            data = self.models[name](data)
            for target in self.targets:
                data.out[target] += data.out[name][target]
            # Aggregate virials/stress if present
            for extra_key in (VIRIALS_KEY, STRESS_KEY):
                if extra_key in data.out.get(name, {}):
                    if extra_key not in data.out:
                        data.out[extra_key] = torch.zeros_like(
                            data.out[name][extra_key]
                        )
                    data.out[extra_key] = (
                        data.out[extra_key] + data.out[name][extra_key]
                    )
        return data

    def neighbor_list(self, **kwargs):
        nl = {}
        for _, model in self.models.items():
            nl.update(**model.neighbor_list(**kwargs))
        return nl


class EnergyOut(torch.nn.Module):
    r"""Extractor for energy computed via an mlcg compatible model.

    Parameters
    ----------
    model:
        model whose target should be extyracted
    targets:
        List of prediction targets that will be extracted

    """

    name: str = "EnergyOut"

    def __init__(
        self,
        model: torch.nn.Module,
        targets: List[str] = None,
    ):
        super().__init__()
        if targets is None:
            targets = [ENERGY_KEY]
        self.targets = targets
        self.model = model
        self.name = self.model.name

    def forward(self, data: AtomicData) -> AtomicData:
        data = self.model(data)
        for target in self.targets:
            data.out[target] = data.out[self.name][target]
        return data


class GradientsOut(torch.nn.Module):
    r"""Gradient wrapper for models.

    Parameters
    ----------
    targets:
        Gradient properties to compute.  Currently ``FORCE_KEY`` is the
        primary target.
    compute_virials:
        If ``True``, compute the virial tensor (following method used by MACE / NequIP).  The virial is
        stored under ``VIRIALS_KEY``.
    compute_stress:
        If ``True`` (implies *compute_virials*), also compute
        ``stress = virial / volume`` and store under ``STRESS_KEY``.
        Requires a non-zero cell on the input data.
    Example
    -------
        To predict forces from an energy model, one would supply a model that
        predicts a scalar atom property (an energy) and specify the `FORCE_KEY`
        in the targets.
    """

    _targets = {FORCE_KEY: ENERGY_KEY}

    def __init__(
        self,
        model: torch.nn.Module,
        targets: str = FORCE_KEY,
        compute_virials: bool = False,
        compute_stress: bool = False,
    ):
        super(GradientsOut, self).__init__()
        self.model = model
        self.name = self.model.name
        self.targets = []
        if isinstance(targets, str):
            self.targets = [targets]
        elif isinstance(targets, Sequence):
            self.targets = targets
        assert any(
            [k in GradientsOut._targets for k in self.targets]
        ), f"targets={self.targets} should be any of {GradientsOut._targets}"

        self.compute_virials = compute_virials or compute_stress
        self.compute_stress = compute_stress

    def _apply_strain(self, data: AtomicData) -> Tuple[AtomicData, torch.Tensor]:
        """Apply symmetric infinitesimal strain for virial computation.

        Follows the approach used in MACE and NequIP.

        References
        ----------
        https://github.com/mir-group/nequip
        https://github.com/ACEsuit/mace
        """
        num_graphs = int(data.ptr.numel() - 1)

        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data.pos.dtype,
            device=data.pos.device,
        )
        displacement.requires_grad_(True)

        symmetric_displacement = 0.5 * (
            displacement + displacement.transpose(-1, -2)
        )

        # Strain positions
        data.pos = data.pos + torch.einsum(
            "bi,bij->bj", data.pos, symmetric_displacement[data.batch]
        )

        # Strain cell
        cell = getattr(data, CELL_KEY, None)
        if cell is not None and cell.numel() > 0:
            cell = cell.view(-1, 3, 3)
            data.cell = (
                cell + torch.matmul(cell, symmetric_displacement)
            ).view(-1, 3, 3)

        # Update cell_shifts in neighbor lists
        nl = getattr(data, NEIGHBOR_LIST_KEY, None)
        if nl is not None:
            for nl_dict in nl.items():
                if not isinstance(nl_dict, dict):
                    continue
                cs = nl_dict.get("cell_shifts")
                if cs is None or not isinstance(cs, torch.Tensor) or cs.numel() == 0:
                    continue
                
                edge_index = nl_dict.get("index_mapping")
                if edge_index is not None:
                    sender = edge_index[0]
                    nl_dict["cell_shifts"] = cs + torch.einsum(
                        "bi,bij->bj",
                        cs,
                        symmetric_displacement[data.batch[sender]],
                    )

        return data, displacement

    def _compute_virials_and_stress(
        self, 
        grad_disp: Optional[torch.Tensor], 
        displacement: torch.Tensor, 
        data: AtomicData
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extracts virials from gradients and computes stress if requested."""
        virials = -grad_disp if grad_disp is not None else torch.zeros_like(displacement)
        stress = torch.zeros_like(virials)

        if getattr(self, "compute_stress", False):
            cell = getattr(data, CELL_KEY, None)
            if cell is not None:
                cell = cell.view(-1, 3, 3)
                volume = torch.linalg.det(cell).abs().unsqueeze(-1)
                stress = virials / volume.view(-1, 1, 1)
                
                # Clamp unphysical values if volume is near zero to avoid NaNs / infs
                stress = torch.where(
                    torch.abs(stress) < 1e10,
                    stress,
                    torch.zeros_like(stress),
                )
        return virials, stress

    def forward(self, data: AtomicData) -> AtomicData:
        """Forward pass through the gradient layer.

        Parameters
        ----------
        data:
            AtomicData instance

        Returns
        -------
        data:
            Updated AtomicData instance, where the "out" field has
            been populated with the base predictions of the model (eg,
            the energy) as well as the target predictions produced through
            gradient operations (forces, and optionally virials / stress).
        """

        data.pos.requires_grad_(True)

        # Backward compat: old checkpoints lack virial attrs
        _compute_virials = getattr(self, "compute_virials", False)

        # Optional: apply symmetric strain for virial computation
        displacement = None
        if _compute_virials:
            data, displacement = self._apply_strain(data)

        data = self.model(data)

        if FORCE_KEY in self.targets:
            if self.name == "SumOut":
                y = data.out[ENERGY_KEY]
            else:
                y = data.out[self.name][ENERGY_KEY]

            wrt = [data.pos] if displacement is None else [data.pos, displacement]

            dy = torch.autograd.grad(
                y.sum(),
                wrt,
                create_graph=self.training,
                allow_unused=True,
            )

            dy_dr = dy[0]
            if self.name == "SumOut":
                data.out[FORCE_KEY] = -dy_dr
            else:
                data.out[self.name][FORCE_KEY] = -dy_dr

            if displacement is not None:
                dy_dd = dy[1] if len(dy) > 1 else None
                virials, stress = self._compute_virials_and_stress(
                    dy_dd, displacement, data
                )
                if self.name == "SumOut":
                    data.out[VIRIALS_KEY] = virials
                    data.out[STRESS_KEY] = stress
                else:
                    data.out[self.name][VIRIALS_KEY] = virials
                    data.out[self.name][STRESS_KEY] = stress

        data.pos = data.pos.detach()

        return data

    def neighbor_list(self, **kwargs: Any):
        return self.model.neighbor_list(**kwargs)
