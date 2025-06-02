import torch
from typing import Sequence, Any, List
from ..data.atomic_data import AtomicData
from ..data._keys import *


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
        return data

    def neighbor_list(self, **kwargs):
        nl = {}
        for _, model in self.models.items():
            nl.update(**model.neighbor_list(**kwargs))
        return nl


class EnergyOut(torch.nn.Module):
    r"""Extractor for energy computed via SchNet

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
            targets = ["enegy"]
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
        The gradient targets to produce from a model output. These can be any
        of the gradient properties referenced in `mlcg.data._keys`.
        At the moment only forces are implemented.

    Example
    -------
        To predict forces from an energy model, one would supply a model that
        predicts a scalar atom property (an energy) and specify the `FORCE_KEY`
        in the targets.
    """

    _targets = {FORCE_KEY: ENERGY_KEY}

    def __init__(self, model: torch.nn.Module, targets: str = FORCE_KEY):
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
            the energy as well as the target predictions produced through
            gradient operations.
        """

        data.pos.requires_grad_(True)
        data = self.model(data)

        if FORCE_KEY in self.targets:
            if self.name == "SumOut":
                y = data.out[ENERGY_KEY]
            else:
                y = data.out[self.name][ENERGY_KEY]

            dy_dr = torch.autograd.grad(
                y.sum(),
                data.pos,
                # grad_outputs=torch.ones_like(y),
                # retain_graph=self.training,
                create_graph=self.training,
            )[0]
            if self.name == "SumOut":
                data.out[FORCE_KEY] = -dy_dr
            else:
                data.out[self.name][FORCE_KEY] = -dy_dr
            # assert not torch.any(torch.isnan(dy_dr)), f"nan in {self.name}"
        data.pos = data.pos.detach()

        return data

    def neighbor_list(self, **kwargs: Any):
        return self.model.neighbor_list(**kwargs)
