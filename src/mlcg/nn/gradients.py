import torch
from typing import Sequence, Any, List, Optional
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
    own_nl_models:
        Optional model keys that should not receive the shared
        ``data.neighbor_list``. Useful in cases where a model needs to
        build its own coordinate-dependent neighbor list and in compilation
        cases. For each key listed here, ``data.neighbor_list``
        is temporarily cleared right before that model's call and restored
        immediately after, leaving the other models unaffected. Defaults to
        ``None``.

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

        # Withhold the shared neighbor_list from SchNet's call:
        full_model = SumOut(
            models,
            targets=[ENERGY_KEY, FORCE_KEY],
            own_nl_models=["SchNet"],
        )


    """

    name: str = "SumOut"

    def __init__(
        self,
        models: torch.nn.ModuleDict,
        targets: Optional[List[str]] = None,
        own_nl_models: Optional[Sequence[str]] = None,
    ):
        super(SumOut, self).__init__()
        if targets is None:
            targets = [ENERGY_KEY, FORCE_KEY]
        self.targets = targets
        self.models = models
        self.own_nl_models = set(own_nl_models) if own_nl_models else set()

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

        own_nl_models = getattr(self, "own_nl_models", set())
        original_neighbor_list = data.neighbor_list if own_nl_models else None

        for name in self.models.keys():
            if name in own_nl_models:
                data.neighbor_list = {}
                data = self.models[name](data)
                data.neighbor_list = original_neighbor_list
            else:
                data = self.models[name](data)
            for target in self.targets:
                data.out[target] += data.out[name][target]
        return data

    def neighbor_list(self, **kwargs):
        nl = {}
        for _, model in self.models.items():
            nl.update(**model.neighbor_list(**kwargs))
        return nl


class NeighborSharingSumOut(SumOut):
    r"""`SumOut` that computes the geometric neighbor list of a network once
    and shares it with other models, e.g. `FastCutoffRepulsion` and
    `FastCutoffExpRepulsion`.

    On every forward pass the neighbor list of ``nls_model_key`` is computed
    and stored under that network's name so it is not recomputed by the
    network itself. For every model in ``shared_nl_models`` the entry
    ``data.neighbor_list[model.name]`` is replaced by a copy of that
    neighbor list, keeping ``index_mapping_exclusions`` (and a cached
    ``exc_table``) from the original entry. The original
    ``data.neighbor_list`` is restored after the forward pass, so the input
    data can be reused by other models.

    Parameters
    ----------
    nls_model_key:
        Key in ``models`` of the network whose neighbor list is shared. The
        network (optionally wrapped in `GradientsOut`) must expose
        ``rbf_layer.cutoff.cutoff_upper``, ``max_num_neighbors`` and
        ``neighbor_list(data, rcut, max_num_neighbors)``, as `SchNet` does.
    models:
        Dictionary of predictors models keyed by their name attribute
    shared_nl_models:
        Keys in ``models`` that receive the shared neighbor list. Models not
        listed here (e.g. bonds and angles priors) keep their own entry.
    targets:
        List of prediction targets that will be pooled
    own_nl_models:
        See `SumOut`.
    """

    def __init__(
        self,
        nls_model_key: str,
        models: torch.nn.ModuleDict,
        shared_nl_models: Sequence[str],
        targets: Optional[List[str]] = None,
        own_nl_models: Optional[Sequence[str]] = None,
    ):
        # Deferred imports: `schnet` and `prior` import `mlcg.geometry`, which
        # in turn imports `mlcg.nn.prior`, so importing them at module level
        # creates a circular import through `mlcg.nn.__init__`.
        # TODO: remove the statistics module which is not being used anymore
        from .prior.repulsion import _BaseFastRepulsion
        from .schnet import SchNet

        super().__init__(models, targets, own_nl_models)
        if nls_model_key not in models:
            raise ValueError(f"{nls_model_key=} is not a key of `models`")
        if nls_model_key in self.own_nl_models:
            raise ValueError(
                f"{nls_model_key=} can't be contained in {own_nl_models=}"
            )
        nls_model = models[nls_model_key]
        # Guard for only using SchNet 
        # TOOO: create an interface for extracting cutoff of different MPNNs
        if not isinstance(getattr(nls_model, "model", nls_model), SchNet):
            raise TypeError(
                "Only subclasses of SchNet are supported as root for the nls"
            )

        shared_nl_models = list(shared_nl_models)
        for key in shared_nl_models:
            if key not in models:
                raise ValueError(f"shared model {key!r} is not in `models`")
            if key == nls_model_key or key in self.own_nl_models:
                raise ValueError(
                    f"shared model {key!r} can't be `nls_model_key` or be "
                    "contained in `own_nl_models`"
                )
            shared = models[key]
            if not isinstance(
                getattr(shared, "model", shared), _BaseFastRepulsion
            ):
                raise TypeError(
                    "Only subclasses of _BaseFastRepulsion are supported for sharing the nls"
                )
        self.nls_model_key = nls_model_key
        self.shared_nl_models = shared_nl_models

    def _network_neighbor_list(self, data: AtomicData):
        model = self.models[self.nls_model_key]
        network = getattr(model, "model", model)  # unwrap GradientsOut
        # extract cutoff:
        nl = network.neighbor_list(
            data,
            network.rbf_layer.cutoff.cutoff_upper,
            network.max_num_neighbors,
        )[network.name]
        return network.name, nl

    def forward(self, data: AtomicData) -> AtomicData:
        original_nl = data.neighbor_list
        network_name, shared_nl = self._network_neighbor_list(data)

        new_nl = dict(original_nl)
        new_nl[network_name] = shared_nl
        shared_names = [self.models[key].name for key in self.shared_nl_models]
        for name in shared_names:
            entry = dict(shared_nl)
            old_entry = original_nl.get(name)
            if old_entry is not None:
                for k in ("index_mapping_exclusions", "exclusion_table"):
                    if k in old_entry:
                        entry[k] = old_entry[k]
            new_nl[name] = entry

        data.neighbor_list = new_nl
        data = super().forward(data)

        # keep the exclusion tables built during the forward pass cached
        # in the original entries so they are not rebuilt every call
        for name in shared_names:
            if name in original_nl and "exclusion_table" in new_nl[name]:
                original_nl[name]["exclusion_table"] = new_nl[name][
                    "exclusion_table"
                ]
        data.neighbor_list = original_nl
        return data


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
        targets: Optional[List[str]] = None,
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
