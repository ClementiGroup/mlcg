import torch
from typing import List, Type, Tuple
from copy import deepcopy
from collections import defaultdict

from ..nn import GradientsOut, Dihedral, SumOut
from ..nn.prior import _Prior
from ..nn.utils import desparsify_prior_module
from ..neighbor_list.neighbor_list import make_neighbor_list
from ..data import AtomicData
from ..nn.prior import (
    _Prior,
    Dihedral,
    GeneralAngles,
    GeneralBonds,
    HarmonicBonds,
    HarmonicAngles,
    HarmonicImpropers,
)

"""It is convinient to define various prior models for many types of
interactions which leads to a large number of priors. While this is convinient
when difining these priors, it results in a clear slowdown of the simulation.
Below are defined Prior models that are specialized to a specific simulation.

The prior models for which such specialization is available using the function

.. autofunction:: condense_all_priors_for_simulation

are listed in the :obj:`to_reduce` dictionary:

.. autodata:: to_reduce
    :annotation:

The names of the specialized prior classes have the name of their original class
with `Static` prepended and they are inherit from :obj:`StaticPrior`.

"""

#: register the types of prior that can be reduced
to_reduce = {
    "Dihedral": Dihedral,
    "GeneralAngles": GeneralAngles,
    "GeneralBonds": GeneralBonds,
    "HarmonicBonds": HarmonicBonds,
    "HarmonicAngles": HarmonicAngles,
    "HarmonicImpropers": HarmonicImpropers,
}


class StaticPrior(torch.nn.Module):
    def __init__(self, **params):
        torch.nn.Module.__init__(self)
        self.param_names = []
        for k, v in params.items():
            self.register_buffer(k, v)
            self.param_names.append(k)

    def data2parameters(self, data):
        params = {k: getattr(self, k) for k in self.param_names}
        return params


def constructor(self, **params):
    StaticPrior.__init__(self, **params)


glbls = globals()
for name, CLS in to_reduce.items():
    glbls[f"Static{name}"] = type(
        f"Static{name}",
        (StaticPrior, CLS),
        {"__init__": constructor, "name": CLS.name},
    )


def condense_all_priors_for_simulation(
    priors: SumOut,
    data_list: List[AtomicData],
) -> Tuple[SumOut, List[AtomicData]]:
    """Specilize the priors for all the prior types registered in

    .. autodata:: to_reduce

    The particular list of atomic structures contained in
    :obj:`data_list` and adapt the neighbor lists in
    :obj:`data_list` accordingly.
    The specialized priors can only be used on a collated version of the
    returned list of atomic structures so this function is only meant to
    speedup simulations.

    Parameters
    ----------
    TargetPrior :
        A prior class that will be condensed into one prior
    priors :
        dictionary of prior models
    data_list :
        list of configuration for which the condensed prior should be produced

    Returns
    -------

    Condensed priors and the list of configuration with adapted neighborlists.
    """
    for TargetPrior in to_reduce.values():
        priors, data_list = condense_prior_for_simulation(
            TargetPrior, priors, data_list
        )
    return priors, data_list


def condense_prior_for_simulation(
    TargetPrior: Type[_Prior],
    priors: SumOut,
    data_list: List[AtomicData],
) -> Tuple[SumOut, List[AtomicData]]:
    """Condense the priors of type TargetPrior into a :obj:`StaticPrior` for
    the particular list of atomic structures contained in
    :obj:`data_list` and adapt the neighbor lists in
    :obj:`data_list` accordingly.
    The :obj:`StaticPrior` can only be used on a collated version of the
    returned list of atomic structures so this function is only meant to
    speedup simulations.

    Parameters
    ----------
    TargetPrior :
        A prior class that will be condensed into one prior
    priors :
        dictionary of prior models
    data_list :
        list of configuration for which the condensed prior should be produced

    Returns
    -------

    Condensed priors and the list of configuration with adapted neighborlists.

    """
    name = None
    for k, CLS in to_reduce.items():
        if CLS == TargetPrior:
            name = f"Static{k}"
            StaticCLS = glbls[name]
            break

    if name is None:
        raise RuntimeError(
            f"The condense target {TargetPrior} has not been registered in {to_reduce}"
        )

    params = defaultdict(list)
    condensed_data_list = deepcopy(data_list)
    condensed_priors = deepcopy(priors)

    n_degs = []
    n_removed = 0
    for k, prior in priors.models.items():
        if isinstance(prior.model, TargetPrior):
            # remove prior that is being condensed
            mod = condensed_priors.models.pop(k)
            n_removed += 1
        if isinstance(prior.model, Dihedral) and TargetPrior == Dihedral:
            n_degs.append(prior.model.n_degs)
        else:
            n_degs.append(1)
    n_degs = max(n_degs)

    # there are no prior to condense for TargetPrior
    if n_removed == 0:
        return condensed_priors, condensed_data_list

    for ii, data in enumerate(condensed_data_list):
        index_mapping = []
        for k, prior in priors.models.items():
            if isinstance(prior.model, TargetPrior):
                prior.model = desparsify_prior_module(prior.model)
                pp = prior.model.data2parameters(data)
                for kv, v in pp.items():
                    if v.ndim == 1:
                        v = torch.unsqueeze(v, dim=-1)
                    ee = torch.zeros((v.shape[0], n_degs))
                    ee[:, : v.shape[1]] = v
                    params[kv].append(ee)

                index_mapping.append(data.neighbor_list[k]["index_mapping"])
                # remove NL entry that is being condensed
                condensed_data_list[ii].neighbor_list.pop(k)

        index_mapping = torch.cat(index_mapping, dim=-1)

        nl = make_neighbor_list(
            tag=name, order=TargetPrior._order, index_mapping=index_mapping
        )
        # add condensed NL
        condensed_data_list[ii].neighbor_list[StaticCLS.name] = nl

    for k in params.keys():
        if TargetPrior == Dihedral:
            # need to complete to the max size of epansion
            params[k] = torch.vstack(params[k])
        else:
            params[k] = torch.cat(params[k]).flatten()
    static_prior = StaticCLS(**params)
    condensed_priors.models[StaticCLS.name] = GradientsOut(static_prior)

    return condensed_priors, condensed_data_list
