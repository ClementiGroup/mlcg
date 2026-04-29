from typing import Dict, List, Optional, Tuple
from copy import deepcopy
import torch
from scipy.integrate import trapezoid

from ..data import AtomicData
from ..nn.prior import Dihedral, Harmonic, _Prior
from ..utils import tensor2tuple
from ._symmetrize import _symmetrise_map, _flip_map


def _get_all_unique_keys(
    unique_types: torch.Tensor, order: int
) -> torch.Tensor:
    """Helper function for returning all unique, symmetrised atom type keys

    Parameters
    ----------
    unique_types:
        Tensor of unique atom types of shape (order, n_unique_atom_types)
    order:
        The order of the interaction type

    Returns
    -------
    torch.Tensor:
       Tensor of unique atom types, symmetrised
    """
    # get all combinations of size order between the elements of unique_types
    keys = torch.cartesian_prod(*[unique_types for ii in range(order)]).t()
    # symmetrize the keys and keep only unique entries
    sym_keys = _symmetrise_map[order](keys)
    unique_sym_keys = torch.unique(sym_keys, dim=1)
    return unique_sym_keys


def _get_bin_centers(
    feature: torch.Tensor, nbins: int, b_min: float, b_max: float
) -> torch.Tensor:
    """Returns bin centers for histograms.

    Parameters
    ----------
    feature:
        1-D input values of a feature.
    nbins:
        Number of bins in the histogram
    b_min
        If specified, the lower bound of bin edges. If not specified, the lower bound
        defaults to the lowest value in the input feature
    b_max
        If specified, the upper bound of bin edges. If not specified, the upper bound
        defaults to the greatest value in the input feature

    Returns
    -------
    torch.Tensor:
        torch tensor containing the locaations of the bin centers
    """

    if b_min >= b_max:
        raise ValueError("b_min must be less than b_max.")

    bin_centers = torch.zeros((nbins,), dtype=torch.float64)

    delta = (b_max - b_min) / nbins
    bin_centers = (
        b_min
        + 0.5 * delta
        + torch.arange(0, nbins, dtype=torch.float64) * delta
    )
    return bin_centers


def compute_statistics(
    data: AtomicData,
    target: str,
    beta: float,
    TargetPrior: _Prior = Harmonic,
    nbins: int = 100,
    bmin: Optional[float] = None,
    bmax: Optional[float] = None,
    fit_from_values: bool = False,
    target_fit_kwargs: Optional[Dict] = None,
) -> Dict:
    r"""Function for computing atom type-specific statistics for
    every combination of atom types present in a collated AtomicData
    structure.

    Parameters
    ----------
    data:
        Input data, in the form of a collated list of individual AtomicData
        structures.
    target:
        The keyword specifiying with `neighbor_list` sub_dictionary should
        be used to gather statisitics
    beta:
        Inverse thermodynamic temperature:

        .. math::

            \beta = \frac{1}{k_B T}

        where :math:`k_B` is Boltzmann's constant and :math:`T` is the temperature.
    TargetPrior:
        The class type of prior for which the statistics will be processed.
    nbins:
        The number of bins over which 1-D feature histograms are constructed
        in order to estimate distributions
    bmin:
        If specified, the lower bound of bin edges. If not specified, the lower bound
        defaults to the lowest value in the input feature
    bmax:
        If specified, the upper bound of bin edges. If not specified, the upper bound
        defaults to the greatest value in the input feature
    fit_from_values:
        If True, the prior parameters are estimated directly from features values
        instead of their implied emperical potentials
    target_fit_kwargs:
        Extra fit options that are prior_specific

    Returns
    -------
    Dict:
        Dictionary of gathered statistics and estimated parameters based on
        the `TargetPrior`. The following key/value pairs are common across all `TargetPrior`
        choices:

        .. code-block:: python

            (*specific_types) : {

                ...

                "p" : torch.Tensor of shape [n_bins], containing the normalized bin counts
                    of the of the 1-D feature corresponding to the atom_type group
                    (*specific_types) = (specific_types[0], specific_types[1], ...)
                "p_bin": : torch.Tensor of shape [n_bins] containing the bin center values
                "V" : torch.tensor of shape [n_bins], containing the emperically estimated
                    free energy curve according to a direct Boltzmann inversion of the
                    normalized probability distribution for the feature.
                "V_bin" : torch_tensor of shape [n_bins], containing the bin center values
            }

        where `...` indicates other sub-key/value pairs apart from those enumerated above,
        which may appear depending on the chosen `TargetPrior`. For example,
        if `TargetPrior` is `HarmonicBonds`, there will also be keys/values associated with
        estimated bond constants and means.

    Example
    -------

    .. code-block:: python

        my_data = AtomicData(
            out={},
            pos=[769600, 3],
            atom_types=[769600],
            n_atoms=[20800],
            neighbor_list={
                bonds={
                  tag=[20800],
                  order=[20800],
                  index_mapping=[2, 748800],
                  cell_shifts=[20800],
                  rcut=[20800],
                  self_interaction=[20800]
                },
                angles={
                  tag=[20800],
                  order=[20800],
                  index_mapping=[3, 977600],
                  cell_shifts=[20800],
                  rcut=[20800],
                  self_interaction=[20800]
                }
            },
            batch=[769600],
            ptr=[20801]
        )

        angle_stats = bond_stats = compute_statistics(my_data,
             'bonds', beta=beta,
             TargetPrior=HarmonicBonds
        )
        dihedral_stats = compute_statistics(my_data,
                                            'dihedrals',
                                            beta=beta,
                                            TargetPrior=Dihedral
        )


    """
    if target_fit_kwargs == None:
        target_fit_kwargs = {}
    unique_types = torch.unique(data.atom_types)
    order = data.neighbor_list[target]["index_mapping"].shape[0]
    unique_keys = _get_all_unique_keys(unique_types, order)

    mapping = data.neighbor_list[target]["index_mapping"]
    pbc = getattr(data, "pbc", None)
    cell = getattr(data, "cell", None)
    values = TargetPrior.compute_features(
        data.pos,
        mapping,
        pbc=pbc,
        cell=cell,
        batch=data.batch,
    )
    interaction_types = torch.vstack(
        [data.atom_types[mapping[ii]] for ii in range(order)]
    )

    interaction_types = _symmetrise_map[order](interaction_types)

    statistics = {}
    for unique_key in unique_keys.t():
        # find which values correspond to unique_key type of interaction
        mask = torch.all(
            torch.vstack(
                [
                    interaction_types[ii, :] == unique_key[ii]
                    for ii in range(order)
                ]
            ),
            dim=0,
        )
        val = values[mask]
        if len(val) == 0:
            continue

        if bmin is None:
            b_min = val.min()
        else:
            b_min = bmin
        if bmax is None:
            b_max = val.max()
        else:
            b_max = bmax

        bin_centers = _get_bin_centers(val, nbins, b_min=b_min, b_max=b_max)
        hist = torch.histc(val, bins=nbins, min=b_min, max=b_max)
        mask = hist > 0
        bin_centers_nz = bin_centers[mask]
        ncounts_nz = hist[mask]
        dG_nz = -torch.log(ncounts_nz) / beta

        if fit_from_values:
            params = TargetPrior.fit_from_values(val, **target_fit_kwargs)

        else:
            params = TargetPrior.fit_from_potential_estimates(
                bin_centers_nz, dG_nz, **target_fit_kwargs
            )

        kk = tensor2tuple(unique_key)
        statistics[kk] = params

        statistics[kk]["p"] = hist / trapezoid(
            hist.cpu().numpy(), x=bin_centers.cpu().numpy()
        )
        statistics[kk]["p_bin"] = bin_centers
        statistics[kk]["V"] = dG_nz
        statistics[kk]["V_bin"] = bin_centers_nz

        kf = tensor2tuple(_flip_map[order](unique_key))
        statistics[kf] = deepcopy(statistics[kk])

    return statistics


def fit_baseline_models(
    data: AtomicData,
    beta: float,
    priors_cls: List[_Prior],
    nbins: int = 100,
    bmin: Optional[float] = None,
    bmax: Optional[float] = None,
) -> Tuple[torch.nn.ModuleDict, Dict]:
    r"""Function for parametrizing a list of priors based on type-specific interactions contained in
    a collated AtomicData structure

    Parameters
    ----------
    data:
        Input data, in the form of a collated list of individual AtomicData
        structures.
    beta:
        Inverse thermodynamic temperature:

        .. math::

        \beta = \frac{1}{k_B T}

        where :math:`k_B` is Boltzmann's constant and :math:`T` is the temperature.
    priors_cls:
       List of priors to parametrize based on the input data
    nbins:
        The number of bins over which 1-D feature histograms are constructed
        in order to estimate distributions
    bmin
        If specified, the lower bound of bin edges. If not specified, the lower bound
        defaults to the lowest value in the input feature
    bmax
        If specified, the upper bound of bin edges. If not specified, the upper bound
        defaults to the greatest value in the input feature

    Returns
    -------
    nn.Module:
        The list of parametrized priors
    Dict:
        Corresponding statistsics for prior fits
    """

    statistics = {}
    models = torch.nn.ModuleDict()
    for TargetPrior in priors_cls:
        k = TargetPrior.name
        statistics[k] = compute_statistics(
            data,
            k,
            beta,
            TargetPrior=TargetPrior,
            nbins=nbins,
            bmin=bmin,
            bmax=bmax,
        )
        models[k] = TargetPrior(statistics[k])
    return models, statistics
