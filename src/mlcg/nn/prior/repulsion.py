import numpy as np
import torch
from typing import Final, Optional, Dict
from torch_geometric.utils import scatter

from .base import _Prior
from ...data.atomic_data import AtomicData
from ...geometry.topology import Topology
from ...geometry.internal_coordinates import (
    compute_distances,
)


class Repulsion(_Prior):
    r"""1-D power law repulsion prior for feature :math:`x` of the form:

    .. math::

        U_{ \textnormal{Repulsion}}(x) = (\sigma/x)^6

    where :math:`\sigma` is the excluded volume.

    Parameters
    ----------
    statistics:
        Dictionary of interaction parameters for each type of atom pair,
        where the keys are tuples of interacting bead types and the
        corresponding values define the interaction parameters. These
        Can be hand-designed or taken from the output of
        `mlcg.geometry.statistics.compute_statistics`, but must minimally
        contain the following information for each key:

        .. code-block:: python

            tuple(*specific_types) : {
                "sigma" : torch.Tensor scalar that describes the excluded
                    volume of the two interacting atoms.
                ...

                }
        The keys can be tuples of 2 integer atom types.
    """

    name: Final[str] = "repulsion"
    _neighbor_list_name = "fully connected"

    def __init__(self, statistics: Dict) -> None:
        super(Repulsion, self).__init__()
        keys = torch.tensor(list(statistics.keys()), dtype=torch.long)
        self.allowed_interaction_keys = list(statistics.keys())
        self.order = 2
        self.name = self.name
        unique_types = torch.unique(keys.flatten())
        assert unique_types.min() >= 0
        max_type = unique_types.max()
        sizes = tuple([max_type + 1 for _ in range(self.order)])
        sigma = torch.zeros(sizes)
        for key in statistics.keys():
            sigma[key] = statistics[key]["sigma"]
        self.register_buffer("sigma", sigma)

    def data2features(self, data: AtomicData) -> torch.Tensor:
        """Computes features for the harmonic interaction from
        an AtomicData instance)

        Parameters
        ----------
        data:
            Input `AtomicData` instance

        Returns
        -------
        torch.Tensor:
            Tensor of computed features
        """

        mapping = data.neighbor_list[self.name]["index_mapping"]
        return Repulsion.compute_features(data.pos, mapping)

    def forward(self, data: AtomicData) -> AtomicData:
        """Forward pass through the repulsion interaction.

        Parameters
        ----------
        data:
            Input AtomicData instance that possesses an appropriate
            neighbor list containing both an 'index_mapping'
            field and a 'mapping_batch' field for accessing
            beads relevant to the interaction and scattering
            the interaction energies onto the correct example/structure
            respectively.

        Returns
        -------
        AtomicData:
            Updated AtomicData instance with the 'out' field
            populated with the predicted energies for each
            example/structure
        """

        mapping = data.neighbor_list[self.name]["index_mapping"]
        mapping_batch = data.neighbor_list[self.name]["mapping_batch"]
        interaction_types = [
            data.atom_types[mapping[ii]] for ii in range(self.order)
        ]
        features = self.data2features(data)
        y = Repulsion.compute(features, self.sigma[interaction_types])
        y = scatter(y, mapping_batch, dim=0, reduce="sum")
        data.out[self.name] = {"energy": y}
        return data

    @staticmethod
    def compute_features(pos, mapping):
        return compute_distances(pos, mapping)

    @staticmethod
    def compute(x, sigma):
        """Method defining the repulsion interaction"""
        rr = (sigma / x) * (sigma / x)
        return rr * rr * rr

    @staticmethod
    def fit_from_values(
        values: torch.Tensor,
        percentile: Optional[float] = 1,
        cutoff: Optional[float] = None,
    ) -> Dict:
        """Method for fitting interaction parameters directly from input features

        Parameters
        ----------
        values:
            Input features as a tensor of shape (n_frames)
        percentile:
            If specified, the sigma value is calculated using the specified
            distance percentile (eg, percentile = 1) sets the sigma value
            at the location of the 1th percentile of pairwise distances. This
            option is useful for estimating repulsions for distance distribtions
            with long lower tails or lower distance outliers. Must be a number from
            0 to 1
        cutoff:
            If specified, only those input values below this cutoff will be used in
            evaluating the percentile

        Returns
        -------
        Dict:
            Dictionary of interaction parameters as retrived through
            `scipy.optimize.curve_fit`
        """
        values = values.numpy()
        if cutoff != None:
            values = values[values < cutoff]
        sigma = torch.tensor(np.percentile(values, percentile))
        stat = {"sigma": sigma}
        return stat

    @staticmethod
    def fit_from_potential_estimates(
        bin_centers_nz: torch.Tensor,
        dG_nz: torch.Tensor,
        percentile: Optional[float] = None,
    ) -> Dict:
        r"""Method for fitting interaction parameters from data

        Parameters
        ----------
        bin_centers:
            Bin centers from a discrete histgram used to estimate the energy
            through logarithmic inversion of the associated Boltzmann factor
        dG_nz:
            The value of the energy :math:`U` as a function of the bin
            centers, as retrived via:

            ..math::

                U(x) = -\frac{1}{\beta}\log{ \left( p(x)\right)}

            where :math:`\beta` is the inverse thermodynamic temperature and
            :math:`p(x)` is the normalized probability distribution of
            :math:`x`.


        Returns
        -------
        Dict:
            Dictionary of interaction parameters as retrived through
            `scipy.optimize.curve_fit`
        """

        delta = bin_centers_nz[1] - bin_centers_nz[0]
        sigma = bin_centers_nz[0] - 0.5 * delta
        stat = {"sigma": sigma}
        return stat

    @staticmethod
    def neighbor_list(topology: Topology) -> Dict:
        """Method for computing a neighbor list from a topology
        and a chosen feature type.

        Parameters
        ----------
        topology:
            A Topology instance with a defined fully-connected
            set of edges.

        Returns
        -------
        Dict:
            Neighborlist of the fully-connected distances
            according to the supplied topology
        """

        return {
            Repulsion.name: topology.neighbor_list(
                Repulsion._neighbor_list_name
            )
        }

class ExpRepulsion(_Prior):
    r"""1-D exp-6 potential repulsion prior for feature :math:`x` of the form:

    .. math::

        U_{ \textnormal{ExpRepulsion}}(x) = (6/\alpha)exp[\alpha(1 - x/r_0)]

    where :math:`\alpha` is the strength of the repulsion and :math:`r_0` is the excluded volume.

    Parameters
    ----------
    statistics:
        Dictionary of interaction parameters for each type of atom pair,
        where the keys are tuples of interacting bead types and the
        corresponding values define the interaction parameters. These
        Can be hand-designed or taken from the output of
        `mlcg.geometry.statistics.compute_statistics`, but must minimally
        contain the following information for each key:

        .. code-block:: python

            tuple(*specific_types) : {
                "alpha" : torch.Tensor scalar that describes the strength of the repulsion.
                "r_0" : torch.Tensor scalar that describes the excluded volume.
                ...

                }
        The keys can be tuples of 2 integer atom types.
    """

    name: Final[str] = "repulsion"
    _neighbor_list_name = "fully connected"

    def __init__(self, statistics: Dict) -> None:
        super(ExpRepulsion, self).__init__()
        keys = torch.tensor(list(statistics.keys()), dtype=torch.long)
        self.allowed_interaction_keys = list(statistics.keys())
        self.order = 2
        self.name = self.name
        unique_types = torch.unique(keys.flatten())
        assert unique_types.min() >= 0
        max_type = unique_types.max()
        sizes = tuple([max_type + 1 for _ in range(self.order)])
        alpha = torch.zeros(sizes)
        r_0 = torch.zeros(sizes)
        for key in statistics.keys():
            alpha[key] = statistics[key]["alpha"]
            r_0[key] = statistics[key]["r_0"]
        self.register_buffer("alpha", alpha)
        self.register_buffer("r_0", r_0)

    def data2features(self, data: AtomicData) -> torch.Tensor:
        """Computes features for the harmonic interaction from
        an AtomicData instance)
        """

        mapping = data.neighbor_list[self.name]["index_mapping"]
        return ExpRepulsion.compute_features(data.pos, mapping)

    def forward(self, data: AtomicData) -> AtomicData:
        """Forward pass through the repulsion interaction.

        Parameters
        ----------
        data:
            Input AtomicData instance that possesses an appropriate
            neighbor list containing both an 'index_mapping'
            field and a 'mapping_batch' field for accessing
            beads relevant to the interaction and scattering
            the interaction energies onto the correct example/structure
            respectively.

        Returns
        -------
        AtomicData:
            Updated AtomicData instance with the 'out' field
            populated with the predicted energies for each
            example/structure
        """

        mapping = data.neighbor_list[self.name]["index_mapping"]
        mapping_batch = data.neighbor_list[self.name]["mapping_batch"]
        interaction_types = [
            data.atom_types[mapping[ii]] for ii in range(self.order)
        ]
        features = self.data2features(data)
        y = ExpRepulsion.compute(features, self.alpha[interaction_types], self.r_0[interaction_types])
        y = scatter(y, mapping_batch, dim=0, reduce="sum")
        data.out[self.name] = {"energy": y}
        return data

    @staticmethod
    def compute_features(pos, mapping):
        return compute_distances(pos, mapping)

    @staticmethod
    def compute(x, alpha, r_0):
        """Method defining the repulsion interaction"""
        rr = 1 - (x / r_0)
        return (6 / alpha) * torch.exp(alpha * rr)


    @staticmethod
    def neighbor_list(topology: Topology) -> Dict:
        """Method for computing a neighbor list from a topology
        and a chosen feature type.
        """
        return {
            ExpRepulsion.name: topology.neighbor_list(
                ExpRepulsion._neighbor_list_name
            )
        }