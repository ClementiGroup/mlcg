import numpy as np
import copy
import torch
from typing import Final, Optional, Dict
from torch_geometric.utils import scatter

from .base import _Prior
from ...data.atomic_data import AtomicData
from ...geometry.topology import Topology
from ...geometry.internal_coordinates import (
    compute_distances,
)

from ..kernels.models.prior import flash_repulsion


class Repulsion(_Prior):
    r"""1-D power law repulsion prior for feature :math:`x` of the form:

    .. math::

        U_{ \textnormal{Repulsion}}(x) = \left(\frac{\sigma}{x}\right)^6

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
                "sigma" : #torch.Tensor scalar that describes the excluded
                    #volume of the two interacting atoms.
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
        pbc = getattr(data, "pbc", None)
        cell = getattr(data, "cell", None)
        return self.compute_features(
            pos=data.pos,
            mapping=mapping,
            pbc=pbc,
            cell=cell,
            batch=data.batch,
        )

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
        interaction_types = tuple(
            data.atom_types[mapping[ii]] for ii in range(self.order)
        )
        features = self.data2features(data)
        y = Repulsion.compute(features, self.sigma[interaction_types])
        num_graphs = data.ptr.numel() - 1 if hasattr(data, "ptr") else None
        y = scatter(y, mapping_batch, dim=0, reduce="sum", dim_size=num_graphs)
        data.out[self.name] = {"energy": y}
        return data

    @staticmethod
    def compute_features(
        pos: torch.Tensor,
        mapping: torch.Tensor,
        pbc: torch.Tensor = None,
        cell: torch.Tensor = None,
        batch: torch.Tensor = None,
    ) -> torch.Tensor:
        if all([feat != None for feat in [pbc, cell]]):
            cell_shifts = _Prior._get_cell_shifts(
                pos, mapping, pbc, cell, batch
            )
        else:
            cell_shifts = None
        return compute_distances(pos, mapping, cell_shifts)

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


class FlashRepulsion(_Prior):
    """
    Computes per-graph repulsion energy:
        e_ij = (sigma[type_i, type_j] / r_ij)^6
    reduced as sum over edges per graph (using mapping_batch).

    Returns: y of shape [num_graphs] (float32).
    """

    name = "repulsion"

    def __init__(
        self,
        sigma: torch.Tensor,
        name: str,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.register_buffer("sigma", sigma)
        self.eps = float(eps)
        self.name = name

    def forward(self, data) -> torch.Tensor:  # int
        # minimal checks
        pos = data.pos  # [N,3],
        atom_types = data.atom_types  # [N], int32 or int64
        index_mapping = data.neighbor_list[self.name]["index_mapping"]
        mapping_batch = data.neighbor_list[self.name]["mapping_batch"]
        num_graphs = data.ptr.numel() - 1 if hasattr(data, "ptr") else None
        # assert pos.is_cuda, "pos must be CUDA"
        assert pos.shape[-1] == 3, "pos must be [N,3]"
        assert index_mapping.shape[0] == 2, "index_mapping must be [2,E]"
        # assert (
        # self.sigma.is_cuda == pos.is_cuda
        # ), "sigma and pos must be on same device"
        y = flash_repulsion(
            pos=pos,
            atom_types=atom_types,
            index_mapping=index_mapping,
            mapping_batch=mapping_batch,
            sigma=self.sigma,
            num_graphs=num_graphs,
            eps=self.eps,
        )
        data.out[self.name] = {"energy": y}
        return data

    @classmethod
    def flash_from_standard(cls, standard_model: Repulsion) -> "FlashRepulsion":
        """Class method to initialize a FlashRepulsion from a preexisting Repulsion model.

        Parameters
        ----------
        standard_model:
            A preexisting Repulsion model from which to initialize the FlashRepulsion. The
            sigma parameter will be taken from the standard_model and used to initialize
            the flash model.
        """

        if not isinstance(standard_model, Repulsion):
            raise ValueError(
                f"Expected input model of type Repulsion, but got {type(standard_model)}"
            )

        return cls(
            sigma=copy.deepcopy(standard_model.sigma),
            name=standard_model.name,
        )
