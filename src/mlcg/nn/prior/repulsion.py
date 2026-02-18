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

    name: str = "repulsion"
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

    def data2features(self, data):
        mapping = data.neighbor_list[self.name]["index_mapping"]
        pbc = getattr(data, "pbc", None)
        cell = getattr(data, "cell", None)
        return self.compute_features(
            pos=data.pos,
            mapping=mapping,
            pbc=pbc,
            cell=cell,
            batch=data.batch
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
        interaction_types = [
            data.atom_types[mapping[ii]] for ii in range(self.order)
        ]
        features = self.data2features(data)
        y = Repulsion.compute(features, self.sigma[interaction_types])
        y = scatter(y, mapping_batch, dim=0, reduce="sum")
        data.out[self.name] = {"energy": y}
        return data

    @staticmethod
    def compute_features(
        pos: torch.Tensor,
        mapping: torch.Tensor,
        pbc: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cell_shifts = _Prior._get_cell_shifts(pos, mapping, pbc, cell, batch)
        return compute_distances(
            pos=pos,
            mapping=mapping,
            cell_shifts=cell_shifts
        )
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


class CutoffRepulsion(Repulsion):
    def __init__(self, statistics: Dict, cutoff: float,name:str = "repulsion") -> None:
        super().__init__(statistics=statistics)
        self.cutoff = cutoff
        self.name=name

    @staticmethod
    def compute_dev(x, sigma):
        """Method defining the repulsion interaction"""
        orig = Repulsion.compute(x,sigma)
        return -6*orig/x
    
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

        features = self.data2features(data)
        #filter all distances that are larger than the cutoff
        mask = features < self.cutoff
        features = features[mask]
        mapping = data.neighbor_list[self.name]["index_mapping"][:,mask]
        mapping_batch = data.neighbor_list[self.name]["mapping_batch"][mask]
        interaction_types = [
            data.atom_types[mapping[ii]] for ii in range(self.order)
        ]
        
        
        y = Repulsion.compute(features, self.sigma[interaction_types])
        yc = Repulsion.compute(self.cutoff, self.sigma[interaction_types])
        #ensure that the cutoff is continupus
        y = y - yc - (features - self.cutoff)*(CutoffRepulsion.compute_dev(self.cutoff, self.sigma[interaction_types]))
        y = scatter(y, mapping_batch, dim=0, reduce="sum")
        data.out[self.name] = {"energy": y}
        return data



class RepulsionBuckMod(_Prior):
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
        super(RepulsionBuckMod, self).__init__()
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
        return RepulsionBuckMod.compute_features(data.pos, mapping)

    def data2parameters(self, data):
        mapping = data.neighbor_list[self.name]["index_mapping"]
        interaction_types = [
            data.atom_types[mapping[ii]] for ii in range(self.order)
        ]
        params = {
            "alpha": self.alpha[interaction_types].flatten(),
            "r_0": self.r_0[interaction_types].flatten(),
        }
        return params

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
        params = self.data2parameters(data)
        features = self.data2features(data).flatten()
        y = RepulsionBuckMod.compute(features, **params)
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
            RepulsionBuckMod.name: topology.neighbor_list(
                RepulsionBuckMod._neighbor_list_name
            )
        }


class LennardJonesShifted(_Prior):
    r"""1-D Lennard-Jones potential with shift modification for feature :math:`x` of the form:

    .. math::

        U_{\text{LJ}}(x) = 4\epsilon \left[ \left(\frac{\sigma}{x}\right)^{12} - \left(\frac{\sigma}{x}\right)^{6} \right] - U_{\text{LJ}}(r_c)

    where :math:`\sigma` is the distance at which the potential is zero, :math:`\epsilon` is the well depth,
    and :math:`r_c` is the cutoff distance. The potential is shifted by :math:`U_{\text{LJ}}(r_c)` to ensure
    it is zero at the cutoff.

    Parameters
    ----------
    statistics:
        Dictionary of interaction parameters for each type of atom pair,
        where the keys are tuples of interacting bead types and the
        corresponding values define the interaction parameters. Must contain:

        .. code-block:: python

            tuple(*specific_types) : {
                "sigma" : torch.Tensor scalar that describes the distance at which 
                    the potential is zero.
                "epsilon" : torch.Tensor scalar that describes the well depth.
                "cutoff" : torch.Tensor scalar that describes the cutoff distance.
                ...
                }
        The keys must be tuples of 2 integer atom types.
    """

    name: Final[str] = "lennard_jones"
    _neighbor_list_name = "fully connected"

    def __init__(self, statistics: Dict) -> None:
        super(LennardJonesShifted, self).__init__()
        keys = torch.tensor(list(statistics.keys()), dtype=torch.long)
        self.allowed_interaction_keys = list(statistics.keys())
        self.order = 2
        self.name = self.name
        unique_types = torch.unique(keys.flatten())
        assert unique_types.min() >= 0
        max_type = unique_types.max()
        sizes = tuple([max_type + 1 for _ in range(self.order)])
        sigma = torch.zeros(sizes)
        epsilon = torch.zeros(sizes)
        cutoff = torch.zeros(sizes)
        for key in statistics.keys():
            # hardcode 0 for now
            sigma[key] = 0
            epsilon[key] = 0
            cutoff[key] = 0
        self.register_buffer("sigma", sigma)
        self.register_buffer("epsilon", epsilon)
        self.register_buffer("cutoff", cutoff)

    def data2features(self, data: AtomicData) -> torch.Tensor:
        """Computes pairwise distances from an AtomicData instance

        Parameters
        ----------
        data:
            Input `AtomicData` instance

        Returns
        -------
        torch.Tensor:
            Tensor of computed distances
        """
        mapping = data.neighbor_list[self.name]["index_mapping"]
        pbc = getattr(data, "pbc", None)
        cell = getattr(data, "cell", None)
        return LennardJonesShifted.compute_features(
            pos=data.pos,
            mapping=mapping,
            pbc=pbc,
            cell=cell,
            batch=data.batch,
        )

    def forward(self, data: AtomicData) -> AtomicData:
        """Forward pass through the Lennard-Jones interaction.

        Parameters
        ----------
        data:
            Input AtomicData instance with appropriate neighbor list

        Returns
        -------
        AtomicData:
            Updated AtomicData instance with the 'out' field containing
            predicted energies for each structure
        """
        mapping = data.neighbor_list[self.name]["index_mapping"]
        mapping_batch = data.neighbor_list[self.name]["mapping_batch"]
        interaction_types = [
            data.atom_types[mapping[ii]] for ii in range(self.order)
        ]
        features = self.data2features(data)
        y = LennardJonesShifted.compute(
            features,
            self.sigma[interaction_types],
            self.epsilon[interaction_types],
            self.cutoff[interaction_types]
        )
        y = scatter(y, mapping_batch, dim=0, reduce="sum")
        data.out[self.name] = {"energy": y}
        return data

    @staticmethod
    def compute_features(
        pos: torch.Tensor,
        mapping: torch.Tensor,
        pbc: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cell_shifts = _Prior._get_cell_shifts(pos, mapping, pbc, cell, batch)
        return compute_distances(
            pos=pos,
            mapping=mapping,
            cell_shifts=cell_shifts
        )

    @staticmethod
    def compute(x: torch.Tensor, sigma: torch.Tensor, epsilon: torch.Tensor, cutoff: torch.Tensor) -> torch.Tensor:
        """Compute the shifted Lennard-Jones potential

        Parameters
        ----------
        x : torch.Tensor
            Distances between particles
        sigma : torch.Tensor
            Distance at which potential is zero
        epsilon : torch.Tensor
            Well depth
        cutoff : torch.Tensor
            Cutoff distance

        Returns
        -------
        torch.Tensor
            Potential energy
        """
        # Compute mask for distances within cutoff
        mask = x < cutoff
        
        # Initialize energy tensor with zeros
        energy = torch.zeros_like(x)
        
        # Compute only for distances within cutoff
        r_scaled = sigma[mask] / x[mask]
        r6 = r_scaled ** 6
        r12 = r6 ** 2
        
        # Compute potential at cutoff for shifting
        rc_scaled = sigma[mask] / cutoff[mask]
        rc6 = rc_scaled ** 6
        rc12 = rc6 ** 2
        v_shift = 4.0 * epsilon[mask] * (rc12 - rc6)
        
        # Compute shifted potential
        energy[mask] = 4.0 * epsilon[mask] * (r12 - r6) - v_shift
        
        return energy

    @staticmethod
    def neighbor_list(topology: Topology) -> Dict:
        """Compute neighbor list from topology

        Parameters
        ----------
        topology:
            A Topology instance with defined fully-connected edges

        Returns
        -------
        Dict:
            Neighborlist of fully-connected distances
        """
        return {
            LennardJonesShifted.name: topology.neighbor_list(
                LennardJonesShifted._neighbor_list_name
            )
        }