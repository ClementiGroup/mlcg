import torch
from torch_scatter import scatter
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
from typing import Final, Optional, Dict
from math import pi
import numpy as np

from ..geometry.topology import Topology
from ..geometry.internal_coordinates import (
    compute_distances,
    compute_angles,
    compute_torsions,
)
from ..data.atomic_data import AtomicData

torch_pi = torch.tensor(pi)


class _Prior(object):
    """Abstract prior class"""

    def __init__(self) -> None:
        super(_Prior, self).__init__()

    @staticmethod
    def fit_from_values():
        raise NotImplementedError


class Harmonic(torch.nn.Module, _Prior):
    r"""1-D Harmonic prior interaction for feature :math:`x` of the form:

    .. math::

        U_{\text{Harmonic}}(x) = k\left( x - x_0 \right)^2

    where :math:`k` is a harmonic/spring constant describing the interaction
    strength and :math:`x_0` is the equilibrium value of the feature :math:`x`.
    A an optimizable constant energy offset is added during the prior parameter
    fitting.

    Parameters
    ----------
    statistics:
        Dictionary of interaction parameters for each type of atom pair/triple,
        where the keys are tuples of interacting bead types and the
        corresponding values define the interaction parameters. These
        Can be hand-designed or taken from the output of
        `mlcg.geometry.statistics.compute_statistics`, but must minimally
        contain the following information for each key:

        .. code-block:: python

            tuple(*specific_types) : {
                "k" : torch.Tensor scalar that describes the strength of the
                    harmonic interaction.
                "x_0" : torch.Tensor scalar that describes the mean feature
                    value.
                ...

                }

        The keys can be tuples of 2 or 3 atom type integers.
    """

    _order_map = {
        "bonds": 2,
        "angles": 3,
        "impropers": 4,
        "omega": 4,
        "gamma_1": 4,
        "gamma_2": 4,
        "dihedrals": 4,
    }
    _compute_map = {
        "bonds": compute_distances,
        "angles": compute_angles,
        "impropers": compute_torsions,
        "omega": compute_torsions,
        "gamma_1": compute_torsions,
        "gamma_2": compute_torsions,
        "dihedrals": compute_torsions,
    }

    def __init__(self, statistics: Dict, name: str) -> None:
        super(Harmonic, self).__init__()
        keys = torch.tensor(list(statistics.keys()), dtype=torch.long)
        self.allowed_interaction_keys = list(statistics.keys())
        self.name = name
        self.order = Harmonic._order_map[name]

        unique_types = torch.unique(keys.flatten())
        assert unique_types.min() >= 0
        max_type = unique_types.max()
        sizes = tuple([max_type + 1 for _ in range(self.order)])
        x_0 = torch.zeros(sizes)
        k = torch.zeros(sizes)
        for key in statistics.keys():
            x_0[key] = statistics[key]["x_0"]
            k[key] = statistics[key]["k"]

        self.register_buffer("x_0", x_0)
        self.register_buffer("k", k)

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
        return Harmonic.compute_features(data.pos, mapping, self.name)

    def data2parameters(self, data):
        mapping = data.neighbor_list[self.name]["index_mapping"]
        interaction_types = [
            data.atom_types[mapping[ii]] for ii in range(self.order)
        ]
        params = {
            "x0": self.x_0[interaction_types].flatten(),
            "k": self.k[interaction_types].flatten(),
        }
        params["V0"] = torch.zeros_like(params["x0"])
        return params

    def forward(self, data: AtomicData) -> AtomicData:
        """Forward pass through the harmonic interaction.

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
        mapping_batch = data.neighbor_list[self.name]["mapping_batch"]
        params = self.data2parameters(data)
        features = self.data2features(data).flatten()
        y = Harmonic.compute(features, **params)
        y = scatter(y, mapping_batch, dim=0, reduce="sum")
        data.out[self.name] = {"energy": y}
        return data

    @staticmethod
    def compute_features(pos, mapping, target):
        return Harmonic._compute_map[target](pos, mapping)

    @staticmethod
    def compute(x, x0, k, V0=0):
        return k * (x - x0) ** 2 + V0

    @staticmethod
    def fit_from_potential_estimates(
        bin_centers_nz: torch.Tensor, dG_nz: torch.Tensor
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

        # remove noise by discarding signals
        integral = torch.tensor(
            float(trapezoid(dG_nz.cpu().numpy(), bin_centers_nz.cpu().numpy()))
        )

        mask = torch.abs(dG_nz) > 1e-4 * torch.abs(integral)
        try:
            popt, _ = curve_fit(
                Harmonic.compute,
                bin_centers_nz[mask],
                dG_nz[mask],
                p0=[bin_centers_nz[torch.argmin(dG_nz[mask])], 60, -1],
            )
            stat = {"k": popt[1], "x_0": popt[0]}
        except:
            print(f"failed to fit potential estimate for Harmonic")
            stat = {
                "k": torch.tensor(float("nan")),
                "x_0": torch.tensor(float("nan")),
            }
        return stat

    @staticmethod
    def neighbor_list(topology: Topology, type: str) -> Dict:
        """Method for computing a neighbor list from a topology
        and a chosen feature type.

        Parameters
        ----------
        topology:
            A Topology instance with defined features relevant to the
            feature type chosen for the neighbor list.
        type:
            A string describing the type of features. Must be one of
            :code:`["bonds", "angles"]`

        Returns
        -------
        Dict:
            Neighborlist of the chosen feature according to the
            supplied topology
        """

        assert type in Harmonic._compute_map
        nl = topology.neighbor_list(type)
        return {type: nl}


class HarmonicBonds(Harmonic):
    """Wrapper class for quickly computing bond priors
    (order 2 Harmonic priors)
    """

    name: Final[str] = "bonds"
    _order = 2

    def __init__(self, statistics) -> None:
        super(HarmonicBonds, self).__init__(statistics, HarmonicBonds.name)

    @staticmethod
    def neighbor_list(topology: Topology) -> dict:
        return Harmonic.neighbor_list(topology, HarmonicBonds.name)

    @staticmethod
    def compute_features(pos, mapping):
        return Harmonic.compute_features(pos, mapping, HarmonicBonds.name)


class HarmonicAngles(Harmonic):
    r"""Wrapper class for quickly computing angle priors
    (order 3 Harmonic priors)

    To avoid numerical instabilities, we use a functional form of the following way:

    .. math::

        U_{\text{HarmonicAngles}}(\theta) = k\left( \cos{\theta} - \cos{\theta}_0 \right)^2

    where :math:`\theta_0` is the value of the angle at equilibrium.

    """

    name: Final[str] = "angles"
    _order = 3

    def __init__(self, statistics) -> None:
        super(HarmonicAngles, self).__init__(statistics, HarmonicAngles.name)

    @staticmethod
    def neighbor_list(topology: Topology) -> dict:
        return Harmonic.neighbor_list(topology, HarmonicAngles.name)

    @staticmethod
    def compute_features(pos, mapping):
        return Harmonic.compute_features(pos, mapping, HarmonicAngles.name)


class HarmonicImpropers(Harmonic):
    name: Final[str] = "impropers"
    _order = 4

    def __init__(self, statistics) -> None:
        super(HarmonicImpropers, self).__init__(
            statistics, HarmonicImpropers.name
        )

    @staticmethod
    def neighbor_list(topology: Topology) -> dict:
        return Harmonic.neighbor_list(topology, HarmonicImpropers.name)

    @staticmethod
    def compute_features(pos, mapping):
        return Harmonic.compute_features(pos, mapping, HarmonicImpropers.name)


class ShiftedPeriodicHarmonicImpropers(Harmonic):
    """HarmonicImproper that can handle discontinuities around the
    pi modulus, eg, for Omega dihedral angles or distributions with this form:


            ###                                   ###
            ###                                   ###
            ###                                   ###
            ####                                  ###
            ####                                  ###
            #####                                ####
            ######                             ,#####
             -pi                0                 pi


    Internally these features are computed such that angles
    lower than zero are phase shifted by 2pi, and then pi is subtracted from
    the resulting distribution to get:

                              #####
                             #######
                            ########.
                           /#########,
                           ###########
                           ###########
                          #############
             -pi                0                 pi

    where the harmonic parameters are fitted over the free energy associated
    with the latter , tranformed distribution.

    """

    name: Final[str] = "impropers"
    _order = 4

    def __init__(self, statistics) -> None:
        super(ShiftedPeriodicHarmonicImpropers, self).__init__(
            statistics, ShiftedPeriodicHarmonicImpropers.name
        )

    @staticmethod
    def neighbor_list(topology: Topology) -> dict:
        return Harmonic.neighbor_list(topology, HarmonicImpropers.name)

    @staticmethod
    def compute_features(pos, mapping):
        # features should be between -pi and pi after data2features()
        # Here, we conditionally shift angles in (-pi, 0) to (pi, 2pi)
        # Then subtract pi in order to center the distribution at 0
        features = Harmonic.compute_features(
            pos, mapping, HarmonicImpropers.name
        )
        features = (
            torch.where(features < 0, features + 2 * torch_pi, features)
            - torch_pi
        )
        return features

    def data2features(self, data):
        mapping = data.neighbor_list[self.name]["index_mapping"]
        return ShiftedPeriodicHarmonicImpropers.compute_features(
            data.pos, mapping
        )

    def forward(self, data):
        mapping_batch = data.neighbor_list[self.name]["mapping_batch"]
        params = self.data2parameters(data)
        features = self.data2features(data).flatten()

        y = Harmonic.compute(features, **params)
        y = scatter(y, mapping_batch, dim=0, reduce="sum")
        data.out[self.name] = {"energy": y}
        return data


class Repulsion(torch.nn.Module, _Prior):
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
        """Method for fitting interaction parameters from data

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


class GeneralBonds(Harmonic):
    """Generalized Bonds"""

    name: Final[str] = "bonds"
    _order = 2

    def __init__(self, statistics, name) -> None:
        super(GeneralBonds, self).__init__(statistics, HarmonicBonds.name)
        self.name = name

    def data2features(self, data):
        mapping = data.neighbor_list[self.name]["index_mapping"]
        return Harmonic.compute_features(data.pos, mapping, "bonds")

    @staticmethod
    def compute_features(pos, mapping):
        return Harmonic.compute_features(pos, mapping, "bonds")


class GeneralAngles(Harmonic):
    """Generalized Angles"""

    name: Final[str] = "angles"
    _order = 3

    def __init__(self, statistics, name) -> None:
        super(GeneralAngles, self).__init__(statistics, HarmonicAngles.name)
        self.name = name

    def data2features(self, data):
        mapping = data.neighbor_list[self.name]["index_mapping"]
        return Harmonic.compute_features(data.pos, mapping, "angles")

    @staticmethod
    def compute_features(pos, mapping):
        return Harmonic.compute_features(pos, mapping, "angles")


class Dihedral(torch.nn.Module, _Prior):
    r"""
    Prior that constrains dihedral planar angles using
    the following energy ansatz:

    .. math::

        V(\theta) = v_0 + \sum_{n=1}^{n_{deg}} k1_n \sin{(n\theta)} + k2_n\cos{(n\theta)}

    where :math:`n_{deg}` is the maximum number of terms to take in the sinusoidal series,
    :math:`v_0` is a constant offset, and :math:`k1_n` and :math:`k2_n` are coefficients
    for each term number :math:`n`.

    Parameters
    ----------
    statistics:
        Dictionary of interaction parameters for each type of atom quadruple,
        where the keys are tuples of interacting bead types and the
        corresponding values define the interaction parameters. These
        Can be hand-designed or taken from the output of
        `mlcg.geometry.statistics.compute_statistics`, but must minimally
        contain the following information for each key:

        .. code-block:: python

            tuple(*specific_types) : {
                "k1s" : torch.Tensor that contains all k1 coefficients
                "k2s" : torch.Tensor that contains all k2 coefficients
                "v_0" : torch.Tensor that contains the constant offset
                ...
                }

        The keys must be tuples of 4 atoms.
    """

    name: Final[str] = "dihedrals"
    _order = 4
    _neighbor_list_name = "dihedrals"

    def __init__(self, statistics: Dict, n_degs: int = 6) -> None:
        super(Dihedral, self).__init__()
        keys = torch.tensor(list(statistics.keys()), dtype=torch.long)
        self.allowed_interaction_keys = list(statistics.keys())
        self.order = self._order
        self.name = self.name
        unique_types = torch.unique(keys.flatten())
        assert unique_types.min() >= 0
        max_type = unique_types.max()
        sizes = tuple([max_type + 1 for _ in range(self.order)])
        # In principle we could extend this to include even more wells if needed.
        self.n_degs = n_degs
        self.k1_names = ["k1_" + str(ii) for ii in range(1, self.n_degs + 1)]
        self.k2_names = ["k2_" + str(ii) for ii in range(1, self.n_degs + 1)]
        k1 = torch.zeros(self.n_degs, *sizes)
        k2 = torch.zeros(self.n_degs, *sizes)
        v_0 = torch.zeros(*sizes)

        for key in statistics.keys():
            for ii in range(self.n_degs):
                k1_name = self.k1_names[ii]
                k2_name = self.k2_names[ii]
                k1[ii][key] = statistics[key]["k1s"][k1_name]
                k2[ii][key] = statistics[key]["k2s"][k2_name]
            v_0[key] = statistics[key]["v_0"]
        self.register_buffer("k1s", k1)
        self.register_buffer("k2s", k2)
        self.register_buffer("v_0", v_0)

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
        return Dihedral.compute_features(data.pos, mapping)

    def forward(self, data: AtomicData) -> AtomicData:
        """Forward pass through the dihedral interaction.
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

        mapping_batch = data.neighbor_list[self.name]["mapping_batch"]

        features = self.data2features(data).flatten()
        params = self.data2parameters(data)
        y = Dihedral.compute(features, **params)
        y = scatter(y, mapping_batch, dim=0, reduce="sum")
        data.out[self.name] = {"energy": y}
        return data

    def data2parameters(self, data: AtomicData) -> Dict:
        mapping = data.neighbor_list[self.name]["index_mapping"]
        interaction_types = [
            data.atom_types[mapping[ii]] for ii in range(self.order)
        ]
        # the parameters have shape n_features x n_degs
        k1s = torch.vstack(
            [self.k1s[ii][interaction_types] for ii in range(self.n_degs)]
        ).t()
        k2s = torch.vstack(
            [self.k2s[ii][interaction_types] for ii in range(self.n_degs)]
        ).t()
        v_0 = self.v_0[interaction_types].view(-1, 1)
        return {"k1s": k1s, "k2s": k2s, "v_0": v_0}

    @staticmethod
    def compute_features(
        pos: torch.Tensor, mapping: torch.Tensor
    ) -> torch.Tensor:
        return compute_torsions(pos, mapping)

    @staticmethod
    def wrapper_fit_func(theta: torch.Tensor, *args) -> torch.Tensor:
        args = args[0]
        v_0 = torch.tensor(args[0])
        k_args = args[1:]
        num_ks = len(k_args) // 2
        k1s, k2s = k_args[:num_ks], k_args[num_ks:]
        k1s = torch.tensor(k1s).view(-1, num_ks)
        k2s = torch.tensor(k2s).view(-1, num_ks)
        return Dihedral.compute(theta, v_0, k1s, k2s)

    @staticmethod
    def compute(
        theta: torch.Tensor,
        v_0: torch.Tensor,
        k1s: torch.Tensor,
        k2s: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the dihedral interaction for a list of angles and models
        parameters. The ineraction is computed as a sin/cos basis expansion up
        to N basis functions.

        Parameters
        ----------
        theta :
            angles to compute the value of the dihedral interaction on
        v_0 :
            constant offset
        k1s :
            list of sin parameters
        k2s :
            list of cos parameters
        Returns
        -------
        torch.Tensor:
            Dihedral interaction energy
        """
        _, n_k = k1s.shape
        n_degs = torch.arange(
            1, n_k + 1, dtype=theta.dtype, device=theta.device
        )
        # expand the features w.r.t the mult integer so that it has the
        # shape of k1s and k2s
        angles = theta.view(-1, 1) * n_degs.view(1, -1)
        V = k1s * torch.sin(angles) + k2s * torch.cos(angles)
        # HOTFIX to avoid shape mismatch when using specialized priors
        # TODO: think of a better fix
        if v_0.ndim > 1:
            v_0 = v_0[:, 0]

        return V.sum(dim=1) + v_0

    @staticmethod
    def neg_log_likelihood(y, yhat):
        """
        Convert dG to probability and use KL divergence to get difference between
        predicted and actual
        """
        L = torch.sum(torch.exp(-y) * torch.log(torch.exp(-yhat)))
        return -L

    @staticmethod
    def _init_parameters(n_degs):
        """Helper method for guessing initial parameter values"""
        p0 = [1.00]  # start with constant offset
        k1s_0 = [1 for _ in range(n_degs)]
        k2s_0 = [1 for _ in range(n_degs)]
        p0.extend(k1s_0)
        p0.extend(k2s_0)
        return p0

    @staticmethod
    def _init_parameter_dict(n_degs):
        """Helper method for initializing the parameter dictionary"""
        stat = {"k1s": {}, "k2s": {}, "v_0": 0.00}
        k1_names = ["k1_" + str(ii) for ii in range(1, n_degs + 1)]
        k2_names = ["k2_" + str(ii) for ii in range(1, n_degs + 1)]
        for ii in range(n_degs):
            k1_name = k1_names[ii]
            k2_name = k2_names[ii]
            stat["k1s"][k1_name] = {}
            stat["k2s"][k2_name] = {}
        return stat

    @staticmethod
    def _make_parameter_dict(stat, popt, n_degs):
        """Helper method for constructing a fitted parameter dictionary"""
        v_0 = popt[0]
        k_popt = popt[1:]
        num_k1s = int(len(k_popt) / 2)
        k1_names = sorted(list(stat["k1s"].keys()))
        k2_names = sorted(list(stat["k2s"].keys()))
        for ii in range(n_degs):
            k1_name = k1_names[ii]
            k2_name = k2_names[ii]
            stat["k1s"][k1_name] = {}
            stat["k2s"][k2_name] = {}
            if len(k_popt) > 2 * ii:
                stat["k1s"][k1_name] = k_popt[ii]
                stat["k2s"][k2_name] = k_popt[num_k1s + ii]
            else:
                stat["k1s"][k1_name] = 0
                stat["k2s"][k2_name] = 0
        stat["v_0"] = v_0
        return stat

    @staticmethod
    def _compute_adjusted_R2(
        bin_centers_nz, dG_nz, mask, popt, free_parameters
    ):
        """
        Method for model selection using adjusted R2
        Higher values imply better model selection
        """
        dG_fit = Dihedral.wrapper_fit_func(bin_centers_nz[mask], *[popt])
        SSres = torch.sum(torch.square(dG_nz[mask] - dG_fit))
        SStot = torch.sum(torch.square(dG_nz[mask] - torch.mean(dG_nz[mask])))
        n_samples = len(dG_nz[mask])
        R2 = 1 - (SSres / (n_samples - free_parameters - 1)) / (
            SStot / (n_samples - 1)
        )
        return R2

    @staticmethod
    def _compute_aic(bin_centers_nz, dG_nz, mask, popt, free_parameters):
        """Method for computing the AIC"""
        aic = (
            2
            * Dihedral.neg_log_likelihood(
                dG_nz[mask],
                Dihedral.wrapper_fit_func(bin_centers_nz[mask], *[popt]),
            )
            + 2 * free_parameters
        )
        return aic

    @staticmethod
    def _linear_regression(bin_centers, targets, n_degs):
        """Vanilla linear regression"""
        features = [torch.ones_like(bin_centers)]
        for n in range(n_degs):
            features.append(torch.sin((n + 1) * bin_centers))
        for n in range(n_degs):
            features.append(torch.cos((n + 1) * bin_centers))
        features = torch.stack(features).t()
        targets = targets.to(features.dtype)
        sol = torch.linalg.lstsq(features, targets.t())
        return sol

    @staticmethod
    def fit_from_potential_estimates(
        bin_centers_nz: torch.Tensor,
        dG_nz: torch.Tensor,
        n_degs: int = 6,
        constrain_deg: Optional[int] = None,
        regression_method: str = "linear",
        metric: str = "aic",
    ) -> Dict:
        """
        Loop over n_degs basins and use either the AIC criterion
        or a prechosen degree to select best fit. Parameter fitting
        occurs over unmaksed regions of the free energy only.
        Parameters
        ----------
        bin_centers_nz:
            Bin centers over which the fit is carried out
        dG_nz:
            The emperical free energy correspinding to the bin centers
        n_degs:
            The maximum number of degrees to attempt to fit if using the AIC
            criterion for prior model selection
        constrain_deg:
            If not None, a single fit is produced for the specified integer
            degree instead of using the AIC criterion for fit selection between
            multiple degrees
        regression_method:
            String specifying which regression method to use. If "nonlinear",
            the default `scipy.optimize.curve_fit` method is used. If 'linear',
            linear regression via `torch.linalg.lstsq` is used
        metric:
            If a constrain deg is not specified, this string specifies whether to
            use either AIC ('aic') or adjusted R squared ('r2') for automated degree
            selection. If the automatic degree determination fails, users should
            consider searching for a proper constrained degree.

        Returns
        -------
        Dict:
            Statistics dictionary with fitted interaction parameters
        """

        integral = torch.tensor(
            float(trapezoid(dG_nz.cpu().numpy(), bin_centers_nz.cpu().numpy()))
        )

        mask = torch.abs(dG_nz) > 1e-4 * torch.abs(integral)

        if constrain_deg != None:
            assert isinstance(constrain_deg, int)
            stat = Dihedral._init_parameter_dict(constrain_deg)
            if regression_method == "linear":
                popt = (
                    Dihedral._linear_regression(
                        bin_centers_nz[mask], dG_nz[mask], constrain_deg
                    )
                    .solution.numpy()
                    .tolist()
                )
            elif regression_method == "nonlinear":
                p0 = Dihedral._init_parameters(constrain_deg)
                popt, _ = curve_fit(
                    lambda theta, *p0: Dihedral.wrapper_fit_func(theta, p0),
                    bin_centers_nz[mask],
                    dG_nz[mask],
                    p0=p0,
                )
            else:
                raise ValueError(
                    "regression method {} is neither 'linear' nor 'nonlinear'".format(
                        regression_method
                    )
                )
            stat = Dihedral._make_parameter_dict(stat, popt, constrain_deg)

        else:
            if metric == "aic":
                metric_func = Dihedral._compute_aic
                best_func = min
            elif metric == "r2":
                metric_func = Dihedral._compute_adjusted_R2
                best_func = max
            else:
                raise ValueError(
                    "metric {} is neither 'aic' nor 'r2'".format(metric)
                )

            # Determine best fit for unknown # of parameters
            stat = Dihedral._init_parameter_dict(n_degs)
            popts = []
            metric_vals = []

            try:
                for deg in range(1, n_degs + 1):
                    free_parameters = 1 + (2 * deg)
                    if regression_method == "linear":
                        popt = (
                            Dihedral._linear_regression(
                                bin_centers_nz[mask], dG_nz[mask], deg
                            )
                            .solution.numpy()
                            .tolist()
                        )
                    elif regression_method == "nonlinear":
                        p0 = Dihedral._init_parameters(deg)
                        popt, _ = curve_fit(
                            lambda theta, *p0: Dihedral.wrapper_fit_func(
                                theta, p0
                            ),
                            bin_centers_nz[mask],
                            dG_nz[mask],
                            p0=p0,
                        )
                    else:
                        raise ValueError(
                            "regression method {} is neither 'linear' nor 'nonlinear'".format(
                                regression_method
                            )
                        )
                    metric_val = metric_func(
                        bin_centers_nz, dG_nz, mask, popt, free_parameters
                    )
                    popts.append(popt)
                    metric_vals.append(metric_val)
                best_val = best_func(metric_vals)
                best_i_val = metric_vals.index(best_val)
                popt = popts[best_i_val]
                stat = Dihedral._make_parameter_dict(stat, popt, n_degs)
            except:
                print(f"failed to fit potential estimate for Dihedral")
                stat = Dihedral._init_parameter_dict(n_degs)
                k1_names = sorted(list(stat["k1s"].keys()))
                k2_names = sorted(list(stat["k2s"].keys()))
                for ii in range(n_degs):
                    k1_name = k1_names[ii]
                    k2_name = k2_names[ii]
                    stat["k1s"][k1_name] = torch.tensor(float("nan"))
                    stat["k2s"][k2_name] = torch.tensor(float("nan"))
        return stat

    def from_user(*args):
        """
        Direct input of parameters from user. Leave empty for now
        """
        raise NotImplementedError()

    @staticmethod
    def neighbor_list(topology) -> None:
        nl = topology.neighbor_list(Dihedral.name)
        return {Dihedral.name: nl}


class Quartic(torch.nn.Module, _Prior):
    r"""
    Prior that helps in fitting tighter bimodal distributions
    using the following energy ansatz.


    .. math:

        V(x) = a*(x-xa)**2 + b*(x-xb)**3 + c*(x-xc)**4 + d

    N.B. the linear term is missing
    Especially useful for CA angles, to restrain them
    avoiding exploration toward pi
    """

    _order_map = {
        "bonds": 2,
        "angles": 3,
        "dihedrals": 4,
    }
    _compute_map = {
        "bonds": compute_distances,
        "angles": compute_angles,
        "dihedrals": compute_torsions,
    }
    _neighbor_list_map = {
        "bonds": "bonds",
        "angles": "angles",
        "dihedrals": "dihedrals",
    }

    def __init__(
        self, statistics, name, order: Optional[int] = None, n_degs: int = 4
    ) -> None:
        super(Quartic, self).__init__()
        keys = torch.tensor(list(statistics.keys()), dtype=torch.long)
        self.allowed_interaction_keys = list(statistics.keys())
        self.name = name
        if order is not None:
            self.order = order
        elif name in Quartic._order_map.keys():
            self.order = Quartic._order_map[self.name]
        else:
            raise ValueError(f"Uncompatible order {order}")
        self.neighbor_list_type = Quartic._neighbor_list_map[self.name]

        unique_types = torch.unique(keys.flatten())
        assert unique_types.min() >= 0
        max_type = unique_types.max()
        sizes = tuple([max_type + 1 for _ in range(self.order)])

        self.n_degs = n_degs
        self.k_names = ["k_" + str(ii) for ii in range(2, self.n_degs + 1)]
        self.x0_names = ["x0_" + str(ii) for ii in range(2, self.n_degs + 1)]

        k = torch.zeros(self.n_degs - 1, *sizes)
        x0 = torch.zeros(self.n_degs - 1, *sizes)
        v_0 = torch.zeros(*sizes)

        for key in statistics.keys():
            for ii in range(self.n_degs - 1):
                k_name = self.k_names[ii]
                x0_name = self.x0_names[ii]
                k[ii][key] = statistics[key]["ks"][k_name]
                x0[ii][key] = statistics[key]["x0s"][x0_name]
            v_0[key] = statistics[key]["v_0"]
        self.register_buffer("ks", k)
        self.register_buffer("v_0", v_0)
        self.register_buffer("x0s", x0)

    @staticmethod
    def compute_features(pos, mapping, target):
        compute_map_type = Quartic._neighbor_list_map[target]
        return Quartic._compute_map[compute_map_type](pos, mapping)

    def data2features(self, data):
        mapping = data.neighbor_list[self.name]["index_mapping"]
        return Quartic.compute_features(data.pos, mapping, self.name)

    def data2parameters(self, data):
        mapping = data.neighbor_list[self.name]["index_mapping"]
        interaction_types = [
            data.atom_types[mapping[ii]] for ii in range(self.order)
        ]
        # the parameters have shape n_features x n_degs-1 since
        # linear term is missing
        ks = torch.vstack(
            [self.ks[ii][interaction_types] for ii in range(self.n_degs - 1)]
        ).t()

        x0s = torch.vstack(
            [self.x0s[ii][interaction_types] for ii in range(self.n_degs - 1)]
        ).t()
        v_0s = self.v_0[interaction_types].t()
        return {"ks": ks, "x0s": x0s, "v_0s": v_0s}

    def forward(self, data):
        mapping_batch = data.neighbor_list[self.name]["mapping_batch"]
        features = self.data2features(data).flatten()
        params = self.data2parameters(data)
        V0s = params["v_0s"].t()
        ks = params["ks"].t()
        x0s = params["x0s"].t()
        y = Quartic.compute(
            features,
            ks,
            V0s,
            x0s,
        )
        y = scatter(y, mapping_batch, dim=0, reduce="sum")
        data.out[self.name] = {"energy": y}
        return data

    @staticmethod
    def compute(
        x: torch.Tensor, ks: torch.Tensor, V0: torch.Tensor, x0s: torch.Tensor
    ):
        """Quartic potential interaction with missing linear term.

        .. math:

            V(r) = V0 + \sum_{n=2}^{4} k_n (x-x_n)^n

        """
        V = 0
        for i in range(3):
            V += ks[i] * (x - x0s[i]) ** (i + 2)

        V += V0
        return V

    @staticmethod
    def _quartic_model(x, a, b, c, d, xa, xb, xc):
        return a * (x - xa) ** 2 + b * (x - xb) ** 3 + c * (x - xc) ** 4 + d

    @staticmethod
    def _init_quartic_parameters(n_degs):
        """
        Helper method for guessing initial parameter values
        Not used for now
        """
        ks = [1.0 for _ in range(n_degs - 1)]
        x0s = [0.0 for _ in range(n_degs - 1)]
        V0 = -1.0
        p0 = [V0]
        p0.extend(ks)
        p0.extend(x0s)
        return p0

    @staticmethod
    def _init_quartic_parameter_dict(n_degs):
        """Helper method for initializing the parameter dictionary"""
        stat = {"ks": {}, "x0s": {}, "v_0": 0.0}
        k_names = ["k_" + str(ii) for ii in range(2, n_degs + 1)]
        x0_names = ["x0_" + str(ii) for ii in range(2, n_degs + 1)]
        for ii in range(n_degs - 1):
            k_name = k_names[ii]
            x0_name = x0_names[ii]
            stat["ks"][k_name] = {}
            stat["x0s"][x0_name] = {}
        return stat

    @staticmethod
    def _make_quartic_dict(stat, popt, n_degs):
        """Helper method for constructing a fitted parameter dictionary"""
        stat["v_0"] = popt[0]
        k_names = sorted(list(stat["ks"].keys()))
        x0_names = sorted(list(stat["x0s"].keys()))
        for ii in range(n_degs - 1):
            k_name = k_names[ii]
            x0_name = x0_names[ii]
            stat["ks"][k_name] = popt[ii]
            stat["x0s"][x0_name] = popt[ii + n_degs]

        return stat

    @staticmethod
    def fit_quartic_from_potential_estimates(
        bin_centers_nz: torch.Tensor,
        dG_nz: torch.Tensor,
        **kwargs,
    ):
        """

        Parameters
        ----------
        bin_centers_nz:
            Bin centers over which the fit is carried out
        dG_nz:
            The free energy values correspinding to the bin centers

        Returns
        -------
            Statistics dictionary with fitted quartic parameters
        """
        n_degs = 4

        integral = torch.tensor(
            float(trapezoid(dG_nz.cpu().numpy(), bin_centers_nz.cpu().numpy()))
        )

        mask = torch.abs(dG_nz) > 1e-4 * torch.abs(integral)
        try:
            stat = Quartic._init_quartic_parameter_dict(n_degs)
            popt, _ = curve_fit(
                Quartic._quartic_model,
                bin_centers_nz.cpu().numpy()[mask],
                dG_nz.cpu().numpy()[mask],
                p0=[1, 0, 0, torch.argmin(dG_nz[mask]), 0, 0, 0],
                bounds=(
                    (0, 0, 0, -np.inf, -np.pi, -np.pi, -np.pi),
                    (np.inf, np.inf, np.inf, np.inf, np.pi, np.pi, np.pi),
                ),
                maxfev=5000,
            )
            stat = Quartic._make_quartic_dict(stat, popt, n_degs)
        except:
            print(f"failed to fit potential estimate for the prior Quartic")
            stat = Quartic._init_quartic_parameter_dict(n_degs)
            k_names = sorted(list(stat["ks"].keys()))
            x_0_names = sorted(list(stat["x0s"].keys()))
            for ii in range(n_degs - 1):
                k1_name = k_names[ii]
                k2_name = x_0_names[ii]
                stat["ks"][k1_name] = torch.tensor(float("nan"))
                stat["x0s"][k2_name] = torch.tensor(float("nan"))

        return stat
