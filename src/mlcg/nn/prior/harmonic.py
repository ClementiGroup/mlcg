from math import pi
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
import torch
from torch_geometric.utils import scatter
from typing import Final, Dict


from .base import _Prior
from ...data.atomic_data import AtomicData
from ...geometry.topology import Topology
from ...geometry.internal_coordinates import (
    compute_distances,
    compute_angles_cos,
    compute_angles_raw,
    compute_torsions,
)

torch_pi = torch.tensor(pi)


class Harmonic(_Prior):
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
                "k" : #torch.Tensor scalar that describes the strength of the
                    #harmonic interaction.
                "x_0" : #torch.Tensor scalar that describes the mean feature
                    #value.
                ...

                }

        The keys can be tuples of 2 or 3 atom type integers.
    """

    def __init__(self, statistics: Dict, name: str, order: int) -> None:
        super(Harmonic, self).__init__()
        keys = torch.tensor(list(statistics.keys()), dtype=torch.long)
        self.allowed_interaction_keys = list(statistics.keys())
        self.name = name
        self.order = order

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

    def data2parameters(self, data: AtomicData):
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

        nl = topology.neighbor_list(type)
        return {type: nl}


class HarmonicBonds(Harmonic):
    """Wrapper class for quickly computing bond priors
    (order 2 Harmonic priors)
    """

    name: Final[str] = "bonds"
    _order = 2

    def __init__(self, statistics) -> None:
        super(HarmonicBonds, self).__init__(
            statistics, HarmonicBonds.name, order=2
        )

    def data2features(self, data: AtomicData) -> torch.Tensor:
        mapping = data.neighbor_list[self.name]["index_mapping"]
        return self.compute_features(data.pos, mapping)

    @staticmethod
    def neighbor_list(topology: Topology) -> dict:
        return Harmonic.neighbor_list(topology, HarmonicBonds.name)

    @staticmethod
    def compute_features(
        pos: torch.Tensor, mapping: torch.Tensor
    ) -> torch.Tensor:
        return compute_distances(pos, mapping)


class HarmonicAngles(Harmonic):
    r"""Wrapper class for quickly computing angle priors
    (order 3 Harmonic priors)

    To avoid numerical instabilities, we use a functional form of the following way:

    .. math::

        U_{\text{HarmonicAngles}}(\theta) = k\left( \cos{(\theta)} - \cos{(\theta_0)} \right)^2

    where :math:`\theta_0` is the value of the angle at equilibrium.

    """

    name: Final[str] = "angles"
    _order = 3

    def __init__(self, statistics: Dict) -> None:
        super(HarmonicAngles, self).__init__(
            statistics, HarmonicAngles.name, order=3
        )

    def data2features(self, data: torch.Tensor) -> torch.Tensor:
        mapping = data.neighbor_list[self.name]["index_mapping"]
        return self.compute_features(data.pos, mapping)

    @staticmethod
    def neighbor_list(topology: Topology) -> dict:
        return Harmonic.neighbor_list(topology, HarmonicAngles.name)

    @staticmethod
    def compute_features(
        pos: torch.Tensor, mapping: torch.Tensor
    ) -> torch.Tensor:
        return compute_angles_cos(pos, mapping)


class HarmonicAnglesRaw(Harmonic):
    r"""Wrapper class for quickly computing angle priors
    (order 3 Harmonic priors)

    .. math::

        \theta_{ijk} = &\text{atan2}(\Vert \hat{\mathbf{n}} \vert, \mathbf{r}_{ij} \cdot \mathbf{r}_{kj} ) \\
        \mathbf{r}_{ij} &= \mathbf{r}_i - \mathbf{r}_j \\
        \mathbf{r}_{kj} &= \mathbf{r}_k - \mathbf{r}_j \\
        \mathbf{\hat{n}} &= \frac{\mathbf{r}_{ij} \times \mathbf{r}_{kj}}{\Vert \mathbf{r}_{ij} \times \mathbf{r}_{kj} \Vert} 

    """

    name: Final[str] = "angles"
    _order = 3

    def __init__(self, statistics, name) -> None:
        super(HarmonicAnglesRaw, self).__init__(
            statistics, HarmonicAnglesRaw.name
        )
        self.name = name

    @staticmethod
    def neighbor_list(topology: Topology) -> dict:
        return Harmonic.neighbor_list(topology, HarmonicAnglesRaw.name)

    def data2features(self, data):
        mapping = data.neighbor_list[self.name]["index_mapping"]
        return HarmonicAnglesRaw.compute_features(data.pos, mapping)

    @staticmethod
    def compute_features(pos, mapping):
        return compute_angles_raw(pos, mapping)


class HarmonicImpropers(Harmonic):
    name: Final[str] = "impropers"
    _order = 4

    def __init__(self, statistics) -> None:
        super(HarmonicImpropers, self).__init__(
            statistics, HarmonicImpropers.name, order=HarmonicImpropers._order
        )

    @staticmethod
    def data2features(self, data: torch.Tensor) -> torch.Tensor:
        mapping = data.neighbor_list[self.name]["index_mapping"]
        return self.compute_features(data.pos, mapping)

    @staticmethod
    def neighbor_list(topology: Topology) -> dict:
        return Harmonic.neighbor_list(topology, HarmonicImpropers.name)

    @staticmethod
    def compute_features(
        pos: torch.Tensor, mapping: torch.Tensor
    ) -> torch.Tensor:
        return compute_torsions(pos, mapping)


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
            statistics,
            ShiftedPeriodicHarmonicImpropers.name,
            order=ShiftedPeriodicHarmonicImpropers._order,
        )

    @staticmethod
    def neighbor_list(topology: Topology) -> dict:
        return Harmonic.neighbor_list(topology, HarmonicImpropers.name)

    @staticmethod
    def compute_features(
        pos: torch.Tensor, mapping: torch.Tensor
    ) -> torch.Tensor:
        # features should be between -pi and pi after data2features()
        # Here, we conditionally shift angles in (-pi, 0) to (pi, 2pi)
        # Then subtract pi in order to center the distribution at 0
        features = compute_torsions(pos, mapping)
        features = (
            torch.where(features < 0, features + 2 * torch_pi, features)
            - torch_pi
        )
        return features

    def data2features(self, data: AtomicData) -> torch.Tensor:
        mapping = data.neighbor_list[self.name]["index_mapping"]
        return ShiftedPeriodicHarmonicImpropers.compute_features(
            data.pos, mapping
        )

    def forward(self, data: AtomicData) -> AtomicData:
        mapping_batch = data.neighbor_list[self.name]["mapping_batch"]
        params = self.data2parameters(data)
        features = self.data2features(data).flatten()

        y = Harmonic.compute(features, **params)
        y = scatter(y, mapping_batch, dim=0, reduce="sum")
        data.out[self.name] = {"energy": y}
        return data


class GeneralBonds(Harmonic):
    """Generalized Bonds"""

    name: Final[str] = "bonds"
    _order = 2

    def __init__(self, statistics, name) -> None:
        super(GeneralBonds, self).__init__(
            statistics, HarmonicBonds.name, order=GeneralBonds._order
        )
        self.name = name

    def data2features(self, data: AtomicData) -> torch.Tensor:
        mapping = data.neighbor_list[self.name]["index_mapping"]
        return compute_distances(data.pos, mapping)

    @staticmethod
    def compute_features(
        pos: torch.Tensor, mapping: torch.Tensor
    ) -> torch.Tensor:
        return compute_distances(pos, mapping)


class GeneralAngles(Harmonic):
    """Generalized Angles"""

    name: Final[str] = "angles"
    _order = 3

    def __init__(self, statistics, name) -> None:
        super(GeneralAngles, self).__init__(
            statistics, HarmonicAngles.name, order=GeneralAngles._order
        )
        self.name = name

    def data2features(self, data: torch.Tensor) -> torch.Tensor:
        mapping = data.neighbor_list[self.name]["index_mapping"]
        return compute_angles_cos(data.pos, mapping)

    @staticmethod
    def compute_features(
        pos: torch.Tensor, mapping: torch.Tensor
    ) -> torch.Tensor:
        return compute_angles_cos(pos, mapping)
