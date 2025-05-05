from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
import torch
from torch_scatter import scatter
from typing import Optional, Dict, Final


from .base import _Prior
from ...data.atomic_data import AtomicData
from ...geometry.topology import Topology
from ...geometry.internal_coordinates import (
    compute_torsions,
)


class FourierSeries(torch.nn.Module, _Prior):
    r"""
    Prior class representing a fourier series of a periodic variable $\theta$.
    The energy computed is given by the follwoing function

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

        The keys must be tuples of `order` atoms.
    """

    def __init__(
        self, statistics: Dict, name: str = "", n_degs: int = 6, order: int = 4
    ) -> None:
        super(FourierSeries, self).__init__()
        keys = torch.tensor(list(statistics.keys()), dtype=torch.long)
        self.allowed_interaction_keys = list(statistics.keys())
        self.order = order
        self.name = name
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
        return self.compute_features(data.pos, mapping)

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
        y = FourierSeries.compute(features, **params)
        y = scatter(y, mapping_batch, dim=0, reduce="sum")
        data.out[self.name] = {"energy": y}
        return data

    @staticmethod
    def wrapper_fit_func(theta: torch.Tensor, *args) -> torch.Tensor:
        args = args[0]
        v_0 = torch.tensor(args[0])
        k_args = args[1:]
        num_ks = len(k_args) // 2
        k1s, k2s = k_args[:num_ks], k_args[num_ks:]
        k1s = torch.tensor(k1s).view(-1, num_ks)
        k2s = torch.tensor(k2s).view(-1, num_ks)
        return FourierSeries.compute(theta, v_0, k1s, k2s)

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
            FourierSeries interaction energy
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
        dG_fit = FourierSeries.wrapper_fit_func(bin_centers_nz[mask], *[popt])
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
            * FourierSeries.neg_log_likelihood(
                dG_nz[mask],
                FourierSeries.wrapper_fit_func(bin_centers_nz[mask], *[popt]),
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
            stat = FourierSeries._init_parameter_dict(constrain_deg)
            if regression_method == "linear":
                popt = (
                    FourierSeries._linear_regression(
                        bin_centers_nz[mask], dG_nz[mask], constrain_deg
                    )
                    .solution.numpy()
                    .tolist()
                )
            elif regression_method == "nonlinear":
                p0 = FourierSeries._init_parameters(constrain_deg)
                popt, _ = curve_fit(
                    lambda theta, *p0: FourierSeries.wrapper_fit_func(
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
            stat = FourierSeries._make_parameter_dict(stat, popt, constrain_deg)

        else:
            if metric == "aic":
                metric_func = FourierSeries._compute_aic
                best_func = min
            elif metric == "r2":
                metric_func = FourierSeries._compute_adjusted_R2
                best_func = max
            else:
                raise ValueError(
                    "metric {} is neither 'aic' nor 'r2'".format(metric)
                )

            # Determine best fit for unknown # of parameters
            stat = FourierSeries._init_parameter_dict(n_degs)
            popts = []
            metric_vals = []

            try:
                for deg in range(1, n_degs + 1):
                    free_parameters = 1 + (2 * deg)
                    if regression_method == "linear":
                        popt = (
                            FourierSeries._linear_regression(
                                bin_centers_nz[mask], dG_nz[mask], deg
                            )
                            .solution.numpy()
                            .tolist()
                        )
                    elif regression_method == "nonlinear":
                        p0 = FourierSeries._init_parameters(deg)
                        popt, _ = curve_fit(
                            lambda theta, *p0: FourierSeries.wrapper_fit_func(
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
                stat = FourierSeries._make_parameter_dict(stat, popt, n_degs)
            except:
                print(f"failed to fit potential estimate for FourierSeries")
                stat = FourierSeries._init_parameter_dict(n_degs)
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


class Dihedral(FourierSeries):
    r"""
    Class to represent a Dihedral potential using a fourier series
    """

    name: Final[str] = "dihedrals"
    _order: Final[int] = 4

    def __init__(
        self,
        statistics: Dict,
        n_degs: int = 3,
        name: str = "dihedrals",
    ) -> None:
        super(Dihedral, self).__init__(
            statistics, name=name, n_degs=n_degs, order=self._order
        )

    @staticmethod
    def neighbor_list(topology) -> None:
        nl = topology.neighbor_list(Dihedral.name)
        return {Dihedral.name: nl}

    @staticmethod
    def compute_features(
        pos: torch.Tensor, mapping: torch.Tensor
    ) -> torch.Tensor:
        return compute_torsions(pos, mapping)
