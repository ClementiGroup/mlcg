import torch
from torch import nn
from typing import Union

from .base import _RadialBasis
from ..cutoff import _Cutoff, CosineCutoff, IdentityCutoff


class SpacedExpBasis(_RadialBasis):
    r"""Class for generating a set of exponential normal radial basis functions,
    with means and standard deviations designed to capture the physics around 2-* A.
    The functions have an exponential form with the following means and std:

    .. math::
        \sigma_0 = \sigma_f/s
        \sigma_1 = \sigma_${min}
        \sigma_2 = \sigma_f*\sigma_1
        ...
        \sigma_n = \sigma_f*\sigma_${n-1}

        \mu_0 = 0
        \mu_1 = \sigma_f
        \mu_2 = \mu_1 + s*\sigma_1
        ...
        \mu_n = \mu_${n-1} + s*\sigma_${n-1}

    Parameters
    ----------
    cutoff:
        Defines the smooth cutoff function. If a float is provided, it will be interpreted as
        an IdentityCutoff ranging over [0, cutoff]. Otherwise,
        a chosen `_Cutoff` instance can be supplied.
    sigma_min:
        Width of first non-zero-centered basis function
    sigma_factor:
        Location of first non-zero-centered basis function and multiplicative factor to spread std of each new peak by
    mean_spacing:
        this time previous sigma indicates how much to distance the mean of subsequent gaussian by
    trainable:
        If True, the parameters of the basis (the centers and widths of each
        function) are registered as optimizable parameters that will be updated
        during backpropagation. If False, these parameters will be instead fixed
        in an unoptimizable buffer.
    """

    def __init__(
        self,
        cutoff: Union[int, float, _Cutoff],
        sigma_min: float = 0.25,
        sigma_factor: float = 2.0,
        mean_spacing: float = 2.0,
        trainable: bool = True,
    ):
        super(SpacedExpBasis, self).__init__()
        if isinstance(cutoff, (float, int)):
            self.cutoff = IdentityCutoff(0, cutoff)
        elif isinstance(cutoff, _Cutoff):
            self.cutoff = cutoff
        else:
            raise TypeError(
                "Supplied cutoff {} is neither a number nor a _Cutoff instance.".format(
                    cutoff
                )
            )

        self.sigma_min = sigma_min
        self.sigma_factor = sigma_factor
        self.mean_spacing = mean_spacing

        self.check_cutoff()
        if self.sigma_factor < 1.0:
            raise ValueError(
                "Sigma factor must be greater than or equal to 1.0, but {} was supplied.".format(
                    self.sigma_factor
                )
            )
        self.trainable = trainable

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)
        self.num_rbf = len(means)

    def _initial_params(self):
        r"""
        Method for initializing the basis function parameters
        """

        mus = [0, self.sigma_factor]
        sigmas = [self.sigma_factor / self.mean_spacing, self.sigma_min]
        while mus[-1] < self.cutoff.cutoff_upper:
            mus.append(mus[-1] + self.mean_spacing * sigmas[-1])
            sigmas.append(self.sigma_factor * sigmas[-1])
        means = torch.tensor(mus)
        betas = 2 * (torch.tensor(sigmas) ** 2)
        return means, betas

    def reset_parameters(self):
        r"""
        Method to reset the parameters of the basis functions to their
        initial values.
        """
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        r"""
        Expansion of distances through the radial basis function set.

        Parameters
        ----------
        dist: torch.Tensor
            Input pairwise distances of shape (total_num_edges)

        Return
        ------
        expanded_distances: torch.Tensor
            Distances expanded in the radial basis with shape (total_num_edges, num_rbf)
        """

        dist = dist.unsqueeze(-1)
        return self.cutoff(dist) * torch.exp(
            -((dist - self.means) ** 2) / self.betas
        )
