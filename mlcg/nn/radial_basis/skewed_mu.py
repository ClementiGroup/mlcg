import torch
from torch import nn
from typing import Union

from .base import _RadialBasis
from ..cutoff import _Cutoff, IdentityCutoff


class SkewedMuGaussianBasis(_RadialBasis):
    r"""Class that generates a set of non-equidistant 1-D gaussian basis functions
    with their means scattered between 0 and 1:

    .. math::

        f_n = \exp{ \left( -\gamma(r-c_n)^2 \right) }

    Parameters
    ----------
    num_rbf:
        The number of gaussian functions in the basis set.
    zeta:
        It controls the skewness of the distribution of the center of gaussian bases.
        When 0 < zeta < 1: the distribution is denser closer to one. The smaller zeta
        is, the farther away the distribution deviates from the uniform case.
        When zeta > 1: the distribution is denser closer to 1.
        When zeta is very close to 1, the distribution is close to uniform like in a
        normal Gaussian RBF.

    """

    def __init__(
        self,
        num_rbf: int = 32,
        zeta: float = 1.0,
    ):
        super(SkewedMuGaussianBasis, self).__init__()

        self.num_rbf = num_rbf
        self.zeta = zeta

        offset, coeff = self._initial_params()
        self.register_buffer("coeff", coeff)
        self.register_buffer("offset", offset)

    def _initial_params(self):
        r"""Method for generating the initial parameters of the basis.
        The functions are set to have equidistant centers between the
        lower and cupper cutoff, while the variance of each function
        is set based on the difference between the lower and upper
        cutoffs.
        """
        if abs(self.zeta - 1.0) < 1e-3:
            # we switch to uniform dtrisbution of mu to avoid numerical instability near 1
            offset = torch.linspace(0, 1, self.num_rbf)
            coeff = -0.5 / (offset[1] - offset[0]) ** 2
        else:
            xs = torch.linspace(0, 1, self.num_rbf)
            offset = (1 - self.zeta**xs) / (1 - self.zeta)
            xs1 = torch.linspace(
                0 - 0.5 / (self.num_rbf - 1),
                1 + 0.5 / (self.num_rbf - 1),
                self.num_rbf + 1,
            )
            coeff = (
                -0.5
                / (
                    (self.zeta ** xs1[:-1] - self.zeta ** xs1[1:])
                    / (1 - self.zeta)
                )
                ** 2
            )
        return offset, coeff

    def reset_parameters(self):
        r"""Method for resetting the basis to its initial state"""
        offset, coeff = self._initial_params()
        self.offset.data.copy_(offset)
        self.coeff.data.copy_(coeff)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        r"""Expansion of distances through the radial basis function set.

        Parameters
        ----------
        dist:
            Input pairwise distances of shape (total_num_edges)

        Return
        ------
        expanded_distances:
            Distances expanded in the radial basis with shape (total_num_edges,
            num_rbf)
        """

        dist = dist.unsqueeze(-1)
        expanded_distances = torch.exp(
            self.coeff * torch.pow(dist - self.offset, 2)
        ) * self.cutoff(dist)
        return expanded_distances
