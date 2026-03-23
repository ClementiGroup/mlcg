import torch
from torch import nn
from typing import Union

from .base import _RadialBasis
from ..cutoff import _Cutoff, IdentityCutoff


class GaussianBasis(_RadialBasis):
    r"""Class that generates a set of equidistant 1-D gaussian basis functions
    scattered between a specified lower and upper cutoff:

    .. math::

        f_n = \exp{ \left( -\gamma(r-c_n)^2 \right) }

    Parameters
    ----------
    cutoff:
        Defines the smooth cutoff function. If a float is provided, it will be interpreted as
        an upper cutoff and an IdentityCutoff will be used between 0 and the provided float. Otherwise,
        a chosen _Cutoff instance can be supplied.
    num_rbf:
        The number of gaussian functions in the basis set.
    trainable:
        If True, the parameters of the gaussian basis (the centers and widths of
        each function) are registered as optimizable parameters that will be
        updated during backpropagation. If False, these parameters will be
        instead fixed in an unoptimizable buffer.

    """

    def __init__(
        self,
        cutoff: Union[int, float, _Cutoff],
        num_rbf: int = 50,
        trainable: bool = False,
    ):
        super(GaussianBasis, self).__init__()
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

        self.check_cutoff()

        self.num_rbf = num_rbf
        self.trainable = trainable

        offset, coeff = self._initial_params()
        if trainable:
            self.register_parameter("coeff", nn.Parameter(coeff))
            self.register_parameter("offset", nn.Parameter(offset))
        else:
            self.register_buffer("coeff", coeff)
            self.register_buffer("offset", offset)

    def _initial_params(self):
        r"""Method for generating the initial parameters of the basis.
        The functions are set to have equidistant centers between the
        lower and cupper cutoff, while the variance of each function
        is set based on the difference between the lower and upper
        cutoffs.
        """
        offset = torch.linspace(
            self.cutoff.cutoff_lower, self.cutoff.cutoff_upper, self.num_rbf
        )
        coeff = -0.5 / (offset[1] - offset[0]) ** 2
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
