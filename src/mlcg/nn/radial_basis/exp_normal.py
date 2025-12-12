import torch
from torch import nn
from typing import Union

from .base import _RadialBasis
from ..cutoff import _Cutoff, CosineCutoff


class ExpNormalBasis(_RadialBasis):
    r"""Class for generating a set of exponential normal radial basis functions,
    as described in [Physnet]_ . The functions have the following form:

    .. math::

        f_n(r_{ij};\alpha, r_{low},r_{high}) = f_{cut}(r_{ij},r_{low},r_{high})
        \times \exp\left[-\beta_n \left(e^{\alpha (r_{ij} -r_{high}) }
        - \mu_n \right)^2\right]

    where

    .. math::

        \alpha = 5.0/(r_{high} - r_{low})

    is a distance rescaling factor, and, by default

    .. math::

        f_{cut} ( r_{ij},r_{low},r_{high} ) =  \cos{\left( r_{ij} \times \pi / r_{high}\right)} + 1.0

    represents a cosine cutoff function (though users can specify their own cutoff function
    if they desire).
    Parameters
    ----------
    cutoff:
        Defines the smooth cutoff function. If a float is provided, it will be interpreted as
        an upper cutoff and a CosineCutoff will be used between 0 and the provided float. Otherwise,
        a chosen `_Cutoff` instance can be supplied.
    num_rbf:
        The number of functions in the basis set.
    trainable:
        If True, the parameters of the basis (the centers and widths of each
        function) are registered as optimizable parameters that will be updated
        during backpropagation. If False, these parameters will be instead fixed
        in an unoptimizable buffer.
    """

    def __init__(
        self,
        cutoff: Union[int, float, _Cutoff],
        num_rbf: int = 50,
        trainable: bool = True,
    ):
        super(ExpNormalBasis, self).__init__()
        if isinstance(cutoff, (float, int)):
            self.cutoff = CosineCutoff(0, cutoff)
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
        self.alpha = 5.0 / (self.cutoff.cutoff_upper - self.cutoff.cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        r"""Method for initializing the basis function parameters, as described in
        https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181 .
        """

        start_value = torch.exp(
            torch.scalar_tensor(
                -self.cutoff.cutoff_upper + self.cutoff.cutoff_lower
            )
        )
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def reset_parameters(self):
        r"""Method to reset the parameters of the basis functions to their
        initial values.
        """
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        r"""Expansion of distances through the radial basis function set.
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
            -self.betas
            * (
                torch.exp(self.alpha * (-dist + self.cutoff.cutoff_lower))
                - self.means
            )
            ** 2
        )


class ExtendedExpNormalBasis(ExpNormalBasis):
    r"""ExpNormalBasis that allows for arbitrary lower cutoffs not tied to
    a supplied `_Cutoff`. This basis follows the definition in [Physnet]_.
    Parameters
    ----------
    internal_cutoff:
        if specified, this lower cutoff overrides any supplied `_Cutoff` lower
        cutoff.
    trainable:
        If True, the parameters of the basis (the centers and widths of each
        function) are registered as optimizable parameters that will be updated
        during backpropagation. If False, these parameters will be instead fixed
        in an unoptimizable buffer.
    """

    def __init__(
        self,
        *args,
        internal_cutoff_lower: Union[None, float] = None,
        trainable=False,
        **kwargs,
    ):
        super(ExtendedExpNormalBasis, self).__init__(*args, **kwargs)

        if internal_cutoff_lower != None:
            self.internal_cutoff_lower = internal_cutoff_lower
        else:
            self.internal_cutoff_lower = self.cutoff.cutoff_lower

        self.internal_cutoff_upper = self.cutoff.cutoff_upper
        self.alpha = 5.0 / (
            self.internal_cutoff_upper - self.internal_cutoff_lower
        )
        means, betas = self.extended_initial_params()

        # reset attributes
        if trainable:
            delattr(self, "means")
            delattr(self, "betas")
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            delattr(self, "means")
            delattr(self, "betas")
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def extended_initial_params(self):
        r"""Method for initializing the basis function parameters, as described in
        https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181 .
        """

        start_value = torch.exp(
            torch.scalar_tensor(
                -self.internal_cutoff_upper + self.internal_cutoff_lower
            )
        )
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        r"""Expansion of distances through the radial basis function set.
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
            -self.betas
            * (
                torch.exp(self.alpha * (-dist + self.internal_cutoff_lower))
                - self.means
            )
            ** 2
        )
