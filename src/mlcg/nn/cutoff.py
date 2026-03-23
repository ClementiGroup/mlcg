from typing import Union
import torch
import torch.nn as nn
import numpy as np
import math


class _Cutoff(nn.Module):
    r"""Abstract cutoff class"""

    def __init__(self):
        super(_Cutoff, self).__init__()
        self.cutoff_lower = None
        self.cutoff_upper = None

    def check_cutoff(self):
        if self.cutoff_upper < self.cutoff_lower:
            raise ValueError(
                "Upper cutoff {} is less than lower cutoff {}".format(
                    self.cutoff_upper, self.cutoff_lower
                )
            )

    def forward(self):
        raise NotImplementedError


class _OneSidedCutoff(_Cutoff):
    r"""Abstract classs for cutoff functions with a fuxed lower cutoff of 0"""

    def __init__(self):
        super(_OneSidedCutoff, self).__init__()
        self.cutoff_lower = 0
        self.cutoff_upper = None

    def forward(self):
        raise NotImplementedError


class IdentityCutoff(_Cutoff):
    r"""Cutoff function that is one everywhere, but retains
    cutoff_lower and cutoff_upper attributes

    Parameters
    ----------
    cutoff_lower:
        left bound for the radial cutoff distance
    cutoff_upper:
        right bound for the radial cutoff distance
    """

    def __init__(self, cutoff_lower: float = 0, cutoff_upper: float = np.inf):
        super(IdentityCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

        self.check_cutoff()

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        r"""Fowrad method that returns a cutoff enevlope where all values are
        one

        Parameters
        ----------
        distances:
            Input distances of shape (total_num_distances)

        Returns
        -------
            Cutoff envelope filled with ones, of shape (total_num_edges)
        """
        return torch.ones_like(distances)


class CosineCutoff(_Cutoff):
    r"""Class implementing a cutoff envelope based a cosine signal in the
    interval `[lower_cutoff, upper_cutoff]`:

    .. math::

        \cos{\left( r_{ij} \times \pi / r_{high}\right)} + 1.0

    NOTE: The behavior of the cutoff is qualitatively different for lower
    cutoff values greater than zero when compared to the zero lower cutoff
    default. We recommend visualizing your basis to see if it makes physical
    sense.

    .. math::

        0.5 \cos{ \left[ \pi \left(2 \frac{r_{ij} - r_{low}}{r_{high}
         - r_{low}} + 1.0 \right)\right]} + 0.5

    """

    def __init__(self, cutoff_lower: float = 0.0, cutoff_upper: float = 5.0):
        super(CosineCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

        self.check_cutoff()

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Applies cutoff envelope to distances.

        Parameters
        ----------
        distances:
            Distances of shape (total_num_edges)

        Returns
        -------
        cutoffs:
            Distances multiplied by the cutoff envelope, with shape
            (total_num_edges)
        """
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (
                torch.cos(
                    math.pi
                    * (
                        2
                        * (distances - self.cutoff_lower)
                        / (self.cutoff_upper - self.cutoff_lower)
                        + 1.0
                    )
                )
                + 1.0
            )
            # remove contributions below the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).to(
                distances.dtype
            )
            cutoffs = cutoffs * (distances > self.cutoff_lower).to(
                distances.dtype
            )
            return cutoffs
        else:
            cutoffs = 0.5 * (
                torch.cos(distances * math.pi / self.cutoff_upper) + 1.0
            )
            # remove contributions beyond the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).to(
                distances.dtype
            )
            return cutoffs


class ShiftedCosineCutoff(_OneSidedCutoff):
    r"""Class of Behler cosine cutoff with an additional smoothing parameter.

    .. math::

        0.5 + 0.5  \cos{ \left[ \pi \left( \frac{r_{ij} - r_{high} +
        \sigma}{\sigma}\right)\right]}

    where :math:`\sigma` is the smoothing width.

    Parameters
    ----------
    cutoff:
        cutoff radius
    smooth_width:
        parameter that controls the extent of smoothing in the cutoff envelope.

    """

    def __init__(
        self,
        cutoff: Union[int, float] = 5.0,
        smooth_width: Union[int, float] = 0.5,
    ):
        super(ShiftedCosineCutoff, self).__init__()
        self.cutoff_upper = cutoff
        self.smooth_width = smooth_width
        # del self.cutoff_upper
        # self.register_buffer("cutoff_upper", torch.Tensor([cutoff]))
        # self.register_buffer("smooth_width", torch.Tensor([smooth_width]))

    def forward(self, distances):
        """Compute cutoff function.

        Args:
            distances (torch.Tensor): values of interatomic distances.

        Returns:
            torch.Tensor: values of cutoff function.

        """
        # Compute values of cutoff function
        cutoffs = torch.ones_like(distances)
        mask = distances > self.cutoff_upper - self.smooth_width
        cutoffs[mask] = 0.5 + 0.5 * torch.cos(
            math.pi
            * (distances[mask] - self.cutoff_upper + self.smooth_width)
            / self.smooth_width
        )
        cutoffs[distances > self.cutoff_upper] = 0.0

        return cutoffs.view(-1)
