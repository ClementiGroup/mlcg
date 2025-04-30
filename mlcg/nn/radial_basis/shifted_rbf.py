import torch
from torch import nn
from typing import Union

from .base import _RadialBasis



class ShiftedRBF(_RadialBasis):
    r"""Subclass of ExpNormalBasis which shifts the distance by a value
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
    shift:
        Zero value of the functions.
    """

    def __init__(
        self,
        rbf : _RadialBasis,
        shift: float = 0.0,
    ):
        super(ShiftedRBF, self).__init__()
        self.rbf = rbf
        self.num_rbf = self.rbf.num_rbf
        self.cutoff = self.rbf.cutoff
        assert shift >= 0.0
        self.shift = shift
    
    def reset_parameters(self):
        r"""Method to reset the parameters of the basis functions to their
        initial values.
        """
        self.rbf.reset_parameters()

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
        return self.rbf.forward(dist - self.shift)