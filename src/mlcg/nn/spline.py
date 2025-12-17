import torch.nn as nn
import torch
from typing import List


class NaturalCubicSpline(nn.Module):
    """Calculates the natural cubic spline approximation for a set of functions that are defined on the same points. Also calculates their derivatives.
    The supporting grid, :obj:`t`, to train the splines is assumed to be equispaced.

    The cubic spline is given by::

        ((d*f + c)*f + b)*f + a

    where :obj:`f` is the fractional part of the input w.r.t the grid.

    Parameters
    ----------
    coeffs:
        list of coefficients needed to do the interpolation in this order:
        :obj:`[t, a, b, c, d]`
    """

    def __init__(self, coeffs: List[torch.Tensor]):
        super(NaturalCubicSpline, self).__init__()
        t, a, b, c, d = coeffs

        self.register_buffer("_t", t)
        self.register_buffer("_a", a)
        self.register_buffer("_b", b)
        self.register_buffer("_c", c)
        self.register_buffer("_d", d)

        n_bin = self._t.shape[0] - 1
        rng = self._t[-1] - self._t[0]
        self.factor = n_bin / rng

    def __len__(self):
        """Method to return basis size"""
        return self._b.shape[-1]

    def _interpret_t(self, t):
        index = torch.floor(self.factor * t).to(dtype=torch.long)
        fractional_part = (t - self._t[index]).unsqueeze(-1)
        return fractional_part, index

    def forward(self, t):
        fractional_part, index = self._interpret_t(t)
        inner = (
            self._c[..., index, :] + self._d[..., index, :] * fractional_part
        )
        inner = self._b[..., index, :] + inner * fractional_part
        return self._a[..., index, :] + inner * fractional_part

    def backward(self, t):
        fractional_part, index = self._interpret_t(t)
        inner = (
            2 * self._c[..., index, :]
            + 3 * self._d[..., index, :] * fractional_part
        )
        deriv = self._b[..., index, :] + inner * fractional_part
        return deriv
