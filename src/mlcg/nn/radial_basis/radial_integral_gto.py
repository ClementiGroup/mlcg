import torch
import numpy as np
from mpmath import hyp1f1, gamma, exp, power
from itertools import product
from scipy import interpolate
from typing import Union

from ..spline import NaturalCubicSpline
from .base import _RadialBasis
from ..cutoff import _Cutoff, ShiftedCosineCutoff


class RIGTOBasis(_RadialBasis):
    r"""This radial basis is the effective basis when expanding an atomic
    density smeared by Gaussains of width :math:`\sigma` on a set of
    :math:`n_{max}` orthonormal Gaussian Type Orbitals (GTOs) and
    :math:`l_{max}+1` Spherical Harmonics (SPHs) namely.
    This radial basis set is interpolated using natural cubic splines for
    efficiency and the cutoff is included into the splined functions.

    The basis is defined as

    .. math::
        R_{nl}(r) = f_c(r) \mathcal{N}_n \frac{\Gamma(\frac{n+l+3}{2})}{\Gamma(l+\frac{3}{2})}
        c^l r^l(c+b_n)^{-\frac{(n+l+3)}{2}}
        {}_1F_1\left(\frac{n+l+3}{2},l+\frac{3}{2};\frac{c^2 r^2}{c+b_n}\right),

    where :math:`{}_1F_1` is the confluent hypergeometric function,
    :math:`\Gamma` is the gamma function, :math:`f_c` is a cutoff function,
    :math:`b_n=\frac{1}{2\sigma_n^2}`, :math:`c= 1 / (2\sigma^2`,
    :math:`\sigma_n = r_\text{cut} \max(\sqrt{n},1)/n_{max}` and
    :math:`\mathcal{N}_n^2 = \frac{2}{\sigma_n^{2n + 3}\Gamma(n + 3/2)}`.

    For more details on the derivation, refer to `appendix A <https://doi.org/10.5075/epfl-thesis-7997>`_.

    Parameters
    ----------
    nmax:
        number of radial basis
    lmax:
        maximum spherical order (lmax included so there are lmax+1 orders)
    sigma:
        smearing of the atomic density
    cutoff:
        Defines the smooth cutoff function. If a float is provided, it will be
        interpreted as an upper cutoff and a CosineCutoff will be used between
        0 and the provided float. Otherwise, a chosen _Cutoff instance can be
        supplied.
    mesh_size:
        number of points used to interpolate with splines the radial basis spanning uniformly the range difined by the cutoff :math:`[0, r_c]`.

    """

    def __init__(
        self,
        cutoff: Union[int, float, _Cutoff],
        nmax: int = 5,
        lmax: int = 5,
        sigma: float = 0.4,
        mesh_size: int = 300,
    ):
        super(RIGTOBasis, self).__init__()
        if isinstance(cutoff, (float, int)):
            self.cutoff = ShiftedCosineCutoff(float(cutoff), 0.5)
        elif isinstance(cutoff, _Cutoff):
            self.cutoff = cutoff
        else:
            raise TypeError(
                f"Supplied cutoff {cutoff} is neither a number nor a _Cutoff instance."
            )

        self.check_cutoff()
        self.nmax = nmax
        self.lmax = lmax
        self.sigma = sigma
        self.mesh_size = mesh_size
        self.num_rbf = self.nmax * (self.lmax + 1)
        self.Rln = splined_radial_integrals(
            nmax,
            lmax + 1,
            self.cutoff.cutoff_upper,
            sigma,
            self.cutoff,
            mesh_size,
        )

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        r"""Expansion of distances through the radial basis function set.

        Parameters
        ----------
        dist: torch.Tensor
            Input pairwise distances of shape (total_num_edges)

        Return
        ------
        expanded_distances: torch.Tensor
            Distances expanded in the radial basis with shape (total_num_edges, lmax + 1, nmax)
        """
        if self.lmax == 0:
            return self.Rln(dist).view(-1, self.nmax)
        else:
            return self.Rln(dist).view(-1, self.lmax + 1, self.nmax)

    def plot(self):
        """Plot the set of radial basis function."""
        import matplotlib.pyplot as plt

        dist = torch.linspace(0, self.cutoff.cutoff_upper, 200)
        y = self.forward(dist).numpy()
        for l in range(self.lmax + 1):
            for n in range(self.nmax):
                plt.plot(dist.numpy(), y[:, l, n], label=f"n={n}")
            plt.title(f"l={l}")
            plt.legend()
            plt.show()

    def reset_parameters(self):
        pass


def fit_splined_radial_integrals(nmax, lmax, rc, sigma, cutoff, mesh_size):
    c = 0.5 / sigma**2
    length, channels = mesh_size, nmax * lmax

    dists = np.linspace(0, rc + 1e-6, length)
    x = o_ri_gto(rc, nmax, lmax, dists, c).reshape((length, lmax, nmax))
    x *= cutoff(torch.from_numpy(dists)).numpy()[:, None, None]
    coeffs = torch.zeros(((4, length - 1, lmax, nmax)))
    for l in range(lmax):
        for n in range(nmax):
            ispl = interpolate.CubicSpline(dists, x[:, l, n], bc_type="natural")
            for i in range(4):
                coeffs[i, :, l, n] = torch.from_numpy(ispl.c[-i - 1])

    coeffs = coeffs.view(4, length - 1, -1)
    coeffs = (
        torch.from_numpy(dists),
        coeffs[0],
        coeffs[1],
        coeffs[2],
        coeffs[3],
    )
    return coeffs


def splined_radial_integrals(nmax, lmax, rc, sigma, cutoff, mesh_size=600):
    coeffs = fit_splined_radial_integrals(
        nmax, lmax, rc, sigma, cutoff, mesh_size
    )
    Rnl = NaturalCubicSpline(coeffs)
    return Rnl


def sn(n, rcut, nmax):
    return rcut * max(np.sqrt(n), 1) / nmax


def dn(n, rcut, nmax):
    s_n = sn(n, rcut, nmax)
    return 0.5 / (s_n) ** 2


def gto_norm(n, rcut, nmax):
    s_n = sn(n, rcut, nmax)
    norm2 = 0.5 / (np.power(s_n, 2 * n + 3) * float(gamma(n + 1.5)))
    return np.sqrt(norm2)


def ortho_Snn(rcut, nmax):
    Snn = np.zeros((nmax, nmax))
    norms = np.array([gto_norm(n, rcut, nmax) for n in range(nmax)])
    bn = np.array([dn(n, rcut, nmax) for n in range(nmax)])
    for n, m in product(range(nmax), range(nmax)):
        Snn[n, m] = (
            norms[n]
            * norms[m]
            * 0.5
            * np.power(bn[n] + bn[m], -0.5 * (3 + n + m))
            * float(gamma(0.5 * (3 + m + n)))
        )
    eigenvalues, unitary = np.linalg.eigh(Snn)
    diagoverlap = np.diag(np.sqrt(eigenvalues))
    newoverlap = unitary @ diagoverlap @ unitary.T
    orthomatrix = np.linalg.inv(newoverlap)
    return orthomatrix, Snn


def gto(rcut, nmax, r):
    ds = np.array([dn(n, rcut, nmax) for n in range(nmax)])
    ortho, Snn = ortho_Snn(rcut, nmax)
    norms = np.array([gto_norm(n, rcut, nmax) for n in range(nmax)])
    res = np.zeros((r.shape[0], nmax))
    for n in range(nmax):
        res[:, n] = norms[n] * np.power(r, n + 1) * np.exp(-ds[n] * r**2)
    res = res @ ortho
    return res


def ri_gto(n, l, rij, c, d, norm):
    res = (
        exp(-c * rij**2)
        * (gamma(0.5 * (l + n + 3)) / gamma(l + 1.5))
        * power(c * rij, l)
        * power(c + d, -0.5 * (l + n + 3))
    )
    res *= hyp1f1(0.5 * (n + l + 3), l + 1.5, power(c * rij, 2) / (c + d))
    return norm * float(res)


def o_ri_gto(rcut, nmax, lmax, rij, c):
    ds = np.array([dn(n, rcut, nmax) for n in range(nmax)])
    norms = np.array([gto_norm(n, rcut, nmax) for n in range(nmax)])
    ortho, Snn = ortho_Snn(rcut, nmax)
    res = np.zeros((rij.shape[0], lmax, nmax))
    for ii, dist in enumerate(rij):
        for l in range(lmax):
            for n in range(nmax):
                res[ii, l, n] = ri_gto(n, l, float(dist), c, ds[n], norms[n])
    res = res @ ortho
    return res
