"""
Flash paths of the priors of :mod:`mlcg.nn.prior`, as companion mixins
(:class:`~mlcg.nn.flash_models.SupportsFlash`) listed first in the bases of
the standard classes, as in :mod:`mlcg.nn.flash_models.schnet`.
"""

from ...data import AtomicData
from .mixin import SupportsFlash


class PriorFlashPath(SupportsFlash):
    """Flash path of :class:`~mlcg.nn.prior.base._Prior`: a prior class
    with a fused energy op sets ``_kernel_op``; :meth:`enable_kernels` then
    computes its energy with that op from the parameters of
    ``data2parameters`` (per-type tables or the per-interaction buffers of
    a specialised prior). The ops have no periodic cell shifts, so
    periodic data runs the standard path. Each prior family's companion
    maps its parameters to the op's arguments in ``flash_forward``."""

    _kernel_op = None

    def enable_kernels(self, enabled: bool = True):
        super().enable_kernels(enabled and self._kernel_op is not None)

    def _uses_kernels(self, data: AtomicData) -> bool:
        return self._kernels and (
            data.get("pbc") is None or data.get("cell") is None
        )

    def _kernel_forward(self, data: AtomicData, **params) -> AtomicData:
        neighbor_list = data.neighbor_list[self.name]
        num_graphs = data.ptr.numel() - 1 if hasattr(data, "ptr") else None
        energy = self._kernel_op(
            data.pos,
            data.atom_types,
            neighbor_list["index_mapping"],
            neighbor_list["mapping_batch"],
            num_graphs=num_graphs,
            **params,
        )
        data.out.setdefault(self.name, {})["energy"] = energy
        return data


class HarmonicFlashPath(PriorFlashPath):
    """Flash path of :class:`~mlcg.nn.prior.harmonic.Harmonic`."""

    def flash_forward(self, data: AtomicData) -> AtomicData:
        params = self.data2parameters(data)
        return self._kernel_forward(data, k=params["k"], x_0=params["x0"])


class FourierSeriesFlashPath(PriorFlashPath):
    """Flash path of :class:`~mlcg.nn.prior.fourier_series.FourierSeries`."""

    def flash_forward(self, data: AtomicData) -> AtomicData:
        params = self.data2parameters(data)
        return self._kernel_forward(
            data,
            k1=params["k1s"],
            k2=params["k2s"],
            v_0=params["v_0"],
            deg=self.n_degs,
        )


class RepulsionFlashPath(PriorFlashPath):
    """Flash path of :class:`~mlcg.nn.prior.repulsion.Repulsion`."""

    def flash_forward(self, data: AtomicData) -> AtomicData:
        return self._kernel_forward(data, sigma=self.sigma)
