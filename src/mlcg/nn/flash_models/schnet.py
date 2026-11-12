"""
Flash paths of the SchNet modules of :mod:`mlcg.nn.schnet`, as companion
mixins (:class:`~mlcg.nn.flash_models.SupportsFlash`): each standard class
lists its companion first in its bases, so it keeps one class, one state
dict and the standard checkpoint format. :mod:`mlcg.nn.schnet` imports
this module, so this module imports it only inside functions.
"""

import math
import warnings
from typing import Optional

import torch
from torch_geometric.utils import scatter

from ...data.atomic_data import ENERGY_KEY, AtomicData
from ...neighbor_list.neighbor_list import data_excluded_order
from ..kernels import (
    RadiusCSR,
    build_filter_table,
    fused_cfconv,
    tabulated_cfconv,
)
from ..kernels.pbc import check_cells, minimum_image_vectors, periodic_cells
from .mixin import (
    FIRST_NUM_KNOTS,
    MAX_NUM_KNOTS,
    TABLE_TOLERANCE,
    SupportsFlash,
    check_tolerance,
)


class CFConvFlashPath(SupportsFlash):
    """Flash path of :class:`~mlcg.nn.schnet.CFConv` (``aggr="add"`` only),
    when called with ``flash=`` (see :meth:`flash_forward`): in eval mode
    with a table (:meth:`build_table`), :func:`tabulated_cfconv` reads the
    filter from it; otherwise :func:`fused_cfconv` fuses the message passing
    on the given edge features, without materializing per-edge messages."""

    _runtime_attributes = ("_kernels", "_table")

    def drop_inference(self):
        self._buffers.pop("_table", None)

    @property
    def table(self) -> Optional[torch.Tensor]:
        """Hermite table of the filter, ``[num_knots, 3, num_filters]``."""
        return self._buffers.get("_table")

    @property
    def tabulated(self) -> bool:
        """Whether :meth:`flash_forward` reads the table (eval mode only)."""
        return self.table is not None and not self.training

    def flash_forward(
        self,
        x,
        row_ptr,
        col,
        features=None,
        pos=None,
        batch=None,
        cell=None,
        pbc=None,
        cutoff=None,
    ):
        """Message passing on a destination CSR (``row_ptr``, ``col``) of a
        *symmetric* neighbour list. If :attr:`tabulated`: the filter table,
        with distances recomputed from ``pos`` (minimum image with
        ``cell``/``pbc``, edges beyond ``cutoff`` ignored). Otherwise the
        exact fused kernels (trainable) on ``features = (distances, rbf)``
        in row order."""
        x = self.lin1(x)
        if self.tabulated:
            x = tabulated_cfconv(
                x, pos, row_ptr, col, self.table, cutoff, batch, cell, pbc
            )
        else:
            distances, rbf = features
            filters = self.filter_network(rbf)
            weight = self.cutoff(distances)
            x = fused_cfconv(x, filters, weight, row_ptr, col)
        return self.lin2(x)

    def build_table(
        self,
        rbf_layer: torch.nn.Module,
        num_knots: Optional[int] = None,
        tolerance: float = TABLE_TOLERANCE,
    ):
        """Tabulates the filter (times the cutoff) of ``rbf_layer``'s
        expansion from the current weights (float64, cubic Hermite over
        ``rbf_layer``'s cutoff) and keeps it as :attr:`table`. It uses
        ``num_knots`` knots, or, if ``None``, the smallest power of two
        meeting ``tolerance`` (relative): the next size is predicted from
        the O(h^4) error and then verified, up to ``MAX_NUM_KNOTS``. Raises
        ``ValueError`` if the filter can't meet ``tolerance`` (e.g.
        non-smooth activations) or if ``tolerance`` is below the float32
        floor. Weights changed by hand afterwards need another call
        (``train()`` and ``load_state_dict`` drop the table)."""
        check_tolerance(tolerance)
        cutoff_upper = float(rbf_layer.cutoff.cutoff_upper)

        def table_on(knots):
            return build_filter_table(
                self.filter_network, rbf_layer, self.cutoff, cutoff_upper, knots
            )

        knots = num_knots or FIRST_NUM_KNOTS
        table, error = table_on(knots)
        while num_knots is None and error > tolerance and knots < MAX_NUM_KNOTS:
            predicted = knots * (error / tolerance) ** 0.25
            knots = min(MAX_NUM_KNOTS, 1 << math.ceil(math.log2(predicted)))
            previous = error
            table, error = table_on(knots)
            if error >= previous:  # at the float32 floor: more knots won't help
                break
        if error > tolerance:
            raise ValueError(
                f"Filter table error {error:.2e} exceeds {tolerance:.0e} "
                f"with {knots} knots"
            )
        self.register_buffer("_table", table, persistent=False)


class StandardSchNetFlashPath(SupportsFlash):
    """Flash path of :class:`~mlcg.nn.schnet.StandardSchNet`: with
    :meth:`enable_kernels`, the radius graph comes from a sync-free
    neighbour search (a CSR) and every block runs fused Triton kernels
    (trainable, double backward). After :meth:`prepare_inference`, in eval
    mode, the filters are read from cubic Hermite tables instead of being
    evaluated (inference only; periodic cells as in
    :mod:`~mlcg.nn.kernels.pbc`). ``train()`` and loading a state dict drop
    the tables. Needs every block to have a flash path
    (:meth:`~mlcg.nn.schnet.InteractionBlock.has_flash_path`)."""

    _runtime_attributes = ("_kernels", "neighbors")

    def forward(self, data: AtomicData) -> AtomicData:
        if self._kernels:
            return self.flash_forward(data)
        return super().forward(data)

    def enable_kernels(self, enabled: bool = True):
        """Switches the flash path of every block on or off. Refused, with a
        warning, unless every block is an :class:`InteractionBlock` with a
        flash path (and as in :meth:`SupportsFlash.enable_kernels`)."""
        from ..schnet import InteractionBlock  # it imports this module

        if enabled and not all(
            isinstance(block, InteractionBlock) and block.has_flash_path()
            for block in self.interaction_blocks
        ):
            warnings.warn(
                f"{type(self).__name__} has no flash path: its blocks are not "
                "all InteractionBlock + CFConv with aggr='add'"
            )
            return
        super().enable_kernels(enabled)
        if self._kernels and not hasattr(self, "neighbors"):
            cutoff = float(self.rbf_layer.cutoff.cutoff_upper)
            self.neighbors = RadiusCSR(cutoff)

    def prepare_inference(
        self,
        data: AtomicData,
        num_knots: Optional[int] = None,
        tolerance: float = TABLE_TOLERANCE,
    ):
        """Filter tables of the blocks (:meth:`CFConv.build_table`); a block
        whose filter can't be tabulated keeps its filter network, with a
        warning, and the CSR of ``data``'s excluded pairs, used while the
        pairs keep their shape (other pairs of that shape give NaN
        energies). Raises ``ValueError`` for a periodic cell the flash path
        can't handle (see :func:`~mlcg.nn.kernels.pbc.check_cells`) or
        excluded pairs of another order than the model's."""
        if not self._kernels:
            return
        self._prepare_search(data)
        for i, block in enumerate(self.interaction_blocks):
            try:
                block.conv.build_table(self.rbf_layer, num_knots, tolerance)
            except ValueError as error:
                warnings.warn(
                    f"{error}; block {i} of {self.name} uses its filter network"
                )

    def flash_forward(self, data: AtomicData) -> AtomicData:
        """Flash path: the blocks get the radius graph as ``flash=(row_ptr,
        col, features, pos, batch, cell, pbc, cutoff)`` and read their filter
        tables in eval mode, or run the exact fused kernels on ``features =
        (distances, rbf)`` in row order. The per-edge tensors (and their
        host sync) are computed only if a block has no table in use.
        Energies of graphs whose neighbour search overflowed are NaN."""
        row_ptr, col, valid, cell, pbc = self._radius_graph(data)
        features = None
        if not all(block.conv.tabulated for block in self.interaction_blocks):
            _, _, distances = self._edge_distances(
                data, row_ptr, col, cell, pbc
            )
            features = (distances, self.rbf_layer(distances))
        graph = (
            row_ptr,
            col,
            features,
            data.pos,
            data.batch,
            cell,
            pbc,
            self.neighbors.cutoff,
        )
        x = self.embedding_layer(data.atom_types)
        for block in self.interaction_blocks:
            x = x + block(x, flash=graph)
        energy = self.output_network(x, data) * valid
        num_graphs = data.ptr.numel() - 1 if hasattr(data, "ptr") else None
        energy = scatter(
            energy, data.batch, dim=0, reduce="sum", dim_size=num_graphs
        )
        data.out.setdefault(self.name, {}).update(
            {ENERGY_KEY: energy.flatten()}
        )
        return data

    def _radius_graph(self, data: AtomicData):
        """The sync-free radius graph as a destination CSR of a symmetric
        list (minimum image with a cell): ``row_ptr, col, valid, cell, pbc``,
        with ``valid`` 1, or NaN for a graph whose search overflowed."""
        ptr = (
            data.ptr
            if hasattr(data, "ptr")
            else data.pos.new_tensor([0, data.pos.shape[0]], dtype=torch.long)
        )
        cell, pbc = periodic_cells(
            data.get("cell"), data.get("pbc"), ptr.numel() - 1, data.pos.device
        )
        row_ptr, col, valid = self.neighbors(
            data.pos.detach(),
            data.batch,
            ptr,
            cell,
            pbc,
            self._excluded_pairs(data),
        )
        return row_ptr, col, valid, cell, pbc

    def _excluded_pairs(self, data: AtomicData) -> Optional[torch.Tensor]:
        """The data's excluded pairs, ``None`` without them. Every call
        checks only that they are present exactly when the model excludes
        pairs (no host sync); their order is checked once, where the data
        enters the flash path (:meth:`_check_flash_data`)."""
        excluded = data.get("excluded_pairs")
        order = getattr(self, "excluded_order", 1)
        # branches, not `bool(excluded) != (order > 1)`: under
        # torch.compile(dynamic=True) `order` is a SymInt
        if excluded:
            if order == 1:
                raise ValueError(
                    "The data has excluded pairs, the model excludes none "
                    "(excluded_order 1)"
                )
            return excluded["index_mapping"]
        if order > 1:
            raise ValueError(
                f"The model has excluded_order {order}, the data has no "
                "excluded pairs"
            )
        return None

    def drop_inference(self):
        if hasattr(self, "neighbors"):
            self.neighbors.store_exclusions(None)

    def _prepare_search(self, data: AtomicData):
        """Checks ``data`` (:meth:`_check_flash_data`) and stores the CSR
        of its excluded pairs for the following calls (``data`` may be
        an empty ``AtomicData`` for a model without exclusions)."""
        self._check_flash_data(data)
        pairs = self._excluded_pairs(data)
        self.neighbors.store_exclusions(
            pairs, 0 if pairs is None else data.pos.shape[0]
        )

    def _check_flash_data(self, data: AtomicData):
        """Raises for data the flash path can't use: an unsupported cell, or
        excluded pairs of another order than the model's (one host sync)."""
        check_cells(data.get("cell"), data.get("pbc"), self.neighbors.cutoff)
        order = getattr(self, "excluded_order", 1)
        if data_excluded_order(data) != order:
            raise ValueError(
                f"The data has excluded_order {data_excluded_order(data)}, "
                f"the model {order}"
            )

    def _edge_distances(self, data: AtomicData, row_ptr, col, cell, pbc):
        """Sources, destinations and distances of the edges in row order (one
        host sync: the number of edges)."""
        num_edges = int(row_ptr[-1])
        dst = torch.repeat_interleave(
            torch.arange(len(row_ptr) - 1, device=col.device),
            row_ptr.diff(),
            output_size=num_edges,
        )
        src = col[:num_edges].long()
        vectors = minimum_image_vectors(
            data.pos[dst] - data.pos[src], cell, pbc, data.batch[dst]
        )
        return src, dst, vectors.norm(dim=1)


class RBFRegularizedSchNetFlashPath(SupportsFlash):
    """Flash path of :class:`~mlcg.nn.schnet.RBFRegularizedSchNet`: the exact
    fused kernels only, without filter tables, since the filters depend on
    the bead types."""

    def prepare_inference(
        self,
        data: AtomicData,
        num_knots: Optional[int] = None,
        tolerance: float = TABLE_TOLERANCE,
    ):
        """No filter tables: the filters depend on the types of the pair, not
        only on the distance. Raises ``ValueError`` for a periodic cell the
        flash path can't handle (see :func:`~mlcg.nn.kernels.pbc.check_cells`).
        """
        if self._kernels:
            self._prepare_search(data)

    def flash_forward(self, data: AtomicData) -> AtomicData:
        """Flash path: the exact fused kernels on the radius graph of the
        sync-free neighbour search, each block with its own regularized
        expansion (per-edge tensors: one host sync per step). The expansion
        is symmetric in the pair, as the kernels need."""
        row_ptr, col, valid, cell, pbc = self._radius_graph(data)
        src, dst, distances = self._edge_distances(
            data, row_ptr, col, cell, pbc
        )
        types = data.atom_types
        rbf_expansion = self.rbf_layer(distances, types[src], types[dst])
        x = self.embedding_layer(types)
        for block, rbf in zip(self.interaction_blocks, rbf_expansion):
            graph = (
                row_ptr,
                col,
                (distances, rbf),
                data.pos,
                data.batch,
                cell,
                pbc,
                self.neighbors.cutoff,
            )
            x = x + block(x, flash=graph)
        energy = self.output_network(x, data) * valid
        energy = scatter(energy, data.batch, dim=0, reduce="sum")
        data.out[self.name] = {
            ENERGY_KEY: energy.flatten(),
            "radial_filters": self.rbf_layer.get_regularization_parameters(),
        }
        return data
