import math
import warnings
import copy
from typing import List, Optional
import torch
from torch_geometric.utils import scatter
from ...data.atomic_data import AtomicData, ENERGY_KEY
from ..mlp import MLP
from ..schnet import SchNet, StandardSchNet, InteractionBlock, CFConv
from ...geometry.internal_coordinates import compute_distances
from ...neighbor_list.neighbor_list import make_neighbor_list
from ..kernels.pbc import check_cells, minimum_image_vectors, periodic_cells
from ..kernels import (
    RadiusCSR,
    build_csr_representation_from_edges,
    build_filter_table,
    fused_cfconv,
    tabulated_cfconv,
)

TABLE_TOLERANCE = 1e-5
MIN_TOLERANCE = 1e-7  # float32 storage and arithmetic floor
FIRST_NUM_KNOTS = 2048
MAX_NUM_KNOTS = 16384


def check_tolerance(tolerance: float):
    if tolerance < MIN_TOLERANCE:
        raise ValueError(
            f"Table tolerance {tolerance:.0e} is below the float32 floor "
            f"{MIN_TOLERANCE:.0e}"
        )


def _filter_table(conv, rbf_layer, cutoff_upper, num_knots, tolerance):
    """Table of ``conv``'s filter on ``num_knots`` knots, or, if ``None``, on
    the smallest power of two meeting ``tolerance``: the next size is
    predicted from the O(h^4) error and then verified, up to
    ``MAX_NUM_KNOTS``."""

    def table_on(knots):
        return build_filter_table(
            conv.filter_network, rbf_layer, conv.cutoff, cutoff_upper, knots
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
    return table


def _drop_tables_hook(module, incompatible_keys):
    module.drop_tables()


class FlashSchNet(SchNet):
    r"""
    See :class:`SchNet` for the full API documentation.

    This implementation replaces the standard message-passing kernels
    with fused Triton kernels. After :meth:`build_tables`, in eval mode, every
    interaction filter is read from a cubic Hermite table instead of being
    evaluated, with a sync-free neighbor search (inference only; periodic
    cells as in :mod:`~mlcg.nn.kernels.pbc`). In train
    mode, or without tables, it runs the filter networks (trainable, double
    backward). ``train()`` and loading a state dict drop the tables.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.neighbors = RadiusCSR(float(self.rbf_layer.cutoff.cutoff_upper))
        self.register_load_state_dict_post_hook(_drop_tables_hook)

    def build_tables(
        self,
        num_knots: Optional[int] = None,
        tolerance: float = TABLE_TOLERANCE,
    ) -> "FlashSchNet":
        """Tabulate every interaction filter from the current weights
        (float64, cubic Hermite over the cutoff), each on the fewest
        power-of-two knots that keep its relative error within ``tolerance``,
        or on ``num_knots`` knots if given. Raises ``ValueError`` if a filter
        can't meet ``tolerance`` (e.g. non-smooth activations) or if
        ``tolerance`` is below the float32 floor. The tables reflect the
        weights at this call: weights changed by hand afterwards need another
        call (``train()`` and ``load_state_dict`` drop the tables)."""
        check_tolerance(tolerance)
        cutoff_upper = float(self.rbf_layer.cutoff.cutoff_upper)
        tables = [
            _filter_table(
                block.conv, self.rbf_layer, cutoff_upper, num_knots, tolerance
            )
            for block in self.interaction_blocks
        ]
        for block, table in zip(self.interaction_blocks, tables):
            block.conv.table = table
        return self

    def drop_tables(self):
        for block in self.interaction_blocks:
            block.conv.table = None

    @property
    def tabulated(self) -> bool:
        return self.interaction_blocks[0].conv.table is not None

    def train(self, mode: bool = True) -> "FlashSchNet":
        if mode:
            self.drop_tables()
        return super().train(mode)

    def forward(self, data: AtomicData) -> AtomicData:
        if not self.training and self.tabulated:
            return self._forward_tabulated(data)
        return self._forward_filters(data)

    def _neighbor_graph(self, data: AtomicData):
        """Sync-free CSR radius graph (minimum image with a cell), the
        per-graph cells and the ``valid`` factor (1, or NaN)."""
        ptr = (
            data.ptr
            if hasattr(data, "ptr")
            else data.pos.new_tensor([0, data.pos.shape[0]], dtype=torch.long)
        )
        periodic = periodic_cells(
            data.get("cell"), data.get("pbc"), ptr.numel() - 1, data.pos.device
        )
        row_ptr, col, valid = self.neighbors(
            data.pos.detach(), data.batch, ptr, *periodic
        )
        return row_ptr, col, valid, periodic

    def _forward_tabulated(self, data: AtomicData) -> AtomicData:
        row_ptr, col, valid, periodic = self._neighbor_graph(data)
        edges = (row_ptr, col, self.neighbors.cutoff, data.batch, *periodic)
        x = self.embedding_layer(data.atom_types)
        for block in self.interaction_blocks:
            x = x + block.tabulated(x, data.pos, *edges)
        return self._readout(x, data, valid)

    def _forward_filters(self, data: AtomicData) -> AtomicData:
        r"""Forward pass through the SchNet architecture.

        Parameters
        ----------
        data:
            Input data object containing batch atom/bead positions
            and atom/bead types.

        Returns
        -------
        data:
           Data dictionary, updated with predicted energy of shape
           (num_examples * num_atoms, 1), as well as neighbor list
           information.
        """

        neighbor_list = data.neighbor_list.get(self.name)
        if not self.is_nl_compatible(neighbor_list):
            neighbor_list = self.neighbor_list(
                data, self.rbf_layer.cutoff.cutoff_upper
            )[self.name]
        edge_index = neighbor_list["index_mapping"]
        distances = compute_distances(
            data.pos, edge_index, neighbor_list["cell_shifts"]
        )
        x = self.embedding_layer(data.atom_types)

        rbf_expansion = self.rbf_layer(distances)

        csr_data = build_csr_representation_from_edges(edge_index, x.shape[0])

        for block in self.interaction_blocks:
            x = x + block(
                x,
                edge_index,
                distances,
                rbf_expansion,
                csr_data=csr_data,
            )

        return self._readout(x, data)

    @torch.no_grad()
    def neighbor_list(
        self, data: AtomicData, rcut: float, max_num_neighbors: int = 1000
    ) -> dict:
        """mlcg neighbour list of the radius graph (``self.neighbors``, no
        neighbour cap), with minimum-image ``cell_shifts``. Raises
        ``ValueError`` for an unsupported periodic cell."""
        if float(rcut) != self.neighbors.cutoff:
            raise ValueError(f"rcut {rcut} != cutoff {self.neighbors.cutoff}")
        check_cells(data.get("cell"), data.get("pbc"), self.neighbors.cutoff)
        row_ptr, col, _, (cell, pbc) = self._neighbor_graph(data)
        num_edges = int(row_ptr[-1])
        dst = torch.repeat_interleave(
            torch.arange(len(row_ptr) - 1, device=col.device),
            row_ptr.diff(),
            output_size=num_edges,
        )
        src = col[:num_edges].long()
        d = data.pos[dst] - data.pos[src]
        shifts = minimum_image_vectors(d, cell, pbc, data.batch[dst]) - d
        return {
            self.name: make_neighbor_list(
                tag=self.name,
                order=2,
                index_mapping=torch.stack([src, dst]),
                cell_shifts=shifts,
                rcut=float(rcut),
                self_interaction=False,
            )
        }

    def _readout(self, x, data, scale=1.0):
        energy = self.output_network(x, data) * scale
        num_graphs = data.ptr.numel() - 1 if hasattr(data, "ptr") else None
        energy = scatter(
            energy, data.batch, dim=0, reduce="sum", dim_size=num_graphs
        )
        data.out.setdefault(self.name, {}).update(
            {ENERGY_KEY: energy.flatten()}
        )
        return data


class FlashInteractionBlock(InteractionBlock):
    r"""
    See :class:`InteractionBlock` for the full API documentation.

    This implementation is used within :class:`FlashSchNet` to
    pass csr_data dictionary to FlashCFConv layer.
    """

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_attr: torch.Tensor,
        *args,
        csr_data: dict = None,
    ) -> torch.Tensor:
        r"""Forward pass through the interaction block.

        Parameters
        ----------
        x:
            Embedded features of shape (num_examples, num_atoms,
            hidden_channels)
        edge_index:
            Graph edge index tensor of shape (2, total_num_edges)
        edge_weight:
            Graph edge weight (eg, distances), of shape (total_num_edges)
        edge_attr:
            Graph edge attributes (eg, expanded distances), of shape
            (total_num_edges, num_rbf)
        csr_data:
            Optional dict with CSR data for segment reduce:
            {"dst_ptr", "csr_perm", "edge_src", "edge_dst"}

        Returns
        -------
        x:
            Updated embedded features of shape (num_examples * num_atoms,
            hidden_channels)
        """

        x = self.conv(x, edge_index, edge_weight, edge_attr, csr_data=csr_data)
        x = self.activation(x)
        x = self.lin(x)
        return x

    def tabulated(self, x, pos, *edges):
        return self.lin(self.activation(self.conv.tabulated(x, pos, *edges)))


class FlashCFConv(CFConv):
    r"""
    See :class:`CFConv` for the full API documentation.

    This implementation use :func:`fused_cfconv` to fuse
    the message passing operation performed by SchNet without
    materializing intermediate expanded representations, or, with a filter
    table (see :meth:`FlashSchNet.build_tables`), :func:`tabulated_cfconv`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("table", None, persistent=False)

    def tabulated(self, x, pos, row_ptr, col, cutoff_upper, batch, cell, pbc):
        x = tabulated_cfconv(
            self.lin1(x),
            pos,
            row_ptr,
            col,
            self.table,
            cutoff_upper,
            batch,
            cell,
            pbc,
        )
        return self.lin2(x)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_attr: torch.Tensor,
        csr_data: dict = None,
    ) -> torch.Tensor:
        r"""Forward pass through the continuous filter convolution.

        Parameters
        ----------
        x:
            Embedded features of shape (num_examples * num_atoms,
            hidden_channels)
        edge_index
            Graph edge index tensor of shape (2, total_num_edges)
        edge_weight:
            Graph edge weight (eg, distances), of shape (total_num_edges)
        edge_attr:
            Graph edge attributes (eg, expanded distances), of shape
            (total_num_edges, num_rbf)
        csr_data:
            Optional dict with CSR data for segment reduce:
            {"dst_ptr", "csr_perm", "src_ptr", "src_perm"}

        Returns
        -------
        x:
            Updated embedded features of shape (num_examples * num_atoms,
            hidden_channels)
        """

        x = self.lin1(x)
        num_nodes = x.shape[0]
        filter_out = self.filter_network(edge_attr)
        edge_weight = self.cutoff(edge_weight)

        x = fused_cfconv(
            x,
            filter_out,
            edge_weight,
            edge_index[0],
            edge_index[1],
            csr_data["dst_ptr"],
            csr_data["csr_perm"],
            num_nodes,
            csr_data["src_ptr"],
            csr_data["src_perm"],
        )

        x = self.lin2(x)
        return x


class StandardFlashSchNet(FlashSchNet):
    """Small wrapper class for :ref:`SchNet` to simplify the definition of the
    SchNet model through an input file. The upper distance cutoff attribute
    in is set by default to match the upper cutoff value in the cutoff function.
    This class uses custom triton operations to speed up the messages computation
    in the model CFConv and to reduce the memory associated with materialization
    of intermediate tensors.

    Parameters
    ----------
    rbf_layer:
        radial basis function used to project the distances :math:`r_{ij}`.
    cutoff:
        smooth cutoff function to supply to the CFConv
    output_hidden_layer_widths:
        List giving the number of hidden nodes of each hidden layer of the MLP
        used to predict the target property from the learned representation.
    hidden_channels:
        dimension of the learned representation, i.e. dimension of the embeding projection, convolution layers, and interaction block.
    embedding_size:
        dimension of the input embeddings (should be larger than :obj:`AtomicData.atom_types.max()+1`).
    num_filters:
        number of nodes of the networks used to filter the projected distances
    num_interactions:
        number of interaction blocks
    activation:
        activation function
    max_num_neighbors:
        The maximum number of neighbors to return for each atom in :obj:`data`.
        If the number of actual neighbors is greater than
        :obj:`max_num_neighbors`, returned neighbors are picked randomly.
    aggr:
        Aggregation scheme for continuous filter output. For all options,
        see `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html?highlight=MessagePassing#the-messagepassing-base-class>`_
        for more options.

    """

    def __init__(
        self,
        rbf_layer: torch.nn.Module,
        cutoff: torch.nn.Module,
        output_hidden_layer_widths: List[int],
        hidden_channels: int = 128,
        embedding_size: int = 100,
        num_filters: int = 128,
        num_interactions: int = 3,
        activation: torch.nn.Module = torch.nn.Tanh(),
        max_num_neighbors: int = 1000,
        aggr: str = "add",
    ):
        if num_interactions < 1:
            raise ValueError("At least one interaction block must be specified")

        if cutoff.cutoff_lower != rbf_layer.cutoff.cutoff_lower:
            warnings.warn(
                "Cutoff function lower cutoff, {}, and radial basis function "
                " lower cutoff, {}, do not match.".format(
                    cutoff.cutoff_lower, rbf_layer.cutoff.cutoff_lower
                )
            )
        if cutoff.cutoff_upper != rbf_layer.cutoff.cutoff_upper:
            warnings.warn(
                "Cutoff function upper cutoff, {}, and radial basis function "
                " upper cutoff, {}, do not match.".format(
                    cutoff.cutoff_upper, rbf_layer.cutoff.cutoff_upper
                )
            )

        if aggr != "add":
            raise NotImplementedError(
                f"Only aggr='add' is supported for FlashSchNet, you provided {aggr}"
            )

        embedding_layer = torch.nn.Embedding(embedding_size, hidden_channels)

        interaction_blocks = []
        for _ in range(num_interactions):
            filter_network = MLP(
                layer_widths=[rbf_layer.num_rbf, num_filters, num_filters],
                activation_func=activation,
                last_bias=False,
            )

            cfconv = FlashCFConv(
                filter_network,
                cutoff=cutoff,
                num_filters=num_filters,
                in_channels=hidden_channels,
                out_channels=hidden_channels,
            )
            block = FlashInteractionBlock(cfconv, hidden_channels, activation)
            interaction_blocks.append(block)
        output_layer_widths = (
            [hidden_channels] + output_hidden_layer_widths + [1]
        )
        output_network = MLP(
            output_layer_widths, activation_func=activation, last_bias=False
        )
        super(StandardFlashSchNet, self).__init__(
            embedding_layer,
            interaction_blocks,
            rbf_layer,
            output_network,
            max_num_neighbors=max_num_neighbors,
        )

    @classmethod
    def flash_from_standard(
        cls, standard_model: StandardSchNet
    ) -> "StandardFlashSchNet":
        """Class method to initialize a StandardFlashSchNet from a preexisting StandardSchNet model.

        Parameters
        ----------
        standard_model:
            A StandardSchNet model that will be used to create a
            StandardFlashSchNet with the same architecture and weights.

        Returns
        -------
        flash_model:
            A StandardFlashSchNet model with the same architecture and
            weights as the input standard_model.
        """

        if not isinstance(standard_model, StandardSchNet):
            raise ValueError(
                f"Expected input model of type StandardSchNet, but got {type(standard_model)}"
            )

        linear_layers = [
            l
            for l in standard_model.output_network.layers
            if isinstance(l, torch.nn.Linear)
        ]
        widths = [linear_layers[0].in_features] + [
            l.out_features for l in linear_layers
        ]

        instance = cls(
            rbf_layer=copy.deepcopy(standard_model.rbf_layer),
            cutoff=copy.deepcopy(
                standard_model.interaction_blocks[0].conv.cutoff
            ),
            output_hidden_layer_widths=widths[1:-1],
            hidden_channels=widths[0],
            embedding_size=standard_model.embedding_layer.num_embeddings,
            num_filters=standard_model.interaction_blocks[
                0
            ].conv.lin1.out_features,
            num_interactions=len(standard_model.interaction_blocks),
            activation=copy.deepcopy(
                standard_model.interaction_blocks[0].activation
            ),
            max_num_neighbors=standard_model.max_num_neighbors,
            aggr=standard_model.interaction_blocks[0].conv.aggr,
        )
        instance.load_state_dict(standard_model.state_dict(), strict=True)
        # the constructor builds every MLP with the block activation; copy the
        # source's own modules so that each filter keeps its activation
        for flash_block, block in zip(
            instance.interaction_blocks, standard_model.interaction_blocks
        ):
            flash_block.conv.filter_network = copy.deepcopy(
                block.conv.filter_network
            )
            flash_block.activation = copy.deepcopy(block.activation)
        instance.output_network = copy.deepcopy(standard_model.output_network)
        device = next(standard_model.parameters()).device
        return instance.to(device).train(standard_model.training)
