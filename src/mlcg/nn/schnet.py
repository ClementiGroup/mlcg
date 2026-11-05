import math
import warnings
from typing import Optional, List, Final
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
from ..neighbor_list.neighbor_list import (
    atomic_data2neighbor_list,
    validate_neighborlist,
)
from ..data.atomic_data import AtomicData, ENERGY_KEY
from ..geometry.internal_coordinates import compute_distances
from .mlp import MLP
from ._module_init import init_xavier_uniform
from .attention import (
    AttentiveInteractionBlock,
    AttentiveInteractionBlock2,
    Nonlocalinteractionblock,
)

from .radial_basis import RegularizedBasis
from .flash_models.mixin import SupportsFlash, TABLE_TOLERANCE, check_tolerance
from .kernels import (
    RadiusCSR,
    build_filter_table,
    fused_cfconv,
    tabulated_cfconv,
)
from .kernels.pbc import check_cells, minimum_image_vectors, periodic_cells

try:
    from mlcg_opt_radius.radius import radius_distance
except ImportError:
    print(
        "`mlcg_opt_radius` not installed. Please check the"
        + "`opt_radius` folder and follow the instructions."
    )
    radius_distance = None

FIRST_NUM_KNOTS = 2048
MAX_NUM_KNOTS = 16384


class SchNet(torch.nn.Module):
    r"""PyTorch Geometric implementation of SchNet
    Code adapted from [PT_geom_schnet]_  which is based on the architecture
    described in [Schnet]_ .

    Parameters
    ----------
    embedding_layer:
        Initial embedding layer that transforms atoms/coarse grain bead
        types into embedded features
    interaction_blocks: list of torch.nn.Module or torch.nn.Sequential
        Sequential interaction blocks of the model, where each interaction
        block applies
    rbf_layer:
        The set of radial basis functions that expands pairwise distances
        between atoms/CG beads.
    output_network:
        Output neural network that predicts scalar energies from SchNet
        features. This network should transform (num_examples * num_atoms,
        hidden_channels) to (num_examples * num atoms, 1).
    upper_distance_cutoff:
        Upper distance cutoff used for making neighbor lists.
    self_interaction:
        If True, self interactions/distancess are calculated. But it never
        had a function due to a bug in the implementation (see static method
        `neighbor_list`). Should be kept False. This option shall not be
        deleted for compatibility.
    max_num_neighbors:
        Maximum number of neighbors to return for a
        given node/atom when constructing the molecular graph during forward
        passes. This attribute is passed to the torch_cluster radius_graph
        routine keyword max_num_neighbors, which normally defaults to 32.
        Users should set this to higher values if they are using higher upper
        distance cutoffs and expect more than 32 neighbors per node/atom.
    nls_distance_method:
        Method for computing a neighbor list. Supported values are
        `torch`, `nvalchemi_naive`, `nvalchemi_cell`, `nvalchemi_raw`
        and `custom_kernel`.
    """

    name: Final[str] = "SchNet"

    def __init__(
        self,
        embedding_layer: torch.nn.Module,
        interaction_blocks: List[torch.nn.Module],
        rbf_layer: torch.nn.Module,
        output_network: torch.nn.Module,
        self_interaction: bool = False,
        max_num_neighbors: int = 1000,
        nls_distance_method: str = "torch",
    ):
        super(SchNet, self).__init__()

        self.embedding_layer = embedding_layer
        self.rbf_layer = rbf_layer
        self.max_num_neighbors = max_num_neighbors
        if self_interaction:
            raise NotImplementedError(
                "The option `self_interaction` did not have function due to a bug. It only exists for compatibility and should stay `False`."
            )
        self.self_interaction = self_interaction

        if isinstance(interaction_blocks, List):
            self.interaction_blocks = torch.nn.Sequential(*interaction_blocks)
        elif isinstance(interaction_blocks, InteractionBlock):
            self.interaction_blocks = torch.nn.Sequential(interaction_blocks)
        else:
            raise RuntimeError(
                "interaction_blocks must be a single InteractionBlock or "
                "a list of InteractionBlocks."
            )
        self.nls_distance_method = nls_distance_method
        self.output_network = output_network
        self.reset_parameters()

    def reset_parameters(self):
        """Method for resetting linear layers in each SchNet component"""
        self.embedding_layer.reset_parameters()
        self.rbf_layer.reset_parameters()
        for block in self.interaction_blocks:
            block.reset_parameters()
        self.output_network.reset_parameters()

    def forward(self, data: AtomicData) -> AtomicData:
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
        x = self.embedding_layer(data.atom_types)

        neighbor_list = data.neighbor_list.get(self.name)

        if not self.is_nl_compatible(neighbor_list):
            neighbor_list = self.neighbor_list(
                data,
                self.rbf_layer.cutoff.cutoff_upper,
                self.max_num_neighbors,
            )[self.name]

        edge_index = neighbor_list["index_mapping"]
        distances = compute_distances(
            data.pos,
            edge_index,
            neighbor_list["cell_shifts"],
        )

        rbf_expansion = self.rbf_layer(distances)
        num_batch = data.batch[-1] + 1
        for block in self.interaction_blocks:
            x = x + block(
                x, edge_index, distances, rbf_expansion, num_batch, data.batch
            )

        energy = self.output_network(x, data)
        num_graphs = data.ptr.numel() - 1 if hasattr(data, "ptr") else None
        energy = scatter(
            energy, data.batch, dim=0, reduce="sum", dim_size=num_graphs
        )
        energy = energy.flatten()
        data.out.setdefault(self.name, {}).update({ENERGY_KEY: energy})

        return data

    def is_nl_compatible(self, nl):
        is_compatible = False
        if validate_neighborlist(nl):
            if (
                nl["order"] == 2
                and nl["self_interaction"] == False
                and nl["rcut"] == self.cutoff.cutoff_upper
            ):
                is_compatible = True
        return is_compatible

    def neighbor_list(
        self,
        data: AtomicData,
        rcut: float,
        max_num_neighbors: int = 1000,
    ) -> dict:
        """Computes the neighborlist for :obj:`data` using a strict cutoff of :obj:`rcut`."""
        if not hasattr(self, "nls_distance_method"):
            self.nls_distance_method = "torch"
        return {
            SchNet.name: atomic_data2neighbor_list(
                data,
                rcut,
                self_interaction=False,
                max_num_neighbors=max_num_neighbors,
                nls_distance_method=self.nls_distance_method,
            )
        }


class InteractionBlock(torch.nn.Module):
    r"""Interaction blocks for SchNet. Consists of atomwise
    transformations of embedded features that are continuously
    convolved with filters generated from radial basis function-expanded
    pairwise distances.

    Parameters
    ----------
    cfconv_layer:
        Continuous filter convolution layer for convolutions of radial basis
        function-expanded distances with embedded features
    hidden_channels:
        Hidden dimension of embedded features
    activation:
        Activation function applied to linear layer outputs
    """

    def __init__(
        self,
        cfconv_layer: torch.nn.Module,
        hidden_channels: int = 128,
        activation: torch.nn.Module = torch.nn.Tanh(),
    ):
        super(InteractionBlock, self).__init__()
        self.conv = cfconv_layer
        self.activation = activation
        self.lin = torch.nn.Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
        init_xavier_uniform(self.lin)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        *args,
        flash: Optional[tuple] = None,
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
        flash:
            Flash path only (keyword), in place of the edge arguments: the
            graph ``(row_ptr, col, features, pos, batch, cell, pbc,
            cutoff)``, where ``row_ptr``, ``col`` are the destination CSR of
            a *symmetric* neighbour list (int32) and ``features`` is
            ``(distances, rbf)`` in row order, or ``None`` when the filter
            table is used (see :meth:`CFConv.flash_forward`).

        Returns
        -------
        x:
            Updated embedded features of shape (num_examples * num_atoms,
            hidden_channels)
        """
        if flash is None:
            x = self.conv(x, edge_index, edge_weight, edge_attr)
        else:
            x = self.conv(x, flash=flash)
        x = self.activation(x)
        x = self.lin(x)
        return x

    def has_flash_path(self) -> bool:
        """Whether this block takes ``flash=``: its ``forward`` is this
        class's and its convolution has a flash path."""
        return (
            type(self).forward is InteractionBlock.forward
            and isinstance(self.conv, CFConv)
            and self.conv.has_flash_path()
        )


class CFConv(SupportsFlash, MessagePassing):
    r"""Continuous filter convolutions for `SchNet`.

    Parameters
    ----------
    filter_net:
        Neural network for generating filters from expanded pairwise distances
    cutoff:
        Cutoff envelope to apply to the output of the filter generating network.
    in_channels:
        Hidden input dimensions
    out_channels:
        Hidden output dimensions
    num_filters:
        Number of filters
    aggr:
        Aggregation scheme for continuous filter output. For all options,
        see `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html?highlight=MessagePassing#the-messagepassing-base-class>`.

    Flash path (:class:`~mlcg.nn.flash_models.SupportsFlash`, ``aggr="add"``
    only), when called with ``flash=`` (see :meth:`flash_forward`): in eval
    mode with a table (:meth:`build_table`), :func:`tabulated_cfconv` reads
    the filter from it; otherwise :func:`fused_cfconv` fuses the message
    passing on the given edge features, without materializing per-edge
    messages.
    """

    _runtime_attributes = ("_kernels", "_table")

    def __init__(
        self,
        filter_network: torch.nn.Module,
        cutoff: torch.nn.Module,
        in_channels: int = 128,
        out_channels: int = 128,
        num_filters: int = 128,
        aggr: str = "add",
    ):
        super(CFConv, self).__init__(aggr=aggr)
        self.lin1 = torch.nn.Linear(in_channels, num_filters, bias=False)
        self.lin2 = torch.nn.Linear(num_filters, out_channels)
        self.filter_network = filter_network
        self.cutoff = cutoff
        self.reset_parameters()

    def reset_parameters(self):
        r"""Method for resetting the weights of the linear
        layers and filter network according the the
        Xavier uniform strategy. Biases
        are set to 0.
        """

        self.filter_network.reset_parameters()
        init_xavier_uniform(self.lin1)
        init_xavier_uniform(self.lin2)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        flash: Optional[tuple] = None,
    ) -> torch.Tensor:
        r"""Forward pass through the continuous filter convolution.

        Parameters
        ----------
        x:
            Embedded features of shape (num_examples * num_atoms,
            hidden_channels)
        edge_index:
            Graph edge index tensor of shape (2, total_num_edges)
        edge_weight:
            Graph edge weight (eg, distances), of shape (total_num_edges)
        edge_attr:
            Graph edge attributes (eg, expanded distances), of shape
            (total_num_edges, num_rbf)
        flash:
            Flash path only (keyword), in place of the edge arguments: the
            graph ``(row_ptr, col, features, pos, batch, cell, pbc,
            cutoff)``, where ``row_ptr``, ``col`` are the destination CSR of
            a *symmetric* neighbour list (int32) and ``features`` is
            ``(distances, rbf)`` in row order, or ``None`` when the filter
            table is used (see :meth:`CFConv.flash_forward`).

        Returns
        -------
        x:
            Updated embedded features of shape (num_examples * num_atoms,
            hidden_channels)
        """
        if flash is not None:
            return self.flash_forward(x, *flash)
        C = self.cutoff(edge_weight)
        W = self.filter_network(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        # propagate_type: (x: Tensor, W: Tensor)
        # Perform the continuous filter convolution
        x = self.propagate(edge_index, x=x, W=W, size=None)
        x = self.lin2(x)
        return x

    def message(self, x_j: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        r"""Message passing operation to perform the continuous filter
        convolution through element-wise multiplcation of embedded
        features with the output of the filter network.

        Parameters
        ----------
        x_j:
            Tensor of embedded features of shape (total_num_edges,
            hidden_channels)
        W:
            Tensor of filter values of shape (total_num_edges, num_filters)

        Returns
        -------
        x_j * W:
            Elementwise multiplication of the filters with embedded features.
        """
        return x_j * W

    def has_flash_path(self) -> bool:
        """Whether :meth:`flash_forward` computes what ``forward`` does: no
        ``forward`` override without its own ``flash_forward``, ``message``
        is this class's and ``aggr="add"``."""
        return (
            self._flash_forward_is_current()
            and type(self).message is CFConv.message
            and self.aggr == "add"
        )

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


class StandardSchNet(SupportsFlash, SchNet):
    """Small wrapper class for :ref:`mlcg.nn.SchNet` to simplify the definition of the
    SchNet model through an input file. The upper distance cutoff attribute
    in is set by default to match the upper cutoff value in the cutoff function.

    Flash path (:class:`~mlcg.nn.flash_models.SupportsFlash`): with
    :meth:`enable_kernels`, the radius graph comes from a sync-free
    neighbour search (a CSR) and every block runs fused Triton kernels
    (trainable, double backward). After
    :meth:`prepare_inference`, in eval mode, the filters are read from cubic
    Hermite tables instead of being evaluated (inference only; periodic
    cells as in :mod:`~mlcg.nn.kernels.pbc`). ``train()`` and loading a
    state dict drop the tables. Needs every block to be an
    :class:`InteractionBlock` with a flash path
    (:meth:`InteractionBlock.has_flash_path`).

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
    nls_distance_method:
        Method for computing a neighbor list. Supported values are
        `torch`, `nvalchemi_naive`, `nvalchemi_cell`, `nvalchemi_raw`
        and `custom_kernel`.

    """

    _runtime_attributes = ("_kernels", "neighbors")

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
        nls_distance_method: str = "torch",
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

        embedding_layer = torch.nn.Embedding(embedding_size, hidden_channels)

        interaction_blocks = []
        for _ in range(num_interactions):
            filter_network = MLP(
                layer_widths=[rbf_layer.num_rbf, num_filters, num_filters],
                activation_func=activation,
                last_bias=False,
            )

            cfconv = CFConv(
                filter_network,
                cutoff=cutoff,
                num_filters=num_filters,
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                aggr=aggr,
            )
            block = InteractionBlock(cfconv, hidden_channels, activation)
            interaction_blocks.append(block)
        output_layer_widths = (
            [hidden_channels] + output_hidden_layer_widths + [1]
        )
        output_network = MLP(
            output_layer_widths, activation_func=activation, last_bias=False
        )
        super(StandardSchNet, self).__init__(
            embedding_layer,
            interaction_blocks,
            rbf_layer,
            output_network,
            max_num_neighbors=max_num_neighbors,
            nls_distance_method=nls_distance_method,
        )

    def forward(self, data: AtomicData) -> AtomicData:
        if self._kernels:
            return self.flash_forward(data)
        return super().forward(data)

    def enable_kernels(self, enabled: bool = True):
        """Switches the flash path of every block on or off. Refused, with a
        warning, unless every block is an :class:`InteractionBlock` with a
        flash path (and as in :meth:`SupportsFlash.enable_kernels`)."""
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
        warning. Raises ``ValueError`` for a periodic cell the flash path
        can't handle (see :func:`~mlcg.nn.kernels.pbc.check_cells`)."""
        if not self._kernels:
            return
        check_cells(data.get("cell"), data.get("pbc"), self.neighbors.cutoff)
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
            data.pos.detach(), data.batch, ptr, cell, pbc
        )
        return row_ptr, col, valid, cell, pbc

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


class AttentiveSchNet(SchNet):
    """Small wrapper class for :ref:`mlcg.nn.SchNet` to simplify the definition of the
    SchNet model with an Interaction block that includes attention through an input file. The upper distance cutoff attribute
    in is set by default to match the upper cutoff value in the cutoff function.

    Parameters
    ----------
    rbf_layer:
        radial basis function used to project the distances :math:`r_{ij}`.
    cutoff:
        smooth cutoff function to supply to the CFConv
    output_hidden_layer_widths:
        List giving the number of hidden nodes of each hidden layer of the MLP
        used to predict the target property from the learned representation.
    num_features_in:
        size of each input sample for linear layer
    num_features_out:
        size of each output sample for liner layer
    num_residuals_q, num_residuals_k, num_residuals_v:
        Number of residual blocks applied to features via self-attention
        for queries, keys, and values
    attention_block:
        Specify if you want to use softmax attention (input: ExactAttention) or favor+ (input: FavorAttention)
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
    activation_first:
        Inverting the order of linear layers and activation functions.
    attention_version:
        Specifiy which interaction block architecture to choose. By default normal.
    nls_distance_method:
        Method for computing a neighbor list. Supported values are
        `torch`, `nvalchemi_naive`, `nvalchemi_cell`, `nvalchemi_raw`
        and `custom_kernel`.

    """

    def __init__(
        self,
        rbf_layer: torch.nn.Module,
        cutoff: torch.nn.Module,
        output_hidden_layer_widths: List[int],
        num_features_in: int,
        num_features_out: int,
        num_residual_q: int,
        num_residual_k: int,
        num_residual_v: int,
        attention_block: torch.nn.Module,
        hidden_channels: int = 128,
        embedding_size: int = 100,
        num_filters: int = 128,
        num_interactions: int = 3,
        activation_func: torch.nn.Module = torch.torch.nn.Tanh(),
        max_num_neighbors: int = 1000,
        layer_widths: List[int] = None,
        activation_first: bool = False,
        aggr: str = "add",
        attention_version: str = "normal",
        nls_distance_method: str = "torch",
    ):
        if layer_widths is None:
            layer_widths = [128, 128]
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

        embedding_layer = torch.nn.Embedding(embedding_size, hidden_channels)

        if attention_version == "normal":
            attention_cls = AttentiveInteractionBlock
        elif attention_version == "2":
            attention_cls = AttentiveInteractionBlock2
        else:
            raise RuntimeError("attention_version not recognized")

        interaction_blocks = []
        for _ in range(num_interactions):
            filter_network = MLP(
                layer_widths=[rbf_layer.num_rbf, num_filters, num_filters],
                activation_func=activation_func,
            )

            cfconv = CFConv(
                filter_network,
                cutoff=cutoff,
                num_filters=num_filters,
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                aggr=aggr,
            )

            if all(
                [
                    arg != None
                    for arg in [
                        num_features_in,
                        num_features_out,
                        num_residual_q,
                        num_residual_k,
                        num_residual_v,
                        attention_block,
                    ]
                ]
            ):
                int_block = Nonlocalinteractionblock(
                    num_features_in,
                    num_features_out,
                    num_residual_q,
                    num_residual_k,
                    num_residual_v,
                    attention_block,
                )
            elif all(
                [
                    arg == None
                    for arg in [
                        num_features_in,
                        num_features_out,
                        num_residual_q,
                        num_residual_k,
                        num_residual_v,
                        attention_block,
                    ]
                ]
            ):
                int_block = None
            else:
                raise ValueError(
                    "To use Attention, you must specify 'num_features_in','num_features_out','num_residual_q','num_residual_k', 'num_residual_v' and 'attention_block', but only {} was specified".format(
                        [
                            arg
                            for arg in [
                                num_features_in,
                                num_features_out,
                                num_residual_q,
                                num_residual_k,
                                num_residual_v,
                                attention_block,
                            ]
                            if arg != None
                        ]
                    )
                )

            block = attention_cls(
                cfconv_layer=cfconv,
                hidden_channels=hidden_channels,
                activation_func=activation_func,
                attention_block=int_block,
            )

            interaction_blocks.append(block)
        output_layer_widths = (
            [hidden_channels] + output_hidden_layer_widths + [1]
        )
        output_network = MLP(
            output_layer_widths,
            activation_func=activation_func,
            last_bias=False,
        )
        super(AttentiveSchNet, self).__init__(
            interaction_blocks=interaction_blocks,
            rbf_layer=rbf_layer,
            embedding_layer=embedding_layer,
            output_network=output_network,
            max_num_neighbors=max_num_neighbors,
            nls_distance_method=nls_distance_method,
        )


class RBFRegularizedSchNet(StandardSchNet):
    __doc = StandardSchNet.__doc__
    __params_onwards = __doc[__doc.index("Parameters") :]
    __doc__ = (
        """
    This is a StandardSchNet model where the RBF components are weighted via Hadamard product with couple-specific
    vectors. Each tuple of bead types has its own vector.
    The models also adds the key radial_filters to the output dictionary, containing the regularization parameters.
    Flash path (:meth:`flash_forward`): the exact fused kernels only, without
    filter tables, since the filters depend on the bead types.

    """
        + __params_onwards
        + """
    independent_regularizations (bool):
        If True each interaction block has its own set of regularization parameters,
        otherwise the regularization parameters are shared across all interaction blocks.
    """
    )

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
        nls_distance_method: str = "torch",
        independent_regularizations: bool = False,
    ):

        rbf_layer = RegularizedBasis(
            basis_function=rbf_layer,
            types=embedding_size,
            n_basis_set=num_interactions,
            independent_regularizations=independent_regularizations,
        )
        super(RBFRegularizedSchNet, self).__init__(
            rbf_layer=rbf_layer,
            cutoff=cutoff,
            output_hidden_layer_widths=output_hidden_layer_widths,
            hidden_channels=hidden_channels,
            embedding_size=embedding_size,
            num_filters=num_filters,
            num_interactions=num_interactions,
            activation=activation,
            max_num_neighbors=max_num_neighbors,
            aggr=aggr,
            nls_distance_method=nls_distance_method,
        )

    def forward(self, data: AtomicData) -> AtomicData:
        r"""Forward pass through the SchNet architecture with independent regularizations."""
        if self._kernels:
            return self.flash_forward(data)
        x = self.embedding_layer(data.atom_types)

        neighbor_list = data.neighbor_list.get(self.name)

        if not self.is_nl_compatible(neighbor_list):
            neighbor_list = self.neighbor_list(
                data,
                self.rbf_layer.cutoff.cutoff_upper,
                self.max_num_neighbors,
            )[self.name]

        edge_index = neighbor_list["index_mapping"]
        distances = compute_distances(
            data.pos,
            edge_index,
            neighbor_list["cell_shifts"],
        )

        rbf_expansion = self.rbf_layer(
            distances,
            data.atom_types[edge_index[0]],
            data.atom_types[edge_index[1]],
        )
        num_batch = data.batch[-1] + 1
        for block_id, block in enumerate(self.interaction_blocks):
            x = x + block(
                x,
                edge_index,
                distances,
                rbf_expansion[block_id],
                num_batch,
                data.batch,
            )

        energy = self.output_network(x, data)
        energy = scatter(energy, data.batch, dim=0, reduce="sum")
        energy = energy.flatten()
        data.out[self.name] = {
            ENERGY_KEY: energy,
            "radial_filters": self.rbf_layer.get_regularization_parameters(),
        }

        return data

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
            check_cells(
                data.get("cell"), data.get("pbc"), self.neighbors.cutoff
            )

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
