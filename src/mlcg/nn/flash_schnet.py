import logging
import os
import warnings
from typing import Optional, List, Final
import torch
import nvtx
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

logger = logging.getLogger(__name__)

try:
    from mlcg_opt_radius.radius import radius_distance
except ImportError:
    logger.info(
        "mlcg_opt_radius not installed. Please check the `opt_radius` folder and follow the instructions."
    )
    radius_distance = None

try:
    from flashmd.kernels.cfconv_kernels import (
        fused_cutoff_gather_multiply_scatter_autograd,
        fused_distance_gaussian_rbf_cutoff_autograd,
        fused_tanh_linear_autograd,
    )
    from flashmd.kernels.csr_kernels import (
        build_csr_index,
        build_src_csr_index,
        fused_csr_cfconv_autograd,
    )

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    fused_cutoff_gather_multiply_scatter_autograd = None
    fused_distance_gaussian_rbf_cutoff_autograd = None
    fused_tanh_linear_autograd = None
    build_csr_index = None
    build_src_csr_index = None
    fused_csr_cfconv_autograd = None

# All Triton optimizations are ON by default. Use --disable_optim CLI flag
# or set individual env vars to "0" to disable.
USE_TRITON_MESSAGE_PASSING = (
    os.environ.get("MLCG_USE_TRITON_MESSAGE_PASSING", "1") == "1"
)
USE_FUSED_RBF = os.environ.get("MLCG_USE_FUSED_RBF", "1")
USE_FUSED_TANH_LINEAR = os.environ.get("MLCG_USE_FUSED_TANH_LINEAR", "1") == "1"
USE_CSR = os.environ.get("MLCG_USE_CSR", "1") == "1"
USE_SRC_CSR_GRAD_X = os.environ.get("MLCG_USE_SRC_CSR_GRAD_X", "1") == "1"

# Log Triton optimization status
_all_on = (
    USE_TRITON_MESSAGE_PASSING
    and USE_FUSED_RBF == "1"
    and USE_FUSED_TANH_LINEAR
    and USE_CSR
    and USE_SRC_CSR_GRAD_X
)
_all_off = (
    not USE_TRITON_MESSAGE_PASSING
    and USE_FUSED_RBF == "0"
    and not USE_FUSED_TANH_LINEAR
    and not USE_CSR
    and not USE_SRC_CSR_GRAD_X
)
if not TRITON_AVAILABLE:
    print("[flashmd] Triton not available, falling back to PyTorch")
elif _all_off:
    print("[flashmd] Triton optimizations disabled")
elif _all_on:
    print("[flashmd] All Triton optimizations enabled")
else:
    _enabled = []
    if USE_TRITON_MESSAGE_PASSING:
        _enabled.append("cfconv")
    if USE_FUSED_RBF == "1":
        _enabled.append("fused_rbf")
    if USE_FUSED_TANH_LINEAR:
        _enabled.append("fused_tanh_linear")
    if USE_CSR:
        _enabled.append("csr")
    if USE_SRC_CSR_GRAD_X:
        _enabled.append("src_csr_grad_x")
    print(f"[flashmd] Triton optimizations: {', '.join(_enabled)}")


class FlashSchNet(torch.nn.Module):
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
        super().__init__()

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

        # Cache gamma value to avoid GPU-CPU sync in forward
        # coeff is a buffer (not trainable), so this is safe to cache at init
        self._cached_gamma = None

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
        edge_src = edge_index[0].contiguous()
        edge_dst = edge_index[1].contiguous()
        pos_contiguous = data.pos.contiguous()

        # Extract GaussianBasis parameters
        centers = self.rbf_layer.offset
        # Use cached gamma to avoid GPU-CPU sync
        if self._cached_gamma is None:
            self._cached_gamma = self.rbf_layer.coeff.item()
        gamma = self._cached_gamma
        cutoff_upper = self.rbf_layer.cutoff.cutoff_upper

        distances, rbf_expansion = fused_distance_gaussian_rbf_cutoff_autograd(
            pos_contiguous,
            edge_src,
            edge_dst,
            centers,
            gamma,
            cutoff_upper,
        )

        num_batch = data.batch[-1] + 1

        # Build CSR index if enabled (once, reused across all interaction blocks)
        csr_data = None
        need_dst_csr = USE_CSR
        need_src_csr = USE_SRC_CSR_GRAD_X
        if (
            (need_dst_csr or need_src_csr)
            and TRITON_AVAILABLE
            and build_csr_index is not None
            and x.is_cuda
        ):
            with nvtx.annotate("SchNet_build_csr", color="magenta"):
                edge_src = edge_index[0].contiguous()
                edge_dst = edge_index[1].contiguous()
                num_nodes = x.shape[0]

                csr_data = {
                    "edge_src": edge_src,
                    "edge_dst": edge_dst,
                }

                # Build dst-CSR for forward scatter (if USE_CSR)
                if need_dst_csr:
                    dst_ptr, csr_perm = build_csr_index(edge_dst, num_nodes)
                    csr_data["dst_ptr"] = dst_ptr
                    csr_data["csr_perm"] = csr_perm

                # Build src-CSR for backward grad_x (if USE_SRC_CSR_GRAD_X)
                if need_src_csr and build_src_csr_index is not None:
                    src_ptr, src_perm = build_src_csr_index(edge_src, num_nodes)
                    csr_data["src_ptr"] = src_ptr
                    csr_data["src_perm"] = src_perm

        for i, block in enumerate(self.interaction_blocks):

            x = x + block(
                x,
                edge_index,
                distances,
                rbf_expansion,
                num_batch,
                data.batch,
                csr_data=csr_data,
            )

            energy = self.output_network(x, data)

        num_graphs = data.ptr.numel() - 1 if hasattr(data, "ptr") else None
        energy = scatter(
            energy, data.batch, dim=0, reduce="sum", dim_size=num_graphs
        )
        energy = energy.flatten()

        data.out[self.name] = {ENERGY_KEY: energy}

        # Store edge_index if dump_neighbor_list flag is set
        if getattr(data, "_dump_neighbor_list", False):
            data.out[self.name]["edge_index"] = edge_index

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

    def _should_use_fused_rbf(self, pos: torch.Tensor, num_edges: int) -> bool:
        """
        Determine if the fused distance+RBF+cutoff kernel should be used.

        The fused kernel is beneficial for large edge counts (>100k edges) as it
        eliminates intermediate memory traffic by computing distance, Gaussian RBF,
        and cosine cutoff in a single kernel pass.

        Conditions for using fused kernel:
        1. Triton kernels must be available
        2. Data must be on CUDA
        3. rbf_layer must be GaussianBasis (checked by having 'offset' and 'coeff' attrs)
        4. rbf_layer.cutoff must be CosineCutoff with cutoff_lower=0
        5. Either MLCG_USE_FUSED_RBF=1 (force enable) or edge count > threshold

        Returns
        -------
        bool
            True if fused kernel should be used, False otherwise
        """
        # Check Triton availability
        if not TRITON_AVAILABLE:
            return False

        # Check if fused kernel function is available
        if fused_distance_gaussian_rbf_cutoff_autograd is None:
            return False

        # Check CUDA
        if not pos.is_cuda:
            return False

        # Check rbf_layer compatibility (must be GaussianBasis)
        if not hasattr(self.rbf_layer, "offset") or not hasattr(
            self.rbf_layer, "coeff"
        ):
            return False

        # Check cutoff compatibility (must be CosineCutoff with cutoff_lower=0)
        # Import here to avoid circular imports
        from .cutoff import CosineCutoff

        cutoff = self.rbf_layer.cutoff
        if not isinstance(cutoff, CosineCutoff):
            return False
        if cutoff.cutoff_lower != 0:
            return False

        # Check environment variable
        if USE_FUSED_RBF == "1":
            # Force enabled
            return True
        elif USE_FUSED_RBF == "0":
            # Force disabled
            return False
        else:
            # Auto mode: enable if edge count exceeds threshold
            return num_edges > FUSED_RBF_EDGE_THRESHOLD

    @staticmethod
    def neighbor_list(
        data: AtomicData, rcut: float, max_num_neighbors: int = 1000
    ) -> dict:
        """Computes the neighborlist for :obj:`data` using a strict cutoff of :obj:`rcut`."""
        return {
            SchNet.name: atomic_data2neighbor_list(
                data,
                rcut,
                self_interaction=False,
                max_num_neighbors=max_num_neighbors,
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

        # Check if fused tanh+linear kernel is available and enabled
        # Only works with Tanh activation
        self.use_fused_tanh_linear = (
            TRITON_AVAILABLE
            and USE_FUSED_TANH_LINEAR
            and fused_tanh_linear_autograd is not None
            and isinstance(activation, torch.nn.Tanh)
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
        init_xavier_uniform(self.lin)

    @nvtx.annotate("InteractionBlock.forward", color="teal")
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

        with nvtx.annotate("InteractionBlock_cfconv", color="lime"):
            x = self.conv(
                x, edge_index, edge_weight, edge_attr, csr_data=csr_data
            )

        if self.use_fused_tanh_linear and x.is_cuda:
            # Fused path: tanh + linear in single kernel
            # y = tanh(x) @ W.T + b  (Linear stores weight as [out, in])
            with nvtx.annotate(
                "InteractionBlock_fused_tanh_linear", color="olive"
            ):
                # Linear layer weight is [out_features, in_features], need to transpose
                weight = self.lin.weight.t().contiguous()
                bias = self.lin.bias
                x = fused_tanh_linear_autograd(x.contiguous(), weight, bias)
        else:
            # Standard path: separate tanh and linear
            with nvtx.annotate("InteractionBlock_activation", color="olive"):
                x = self.activation(x)
            with nvtx.annotate("InteractionBlock_linear", color="navy"):
                x = self.lin(x)
        return x


class CFConv(MessagePassing):
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
    use_triton:
        Whether to use the fused Triton kernel for message passing (default: True).
    """

    def __init__(
        self,
        filter_network: torch.nn.Module,
        cutoff: torch.nn.Module,
        in_channels: int = 128,
        out_channels: int = 128,
        num_filters: int = 128,
        aggr: str = "add",
        use_triton: bool = True,
    ):
        super(CFConv, self).__init__(aggr=aggr)
        self.lin1 = torch.nn.Linear(in_channels, num_filters, bias=False)
        self.lin2 = torch.nn.Linear(num_filters, out_channels)
        self.filter_network = filter_network
        self.cutoff = cutoff
        self.use_triton = (
            use_triton and TRITON_AVAILABLE and USE_TRITON_MESSAGE_PASSING
        )
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

    @nvtx.annotate("CFConv.forward", color="aqua")
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
            {"dst_ptr", "csr_perm", "edge_src", "edge_dst"}

        Returns
        -------
        x:
            Updated embedded features of shape (num_examples * num_atoms,
            hidden_channels)
        """
        # ===== CFConv Computational Flow =====
        # Input shapes:
        #   x: [num_nodes, hidden_channels]
        #   edge_index: [2, num_edges] - (src, dst) pairs
        #   edge_weight: [num_edges] - distances
        #   edge_attr: [num_edges, num_rbf] - RBF expanded distances
        #
        # Step 1: Linear projection
        with nvtx.annotate("CFConv_lin1", color="coral"):
            x = self.lin1(x)  # [num_nodes, num_filters]

        # Step 2-5: Message passing (propagate_type: x: Tensor, W: Tensor)
        with nvtx.annotate("CFConv_message_passing", color="crimson"):
            if (
                csr_data is not None
                and "dst_ptr" in csr_data
                and fused_csr_cfconv_autograd is not None
            ):
                # === CSR Segment Reduce Path ===
                # Uses CSR format to avoid atomics in scatter-add
                num_nodes = x.shape[0]

                # Step 2a: Filter network (MLP, not fused)
                with nvtx.annotate("CFConv_filter_network", color="salmon"):
                    filter_out = self.filter_network(edge_attr)

                # Steps 2b-5: CSR segment reduce (no atomics!)
                cutoff_upper = self.cutoff.cutoff_upper

                # Get src-CSR for backward if available (USE_SRC_CSR_GRAD_X=1)
                src_ptr = csr_data.get("src_ptr")
                src_perm = csr_data.get("src_perm")

                x = fused_csr_cfconv_autograd(
                    x.contiguous(),
                    filter_out.contiguous(),
                    edge_weight.contiguous(),
                    csr_data["edge_src"],
                    csr_data["edge_dst"],
                    csr_data["dst_ptr"],
                    csr_data["csr_perm"],
                    num_nodes,
                    cutoff_upper,
                    src_ptr,
                    src_perm,
                )
            elif self.use_triton and x.is_cuda:
                # === Triton Optimized Path (atomic scatter) ===
                # Fuses steps 2-5 into single kernel (except filter_network MLP)
                num_nodes = x.shape[0]
                edge_src = edge_index[0].contiguous()
                edge_dst = edge_index[1].contiguous()

                # Step 2a: Filter network (MLP, not fused)
                with nvtx.annotate("CFConv_filter_network", color="salmon"):
                    filter_out = self.filter_network(
                        edge_attr
                    )  # [num_edges, num_filters]

                # Steps 2b-5 FUSED: cutoff * filter_out * x[src] -> scatter_add to dst
                cutoff_upper = self.cutoff.cutoff_upper

                # Get src-CSR for backward if available (USE_SRC_CSR_GRAD_X=1)
                src_ptr = csr_data.get("src_ptr") if csr_data else None
                src_perm = csr_data.get("src_perm") if csr_data else None

                x = fused_cutoff_gather_multiply_scatter_autograd(
                    x.contiguous(),
                    filter_out.contiguous(),
                    edge_weight.contiguous(),
                    edge_src,
                    edge_dst,
                    num_nodes,
                    cutoff_upper,
                    src_ptr,
                    src_perm,
                )
            else:
                # === Original PyTorch Path ===
                # Step 2a: Compute cutoff envelope
                with nvtx.annotate("CFConv_cutoff", color="pink"):
                    C = self.cutoff(edge_weight)  # [num_edges]
                # Step 2b: Filter network + apply cutoff
                with nvtx.annotate("CFConv_filter_network", color="salmon"):
                    W = self.filter_network(edge_attr) * C.view(
                        -1, 1
                    )  # [num_edges, num_filters]
                # Steps 3-5: Message passing (gather x[src], multiply by W, scatter_add to dst)
                x = self.propagate(
                    edge_index, x=x, W=W, size=None
                )  # calls message() then aggregates

        # Step 6: Final linear projection
        with nvtx.annotate("CFConv_lin2", color="coral"):
            x = self.lin2(x)  # [num_nodes, out_channels]
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


class StandardSchNet(SchNet):
    """Small wrapper class for :ref:`SchNet` to simplify the definition of the
    SchNet model through an input file. The upper distance cutoff attribute
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
        )
