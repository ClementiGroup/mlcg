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

try:
    from mlcg_opt_radius.radius import radius_distance
except ImportError:
    print(
        "`mlcg_opt_radius` not installed. Please check the `opt_radius` folder and follow the instructions."
    )
    radius_distance = None


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
            # we need to generate the neighbor list
            # check whether we are using the custom kernel
            # 1. mlcg_opt_radius is installed
            # 2. input data is on CUDA
            # 3. not using PBC (TODO)
            use_custom_kernel = False
            if (radius_distance is not None) and x.is_cuda:
                use_custom_kernel = True
            if not use_custom_kernel:
                if hasattr(data, "exc_pair_index"):
                    raise NotImplementedError(
                        "Excluding pairs requires `mlcg_opt_radius` "
                        "to be available and model running with CUDA."
                    )
                neighbor_list = self.neighbor_list(
                    data,
                    self.rbf_layer.cutoff.cutoff_upper,
                    self.max_num_neighbors,
                )[self.name]
        if use_custom_kernel:
            distances, edge_index = radius_distance(
                data.pos,
                self.rbf_layer.cutoff.cutoff_upper,
                data.batch,
                False,  # no loop edges due to compatibility & backward breaks with zero distance
                self.max_num_neighbors,
                exclude_pair_indices=data.get("exc_pair_index"),
            )
        else:
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
        energy = scatter(energy, data.batch, dim=0, reduce="sum")
        energy = energy.flatten()
        data.out[self.name] = {ENERGY_KEY: energy}

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

        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
        init_xavier_uniform(self.lin)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_attr: torch.Tensor,
        *args,
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

        Returns
        -------
        x:
            Updated embedded features of shape (num_examples * num_atoms,
            hidden_channels)
        """

        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.activation(x)
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
    """

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
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_attr: torch.Tensor,
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

        Returns
        -------
        x:
            Updated embedded features of shape (num_examples * num_atoms,
            hidden_channels)
        """
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


class AttentiveSchNet(SchNet):
    """Small wrapper class for :ref:`SchNet` to simplify the definition of the
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
        )


class RBFRegularizedSchNet(StandardSchNet):
    """
    This is a StandardSchNet model where the RBF components are weighted via Hadamard product with couple-specific
    vectors. Each tuple of bead types has its own vector. The models also adds the key radial_filters to the

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

    CLASS SPECIFIC PARAMETERS -------------------------------------------------------------

    independent_regularizations: bool
        If True each interaction block has its own set of regularization parameters,
        otherwise the regularization parameters are shared across all interaction blocks.
    define_ranges: Dict
        A dictionary containing the ranges of the type couples to be tracked. The keys are the bead-type couples
        given as strings, and the values are the lower bounds of the ranges to be considered.
        Example: {'(i,j)': min_value, '(k,l)': min_value, ...}
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
        )

    def forward(self, data: AtomicData) -> AtomicData:
        r"""Forward pass through the SchNet architecture with independent regularizations."""
        x = self.embedding_layer(data.atom_types)

        neighbor_list = data.neighbor_list.get(self.name)

        if not self.is_nl_compatible(neighbor_list):
            # we need to generate the neighbor list
            # check whether we are using the custom kernel
            # 1. mlcg_opt_radius is installed
            # 2. input data is on CUDA
            # 3. not using PBC (TODO)
            use_custom_kernel = False
            if (radius_distance is not None) and x.is_cuda:
                use_custom_kernel = True
            if not use_custom_kernel:
                if hasattr(data, "exc_pair_index"):
                    raise NotImplementedError(
                        "Excluding pairs requires `mlcg_opt_radius` "
                        "to be available and model running with CUDA."
                    )
                neighbor_list = self.neighbor_list(
                    data,
                    self.rbf_layer.cutoff.cutoff_upper,
                    self.max_num_neighbors,
                )[self.name]
        if use_custom_kernel:
            distances, edge_index = radius_distance(
                data.pos,
                self.rbf_layer.cutoff.cutoff_upper,
                data.batch,
                False,  # no loop edges due to compatibility & backward breaks with zero distance
                self.max_num_neighbors,
                exclude_pair_indices=data.get("exc_pair_index"),
            )
        else:
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
