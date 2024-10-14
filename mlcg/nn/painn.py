"""Code adapted from https://github.com/atomistic-machine-learning/schnetpack"""

from typing import Callable, Union, List, Final

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
import warnings

from mlcg.data import AtomicData
from mlcg.data._keys import ENERGY_KEY
from mlcg.nn import MLP
from mlcg.nn._module_init import init_xavier_uniform
from mlcg.geometry.internal_coordinates import compute_distance_vectors
from mlcg.neighbor_list.neighbor_list import (
    atomic_data2neighbor_list,
    validate_neighborlist,
)


class PaiNNInteraction(MessagePassing):
    r"""PyTorch Geometric implementation of PaiNN block for modeling equivariant interactions.
    Code adapted from https://schnetpack.readthedocs.io/en/latest/api/generated/representation.PaiNN.html

    Parameters
    ----------
    hidden_channels:
        Hidden channel dimension, i.e. node feature size used for the node embedding.
    edge_attr_dim:
        Edge attributes dimension, i.e. number of radial basis functions.
    cutoff:
        Cutoff function
    activation:
        Activation function applied to linear layer outputs.
    aggr:
        Aggregation scheme for continuous filter output. For all options,
        see `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html?highlight=MessagePassing#the-messagepassing-base-class>`.
    """

    def __init__(
        self,
        hidden_channels: int,
        edge_attr_dim: int,
        cutoff: nn.Module,
        activation: Callable,
        aggr: str = "add",
    ):
        super().__init__(aggr=aggr)
        self.hidden_channels = hidden_channels

        self.interatomic_context_net = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            activation,
            nn.Linear(hidden_channels, 3 * hidden_channels),
        )
        self.filter_network = nn.Linear(edge_attr_dim, 3 * hidden_channels)
        self.cutoff = cutoff

    def reset_parameters(self):
        r"""Method for resetting the weights of the linear
        layers and filter network according the the
        Xavier uniform strategy. Biases
        are set to 0.
        """

        for module in self.interatomic_context_net:
            init_xavier_uniform(module)
        init_xavier_uniform(self.filter_network)

    def forward(
        self,
        scalar_node_features: torch.Tensor,  # (n_nodes, 1, n_feat)
        vector_node_features: torch.Tensor,  # (n_nodes, 3, n_feat)
        normdir: torch.Tensor,  # (n_edges, 3)
        edge_index: torch.Tensor,  # (2, n_edges)
        edge_weight: torch.Tensor,  # (n_edges)
        edge_attr: torch.Tensor,  # (n_edges, edge_attr_dim)
    ):
        r"""Compute interaction output.

        Parameters
        ----------
        scalar_node_features:
            Scalar input embedding per node with shape (n_nodes, 1, n_feat).
        vector_node_features:
            Vector input embedding per node with shape (n_nodes, 3, n_feat).
        normdir:
            Normalized directions for every edge with shape (total_num_edges, 3).
        edge_index:
            Graph edge index tensor of shape (2, total_num_edges).
        edge_weight:
            Scalar edge weight, i.e. distances, of shape (total_num_edges, 1).
        edge_attr
            Edge attributes, i.e. rbf projection, of shape (total_num_edges, edge_attr_dim).

        Returns
        -------
        x_scalar:
            Updated scalar embedding per node with shape (n_nodes, 1, n_feat).
        x_vector:
            Updated vector embedding per node with shape (n_nodes, 3, n_feat).
        """
        C = self.cutoff(edge_weight)
        W = self.filter_network(edge_attr) * C.unsqueeze(-1)

        x_scalar = scalar_node_features.squeeze(1)  # (n_nodes, n_feat)
        n_nodes, _ = x_scalar.shape
        x_vector = vector_node_features.view(n_nodes, -1)  # (n_nodes, 3*n_feat)
        x = self.interatomic_context_net(x_scalar)

        return self.propagate(
            edge_index,
            x=x,
            x_scalar=x_scalar,
            x_vector=x_vector,
            W=W,
            normdir=normdir,
        )

    def message(
        self,
        x_j: torch.Tensor,
        x_vector_j: torch.Tensor,
        W: torch.Tensor,
        normdir: torch.Tensor,
    ):
        r"""Message passing operation to generate messages for
        scalar and vectorial features.

        Parameters
        ----------
        x_j:
            Tensor of embedded features of shape
            (total_num_edges,3*hidden_channels)
        x_vector_j:
            Tensor of vectorial features of shape
            (total_num_edges,3*hidden_channels)
        W:
            Tensor of filter values of shape
            (total_num_edges, 3*hidden_channels)
        normdir:
            Tensor of distances versors of shape
            (total_num_edges, 3)

        Returns
        -------
        dq:
            Scalar updates for all nodes.
        dmu:
            Vectorial updates for all nodes
        """
        x_j = x_j.unsqueeze(1)  # reshape as (total_num_edges, 1, 3*n_feats)
        x_vector_j = x_vector_j.view(
            x_j.shape[0], 3, -1
        )  # reshape as (total_num_edges, 3, n_feats)
        x = W * x_j
        dq, dmuR, dmumu = torch.split(x, self.hidden_channels, dim=-1)
        dmu = dmuR * normdir.unsqueeze(-1) + dmumu * x_vector_j

        return dq, dmu

    def aggregate(
        self, inputs: torch.Tensor, index: torch.Tensor, dim_size: int
    ):
        dq, dmu = inputs
        dq = scatter(dq, index, dim=0, dim_size=dim_size)
        dmu = scatter(dmu, index, dim=0, dim_size=dim_size)

        return dq, dmu

    def update(
        self,
        inputs: torch.Tensor,
        x_scalar: torch.Tensor,
        x_vector: torch.Tensor,
    ):
        dq, dmu = inputs
        x_scalar = x_scalar.unsqueeze(1) + dq
        x_vector = x_vector.view(x_scalar.shape[0], 3, -1) + dmu

        return x_scalar, x_vector


class PaiNNMixing(nn.Module):
    r"""PaiNN interaction block for mixing on scalar and vectorial atom features.
    Code adapted from https://schnetpack.readthedocs.io/en/latest/api/generated/representation.PaiNN.html

    Parameters
    -----------
    hidden_channels:
        Hidden channel dimension, i.e. node feature size used for the node embedding.
    activation:
        Activation function applied to linear layer outputs.
    epsilon:
        Stability constant added in norm to prevent numerical instabilities.
    """

    def __init__(
        self, hidden_channels: int, activation: Callable, epsilon: float = 1e-8
    ):
        super(PaiNNMixing, self).__init__()
        self.hidden_channels = hidden_channels

        self.intraatomic_context_net = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            activation,
            nn.Linear(hidden_channels, 3 * hidden_channels),
        )
        self.mu_channel_mix = nn.Linear(
            hidden_channels, 2 * hidden_channels, bias=False
        )
        self.epsilon = epsilon

    def reset_parameters(self):
        r"""Method for resetting the weights of the linear
        layers and filter network according the the
        Xavier uniform strategy. Biases
        are set to 0.
        """
        for module in self.intraatomic_context_net:
            init_xavier_uniform(module)
        init_xavier_uniform(self.mu_channel_mix)

    def forward(
        self,
        scalar_node_features: torch.Tensor,
        vector_node_features: torch.Tensor,
    ):
        r"""Compute intraatomic mixing.

        Parameters
        ----------
        scalar_node_features:
            Tensor of scalar features of shape (n_nodes, 1, n_feat).
        vector_node_features:
            Tensor of vectorial features of shape (n_nodes, 3, n_feat).

        Returns
        -------
        scalar_node_features:
            Updated tensor of scalar features, mixed with vectorial features.
        vector_node_features:
            Updated tensor of vectorial features, mixed with scalar features.
        """

        # intra-atomic
        mu_mix = self.mu_channel_mix(vector_node_features)
        mu_V, mu_W = torch.split(mu_mix, self.hidden_channels, dim=-1)
        mu_Vn = torch.sqrt(
            torch.sum(mu_V**2, dim=-2, keepdim=True) + self.epsilon
        )

        ctx = torch.cat([scalar_node_features, mu_Vn], dim=-1)
        x = self.intraatomic_context_net(ctx)

        dq_intra, dmu_intra, dqmu_intra = torch.split(
            x, self.hidden_channels, dim=-1
        )
        dmu_intra = dmu_intra * mu_W

        dqmu_intra = dqmu_intra * torch.sum(mu_V * mu_W, dim=1, keepdim=True)

        scalar_node_features = scalar_node_features + dq_intra + dqmu_intra
        vector_node_features = vector_node_features + dmu_intra
        return scalar_node_features, vector_node_features


class PaiNN(nn.Module):
    r"""Implementation of PaiNN - polarizable interaction neural network
    Code adapted from https://schnetpack.readthedocs.io/en/latest/api/generated/representation.PaiNN.html
    which is based on the architecture described in http://proceedings.mlr.press/v139/schutt21a.html

    Parameters
    ----------
    embedding_layer:
        Initial embedding layer that transforms atoms/coarse grain bead
        types into embedded features
    interaction_blocks:
        list of PaiNNInteraction or single PaiNNInteraction block.
        Sequential interaction blocks of the model, where each interaction
        block applies.
    mixing_blocks:
        List of PaiNNMixing or single PaiNNMixing block.
        Sequential mixing blocks of the model, where each mixing
        applies after each interaction block.
    rbf_layer:
        The set of radial basis functions that expands pairwise distances
        between atoms/CG beads.
    output_network:
        Output neural network that predicts scalar energies from scalar PaiNN
        features. This network should transform (num_examples * num_atoms,
        hidden_channels) to (num_examples * num atoms, 1).
    max_num_neighbors:
        Maximum number of neighbors to return for a
        given node/atom when constructing the molecular graph during forward
        passes. This attribute is passed to the torch_cluster radius_graph
        routine keyword max_num_neighbors, which normally defaults to 32.
        Users should set this to higher values if they are using higher upper
        distance cutoffs and expect more than 32 neighbors per node/atom.
    """

    name: Final[str] = "PaiNN"

    def __init__(
        self,
        embedding_layer: nn.Module,
        interaction_blocks: Union[PaiNNInteraction, List[PaiNNInteraction]],
        mixing_blocks: Union[PaiNNMixing, List[PaiNNMixing]],
        rbf_layer: torch.nn.Module,
        output_network: nn.Module,
        max_num_neighbors: int,
    ):
        super(PaiNN, self).__init__()

        self.embedding_layer = embedding_layer
        if isinstance(interaction_blocks, List) or isinstance(
            interaction_blocks, PaiNNInteraction
        ):
            self.interaction_blocks = torch.nn.ModuleList(interaction_blocks)
        else:
            raise RuntimeError(
                "interaction_blocks must be a single InteractionBlock or "
                "a list of InteractionBlocks."
            )
        if isinstance(mixing_blocks, List) or isinstance(
            mixing_blocks, PaiNNMixing
        ):
            self.mixing_blocks = torch.nn.ModuleList(mixing_blocks)
        else:
            raise RuntimeError(
                "mixing_blocks must be a single PaiNNMixing or "
                "a list of PaiNNMixing."
            )
        if len(self.mixing_blocks) != len(self.interaction_blocks):
            raise RuntimeError(
                "The number of mixing and interaction blocks must be equal "
                f"but you provided {len(self.mixing_blocks)} mixing "
                f"and {len(self.interaction_blocks)} interactions"
            )
        self.output_network = output_network
        self.rbf_layer = rbf_layer
        self.max_num_neighbors = max_num_neighbors

        self.reset_parameters()

    def reset_parameters(self):
        """Method for resetting linear layers in each PaiNN component"""
        self.embedding_layer.reset_parameters()
        self.rbf_layer.reset_parameters()
        for block in self.interaction_blocks:
            block.reset_parameters()
        for block in self.mixing_blocks:
            block.reset_parameters()
        self.output_network.reset_parameters()

    def forward(self, data: AtomicData):
        r"""Forward pass through the PaiNN architecture.

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
                data, self.rbf_layer.cutoff.cutoff_upper, self.max_num_neighbors
            )[self.name]

        edge_index = neighbor_list["index_mapping"]

        distances, normdir = compute_distance_vectors(
            data.pos,
            edge_index,
            neighbor_list["cell_shifts"],
        )

        rbf_expansion = self.rbf_layer(distances)

        q = self.embedding_layer(data.atom_types)  # (n_atoms, n_features)
        q = q.unsqueeze(1)  # (n_atoms, 1, n_features)
        mu = torch.zeros(
            (q.shape[0], 3, q.shape[2]), device=q.device
        )  # (n_atoms, 3, n_features)

        for i, (interaction, mixing) in enumerate(
            zip(self.interaction_blocks, self.mixing_blocks)
        ):
            q, mu = interaction(
                q, mu, normdir, edge_index, distances, rbf_expansion
            )
            q, mu = mixing(q, mu)
        q = q.squeeze(1)

        energy = self.output_network(q, data)
        energy = scatter(energy, data.batch, dim=0, reduce="sum")
        energy = energy.flatten()
        data.out[self.name] = {ENERGY_KEY: energy}

        return data

    def is_nl_compatible(self, nl):
        is_compatible = False
        if validate_neighborlist(nl):
            if (
                nl["order"] == 2
                and nl["self_interaction"] is False
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
            PaiNN.name: atomic_data2neighbor_list(
                data,
                rcut,
                self_interaction=False,
                max_num_neighbors=max_num_neighbors,
            )
        }


class StandardPaiNN(PaiNN):
    r"""Small wrapper class for :ref:`PaiNN` to simplify the definition of the
    PaiNN model through an input file. The upper distance cutoff attribute
    in is set by default to match the upper cutoff value in the cutoff function.

    Parameters
    ----------
    rbf_layer:
        Radial basis function used to project the distances :math:`r_{ij}`.
    cutoff:
        Smooth cutoff function to supply to the PaiNNInteraction.
    output_hidden_layer_widths:
        List giving the number of hidden nodes of each hidden layer of the MLP
        used to predict the target property from the learned scalar representation.
    hidden_channels:
        Dimension of the learned representation, i.e. dimension of the embedding projection, convolution layers, and interaction block.
    embedding_size:
        Dimension of the input embeddings (should be larger than :obj:`AtomicData.atom_types.max()+1`).
    num_interactions:
        Number of interaction blocks.
    activation:
        Activation function.
    max_num_neighbors:
        The maximum number of neighbors to return for each atom in :obj:`data`.
        If the number of actual neighbors is greater than
        :obj:`max_num_neighbors`, returned neighbors are picked randomly.
    aggr:
        Aggregation scheme for continuous filter output. For all options,
        see `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html?highlight=MessagePassing#the-messagepassing-base-class>`_
        for more options.
    epsilon:
        Stability constant added in norm to prevent numerical instabilities.
    """

    def __init__(
        self,
        rbf_layer: torch.nn.Module,
        cutoff: torch.nn.Module,
        output_hidden_layer_widths: List[int],
        hidden_channels: int = 128,
        embedding_size: int = 100,
        num_interactions: int = 3,
        activation: torch.nn.Module = torch.nn.Tanh(),
        max_num_neighbors: int = 1000,
        aggr: str = "add",
        epsilon: float = 1e-8,
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
        mixing_blocks = []
        for _ in range(num_interactions):
            interaction_blocks.append(
                PaiNNInteraction(
                    hidden_channels, rbf_layer.num_rbf, cutoff, activation, aggr
                )
            )
            mixing_blocks.append(
                PaiNNMixing(hidden_channels, activation, epsilon)
            )

        output_layer_widths = (
            [hidden_channels] + output_hidden_layer_widths + [1]
        )
        output_network = MLP(
            output_layer_widths, activation_func=activation, last_bias=False
        )
        super(StandardPaiNN, self).__init__(
            embedding_layer,
            interaction_blocks,
            mixing_blocks,
            rbf_layer,
            output_network,
            max_num_neighbors,
        )
