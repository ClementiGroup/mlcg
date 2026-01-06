import torch
from .schnet import StandardSchNet
from typing import Optional, List, Final
from torch_scatter import scatter
from ..data.atomic_data import AtomicData, ENERGY_KEY
from ..geometry.internal_coordinates import compute_distances


class SimpleNoiseEmbedSchNet(StandardSchNet):
    """Small wrapper class for :ref:`SchNet` to simplify the definition of the
    SchNet model through an input file. It is very similar to StandardSchNet
    except that it expects a tensor property `noise_levels` in the input `AtomicData`,
    which has the same shape as `atom_types`. The noise level will be transformed
    by `noise_level_rbf_layer`, whose output vector is then transformed by two dense
    layers in `NoiseAwareEmbedding`. The output will be summed with the standard atom
    type embeddings and the sum serves as the starting features for the SchNet.
    (Essentially, the noise level modulates a bias vector to be added up to atom
    type embeddings.)

    Parameters
    ----------
    rbf_layer:
        radial basis function used to project the distances :math:`r_{ij}`.
    cutoff:
        smooth cutoff function to supply to the CFConv
    noise_level_rbf_layer:
        An rbf layer that maps real number `data.noise_levels` corresponding to each
        particle in the input (same shape as `data.atom_types`) to a vector. Supposely
        one can use GaussianBasis or SkewedMuGaussianBasis for this purpose.
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
        noise_num_embedding: int,
        output_hidden_layer_widths: List[int],
        hidden_channels: int = 128,
        embedding_size: int = 100,
        num_filters: int = 128,
        num_interactions: int = 3,
        activation: torch.nn.Module = torch.nn.Tanh(),
        max_num_neighbors: int = 1000,
        aggr: str = "add",
        include_noise_signal_at_embedding_stage: bool = True,
        include_noise_signal_at_output_stage: bool = False,
    ):
        super(SimpleNoiseEmbedSchNet, self).__init__(
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
        # replace the embedding layer for incorporating the noise level
        # del self.embedding_layer
        # self.embedding_layer = NoiseAwareEmbedding(
        #    embedding_size,
        #    hidden_channels,
        #    noise_level_rbf=noise_level_rbf_layer,
        # )
        self.include_noise_signal_at_embedding_stage = (
            include_noise_signal_at_embedding_stage
        )
        if self.include_noise_signal_at_embedding_stage:
            self.noise_embedding = torch.nn.Embedding(
                noise_num_embedding, hidden_channels
            )
        self.include_noise_signal_at_output_stage = (
            include_noise_signal_at_output_stage
        )
        if self.include_noise_signal_at_output_stage:
            self.noise_embedding2 = torch.nn.Embedding(
                noise_num_embedding, hidden_channels
            )

    def schnet_forward(self, data: AtomicData, embeddings: torch.Tensor):
        """The main forward transform used by the SchNet expect the embedding generations."""

        x = embeddings
        neighbor_list = data.neighbor_list.get(self.name)

        if not self.is_nl_compatible(neighbor_list):
            neighbor_list = self.neighbor_list(
                data, self.rbf_layer.cutoff.cutoff_upper, self.max_num_neighbors
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
        if self.include_noise_signal_at_output_stage:
            # we include the noise level information as a multiplicative factor
            x *= self.noise_embedding2(data.noise_levels)
        energy = self.output_network(x, data)
        energy = scatter(energy, data.batch, dim=0, reduce="sum")
        energy = energy.flatten()
        data.out[self.name] = {ENERGY_KEY: energy}

        return data

    def forward(self, data: AtomicData) -> AtomicData:
        embeddings = self.embedding_layer(data.atom_types)
        if self.include_noise_signal_at_embedding_stage:
            embeddings += self.noise_embedding(data.noise_levels)
        return self.schnet_forward(data, embeddings)


class SimpleNoiseEmbedSchNetII(StandardSchNet):
    """Small wrapper class for :ref:`SchNet` to simplify the definition of the
    SchNet model through an input file. It is very similar to StandardSchNet
    except that it expects a tensor property `noise_levels` in the input `AtomicData`,
    which has the same shape as `atom_types`. The noise level will be transformed
    by `noise_level_rbf_layer`, whose output vector is then transformed by two dense
    layers in `NoiseAwareEmbedding`. The output will be summed with the standard atom
    type embeddings and the sum serves as the starting features for the SchNet.
    (Essentially, the noise level modulates a bias vector to be added up to atom
    type embeddings.)

    Parameters
    ----------
    rbf_layer:
        radial basis function used to project the distances :math:`r_{ij}`.
    cutoff:
        smooth cutoff function to supply to the CFConv
    noise_level_rbf_layer:
        An rbf layer that maps real number `data.noise_levels` corresponding to each
        particle in the input (same shape as `data.atom_types`) to a vector. Supposely
        one can use GaussianBasis or SkewedMuGaussianBasis for this purpose.
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
        noise_num_embedding: int,
        output_hidden_layer_widths: List[int],
        hidden_channels: int = 128,
        embedding_size: int = 100,
        num_filters: int = 128,
        num_interactions: int = 3,
        activation: torch.nn.Module = torch.nn.Tanh(),
        max_num_neighbors: int = 1000,
        aggr: str = "add",
        include_noise_signal_at_embedding_stage: bool = True,
        include_noise_signal_at_output_stage: bool = False,
    ):
        super(SimpleNoiseEmbedSchNetII, self).__init__(
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
        # replace the embedding layer for incorporating the noise level
        # del self.embedding_layer
        # self.embedding_layer = NoiseAwareEmbedding(
        #    embedding_size,
        #    hidden_channels,
        #    noise_level_rbf=noise_level_rbf_layer,
        # )
        self.include_noise_signal_at_embedding_stage = (
            include_noise_signal_at_embedding_stage
        )
        if self.include_noise_signal_at_embedding_stage:
            self.noise_embedding = torch.nn.Embedding(
                noise_num_embedding, hidden_channels
            )
        self.include_noise_signal_at_output_stage = (
            include_noise_signal_at_output_stage
        )
        if self.include_noise_signal_at_output_stage:
            self.noise_embedding2 = torch.nn.Embedding(
                noise_num_embedding, hidden_channels
            )

    def schnet_forward(self, data: AtomicData, embeddings: torch.Tensor):
        """The main forward transform used by the SchNet expect the embedding generations."""

        x = embeddings
        neighbor_list = data.neighbor_list.get(self.name)

        if not self.is_nl_compatible(neighbor_list):
            neighbor_list = self.neighbor_list(
                data, self.rbf_layer.cutoff.cutoff_upper, self.max_num_neighbors
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
        if self.include_noise_signal_at_output_stage:
            # we include the noise level information as a multiplicative factor
            x *= self.noise_embedding2(data.noise_levels)
        energy = self.output_network(x, data)
        energy = scatter(energy, data.batch, dim=0, reduce="sum")
        energy = energy.flatten()
        data.out[self.name] = {ENERGY_KEY: energy}

        return data

    def forward(self, data: AtomicData) -> AtomicData:
        embeddings = self.embedding_layer(data.atom_types)
        if self.include_noise_signal_at_embedding_stage:
            embeddings += self.noise_embedding(data.noise_levels)
        return self.schnet_forward(data, embeddings)
