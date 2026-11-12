import torch
from .schnet import StandardSchNet
from typing import List
from torch_geometric.utils import scatter
from ..data.atomic_data import AtomicData, ENERGY_KEY
from ..geometry.internal_coordinates import compute_distances
from .mlp import MLP
from .radial_basis import SkewedMuGaussianBasis

__all__ = ["NoiseAwareEmbedding", "SimpleNoiseEmbedSchNet", "NoiseEmbedSchNet"]


class NoiseAwareEmbedding(torch.nn.Embedding):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        noise_level_rbf: SkewedMuGaussianBasis,
        noise_network_activation_func: torch.nn.Module = torch.nn.SiLU(),
        **kwargs,
    ):
        """A modified Embedding layer that introduces a noise-level-dependent bias on
        all embedding channels."""
        super(NoiseAwareEmbedding, self).__init__(
            num_embeddings, embedding_dim, **kwargs
        )
        self._embedding_dim = embedding_dim
        self.noise_level_rbf = noise_level_rbf
        num_noise_level_rbf = self.noise_level_rbf.num_rbf
        self.filter_network = MLP(
            layer_widths=[num_noise_level_rbf, embedding_dim, embedding_dim],
            activation_func=noise_network_activation_func,
            last_bias=False,
        )

    def forward(
        self,
        discrete_types: torch.Tensor,
        noise_levels: torch.Tensor,
    ) -> torch.Tensor:
        y = super().forward(discrete_types)

        z = self.filter_network(self.noise_level_rbf(noise_levels))
        return y + z


class SimpleNoiseEmbedSchNet(StandardSchNet):
    r"""Noise-conditioned :ref:`StandardSchNet` with a *discrete* noise level.

    Same architecture as :class:`StandardSchNet`, but the input
    :obj:`AtomicData` must also carry an integer tensor ``noise_levels`` with
    the same shape as ``atom_types`` (one entry per bead). Each entry is an
    index into a learnable lookup table of size ``noise_num_embedding``, so
    the model is conditioned on a finite set of noise levels
    :math:`k \in \{0, \dots, \texttt{noise\_num\_embedding} - 1\}`. For a
    continuous noise level, use :class:`NoiseEmbedSchNet`.

    The noise signal can be injected at two points, controlled by the
    ``include_noise_signal_at_*`` flags:

    * **Embedding stage** (additive): the initial node features are

      .. math::

          h_i^{(0)} = E_{\mathrm{type}}[z_i] + E_{\mathrm{noise}}[k_i]

    * **Output stage** (multiplicative): after the last interaction block,
      the node features are rescaled element-wise before the output MLP,

      .. math::

          h_i \leftarrow h_i \odot E'_{\mathrm{noise}}[k_i]

    If both flags are ``False`` the noise level is ignored and the model
    reduces to :class:`StandardSchNet`.

    Parameters
    ----------
    rbf_layer:
        Radial basis function used to project the distances :math:`r_{ij}`.
    cutoff:
        Smooth cutoff function to supply to the CFConv.
    noise_num_embedding:
        Number of distinct noise levels, i.e. size of the noise lookup
        table(s). Every value in ``data.noise_levels`` must lie in
        ``[0, noise_num_embedding)``.
    output_hidden_layer_widths:
        List giving the number of hidden nodes of each hidden layer of the MLP
        used to predict the target property from the learned representation.
    hidden_channels:
        Dimension of the learned representation, i.e. dimension of the
        embedding projection, convolution layers, and interaction blocks.
        Also the dimension of the noise embeddings.
    embedding_size:
        Number of atom-type embeddings (should be larger than
        :obj:`AtomicData.atom_types.max()+1`).
    num_filters:
        Number of nodes of the networks used to filter the projected distances.
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
        see `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html?highlight=MessagePassing#the-messagepassing-base-class>`_.
    include_noise_signal_at_embedding_stage:
        If ``True``, add a learned noise embedding to the atom-type embedding
        before the interaction blocks.
    include_noise_signal_at_output_stage:
        If ``True``, multiply the node features element-wise by a second,
        independent learned noise embedding before the output network.

    Notes
    -----
    Unlike :class:`StandardSchNet`, this class does not expose
    ``nls_distance_method`` or ``excluded_order``; the parent defaults are
    used. Its :meth:`schnet_forward` also skips the ``excluded_order``
    consistency check of the parent.
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
        """Main SchNet forward pass starting from precomputed node embeddings.

        Runs the interaction blocks, optionally applies the multiplicative
        output-stage noise embedding, then the output network, and sums the
        per-bead energies over each structure.

        Parameters
        ----------
        data:
            Input batch; must contain ``pos``, ``batch`` and, if the output
            stage is enabled, ``noise_levels``.
        embeddings:
            Initial node features of shape ``(n_atoms, hidden_channels)``.

        Returns
        -------
        data:
            The input :obj:`AtomicData` with the predicted energy stored in
            ``data.out[self.name][ENERGY_KEY]`` (shape ``(n_structures,)``).
        """

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
        """Embed atom types (plus the additive noise embedding, if enabled)
        and run :meth:`schnet_forward`.

        Parameters
        ----------
        data:
            Input batch; must contain ``atom_types`` and an integer
            ``noise_levels`` tensor of the same shape.

        Returns
        -------
        data:
            The input :obj:`AtomicData` with the predicted energy stored in
            ``data.out[self.name][ENERGY_KEY]``.
        """
        embeddings = self.embedding_layer(data.atom_types)
        if self.include_noise_signal_at_embedding_stage:
            embeddings += self.noise_embedding(data.noise_levels)
        return self.schnet_forward(data, embeddings)


class NoiseEmbedSchNet(StandardSchNet):
    r"""Noise-conditioned :ref:`StandardSchNet` with a *continuous* noise level.

    Same architecture as :class:`StandardSchNet`, but the input
    :obj:`AtomicData` must also carry a real-valued tensor ``noise_levels``
    with the same shape as ``atom_types`` (one entry per bead). The atom-type
    embedding layer is replaced by a :class:`NoiseAwareEmbedding`, which
    expands the noise level :math:`\sigma_i` in a radial basis and maps it
    through a two-layer MLP to a bias vector that is added to the atom-type
    embedding:

    .. math::

        h_i^{(0)} = E_{\mathrm{type}}[z_i]
                    + \mathrm{MLP}\big(\mathrm{RBF}(\sigma_i)\big)

    The noise level only enters through these initial features; the
    interaction blocks and output network are unchanged. For a finite set of
    discrete noise levels, see :class:`SimpleNoiseEmbedSchNet`.

    Parameters
    ----------
    rbf_layer:
        Radial basis function used to project the distances :math:`r_{ij}`.
    cutoff:
        Smooth cutoff function to supply to the CFConv.
    noise_level_rbf_layer:
        Basis expansion applied to ``data.noise_levels`` before the noise MLP,
        e.g. :class:`GaussianBasis` or :class:`SkewedMuGaussianBasis`. Must
        expose a ``num_rbf`` attribute, which sets the MLP input width.
    output_hidden_layer_widths:
        List giving the number of hidden nodes of each hidden layer of the MLP
        used to predict the target property from the learned representation.
    hidden_channels:
        Dimension of the learned representation, i.e. dimension of the
        embedding projection, convolution layers, and interaction blocks.
        Also the hidden and output width of the noise MLP.
    embedding_size:
        Number of atom-type embeddings (should be larger than
        :obj:`AtomicData.atom_types.max()+1`).
    num_filters:
        Number of nodes of the networks used to filter the projected distances.
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
        see `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html?highlight=MessagePassing#the-messagepassing-base-class>`_.
    nls_distance_method:
        Method for computing a neighbor list, see :ref:`SchNet`.
    excluded_order:
        Order of excluded interactions in the neighbor list, see :ref:`SchNet`.

    """

    def __init__(
        self,
        rbf_layer: torch.nn.Module,
        cutoff: torch.nn.Module,
        noise_level_rbf_layer: torch.nn.Module,
        output_hidden_layer_widths: List[int],
        hidden_channels: int = 128,
        embedding_size: int = 100,
        num_filters: int = 128,
        num_interactions: int = 3,
        activation: torch.nn.Module = torch.nn.Tanh(),
        max_num_neighbors: int = 1000,
        aggr: str = "add",
        nls_distance_method: str = "torch",
        excluded_order: int = 1,
    ):
        super(NoiseEmbedSchNet, self).__init__(
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
            excluded_order=excluded_order,
        )
        # replace the embedding layer for incorporating the noise level
        del self.embedding_layer
        self.embedding_layer = NoiseAwareEmbedding(
            embedding_size,
            hidden_channels,
            noise_level_rbf=noise_level_rbf_layer,
        )

    def forward(self, data: AtomicData) -> AtomicData:
        """Build noise-aware embeddings and run :meth:`schnet_forward`.

        Parameters
        ----------
        data:
            Input batch; must contain ``atom_types`` and a real-valued
            ``noise_levels`` tensor of the same shape.

        Returns
        -------
        data:
            The input :obj:`AtomicData` with the predicted energy stored in
            ``data.out[self.name][ENERGY_KEY]``.
        """
        embeddings = self.embedding_layer(data.atom_types, data.noise_levels)
        return self.schnet_forward(data, embeddings)
