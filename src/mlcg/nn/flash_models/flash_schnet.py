import warnings
import copy
from typing import List
import torch
from torch_geometric.utils import scatter
from ...data.atomic_data import AtomicData, ENERGY_KEY
from ..mlp import MLP
from ..schnet import SchNet, StandardSchNet, InteractionBlock, CFConv
from ..kernels.csr import build_csr_representation_from_edges
from ..kernels.models.schnet import fused_cfconv
from ...geometry.internal_coordinates import compute_distances


class FlashSchNet(SchNet):
    r"""
    See :class:`SchNet` for the full API documentation.

    This implementation replaces the standard message-passing kernels
    with fused Triton kernels.
    """

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

        # Saving edge index as contiguous array for speed up triton calls
        edge_index = neighbor_list["index_mapping"].contiguous()
        distances = compute_distances(
            data.pos,
            edge_index,
            neighbor_list["cell_shifts"],
        )

        rbf_expansion = self.rbf_layer(distances)
        num_graphs = data.ptr.numel() - 1 if hasattr(data, "ptr") else None

        csr_data = build_csr_representation_from_edges(edge_index, x.shape[0])

        for i, block in enumerate(self.interaction_blocks):

            x = x + block(
                x,
                edge_index,
                distances,
                rbf_expansion,
                csr_data=csr_data,
            )

            energy = self.output_network(x, data)

        energy = scatter(
            energy, data.batch, dim=0, reduce="sum", dim_size=num_graphs
        )

        energy = energy.flatten()
        data.out.setdefault(self.name, {}).update({ENERGY_KEY: energy})
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


class FlashCFConv(CFConv):
    r"""
    See :class:`CFConv` for the full API documentation.

    This implementation use :func:`fused_cfconv` to fuse 
    the message passing operation performed by SchNet without
    materializing intermediate expanded representations.
    """

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

        return instance
