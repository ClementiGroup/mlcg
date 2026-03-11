from mlcg.nn.schnet import SchNet, CFConv, InteractionBlock, StandardSchNet
import warnings
from typing import List
import torch
from torch_geometric.utils import scatter
from mlcg.data.atomic_data import AtomicData, ENERGY_KEY
from mlcg.geometry.internal_coordinates import compute_distances
from mlcg.nn.mlp import MLP

from .radial_basis import RegularizedBasis

try:
    from mlcg_opt_radius.radius import radius_distance
except ImportError:
    print(
        "`mlcg_opt_radius` not installed. Please check the"
        + "`opt_radius` folder and follow the instructions."
    )
    radius_distance = None


class EdgeAwareInteractionBlock(InteractionBlock):
    def __init__(
        self,
        cfconv_layer: torch.nn.Module,
        hidden_channels: int = 128,
        activation: torch.nn.Module = torch.nn.Tanh(),
    ):
        super().__init__(cfconv_layer, hidden_channels, activation)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_params: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv(x, edge_index, edge_weight, edge_attr, edge_params)
        x = self.activation(x)
        x = self.lin(x)
        return x


class EdgeAwareCFConv(CFConv):
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_params: torch.Tensor,
    ) -> torch.Tensor:
        C = self.cutoff(edge_weight)
        W = self.filter_network(edge_attr) * C.view(-1, 1)
        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W, edge_params=edge_params)
        x = self.lin2(x)
        return x

    def message(
        self, x_j: torch.Tensor, W: torch.Tensor, edge_params: torch.Tensor
    ) -> torch.Tensor:
        return x_j * (torch.abs(edge_params.view(-1, 1)) * W)

    def propagate(self, edge_index, size=None, **kwargs):
        # Manually control the flow of arguments
        kwargs["edge_params"] = kwargs.get("edge_params")
        return super().propagate(edge_index, size=size, **kwargs)


class EdgeRBFRegularizedSchNet(StandardSchNet):
    """
    This is a StandardSchNet model where the RBF components are weighted via Hadamard product with couple-specific
    vectors. Each tuple of bead types has its own vector.
    The models also adds the key radial_filters to the output dictionary, containing the regularization parameters.

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

    """

    def __init__(
        self,
        rbf_layer: torch.nn.Module,
        cutoff: torch.nn.Module,
        output_hidden_layer_widths: List[int],
        max_bead_type: int,
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
        super(EdgeRBFRegularizedSchNet, self).__init__(
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

        # Replace standard interaction blocks with EdgeAware versions
        edge_aware_blocks = []
        for block in self.interaction_blocks:
            cfconv = EdgeAwareCFConv(
                block.conv.filter_network,
                cutoff=block.conv.cutoff,
                num_filters=num_filters,
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                aggr=aggr,
            )
            edge_aware_blocks.append(
                EdgeAwareInteractionBlock(cfconv, hidden_channels, activation)
            )
        self.interaction_blocks = torch.nn.ModuleList(edge_aware_blocks)

        edge_modulation = torch.ones(
            num_interactions, max_bead_type, max_bead_type
        ).float()

        i, j = torch.tril_indices(max_bead_type, max_bead_type, offset=-1)

        edge_modulation[:, i, j] *= 0

        self.register_parameter(
            "edge_parameters",
            torch.nn.Parameter(edge_modulation, requires_grad=True),
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
        types_i = data.atom_types[edge_index[0]]
        types_j = data.atom_types[edge_index[1]]
        senders = torch.min(types_i, types_j)
        receivers = torch.max(types_i, types_j)

        for block_id, block in enumerate(self.interaction_blocks):
            x = x + block(
                x,
                edge_index,
                distances,
                rbf_expansion[block_id],
                self.edge_parameters[block_id, senders, receivers],
            )

        energy = self.output_network(x, data)
        energy = scatter(energy, data.batch, dim=0, reduce="sum")
        energy = energy.flatten()
        data.out[self.name] = {
            ENERGY_KEY: energy,
            "radial_filters": self.rbf_layer.get_regularization_parameters(),
            "edge_parameters": self.edge_parameters,
        }

        return data