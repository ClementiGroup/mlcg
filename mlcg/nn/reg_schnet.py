import warnings
from typing import List, Union, Dict
import torch
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from ..data.atomic_data import AtomicData, ENERGY_KEY
from ..geometry.internal_coordinates import compute_distances
from .mlp import MLP
from .schnet import SchNet, CFConv, InteractionBlock

try:
    from mlcg_opt_radius.radius import radius_distance
except ImportError:
    print(
        "`mlcg_opt_radius` not installed. Please check the `opt_radius` folder and follow the instructions."
    )
    radius_distance = None


class RepulsionFilteredSchNet(SchNet):
    """
    Code from Lorenzo Giambagli
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

    max_bead_type: int
        The maximal value of the bead_type needed to calculate all possible couples in the system.
        This considers that all bead_type couples to be tracked are in the range [0, max_bead_type-1].
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
        # Parameters for RBF rescaling
        max_bead_type: int = 25,
        define_ranges: Union[None, Dict] = None,
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

        super(RepulsionFilteredSchNet, self).__init__(
            embedding_layer,
            interaction_blocks,
            rbf_layer,
            output_network,
            max_num_neighbors=max_num_neighbors,
        )

        i, j = torch.tril_indices(max_bead_type, max_bead_type, offset=-1)
        filters = torch.ones(
            max_bead_type, max_bead_type, self.rbf_layer.num_rbf
        ).float()
        filters[i, j] *= 0

        if define_ranges is not None:
            centers = self.rbf_layer.offset
            for pairs, value in define_ranges:
                couple = sorted(ast.literal_eval(pairs))
                mask = centers < value
                filters[couple[0], couple[1], mask] *= 0
                filters[couple[1], couple[0], mask] *= 0
                self.register_buffer("radial_filters", filters)
        else:
            self.register_parameter(
                "radial_filters",
                torch.nn.Parameter(filters, requires_grad=True),
            )

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

        # Addition for filtering
        senders = data.atom_types[edge_index[0]]
        receivers = data.atom_types[edge_index[1]]

        # Ensure senders is always smaller than receivers, flip otherwise
        mask = senders > receivers
        senders[mask], receivers[mask] = receivers[mask], senders[mask]

        rbf_filters = self.radial_filters[senders, receivers]

        rbf_expansion = self.rbf_layer(distances) * torch.abs(rbf_filters)
        # End of addition

        num_batch = data.batch[-1] + 1
        for block in self.interaction_blocks:
            x = x + block(
                x, edge_index, distances, rbf_expansion, num_batch, data.batch
            )

        energy = self.output_network(x, data)
        energy = scatter(energy, data.batch, dim=0, reduce="sum")
        energy = energy.flatten()
        data.out[self.name] = {
            ENERGY_KEY: energy,
            "radial_filters": self.radial_filters,
        }

        return data
