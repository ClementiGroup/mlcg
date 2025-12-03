import warnings
from typing import Optional, List, Final, Dict
import torch
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
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

try:
    from mlcg_opt_radius.radius import radius_distance
except ImportError:
    print(
        "`mlcg_opt_radius` not installed. Please check the `opt_radius` folder and follow the instructions."
    )
    radius_distance = None

from torch_geometric.data.collate import collate

class RBFFilter():
    """
    Class to define rbf filter.

    Parameters:
    -----------
    rbf_filter: torch.Tensor
        A filter array of size NxNxM where N is number
        of atom types and M is number of basis functions
    use_boolean_filter: bool, default False
        If true, a boolean filter will be generated
        based on condition rbf_filter > boolean_filter_cutoff
    """

    def __init__(self,
                 rbf_filter: torch.Tensor,
                 use_boolean_filter: bool =False,
                 boolean_filter_cutoff: float  = 0.0):
        if use_boolean_filter:
            masked_filters = rbf_filters > mask_cutoff
            self.filter = masked_filters
        else:
            self.filter = rbf_filter

    @classmethod
    def from_file(cls, path: str, **kwargs) -> "RBFFilter":
        filter_tensor = torch.load(path)
        return cls(filter_tensor, **kwargs)


class BinaryRBFFilter(RBFFilter):
    """
    Generate an RBF filter according to the following
    rules:

    1) for a given pair of atom types, set first N < M 
        filter values to 0, the rest set to 1

    2) N is determined as the index of the fist value above 
        the cutoff

    3) padd the filtering mask to the specified last dimention
    K > M with 1
    """
    def __init__(self,
                 rbf_filter: torch.Tensor,
                 n_rbfs: int, 
                 cutoff: float):
        self.filter = self._get_mask(rbf_filter, n_rbfs, cutoff)


    def _get_mask(self, rbf_filter, n_rbfs, cutoff):

        n_atom_types_0 = rbf_filter.size()[0]
        n_atom_types_1 = rbf_filter.size()[1]
        # get a standard boolean mask 
        cutoff_mask = rbf_filter > cutoff

        # find  an index of first occurence of a useful rbf
        target_index = cutoff_mask.int().argmax(dim=-1, keepdim=True)
        
        index_tensor = torch.arange(n_rbfs)
        index_tensor = index_tensor[torch.newaxis, torch.newaxis, :]
        tiled_tensor = index_tensor.tile(dims=(n_atom_types_0, n_atom_types_1, 1))
        mask = tiled_tensor >= target_index
        return mask 

    @classmethod
    def from_file(cls, path: str,  n_rbfs: int, cutoff: float) -> "BinaryRBFFilter":
        """
        Need to reimplement it to pass proper kargs via 
        Pytorch lightning
        """
        filter_tensor = torch.load(path)
        return cls(filter_tensor, n_rbfs=n_rbfs, cutoff=cutoff)


class RepulsionFilteredLinear(torch.nn.Module):
    """
 
    This is a linear RBF model with the RBF components are weighted via Hadamard product with couple-specific
    vectors. Each tuple of bead types has its own vector. 

    Parameters
    ----------
    rbf_layer:
        radial basis function used to project the distances :math:`r_{ij}`.
 
    max_num_neighbors:
        The maximum number of neighbors to return for each atom in :obj:`data`.
        If the number of actual neighbors is greater than
        :obj:`max_num_neighbors`, returned neighbors are picked randomly.

    CLASS SPECIFIC PARAMETERS -------------------------------------------------------------

    max_bead_type: int
        The maximal value of the bead_type needed to calculate all possible couples in the system.
        This considers that all bead_type couples to be tracked are in the range [0, max_bead_type-1].
    define_ranges: Dict
        A dictionary containing the ranges of the type couples to be tracked. The keys are the bead-type couples
        given as strings, and the values are the lower bounds of the ranges to be considered.
        Example: {'(i,j)': min_value, '(k,l)': min_value, ...}
    edge_reg: bool
       If True, the edge rescaling is applied. Every mesage is mutliplied by a scalar that is the regularized.
    total_nodes: int
        If edge_reg is True, the total number of nodes in the system is needed to calculate the regularization factor as each
        interaction has its own value. (N^2 - N)/2 new parameters.
    """
    name: Final[str] = "linear_rbf"

    def __init__(
        self,
        rbf_layer: torch.nn.Module,
        rbf_filters: RBFFilter | BinaryRBFFilter | None = None,
        # Parameters for RBF rescaling
        max_bead_type: int = 25,
        max_num_neighbors = 1000,
        filter_rbfs:bool = True, 
    ):
        super(RepulsionFilteredLinear, self).__init__()
    
        self.max_num_neighbors = max_num_neighbors
        self.max_bead_type = max_bead_type
        self.rbf_layer = rbf_layer
        self.filter_rbfs = filter_rbfs
        # Register  rbf_filters as a buffer 
        if filter_rbfs:
            filter_values = torch.clamp(rbf_filters.filter, min=0.0, max=1.0)
            self.register_buffer("radial_filters", torch.nn.Buffer(data=filter_values, persistent=True))

        # Creating the tensors for the coefficients 
        i, j = torch.tril_indices(max_bead_type, max_bead_type, offset=-1)
        rbf_params = torch.ones(
            max_bead_type, max_bead_type, self.rbf_layer.num_rbf
        ).float()
        rbf_params[i, j] *= 0

        self.register_parameter(
                "rbf_params", torch.nn.Parameter(rbf_params, requires_grad=True)
            )
        

    def forward(self, data: AtomicData) -> AtomicData:
        r"""Forward pass through the linear architecture with
        RBF regularization 
        
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
            if hasattr(data, "exc_pair_index"):
                raise NotImplementedError("Excluding pairs requires `mlcg_opt_radius` "
                    "to be available and model running with CUDA."
                )
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

        # Addition for filtering
        senders = data.atom_types[edge_index[0]]
        receivers = data.atom_types[edge_index[1]]
        # Ensure senders is always smaller than receivers, flip otherwise
        mask = senders > receivers
        senders[mask], receivers[mask] = receivers[mask], senders[mask]

        rbf_params = self.rbf_params[senders, receivers] 

        rbf_expansion = self.rbf_layer(distances)
        if self.filter_rbfs:
            rbf_filters = self.radial_filters[senders, receivers]
            rbf_expansion  *= rbf_filters
        weighted_expansion = rbf_expansion * rbf_params 
        # Energy per edge 
        energy_per_edge = torch.sum(weighted_expansion, dim=1) 
 
        # batch corresponding to each of the  edges 
        batch_map = data.batch[edge_index[0]] 
        energy = scatter(energy_per_edge, batch_map, dim=0, reduce="sum")
        energy = energy.flatten()
        print(energy.size())
        data.out[self.name] = {
            ENERGY_KEY: energy,
            "radial_params": self.radial_filters,
        }
        return data
    
    def is_nl_compatible(self, nl):
        is_compatible = False
        if validate_neighborlist(nl):
            if (
                nl["order"] == 2
                and nl["self_interaction"] is False
                and nl["rcut"] == self.rbf_layer.cutoff.cutoff_upper
            ):
                is_compatible = True
        return is_compatible
    

    @staticmethod
    def neighbor_list(
        data: AtomicData, rcut: float, max_num_neighbors: int = 1000
    ) -> dict:
        """Computes the neighborlist for :obj:`data` using a strict cutoff of :obj:`rcut`."""
        return {
            RepulsionFilteredLinear.name: atomic_data2neighbor_list(
                data,
                rcut,
                self_interaction=False,
                max_num_neighbors=max_num_neighbors,
            )
        }

