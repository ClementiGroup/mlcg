from torch_scatter import scatter
from mlcg.nn.prior import _Prior
import torch
from typing import Final, Optional, Dict

from mlcg.geometry.topology import Topology
from mlcg.geometry.internal_coordinates import compute_distances
from mlcg.data.atomic_data import AtomicData
from mlcg.nn.gradients import GradientsOut, SumOut


class Repulsion(_Prior):
    r"""1-D power law repulsion prior for feature :math:`x` of the form:

    .. math::

        U_{ \textnormal{Repulsion}}(x) = (\sigma/x)^6

    where :math:`\sigma` is the excluded volume.

    Parameters
    ----------
    statistics:
        Dictionary of interaction parameters for each type of atom pair,
        where the keys are tuples of interacting bead types and the
        corresponding values define the interaction parameters. These
        Can be hand-designed or taken from the output of
        `mlcg.geometry.statistics.compute_statistics`, but must minimally
        contain the following information for each key:

        .. code-block:: python

            tuple(*specific_types) : {
                "sigma" : torch.Tensor scalar that describes the excluded
                    volume of the two interacting atoms.
                ...

                }
        The keys can be tuples of 2 integer atom types.
    """

    name: Final[str] = "repulsion"
    _neighbor_list_name = "fully connected"

    def __init__(self, statistics: Dict) -> None:
        super(Repulsion, self).__init__()
        keys = torch.tensor(list(statistics.keys()), dtype=torch.long)
        self.allowed_interaction_keys = list(statistics.keys())
        self.order = 2
        self.name = self.name
        unique_types = torch.unique(keys.flatten())
        assert unique_types.min() >= 0
        max_type = unique_types.max()
        sizes = tuple([max_type + 1 for _ in range(self.order)])
        alpha = torch.zeros(sizes)
        r_0 = torch.zeros(sizes)
        for key in statistics.keys():
            alpha[key] = statistics[key]["alpha"]
            r_0[key] = statistics[key]["r_0"]
        self.register_buffer("alpha", alpha)
        self.register_buffer("r_0", r_0)

    def data2features(self, data: AtomicData) -> torch.Tensor:
        """Computes features for the harmonic interaction from
        an AtomicData instance)

        Parameters
        ----------
        data:
            Input `AtomicData` instance

        Returns
        -------
        torch.Tensor:
            Tensor of computed features
        """

        mapping = data.neighbor_list[self.name]["index_mapping"]
        return Repulsion.compute_features(data.pos, mapping)
    
    def data2parameters(self, data):
        mapping = data.neighbor_list[self.name]["index_mapping"]
        interaction_types = [
            data.atom_types[mapping[ii]] for ii in range(self.order)
        ]
        params = {
            "alpha": self.alpha[interaction_types].flatten(),
            "r_0": self.r_0[interaction_types].flatten(),
        }
        return params
    
    def forward(self, data: AtomicData) -> AtomicData:
        """Forward pass through the repulsion interaction.

        Parameters
        ----------
        data:
            Input AtomicData instance that possesses an appropriate
            neighbor list containing both an 'index_mapping'
            field and a 'mapping_batch' field for accessing
            beads relevant to the interaction and scattering
            the interaction energies onto the correct example/structure
            respectively.

        Returns
        -------
        AtomicData:
            Updated AtomicData instance with the 'out' field
            populated with the predicted energies for each
            example/structure
        """

        mapping = data.neighbor_list[self.name]["index_mapping"]
        mapping_batch = data.neighbor_list[self.name]["mapping_batch"]
        interaction_types = [
            data.atom_types[mapping[ii]] for ii in range(self.order)
        ]
        params = self.data2parameters(data)
        features = self.data2features(data).flatten()
        y = Repulsion.compute(features, **params)
        y = scatter(y, mapping_batch, dim=0, reduce="sum")
        data.out[self.name] = {"energy": y}
        return data

    @staticmethod
    def compute_features(pos, mapping):
        return compute_distances(pos, mapping)

    @staticmethod
    def compute(x, alpha, r_0):
        """Method defining the repulsion interaction"""
        rr = 1 - (x / r_0)
        return (6 / alpha) * torch.exp(alpha * rr)



    @staticmethod
    def neighbor_list(topology: Topology) -> Dict:
        """Method for computing a neighbor list from a topology
        and a chosen feature type.

        Parameters
        ----------
        topology:
            A Topology instance with a defined fully-connected
            set of edges.

        Returns
        -------
        Dict:
            Neighborlist of the fully-connected distances
            according to the supplied topology
        """

        return {
            Repulsion.name: topology.neighbor_list(
                Repulsion._neighbor_list_name
            )
        }
