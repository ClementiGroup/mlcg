import torch
from torch_geometric.utils import scatter
from typing import Optional, Dict


from .base import _Prior
from ...data.atomic_data import AtomicData
from ...geometry.internal_coordinates import (
    compute_angles_cos,
)


class Polynomial(_Prior):
    r"""
    Prior representing a polynomial with
    the following energy ansatz:

    .. math::

        V(x) = V_0 + \sum_{n=1}^{n_{deg}} k_n x^n


    Parameters
    ----------
    statistics:
        Dictionary of interaction parameters for each type of atom combination,
        where the keys are tuples of interacting bead types and the
        corresponding values define the interaction parameters. These
        Can be hand-designed or taken from the output of
        `mlcg.geometry.statistics.compute_statistics`, but must minimally
        contain the following information for each key:

        .. code-block:: python

            tuple(*specific_types) : {
                "ks" : ,#torch.Tensor that contains all k_1,..,k_{n_degs} coefficients
                "v_0" : ,#torch.Tensor that contains the constant offset
                ...
                }

        The keys must be tuples of 2,3,4 atoms.
    """

    def __init__(
        self,
        statistics: dict,
        name: str,
        order: Optional[int] = None,
        n_degs: int = 4,
    ) -> None:
        r""" """
        super(Polynomial, self).__init__()
        keys = torch.tensor(list(statistics.keys()), dtype=torch.long)
        self.allowed_interaction_keys = list(statistics.keys())
        self.name = name
        self.order = order
        unique_types = torch.unique(keys.flatten())
        assert unique_types.min() >= 0

        max_type = unique_types.max()
        sizes = tuple([max_type + 1 for _ in range(self.order)])

        unique_degs = torch.unique(
            torch.tensor([len(val["ks"]) for _, val in statistics.items()])
        )
        assert (
            len(unique_degs) == 1
        ), "ks in the statistics dictionary must be of the same size for all the keys"
        assert (
            unique_degs[0] == n_degs
        ), f"length of parameters {unique_degs[0]} doesn't match degrees {n_degs}"

        self.n_degs = n_degs
        self.k_names = ["k_" + str(ii) for ii in range(1, self.n_degs + 1)]
        k = torch.zeros(self.n_degs, *sizes)
        v_0 = torch.zeros(*sizes)
        for key in statistics.keys():
            for ii in range(self.n_degs):
                k_name = self.k_names[ii]
                k[ii][key] = statistics[key]["ks"][k_name]
            v_0[key] = statistics[key]["v_0"]
        self.register_buffer("ks", k)
        self.register_buffer("v_0", v_0)
        return None

    def data2parameters(self, data: AtomicData) -> Dict:
        mapping = data.neighbor_list[self.name]["index_mapping"]
        interaction_types = [
            data.atom_types[mapping[ii]] for ii in range(self.order)
        ]
        # the parameters have shape n_features x n_degs
        ks = torch.vstack(
            [self.ks[ii][interaction_types] for ii in range(self.n_degs)]
        ).t()
        v_0s = self.v_0[interaction_types].t()
        return {"ks": ks, "v_0s": v_0s}

    def forward(self, data: AtomicData) -> AtomicData:
        r"""Forward pass through the Polynomial interaction.
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
        mapping_batch = data.neighbor_list[self.name]["mapping_batch"]
        features = self.data2features(data).flatten()
        params = self.data2parameters(data)
        # V0s =  params["v_0"] if "v_0" in params.keys() else [0 for ii in range(self.n_degs)]
        V0s = params["v_0s"].t()
        # format parameters
        # ks = [params["ks"][:,i] for i in range(self.n_degs)]
        ks = params["ks"].t()
        y = Polynomial.compute(
            features,
            ks,
            V0s,
        )
        y = scatter(y, mapping_batch, dim=0, reduce="sum")
        data.out[self.name] = {"energy": y}
        return data

    @staticmethod
    def compute(
        x: torch.Tensor, ks: torch.Tensor, V0: torch.Tensor
    ) -> torch.Tensor:
        r"""Harmonic interaction in the form of a series. The shape of the tensors
            should match between each other.

        .. math:

            V(r) = V0 + \sum_{n=1}^{deg} k_n x^n

        """
        V = ks[0] * x
        for p, k in enumerate(ks[1:], start=2):
            V += k * torch.pow(x, p)
        V += V0
        return V


class QuarticAngles(Polynomial):
    r"""Wrapper class for angle priors
    (order 3 Polynomial priors of degree 4)
    """

    def __init__(self, statistics, name="angles", n_degs: int = 4) -> None:
        super(QuarticAngles, self).__init__(
            statistics, name, order=3, n_degs=n_degs
        )

    def data2features(self, data: AtomicData) -> torch.Tensor:
        r"""Computes features for the QuarticAngle interaction from
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
        return self.compute_features(data.pos, mapping)

    @staticmethod
    def compute_features(
        pos: AtomicData, mapping: torch.Tensor
    ) -> torch.Tensor:
        return compute_angles_cos(pos, mapping)
