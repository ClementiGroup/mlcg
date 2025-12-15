from typing import Dict
import torch
from torch_scatter import scatter


from .base import _Prior
from ...data.atomic_data import AtomicData
from ...geometry.internal_coordinates import (
    compute_angles_raw,
)


class RestrictedQuartic(_Prior):
    r"""
    Prior representing an angle potential with the following energy ansatz:

    .. math::

        V(\theta) = a\cos^4(\theta) + b\cos^3(\theta) + c\cos^2(\theta) + d\cos(\theta) + \frac{k}{\sin^2(\theta)} + V_0

    This potential combines a fourth grade polynomial expansion in cosine terms
    with a repulsive term that prevents angles from approaching 0 or pi.
    More information on this potential can be found in the following reference:
    Journal of Chemical Theory and Computation 2013 9 (8), 3282-3292
    DOI: 10.1021/ct400219n

    Parameters
    ----------
    statistics : dict
        Dictionary of interaction parameters for each type of atom combination,
        where the keys are tuples of interacting bead types (must be 3-tuples for angles)
        and the corresponding values define the interaction parameters. These
        can be hand-designed or taken from the output of
        `mlcg.geometry.statistics.compute_statistics`, but must minimally
        contain the following information for each key:

        .. code-block:: python

            tuple(*specific_types) : {
                "a" : torch.Tensor for cos^4 coefficient
                "b" : torch.Tensor for cos^3 coefficient
                "c" : torch.Tensor for cos^2 coefficient
                "d" : torch.Tensor for cos coefficient
                "k" : torch.Tensor for sin^-2 coefficient (repulsive-restriction term)
                "v_0" : torch.Tensor for constant offset
            }

        The keys must be tuples of 3 atoms.

    """

    def __init__(
        self,
        statistics: Dict,
        name: str = "angles",
    ) -> None:
        super(RestrictedQuartic, self).__init__()

        keys = torch.tensor(list(statistics.keys()), dtype=torch.long)
        self.allowed_interaction_keys = list(statistics.keys())
        self.name = name
        self.order = 3  # Just an angle prior for now

        unique_types = torch.unique(keys.flatten())
        assert unique_types.min() >= 0

        max_type = unique_types.max()
        sizes = tuple([max_type + 1 for _ in range(self.order)])

        # Initialize parameter tensors
        a = torch.zeros(sizes)
        b = torch.zeros(sizes)
        c = torch.zeros(sizes)
        d = torch.zeros(sizes)
        k = torch.zeros(sizes)
        v_0 = torch.zeros(sizes)

        # Populate parameters from statistics
        for key in statistics.keys():
            a[key] = statistics[key]["a"]
            b[key] = statistics[key]["b"]
            c[key] = statistics[key]["c"]
            d[key] = statistics[key]["d"]
            k[key] = statistics[key]["k"]
            v_0[key] = statistics[key]["v_0"]

        # Register as buffers (non-trainable parameters)
        self.register_buffer("a", a)
        self.register_buffer("b", b)
        self.register_buffer("c", c)
        self.register_buffer("d", d)
        self.register_buffer("k", k)
        self.register_buffer("v_0", v_0)

    def data2features(self, data: AtomicData) -> torch.Tensor:
        """Computes angular features from an AtomicData instance.

        Parameters
        ----------
        data : AtomicData
            Input `AtomicData` instance containing positions and neighbor lists

        Returns
        -------
        torch.Tensor
            Tensor of computed angle values (in radians)
        """
        mapping = data.neighbor_list[self.name]["index_mapping"]
        features = self.compute_features(data.pos, mapping)
        return features

    def data2parameters(self, data: AtomicData) -> Dict[str, torch.Tensor]:
        """Extracts interaction parameters based on atom types.

        Parameters
        ----------
        data : AtomicData
            Input `AtomicData` instance containing atom types

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing parameter tensors for this batch
        """
        mapping = data.neighbor_list[self.name]["index_mapping"]
        interaction_types = tuple(
            data.atom_types[mapping[ii]] for ii in range(self.order)
        )

        params = {
            "a": self.a[interaction_types].flatten(),
            "b": self.b[interaction_types].flatten(),
            "c": self.c[interaction_types].flatten(),
            "d": self.d[interaction_types].flatten(),
            "k": self.k[interaction_types].flatten(),
            "v_0": self.v_0[interaction_types].flatten(),
        }
        return params

    def forward(self, data: AtomicData) -> AtomicData:
        """Forward pass through the double angle interaction.

        Parameters
        ----------
        data : AtomicData
            Input AtomicData instance that possesses an appropriate
            neighbor list containing both an 'index_mapping'
            field and a 'mapping_batch' field for accessing
            beads relevant to the interaction and scattering
            the interaction energies onto the correct example/structure
            respectively.

        Returns
        -------
        AtomicData
            Updated AtomicData instance with the 'out' field
            populated with the predicted energies for each
            example/structure
        """
        mapping_batch = data.neighbor_list[self.name]["mapping_batch"]
        features = self.data2features(data).flatten()
        params = self.data2parameters(data)

        y = self.compute(features, **params)
        y = scatter(y, mapping_batch, dim=0, reduce="sum")

        data.out[self.name] = {"energy": y}
        return data

    @staticmethod
    def compute_features(
        pos: torch.Tensor, mapping: torch.Tensor
    ) -> torch.Tensor:
        """Computes angles from atomic positions.

        Parameters
        ----------
        pos : torch.Tensor
            Atomic positions with shape (n_atoms, 3)
        mapping : torch.Tensor
            Index mapping for angle triplets with shape (n_angles, 3)

        Returns
        -------
        torch.Tensor
            Computed angle values in radians
        """
        return compute_angles_raw(pos, mapping)

    @staticmethod
    def compute(
        x: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        d: torch.Tensor,
        k: torch.Tensor,
        v_0: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the double angle potential energy.

        .. math::

            V(\theta) = a\cos^4(\theta) + b\cos^3(\theta) + c\cos^2(\theta) +
                        d\cos(\theta) + \frac{k}{\sin^2(\theta)} + V_0

        Parameters
        ----------
        x : torch.Tensor
            Angle values in radians
        a, b, c, d : torch.Tensor
            Coefficients for the cosine polynomial terms
        k : torch.Tensor
            Coefficient for the repulsive sine term
        v_0 : torch.Tensor
            Constant energy offset

        Returns
        -------
        torch.Tensor
            Computed potential energy values
        """

        cos = torch.cos(x)
        sin = torch.sin(x)

        quart = (
            a * torch.pow(cos, 4)
            + b * torch.pow(cos, 3)
            + c * torch.pow(cos, 2)
            + d * cos
        )
        rep = k / (sin**2)
        V = quart + rep + v_0

        return V
