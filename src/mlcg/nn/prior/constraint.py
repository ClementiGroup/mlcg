import numpy as np
import torch
from typing import Final, Optional, Dict, List
from torch_geometric.utils import scatter

from .base import _Prior
from ...data.atomic_data import AtomicData
from ...geometry.topology import Topology
from ...geometry.internal_coordinates import (
    safe_norm
)


class Constraint(_Prior):
    r"""1-D power law constraint prior for feature :math:`x` of the form:

    .. math::

        U_{ \textnormal{Constrain}}(x) = ReLU(x-x_0)*(x-x_0)

    where :math:`x_0` is the center of mass of the protein.

    Parameters
    ----------
    """

    name: Final[str] = "constraint"
    _neighbor_list_name: "full"

    def __init__(self, constraint_atom_types: List[int], r_s: float) -> None:
        super(Constraint, self).__init__()
        self.register_buffer("constraint_atom_types", torch.tensor(constraint_atom_types, dtype=torch.int))
        self.register_buffer("r_s", torch.tensor(r_s))

    def data2features(
            self, 
            data: AtomicData, 
            constraint_mapping: torch.Tensor, 
            protein_mapping: torch.Tensor, 
            constraint_mapping_batch: torch.Tensor, 
            protein_mapping_batch: torch.Tensor
        ) -> torch.Tensor:
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
        # We want this function to compute, for each atom that has a type corresponding to constraint_atom_types 
        # to return the distance between this atom and the center of mass of everything that is not constrained

        protein_pos = data.pos[protein_mapping]
        protein_com = scatter(protein_pos, protein_mapping_batch, reduce="mean") 
        com_per_constraint = protein_com[constraint_mapping_batch]
        constraint_pos = data.pos[constraint_mapping]

        dr = constraint_pos - com_per_constraint

        return safe_norm(dr, dim=[1]).reshape(-1)
    
    def data2parameters(self, data:AtomicData) -> torch.Tensor:
        n_elts = torch.isin(data.atom_types, self.constraint_atom_types).sum()
        r_s = self.r_s * torch.ones(n_elts)
        return r_s

    def forward(self, data: AtomicData) -> AtomicData:
        """Forward pass through the Constraint interaction.

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
        
        constraint_mapping = torch.where(torch.isin(data.atom_types, self.constraint_atom_types))[0] #the indices where atom_types[indices] correspond to constain_atom_types
        protein_mapping = torch.where(~torch.isin(data.atom_types, self.constraint_atom_types))[0] # the indices there atom_types[indices] are NOT in constrain_atom_types
        
        protein_mapping_batch = data.neighbor_list[self.name]["mapping_batch"][protein_mapping]
        constraint_mapping_batch = data.neighbor_list[self.name]["mapping_batch"][constraint_mapping]

        r_s = self.data2parameters(data)
        features = self.data2features(data, 
                                      constraint_mapping, 
                                      protein_mapping, 
                                      constraint_mapping_batch, 
                                      protein_mapping_batch)
        y = Constraint.compute(features, r_s)
        y = scatter(y, constraint_mapping_batch, dim=0, reduce="sum") # h
        data.out[self.name] = {"energy": y}
        return data

    @staticmethod
    def compute(r, r_s):
        """Method defining the Constraint interaction"""
        rr = (r - r_s)
        return torch.relu(rr) * rr


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
            Constraint.name: topology.neighbor_list(
                Constraint._neighbor_list_name
            )
        }