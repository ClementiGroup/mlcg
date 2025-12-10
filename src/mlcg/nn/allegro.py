"""
Unraveled implementation of the Allegro neural network potential.

This module provides an unraveled implementation of the Allegro model, a
message-passing neural network for molecular dynamics simulations. It computes
atomic energies and forces using spherical harmonic representations of atomic
environments.

Unlike the original implementation, this version explicitly tracks energy
contributions through each component, making it more transparent and easier
to debug or modify.
"""

import math
from e3nn import o3
import torch

from nequip.data import AtomicDataDict
from nequip.nn import (
    ScalarMLP,
    PerTypeScaleShift,
)

from nequip.nn.embedding import (
    EdgeLengthNormalizer,
)

from allegro.nn import (
    TwoBodySphericalHarmonicTensorEmbed,
    EdgewiseReduce,
    Allegro_Module,
)

from typing import Sequence, Union, Optional, Dict, Final, List
from mlcg.data.atomic_data import AtomicData, ENERGY_KEY, FORCE_KEY
from torch_geometric.utils import scatter
from mlcg.neighbor_list.neighbor_list import (
    atomic_data2neighbor_list,
    validate_neighborlist,
)

from nequip.nn.mlp import ScalarLinearLayer


def init_xavier_uniform(
    module: torch.nn.Module, zero_bias: bool = True
) -> None:
    r"""initialize (in place) weights of the input module using xavier uniform.
    Works only on `torch.nn.Linear` at the moment and the bias are set to 0
    by default.

    Parameters
    ----------
    module:
        a torch module
    zero_bias:
        If True, the bias will be filled with zeroes. If False,
        the bias will be filled according to the torch.nn.Linear
        default distribution:

        .. math:

            \text{Uniform}(-\sqrt{k}, sqrt{k}) ; \quad \quad k = \frac{1}{\text{in_features}}

    """
    if isinstance(module, ScalarLinearLayer):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None and zero_bias == True:
            torch.nn.init.constant_(module.bias, 0.0)


class Allegro(torch.nn.Module):
    """
    Base Allegro implementation for energy and force prediction.

    This class provides the core functionality for the Allegro model,
    connecting various components like edge embeddings, spherical harmonic
    representations, and readout layers to predict per-molecule energies.

    Parameters
    ----------
    edge_norm : torch.nn.Module
        Module that normalizes edge vectors and lengths.
    radial_chemical_embed : torch.nn.Module
        Module that embeds edge distances and atom types.
    scalar_embed_mlp : torch.nn.Module
        MLP that processes scalar edge embeddings.
    tensor_embed : torch.nn.Module
        Module that embeds edge features in spherical harmonic space.
    allegro : torch.nn.Module
        The main Allegro message-passing module.
    edge_readout : torch.nn.Module
        Module that converts edge features to edge energies.
    edge_eng_sum : torch.nn.Module
        Module that aggregates edge energies to per-atom energies.
    per_type_energy_scale_shift : torch.nn.Module
        Module that applies type-dependent scaling and shifting to energies.
    pair_potential : Optional[Dict], default=None
        Optional pair potential to add to the energy.
    max_num_neighbors : int, default=1000
        Maximum number of neighbors per atom to consider.
    """

    name: Final[str] = "allegro"
    # list of labels that need to be removed after a forward pass
    # from an atomic data to avoid mismatches
    cleanup_labels: List[str] = [
        "edge_attrs",
        "edge_cutoff",
        "edge_features",
        "edge_embedding",
        "edge_index",
        "normed_edge_lengths",
        "edge_vectors",
        "edge_energy",
        "edge_lengths",
    ]

    def __init__(
        self,
        edge_norm,
        radial_chemical_embed,
        scalar_embed_mlp,
        tensor_embed,
        allegro,
        edge_readout,
        edge_eng_sum,
        per_type_energy_scale_shift,
        pair_potential=None,
        max_num_neighbors: int = 1000,
    ):
        super().__init__()

        self.edge_norm = edge_norm
        self.radial_chemical_embed = radial_chemical_embed
        self.scalar_embed_mlp = scalar_embed_mlp
        self.tensor_embed = tensor_embed
        self.allegro = allegro
        self.edge_readout = edge_readout
        self.edge_eng_sum = edge_eng_sum
        self.per_type_energy_scale_shift = per_type_energy_scale_shift
        self.pair_potential = pair_potential
        self.max_num_neighbors = max_num_neighbors

        self.reset_parameters()

    def forward(self, data):
        """
        Forward pass of the Allegro model.

        Processes atomic data through the model components to predict per-molecule
        energies and, optionally, forces.

        Parameters
        ----------
        data : AtomicData
            Input atomic data containing positions, atom types, and batch information.

        Returns
        -------
        data : AtomicData
            The input data object with added energy predictions under data.out[self.name].
        """
        dtype = data.pos.dtype

        neighbor_list = data.neighbor_list.get(self.name)

        if not self.is_nl_compatible(neighbor_list):
            neighbor_list = self.neighbor_list(
                data, self.r_max, self.max_num_neighbors
            )[self.name]

        edge_index = neighbor_list["index_mapping"]

        data[AtomicDataDict.EDGE_INDEX_KEY] = edge_index
        data[AtomicDataDict.NUM_NODES_KEY] = data.n_atoms
        # FIXME: check compatibility with pbc

        # populates 'edge_vectors', 'edge_lengths', 'normed_edge_lengths'
        data = self.edge_norm(data)
        # populates 'edge_cutoff', 'edge_embedding'
        data = self.radial_chemical_embed(data)
        data = self.scalar_embed_mlp(data)

        # populates 'edge_attrs', 'edge_features'
        data = self.tensor_embed(data)

        # updates 'edge_features'
        data = self.allegro(data)

        # FIXME: check if we can change from here on

        # populates 'edge_energy'
        data = self.edge_readout(data)

        # aggregate 'edge_energy' in 'atomic_energy'
        data = self.edge_eng_sum(data)

        # update 'atomic_energy'
        data = self.per_type_energy_scale_shift(data)

        energy = (
            scatter(
                data["atomic_energy"],
                data.batch,
                dim=0,
                dim_size=data.n_atoms.size(0),
                reduce="sum",
            )
            .to(dtype)
            .flatten()
        )

        data.out[self.name] = {ENERGY_KEY: energy}
        # cleaning edge-related properties
        for key in self.cleanup_labels:
            del data[key]
        return data

    def reset_parameters(self):
        """
        Reset parameters of all submodules to their initial values.
        Currently accepts default values (the function does nothing),
        but commented code uses Xavier distribution to try to replicate
        Allegro functionality precisely.
        """
        pass

        # init_xavier_uniform(self.edge_norm)
        # init_xavier_uniform(self.scalar_embed_mlp)
        # init_xavier_uniform(self.tensor_embed)
        # init_xavier_uniform(self.allegro)
        # init_xavier_uniform(self.edge_readout)
        # pass

    def is_nl_compatible(self, nl):
        """
        Check if a neighbor list is compatible with this model.

        Parameters
        ----------
        nl : dict
            The neighbor list to check.

        Returns
        -------
        bool
            True if the neighbor list is compatible, False otherwise.
        """
        is_compatible = False
        if validate_neighborlist(nl):
            if (
                nl["order"] == 2
                and nl["self_interaction"] is False
                and nl["rcut"] == self.r_max
            ):
                is_compatible = True
        return is_compatible

    @staticmethod
    def neighbor_list(
        data: AtomicData, rcut: float, max_num_neighbors: int = 1000
    ) -> dict:
        """
        Compute a neighbor list for the given atomic data.

        Parameters
        ----------
        data : AtomicData
            The atomic data to compute neighbor lists for.
        rcut : float
            Cutoff radius for neighbor identification.
        max_num_neighbors : int, default=1000
            Maximum number of neighbors per atom.

        Returns
        -------
        dict
            A dictionary containing the computed neighbor list.
        """
        return {
            Allegro.name: atomic_data2neighbor_list(
                data,
                rcut,
                self_interaction=False,
                max_num_neighbors=max_num_neighbors,
            )
        }


class StandardAllegro(Allegro):
    """
    Standard implementation of the Allegro model with configurable parameters.

    This class provides a higher-level interface to the Allegro model, allowing
    users to specify model parameters directly rather than constructing each
    component manually.

    Parameters
    ----------
    r_max : float, default=4.0
        Cutoff radius for neighbor identification.
    embedding_size : int, default=100
        Dimension of the atom embedding space.
    l_max : int, default=2
        Maximum spherical harmonic degree to use.
    parity : bool, default=False
        Whether to use parity-equivariant features.
    radial_chemical_embed : Optional[dict], default=None
        Configuration for the radial chemical embedding.
    radial_chemical_embed_dim : Optional[int], default=None
        Dimension of the radial chemical embedding.
    per_edge_type_cutoff : Optional[Dict], default=None
        Cutoff parameters for each edge type.
    scalar_embed_mlp_hidden_layers_depth : int, default=1
        Depth of the scalar embedding MLP.
    scalar_embed_mlp_hidden_layers_width : int, default=64
        Width of the scalar embedding MLP.
    scalar_embed_mlp_nonlinearity : Optional[str], default="silu"
        Nonlinearity to use in the scalar embedding MLP.
    num_layers : int, default=2
        Number of message-passing layers.
    num_scalar_features : int, default=64
        Number of scalar features.
    num_tensor_features : int, default=16
        Number of tensor features.
    allegro_mlp_hidden_layers_depth : int, default=1
        Depth of the Allegro MLP.
    allegro_mlp_hidden_layers_width : int, default=64
        Width of the Allegro MLP.
    allegro_mlp_nonlinearity : Optional[str], default="silu"
        Nonlinearity to use in the Allegro MLP.
    tp_path_channel_coupling : bool, default=True
        Whether to use channel coupling in tensor product paths.
    readout_mlp_hidden_layers_depth : int, default=1
        Depth of the readout MLP.
    readout_mlp_hidden_layers_width : int, default=32
        Width of the readout MLP.
    readout_mlp_nonlinearity : Optional[str], default="silu"
        Nonlinearity to use in the readout MLP.
    avg_num_neighbors : Optional[float], default=10.0
        Average number of neighbors per atom for normalization.
    weight_individual_irreps : bool, default=True
        Whether to use separate weights for each irreducible representation.
    per_type_energy_scales : Optional[Union[float, Sequence[float]]], default=None
        Scaling factors for energies of each atom type.
    per_type_energy_shifts : Optional[Union[float, Sequence[float]]], default=None
        Shift values for energies of each atom type.
    per_type_energy_scales_trainable : Optional[bool], default=False
        Whether the energy scales are trainable parameters.
    per_type_energy_shifts_trainable : Optional[bool], default=False
        Whether the energy shifts are trainable parameters.
    pair_potential : Optional[Dict], default=None
        Optional pair potential to add to the energy.
    forward_normalize : bool, default=True
        Whether to normalize weights in the forward pass.
    """

    def __init__(
        self,
        r_max: float = 4.0,
        embedding_size: int = 100,
        # irreps
        l_max: int = 2,
        parity: bool = False,
        # scalar embed
        radial_chemical_embed: Optional[dict] = None,
        radial_chemical_embed_dim: Optional[int] = None,
        per_edge_type_cutoff: Optional[
            Dict[str, Union[float, Dict[str, float]]]
        ] = None,  # FIXME: check if useful
        # scalar embed MLP
        scalar_embed_mlp_hidden_layers_depth: int = 1,
        scalar_embed_mlp_hidden_layers_width: int = 64,
        scalar_embed_mlp_nonlinearity: Optional[str] = "silu",
        # allegro layers
        num_layers: int = 2,
        num_scalar_features: int = 64,
        num_tensor_features: int = 16,
        allegro_mlp_hidden_layers_depth: int = 1,
        allegro_mlp_hidden_layers_width: int = 64,
        allegro_mlp_nonlinearity: Optional[str] = "silu",
        tp_path_channel_coupling: bool = True,  # FIXME: check if useful
        # readout
        readout_mlp_hidden_layers_depth: int = 1,
        readout_mlp_hidden_layers_width: int = 32,
        readout_mlp_nonlinearity: Optional[str] = "silu",
        # edge sum normalization
        avg_num_neighbors: Optional[float] = 10.0,
        # allegro layers defaults
        weight_individual_irreps: bool = True,
        # per atom energy params
        per_type_energy_scales: Optional[Union[float, Sequence[float]]] = None,
        per_type_energy_shifts: Optional[Union[float, Sequence[float]]] = None,
        per_type_energy_scales_trainable: Optional[bool] = False,
        per_type_energy_shifts_trainable: Optional[bool] = False,
        pair_potential: Optional[Dict] = None,
        # weight initialization and normalization
        forward_normalize: bool = True,
    ):

        ## haking jit for module
        _original_script = torch.jit.script
        torch.jit.script = lambda fn: fn  # No-op

        self.r_max = r_max
        irreps_edge_sh = repr(o3.Irreps.spherical_harmonics(l_max, p=-1))
        # set tensor_track_allowed_irreps
        # note that it is treated as a set, so order doesn't really matter
        if parity:
            # we want all irreps up to lmax
            tensor_track_allowed_irreps = o3.Irreps(
                [
                    (1, (this_l, p))
                    for this_l in range(l_max + 1)
                    for p in (1, -1)
                ]
            )
        else:
            # we want only irreps that show up in the original SH
            tensor_track_allowed_irreps = irreps_edge_sh

        type_names = self.embedding_size_to_type_names(embedding_size)

        if radial_chemical_embed is None:
            radial_chemical_embed = {
                "_target_": "allegro.nn.TwoBodyBesselScalarEmbed",
                "num_bessels": 8,
                "polynomial_cutoff_p": 6,
            }

        edge_norm = EdgeLengthNormalizer(
            r_max=r_max,
            type_names=type_names,
            per_edge_type_cutoff=per_edge_type_cutoff,
        )

        from mlcg.pl.model import get_class_from_str

        radial_chemical_embed_module = get_class_from_str(
            radial_chemical_embed["_target_"]
        )(
            num_bessels=radial_chemical_embed["num_bessels"],
            polynomial_cutoff_p=radial_chemical_embed["polynomial_cutoff_p"],
            type_names=type_names,
            module_output_dim=(
                num_scalar_features
                if radial_chemical_embed_dim is None
                else radial_chemical_embed_dim
            ),
            forward_weight_init=forward_normalize,
            scalar_embed_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
            irreps_in=edge_norm.irreps_out,
        )

        scalar_embed_mlp = ScalarMLP(
            output_dim=num_scalar_features,
            hidden_layers_depth=scalar_embed_mlp_hidden_layers_depth,
            hidden_layers_width=scalar_embed_mlp_hidden_layers_width,
            nonlinearity=scalar_embed_mlp_nonlinearity,
            bias=False,
            forward_weight_init=forward_normalize,
            field=AtomicDataDict.EDGE_EMBEDDING_KEY,
            out_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
            irreps_in=radial_chemical_embed_module.irreps_out,
        )

        ##
        tensor_embed = TwoBodySphericalHarmonicTensorEmbed(
            irreps_edge_sh=irreps_edge_sh,
            num_tensor_features=num_tensor_features,
            forward_weight_init=forward_normalize,
            scalar_embedding_in_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
            tensor_basis_out_field=AtomicDataDict.EDGE_ATTRS_KEY,
            tensor_embedding_out_field=AtomicDataDict.EDGE_FEATURES_KEY,
            irreps_in=scalar_embed_mlp.irreps_out,
        )

        allegro = Allegro_Module(
            num_layers=num_layers,
            num_scalar_features=num_scalar_features,
            num_tensor_features=num_tensor_features,
            tensor_track_allowed_irreps=tensor_track_allowed_irreps,
            avg_num_neighbors=avg_num_neighbors,
            # MLP
            latent_kwargs={
                "hidden_layers_depth": allegro_mlp_hidden_layers_depth,
                "hidden_layers_width": allegro_mlp_hidden_layers_width,
                "nonlinearity": allegro_mlp_nonlinearity,
                "bias": False,
                "forward_weight_init": forward_normalize,
            },
            tp_path_channel_coupling=tp_path_channel_coupling,
            # best to use defaults for these
            weight_individual_irreps=weight_individual_irreps,
            # fields
            tensor_basis_in_field=AtomicDataDict.EDGE_ATTRS_KEY,
            tensor_features_in_field=AtomicDataDict.EDGE_FEATURES_KEY,
            scalar_in_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
            scalar_out_field=AtomicDataDict.EDGE_FEATURES_KEY,
            irreps_in=tensor_embed.irreps_out,
        )

        edge_readout = ScalarMLP(
            output_dim=1,
            hidden_layers_depth=readout_mlp_hidden_layers_depth,
            hidden_layers_width=readout_mlp_hidden_layers_width,
            nonlinearity=readout_mlp_nonlinearity,
            bias=False,
            forward_weight_init=forward_normalize,
            field=AtomicDataDict.EDGE_FEATURES_KEY,
            out_field=AtomicDataDict.EDGE_ENERGY_KEY,
            irreps_in=allegro.irreps_out,
        )

        edge_eng_sum = EdgewiseReduce(
            field=AtomicDataDict.EDGE_ENERGY_KEY,
            out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
            factor=1.0 / math.sqrt(2 * avg_num_neighbors),
            # ^ factor of 2 to normalize dE/dr_i which includes both contributions from dE/dr_ij and every other derivative against r_ji
            irreps_in=edge_readout.irreps_out,
        )

        per_type_energy_scale_shift = PerTypeScaleShift(
            type_names=type_names,
            field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
            out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
            scales=per_type_energy_scales,
            shifts=per_type_energy_shifts,
            scales_trainable=per_type_energy_scales_trainable,
            shifts_trainable=per_type_energy_shifts_trainable,
            irreps_in=edge_eng_sum.irreps_out,
        )

        super().__init__(
            edge_norm=edge_norm,
            radial_chemical_embed=radial_chemical_embed_module,
            scalar_embed_mlp=scalar_embed_mlp,
            tensor_embed=tensor_embed,
            allegro=allegro,
            edge_readout=edge_readout,
            edge_eng_sum=edge_eng_sum,
            per_type_energy_scale_shift=per_type_energy_scale_shift,
            pair_potential=pair_potential,
        )
        ## Restoring jit
        torch.jit.script = _original_script

    @staticmethod
    def embedding_size_to_type_names(embedding_size: int) -> List[str]:
        """
        Generate type names for a given embedding size.

        Parameters
        ----------
        embedding_size : int
            Size of the embedding space.

        Returns
        -------
        List[str]
            List of type names.
        """
        return [f"emb_{i}" for i in range(embedding_size + 1)]
