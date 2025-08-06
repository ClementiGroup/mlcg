import math
from e3nn import o3
import torch

from nequip.data import AtomicDataDict
from nequip.model import model_builder
from nequip.nn import (
    SequentialGraphNetwork,
    ScalarMLP,
    PerTypeScaleShift,
    ForceStressOutput,
)

from nequip.nn.embedding import (
    EdgeLengthNormalizer,
    AddRadialCutoffToData,
    PolynomialCutoff,
)
from allegro.nn import (
    TwoBodySphericalHarmonicTensorEmbed,
    EdgewiseReduce,
    Allegro_Module,
)
from nequip.utils import RankedLogger

from hydra.utils import instantiate
from typing import Sequence, Union, Optional, Dict, Final, List
from mlcg.data.atomic_data import AtomicData, ENERGY_KEY, FORCE_KEY
from torch_geometric.utils import scatter
from mlcg.neighbor_list.neighbor_list import (
    atomic_data2neighbor_list,
    validate_neighborlist,
)

from torch_cluster import radius_graph

from nequip.nn.mlp import ScalarLinearLayer

def init_xavier_uniform(
    module: torch.nn.Module, zero_bias: bool = True
) -> None:
    """initialize (in place) weights of the input module using xavier uniform.
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
    name: Final[str] = "allegro"

    def __init__(self,
                 edge_norm,
                 radial_chemical_embed,
                 scalar_embed_mlp,
                 tensor_embed,
                 allegro,
                 edge_readout,
                 edge_eng_sum,
                 per_type_energy_scale_shift,
                 pair_potential = None,
                 max_num_neighbors: int = 1000
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

        dtype = data.pos.dtype
        
        neighbor_list = data.neighbor_list.get(self.name)

        if not self.is_nl_compatible(neighbor_list):
            neighbor_list = self.neighbor_list(
                data, self.r_max, self.max_num_neighbors
            )[self.name]

        edge_index = neighbor_list["index_mapping"]

        data[AtomicDataDict.EDGE_INDEX_KEY] = edge_index
        data[AtomicDataDict.NUM_NODES_KEY] = data.n_atoms
        #FIXME: check compatibility with pbc

        # populates 'edge_vectors', 'edge_lengths', 'normed_edge_lengths'
        data = self.edge_norm(data)

        # populates 'edge_cutoff', 'edge_embedding'
        data = self.radial_chemical_embed(data)
        data = self.scalar_embed_mlp(data)

        # populates 'edge_attrs', 'edge_features'
        data = self.tensor_embed(data)

        # updates 'edge_features'
        data = self.allegro(data)

        #FIXME: check if we can change from here on

        # populates 'edge_energy'
        data = self.edge_readout(data)

        # aggregate 'edge_energy' in 'atomic_energy'
        data = self.edge_eng_sum(data)

        # update 'atomic_energy'
        data = self.per_type_energy_scale_shift(data)

        energy = scatter(data['atomic_energy'],
                         data.batch,
                         dim=0,
                         dim_size=data.n_atoms.size(0),
                         reduce='sum').to(dtype).flatten()

        data.out[self.name] = {ENERGY_KEY: energy}

        return data


    def reset_parameters(self):
        pass
        
        # init_xavier_uniform(self.edge_norm)
        # init_xavier_uniform(self.scalar_embed_mlp)
        # init_xavier_uniform(self.tensor_embed)
        # init_xavier_uniform(self.allegro)
        # init_xavier_uniform(self.edge_readout)
        # pass

    def is_nl_compatible(self, nl):
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
        """Computes the neighborlist for :obj:`data` using a strict cutoff of :obj:`rcut`."""
        return {
            Allegro.name: atomic_data2neighbor_list(
                data,
                rcut,
                self_interaction=False,
                max_num_neighbors=max_num_neighbors,
            )
        }

class StandardAllegro(Allegro):
    def __init__(
            self,
            r_max: float = 4.0,
            embedding_size: int = 100,
            # irreps
            l_max: int = 2,
            parity: bool = False,
            # scalar embed
            radial_chemical_embed : Optional[dict] = None,
            radial_chemical_embed_dim: Optional[int] = None,
            per_edge_type_cutoff: Optional[Dict[str, Union[float, Dict[str, float]]]] = None, #FIXME: check if useful
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
            tp_path_channel_coupling: bool = True, #FIXME: check if useful
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
        
        self.r_max = r_max
        irreps_edge_sh = repr(o3.Irreps.spherical_harmonics(l_max, p=-1))
        # set tensor_track_allowed_irreps
        # note that it is treated as a set, so order doesn't really matter
        if parity:
            # we want all irreps up to lmax
            tensor_track_allowed_irreps = o3.Irreps(
                [(1, (this_l, p)) for this_l in range(l_max + 1) for p in (1, -1)]
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
        radial_chemical_embed_module = get_class_from_str(radial_chemical_embed["_target_"])(
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
        # radial_chemical_embed_module = instantiate(
        #     radial_chemical_embed,
        #     type_names=type_names,
        #     module_output_dim=(
        #         num_scalar_features
        #         if radial_chemical_embed_dim is None
        #         else radial_chemical_embed_dim
        #     ),
        #     forward_weight_init=forward_normalize,
        #     scalar_embed_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
        #     irreps_in=edge_norm.irreps_out,
        # )

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
                pair_potential=pair_potential)


    @staticmethod
    def embedding_size_to_type_names(embedding_size:int) -> List[str]:
        return [f"emb_{i}" for i in range(embedding_size+1)]