import torch
from ..data.atomic_data import AtomicData, ENERGY_KEY, FORCE_KEY
from allegro.model import AllegroModel
from allegro.nn import TwoBodyBesselScalarEmbed,TwoBodySplineScalarEmbed, TwoBodySphericalHarmonicTensorEmbed
from nequip.data import AtomicDataDict
from nequip.utils.global_state import set_global_state
from torch_cluster import radius_graph
from typing import Final, Optional, List

DEFAULT_ALLEGRO_MODEL_PARAMS = {
    "seed": 123,
    "type_names": ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU", "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"],
    "model_dtype": "float32",
    "r_max": 4.0,
    "avg_num_neighbors": 20.0,
    "radial_chemical_embed_dim": 16,
    "scalar_embed_mlp_hidden_layers_depth": 1,
    "scalar_embed_mlp_hidden_layers_width": 32,
    "num_layers": 2,
    "l_max": 2,
    "num_scalar_features": 32,
    "num_tensor_features": 4,
    "allegro_mlp_hidden_layers_depth": 2,
    "allegro_mlp_hidden_layers_width": 32,
    "readout_mlp_hidden_layers_depth": 1,
    "readout_mlp_hidden_layers_width": 8,
    "radial_chemical_embed": {
        "_target_": "allegro.nn.TwoBodyBesselScalarEmbed",
        "num_bessels": 8,
    },
}


def embedding_size_to_type_names(embedding_size:int) -> List[str]:
    return [f"emb_{i}" for i in range(embedding_size+1)]

class StandardAllegro(torch.nn.Module):
    r"""Small wrapper class for an allegro model. The implementation is taken
    from the allegro package. We give access to some of the hyperparameters, 
    and take the documentation from the allegro package.

    Parameters
    ----------
    embedding_size: 
        number of different atom types in the system
    r_max: 
        cutoff radius
    l_max: 
        maximum order :math:`\ell` to use in spherical harmonics embedding, 1 is baseline (fast), 2 is more accurate, but slower, 3 highly accurate but slow
    radial_chemical_embed: 
        an Allegro-compatible two-body radial-chemical embedding module, e.g. :class:`allegro.nn.TwoBodyBesselScalarEmbed`.
        (default `None` will use `allegro.nn.TwoBodyBesselScalarEmbed` with `n_bessels=8`)
    radial_chemical_output_dim: 
        output dimension of the radial_che embedding module (default ``None`` will use ``num_scalar_features``)
    num_layers: 
        number of Allegro layers
    num_scalar_features: 
        multiplicity of scalar features in the Allegro layers 
    num_tensor_features: 
        multiplicity of tensor features in the Allegro layers
    allegro_mlp_hidden_layers_depth: 
        number of hidden layers in the Allegro scalar MLPs
    allegro_mlp_hidden_layers_width: 
        width of hidden layers in the Allegro scalar MLPs (reasonable to set it to be the same as ``num_scalar_features``)
    allegro_mlp_nonlinearity: 
        ``silu``, ``mish``, ``gelu``, or ``None`` (default ``silu``)
    readout_mlp_hidden_layers_depth: 
        number of hidden layers in the readout MLP (default ``1``)
    readout_mlp_hidden_layers_width: 
        width of hidden layers in the readout MLP (reasonable to set it to be the same as ``num_scalar_features``)
    readout_mlp_nonlinearity: 
        ``silu``, ``mish``, ``gelu``, or ``None`` (default ``silu``)
    avg_num_neighbors: 
        used to normalize edge sums for better numerics (default ``None``)
    parity: 
        whether to include features with odd mirror parity (default ``True``)
    seed:
        seed used to initialize the allegro model.
    dtype: 
        data typed used for the model 

    Notes
    ------

    -Allegro by default provides an interface to obtain forces from their 
    model. To avoid any further problems, we use their interface by default.
    This means that the allegro model doesn't need to be wrapped in a 
    `GradientsOut` to get the forces. 

    -Nequip (on which allegro depends) requires to set a global state in 
    which a seed is hardcoded to 123 and the dtype to float32. Check 
    `nequi/utils/global-state.py:set_global_state` for more details.

    """
    name: Final[str] = "Allegro"
    def __init__(
        self,
        embedding_size: int = 100,
        r_max: float = 4.0,
        l_max: int = 2,
        # scalar embed
        radial_chemical_embed : Optional[dict] = None,
        radial_chemical_embed_dim: Optional[int] = None,
        # scalar embed mlp
        scalar_embed_mlp_hidden_layers_depth: int = 1,
        scalar_embed_mlp_hidden_layers_width: int = 32,
        scalar_embed_mlp_nonlinearity: Optional[str] = "silu",
        # allegro layers
        num_layers: int = 2,
        num_scalar_features: int = 32,
        num_tensor_features: int = 4,
        allegro_mlp_hidden_layers_depth: int = 2,
        allegro_mlp_hidden_layers_width: int = 32,
        allegro_mlp_nonlinearity: Optional[str] = "silu",
        # readout
        readout_mlp_hidden_layers_depth: int = 1,
        readout_mlp_hidden_layers_width: int = 8,
        readout_mlp_nonlinearity: Optional[str] = "silu",
        # edge sum normalization
        avg_num_neighbors: Optional[float] = 10.0,
        # extra
        parity: bool = False,
        seed: int = 123,
        model_dtype: str = "float32",
    ):
        super(StandardAllegro,self).__init__()
        set_global_state()
        local_params = {k:v for k,v in locals().items() if k not in ["__class__","self","embedding_size"]}
        print(local_params.keys())
        if local_params["radial_chemical_embed"] is None:
            local_params["radial_chemical_embed"] = {
                "_target_": "allegro.nn.TwoBodyBesselScalarEmbed",
                "num_bessels": 8,
            },
        model_params = local_params
        model_params["type_names"] = embedding_size_to_type_names(embedding_size)
        print("--------------------------------------------")
        print()
        print("Initializing allegro model with hyperparams:")
        print()
        print("--------------------------------------------")
        print()
        for key,val in model_params.items():
            print(f"     {key}: {val}")
        print()
        print()
        print("--------------------------------------------")
        self.current_model_params = model_params
        self.r_max = model_params["r_max"]
        self.allegro_model = AllegroModel(**model_params)


    def forward(self, data: AtomicData):
        r"""Evaluate an the model in an AtomicData object.
        
        Remember that the function populates already both energy and forces key
        """
        edge_index_subset = radius_graph(
            data.pos,
            r=self.r_max,
            batch=data.batch
        )
        data.pos.requires_grad_(True)
        atoms_per_batch = torch.tensor([(data.batch == i).sum().item() for i in torch.unique(data.batch)])
        allegro_subset_data = {
            AtomicDataDict.POSITIONS_KEY: data.pos,
            AtomicDataDict.ATOM_TYPE_KEY: data.atom_types,
            AtomicDataDict.EDGE_INDEX_KEY: edge_index_subset,
            AtomicDataDict.BATCH_KEY: data.batch,
            AtomicDataDict.NUM_NODES_KEY: atoms_per_batch,
        }
        allegro_output = self.allegro_model(allegro_subset_data)
        data.out[self.name] = {ENERGY_KEY: allegro_output["total_energy"].flatten(),FORCE_KEY: allegro_output["forces"]}
        data.pos = data.pos.detach()
        return data
    
