from .gradients import GradientsOut, SumOut, EnergyOut
from .schnet import SchNet, StandardSchNet
from .radial_basis import GaussianBasis, ExpNormalBasis
from .cutoff import CosineCutoff, IdentityCutoff
from .losses import ForceMSE, ForceRMSE, Loss
from .prior import Harmonic, HarmonicAngles, HarmonicBonds, Repulsion, Dihedral
from .mlp import MLP, TypesMLP
from .attention import ExactAttention, FavorAttention, Nonlocalinteractionblock
from .pyg_forward_compatibility import (
    get_refreshed_cfconv_layer,
    refresh_module_with_schnet_,
    load_and_adapt_old_checkpoint,
    fixed_pyg_inspector,
)
from .painn import PaiNN, StandardPaiNN
from .lr_scheduler import CustomStepLR
from .utils import sparsify_prior_module, desparsify_prior_module

try:
    from .mace_interface import MACEInterface
except Exception as e:
    print(e)
    print("MACE installation not found ...")

__all__ = [
    "GradientsOut",
    "SumOut",
    "EnergyOut",
    "SchNet",
    "StandardSchNet",
    "GaussianBasis",
    "ExpNormalBasis",
    "CosineCutoff",
    "IdentityCutoff",
    "ForceMSE",
    "ForceRMSE",
    "Loss",
    "Harmonic",
    "HarmonicAngles",
    "HarmonicBonds",
    "Repulsion",
    "MLP",
    "TypesMLP",
    "Attention",
    "Residual",
    "Residual_MLP",
    "ResidualStack",
    "ExactAttention",
    "FavorAttention",
    "Nonlocalinteractionblock",
    "get_refreshed_cfconv_layer",
    "refresh_module_with_schnet_",
    "load_and_adapt_old_checkpoint",
    "fixed_pyg_inspector",
    "PaiNN",
    "StandardPaiNN",
    "CustomStepLR",
]
