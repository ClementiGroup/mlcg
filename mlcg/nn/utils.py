import torch
from warnings import warn
from .prior import _Prior, Dihedral, Harmonic


def sparsify_prior_module(module: _Prior) -> torch.nn.Module:
    r"""
    Converts buffer tensors to sparse tensors inplace for Harmonic and Dihedral objects
    """
    if isinstance(module, Dihedral):
        module.v_0 = module.v_0.to_sparse()
        module.k1s = module.k1s.to_sparse()
        module.k2s = module.k2s.to_sparse()
    elif issubclass(type(module), Harmonic):
        module.x_0 = module.x_0.to_sparse()
        module.k = module.k.to_sparse()
    else:
        warn(
            f"Module is not supported for sparsification. It will be returned as is"
        )
    return module


def desparsify_prior_module(module: _Prior) -> torch.nn.Module:
    r"""
    Converts parameter tensors inplace to dense tensors in Harmonic and Dihedral objects
    """
    if isinstance(module, Dihedral):
        module.v_0 = module.v_0.to_dense()
        module.k1s = module.k1s.to_dense()
        module.k2s = module.k2s.to_dense()
    elif issubclass(type(module), Harmonic):
        module.x_0 = module.x_0.to_dense()
        module.k = module.k.to_dense()
    return module
