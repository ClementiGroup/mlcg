"""
Set of utilities to convert standard models to flash models.
"""

import torch
from warnings import warn
from ..schnet import StandardSchNet
from .flash_schnet import StandardFlashSchNet
from .prior import (
    FlashDihedral,
    FlashRepulsion,
    FlashHarmonicBonds,
    FlashHarmonicAngles,
)
from ..prior import (
    Repulsion,
    HarmonicBonds,
    HarmonicAngles,
    Dihedral,
    GeneralBonds,
    GeneralAngles,
)
from ..gradients import GradientsOut, SumOut, EnergyOut
from ...simulation.specialize_prior import glbls

_SUPPORTED_CONVERSIONS = {
    StandardSchNet: StandardFlashSchNet,
    Repulsion: FlashRepulsion,
    HarmonicBonds: FlashHarmonicBonds,
    HarmonicAngles: FlashHarmonicAngles,
    Dihedral: FlashDihedral,
    GeneralBonds: FlashHarmonicBonds,  # FIXME: check if this conversion is always valid
    GeneralAngles: FlashHarmonicAngles,  # FIXME: check if this conversion is always valid
    glbls["StaticHarmonicBonds"]: FlashHarmonicBonds,
    glbls["StaticHarmonicAngles"]: FlashHarmonicAngles,
    glbls["StaticDihedral"]: FlashDihedral,
    glbls["StaticGeneralBonds"]: FlashHarmonicBonds,
    glbls["StaticGeneralAngles"]: FlashHarmonicAngles,
}


def convert_single_standard_model_to_flash(
    standard_model: torch.nn.Module,
) -> torch.nn.Module:
    """
    Convert a standard model into its flash counterpart.
    Only maps supported classes; flash models are returned unchanged.

    Args:
        standard_model (torch.nn.Module): The original SchNet model to be converted.

    """
    model_type = type(standard_model)
    if model_type in _SUPPORTED_CONVERSIONS.values():
        return standard_model
    if model_type not in _SUPPORTED_CONVERSIONS:
        warn(
            f"Model type {model_type} not supported for flash conversion, returning original model"
        )
        return standard_model

    flash_cls = _SUPPORTED_CONVERSIONS[model_type]
    return flash_cls.flash_from_standard(standard_model)


def convert_standard_model_to_flash(
    standard_model: torch.nn.Module,
) -> torch.nn.Module:
    """
    Convert a standard model into its flash counterpart.
    Only maps supported classes. The result keeps the standard parameter
    names and is trainable; for fast inference see
    :func:`~mlcg.nn.flash_models.optimize_for_inference`.

    Args:
        standard_model (torch.nn.Module): The original SchNet model to be converted.

    """
    convert = convert_standard_model_to_flash

    if isinstance(standard_model, GradientsOut):
        return GradientsOut(
            convert(standard_model.model), targets=standard_model.targets
        )

    elif isinstance(standard_model, EnergyOut):
        return EnergyOut(
            convert(standard_model.model), targets=standard_model.targets
        )

    elif isinstance(standard_model, SumOut):
        flash_inner = torch.nn.ModuleDict(
            {
                name: convert(model)
                for name, model in standard_model.models.items()
            }
        )
        return SumOut(flash_inner, targets=standard_model.targets)

    return convert_single_standard_model_to_flash(standard_model)
