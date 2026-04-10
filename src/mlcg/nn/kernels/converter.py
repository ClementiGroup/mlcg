"""
Set of utilities to convert standard models to flash models.
"""

import torch
from warnings import warn
from ..schnet import StandardSchNet
from ..flash_schnet import StandardFlashSchNet
from ..prior import (
    FlashDihedral,
    FlashRepulsion,
    FlashHarmonicBonds,
    FlashHarmonicAngles,
    Repulsion,
    HarmonicBonds,
    HarmonicAngles,
    Dihedral,
)
from ..gradients import GradientsOut, SumOut, EnergyOut


def convert_single_standard_model_to_flash(
    standard_model: torch.nn.Module,
) -> torch.nn.Module:
    """
    Convert a standard model into its flash counterpart.
    Only maps supported classes.

    Args:
        standard_model (torch.nn.Module): The original SchNet model to be converted.

    """

    _supported_conversions = {
        StandardSchNet: StandardFlashSchNet,
        Repulsion: FlashRepulsion,
        HarmonicBonds: FlashHarmonicBonds,
        HarmonicAngles: FlashHarmonicAngles,
        Dihedral: FlashDihedral,
    }
    model_type = type(standard_model)
    if model_type not in _supported_conversions:
        warn(
            f"Model type {model_type} not supported for flash conversion, returning original model"
        )
        return standard_model

    flash_cls = _supported_conversions[model_type]
    return flash_cls.flash_from_standard(standard_model)


def convert_standard_model_to_flash(
    standard_model: torch.nn.Module,
) -> torch.nn.Module:
    """
    Convert a standard model into its flash counterpart.
    Only maps supported classes.

    Args:
        standard_model (torch.nn.Module): The original SchNet model to be converted.

    """

    if isinstance(standard_model, GradientsOut):
        flash_inner = convert_standard_model_to_flash(standard_model.model)
        return GradientsOut(flash_inner, targets=standard_model.targets)

    elif isinstance(standard_model, EnergyOut):
        flash_inner = convert_standard_model_to_flash(standard_model.model)
        return EnergyOut(flash_inner, targets=standard_model.targets)

    elif isinstance(standard_model, SumOut):
        flash_inner = torch.nn.ModuleDict(
            {
                name: convert_standard_model_to_flash(model)
                for name, model in standard_model.models.items()
            }
        )
        return SumOut(flash_inner, targets=standard_model.targets)

    return convert_single_standard_model_to_flash(standard_model)
