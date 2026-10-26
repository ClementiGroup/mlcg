"""
Set of utilities to convert standard models to flash models.
"""

import torch
from warnings import warn
from ..schnet import StandardSchNet
from .flash_schnet import StandardFlashSchNet
from .tabulated_schnet import TabulatedSchNet
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


def convert_single_standard_model_to_flash(
    standard_model: torch.nn.Module,
    tabulate_filters: bool = True,
) -> torch.nn.Module:
    """
    Convert a standard model into its flash counterpart.
    Only maps supported classes.

    Args:
        standard_model (torch.nn.Module): The original SchNet model to be converted.
        tabulate_filters (bool): convert a ``StandardSchNet`` to the
            inference-only :class:`TabulatedSchNet`, falling back to
            :class:`StandardFlashSchNet` if its filters can't be tabulated.

    """
    if tabulate_filters and type(standard_model) is StandardSchNet:
        try:
            return TabulatedSchNet.flash_from_standard(standard_model)
        except ValueError as error:
            warn(f"{error}; using StandardFlashSchNet instead")

    _supported_conversions = {
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
    tabulate_filters: bool = True,
) -> torch.nn.Module:
    """
    Convert a standard model into its flash counterpart.
    Only maps supported classes.

    Args:
        standard_model (torch.nn.Module): The original SchNet model to be converted.
        tabulate_filters (bool): see :func:`convert_single_standard_model_to_flash`.
            The tabulated SchNet is for inference (simulations) only.

    """

    def convert(model):
        return convert_standard_model_to_flash(model, tabulate_filters)

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

    return convert_single_standard_model_to_flash(
        standard_model, tabulate_filters
    )
