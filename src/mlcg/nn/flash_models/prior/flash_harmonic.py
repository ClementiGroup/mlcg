import copy
import torch
from typing import Union

from ...prior.base import _Prior
from ...prior.harmonic import HarmonicBonds, HarmonicAngles, GeneralBonds, GeneralAngles
from ...kernels.models.prior import flash_harmonic_bonds, flash_harmonic_angles


class FlashHarmonicBonds(_Prior):
    """
    Compute fused harmonic bonds energy.

    Attention!
    ----------
    At the moment only non periodic systems are supported
    """

    name = "bonds"

    def __init__(
        self,
        k: torch.Tensor,
        x_0: torch.Tensor,
        name: str,
    ):
        super().__init__()
        self.register_buffer("k", k)
        self.register_buffer("x_0", x_0)
        self.name = name

    def forward(self, data) -> torch.Tensor:  # int
        # minimal checks
        pos = data.pos  # [N,3],
        atom_types = data.atom_types  # [N], int32 or int64
        index_mapping = data.neighbor_list[self.name]["index_mapping"]
        mapping_batch = data.neighbor_list[self.name]["mapping_batch"]
        num_graphs = data.ptr.numel() - 1 if hasattr(data, "ptr") else None
        assert pos.shape[-1] == 3, "pos must be [N,3]"
        assert index_mapping.shape[0] == 2, "index_mapping must be [2,E]"
        y = flash_harmonic_bonds(
            pos=pos,
            atom_types=atom_types,
            index_mapping=index_mapping,
            mapping_batch=mapping_batch,
            k=self.k,
            x_0=self.x_0,
            num_graphs=num_graphs,
        )
        data.out.setdefault(self.name, {}).update({"energy": y})
        return data

    @classmethod
    def flash_from_standard(
        cls, standard_model: Union[HarmonicBonds, GeneralBonds]
    ) -> "FlashHarmonicBonds":
        """Class method to initialize a FlashHarmonicBonds from a preexisting
        HarmonicBonds or GeneralBonds model.

        Parameters
        ----------
        standard_model:
            A preexisting HarmonicBonds model from which to initialize the FlashHarmonicBonds.
        """

        if not (
            isinstance(standard_model, HarmonicBonds)
            or isinstance(standard_model, GeneralBonds)
        ):
            raise ValueError(
                f"Expected input model of type HarmonicBonds, but got {type(standard_model)}"
            )

        return cls(
            k=copy.deepcopy(standard_model.k),
            x_0=copy.deepcopy(
                standard_model.x_0
                if hasattr(standard_model, "x_0")
                else standard_model.x0
            ),
            name=standard_model.name,
        )


class FlashHarmonicAngles(_Prior):
    """
    ompute fused harmonic angles energy.

    Attention!
    ----------
    At the moment only non periodic systems are supported
    """

    name = "angles"

    def __init__(
        self,
        k: torch.Tensor,
        x_0: torch.Tensor,
        name: str,
    ):
        super().__init__()
        self.register_buffer("k", k)
        self.register_buffer("x_0", x_0)
        self.name = name

    def forward(self, data) -> torch.Tensor:  # int
        # minimal checks
        pos = data.pos  # [N,3],
        atom_types = data.atom_types  # [N], int32 or int64
        index_mapping = data.neighbor_list[self.name]["index_mapping"]
        mapping_batch = data.neighbor_list[self.name]["mapping_batch"]
        num_graphs = data.ptr.numel() - 1 if hasattr(data, "ptr") else None
        assert pos.shape[-1] == 3, "pos must be [N,3]"
        assert index_mapping.shape[0] == 3, "index_mapping must be [3,E]"
        y = flash_harmonic_angles(
            pos=pos,
            atom_types=atom_types,
            index_mapping=index_mapping,
            mapping_batch=mapping_batch,
            k=self.k,
            x_0=self.x_0,
            num_graphs=num_graphs,
        )
        data.out.setdefault(self.name, {}).update({"energy": y})
        return data

    @classmethod
    def flash_from_standard(
        cls, standard_model: Union[HarmonicAngles, GeneralAngles]
    ) -> "FlashHarmonicAngles":
        """Class method to initialize a FlashHarmonicAngles from a preexisting
        HarmonicAngles  or GeneralAngles model.

        Parameters
        ----------
        standard_model:
            A preexisting HarmonicAngles model from which to initialize the FlashHarmonicAngles.
        """

        if not (
            isinstance(standard_model, HarmonicAngles)
            or isinstance(standard_model, GeneralAngles)
        ):
            raise ValueError(
                f"Expected input model of type HarmonicAngles, but got {type(standard_model)}"
            )

        return cls(
            k=copy.deepcopy(standard_model.k),
            x_0=copy.deepcopy(
                standard_model.x_0
                if hasattr(standard_model, "x_0")
                else standard_model.x0
            ),
            name=standard_model.name,
        )
