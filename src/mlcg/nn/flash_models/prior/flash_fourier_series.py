import copy
import torch
from typing import Final
from ...prior.base import _Prior
from ...prior.fourier_series import FourierSeries, Dihedral
from ...kernels.models.prior import flash_dihedral


class FlashDihedral(_Prior):
    """
    Computes per-graph dihedral energy:
        e_ijk = k[type_i, type_j, type_k]* (cos(theta_ijk) - cos0_ijk)^6
    reduced as sum over edges per graph (using mapping_batch).

    Returns: y of shape [num_graphs] (float32).
    """

    name = "dihedral"
    _order: Final[int] = 4

    def __init__(
        self,
        k1s: torch.Tensor,
        k2s: torch.Tensor,
        v_0: torch.Tensor,
        n_degs: int,
        name: str,
    ):
        super().__init__()
        self.register_buffer("k1s", k1s)
        self.register_buffer("k2s", k2s)
        self.register_buffer("v_0", v_0)
        self.n_degs = n_degs
        self.name = name

    def forward(self, data) -> torch.Tensor:  # int
        # minimal checks
        pos = data.pos  # [N,3],
        atom_types = data.atom_types  # [N], int32 or int64
        index_mapping = data.neighbor_list[self.name]["index_mapping"]
        mapping_batch = data.neighbor_list[self.name]["mapping_batch"]
        num_graphs = data.ptr.numel() - 1 if hasattr(data, "ptr") else None
        assert pos.shape[-1] == 3, "pos must be [N,3]"
        assert index_mapping.shape[0] == 4, "index_mapping must be [4,E]"
        y = flash_dihedral(
            pos=pos,
            atom_types=atom_types,
            index_mapping=index_mapping,
            mapping_batch=mapping_batch,
            k1=self.k1s,
            k2=self.k2s,
            v_0=self.v_0,
            deg=self.n_degs,
            num_graphs=num_graphs,
        )
        # data.out[self.name] = {"energy": y}
        data.out.setdefault(self.name, {})["energy"] = y
        return data

    @classmethod
    def flash_from_standard(
        cls, standard_model: FourierSeries
    ) -> "FlashDihedral":
        """Class method to initialize a FlashDihedral from a preexisting Dihedral model.

        Parameters
        ----------
        standard_model:
            A preexisting Dihedral model from which to initialize the FlashDihedral.
        """

        if not isinstance(standard_model, Dihedral):
            raise ValueError(
                f"Expected input model of type Dihedral, but got {type(standard_model)}"
            )

        return cls(
            k1s=copy.deepcopy(standard_model.k1s),
            k2s=copy.deepcopy(standard_model.k2s),
            v_0=copy.deepcopy(standard_model.v_0),
            n_degs=standard_model.n_degs,
            name=standard_model.name,
        )
