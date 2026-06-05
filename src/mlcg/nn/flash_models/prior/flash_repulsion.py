import copy
import torch

from ...prior.base import _Prior
from ...prior.repulsion import Repulsion
from ...kernels.models.prior import flash_repulsion


class FlashRepulsion(_Prior):
    """
    Computes per-graph repulsion energy:
        e_ij = (sigma[type_i, type_j] / r_ij)^6
    reduced as sum over edges per graph (using mapping_batch).

    Returns: y of shape [num_graphs] (float32).
    """

    name = "repulsion"

    def __init__(
        self,
        sigma: torch.Tensor,
        name: str,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.register_buffer("sigma", sigma)
        self.eps = float(eps)
        self.name = name

    def forward(self, data) -> torch.Tensor:  # int
        # minimal checks
        pos = data.pos  # [N,3],
        atom_types = data.atom_types  # [N], int32 or int64
        index_mapping = data.neighbor_list[self.name]["index_mapping"]
        mapping_batch = data.neighbor_list[self.name]["mapping_batch"]
        num_graphs = data.ptr.numel() - 1 if hasattr(data, "ptr") else None
        # assert pos.is_cuda, "pos must be CUDA"
        assert pos.shape[-1] == 3, "pos must be [N,3]"
        assert index_mapping.shape[0] == 2, "index_mapping must be [2,E]"
        y = flash_repulsion(
            pos=pos,
            atom_types=atom_types,
            index_mapping=index_mapping,
            mapping_batch=mapping_batch,
            sigma=self.sigma,
            num_graphs=num_graphs,
            eps=self.eps,
        )
        data.out.setdefault(self.name, {})["energy"] = y
        return data

    @classmethod
    def flash_from_standard(cls, standard_model: Repulsion) -> "FlashRepulsion":
        """Class method to initialize a FlashRepulsion from a preexisting Repulsion model.

        Parameters
        ----------
        standard_model:
            A preexisting Repulsion model from which to initialize the FlashRepulsion. The
            sigma parameter will be taken from the standard_model and used to initialize
            the flash model.
        """

        if not isinstance(standard_model, Repulsion):
            raise ValueError(
                f"Expected input model of type Repulsion, but got {type(standard_model)}"
            )

        return cls(
            sigma=copy.deepcopy(standard_model.sigma),
            name=standard_model.name,
        )
