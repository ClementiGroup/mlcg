"""
Wrapper around metatrain's pure-PyTorch PET backend for use with mlcg's
AtomicData format, following the same base/Standard split used by the other
external-model wrappers in :ref:`mlcg.nn` (e.g. :class:`~mlcg.nn.mace.MACE`,
:class:`~mlcg.nn.allegro.Allegro`).
"""

from typing import Dict, Final, List, Optional

import torch
from torch_geometric.utils import scatter

from metatrain.pet.modules.backend import PETBackend

from mlcg.data.atomic_data import AtomicData, ENERGY_KEY, FORCE_KEY
from mlcg.neighbor_list.neighbor_list import (
    atomic_data2neighbor_list,
    validate_neighborlist,
)


class UPET(torch.nn.Module):
    """
    Base uPET (metatrain PET backend) implementation for energy prediction.

    As with :class:`~mlcg.nn.mace.MACE` and :class:`~mlcg.nn.allegro.Allegro`,
    this class only predicts the energy; forces should be obtained by
    wrapping an instance in :class:`~mlcg.nn.gradients.GradientsOut`.

    Parameters
    ----------
    backend:
        A configured ``metatrain.pet.modules.backend.PETBackend`` instance,
        with the ``"energy"`` output already registered via
        ``backend.add_output(...)``.
    atomic_types:
        Sorted list of species (CG bead types / fake atomic numbers) the
        model supports.
    r_max:
        Cutoff radius used to build the neighbor list, matching the
        ``cutoff`` the backend was configured with.
    cutoff_width_adaptive:
        Width of the smooth cutoff taper for the adaptive cutoff scheme.
    max_num_neighbors:
        Passed through to mlcg's neighbor list builder.
    nls_distance_method:
        Method for computing a neighbor list. Supported values are
        `torch`, `nvalchemi_naive`, `nvalchemi_cell`, `nvalchemi_raw`
        and `custom_kernel`.
    """

    name: Final[str] = "upet"

    def __init__(
        self,
        backend: PETBackend,
        atomic_types: List[int],
        r_max: float,
        cutoff_width_adaptive: float = 1.0,
        max_num_neighbors: int = 1000,
        nls_distance_method: str = "torch",
    ):
        super().__init__()
        self.backend = backend
        self.atomic_types = atomic_types
        self.r_max = r_max
        self.cutoff_width_adaptive = cutoff_width_adaptive
        self.max_num_neighbors = max_num_neighbors
        self.nls_distance_method = nls_distance_method

    def forward(self, data: AtomicData) -> AtomicData:
        """
        Forward pass of the uPET model.

        Parameters
        ----------
        data:
            Input atomic data containing positions, atom types, and batch
            information.

        Returns
        -------
        data:
            The input data object with the predicted energy added under
            ``data.out[self.name]``.
        """
        pos = data.pos
        species = data.atom_types

        if "batch" in data:
            system_indices = data.batch
        else:
            system_indices = torch.zeros(
                pos.shape[0], dtype=torch.long, device=pos.device
            )
        num_systems = int(system_indices.max().item()) + 1

        if "cell" in data and data.cell is not None:
            cells = data.cell.to(pos.dtype)
        else:
            cells = torch.zeros(
                num_systems, 3, 3, dtype=pos.dtype, device=pos.device
            )

        neighbor_list = data.neighbor_list.get(self.name)
        if not self.is_nl_compatible(neighbor_list):
            neighbor_list = self.neighbor_list(
                data, self.r_max, self.max_num_neighbors
            )[self.name]

        centers, neighbors = (
            neighbor_list["index_mapping"][0],
            neighbor_list["index_mapping"][1],
        )
        cell_shifts = neighbor_list["cell_shifts"].long()

        batch_data = self.backend.preprocess(
            pos,
            centers,
            neighbors,
            species,
            cells,
            cell_shifts,
            system_indices,
            cutoff_width_adaptive=self.cutoff_width_adaptive,
        )
        node_feats, edge_feats = self.backend.calculate_features(batch_data)
        atomic_predictions, _, _ = self.backend.predict(
            node_feats,
            edge_feats,
            batch_data,
            cells,
            system_indices,
            requested_output_names=["energy"],
        )
        per_atom_energy = atomic_predictions["energy"][0].squeeze(-1)
        total_energy = scatter(
            per_atom_energy,
            system_indices,
            dim=0,
            dim_size=num_systems,
            reduce="sum",
        )

        data.out[self.name] = {ENERGY_KEY: total_energy}
        return data

    def is_nl_compatible(self, nl):
        """
        Check if a neighbor list is compatible with this model.

        Parameters
        ----------
        nl:
            The neighbor list to check.

        Returns
        -------
        bool:
            True if the neighbor list is compatible, False otherwise.
        """
        is_compatible = False
        if validate_neighborlist(nl):
            if (
                nl["order"] == 2
                and nl["self_interaction"] is False
                and nl["rcut"] == self.r_max
            ):
                is_compatible = True
        return is_compatible

    def neighbor_list(
        self,
        data: AtomicData,
        rcut: float,
        max_num_neighbors: int = 1000,
    ) -> dict:
        """Computes the neighborlist for :obj:`data` using a strict cutoff of :obj:`rcut`."""
        if not hasattr(self, "nls_distance_method"):
            self.nls_distance_method = "torch"
        return {
            UPET.name: atomic_data2neighbor_list(
                data,
                rcut,
                self_interaction=False,
                max_num_neighbors=max_num_neighbors,
                nls_distance_method=self.nls_distance_method,
            )
        }


class StandardUPET(UPET):
    """
    Standard implementation of the uPET model with configurable parameters.

    Builds a ``metatrain.pet.modules.backend.PETBackend`` from flat
    hyperparameters (see ``metatrain.pet.documentation.ModelHypers``) rather
    than requiring a pre-built backend, mirroring
    :class:`~mlcg.nn.allegro.StandardAllegro`/
    :class:`~mlcg.nn.mace.StandardMACE`.

    Parameters
    ----------
    atomic_types:
        Sorted list of species (CG bead types / fake atomic numbers) the
        model supports.
    cutoff:
        Cutoff radius used both by the PET backend and mlcg's neighbor list
        builder.
    cutoff_function:
        Cutoff taper shape, one of ``"Cosine"``/``"Bump"``.
    cutoff_width:
        Width of the cutoff taper.
    num_neighbors_adaptive:
        Target average number of neighbors for the adaptive cutoff scheme,
        or ``None`` to disable it.
    adaptive_cutoff_method:
        One of ``"grid"``/``"solver"``.
    d_pet:
        PET internal feature dimension.
    d_node:
        Node feature dimension.
    d_head:
        Attention head dimension.
    d_feedforward:
        Feedforward layer dimension.
    num_heads:
        Number of attention heads.
    num_attention_layers:
        Number of attention layers per GNN layer.
    num_gnn_layers:
        Number of message-passing layers.
    normalization:
        One of ``"RMSNorm"``/``"LayerNorm"``.
    activation:
        One of ``"SiLU"``/``"SwiGLU"``.
    attention_temperature:
        Softmax temperature used in attention.
    transformer_type:
        One of ``"PreLN"``/``"PostLN"``.
    featurizer_type:
        One of ``"residual"``/``"feedforward"``.
    system_conditioning:
        Whether to condition on system-level (e.g. charge/spin) inputs.
    long_range:
        Dict of long-range/Ewald settings, see
        ``metatrain.pet.documentation.ModelHypers``.
    cutoff_width_adaptive:
        Width of the smooth cutoff taper for the adaptive cutoff scheme.
    max_num_neighbors:
        Passed through to mlcg's neighbor list builder.
    nls_distance_method:
        Method for computing a neighbor list. Supported values are
        `torch`, `nvalchemi_naive`, `nvalchemi_cell`, `nvalchemi_raw`
        and `custom_kernel`.
    """

    def __init__(
        self,
        atomic_types: List[int],
        cutoff: float = 7.5,
        cutoff_function: str = "Bump",
        cutoff_width: float = 0.5,
        num_neighbors_adaptive: Optional[float] = None,
        adaptive_cutoff_method: str = "solver",
        d_pet: int = 128,
        d_node: int = 256,
        d_head: int = 128,
        d_feedforward: int = 256,
        num_heads: int = 8,
        num_attention_layers: int = 2,
        num_gnn_layers: int = 2,
        normalization: str = "RMSNorm",
        activation: str = "SwiGLU",
        attention_temperature: float = 1.0,
        transformer_type: str = "PreLN",
        featurizer_type: str = "feedforward",
        system_conditioning: bool = False,
        long_range: Optional[Dict] = None,
        cutoff_width_adaptive: float = 1.0,
        max_num_neighbors: int = 1000,
        nls_distance_method: str = "torch",
    ):
        if long_range is None:
            long_range = dict(
                enable=False,
                use_ewald=False,
                smearing=1.4,
                kspace_resolution=1.33,
                interpolation_nodes=5,
            )

        hypers = dict(
            cutoff=cutoff,
            cutoff_function=cutoff_function,
            cutoff_width=cutoff_width,
            num_neighbors_adaptive=num_neighbors_adaptive,
            adaptive_cutoff_method=adaptive_cutoff_method,
            d_pet=d_pet,
            d_node=d_node,
            d_head=d_head,
            d_feedforward=d_feedforward,
            num_heads=num_heads,
            num_attention_layers=num_attention_layers,
            num_gnn_layers=num_gnn_layers,
            normalization=normalization,
            activation=activation,
            attention_temperature=attention_temperature,
            transformer_type=transformer_type,
            featurizer_type=featurizer_type,
            system_conditioning=system_conditioning,
            long_range=long_range,
        )

        backend = PETBackend(hypers, atomic_types)
        backend.add_output("energy", {"energy": [1]})

        super().__init__(
            backend=backend,
            atomic_types=atomic_types,
            r_max=cutoff,
            cutoff_width_adaptive=cutoff_width_adaptive,
            max_num_neighbors=max_num_neighbors,
            nls_distance_method=nls_distance_method,
        )
