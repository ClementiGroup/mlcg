"""
Wrapper around metatrain's pure-PyTorch PET backend for use with mlcg's
AtomicData format, following the same base/Standard split used by the other
external-model wrappers in :ref:`mlcg.nn` (e.g. :class:`~mlcg.nn.mace.MACE`,
:class:`~mlcg.nn.allegro.Allegro`).

Two force modes are supported (``force_mode``):

- ``"conservative"`` (default): the backend predicts a per-atom energy that
  is summed per structure; wrap the model in
  :class:`~mlcg.nn.gradients.GradientsOut` to get forces as ``-dE/dx``.
  Training then requires a double backward, which also forces PET onto its
  unfused manual-attention path.
- ``"direct"``: a per-atom 3-vector head (the same mechanism as metatrain's
  ``non_conservative_forces`` target) predicts forces directly, used
  *without* ``GradientsOut``. No derivative w.r.t. positions is taken, so
  training needs a single backward and PET keeps its fused attention. A zero
  placeholder energy is emitted so that :class:`~mlcg.nn.SumOut` and the
  simulation code, which expect an energy key, work unchanged -- it is not a
  learned quantity.

PET is not rotation-equivariant by construction (it consumes raw Cartesian
edge vectors), so a direct force head in particular should be trained with
``rotation_augment=True``.
"""

import math
from typing import Dict, Final, List, Optional

import torch
from torch_geometric.utils import scatter

from metatrain.pet.modules.backend import PETBackend

from mlcg.data.atomic_data import AtomicData, ENERGY_KEY, FORCE_KEY
from mlcg.neighbor_list.neighbor_list import (
    atomic_data2neighbor_list,
    validate_neighborlist,
)


def _random_rotations(n: int, device, dtype) -> torch.Tensor:
    """``n`` uniformly random proper rotation matrices, shape ``(n, 3, 3)``.

    Shoemake's method (Graphics Gems III, 1992): three uniforms map to a unit
    quaternion uniform on S^3, which pushes forward to the Haar measure on
    SO(3). Same as ``mlcg.nn.et._random_rotations`` in the ET integration.
    """
    work = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    u1, u2, u3 = torch.rand(3, n, device=device, dtype=work).unbind(0)
    a, b = torch.sqrt(1.0 - u1), torch.sqrt(u1)
    t2, t3 = 2.0 * math.pi * u2, 2.0 * math.pi * u3
    w, x = a * torch.sin(t2), a * torch.cos(t2)
    y, z = b * torch.sin(t3), b * torch.cos(t3)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = torch.stack(
        [
            1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
        ],
        dim=-1,
    ).reshape(n, 3, 3)
    return R.to(dtype)


def _remove_net_force_and_torque(
    pos: torch.Tensor,
    forces: torch.Tensor,
    system_indices: torch.Tensor,
    num_systems: int,
) -> torch.Tensor:
    """Project out each structure's net force and net torque (about its
    centroid), on mlcg's flat ``(N_total, 3)`` layout.

    Same math as md-et's ``remove_net_force``/``remove_net_torque``, but works
    on ragged batches. The torque correction ``r_i x mu`` sums to zero force
    (``sum_i r_i = 0`` about the centroid), so the order of the two is free.
    """
    counts = scatter(
        torch.ones_like(pos[:, 0]), system_indices, dim=0,
        dim_size=num_systems, reduce="sum",
    ).unsqueeze(-1)
    net_force = scatter(forces, system_indices, dim=0, dim_size=num_systems, reduce="sum")
    forces = forces - (net_force / counts)[system_indices]

    com = scatter(pos, system_indices, dim=0, dim_size=num_systems, reduce="mean")
    r = pos - com[system_indices]
    tau = scatter(
        torch.linalg.cross(r, forces), system_indices, dim=0,
        dim_size=num_systems, reduce="sum",
    )
    # sum_i r_i x (r_i x mu) = (S - s I) mu, with S = sum_i r_i r_i^T, s = tr S
    S = scatter(
        r.unsqueeze(-1) * r.unsqueeze(-2), system_indices, dim=0,
        dim_size=num_systems, reduce="sum",
    )
    s = S.diagonal(dim1=-2, dim2=-1).sum(-1)
    eye = torch.eye(3, dtype=pos.dtype, device=pos.device)
    M = S - s[:, None, None] * eye
    # pinv: a collinear structure (or a single bead) makes M singular
    mu = (torch.linalg.pinv(M) @ (-tau).unsqueeze(-1)).squeeze(-1)
    return forces + torch.linalg.cross(r, mu[system_indices])


class UPET(torch.nn.Module):
    """
    Base uPET (metatrain PET backend) implementation.

    In ``"conservative"`` mode this class only predicts the energy, as with
    :class:`~mlcg.nn.mace.MACE` and :class:`~mlcg.nn.allegro.Allegro`; forces
    should be obtained by wrapping an instance in
    :class:`~mlcg.nn.gradients.GradientsOut`. In ``"direct"`` mode it
    predicts forces itself and must *not* be wrapped.

    Parameters
    ----------
    backend:
        A configured ``metatrain.pet.modules.backend.PETBackend`` instance,
        with the ``"energy"`` output (conservative mode, shape ``[1]``) or
        the ``"forces"`` output (direct mode, shape ``[3, 1]``) already
        registered via ``backend.add_output(...)``.
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
    force_mode:
        ``"conservative"`` or ``"direct"``, see the module docstring.
    rotation_augment:
        During training, evaluate the model on a randomly rotated copy of each
        structure (one rotation per structure), equivalent to rotating inputs
        and targets together. Direct mode rotates the predicted forces back
        explicitly; conservative mode needs no back-rotation, since
        ``GradientsOut`` differentiates w.r.t. the unrotated positions.
        Ignored in eval mode. Default False keeps older configs unchanged.
    filter_forces:
        Direct mode only. Project out each structure's net force and net
        torque. Physically exact for an isolated molecule with no periodic box.
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
        force_mode: str = "conservative",
        rotation_augment: bool = False,
        filter_forces: bool = False,
    ):
        super().__init__()
        if force_mode not in ("conservative", "direct"):
            raise ValueError(
                f"force_mode must be 'conservative' or 'direct', got {force_mode}"
            )
        if filter_forces and force_mode != "direct":
            raise ValueError("filter_forces only applies to force_mode='direct'")
        self.backend = backend
        self.atomic_types = atomic_types
        self.r_max = r_max
        self.cutoff_width_adaptive = cutoff_width_adaptive
        self.max_num_neighbors = max_num_neighbors
        self.nls_distance_method = nls_distance_method
        self.force_mode = force_mode
        self.rotation_augment = rotation_augment
        self.filter_forces = filter_forces
        # Read by mlcg.pl.PLModel: direct forces need no autograd w.r.t.
        # positions, so validation can run under no_grad.
        self.derivative = force_mode == "conservative"

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
            The input data object with the predicted energy (conservative
            mode) or forces plus a zero placeholder energy (direct mode)
            added under ``data.out[self.name]``.
        """
        direct = self.force_mode == "direct"
        # Direct forces don't differentiate w.r.t. positions; detaching keeps
        # uPET out of any autograd graph (e.g. when a GradientsOut prior in the
        # same SumOut has set pos.requires_grad during simulation), which also
        # lets PET use its fused attention kernel.
        pos = data.pos.detach() if direct else data.pos
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

        # Rotation augmentation: predict in a randomly rotated frame. The
        # neighbor list only depends on distances, so it carries over; the
        # cell rotates with the positions so periodic edge vectors stay
        # consistent. In conservative mode the rotation stays in the autograd
        # graph, so GradientsOut's -dE/dx is already in the original frame.
        rotations = None
        model_pos = pos
        if self.rotation_augment and self.training:
            rotations = _random_rotations(num_systems, pos.device, pos.dtype)
            # x' = R x, row-vector form x' = x R^T
            model_pos = torch.bmm(
                pos.unsqueeze(1), rotations.transpose(1, 2)[system_indices]
            ).squeeze(1)
            cells = cells @ rotations.transpose(1, 2)

        batch_data = self.backend.preprocess(
            model_pos,
            centers,
            neighbors,
            species,
            cells,
            cell_shifts,
            system_indices,
            cutoff_width_adaptive=self.cutoff_width_adaptive,
        )
        node_feats, edge_feats = self.backend.calculate_features(batch_data)
        output_name = "forces" if direct else "energy"
        atomic_predictions, _, _ = self.backend.predict(
            node_feats,
            edge_feats,
            batch_data,
            cells,
            system_indices,
            requested_output_names=[output_name],
        )

        if direct:
            forces = atomic_predictions["forces"][0].reshape(-1, 3).to(pos.dtype)
            if rotations is not None:
                # F = R^T F', row-vector form F = F' R
                forces = torch.bmm(
                    forces.unsqueeze(1), rotations[system_indices]
                ).squeeze(1)
            if self.filter_forces:
                # In the original frame, matching pos.
                forces = _remove_net_force_and_torque(
                    pos, forces, system_indices, num_systems
                )
            data.out[self.name] = {
                FORCE_KEY: forces,
                # Placeholder so SumOut / the simulation code find an energy.
                # Not learned: a simulated "potential" is prior-only.
                ENERGY_KEY: torch.zeros(
                    num_systems, dtype=pos.dtype, device=pos.device
                ),
            }
            return data

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
    force_mode, rotation_augment, filter_forces:
        See :class:`UPET`. ``force_mode`` picks which head is registered on
        the backend (``"energy"`` or ``"forces"``); the trunk is the same, so
        a direct-mode checkpoint's trunk weights can initialize a
        conservative model.
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
        force_mode: str = "conservative",
        rotation_augment: bool = False,
        filter_forces: bool = False,
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
        if force_mode == "direct":
            # per-atom Cartesian 3-vector, as metatrain's non_conservative_forces
            backend.add_output("forces", {"forces": [3, 1]})
        else:
            backend.add_output("energy", {"energy": [1]})

        super().__init__(
            backend=backend,
            atomic_types=atomic_types,
            r_max=cutoff,
            cutoff_width_adaptive=cutoff_width_adaptive,
            max_num_neighbors=max_num_neighbors,
            nls_distance_method=nls_distance_method,
            force_mode=force_mode,
            rotation_augment=rotation_augment,
            filter_forces=filter_forces,
        )
