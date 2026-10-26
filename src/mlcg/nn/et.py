"""
Wrapper around md-et's ``PairEncoder`` (edge-transformer) model for use with
mlcg's AtomicData format, following the same base/Standard split used by the
other external-model wrappers in :ref:`mlcg.nn` (e.g.
:class:`~mlcg.nn.mace.MACE`, :class:`~mlcg.nn.allegro.Allegro`,
:class:`~mlcg.nn.upet.UPET`).

Unlike those wrappers, ``PairEncoder`` does dense all-pairs attention over a
padded batch (no radius cutoff, no neighbor list) -- so this module converts
mlcg's flat/scatter batch layout to md-et's ``(batch, max_atoms, ...)``
padded layout via :func:`torch_geometric.utils.to_dense_batch` instead of
building a neighbor list.
"""

from typing import Final, List

import torch
from torch_geometric.utils import to_dense_batch

from md_et.nn.pair_encoder import PairEncoder
from md_et.nn.types import Property as Props

from mlcg.data.atomic_data import AtomicData, ENERGY_KEY


class ET(torch.nn.Module):
    """
    Base edge-transformer (md-et ``PairEncoder``) implementation for energy
    prediction.

    As with :class:`~mlcg.nn.mace.MACE`, :class:`~mlcg.nn.allegro.Allegro`
    and :class:`~mlcg.nn.upet.UPET`, this class only predicts the energy;
    forces should be obtained by wrapping an instance in
    :class:`~mlcg.nn.gradients.GradientsOut`.

    Parameters
    ----------
    encoder:
        A configured ``md_et.nn.pair_encoder.PairEncoder`` instance, whose
        ``target_heads`` must be exactly ``["formation_energy"]`` with
        ``energy_conserving=False`` -- this wrapper only reads
        ``Props.formation_energy`` from its output and relies on
        :class:`~mlcg.nn.gradients.GradientsOut` for forces, so any other
        head configuration is not supported here.
    atomic_types:
        Sorted list of species (CG bead types / fake atomic numbers) the
        model supports. Not consumed by ``PairEncoder`` itself (species are
        used directly as embedding indices), kept only for interface parity
        with the other :ref:`mlcg.nn` wrappers.
    """

    name: Final[str] = "et"

    def __init__(
        self,
        encoder: PairEncoder,
        atomic_types: List[int],
    ):
        super().__init__()
        self.encoder = encoder
        self.atomic_types = atomic_types

    def forward(self, data: AtomicData) -> AtomicData:
        """
        Forward pass of the ET model.

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
        species = data.atom_types.long()

        if "batch" in data:
            system_indices = data.batch
        else:
            system_indices = torch.zeros(
                pos.shape[0], dtype=torch.long, device=pos.device
            )
        num_systems = data.n_atoms.size(0)

        positions_padded, mask = to_dense_batch(
            pos, system_indices, batch_size=num_systems
        )
        atomic_numbers_padded, _ = to_dense_batch(
            species, system_indices, batch_size=num_systems
        )

        charge = torch.zeros(
            num_systems, 1, dtype=torch.long, device=pos.device
        )
        multiplicity = torch.ones(
            num_systems, 1, dtype=torch.long, device=pos.device
        )

        inputs = {
            Props.positions: positions_padded,
            Props.atomic_numbers: atomic_numbers_padded,
            Props.mask: mask,
            Props.charge: charge,
            Props.multiplicity: multiplicity,
        }

        outputs = self.encoder(inputs)
        total_energy = outputs[Props.formation_energy].squeeze(-1).to(pos.dtype)

        data.out[self.name] = {ENERGY_KEY: total_energy}
        return data


class StandardET(ET):
    """
    Standard implementation of the ET model with configurable parameters.

    Builds a ``md_et.nn.pair_encoder.PairEncoder`` from flat hyperparameters,
    mirroring :class:`~mlcg.nn.allegro.StandardAllegro`/
    :class:`~mlcg.nn.mace.StandardMACE`/:class:`~mlcg.nn.upet.StandardUPET`.
    ``target_heads``/``energy_conserving`` are fixed internally (energy-only
    output, forces via :class:`~mlcg.nn.gradients.GradientsOut`) rather than
    exposed as knobs, matching the contract described on :class:`ET`.

    Parameters
    ----------
    atomic_types:
        Sorted list of species (CG bead types / fake atomic numbers) the
        model supports. See the note on :class:`ET` -- not consumed by
        ``PairEncoder`` itself, kept for interface parity.
    n_layers:
        Number of edge-transformer layers.
    embd_dim:
        Embedding / hidden dimension.
    num_3d_kernels:
        Number of radial-basis kernels used by the pairwise distance
        embedding (``distance_embed_type``).
    num_heads:
        Number of attention heads.
    ffn_multiplier:
        Width multiplier for the feed-forward blocks.
    attention_dropout:
        Dropout applied inside attention.
    ffn_dropout:
        Dropout applied inside the feed-forward blocks.
    head_dropout:
        Dropout applied inside the output (energy) head.
    head_project_down:
        Whether the output head progressively projects its hidden dimension
        down (``embd_dim`` -> ``embd_dim/2`` -> ``embd_dim/4``) before the
        final linear layer.
    distance_embed_type:
        One of ``"gaussian"``/``"bessel"``.
    embed_edge_types:
        Whether pairwise distance embeddings are conditioned on the
        (ordered) pair of species, rather than being species-agnostic.

    Notes
    -----
    ``PairEncoder`` also accepts (but, verified by reading its source,
    silently ignores) ``cls_token``, ``activation``, ``norm_first``,
    ``norm``, ``decomposer_type``, ``compose_dipole_from_charges``,
    ``use_electronic_embeddings`` and ``directional_embed_type`` -- none of
    these are wired to anything in the current ``md-et`` release, so they
    are intentionally not exposed here.
    """

    def __init__(
        self,
        atomic_types: List[int],
        n_layers: int = 12,
        embd_dim: int = 192,
        num_3d_kernels: int = 32,
        num_heads: int = 8,
        ffn_multiplier: int = 2,
        attention_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        head_dropout: float = 0.0,
        head_project_down: bool = True,
        distance_embed_type: str = "gaussian",
        embed_edge_types: bool = False,
    ):
        encoder = PairEncoder(
            n_layers=n_layers,
            embd_dim=embd_dim,
            num_3d_kernels=num_3d_kernels,
            cls_token=False,
            num_heads=num_heads,
            activation="gelu",
            ffn_multiplier=ffn_multiplier,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            head_dropout=head_dropout,
            norm_first=False,
            norm="layer",
            decomposer_type="pooling",
            target_heads=["formation_energy"],
            head_project_down=head_project_down,
            energy_conserving=False,
            distance_embed_type=distance_embed_type,
            embed_edge_types=embed_edge_types,
        )

        super().__init__(encoder=encoder, atomic_types=atomic_types)
