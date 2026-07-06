from typing import Any, Callable, Dict, List, Optional, Union, Final, Tuple

import torch
from e3nn import nn, o3
from e3nn.util.jit import compile_mode

try:
    from mace.modules.radial import ZBLBasis
    from mace.tools.scatter import scatter_sum
    from mace.tools import to_one_hot

    from mace.modules.blocks import (
        EquivariantProductBasisBlock,
        LinearNodeEmbeddingBlock,
        LinearReadoutBlock,
        NonLinearReadoutBlock,
        RadialEmbeddingBlock,
        InteractionBlock,
    )
    from mace.modules.utils import get_edge_vectors_and_lengths
    from mace.modules.wrapper_ops import (
        Linear,
        TensorProduct,
        FullyConnectedTensorProduct,
    )
    from mace.modules.irreps_tools import (
        reshape_irreps,
        tp_out_irreps_with_instructions,
    )

except ImportError as e:
    print(e)
    print(
        "Please install or set mace to your path before using this interface. "
        + "To install you can either run 'pip install git+https://github.com/ACEsuit/mace.git@v0.3.13', "
        + "or clone the repository and add it to your PYTHONPATH."
        ""
    )

try:
    import cuequivariance as cue
    import cuequivariance_torch as cuet

    CUET_AVAILABLE = True
except ImportError:
    CUET_AVAILABLE = False
    print(
        "cuEquivariance is not installed. cuEquivariance features will be disabled. It is recommended to install cuEquivariance for better performance. "
        + "To install cuEquivariance run pip install cuequivariance cuequivariance-torch cuequivariance-ops-torch-cu12 "
        + 'Replace "cu12" with "cu11" if you are using CUDA 11.'
    )

if CUET_AVAILABLE:
    from mace.modules.wrapper_ops import CuEquivarianceConfig

# from ..pl.model import get_class_from_str
from ..data.atomic_data import AtomicData, ENERGY_KEY
from ..neighbor_list.neighbor_list import (
    atomic_data2neighbor_list,
    validate_neighborlist,
)

from e3nn.util.jit import compile_mode


# This is a copy of the residual RealAgnosticResidualInteractionBlock as it is in
# mace v0.3.13
# https://github.com/ACEsuit/mace/blob/b5faaa076c49778fc17493edfecebcabeb960155/mace/modules/blocks.py#L474
@compile_mode("script")
class CustomRealAgnosticResidualInteractionBlock(InteractionBlock):
    r"""version of the mace.modules.blocks RealAgnosticResidualInteractionBlock
    without a hardcoded tanh gate.

    We avoid doing a general AgnosticResidualInteractionBlock as it would require
    a larger rewritting of our MACE implementation
    """

    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None
        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.tanh,  # gate
        )

        # Linear
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.node_feats_irreps,
            self.node_attrs_irreps,
            self.hidden_irreps,
            cueq_config=self.cueq_config,
        )
        self.reshape = reshape_irreps(
            self.irreps_out, cueq_config=self.cueq_config
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        lammps_class: Optional[Any] = None,
        lammps_natoms: Tuple[int, int] = (0, 0),
        first_layer: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        n_real = lammps_natoms[0] if lammps_class is not None else None
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        node_feats = self.handle_lammps(
            node_feats,
            lammps_class=lammps_class,
            lammps_natoms=lammps_natoms,
            first_layer=first_layer,
        )
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.truncate_ghosts(message, n_real)
        node_attrs = self.truncate_ghosts(node_attrs, n_real)
        sc = self.truncate_ghosts(sc, n_real)
        message = self.linear(message) / self.avg_num_neighbors
        return (
            self.reshape(message),
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2]


@compile_mode("script")
class MACE(torch.nn.Module):
    """
    Implementation of MACE neural network model from https://github.com/ACEsuit/mace

    Args:
        atomic_numbers (torch.Tensor):
            Tensor of atomic numbers present in the system.
        node_embedding (torch.nn.Module):
            Module for embedding node (atom) attributes.
        radial_embedding (torch.nn.Module):
            Module for embedding radial (distance) features.
        spherical_harmonics (torch.nn.Module):
            Module for computing spherical harmonics of edge vectors.
        interactions (List[torch.nn.Module]):
            List of interaction blocks.
        products (List[torch.nn.Module]):
            List of product basis blocks.
        readouts (List[torch.nn.Module]):
            List of readout blocks.
        r_max (float):
            Cutoff radius for neighbor list.
        max_num_neighbors (int):
            Maximum number of neighbors per atom.
        pair_repulsion_fn (torch.nn.Module, optional):
            Optional pairwise repulsion energy function.
        nls_distance_method:
        Method for computing a neighbor list. Supported values are
        `torch`, `nvalchemi_naive`, `nvalchemi_cell`, `nvalchemi_raw`
        and `custom_kernel`.
    """

    name: Final[str] = "mace"

    def __init__(
        self,
        atomic_numbers: torch.Tensor,
        node_embedding: torch.nn.Module,
        radial_embedding: torch.nn.Module,
        spherical_harmonics: torch.nn.Module,
        interactions: List[torch.nn.Module],
        products: List[torch.nn.Module],
        readouts: List[torch.nn.Module],
        r_max: float,
        max_num_neighbors: int,
        pair_repulsion_fn: torch.nn.Module = None,
        nls_distance_method: str = "torch",
    ):
        super().__init__()

        self.register_buffer("atomic_numbers", atomic_numbers)
        self.node_embedding = node_embedding
        self.radial_embedding = radial_embedding
        self.spherical_harmonics = spherical_harmonics
        self.interactions = torch.nn.ModuleList(interactions)
        self.products = torch.nn.ModuleList(products)
        self.readouts = torch.nn.ModuleList(readouts)
        self.r_max = r_max
        self.max_num_neighbors = max_num_neighbors
        self.pair_repulsion_fn = pair_repulsion_fn
        self.nls_distance_method = nls_distance_method

        self.register_buffer(
            "types_mapping",
            -1 * torch.ones(atomic_numbers.max() + 1, dtype=torch.long),
        )
        self.types_mapping[atomic_numbers] = torch.arange(
            atomic_numbers.shape[0]
        )

    def get_node_feats_and_attrs(
        self, data: AtomicData
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the initial node features and the node attribute tags.

        Returns a tuple ``(node_feats, node_attrs)`` where:

        * ``node_feats`` is the initial per-node hidden representation that is
          propagated and *updated* through the interaction/product blocks (the
          MACE analogue of SchNet's ``x``).
        * ``node_attrs`` is the *fixed* per-node identity tag, a one-hot over
          ``atomic_numbers`` of shape ``(n_nodes, num_elements)``, re-injected
          unchanged at every layer to select the element-indexed weights of the
          product and interaction blocks.

        Subclasses override this to source the initial features from something
        other than a learned type embedding (e.g. a frozen per-type table or
        precomputed per-bead embeddings); see
        :ref:`mlcg.nn.FrozenTypEmbeddingMACE` and
        :ref:`mlcg.nn.FrozenResEmbeddingMACE`.
        """
        types_ids = self.types_mapping[data.atom_types].view(-1, 1)
        node_attrs = to_one_hot(types_ids, self.atomic_numbers.shape[0])
        node_feats = self.node_embedding(node_attrs)
        return node_feats, node_attrs

    def forward(self, data: AtomicData) -> AtomicData:
        """
        Forward pass of the MACE model.

        Args:
            data (AtomicData):
                Input atomic data object.

        Returns:
            AtomicData:
                Output data with predicted energies in `data.out`.
        """
        # Setup
        num_atoms_arange = torch.arange(data.pos.shape[0])
        num_graphs = data.ptr.numel() - 1  # data.batch.max()
        node_heads = torch.zeros_like(data.batch)

        # Embeddings (see get_node_feats_and_attrs; overridable by subclasses)
        node_feats, node_attrs = self.get_node_feats_and_attrs(data)

        neighbor_list = data.neighbor_list.get(self.name)

        if not self.is_nl_compatible(neighbor_list):
            neighbor_list = self.neighbor_list(
                data, self.r_max, self.max_num_neighbors
            )[self.name]

        edge_index = neighbor_list["index_mapping"]

        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data.pos,
            edge_index=edge_index,
            shifts=neighbor_list["cell_shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(
            lengths, node_attrs, edge_index, self.atomic_numbers
        )

        if self.pair_repulsion_fn:
            pair_node_energy = self.pair_repulsion_fn(
                lengths, node_attrs, edge_index, self.atomic_numbers
            )
            pair_energy = scatter_sum(
                src=pair_node_energy,
                index=data["batch"],
                dim=-1,
                dim_size=num_graphs,
            )  # [n_graphs,]
        else:
            pair_energy = torch.zeros(
                data.batch.max() + 1,
                device=data.pos.device,
                dtype=data.pos.dtype,
            )

        # Interactions
        energies = [pair_energy]
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
            )
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=node_attrs
            )
            node_energies = readout(node_feats, node_heads)[
                num_atoms_arange, node_heads
            ]  # [n_nodes, len(heads)]
            energy = scatter_sum(
                src=node_energies,
                index=data["batch"],
                dim=0,
                dim_size=num_graphs,
            )  # [n_graphs,]
            energies.append(energy)

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]

        data.out[self.name] = {ENERGY_KEY: total_energy}

        return data

    def is_nl_compatible(self, nl):
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
            MACE.name: atomic_data2neighbor_list(
                data,
                rcut,
                self_interaction=False,
                max_num_neighbors=max_num_neighbors,
                nls_distance_method=self.nls_distance_method,
            )
        }


@compile_mode("script")
class StandardMACE(MACE):
    """
    Standard configuration of the MACE model.

    This class provides a convenient interface for constructing a MACE model
    with typical settings and block choices, including embedding, interaction,
    and readout modules.

    Args:
        r_max (float):
            Cutoff radius for neighbor list.
        num_bessel (int):
            Number of Bessel functions for radial basis.
        num_polynomial_cutoff (int):
            Number of polynomial cutoff functions.
        max_ell (int):
            Maximum angular momentum for spherical harmonics.
        interaction_cls (str):
            Class name for interaction blocks.
        interaction_cls_first (str):
            Class name for the first interaction block.
        num_interactions (int):
            Number of interaction blocks.
        hidden_irreps (str):
            Irreducible representations for hidden features. For example if only
            a scalar representation with 128 channels is used can be "128x0e". If
            also a vector representation is used can be "128x0e + 128x1o".
        MLP_irreps (str):
            Irreducible representations for MLP layers.
        avg_num_neighbors (float):
            Average number of neighbors per atom used for normalization and numerical stability.
        atomic_numbers (List[int]):
            List of atomic numbers in the system.
        correlation (Union[int, List[int]]):
            Correlation order(s) for product blocks.
        gate (Optional[Callable]):
            Activation function for non-linearities.
        max_num_neighbors (int, optional):
            Maximum number of neighbors per atom.
        pair_repulsion (bool, optional):
            Whether to use pairwise repulsion.
        distance_transform (str, optional):
            Distance transformation type.
        radial_MLP (Optional[List[int]], optional):
            Radial MLP architecture.
        radial_type (Optional[str], optional):
            Radial basis type.
        cueq_config (Optional[Dict[str, Any]], optional):
            cuEquivariance configuration.
        use_cueq (Optional[bool], optional):
            Whether to use cuEquivariance acceleration.
        nls_distance_method:
            Method for computing a neighbor list. Supported values are
            `torch`, `nvalchemi_naive`, `nvalchemi_cell`, `nvalchemi_raw`
            and `custom_kernel`.
    """

    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: str,
        interaction_cls_first: str,
        num_interactions: int,
        hidden_irreps: str,
        MLP_irreps: str,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: Union[int, List[int]],
        gate: Optional[Callable],
        max_num_neighbors: int = 1000,
        pair_repulsion: bool = False,
        distance_transform: str = "None",
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        cueq_config: Optional[Any] = None,
        use_cueq: Optional[
            bool
        ] = False,  # defaults to False for backwards compatibility
        nls_distance_method: str = "torch",
    ):
        from mlcg.pl.model import get_class_from_str

        atomic_numbers.sort()
        atomic_numbers = torch.as_tensor(atomic_numbers)
        num_elements = atomic_numbers.shape[0]

        hidden_irreps = o3.Irreps(hidden_irreps)
        MLP_irreps = o3.Irreps(MLP_irreps)
        # Default to create CuEquivariance config if installed
        if CUET_AVAILABLE and use_cueq:
            print("=" * 60)
            print("INITIALIZING CUEQUIVARIANCE")
            print("=" * 60)
            print("Note: CuEquivariance kernels will be compiled on first use.")
            print(
                "This may take a few minutes but only happens once per configuration."
            )
            print("=" * 60)
            cueq_config = CuEquivarianceConfig(
                enabled=True,
                layout="ir_mul",  # irreps, multiplicity
                group="O3",
                optimize_all=True,
            )
        else:
            print("Using e3nn. cuEquivariance acceleration is disabled.")
            cueq_config = None
        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps(
            [(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))]
        )
        node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps,
            irreps_out=node_feats_irreps,
            cueq_config=cueq_config,
        )
        radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
            distance_transform=distance_transform,
        )
        edge_feats_irreps = o3.Irreps(f"{radial_embedding.out_dim}x0e")

        pair_repulsion_fn = None
        if pair_repulsion:
            pair_repulsion_fn = ZBLBasis(p=num_polynomial_cutoff)

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]

        # Interactions and readout
        inter = get_class_from_str(interaction_cls_first)(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
            cueq_config=cueq_config,
        )
        interactions = [inter]

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in interaction_cls_first:
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation[0],
            num_elements=num_elements,
            use_sc=use_sc_first,
            cueq_config=cueq_config,
        )
        products = [prod]

        readouts = [
            LinearReadoutBlock(hidden_irreps, o3.Irreps("1x0e"), cueq_config)
        ]

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                hidden_irreps_out = str(
                    hidden_irreps[0]
                )  # Select only scalars for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = get_class_from_str(interaction_cls)(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
                cueq_config=cueq_config,
            )
            interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i + 1],
                num_elements=num_elements,
                use_sc=True,
                cueq_config=cueq_config,
            )
            products.append(prod)
            if i == num_interactions - 2:
                readouts.append(
                    NonLinearReadoutBlock(
                        hidden_irreps_out,
                        (1 * MLP_irreps).simplify(),
                        gate,
                        o3.Irreps("1x0e"),
                        1,
                        cueq_config,
                    )
                )
            else:
                readouts.append(
                    LinearReadoutBlock(
                        hidden_irreps, o3.Irreps("1x0e"), cueq_config
                    )
                )

        super().__init__(
            atomic_numbers,
            node_embedding,
            radial_embedding,
            spherical_harmonics,
            interactions,
            products,
            readouts,
            r_max,
            max_num_neighbors,
            pair_repulsion_fn,
            nls_distance_method=nls_distance_method,
        )


@compile_mode("script")
class FrozenTypEmbeddingMACE(StandardMACE):
    """MACE variant whose initial node features come from a frozen, pretrained
    **per-type** embedding table instead of the learned
    :class:`LinearNodeEmbeddingBlock`.

    This is the MACE analogue of :ref:`mlcg.nn.FrozenTypEmbeddingSchNet`: every
    bead of the same type receives the same fixed feature vector, kept **frozen**
    (``requires_grad=False``) for the whole training. The one-hot ``node_attrs``
    are left unchanged, so MACE's element-indexed product and interaction weights
    remain per bead type exactly as in :ref:`mlcg.nn.StandardMACE` -- only the
    *initial* node features are frozen. For fixed per-bead (residue) embeddings
    that differ within a type, use :ref:`mlcg.nn.FrozenResEmbeddingMACE` instead.

    The table is built one of two ways:

    * If ``embedding_path`` is given, it is loaded from disk (a tensor of shape
      ``(n_types, num_features)``, where ``num_features`` is the scalar (``0e``)
      channel count of ``hidden_irreps``) and frozen.
    * Otherwise a fresh ``torch.nn.Embedding(n_types, num_features)`` is randomly
      initialized (under the active ``seed_everything``) and frozen.

    ``n_types`` is the number of entries in ``atomic_numbers``. All other
    arguments are identical to :ref:`mlcg.nn.StandardMACE`.

    Parameters
    ----------
    embedding_path:
        Optional path to a ``(n_types, num_features)`` tensor used as the frozen
        embedding table. If empty, the table is randomly initialized and frozen.
    """

    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: str,
        interaction_cls_first: str,
        num_interactions: int,
        hidden_irreps: str,
        MLP_irreps: str,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: Union[int, List[int]],
        gate: Optional[Callable],
        max_num_neighbors: int = 1000,
        pair_repulsion: bool = False,
        distance_transform: str = "None",
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        cueq_config: Optional[Any] = None,
        use_cueq: Optional[bool] = False,
        nls_distance_method: str = "torch",
        embedding_path: str = "",
    ):
        super().__init__(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            max_ell=max_ell,
            interaction_cls=interaction_cls,
            interaction_cls_first=interaction_cls_first,
            num_interactions=num_interactions,
            hidden_irreps=hidden_irreps,
            MLP_irreps=MLP_irreps,
            avg_num_neighbors=avg_num_neighbors,
            atomic_numbers=atomic_numbers,
            correlation=correlation,
            gate=gate,
            max_num_neighbors=max_num_neighbors,
            pair_repulsion=pair_repulsion,
            distance_transform=distance_transform,
            radial_MLP=radial_MLP,
            radial_type=radial_type,
            cueq_config=cueq_config,
            use_cueq=use_cueq,
            nls_distance_method=nls_distance_method,
        )

        num_features = o3.Irreps(hidden_irreps).count(o3.Irrep(0, 1))
        num_elements = int(self.atomic_numbers.shape[0])

        if embedding_path:
            table = torch.load(embedding_path)
            table = torch.as_tensor(table, dtype=torch.get_default_dtype())
            assert tuple(table.shape) == (num_elements, num_features), (
                f"frozen type-embedding table has shape {tuple(table.shape)} "
                f"but expected ({num_elements}, {num_features})"
            )
            frozen_embedding = torch.nn.Embedding.from_pretrained(
                table, freeze=True
            )
        else:
            frozen_embedding = torch.nn.Embedding(num_elements, num_features)

        # Freeze the type-embedding table: it is never trained. This replaces the
        # learned LinearNodeEmbeddingBlock built by StandardMACE (which is no
        # longer referenced, avoiding unused-parameter errors under DDP).
        frozen_embedding.weight.requires_grad_(False)
        self.node_embedding = frozen_embedding

    def get_node_feats_and_attrs(
        self, data: AtomicData
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        types_ids = self.types_mapping[data.atom_types].view(-1, 1)
        node_attrs = to_one_hot(types_ids, self.atomic_numbers.shape[0])
        node_feats = self.node_embedding(types_ids.squeeze(-1))
        return node_feats, node_attrs


@compile_mode("script")
class FrozenResEmbeddingMACE(StandardMACE):
    """MACE variant whose initial node features come from frozen, precomputed
    **per-residue/bead** embeddings carried on the input
    (``data.precomputed_embeddings``) rather than from a learned type embedding.

    This is the MACE analogue of :ref:`mlcg.nn.FrozenResEmbeddingSchNet`. Each
    individual bead gets its own fixed vector, mapped into the network's scalar
    node-feature channels by a learned linear projection, so ``embedding_dim``
    and ``hidden_irreps`` may differ.

    Unlike the SchNet version, MACE re-injects the fixed node identity tag
    (``node_attrs``) at every layer. Here that tag is collapsed to a **single
    dummy element** (``node_attrs = ones(n_nodes, 1)``, ``num_elements = 1``), so
    *all* residue/bead identity enters the model exclusively through the
    precomputed embedding and the product/interaction blocks are
    element-agnostic. ``atomic_numbers`` is therefore not an argument: it is
    fixed internally to a single dummy element. Keep ``pair_repulsion=False``
    (ZBL repulsion is keyed on physical atomic numbers, which are meaningless
    here).

    The ``data.precomputed_embeddings`` consumed here are populated upstream by
    the dataset/simulation: during training a per-bead embedding is drawn from a
    randomly sampled frame of the same molecule (data augmentation), while
    validation, inference and simulation use a single fixed frame.

    Parameters
    ----------
    embedding_dim:
        Dimension of the precomputed per-bead embeddings supplied in
        ``data.precomputed_embeddings`` (shape ``(n_beads, embedding_dim)``).

    All remaining arguments are identical to :ref:`mlcg.nn.StandardMACE`, except
    that ``atomic_numbers`` is omitted (fixed to a single dummy element).
    """

    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: str,
        interaction_cls_first: str,
        num_interactions: int,
        hidden_irreps: str,
        MLP_irreps: str,
        avg_num_neighbors: float,
        embedding_dim: int,
        correlation: Union[int, List[int]],
        gate: Optional[Callable],
        max_num_neighbors: int = 1000,
        pair_repulsion: bool = False,
        distance_transform: str = "None",
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        cueq_config: Optional[Any] = None,
        use_cueq: Optional[bool] = False,
        nls_distance_method: str = "torch",
    ):
        # A single dummy element: node_attrs carry no type information, so
        # products/interactions are element-agnostic and identity comes entirely
        # from the precomputed embedding projected below.
        super().__init__(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            max_ell=max_ell,
            interaction_cls=interaction_cls,
            interaction_cls_first=interaction_cls_first,
            num_interactions=num_interactions,
            hidden_irreps=hidden_irreps,
            MLP_irreps=MLP_irreps,
            avg_num_neighbors=avg_num_neighbors,
            atomic_numbers=[1],
            correlation=correlation,
            gate=gate,
            max_num_neighbors=max_num_neighbors,
            pair_repulsion=pair_repulsion,
            distance_transform=distance_transform,
            radial_MLP=radial_MLP,
            radial_type=radial_type,
            cueq_config=cueq_config,
            use_cueq=use_cueq,
            nls_distance_method=nls_distance_method,
        )

        num_features = o3.Irreps(hidden_irreps).count(o3.Irrep(0, 1))
        # Learned projection of the (frozen) precomputed embedding into the
        # scalar node-feature channels; replaces StandardMACE's
        # LinearNodeEmbeddingBlock as the initial-feature source.
        self.node_embedding = torch.nn.Linear(embedding_dim, num_features)

    def get_node_feats_and_attrs(
        self, data: AtomicData
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = data.precomputed_embeddings.to(self.node_embedding.weight.dtype)
        node_feats = self.node_embedding(emb)
        node_attrs = torch.ones(
            (node_feats.shape[0], 1),
            dtype=node_feats.dtype,
            device=node_feats.device,
        )
        return node_feats, node_attrs
