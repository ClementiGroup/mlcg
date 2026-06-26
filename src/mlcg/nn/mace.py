from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
    Final,
    Tuple,
    Type,
)

import torch
from e3nn import nn, o3
from e3nn.util.jit import compile_mode
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

try:
    import openequivariance as oeq

    OEQ_AVAILABLE = True

except ImportError:
    OEQ_AVAILABLE = False
    print(
        "openequivariance is not installed. openequivariance features will be disabled. It is recommended to install openequivariance for better performance. "
        + "To install openequivariance run pip install openequivariance"
    )
if OEQ_AVAILABLE:
    from mace.modules.wrapper_ops import OEQConfig

from ..data.atomic_data import AtomicData, ENERGY_KEY
from ..neighbor_list.neighbor_list import (
    atomic_data2neighbor_list,
    validate_neighborlist,
)


# This is a copy of the residual RealAgnosticResidualInteractionBlock as it is in
# mace v0.3.16
# https://github.com/ACEsuit/mace/blob/b5faaa076c49778fc17493edfecebcabeb960155/mace/modules/blocks.py#L474
class CustomRealAgnosticResidualInteractionBlock(InteractionBlock):
    r"""version of the mace.modules.blocks RealAgnosticResidualInteractionBlock
    without a hardcoded tanh gate.

    We avoid doing a general AgnosticResidualInteractionBlock as it would require
    a larger rewritting of our MACE implementation
    """

    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None
        if not hasattr(self, "oeq_config"):
            self.oeq_config = None

        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.edge_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.edge_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.edge_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
            oeq_config=self.oeq_config,
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
        cutoff: Optional[torch.Tensor] = None,
        lammps_class: Optional[Any] = None,
        lammps_natoms: Tuple[int, int] = (0, 0),
        first_layer: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        if cutoff is not None:
            tp_weights = tp_weights * cutoff
        message = None
        if hasattr(self, "conv_fusion"):
            message = self.conv_tp(
                node_feats, edge_attrs, tp_weights, edge_index
            )
        else:
            mji = self.conv_tp(
                node_feats[edge_index[0]], edge_attrs, tp_weights
            )  # [n_nodes, irreps]
            message = scatter_sum(
                src=mji,
                index=edge_index[1],
                dim=0,
                dim_size=node_feats.shape[0],
            )
        message = self.truncate_ghosts(message, n_real)
        node_attrs = self.truncate_ghosts(node_attrs, n_real)
        sc = self.truncate_ghosts(sc, n_real)
        message = self.linear(message) / self.avg_num_neighbors
        return (
            self.reshape(message),
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2]


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

        types_ids = self.types_mapping[data.atom_types].view(-1, 1)
        node_attrs = to_one_hot(types_ids, self.atomic_numbers.shape[0])

        # Embeddings
        node_feats = self.node_embedding(node_attrs)

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
        edge_feats, cutoff = self.radial_embedding(
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

        energies = [pair_energy]

        # Interactions
        for i, (interaction, product, readout) in enumerate(
            zip(self.interactions, self.products, self.readouts)
        ):
            node_feats, sc = interaction(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
                cutoff=cutoff,
                first_layer=(i == 0),
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=node_attrs.to(dtype=node_feats.dtype),
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
        apply_cutoff (bool, optional):
            Whether to apply an envelope cutoff to the radial embedding.
        use_reduced_cg (bool, optional):
            Whether to use reduced Clebsch-Gordan coefficients in the product blocks.
        use_so3 (bool, optional):
            Whether to use SO(3) spherical harmonics (parity +1 only) instead of O(3).
        use_agnostic_product (bool, optional):
            Whether to use element-agnostic symmetric contraction in product blocks.
        edge_irreps (Optional[o3.Irreps], optional):
            Custom irreps for edge features in interaction blocks (all layers except first).
            If None, defaults to the standard interaction irreps.
        use_edge_irreps_first (bool, optional):
            Whether to apply `edge_irreps` also to the first interaction block's scalar channels.
        cueq_config (Optional[Dict[str, Any]], optional):
            cuEquivariance configuration.
        use_cueq (Optional[bool], optional):
            Whether to use cuEquivariance acceleration.
        oeq_config (Optional[Dict[str, Any]], optional):
            openequivariance configuration.
        use_oeq (Optional[bool], optional):
            Whether to use openequivariance acceleration.
        readout_cls (Optional[Type[NonLinearReadoutBlock]], optional):
            Class to use for the final (non-linear) readout block.
        keep_last_layer_irreps (bool, optional):
            If False, the last interaction layer outputs only scalar irreps, reducing
            parameters. Defaults to True for backward compatibility.
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
        apply_cutoff: bool = True,
        use_reduced_cg: bool = True,
        use_so3: bool = False,
        use_agnostic_product: bool = False,
        distance_transform: str = "None",
        edge_irreps: Optional[o3.Irreps] = None,
        use_edge_irreps_first: bool = False,
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        cueq_config: Optional[Any] = None,
        use_cueq: Optional[
            bool
        ] = False,  # Defaults to False for backwards compatibility
        oeq_config: Optional[Dict[str, Any]] = None,
        use_oeq: Optional[
            bool
        ] = False,  # Defaults to False for backwards compatibility
        readout_cls: Optional[
            Type[NonLinearReadoutBlock]
        ] = NonLinearReadoutBlock,
        keep_last_layer_irreps: bool = True,  # Default to True for backward compatibility
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
            oeq_config = None
        elif OEQ_AVAILABLE and use_oeq:
            print("=" * 60)
            print("INITIALIZING OPEN EQUIVARIANCE")
            print("=" * 60)
            print("This may take a few minutes.")
            print("=" * 60)
            oeq_config = OEQConfig(
                enabled=True,
                optimize_all=True,
                conv_fusion="atomic",
            )
            cueq_config = None
        else:
            print(
                "Using e3nn. cuEquivariance and openequivariance acceleration are disabled."
            )
            cueq_config = None
            oeq_config = None
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
            apply_cutoff=apply_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f"{radial_embedding.out_dim}x0e")

        pair_repulsion_fn = None
        if pair_repulsion:
            pair_repulsion_fn = ZBLBasis(p=num_polynomial_cutoff)

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        if not use_so3:
            sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        else:
            sh_irreps = o3.Irreps.spherical_harmonics(max_ell, p=1)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        # interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        sh_irreps_inter = sh_irreps
        if hidden_irreps.count(o3.Irrep(0, -1)) > 0:
            sh_irreps_inter = o3.Irreps(
                "+".join([f"1x{i}e+1x{i}o" for i in range(max_ell + 1)])
            )
        interaction_irreps = (
            (sh_irreps_inter * num_features).sort()[0].simplify()
        )
        interaction_irreps_first = (
            (sh_irreps * num_features).sort()[0].simplify()
        )

        spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]

        # Interactions and readout
        if num_interactions == 1 and not keep_last_layer_irreps:
            hidden_irreps_out = str(hidden_irreps[0])
        else:
            hidden_irreps_out = hidden_irreps
        edge_irreps_first = None
        if use_edge_irreps_first and edge_irreps is not None:
            edge_irreps_first = o3.Irreps(
                f"{edge_irreps.count(o3.Irrep(0, 1))}x0e"
            )
        inter = get_class_from_str(interaction_cls_first)(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps_first,
            hidden_irreps=hidden_irreps_out,
            edge_irreps=edge_irreps_first,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
            cueq_config=cueq_config,
            oeq_config=oeq_config,
        )
        interactions = [inter]

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in interaction_cls_first:
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps_out,
            correlation=correlation[0],
            num_elements=num_elements,
            use_sc=use_sc_first,
            cueq_config=cueq_config,
            oeq_config=oeq_config,
            use_reduced_cg=use_reduced_cg,
            use_agnostic_product=use_agnostic_product,
        )
        products = [prod]

        readouts = [
            LinearReadoutBlock(hidden_irreps, o3.Irreps("1x0e"), cueq_config)
        ]

        for i in range(num_interactions - 1):
            if i == num_interactions - 2 and not keep_last_layer_irreps:
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
                edge_irreps=edge_irreps,
                radial_MLP=radial_MLP,
                cueq_config=cueq_config,
                oeq_config=oeq_config,
            )
            interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i + 1],
                num_elements=num_elements,
                use_sc=True,
                cueq_config=cueq_config,
                oeq_config=oeq_config,
                use_reduced_cg=use_reduced_cg,
                use_agnostic_product=use_agnostic_product,
            )
            products.append(prod)
            if i == num_interactions - 2:
                readouts.append(
                    readout_cls(
                        hidden_irreps_out,
                        (1 * MLP_irreps).simplify(),
                        gate,
                        o3.Irreps("1x0e"),
                        1,
                        cueq_config,
                        oeq_config,
                    )
                )
            else:
                readouts.append(
                    LinearReadoutBlock(
                        hidden_irreps,
                        o3.Irreps("1x0e"),
                        cueq_config,
                        oeq_config,
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
