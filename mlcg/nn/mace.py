from typing import Any, Callable, Dict, List, Optional, Union, Final

import torch
from e3nn import o3

try:
    from mace.modules.radial import ZBLBasis
    from mace.tools.scatter import scatter_sum
    from mace.tools import to_one_hot

    from mace.modules.blocks import (
        EquivariantProductBasisBlock,
        LinearNodeEmbeddingBlock,
        LinearReadoutBlock,
        RadialEmbeddingBlock,
    )
    from mace.modules.utils import get_edge_vectors_and_lengths

except ImportError as e:
    print(e)
    print(
        "Please install or set mace to your path before using this interface. "
        + "To install you can either run 'pip install git+https://github.com/ACEsuit/mace.git@v0.3.12', "
        + "or clone the repository and add it to your PYTHONPATH."
        ""
    )

# from ..pl.model import get_class_from_str
from ..data.atomic_data import AtomicData, ENERGY_KEY
from ..neighbor_list.neighbor_list import (
    atomic_data2neighbor_list,
    validate_neighborlist,
)

from e3nn.util.jit import compile_mode


@compile_mode("script")
class MACE(torch.nn.Module):
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

        self.register_buffer(
            "types_mapping",
            -1 * torch.ones(atomic_numbers.max() + 1, dtype=torch.long),
        )
        self.types_mapping[atomic_numbers] = torch.arange(
            atomic_numbers.shape[0]
        )

    def forward(self, data: AtomicData) -> AtomicData:
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

        # Interactions
        energies = [pair_energy]
        node_feats_concat: List[torch.Tensor] = []

        for i, (interaction, product) in enumerate(
            zip(self.interactions, self.products)
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
                node_feats=node_feats, sc=sc, node_attrs=node_attrs
            )
            node_feats_concat.append(node_feats)

        for i, readout in enumerate(self.readouts):
            feat_idx = -1 if len(self.readouts) == 1 else i
            node_energies = readout(node_feats_concat[feat_idx], node_heads)[
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

    @staticmethod
    def neighbor_list(
        data: AtomicData, rcut: float, max_num_neighbors: int = 1000
    ) -> dict:
        """Computes the neighborlist for :obj:`data` using a strict cutoff of :obj:`rcut`."""
        return {
            MACE.name: atomic_data2neighbor_list(
                data,
                rcut,
                self_interaction=False,
                max_num_neighbors=max_num_neighbors,
            )
        }


@compile_mode("script")
class StandardMACE(MACE):
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
        pair_repulsion: bool = False,
        apply_cutoff: bool = True,
        use_reduced_cg: bool = True,
        use_so3: bool = False,
        use_agnostic_product: bool = False,
        use_last_readout_only: bool = False,
        distance_transform: str = "None",
        edge_irreps: Optional[o3.Irreps] = None,
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        heads: Optional[List[str]] = None,
        cueq_config: Optional[Dict[str, Any]] = None,
        oeq_config: Optional[Dict[str, Any]] = None,
        readout_cls: Optional[str] = "mace.modules.blocks.NonLinearReadoutBlock",
        max_num_neighbors: int = 1000,
    ):
        from mlcg.pl.model import get_class_from_str

        atomic_numbers.sort()
        atomic_numbers = torch.as_tensor(atomic_numbers)
        num_elements = atomic_numbers.shape[0]

        hidden_irreps = o3.Irreps(hidden_irreps)
        MLP_irreps = o3.Irreps(MLP_irreps)

        if heads is None:
            heads = ["Default"]
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

        if not use_so3:
            sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        else:
            sh_irreps = o3.Irreps.spherical_harmonics(max_ell, p=1)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))

        def generate_irreps(l):
            str_irrep = "+".join([f"1x{i}e+1x{i}o" for i in range(l + 1)])
            return o3.Irreps(str_irrep)

        sh_irreps_inter = sh_irreps
        if hidden_irreps.count(o3.Irrep(0, -1)) > 0:
            sh_irreps_inter = generate_irreps(max_ell)
        interaction_irreps = (sh_irreps_inter * num_features).sort()[0].simplify()
        interaction_irreps_first = (sh_irreps * num_features).sort()[0].simplify()

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
            target_irreps=interaction_irreps_first,
            hidden_irreps=hidden_irreps,
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
            target_irreps=hidden_irreps,
            correlation=correlation[0],
            num_elements=num_elements,
            use_sc=use_sc_first,
            cueq_config=cueq_config,
            oeq_config=oeq_config,
            use_reduced_cg=use_reduced_cg,
            use_agnostic_product=use_agnostic_product,
        )
        products = [prod]

        readouts = []
        if not use_last_readout_only:
            readouts.append(
                LinearReadoutBlock(
                    hidden_irreps,
                    o3.Irreps(f"{len(heads)}1x0e"),
                    cueq_config,
                    oeq_config,
                )
            )

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
                    get_class_from_str(readout_cls)(
                        hidden_irreps_out,
                        (len(heads) * MLP_irreps).simplify(),
                        gate,
                        o3.Irreps(f"{len(heads)}x0e"),
                        len(heads),
                        cueq_config,
                        oeq_config,
                    )
                )
            elif not use_last_readout_only:
                readouts.append(
                    LinearReadoutBlock(
                        hidden_irreps,
                        o3.Irreps("1x0e"),
                        cueq_config,
                        oeq_config,
                    )
                )

        super().__init__(
            atomic_numbers=atomic_numbers,
            node_embedding=node_embedding,
            radial_embedding=radial_embedding,
            spherical_harmonics=spherical_harmonics,
            interactions=interactions,
            products=products,
            readouts=readouts,
            r_max=r_max,
            max_num_neighbors=max_num_neighbors,
            pair_repulsion_fn=pair_repulsion_fn,
        )


@compile_mode("script")
class ScaleShiftBlock(torch.nn.Module):
    def __init__(self, scale: float, shift: float) -> None:
        super().__init__()

        self.register_buffer(
            "scale",
            torch.tensor(scale, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "shift",
            torch.tensor(shift, dtype=torch.get_default_dtype()),
        )

    def forward(self, x: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
        # Ensure scale and shift are at least 1D for indexing
        scale_vals = torch.atleast_1d(self.scale)[head]
        shift_vals = torch.atleast_1d(self.shift)[head]

        return scale_vals * x + shift_vals


@compile_mode("script")
class ScaleShiftMACE(MACE):
    def __init__(
        self,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Add scale-shift transformation for interaction energies
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale,
            shift=atomic_inter_shift
        )

    def forward(self, data: AtomicData) -> AtomicData:
        # Setup
        num_atoms_arange = torch.arange(data.pos.shape[0])
        num_graphs = data.ptr.numel() - 1
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

        # Compute atomic reference energies (E0) - not scaled/shifted
        # This assumes atomic energies are handled elsewhere (e.g., preprocessing)
        # or set to zero if already subtracted from the dataset
        atomic_energy = torch.zeros(
            num_graphs,
            device=data.pos.device,
            dtype=data.pos.dtype,
        )

        # Compute pair repulsion energy if available
        if self.pair_repulsion_fn:
            pair_node_energy = self.pair_repulsion_fn(
                lengths, node_attrs, edge_index, self.atomic_numbers
            )
        else:
            pair_node_energy = torch.zeros(
                data.pos.shape[0],
                device=data.pos.device,
                dtype=data.pos.dtype,
            )

        # Interactions
        node_es_list = [pair_node_energy]
        node_feats_list: List[torch.Tensor] = []

        for i, (interaction, product) in enumerate(
            zip(self.interactions, self.products)
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
                node_feats=node_feats, sc=sc, node_attrs=node_attrs
            )
            node_feats_list.append(node_feats)

        for i, readout in enumerate(self.readouts):
            feat_idx = -1 if len(self.readouts) == 1 else i
            node_es_list.append(
                readout(node_feats_list[feat_idx], node_heads)[
                    num_atoms_arange, node_heads
                ]  # [n_nodes, len(heads)]
            )

        # Sum all interaction energy contributions per node
        node_inter_es = torch.sum(torch.stack(node_es_list, dim=0), dim=0)

        # Apply scaling and shifting to interaction energies
        node_inter_es = self.scale_shift(node_inter_es, node_heads)

        interaction_energy = scatter_sum(
            src=node_inter_es,
            index=data["batch"],
            dim=-1,
            dim_size=num_graphs
        )

        # Total energy = atomic reference energies + scaled interaction energies
        total_energy = atomic_energy + interaction_energy

        data.out[self.name] = {
            ENERGY_KEY: total_energy,
            "interaction_energy": interaction_energy,  # Additional output
        }

        return data


@compile_mode("script")
class StandardScaleShiftMACE(ScaleShiftMACE):
    def __init__(
        self,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
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
        pair_repulsion: bool = False,
        apply_cutoff: bool = True,
        use_reduced_cg: bool = True,
        use_so3: bool = False,
        use_agnostic_product: bool = False,
        use_last_readout_only: bool = False,
        distance_transform: str = "None",
        edge_irreps: Optional[o3.Irreps] = None,
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        heads: Optional[List[str]] = None,
        cueq_config: Optional[Dict[str, Any]] = None,
        oeq_config: Optional[Dict[str, Any]] = None,
        readout_cls: Optional[str] = "mace.modules.blocks.NonLinearReadoutBlock",
        max_num_neighbors: int = 1000,
    ):
        from mlcg.pl.model import get_class_from_str

        atomic_numbers.sort()
        atomic_numbers = torch.as_tensor(atomic_numbers)
        num_elements = atomic_numbers.shape[0]

        hidden_irreps = o3.Irreps(hidden_irreps)
        MLP_irreps = o3.Irreps(MLP_irreps)

        if heads is None:
            heads = ["Default"]
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

        if not use_so3:
            sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        else:
            sh_irreps = o3.Irreps.spherical_harmonics(max_ell, p=1)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))

        def generate_irreps(l):
            str_irrep = "+".join([f"1x{i}e+1x{i}o" for i in range(l + 1)])
            return o3.Irreps(str_irrep)

        sh_irreps_inter = sh_irreps
        if hidden_irreps.count(o3.Irrep(0, -1)) > 0:
            sh_irreps_inter = generate_irreps(max_ell)
        interaction_irreps = (sh_irreps_inter * num_features).sort()[0].simplify()
        interaction_irreps_first = (sh_irreps * num_features).sort()[0].simplify()

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
            target_irreps=interaction_irreps_first,
            hidden_irreps=hidden_irreps,
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
            target_irreps=hidden_irreps,
            correlation=correlation[0],
            num_elements=num_elements,
            use_sc=use_sc_first,
            cueq_config=cueq_config,
            oeq_config=oeq_config,
            use_reduced_cg=use_reduced_cg,
            use_agnostic_product=use_agnostic_product,
        )
        products = [prod]

        readouts = []
        if not use_last_readout_only:
            readouts.append(
                LinearReadoutBlock(
                    hidden_irreps,
                    o3.Irreps(f"{len(heads)}1x0e"),
                    cueq_config,
                    oeq_config,
                )
            )

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
                    get_class_from_str(readout_cls)(
                        hidden_irreps_out,
                        (len(heads) * MLP_irreps).simplify(),
                        gate,
                        o3.Irreps(f"{len(heads)}x0e"),
                        len(heads),
                        cueq_config,
                        oeq_config,
                    )
                )
            elif not use_last_readout_only:
                readouts.append(
                    LinearReadoutBlock(
                        hidden_irreps,
                        o3.Irreps("1x0e"),
                        cueq_config,
                        oeq_config,
                    )
                )

        super().__init__(
            atomic_inter_scale=atomic_inter_scale,
            atomic_inter_shift=atomic_inter_shift,
            atomic_numbers=atomic_numbers,
            node_embedding=node_embedding,
            radial_embedding=radial_embedding,
            spherical_harmonics=spherical_harmonics,
            interactions=interactions,
            products=products,
            readouts=readouts,
            r_max=r_max,
            max_num_neighbors=max_num_neighbors,
            pair_repulsion_fn=pair_repulsion_fn,
        )
