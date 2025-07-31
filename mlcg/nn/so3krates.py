"""Code adapted from https://github.com/thorben-frank/mlff"""

from typing import List, Final
import os
import warnings

import torch
import torch.nn as nn
from torch_geometric.utils import scatter
import math
import itertools as it

from mlcg.data import AtomicData
from mlcg.data._keys import ENERGY_KEY
from mlcg.nn import MLP
from mlcg.nn._module_init import init_xavier_uniform
from mlcg.nn.angular_basis.spherical_harmonics import SphericalHarmonics
from mlcg.geometry.internal_coordinates import compute_distance_vectors
from mlcg.neighbor_list.neighbor_list import (
    atomic_data2neighbor_list,
    validate_neighborlist,
)


def indx_fn(x):
    """Index function for CG matrix indexing."""
    return int((x + 1)**2) if x >= 0 else 0


def load_cgmatrix():
    """Load Clebsch-Gordan matrix from torch file."""
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cgmatrix_path = os.path.join(current_dir, 'cgmatrix.pt')
    return torch.load(cgmatrix_path, map_location='cpu')


def init_clebsch_gordan_matrix(degrees, l_out_max=None):
    """
    Initialize the Clebsch-Gordan matrix for given degrees and maximal output order.

    Args:
        degrees (List): Sequence of degrees l.
        l_out_max (int): Maximal output order. Defaults to max(degrees).

    Returns:
        Clebsch-Gordan matrix tensor
    """
    if l_out_max is None:
        _l_out_max = max(degrees)
    else:
        _l_out_max = l_out_max

    l_in_max = max(degrees)
    l_in_min = min(degrees)

    offset_corr = indx_fn(l_in_min - 1)
    _cg = load_cgmatrix()
    return _cg[offset_corr:indx_fn(_l_out_max), offset_corr:indx_fn(l_in_max), offset_corr:indx_fn(l_in_max)]

class L0Contraction:
    def __init__(self, degrees, dtype=torch.float32):
        """
            Create L0 contraction function that matches the JAX implementation.

            Args:
                degrees (List[int]): List of spherical harmonic degrees
                dtype: Data type for computations

            Methods:
                __call__(sphc): 
                    Contract spherical harmonic coordinates to L0.
        """
        self.degrees = degrees
        self.dtype = dtype

        # Get CG coefficients. Take diagonal for l=0 contraction
        cg = torch.diagonal(init_clebsch_gordan_matrix(degrees=list({0, *degrees}), l_out_max=0), dim1=1, dim2=2)[0]
        cg_rep = []
        unique_degrees, counts = torch.unique(torch.tensor(degrees), return_counts=True)

        for d, r in zip(unique_degrees, counts):
            start_idx = indx_fn(d.item() - 1)
            end_idx = indx_fn(d.item())
            cg_segment = cg[start_idx:end_idx]
            cg_rep.append(cg_segment.repeat(r.item()))

        self.cg_rep = torch.cat(cg_rep).to(dtype) # shape: (m_tot), m_tot = \sum_l 2l+1 for l in degrees

        # Create segment ids for each degree. Using it.chain() is more efficient than .extend()
        self.segment_ids = torch.tensor([
            y for y in it.chain(*[[n] * int(2 * degrees[n] + 1) for n in range(len(degrees))])
        ])

    def __call__(self, sphc):
        """
        Contract spherical harmonic coordinates to L0.

        Args:
            sphc (torch.Tensor): Spherical harmonic coordinates, shape (n, m_tot)

        Returns:
            torch.Tensor: L0 contracted features, shape (n, len(degrees))
        """
        # Element-wise multiplication and contraction
        weighted_sphc = sphc * sphc * self.cg_rep.to(sphc.device).unsqueeze(0)  # shape: (n, m_tot)

        return scatter(weighted_sphc, self.segment_ids.to(sphc.device), dim=1, dim_size=len(self.degrees))


class So3kratesInteraction(nn.Module):
    """SO3krates interaction layer implementing feature and geometric blocks."""

    def __init__(
        self,
        hidden_channels: int,
        degrees: List[int],
        edge_attr_dim: int,
        cutoff: nn.Module,
        fb_rad_filter_features: List[int],
        fb_sph_filter_features: List[int],
        gb_rad_filter_features: List[int],
        gb_sph_filter_features: List[int],
        n_heads: int = 4,
        activation: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.degrees = degrees
        self.n_heads = n_heads
        self.cutoff = cutoff
        self.activation = activation

        # Total number of spherical harmonic coefficients
        self.m_tot = sum(2 * l + 1 for l in degrees)

        # Feature block components
        self.fb_rad_filter = MLP(
            [edge_attr_dim] + fb_rad_filter_features,
            activation_func=activation,
            last_bias=True
        )
        self.fb_sph_filter = MLP(
            [len(degrees)] + fb_sph_filter_features,
            activation_func=activation,
            last_bias=True
        )

        # Geometric block components
        self.gb_rad_filter = MLP(
            [edge_attr_dim] + gb_rad_filter_features,
            activation_func=activation,
            last_bias=True
        )
        self.gb_sph_filter = MLP(
            [len(degrees)] + gb_sph_filter_features,
            activation_func=activation,
            last_bias=True
        )

        self.fb_attention = ConvAttention(hidden_channels, n_heads)
        self.gb_attention = SphConvAttention(hidden_channels, self.m_tot, degrees)

        # Initialize l0 contraction function using CG coefficients
        self.l0_contraction_fn = L0Contraction(degrees)

    def forward(
        self,
        x: torch.Tensor,  # (n_nodes, hidden_channels)
        chi: torch.Tensor,  # (n_nodes, m_tot)
        edge_index: torch.Tensor,  # (2, n_edges)
        edge_attr: torch.Tensor,  # (n_edges, edge_attr_dim)
        sph_ij: torch.Tensor,  # (n_edges, m_tot)
        edge_weight: torch.Tensor,  # (n_edges,)
    ):

        C = self.cutoff(edge_weight)

        # Compute distance-based features for geometric filtering
        chi_ij = chi[edge_index[0]] - chi[edge_index[1]]  # (n_edges, m_tot)

        # Contract to get degree-wise distances
        d_chi_ij_l = self.l0_contraction_fn(chi_ij)  # (n_edges, len(degrees))

        # Feature block
        w_rad = self.fb_rad_filter(edge_attr)  # (n_edges, fb_rad_filter_features[-1])
        w_sph = self.fb_sph_filter(d_chi_ij_l)  # (n_edges, fb_sph_filter_features[-1])
        w_fb = (w_rad + w_sph)  # (n_edges, hidden_channels)

        x_local = self.fb_attention(x, w_fb, edge_index, C)

        # Geometric block
        w_rad_gb = self.gb_rad_filter(edge_attr)  # (n_edges, gb_rad_filter_features[-1])
        w_sph_gb = self.gb_sph_filter(d_chi_ij_l)  # (n_edges, gb_sph_filter_features[-1])
        w_gb = (w_rad_gb + w_sph_gb)  # (n_edges, hidden_channels)

        chi_local = self.gb_attention(chi, sph_ij, x, w_gb, edge_index, C)

        return x_local, chi_local

    def reset_parameters(self):
        self.fb_rad_filter.reset_parameters()
        self.fb_sph_filter.reset_parameters()
        self.gb_rad_filter.reset_parameters()
        self.gb_sph_filter.reset_parameters()
        self.fb_attention.reset_parameters()
        self.gb_attention.reset_parameters()


class ConvAttention(nn.Module):
    def __init__(self, hidden_channels: int, n_heads: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_heads = n_heads
        self.head_dim = hidden_channels // n_heads

        self.q_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        w_ij: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: torch.Tensor,
    ):
        # Split into heads
        q = self.q_proj(x).view(-1, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(-1, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(-1, self.n_heads, self.head_dim)

        # Compute attention coefficients
        q_i = q[edge_index[1]]  # (n_edges, n_heads, head_dim)
        k_j = k[edge_index[0]]  # (n_edges, n_heads, head_dim)
        v_j = v[edge_index[0]]  # (n_edges, n_heads, head_dim)

        # Reshape w_ij for heads
        w_ij = w_ij.view(-1, self.n_heads, self.head_dim)

        # Compute attention scores
        alpha = (q_i * w_ij * k_j).sum(dim=-1) / math.sqrt(self.head_dim)
        alpha = alpha * cutoff  # (n_edges, n_heads)

        # Apply attention to values and aggregate messages
        out = scatter(alpha.unsqueeze(-1) * v_j,  # (n_edges, n_heads, head_dim)
                      edge_index[1],
                      dim=0,
                      dim_size=x.size(0))
        out = out.view(-1, self.hidden_channels)

        return out

    def reset_parameters(self):
        init_xavier_uniform(self.q_proj)
        init_xavier_uniform(self.k_proj)
        init_xavier_uniform(self.v_proj)


class SphConvAttention(nn.Module):
    """Spherical convolution attention for geometric features."""

    def __init__(self, hidden_channels: int, m_tot: int, degrees: List[int]):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_heads = len(degrees)
        self.head_dim = hidden_channels // self.n_heads
        self.m_tot = m_tot
        self.degrees = degrees

        self.q_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)

        # Create repeat indices for expanding attention to full spherical harmonics
        repeats = [2 * l + 1 for l in degrees]
        self.register_buffer("repeats", torch.tensor(repeats))

    def forward(
        self,
        chi: torch.Tensor,
        sph_ij: torch.Tensor,
        x: torch.Tensor,
        w_ij: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: torch.Tensor,
    ):
        # Compute queries and keys from scalar features
        q = self.q_proj(x).view(-1, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(-1, self.n_heads, self.head_dim)

        q_i = q[edge_index[1]]  # (n_edges, n_heads, head_dim)
        k_j = k[edge_index[0]]  # (n_edges, n_heads, head_dim)

        # Reshape w_ij for heads
        w_ij = w_ij.view(-1, self.n_heads, self.head_dim)

        # Compute attention scores
        alpha = (q_i * w_ij * k_j).sum(dim=-1) / math.sqrt(self.head_dim)
        alpha = alpha * cutoff  # (n_edges, n_heads)

        # Expand attention to full spherical harmonics
        # alpha has shape (n_edges, n_heads), repeats has shape (n_heads,)
        alpha_expanded = torch.repeat_interleave(alpha, self.repeats, dim=1)  # (n_edges, m_tot)

        # Apply attention to spherical harmonics and aggregate messages
        chi_out = scatter(alpha_expanded * sph_ij,
                          edge_index[1],
                          dim=0,
                          dim_size=chi.size(0))

        return chi_out

    def reset_parameters(self):
        init_xavier_uniform(self.q_proj)
        init_xavier_uniform(self.k_proj)


class So3kratesLayer(nn.Module):
    """Complete SO3krates layer with interaction and mixing."""

    def __init__(
        self,
        hidden_channels: int,
        degrees: List[int],
        edge_attr_dim: int,
        cutoff: nn.Module,
        fb_rad_filter_features: List[int],
        fb_sph_filter_features: List[int],
        gb_rad_filter_features: List[int],
        gb_sph_filter_features: List[int],
        n_heads: int = 4,
        activation: nn.Module = nn.SiLU(),
        parity: bool = True,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.degrees = degrees
        self.parity = parity

        # Total number of spherical harmonic coefficients
        self.m_tot = sum(2 * l + 1 for l in degrees)

        # Interaction block
        self.interaction = So3kratesInteraction(
            hidden_channels=hidden_channels,
            degrees=degrees,
            edge_attr_dim=edge_attr_dim,
            cutoff=cutoff,
            fb_rad_filter_features=fb_rad_filter_features,
            fb_sph_filter_features=fb_sph_filter_features,
            gb_rad_filter_features=gb_rad_filter_features,
            gb_sph_filter_features=gb_sph_filter_features,
            n_heads=n_heads,
            activation=activation,
        )

        # Feature-spherical interaction
        self.feature_sph_interaction = FeatureSphInteraction(
            hidden_channels, self.m_tot, degrees, activation
        )

        self.x_norm = nn.LayerNorm(hidden_channels)

        self.residual_delta_1 = nn.Sequential(
            activation,
            nn.Linear(hidden_channels, hidden_channels, bias=False),
            activation,
            nn.Linear(hidden_channels, hidden_channels, bias=False),
        )
        self.residual_out_1 = nn.Sequential(
            activation,
            nn.Linear(hidden_channels, hidden_channels, bias=False),
        )

        self.residual_delta_2 = nn.Sequential(
            activation,
            nn.Linear(hidden_channels, hidden_channels, bias=False),
            activation,
            nn.Linear(hidden_channels, hidden_channels, bias=False),
        )
        self.residual_out_2 = nn.Sequential(
            activation,
            nn.Linear(hidden_channels, hidden_channels, bias=False),
        )

    def forward(
        self,
        x: torch.Tensor,
        chi: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        sph_ij: torch.Tensor,
        edge_weight: torch.Tensor,
    ):
        x = self.x_norm(x)

        # Interaction block
        x_local, chi_local = self.interaction(
            x, chi, edge_index, edge_attr, sph_ij, edge_weight
        )

        # First residual connection
        x = x + x_local
        chi = chi + chi_local

        x = self.residual_out_1(x + self.residual_delta_1(x))
        x = self.x_norm(x)

        # Feature-spherical interaction
        delta_x, delta_chi = self.feature_sph_interaction(x, chi)

        # Second residual connection
        x = x + delta_x
        chi = chi + delta_chi

        x = self.residual_out_2(x + self.residual_delta_2(x))

        return x, chi

    def reset_parameters(self):
        self.interaction.reset_parameters()
        self.feature_sph_interaction.reset_parameters()
        for module in self.residual_delta_1:
            init_xavier_uniform(module)
        for module in self.residual_out_1:
            init_xavier_uniform(module)
        for module in self.residual_delta_2:
            init_xavier_uniform(module)
        for module in self.residual_out_2:
            init_xavier_uniform(module)


class FeatureSphInteraction(nn.Module):
    """Interaction between scalar features and spherical harmonics."""

    def __init__(
        self,
        hidden_channels: int,
        m_tot: int,
        degrees: List[int],
        activation: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.m_tot = m_tot
        self.degrees = degrees
        self.activation = activation

        # Initialize L0 contraction function using CG coefficients
        self.l0_contraction_fn = L0Contraction(degrees)

        # MLP for computing interaction coefficients
        self.interaction_mlp = MLP(
            [hidden_channels + len(degrees), hidden_channels + len(degrees)],
            activation_func=activation,
            last_bias=True
        )

        # Create repeat indices
        repeats = [2 * l + 1 for l in degrees]
        self.register_buffer("repeats", torch.tensor(repeats))

    def forward(self, x: torch.Tensor, chi: torch.Tensor):
        # Contract spherical harmonics to scalar
        d_chi = self.l0_contraction_fn(chi)  # (n_nodes, len(degrees))

        # Concatenate features and contracted spherical harmonics
        y = torch.cat([x, d_chi], dim=-1)  # (n_nodes, hidden_channels + len(degrees))

        # Apply MLP
        output = self.interaction_mlp(y)

        # Split output
        delta_x = output[:, :self.hidden_channels]
        coeff = output[:, self.hidden_channels:]

        # Expand coefficients and multiply with spherical harmonics
        coeff_expanded = torch.repeat_interleave(coeff, self.repeats, dim=1)
        delta_chi = coeff_expanded * chi

        return delta_x, delta_chi

    def reset_parameters(self):
        self.interaction_mlp.reset_parameters()


class So3krates(nn.Module):
    name: Final[str] = "So3krates"

    def __init__(
        self,
        embedding_layer: nn.Module,
        rbf_layer: nn.Module,
        interaction_blocks: List[So3kratesLayer],
        layer_norm: nn.Module,
        output_network: nn.Module,
        degrees: List[int],
        max_num_neighbors: int = 1000,
        normalize_sph: bool = True,
    ):
        super().__init__()

        self.embedding_layer = embedding_layer
        self.rbf_layer = rbf_layer

        if isinstance(interaction_blocks, List) or isinstance(
            interaction_blocks, So3kratesLayer
        ):
            self.interaction_blocks = torch.nn.ModuleList(interaction_blocks)
        else:
            raise RuntimeError(
                "interaction_blocks must be a single InteractionBlock or "
                "a list of InteractionBlocks."
            )

        self.layer_norm = layer_norm
        self.output_network = output_network
        self.degrees = degrees
        self.max_num_neighbors = max_num_neighbors

        # Spherical harmonics computation
        self.sph_harmonics = SphericalHarmonics(
            lmax=max(degrees), normalize=normalize_sph
        )

        # Total number of spherical harmonic coefficients
        self.m_tot = sum(2 * l + 1 for l in degrees)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding_layer.reset_parameters()
        self.rbf_layer.reset_parameters()
        for block in self.interaction_blocks:
            block.reset_parameters()
        self.output_network.reset_parameters()

    def forward(self, data: AtomicData):
        neighbor_list = data.neighbor_list.get(self.name)
        if not self.is_nl_compatible(neighbor_list):
            neighbor_list = self.neighbor_list(
                data, self.rbf_layer.cutoff.cutoff_upper, self.max_num_neighbors
            )[self.name]

        edge_index = neighbor_list["index_mapping"]

        distances, directions = compute_distance_vectors(
            data.pos, edge_index, neighbor_list["cell_shifts"]
        )

        rbf_expansion = self.rbf_layer(distances.squeeze(-1))  # Remove last dimension if size 1

        sph_ij = self.sph_harmonics(directions)
        # Concatenate only the required degrees
        sph_ij = torch.cat([sph_ij[l] for l in self.degrees], dim=-1)  # (n_pairs, sum_l 2l + 1 = m_tot)

        x = self.embedding_layer(data.atom_types)  # (n_atoms, hidden_channels)

        # Initialize spherical harmonics features. They can also be initalized with neighbor dependent embeddings
        chi = torch.zeros(
            (x.size(0), self.m_tot), device=x.device, dtype=x.dtype
        )

        for block in self.interaction_blocks:
            x, chi = block(x, chi, edge_index, rbf_expansion, sph_ij, distances)
        x = self.layer_norm(x)

        energy = self.output_network(x, data)
        energy = scatter(energy, data.batch, dim=0, reduce="sum")
        energy = energy.flatten()

        data.out[self.name] = {ENERGY_KEY: energy}
        return data

    def is_nl_compatible(self, nl):
        is_compatible = False
        if validate_neighborlist(nl):
            if (
                nl["order"] == 2
                and nl["self_interaction"] is False
                and nl["rcut"] == self.cutoff.cutoff_upper
            ):
                is_compatible = True
        return is_compatible

    @staticmethod
    def neighbor_list(
        data: AtomicData, rcut: float, max_num_neighbors: int = 1000
    ) -> dict:
        return {
            So3krates.name: atomic_data2neighbor_list(
                data,
                rcut,
                self_interaction=False,
                max_num_neighbors=max_num_neighbors,
            )
        }


class StandardSo3krates(So3krates):
    def __init__(
        self,
        rbf_layer: nn.Module,
        cutoff: nn.Module,
        output_hidden_layer_widths: List[int],
        hidden_channels: int = 132,
        embedding_size: int = 100,
        num_interactions: int = 6,
        degrees: List[int] = [1, 2, 3],
        n_heads: int = 4,
        activation: nn.Module = nn.SiLU(),
        max_num_neighbors: int = 1000,
        normalize_sph: bool = True,
    ):
        if num_interactions < 1:
            raise ValueError("At least one interaction block must be specified")

        if cutoff.cutoff_lower != rbf_layer.cutoff.cutoff_lower:
            warnings.warn(
                "Cutoff function lower cutoff, {}, and radial basis function "
                " lower cutoff, {}, do not match.".format(
                    cutoff.cutoff_lower, rbf_layer.cutoff.cutoff_lower
                )
            )
        if cutoff.cutoff_upper != rbf_layer.cutoff.cutoff_upper:
            warnings.warn(
                "Cutoff function upper cutoff, {}, and radial basis function "
                " upper cutoff, {}, do not match.".format(
                    cutoff.cutoff_upper, rbf_layer.cutoff.cutoff_upper
                )
            )

        assert hidden_channels % len(degrees) == 0, "hidden_channels must be divisible by len(degrees)."
        assert hidden_channels % n_heads == 0, "hidden_channels must be divisible by n_heads."

        fb_rad_filter_features = [hidden_channels, hidden_channels]
        fb_sph_filter_features = [hidden_channels // 4, hidden_channels]
        gb_rad_filter_features = [hidden_channels, hidden_channels]
        gb_sph_filter_features = [hidden_channels // 4, hidden_channels]

        embedding_layer = nn.Embedding(embedding_size, hidden_channels)

        interaction_blocks = []
        for i in range(num_interactions):
            interaction_blocks.append(
                So3kratesLayer(
                    hidden_channels=hidden_channels,
                    degrees=degrees,
                    edge_attr_dim=rbf_layer.num_rbf,
                    cutoff=cutoff,
                    fb_rad_filter_features=fb_rad_filter_features,
                    fb_sph_filter_features=fb_sph_filter_features,
                    gb_rad_filter_features=gb_rad_filter_features,
                    gb_sph_filter_features=gb_sph_filter_features,
                    n_heads=n_heads,
                    activation=activation,
                )
            )

        layer_norm = nn.LayerNorm(hidden_channels)

        output_layer_widths = (
            [hidden_channels] + output_hidden_layer_widths + [1]
        )
        output_network = MLP(
            output_layer_widths, activation_func=activation, last_bias=True
        )

        super(StandardSo3krates, self).__init__(
            embedding_layer,
            rbf_layer,
            interaction_blocks,
            layer_norm,
            output_network,
            degrees,
            max_num_neighbors,
            normalize_sph,
        )
