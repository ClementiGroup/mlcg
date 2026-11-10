import torch
from .mlp import MLP
from .radial_basis import SkewedMuGaussianBasis

__all__ = ["NoiseAwareEmbedding"]


class NoiseAwareEmbedding(torch.nn.Embedding):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        noise_level_rbf: SkewedMuGaussianBasis,
        noise_network_activation_func: torch.nn.Module = torch.nn.SiLU(),
        **kwargs,
    ):
        """A modified Embedding layer that introduces a noise-level-dependent bias on
        all embedding channels."""
        super(NoiseAwareEmbedding, self).__init__(
            num_embeddings, embedding_dim, **kwargs
        )
        self._embedding_dim = embedding_dim
        self.noise_level_rbf = noise_level_rbf
        num_noise_level_rbf = self.noise_level_rbf.num_rbf
        self.filter_network = MLP(
            layer_widths=[num_noise_level_rbf, embedding_dim, embedding_dim],
            activation_func=noise_network_activation_func,
            last_bias=False,
        )

    def forward(
        self,
        discrete_types: torch.Tensor,
        noise_levels: torch.Tensor,
    ) -> torch.Tensor:
        y = super().forward(discrete_types)

        z = self.filter_network(self.noise_level_rbf(noise_levels))
        return y + z
