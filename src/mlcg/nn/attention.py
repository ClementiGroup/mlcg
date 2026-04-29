import torch
from typing import List
from ._module_init import init_xavier_uniform
from typing import Optional
import math

"""MLCG for attention. Based on:

    Unke, O. T. et al. (2021).
    Spookynet: Learning force fields with electronic degrees of freedom and nonlocal effects.
    Nat. Commun. 12(1), 2021, 1-14
    https://www.nature.com/articles/s41467-021-27504-0

    and the corresponding implementation found here:

    https://github.com/OUnke/SpookyNet
"""


class Residual(torch.nn.Module):
    """
    Pre-activation residual block

    Parameters
    ----------
    layer_widths:
        The width of outputs after passing through
        each linear layer.

    activation_func:
        The (non-linear) activation function to apply
        to the output of each linear transformation. Per default tanh.

    activation_first:
        Inverting the order of linear layers and activation functions.
        By default first linear layer than activation fucntion.
    """

    def __init__(
        self,
        layer_widths: List[int] = None,
        activation_func: torch.nn.Module = torch.torch.nn.Tanh(),
        activation_first: bool = False,
    ) -> None:
        super(Residual, self).__init__()
        if layer_widths is None:
            layer_widths = [128, 128]
        layers = []
        for w_in, w_out in zip(layer_widths[:-1], layer_widths[1:]):
            if activation_first:
                layers.append(activation_func)
                layers.append(torch.nn.Linear(w_in, w_out, bias=True))
            if not activation_first:
                layers.append(torch.nn.Linear(w_in, w_out, bias=True))
                layers.append(activation_func)

        self.layers = torch.nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.layers:
            init_xavier_uniform(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return x + self.layers(x)


class ResidualStack(torch.nn.Module):
    """
    Stack of num_residuals pre-activation residual blocks evaluated in sequence.

    Parameters
    ----------
    num_residuals (int):
        Number of residual blocks to be stacked in sequence

    layer_widths:
        The width of outputs after passing through
        each linear layer

    activation_func:
        The (non-linear) activation function to apply
        to the output of each linear transformation. Per default tanh

    activation_first:
        Inverting the order of linear layers and activation functions
    """

    def __init__(
        self,
        num_residual: int,
        layer_widths: List[int] = None,
        activation_func: torch.nn.Module = torch.torch.nn.Tanh(),
        activation_first: bool = False,
    ) -> None:
        """Initializes the ResidualStack class."""
        super(ResidualStack, self).__init__()
        if layer_widths is None:
            layer_widths = [128, 128]
        self.stack = torch.nn.ModuleList(
            [
                Residual(layer_widths, activation_func, activation_first)
                for i in range(num_residual)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for residual in self.stack:
            x = residual(x)
        return x


class Residual_MLP(torch.nn.Module):
    """
    Residual Multilayer Perceptron

    Residual block, followed by a linear layer and an activation function

    Parameters
    ----------

    num_features_in (int):
        size of each input sample for linear layer

    num_features_out (int):
        size of each output sample for liner layer

    num_residuals (int):
        Number of residual blocks to be stacked in sequence

    layer_widths:
        The width of outputs after passing through
        each linear layer

    activation_func:
        The (non-linear) activation function to apply
        to the output of each linear transformation. Per default tanh

    activation_first:
        Inverting the order of linear layers and activation functions
    """

    def __init__(
        self,
        num_features_in: int,
        num_features_out: int,
        num_residual: int,
        layer_widths: List[int] = None,
        activation_func: torch.nn.Module = torch.torch.nn.Tanh(),
        activation_first: bool = False,
    ) -> None:
        super(Residual_MLP, self).__init__()
        if layer_widths is None:
            layer_widths = [128, 128]
        self.linear = torch.nn.Linear(
            num_features_in, num_features_out, bias=True
        )
        self.activation_func = activation_func
        self.residual = ResidualStack(
            num_residual, layer_widths, activation_func, activation_first
        )
        self.activation_first = activation_first
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_xavier_uniform(self.linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Pass"""

        if self.activation_first:
            return self.linear(self.activation_func(self.residual(x)))

        if not self.activation_first:
            return self.activation_func(self.linear(self.residual(x)))


class ExactAttention(torch.nn.Module):
    """Softmax Attention"""

    def __init__(
        self,
    ) -> torch.Tensor:
        super(ExactAttention, self).__init__()
        self.attention_matrix = None

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        num_batch: int,
        batch_seg: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Compute exact attention."""
        d = Q.shape[-1]
        dot = Q @ K.T  # compute dot product

        # exact attention matrix
        A = torch.exp((dot - torch.max(dot)) / d**0.5)

        if num_batch > 1:  # mask out entries of different batches
            brow = batch_seg.view(1, -1).expand(A.shape[-2], -1)
            bcol = batch_seg.view(-1, 1).expand(-1, A.shape[-1])
            mask = torch.where(
                brow == bcol, torch.ones_like(A), torch.zeros_like(A)
            )
            A = A * mask

        norm = torch.sum(A, dim=-1, keepdim=True) + eps
        A_weights = A / norm
        attention = A_weights @ V
        self.attention_matrix = A_weights.clone().detach().cpu().numpy()

        return attention


class FavorAttention(torch.nn.Module):
    """Favor+ Attention"""

    def __init__(self, dimv: int = 128, dimqk: int = 128) -> torch.Tensor:
        super(FavorAttention, self).__init__()

        self.register_buffer("omega", self._omega(dimv, dimqk))

    def _omega(self, nrows: int, ncols: int) -> torch.Tensor:
        """
        Returns nrows x ncols random sample feature orthogonal feature matrix, which
        is used for Favor+.

        Parameters
        ----------
        nrows: (int) number of rows
        ncols: (int) number of columns

        Return
        ----------
        random torch.Tensor
        """
        nblocks = int(nrows / ncols)

        blocks = []

        for i in range(nblocks):
            block = torch.rand(size=(ncols, ncols))
            "qr factorization of a matrix - Gram-Schmidt"
            q, _ = torch.linalg.qr(block)
            blocks.append(q.T)

        missing_row = nrows - nblocks * ncols

        if missing_row > 0:
            block = torch.rand(size=(ncols, ncols))
            q, _ = torch.linalg.qr(block)
            blocks.append(q.T[:missing_row])
        "Renormalize rows so they still follow N(0,1)"
        norm = torch.linalg.norm(
            torch.rand(size=(nrows, ncols)), axis=1, keepdims=True
        )
        return (norm * torch.vstack(blocks)).T

    def phi(
        self,
        x: torch.Tensor,
        is_query: bool,
        num_batch: int,
        batch_seg: torch.Tensor,
        eps: float = 1e-4,
    ) -> torch.Tensor:
        """
        Project Q, K, V using the random feature map omega into feature
        space for approximation of Attention.
        Normalize x and project into random feature space.
        """

        d = x.shape[-1]  # d hidden dimension of latent representation
        m = self.omega.shape[-1]
        U = torch.matmul(x.float() / d**0.25, self.omega.float())
        h = torch.sum(x**2, dim=-1, keepdim=True) / (2 * d**0.5)

        # determine maximum (is subtracted to prevent numerical overflow)
        if is_query:
            maximum, _ = torch.max(U, dim=-1, keepdim=True)
        else:
            if num_batch > 1:
                brow = batch_seg.view(1, -1, 1).expand(
                    num_batch, -1, U.shape[-1]
                )
                bcol = (
                    torch.arange(
                        num_batch,
                        dtype=batch_seg.dtype,
                        device=batch_seg.device,
                    )
                    .view(-1, 1, 1)
                    .expand(-1, U.shape[-2], U.shape[-1])
                )
                mask = torch.where(
                    brow == bcol, torch.ones_like(U), torch.zeros_like(U)
                )
                tmp = U.unsqueeze(0).expand(num_batch, -1, -1)
                tmp, _ = torch.max(mask * tmp, dim=-1)
                tmp, _ = torch.max(tmp, dim=-1)
                if tmp.device.type == "cpu":  # indexing faster on CPU
                    maximum = tmp[batch_seg].unsqueeze(-1)
                else:  # gathering is faster on GPUs
                    maximum = torch.gather(tmp, 0, batch_seg).unsqueeze(-1)
            else:
                maximum = torch.max(U)

        return (torch.exp(U - h - maximum) + eps) / math.sqrt(m)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        num_batch: int,
        batch_seg: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        Q = self.phi(Q, True, num_batch, batch_seg)  # random projection of Q
        K = self.phi(K, False, num_batch, batch_seg)  # random projection of K
        if num_batch > 1:
            d = Q.shape[-1]
            n = batch_seg.shape[0]

            # compute norm
            idx = batch_seg.unsqueeze(-1).expand(-1, d)
            tmp = K.new_zeros(num_batch, d).scatter_add_(0, idx, K)
            norm = torch.gather(Q @ tmp.T, -1, batch_seg.unsqueeze(-1)) + eps

            # the ops below are equivalent to this loop (but more efficient):
            # return torch.cat([Q[b==batch_seg]@(
            #    K[b==batch_seg].transpose(-1,-2)@V[b==batch_seg])
            #    for b in range(num_batch)])/norm
            if mask is None:  # mask can be shared across multiple attentions
                one_hot = torch.nn.functional.one_hot(batch_seg).to(
                    dtype=V.dtype, device=V.device
                )
                mask = one_hot @ one_hot.transpose(-1, -2)

            return (
                (mask * (K @ Q.transpose(-1, -2))).transpose(-1, -2) @ V
            ) / norm
        else:
            norm = Q @ torch.sum(K, 0, keepdim=True).T + eps

            return (Q @ (K.T @ V)) / norm


class Nonlocalinteractionblock(torch.nn.Module):
    """
    Block for updating features through nonlocal interactions via attention.

    Parameters
    ----------
    num_features_in:
        size of each input sample for linear layer

    num_features_out:
        size of each output sample for liner layer

    num_residuals_q, num_residuals_k, num_residuals_v:
        Number of residual blocks applied to features via self-attention
        for queries, keys, and values

    layer_widths:
        The width of outputs after passing through
        each linear layer

    activation_func:
        The (non-linear) activation function to apply
        to the output of each linear transformation. Per default tanh

    activation_first:
        Inverting the order of linear layers and activation functions
    """

    def __init__(
        self,
        num_features_in: int,
        num_features_out: int,
        num_residual_q: int,
        num_residual_k: int,
        num_residual_v: int,
        attention_block: torch.nn.Module,
        layer_widths: List[int] = None,
        activation_func: torch.nn.Module = torch.torch.nn.Tanh(),
        activation_first: bool = False,
    ):
        #  Initializes the NonlocalInteraction class
        super(Nonlocalinteractionblock, self).__init__()
        if layer_widths is None:
            layer_widths = [128, 128]
        self.resmlp_q = Residual_MLP(
            num_features_in,
            num_features_out,
            num_residual_q,
            layer_widths,
            activation_func,
            activation_first,
        )
        self.resmlp_k = Residual_MLP(
            num_features_in,
            num_features_out,
            num_residual_k,
            layer_widths,
            activation_func,
            activation_first,
        )
        self.resmlp_v = Residual_MLP(
            num_features_in,
            num_features_out,
            num_residual_v,
            layer_widths,
            activation_func,
            activation_first,
        )

        self.attention_block = attention_block

    def forward(
        self,
        x: torch.Tensor,
        num_batch: int,
        batch_seg: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Evaluate interaction block - computes attention matrix
        """
        q = self.resmlp_q(x)  # queries
        k = self.resmlp_k(x)  # keys
        v = self.resmlp_v(x)  # values

        return self.attention_block(q, k, v, num_batch, batch_seg, mask)


class AttentiveInteractionBlock(torch.nn.Module):
    r"""Interaction blocks for SchNet. Consists of atomwise
    transformations of embedded features that are continuously
    convolved with filters generated from radial basis function-expanded
    pairwise distances.

    Parameters
    ----------
    cfconv_layer:
        Continuous filter convolution layer for convolutions of radial basis
        function-expanded distances with embedded features
    attention_block:
        Specify if you want to use softmax attention (ExactAttention) or favor+ attention (FavorAttention)
    hidden_channels:
        Hidden dimension of embedded features
    activation_func:
        Activation function applied to linear layer outputs. Default Tanh.
    """

    def __init__(
        self,
        cfconv_layer: torch.nn.Module,
        attention_block: torch.nn.Module,
        hidden_channels: int = 128,
        activation_func: torch.nn.Module = torch.torch.nn.Tanh(),
    ):
        super(AttentiveInteractionBlock, self).__init__()
        self.conv = cfconv_layer
        self.activation = activation_func
        self.lin = torch.nn.Linear(hidden_channels, hidden_channels)
        self.attention_block = attention_block

        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
        init_xavier_uniform(self.lin)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_attr: torch.Tensor,
        num_batch: int,
        batch_seg: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        r"""Forward pass through the interaction block.

        Parameters
        ----------
        x:
            Embedded features of shape (num_examples, num_atoms,
            hidden_channels)
        edge_index:
            Graph edge index tensor of shape (2, total_num_edges)
        edge_weight:
            Graph edge weight (eg, distances), of shape (total_num_edges)
        edge_attr:
            Graph edge attributes (eg, expanded distances), of shape
            (total_num_edges, num_rbf)
        num_batch:
            Number of batch is the number of different molecules (int)
        batch_seg:
            Batch_seg is the index for each atom that specifies
            to which molecule in the batch it belongs (Tensor [N])

        Returns
        -------
        x:
            Updated embedded features of shape (num_examples * num_atoms,
            hidden_channels)
        """

        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.activation(x)
        x = x + self.attention_block(x, num_batch, batch_seg, mask)
        x = self.lin(x)

        return x


class AttentiveInteractionBlock2(torch.nn.Module):
    r"""Interaction blocks for SchNet. Consists of atomwise
    transformations of embedded features that are continuously
    convolved with filters generated from radial basis function-expanded
    pairwise distances.

    Parameters
    ----------
    cfconv_layer:
        Continuous filter convolution layer for convolutions of radial basis
        function-expanded distances with embedded features
    attention_block:
        Specify if you want to use softmax attention (ExactAttention) or favor+ attention (FavorAttention)
    hidden_channels:
        Hidden dimension of embedded features
    activation_func:
        Activation function applied to linear layer outputs. Default Tanh.
    """

    def __init__(
        self,
        cfconv_layer: torch.nn.Module,
        attention_block: torch.nn.Module,
        hidden_channels: int = 128,
        activation_func: torch.nn.Module = torch.torch.nn.Tanh(),
    ):
        super(AttentiveInteractionBlock2, self).__init__()
        self.conv = cfconv_layer
        self.activation = activation_func
        self.lin = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin_a = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin_o = torch.nn.Linear(hidden_channels, hidden_channels)
        self.attention_block = attention_block

        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
        init_xavier_uniform(self.lin)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_attr: torch.Tensor,
        num_batch: int,
        batch_seg: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        r"""Forward pass through the interaction block.

        Parameters
        ----------
        x:
            Embedded features of shape (num_examples, num_atoms,
            hidden_channels)
        edge_index:
            Graph edge index tensor of shape (2, total_num_edges)
        edge_weight:
            Graph edge weight (eg, distances), of shape (total_num_edges)
        edge_attr:
            Graph edge attributes (eg, expanded distances), of shape
            (total_num_edges, num_rbf)
        num_batch:
            Number of batch is the number of different molecules (int)
        batch_seg:
            Batch_seg is the index for each atom that specifies
            to which molecule in the batch it belongs (Tensor [N])

        Returns
        -------
        x:
            Updated embedded features of shape (num_examples * num_atoms,
            hidden_channels)
        """

        x_l = self.conv(x, edge_index, edge_weight, edge_attr)
        x_l = self.activation(x_l)
        x_l = self.lin(x_l)
        x_a = self.attention_block(x, num_batch, batch_seg, mask)
        x_a = self.activation(x_a)
        x_a = self.lin_a(x_a)
        # x = x + self.attention_block(x, num_batch, batch_seg, mask)
        # x = self.lin(x)
        return self.lin_o(x_a + x_l)
