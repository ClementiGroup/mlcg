"""
GPTQ W16A16 quantization for filter networks.

This module provides GPTQ-style weight quantization with FP16 precision
for the filter networks in CFConv layers. Uses Triton FP16 kernels for
Tensor Core acceleration.

IMPORTANT: This module REQUIRES Triton and CUDA. No fallback is provided
to ensure bugs from missed kernel usage are caught immediately.
"""

import torch
import torch.nn as nn

from .kernels.models.linear import (
    fused_linear_tanh_fp16,
    linear_fp16_to_fp16,
    linear_fp16,
)

TRITON_FP16_AVAILABLE = True


class GPTQW16A16FilterNetwork(nn.Module):
    """
    GPTQ W16A16 quantized filter network using FP16 Triton kernels.

    This replaces the standard MLP filter network with FP16 precision:
    - Layer 0: FP16 Linear + Tanh (fused kernel)
    - Layer 1: FP16 Linear (no bias, no activation)

    The weights are stored in FP16 format. Input is converted from FP32
    to FP16, and output is converted back to FP32.

    Parameters
    ----------
    in_features : int
        Input dimension (num_rbf, typically ~50)
    hidden_features : int
        Hidden dimension (num_filters, typically 128)
    out_features : int
        Output dimension (num_filters, typically 128)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
    ):
        super().__init__()

        if not TRITON_FP16_AVAILABLE:
            raise RuntimeError(
                "GPTQ W16A16 requires Triton FP16 kernels but they are not available. "
                "Please ensure triton is installed and CUDA is available."
            )

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        # Layer 0: Linear + Tanh (fused)
        # Store weight as [K, N] = [in_features, hidden_features] for kernel
        self.weight0 = nn.Parameter(
            torch.empty(in_features, hidden_features, dtype=torch.float16)
        )
        self.bias0 = nn.Parameter(
            torch.empty(hidden_features, dtype=torch.float16)
        )

        # Layer 1: Linear only (no bias in original filter network)
        # Store weight as [K, N] = [hidden_features, out_features] for kernel
        self.weight1 = nn.Parameter(
            torch.empty(hidden_features, out_features, dtype=torch.float16)
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights using Xavier uniform."""
        nn.init.xavier_uniform_(self.weight0)
        nn.init.zeros_(self.bias0)
        nn.init.xavier_uniform_(self.weight1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantized filter network.

        Data flow (optimized for bandwidth):
        - Input: FP32
        - Layer 0: FP32 x FP16 -> FP16 (tanh applied)
        - Layer 1: FP16 x FP16 -> FP16 (accumulate FP32 internally, output FP16)
        - Output: FP16

        This saves ~50% memory bandwidth in the fused scatter kernels that
        consume filter_out, as they now read FP16 instead of FP32.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor [num_edges, num_rbf] in FP32

        Returns
        -------
        torch.Tensor
            Output tensor [num_edges, num_filters] in FP16
        """
        # if not x.is_cuda:
        #    raise RuntimeError(
        #        "GPTQ W16A16 filter network requires CUDA input. "
        #        "No CPU fallback is provided to prevent silent bugs."
        #    )

        x = x.contiguous()

        # Layer 0: FP32 input x FP16 weight -> FP16 output (with tanh)
        x = fused_linear_tanh_fp16(x, self.weight0, self.bias0)

        # Layer 1: FP16 input x FP16 weight -> FP16 output
        # (FP32 accumulation internally for numerical stability)
        x = linear_fp16_to_fp16(x, self.weight1)

        return x

    @classmethod
    def from_mlp(cls, mlp: nn.Module) -> "GPTQW16A16FilterNetwork":
        """
        Create a GPTQ W16A16 filter network from an existing MLP.

        Copies weights from the original MLP and converts to FP16.

        Parameters
        ----------
        mlp : nn.Module
            Original MLP filter network (2-layer: Linear+Tanh, Linear)

        Returns
        -------
        GPTQW16A16FilterNetwork
            Quantized filter network with FP16 weights
        """
        # Extract layer information from MLP
        # MLP structure: layers = [Linear, Tanh, Linear] for 2-layer filter net
        # Or with use_fused_first: first_linear + rest_layers

        use_fused = getattr(mlp, "use_fused_first", False)

        if use_fused:
            # Fused path: first_linear (layer 0) + rest_layers (layer 1)
            layer0 = mlp.first_linear
            # rest_layers should be just one Linear layer for filter network
            layer1 = mlp.rest_layers[0] if len(mlp.rest_layers) > 0 else None
        else:
            # Original path: layers = [Linear, Tanh, Linear]
            layers = list(mlp.layers)
            layer0 = layers[0]  # First Linear
            layer1 = layers[2] if len(layers) > 2 else layers[-1]  # Last Linear

        # Get dimensions and device
        in_features = layer0.in_features
        hidden_features = layer0.out_features
        out_features = layer1.out_features
        device = layer0.weight.device

        # Create quantized network
        quantized = cls(in_features, hidden_features, out_features)

        # Copy weights (transpose and convert to FP16)
        # Original Linear weight is [out, in], kernel expects [in, out]
        with torch.no_grad():
            quantized.weight0.copy_(layer0.weight.t().half())
            if layer0.bias is not None:
                quantized.bias0.copy_(layer0.bias.half())
            else:
                quantized.bias0.zero_()

            quantized.weight1.copy_(layer1.weight.t().half())

        # Move to the same device as the original MLP
        quantized = quantized.to(device)

        return quantized


class GPTQW16A16OutputNetwork(nn.Module):
    """
    GPTQ W16A16 quantized output network using FP16 Triton kernels.

    This replaces the standard MLP output network with FP16 precision:
    - Layer 0: FP16 Linear + Tanh (fused kernel)
    - Layer 1: FP16 Linear + Tanh (fused kernel)
    - Layer 2: FP16 Linear (no activation, outputs FP32)

    Typical output_network structure: [128, 128, 64, 1] or [128, 32, 16, 1]

    Parameters
    ----------
    in_features : int
        Input dimension (hidden_channels, typically 128)
    hidden1_features : int
        First hidden dimension
    hidden2_features : int
        Second hidden dimension
    out_features : int
        Output dimension (typically 1 for energy)
    """

    def __init__(
        self,
        in_features: int,
        hidden1_features: int,
        hidden2_features: int,
        out_features: int,
    ):
        super().__init__()

        if not TRITON_FP16_AVAILABLE:
            raise RuntimeError(
                "GPTQ W16A16 requires Triton FP16 kernels but they are not available. "
                "Please ensure triton is installed and CUDA is available."
            )

        self.in_features = in_features
        self.hidden1_features = hidden1_features
        self.hidden2_features = hidden2_features
        self.out_features = out_features

        # Layer 0: Linear + Tanh (fused)
        self.weight0 = nn.Parameter(
            torch.empty(in_features, hidden1_features, dtype=torch.float16)
        )
        self.bias0 = nn.Parameter(
            torch.empty(hidden1_features, dtype=torch.float16)
        )

        # Layer 1: Linear + Tanh (fused)
        self.weight1 = nn.Parameter(
            torch.empty(hidden1_features, hidden2_features, dtype=torch.float16)
        )
        self.bias1 = nn.Parameter(
            torch.empty(hidden2_features, dtype=torch.float16)
        )

        # Layer 2: Linear only (no activation, no bias for last layer)
        self.weight2 = nn.Parameter(
            torch.empty(hidden2_features, out_features, dtype=torch.float16)
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights using Xavier uniform."""
        nn.init.xavier_uniform_(self.weight0)
        nn.init.zeros_(self.bias0)
        nn.init.xavier_uniform_(self.weight1)
        nn.init.zeros_(self.bias1)
        nn.init.xavier_uniform_(self.weight2)

    def forward(self, x: torch.Tensor, data=None) -> torch.Tensor:
        """
        Forward pass through the quantized output network.

        Data flow:
        - Input: FP32
        - Layer 0: FP32 x FP16 -> FP16 (tanh applied)
        - Layer 1: FP16 x FP16 -> FP16 (tanh applied)
        - Layer 2: FP16 x FP16 -> FP32
        - Output: FP32

        Parameters
        ----------
        x : torch.Tensor
            Input tensor [num_atoms, hidden_channels] in FP32
        data : optional
            Unused, for API compatibility with MLP

        Returns
        -------
        torch.Tensor
            Output tensor [num_atoms, 1] in FP32
        """
        # if not x.is_cuda:
        #    raise RuntimeError(
        #        "GPTQ W16A16 output network requires CUDA input. "
        #        "No CPU fallback is provided to prevent silent bugs."
        #    )

        x = x.contiguous()

        # Layer 0: FP32 input x FP16 weight -> FP16 output (with tanh)
        x = fused_linear_tanh_fp16(x, self.weight0, self.bias0)

        # Layer 1: FP16 input x FP16 weight -> FP16 output (with tanh)
        x = fused_linear_tanh_fp16(x, self.weight1, self.bias1)

        # Layer 2: FP16 input x FP16 weight -> FP32 output
        x = linear_fp16(x, self.weight2)

        return x

    @classmethod
    def from_mlp(cls, mlp: nn.Module) -> "GPTQW16A16OutputNetwork":
        """
        Create a GPTQ W16A16 output network from an existing MLP.

        Copies weights from the original MLP and converts to FP16.

        Parameters
        ----------
        mlp : nn.Module
            Original MLP output network (3-layer: Linear+Tanh, Linear+Tanh, Linear)

        Returns
        -------
        GPTQW16A16OutputNetwork
            Quantized output network with FP16 weights
        """
        # MLP structure: layers = [Linear, Tanh, Linear, Tanh, Linear]
        use_fused = getattr(mlp, "use_fused_first", False)

        if use_fused:
            # Fused path: first_linear + rest_layers
            layer0 = mlp.first_linear
            rest = list(mlp.rest_layers)
            # rest_layers: [Linear, Tanh, Linear] for 3-layer output network
            layer1 = rest[0]
            layer2 = rest[2] if len(rest) > 2 else rest[-1]
        else:
            # Original path: layers = [Linear, Tanh, Linear, Tanh, Linear]
            layers = list(mlp.layers)
            layer0 = layers[0]  # First Linear
            layer1 = layers[2]  # Second Linear
            layer2 = layers[4]  # Third Linear

        # Get dimensions and device
        in_features = layer0.in_features
        hidden1_features = layer0.out_features
        hidden2_features = layer1.out_features
        out_features = layer2.out_features
        device = layer0.weight.device

        # Create quantized network
        quantized = cls(
            in_features, hidden1_features, hidden2_features, out_features
        )

        # Copy weights (transpose and convert to FP16)
        with torch.no_grad():
            quantized.weight0.copy_(layer0.weight.t().half())
            if layer0.bias is not None:
                quantized.bias0.copy_(layer0.bias.half())
            else:
                quantized.bias0.zero_()

            quantized.weight1.copy_(layer1.weight.t().half())
            if layer1.bias is not None:
                quantized.bias1.copy_(layer1.bias.half())
            else:
                quantized.bias1.zero_()

            quantized.weight2.copy_(layer2.weight.t().half())

        # Move to the same device as the original MLP
        quantized = quantized.to(device)

        return quantized


def apply_gptq_w16a16_to_model(model: nn.Module) -> nn.Module:
    """
    Apply GPTQ W16A16 quantization to all filter networks and output networks.

    This function finds all CFConv layers and replaces their filter_network
    MLP modules with GPTQ W16A16 quantized versions. It also finds and
    quantizes output_network MLPs in SchNet models.

    Parameters
    ----------
    model : nn.Module
        Model containing CFConv layers with filter networks

    Returns
    -------
    nn.Module
        Model with quantized networks (modified in-place)

    Raises
    ------
    RuntimeError
        If Triton FP16 kernels are not available
    """
    if not TRITON_FP16_AVAILABLE:
        raise RuntimeError(
            "GPTQ W16A16 requires Triton FP16 kernels but they are not available. "
            "Please ensure triton is installed and CUDA is available."
        )

    filter_count = 0
    output_count = 0

    # Quantize filter networks in CFConv layers
    for name, module in model.named_modules():
        if hasattr(module, "filter_network") and hasattr(
            module.filter_network, "layers"
        ):
            quantized_filter = GPTQW16A16FilterNetwork.from_mlp(
                module.filter_network
            )
            module.filter_network = quantized_filter
            filter_count += 1

    # Quantize output_network MLPs
    for name, module in model.named_modules():
        if hasattr(module, "output_network") and hasattr(
            module.output_network, "layers"
        ):
            layers = list(module.output_network.layers)
            num_linear = sum(1 for l in layers if hasattr(l, "weight"))

            if num_linear == 3:
                quantized_output = GPTQW16A16OutputNetwork.from_mlp(
                    module.output_network
                )
                module.output_network = quantized_output
                output_count += 1
            elif num_linear == 2:
                quantized_output = GPTQW16A16FilterNetwork.from_mlp(
                    module.output_network
                )
                module.output_network = quantized_output
                output_count += 1

    if filter_count == 0:
        raise RuntimeError(
            "No filter networks found to quantize. "
            "Ensure the model contains CFConv layers with filter_network MLPs."
        )

    print(
        f"[GPTQ W16A16] Quantized {filter_count} filter networks, {output_count} output networks"
    )

    return model


def validate_gptq_w16a16(model: nn.Module) -> bool:
    """
    Validate that GPTQ W16A16 quantization was applied correctly.

    Checks that all filter networks and output networks are quantized.

    Parameters
    ----------
    model : nn.Module
        Model to validate

    Returns
    -------
    bool
        True if all networks are quantized

    Raises
    ------
    RuntimeError
        If any filter network or output network is not quantized
    """
    for name, module in model.named_modules():
        if hasattr(module, "filter_network"):
            if not isinstance(module.filter_network, GPTQW16A16FilterNetwork):
                raise RuntimeError(
                    f"Filter network in {name} is not quantized! "
                    f"Got {type(module.filter_network).__name__} instead of "
                    f"GPTQW16A16FilterNetwork. GPTQ quantization failed to apply."
                )

        if hasattr(module, "output_network"):
            valid_types = (GPTQW16A16FilterNetwork, GPTQW16A16OutputNetwork)
            if not isinstance(module.output_network, valid_types):
                # Only warn if it's an MLP (some models might have different output networks)
                if hasattr(module.output_network, "layers"):
                    print(
                        f"[GPTQ W16A16] Warning: output_network in {name} is not quantized "
                        f"(got {type(module.output_network).__name__})"
                    )

    return True
