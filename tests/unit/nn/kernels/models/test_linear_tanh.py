import pytest
import torch
from torch.autograd import grad

from mlcg.nn.kernels.models.linear import fused_tanh_linear

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

# Clear triton cache
import os, shutil

shutil.rmtree(os.path.expanduser("~/.triton/cache"), ignore_errors=True)

# Clear torch.compile cache
torch._dynamo.reset()

# compiled version for fast-path comparison (only exercised on CUDA to avoid
# slowing down the test suite on CPU machines).
compiled_fused = torch.compile(fused_tanh_linear)


def reference_tanh_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None
):
    # torch.backends.cuda.matmul.allow_tf32 = False
    out = torch.tanh(x) @ weight
    # torch.backends.cuda.matmul.allow_tf32 = True
    if bias is not None:
        out = out + bias
    return out


# @pytest.mark.parametrize("device", DEVICES) #kernel has problem precision see docs
@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("with_bias", [True, False])
def test_fused_kernel_forward(device, with_bias):
    # random dimensions that are representative of hidden channels
    M, K, N = 10, 16, 16
    x = torch.randn(M, K, device=device, requires_grad=True)
    weight = torch.randn(K, N, device=device, requires_grad=True)
    bias = (
        torch.randn(N, device=device, requires_grad=True) if with_bias else None
    )

    y_ref = reference_tanh_linear(x, weight, bias)
    y_fused = fused_tanh_linear(x, weight, bias)

    torch.testing.assert_close(y_ref, y_fused)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("with_bias", [True, False])
def test_fused_kernel_gradients(device, with_bias):
    M, K, N = 10, 16, 16
    x1 = torch.randn(M, K, device=device, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    weight1 = torch.randn(K, N, device=device, requires_grad=True)
    weight2 = weight1.detach().clone().requires_grad_(True)
    bias1 = (
        torch.randn(N, device=device, requires_grad=True) if with_bias else None
    )
    bias2 = bias1.detach().clone().requires_grad_(True) if with_bias else None

    inputs_ref = [x1, weight1]
    inputs_fused = [x2, weight2]
    if with_bias:
        inputs_ref.append(bias1)
        inputs_fused.append(bias2)

    # reference gradients
    y_ref = reference_tanh_linear(x1, weight1, bias1)
    grad_ref = grad(y_ref.sum(), inputs_ref)

    # fused gradients
    y_fused = fused_tanh_linear(x2, weight2, bias2)
    grad_fused = grad(y_fused.sum(), inputs_fused)

    for i, (g_ref, g_fused) in enumerate(zip(grad_ref, grad_fused)):
        torch.testing.assert_close(g_ref, g_fused)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Compiled kernel tests require CUDA"
)
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("with_bias", [True, False])
def test_fused_kernel_compiled_forward(device, with_bias):
    M, K, N = 10, 16, 16
    x = torch.randn(M, K, device=device, requires_grad=True)
    weight = torch.randn(K, N, device=device, requires_grad=True)
    bias = (
        torch.randn(N, device=device, requires_grad=True) if with_bias else None
    )

    y_fused = fused_tanh_linear(x, weight, bias)
    y_compiled = compiled_fused(x, weight, bias)

    torch.testing.assert_close(y_fused, y_compiled)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Compiled kernel tests require CUDA"
)
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("with_bias", [True, False])
def test_fused_kernel_compiled_gradients(device, with_bias):
    M, K, N = 10, 16, 16
    x1 = torch.randn(M, K, device=device, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    weight1 = torch.randn(K, N, device=device, requires_grad=True)
    weight2 = weight1.detach().clone().requires_grad_(True)
    bias1 = (
        torch.randn(N, device=device, requires_grad=True) if with_bias else None
    )
    bias2 = bias1.detach().clone().requires_grad_(True) if with_bias else None

    inputs_eager = [x1, weight1]
    inputs_comp = [x2, weight2]
    if with_bias:
        inputs_eager.append(bias1)
        inputs_comp.append(bias2)

    # eager/normal gradients
    y_eager = fused_tanh_linear(x1, weight1, bias1)
    grad_eager = grad(y_eager.sum(), inputs_eager)

    # compiled gradients
    y_comp = compiled_fused(x2, weight2, bias2)
    grad_comp = grad(y_comp.sum(), inputs_comp)

    for i, (g_e, g_c) in enumerate(zip(grad_eager, grad_comp)):
        torch.testing.assert_close(g_e, g_c)
