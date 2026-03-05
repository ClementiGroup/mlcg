import pytest
import torch
from torch.autograd import grad

from mlcg.nn.kernels import fused_distance_exp_norm_rbf_cosinecutoff
from mlcg.nn import ExpNormalBasis, CosineCutoff
from mlcg.geometry.internal_coordinates import compute_distances

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
SENDERS = torch.tensor([0, 1, 0, 2, 2, 1, 3, 4, 3, 5])
RECEIVERS = torch.tensor([1, 0, 2, 0, 1, 2, 4, 3, 5, 3])

compiled_fused = torch.compile(fused_distance_exp_norm_rbf_cosinecutoff)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("trainable", [True, False])
def test_fused_kernel_forward(device, trainable):
    basis = ExpNormalBasis(
        CosineCutoff(cutoff_lower=0.0, cutoff_upper=5.0),
        num_rbf=5,
        trainable=trainable,
    ).to(device)

    pos = torch.randn(10, 3, device=device, requires_grad=True)
    senders = SENDERS.to(device)
    receivers = RECEIVERS.to(device)

    distances = compute_distances(pos, torch.stack((senders, receivers)))
    rbf = basis(distances)

    kernel_distances, kernel_rbf = fused_distance_exp_norm_rbf_cosinecutoff(
        pos,
        senders,
        receivers,
        basis.means,
        basis.betas,
        basis.alpha,
        basis.cutoff.cutoff_upper,
    )

    torch.testing.assert_close(distances, kernel_distances)
    torch.testing.assert_close(rbf, kernel_rbf)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("trainable", [True, False])
def test_fused_kernel_gradients(device, trainable):
    basis = ExpNormalBasis(
        CosineCutoff(cutoff_lower=0.0, cutoff_upper=5.0),
        num_rbf=5,
        trainable=trainable,
    ).to(device)

    pos1 = torch.randn(10, 3, device=device, requires_grad=True)
    pos2 = pos1.detach().clone().requires_grad_(True)
    senders = SENDERS.to(device)
    receivers = RECEIVERS.to(device)

    grad_inputs = [pos1, basis.means, basis.betas] if trainable else [pos1]

    # Reference gradients
    distances = compute_distances(pos1, torch.stack((senders, receivers)))
    rbf = basis(distances)
    grad_ref = grad(rbf.sum(), grad_inputs)

    # Fused kernel gradients
    grad_inputs_fused = (
        [pos2, basis.means, basis.betas] if trainable else [pos2]
    )
    kernel_distances, kernel_rbf = fused_distance_exp_norm_rbf_cosinecutoff(
        pos2,
        senders,
        receivers,
        basis.means,
        basis.betas,
        basis.alpha,
        basis.cutoff.cutoff_upper,
    )
    grad_fused = grad(kernel_rbf.sum(), grad_inputs_fused)

    for i, (g_ref, g_fused) in enumerate(zip(grad_ref, grad_fused)):
        torch.testing.assert_close(g_ref, g_fused)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Compiled kernel tests require CUDA"
)
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("trainable", [True, False])
def test_fused_kernel_compiled_forward(device, trainable):
    basis = ExpNormalBasis(
        CosineCutoff(cutoff_lower=0.0, cutoff_upper=5.0),
        num_rbf=5,
        trainable=trainable,
    ).to(device)

    pos = torch.randn(10, 3, device=device, requires_grad=True)
    senders = SENDERS.to(device)
    receivers = RECEIVERS.to(device)

    kernel_distances, kernel_rbf = fused_distance_exp_norm_rbf_cosinecutoff(
        pos,
        senders,
        receivers,
        basis.means,
        basis.betas,
        basis.alpha,
        basis.cutoff.cutoff_upper,
    )
    compiled_distances, compiled_rbf = compiled_fused(
        pos,
        senders,
        receivers,
        basis.means,
        basis.betas,
        basis.alpha,
        basis.cutoff.cutoff_upper,
    )

    torch.testing.assert_close(kernel_distances, compiled_distances)
    torch.testing.assert_close(kernel_rbf, compiled_rbf)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Compiled kernel tests require CUDA"
)
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("trainable", [True, False])
def test_fused_kernel_compiled_gradients(device, trainable):
    basis = ExpNormalBasis(
        CosineCutoff(cutoff_lower=0.0, cutoff_upper=5.0),
        num_rbf=5,
        trainable=trainable,
    ).to(device)

    pos1 = torch.randn(10, 3, device=device, requires_grad=True)
    pos2 = pos1.detach().clone().requires_grad_(True)
    senders = SENDERS.to(device)
    receivers = RECEIVERS.to(device)

    grad_inputs = [pos1, basis.means, basis.betas] if trainable else [pos1]
    grad_inputs_compiled = (
        [pos2, basis.means, basis.betas] if trainable else [pos2]
    )

    # Eager gradients
    _, kernel_rbf = fused_distance_exp_norm_rbf_cosinecutoff(
        pos1,
        senders,
        receivers,
        basis.means,
        basis.betas,
        basis.alpha,
        basis.cutoff.cutoff_upper,
    )
    grad_eager = grad(kernel_rbf.sum(), grad_inputs)

    # Compiled gradients
    _, compiled_rbf = compiled_fused(
        pos2,
        senders,
        receivers,
        basis.means,
        basis.betas,
        basis.alpha,
        basis.cutoff.cutoff_upper,
    )
    grad_compiled = grad(compiled_rbf.sum(), grad_inputs_compiled)

    for i, (g_eager, g_comp) in enumerate(zip(grad_eager, grad_compiled)):
        torch.testing.assert_close(g_eager, g_comp)
