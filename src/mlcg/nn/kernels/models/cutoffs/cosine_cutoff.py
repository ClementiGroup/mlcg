import torch
import triton
import triton.language as tl

triton_pi = tl.constexpr(3.141592653589793)


# ============================================================================
# Utility funcitons triton
# ============================================================================
@triton.jit
def _cosine_cutoff(dist, cutoff_upper: tl.constexpr):
    factor = triton_pi / cutoff_upper
    C = 0.5 * (tl.cos(dist * factor) + 1.0)
    mask_dist = dist < cutoff_upper
    return tl.where(mask_dist, C, 0.0)


@triton.jit
def _d_cosine_cutoff_dd(dist, cutoff_upper: tl.constexpr):
    factor = triton_pi / cutoff_upper
    C = -0.5 * tl.sin(dist * factor) * factor
    mask_dist = dist < cutoff_upper
    return tl.where(mask_dist, C, 0.0)


@triton.jit
def _d2_cosine_cutoff_dd2(dist, cutoff_upper: tl.constexpr):
    factor = triton_pi / cutoff_upper
    C = -0.5 * tl.cos(dist * factor) * factor * factor
    mask_dist = dist < cutoff_upper
    return tl.where(mask_dist, C, 0.0)


# ============================================================================
# Utility funcitons torch
# ============================================================================
def _torch_cosine_cutoff(dist, cutoff_upper):
    C = 0.5 * (torch.cos(dist * torch.pi / cutoff_upper) + 1)
    return C * (dist < cutoff_upper).to(dist.dtype)


def _torch_d_cosine_cutoff_dd(dist, cutoff_upper):
    dC_dd = (
        -0.5
        * torch.sin(dist * torch.pi / cutoff_upper)
        * (torch.pi / cutoff_upper)
    )
    return dC_dd * (dist < cutoff_upper).to(dist.dtype)


def _torch_d2_cosine_cutoff_dd2(dist, cutoff_upper):
    d2C_dd2 = (
        -0.5
        * torch.cos(dist * torch.pi / cutoff_upper)
        * (torch.pi / cutoff_upper) ** 2
    )
    return d2C_dd2 * (dist < cutoff_upper).to(dist.dtype)


def _torch_d3_cosine_cutoff_dd3(dist, cutoff_upper):
    d3C_dd3 = (
        0.5
        * torch.sin(dist * torch.pi / cutoff_upper)
        * (torch.pi / cutoff_upper) ** 3
    )
    return d3C_dd3 * (dist < cutoff_upper).to(dist.dtype)
