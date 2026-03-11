import triton
import triton.language as tl

triton_pi = tl.constexpr(3.141592653589793)


# ============================================================================
# Utility funcitons
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
