"""
Uniform cubic Hermite tables of smooth vector functions of one scalar.

A table has shape ``[num_knots, 3, F]``: row ``k`` holds the value,
``spacing * derivative`` and the difference to the next value
``v_{k+1} - v_k`` at knot ``x_k = k * spacing``, so the segment
``[x_k, x_{k+1}]`` is the contiguous block ``table[k:k+2]``. The difference
is computed in float64 before the table is rounded to float32: taking it
from two float32 values would add ``eps * |v|`` to every interpolated slope,
an error that grows as ``1 / spacing``.
"""

from typing import Callable, Sequence

import torch
import triton
import triton.language as tl


def aligned_num_knots(
    upper: float, num_knots: int, breakpoints: Sequence[float] = ()
) -> int:
    """The smallest knot count in ``[num_knots, 2 num_knots)`` whose uniform
    grid over ``[0, upper]`` puts every breakpoint within 0.1% of a spacing
    from a knot (the closest one found otherwise): no segment then
    straddles a point where the function is not smooth."""
    points = [b / upper for b in breakpoints if 0 < b < upper]

    def misalignment(knots):
        n = knots - 1
        return max((abs(x * n - round(x * n)) for x in points), default=0.0)

    candidates = range(num_knots, 2 * num_knots)
    good = [k for k in candidates if misalignment(k) <= 1e-3]
    return good[0] if good else min(candidates, key=misalignment)


def hermite_table(
    fn: Callable, upper: float, num_knots: int, device
) -> torch.Tensor:
    """Tabulate ``fn: [n] -> [n, F]`` on ``num_knots`` uniform knots over
    ``[0, upper]``. ``fn`` is evaluated in float64 and differentiated with
    forward-mode AD; the table is returned in float32."""
    knots = torch.linspace(
        0.0, upper, num_knots, dtype=torch.float64, device=device
    )
    value, slope = torch.func.jvp(fn, (knots,), (torch.ones_like(knots),))
    spacing = upper / (num_knots - 1)
    step = torch.zeros_like(value)
    step[:-1] = value[1:] - value[:-1]
    table = torch.stack([value, spacing * slope, step], dim=1)
    return table.float().contiguous()


def hermite_relative_error(
    fn: Callable, table: torch.Tensor, upper: float
) -> float:
    """Largest error of the interpolated value and derivative at the
    segment midpoints (where it peaks), relative to ``max |fn|`` and
    ``max |fn'|``."""
    num_knots = table.shape[0]
    spacing = upper / (num_knots - 1)
    mid = torch.arange(num_knots - 1, dtype=torch.float64, device=table.device)
    mid = (mid + 0.5) * spacing
    value, slope = torch.func.jvp(fn, (mid,), (torch.ones_like(mid),))
    v0, m0, dv = (table[:-1, c].double() for c in range(3))
    m1 = table[1:, 1].double()
    approx_value = v0 + 0.5 * dv + 0.125 * (m0 - m1)
    approx_slope = (1.5 * dv - 0.25 * (m0 + m1)) / spacing
    return max(
        float((approx_value - value).abs().max() / value.abs().max()),
        float((approx_slope - slope).abs().max() / slope.abs().max()),
    )


@triton.jit
def locate_knot(x, upper, num_knots):
    """Segment index and local coordinate ``t`` in [0, 1] of ``x`` in
    ``[0, upper]``."""
    s = x * (num_knots - 1) / upper
    k = tl.minimum(s.to(tl.int32), num_knots - 2)
    return k, s - k


@triton.jit
def load_hermite_segment(table_ptr, k, f, mask, F: tl.constexpr):
    """Value ``v0``, slopes ``m0, m1`` and value difference ``dv`` of
    segments ``k`` [BLOCK_E] at features ``f`` [BLOCK_F], as [BLOCK_E,
    BLOCK_F] tiles."""
    row = k.to(tl.int64)[:, None] * (3 * F) + f[None, :]
    v0 = tl.load(table_ptr + row, mask=mask, other=0.0)
    m0 = tl.load(table_ptr + row + F, mask=mask, other=0.0)
    dv = tl.load(table_ptr + row + 2 * F, mask=mask, other=0.0)
    m1 = tl.load(table_ptr + row + 4 * F, mask=mask, other=0.0)
    return v0, m0, dv, m1


@triton.jit
def hermite_value(t, v0, m0, dv, m1):
    t2 = t * t
    t3 = t2 * t
    w1 = 3.0 * t2 - 2.0 * t3
    return v0 + w1 * dv + (t3 - 2.0 * t2 + t) * m0 + (t3 - t2) * m1


@triton.jit
def hermite_slope(t, v0, m0, dv, m1):
    """``spacing * d/dx`` of :func:`hermite_value`."""
    t2 = t * t
    return (
        6.0 * (t - t2) * dv
        + (3.0 * t2 - 4.0 * t + 1.0) * m0
        + (3.0 * t2 - 2.0 * t) * m1
    )
