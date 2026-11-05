import functools

import torch
import triton


def ensure_contiguous(fn):
    """
    Decorator that ensures all torch.Tensor arguments are contiguous
    before passing them to the wrapped function.

    This is essential for Triton kernels, which use flat pointer arithmetic
    (e.g. row * stride + col) that silently produces wrong results if the
    input tensors have non-standard memory layouts (e.g. from transposes,
    slices, or autograd-produced gradients).

    Usage
    -----
    @ensure_contiguous
    def my_triton_wrapper(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ...

    Notes
    -----
    - Non-tensor arguments (ints, floats, bools) are passed through unchanged.
    - If a tensor is already contiguous, `.contiguous()` is a no-op (no copy).
    - Applies to both positional and keyword arguments.
    - Gradients coming from autograd are especially prone to being
      non-contiguous, since ops like linear layers or einsums can produce
      transposed or strided gradient tensors.

    Example
    -------
    >>> x = torch.randn(4, 8).T        # non-contiguous after transpose
    >>> x.is_contiguous()
    False
    >>> @ensure_contiguous
    ... def kernel(t): return t
    >>> kernel(x).is_contiguous()
    True
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        args = tuple(
            (
                a.contiguous()
                if isinstance(a, torch.Tensor) and not a.is_contiguous()
                else a
            )
            for a in args
        )
        kwargs = {
            k: (
                v.contiguous()
                if isinstance(v, torch.Tensor) and not v.is_contiguous()
                else v
            )
            for k, v in kwargs.items()
        }
        return fn(*args, **kwargs)

    return wrapper


def _fitting_tiles(configs, named_args, tile_per_thread, **kwargs):
    """Drop configs wider than the features ``F`` or whose
    ``[BLOCK_E, BLOCK_F]`` tiles exceed ``tile_per_thread`` elements per
    thread (they spill); a config without ``BLOCK_F`` spans all features."""
    width = triton.next_power_of_2(named_args["F"])

    def fits(c):
        block_f = c.kwargs.get("BLOCK_F", width)
        tile = c.kwargs["BLOCK_E"] * block_f / (32 * c.num_warps)
        return block_f <= width and tile <= tile_per_thread

    return [c for c in configs if fits(c)] or [
        min(configs, key=lambda c: c.kwargs["BLOCK_E"])
    ]


def tile_autotune(blocks, warps=(2, 4, 8), tile_per_thread=16):
    """``triton.autotune`` over ``blocks`` x ``warps`` for kernels that loop
    over ``[BLOCK_E, BLOCK_F]`` tiles of edges and features ``F``, keyed on
    ``F`` and pruned by :func:`_fitting_tiles`."""
    return triton.autotune(
        configs=[
            triton.Config(dict(block), num_warps=w)
            for block in blocks
            for w in warps
        ],
        key=["F"],
        prune_configs_by={
            "early_config_prune": functools.partial(
                _fitting_tiles, tile_per_thread=tile_per_thread
            )
        },
    )
