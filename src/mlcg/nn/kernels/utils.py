import functools
import torch


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
