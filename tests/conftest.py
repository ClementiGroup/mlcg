import contextlib
import functools

import pytest
import torch
from pathlib import Path


def _to_tf32(t):
    """Round float32 values to TF32 (10 mantissa bits), keeping the
    gradient of the identity."""
    if not isinstance(t, torch.Tensor) or t.dtype != torch.float32:
        return t
    bits = t.detach().contiguous().view(torch.int32)
    rounded = ((bits + 0x1000) & ~0x1FFF).view(torch.float32).view(t.shape)
    return t + (rounded - t.detach())


def _tf32_inputs(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*(_to_tf32(a) for a in args), **kwargs)

    return wrapper


@pytest.fixture
def tf32_matmul():
    """Context manager: float32 matmul precision "high", with the inputs of
    `torch.matmul`, `mm`, `bmm`, `@` and `einsum` rounded to TF32 as cuBLAS
    does on GPUs that use TF32 for every shape (e.g. Blackwell), on any
    device. Code that must stay exact under `"high"` should not change."""

    @contextlib.contextmanager
    def enabled():
        precision = torch.get_float32_matmul_precision()
        einsum = torch.einsum
        with pytest.MonkeyPatch.context() as mp:
            for name in ("matmul", "mm", "bmm"):
                mp.setattr(torch, name, _tf32_inputs(getattr(torch, name)))
            mp.setattr(
                torch.Tensor,
                "__matmul__",
                _tf32_inputs(torch.Tensor.__matmul__),
            )
            mp.setattr(
                torch,
                "einsum",
                lambda eq, *ops: einsum(eq, *(_to_tf32(o) for o in ops)),
            )
            torch.set_float32_matmul_precision("high")
            try:
                yield
            finally:
                torch.set_float32_matmul_precision(precision)

    return enabled


def pytest_collection_modifyitems(config, items):
    """
    This utility add markers to distinguish between integration, unit, and continuity tests automatically.
    It checks the file paths of the collected tests and assigns the appropriate marker based on their location.
    This allows for easier test selection and organization based on the type of test.

    Run `pytest -m integration` to run only integration tests
    Run `pytest -m unit` to run only unit tests
    Run `pytest -m continuity` to run only continuity tests
    """
    for item in items:
        # Convert nodeid to a path-safe string
        path = Path(str(item.fspath))

        if "tests/integration" in path.as_posix():
            item.add_marker(pytest.mark.integration)
        elif "tests/unit" in path.as_posix():
            item.add_marker(pytest.mark.unit)
        elif "tests/continuity" in path.as_posix():
            item.add_marker(pytest.mark.continuity)
