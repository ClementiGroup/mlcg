"""
One entry point for fast inference (MD) with an mlcg model:
:func:`optimize_for_inference`, built from :func:`prepare_for_inference`
(flash kernels and inference state of every :class:`SupportsFlash`
module, eval mode) and :func:`capture_cuda_graph`.
"""

import copy
import math
import statistics
import time
import warnings
from contextlib import contextmanager
from typing import Callable, Dict, Optional, Tuple, Union

import torch

from ...data.atomic_data import AtomicData
from ...data._keys import CELL_KEY, PBC_KEY, POSITIONS_KEY
from .mixin import (
    TABLE_TOLERANCE,
    SupportsFlash,
    check_tolerance,
    enable_flash_kernels,
)

# With inductor's on-disk caches, a process reuses a graph compiled for an
# earlier system size, and the step runs up to ~13% slower than a cold
# compile (the exact cause is not known; it is not the autotuned tiles). Both
# caches go: the FX graph cache is the one that matters.
COMPILE_OPTIONS = {"autotune_local_cache": False, "fx_graph_cache": False}
# "auto" keeps torch.compile only if it is at least this much faster
AUTO_MARGIN = 0.03
FEATURES = ("flash", "compile", "cuda_graph")


def prepare_for_inference(
    model: torch.nn.Module,
    data: AtomicData,
    num_knots: Optional[int] = None,
    tolerance: float = TABLE_TOLERANCE,
) -> torch.nn.Module:
    """A copy of ``model`` with the flash kernels and the inference state
    (e.g. SchNet filter tables) of every :class:`SupportsFlash` module,
    frozen in eval mode; ``model`` itself is unchanged. Raises
    ``ValueError`` for a periodic cell a module can't handle (see
    :func:`~mlcg.nn.kernels.pbc.check_cells`). One forward on ``data`` fills
    ``data.out``, so that compiling it later traces fewer graphs."""
    check_tolerance(tolerance)
    model = enable_flash_kernels(copy.deepcopy(model))
    for module in model.modules():
        if isinstance(module, SupportsFlash):
            module.prepare_inference(data, num_knots, tolerance)
    model.eval().requires_grad_(False)
    model(data)
    return model


def capture_cuda_graph(
    model: Callable, data: AtomicData, warn: bool = True
) -> Optional["GraphedModel"]:
    """A :class:`GraphedModel` of ``model`` on ``data``, or ``None`` if it
    can't be captured (with a warning, if ``warn``) or ``data`` is not on a
    GPU."""
    if data[POSITIONS_KEY].device.type != "cuda":
        return None
    try:
        graphed = GraphedModel(model, data)
    except Exception as e:
        if warn:
            warnings.warn(
                f"CUDA graph capture failed ({e}); running without it."
            )
        graphed = None
    # frees the warm-up's cached blocks and a failed capture's pool
    torch.cuda.empty_cache()
    return graphed


def optimize_for_inference(
    model: torch.nn.Module,
    data: AtomicData,
    optimize: Union[None, str, Dict[str, bool]] = "auto",
    num_knots: Optional[int] = None,
    tolerance: float = TABLE_TOLERANCE,
    margin: float = AUTO_MARGIN,
) -> Tuple[Callable[[AtomicData], AtomicData], Optional[Dict[str, float]]]:
    """Every inference speed-up in one call, as set by ``optimize``:

    - ``"auto"`` (default): the flash kernels (:func:`prepare_for_inference`)
      and a CUDA graph of the energy + force evaluation
      (:func:`capture_cuda_graph`), and ``torch.compile`` only if it is
      measured faster on ``data`` on this GPU: the eager and the compiled
      model are each timed for about a second, and the compiled one is kept
      if it is at least ``margin`` faster. A model that doesn't compile runs
      eager, with a warning;
    - a dict of ``flash``, ``compile`` and ``cuda_graph``, a missing key
      meaning on (e.g. ``{"compile": False}``). Without ``flash`` the model
      itself is used, not a copy. Raises ``RuntimeError`` if the model
      doesn't compile;
    - ``None``: the model as given.

    ``data`` is a representative input, already on the model's device.
    Returns ``model(data) -> data`` and, for ``"auto"``, a report:
    ``compiled`` (the choice), ``eager_ms`` and ``compiled_ms`` (one call;
    NaN if the compile failed), ``timing_s`` (spent timing) and ``seconds``
    (the whole call, flash preparation and compile included); ``None``
    otherwise.
    """
    if optimize is None:
        return model, None
    if optimize == "auto":
        return _auto(model, data, num_knots, tolerance, margin)
    unknown = set(optimize) - set(FEATURES)
    if unknown:
        raise ValueError(
            f"Unknown `optimize` keys {sorted(unknown)}, use {FEATURES}"
        )
    on = {key: optimize.get(key, True) for key in FEATURES}
    if on["flash"]:
        model = prepare_for_inference(model, data, num_knots, tolerance)
    else:
        # fills data.out, so that compiling traces fewer graphs
        model(data)
    if on["compile"]:
        model = _compile(model, data)
    graphed = capture_cuda_graph(model, data) if on["cuda_graph"] else None
    return graphed or model, None


def describe_optimization(
    optimize: Union[None, str, Dict[str, bool]],
    report: Optional[Dict[str, float]],
    cuda_graph: bool,
) -> str:
    """One line for a log on what :func:`optimize_for_inference` selected:
    the ``optimize`` option, what ``"auto"`` chose with both timings (its
    ``report``), and whether a CUDA graph is used."""
    text = f"Model optimized with optimize={optimize!r}"
    if report is not None:
        kept = "kept" if report["compiled"] else "not kept"
        text += (
            f": torch.compile {kept} (eager {report['eager_ms']:.3f} ms, "
            f"compiled {report['compiled_ms']:.3f} ms per call)"
        )
    return f"{text}; CUDA graph {'on' if cuda_graph else 'off'}"


def _auto(model, data, num_knots, tolerance, margin):
    start = time.perf_counter()
    model = prepare_for_inference(model, data, num_knots, tolerance)
    eager = capture_cuda_graph(model, data)
    # warn once: what fails to capture eager fails compiled for the same reason
    warn = eager is not None
    eager = eager or model
    eager_ms, timing_s = _ms_per_call(eager, data)
    # one graph at a time: two private memory pools may not fit
    del eager
    torch.cuda.empty_cache()
    try:
        compiled = _compile(model, data)
        compiled = capture_cuda_graph(compiled, data, warn) or compiled
        compiled_ms, compiled_timing_s = _ms_per_call(compiled, data)
        timing_s += compiled_timing_s
    except Exception as e:
        warnings.warn(f"Running without torch.compile: {e}")
        compiled, compiled_ms = None, math.nan
    use_compiled = compiled_ms < (1 - margin) * eager_ms
    if use_compiled:
        chosen = compiled
    else:
        compiled = None
        torch.cuda.empty_cache()
        chosen = capture_cuda_graph(model, data, warn) or model
    report = {
        "compiled": use_compiled,
        "eager_ms": eager_ms,
        "compiled_ms": compiled_ms,
        "timing_s": timing_s,
        "seconds": time.perf_counter() - start,
    }
    return chosen, report


def _compile(model, data):
    model = torch.compile(model, dynamic=True, options=COMPILE_OPTIONS)
    try:
        model(data)
    except Exception as e:
        raise RuntimeError(
            f"torch.compile failed on this model: {e}. "
            "Optimize it without compile."
        ) from e
    return model


def _ms_per_call(model, data, chunk_s=0.2, chunks=3):
    """Median time (ms) of one ``model(data)`` over ``chunks`` timed runs
    of at least ``chunk_s`` each (the calls per run double until one is),
    and the seconds spent."""
    start = time.perf_counter()
    calls = 1
    while (elapsed := _seconds(model, data, calls)) < chunk_s:
        calls *= 2
    runs = [elapsed] + [_seconds(model, data, calls) for _ in range(chunks - 1)]
    return 1e3 * statistics.median(runs) / calls, time.perf_counter() - start


def _seconds(model, data, calls):
    on_gpu = data[POSITIONS_KEY].device.type == "cuda"
    if on_gpu:
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(calls):
        model(data)
    if on_gpu:
        torch.cuda.synchronize()
    return time.perf_counter() - start


def _copy_outputs(out):
    """Copies of the outputs a caller reads, since the graph overwrites its
    buffers at the next replay: the totals of a ``SumOut`` (top-level
    tensors), or a single model's own outputs. Per-model entries of a
    ``SumOut`` stay views: a copy each costs ~10 launches per step, +50% of
    a small system's step."""
    if any(torch.is_tensor(value) for value in out.values()):
        return {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in out.items()
        }
    return {
        key: _copy_outputs(value) if isinstance(value, dict) else value
        for key, value in out.items()
    }


@contextmanager
def _fail_on_host_sync():
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(previous)


GRAPH_INPUTS = (POSITIONS_KEY, CELL_KEY, PBC_KEY)


class GraphedModel:
    """Replays a CUDA graph of one ``model(data)`` call (energies and forces).

    The graph is captured on a shallow copy of ``data`` with its own buffers
    for the inputs that change during a run: positions and, if periodic, the
    cell and ``pbc`` (a barostat rescales the box). Each call copies them
    from the given ``data`` and replays, so ``data`` may be a new object.
    Every other input (atom types, batch, ``ptr``, ...) must stay the same
    for the whole simulation. A periodic call to a graph captured without a
    cell, or the reverse, raises ``ValueError``. The model must not
    synchronize with the host: the first warm-up call raises if it does,
    before any capture is tried.
    """

    def __init__(self, model: torch.nn.Module, data: AtomicData, warmup=3):
        self.model = model
        self.data = copy.copy(data)
        self.inputs = {
            key: data[key].detach().clone()
            for key in GRAPH_INPUTS
            if data.get(key) is not None
        }
        for key, buffer in self.inputs.items():
            self.data[key] = buffer
        # data.out keeps the structure the model was compiled with, otherwise
        # Dynamo's guards on it fail and every submodule recompiles.
        self.data.out = {
            key: dict(value) if isinstance(value, dict) else value
            for key, value in data.out.items()
        }
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            with _fail_on_host_sync():
                model(self.data)
            for _ in range(warmup - 1):
                model(self.data)
        torch.cuda.current_stream().wait_stream(stream)
        # the graph's private memory pool cannot reuse blocks cached by the
        # warm-up, so release them before capturing
        torch.cuda.empty_cache()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = model(self.data).out

    def __call__(self, data: AtomicData) -> AtomicData:
        if (data.get(CELL_KEY) is None) == (CELL_KEY in self.inputs):
            raise ValueError(
                "The CUDA graph was captured "
                f"{'with' if CELL_KEY in self.inputs else 'without'} a "
                "periodic cell; capture a new one for this data"
            )
        with torch.no_grad():
            for key, buffer in self.inputs.items():
                buffer.copy_(data[key])
        self.graph.replay()
        data.out = _copy_outputs(self.out)
        return data
