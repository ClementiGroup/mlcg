"""
One entry point for fast inference (MD) with an mlcg model:
:func:`optimize_for_inference`, built from :func:`prepare_for_inference`
(flash conversion, filter tables, eval mode) and :func:`capture_cuda_graph`.
"""

import copy
import warnings
from contextlib import contextmanager
from typing import Callable, Optional

import torch

from ...data.atomic_data import AtomicData
from ...data._keys import CELL_KEY, PBC_KEY, POSITIONS_KEY
from ..kernels.pbc import check_cells
from .converter import convert_standard_model_to_flash
from .flash_schnet import TABLE_TOLERANCE, FlashSchNet, check_tolerance


def prepare_for_inference(
    model: torch.nn.Module,
    data: AtomicData,
    num_knots: Optional[int] = None,
    tolerance: float = TABLE_TOLERANCE,
) -> torch.nn.Module:
    """Flash-convert ``model``, tabulate the filters of every flash SchNet
    (falling back to its filter networks, with a warning, if they can't be
    tabulated) and freeze it in eval mode. Raises ``ValueError`` for a
    periodic cell the tabulated path can't handle (see
    :func:`~mlcg.nn.kernels.pbc.check_cells`). One forward on ``data`` fills
    ``data.out``, so that compiling it later traces fewer graphs."""
    check_tolerance(tolerance)
    model = convert_standard_model_to_flash(model)
    schnets = [m for m in model.modules() if isinstance(m, FlashSchNet)]
    for schnet in schnets:
        cutoff = float(schnet.rbf_layer.cutoff.cutoff_upper)
        check_cells(data.get("cell"), data.get("pbc"), cutoff)
        try:
            schnet.build_tables(num_knots, tolerance)
        except ValueError as error:
            warnings.warn(f"{error}; {schnet.name} uses its filter networks")
    model.eval().requires_grad_(False)
    model(data)
    return model


def capture_cuda_graph(
    model: Callable, data: AtomicData
) -> Optional["GraphedModel"]:
    """A :class:`GraphedModel` of ``model`` on ``data``, or ``None`` (with a
    warning) if it can't be captured."""
    try:
        graphed = GraphedModel(model, data)
    except Exception as e:
        warnings.warn(f"CUDA graph capture failed ({e}); running without it.")
        graphed = None
    # frees the warm-up's cached blocks and a failed capture's pool
    torch.cuda.empty_cache()
    return graphed


def optimize_for_inference(
    model: torch.nn.Module,
    data: AtomicData,
    num_knots: Optional[int] = None,
    tolerance: float = TABLE_TOLERANCE,
    compile: bool = False,
    cuda_graph: bool = True,
) -> Callable[[AtomicData], AtomicData]:
    """Every inference speed-up in one call: :func:`prepare_for_inference`,
    optionally ``torch.compile``, and a CUDA graph of the energy + force
    evaluation (:func:`capture_cuda_graph`). ``data`` is a representative
    input, already on the model's device; returns ``model(data) -> data``.
    """
    model = prepare_for_inference(model, data, num_knots, tolerance)
    if compile:
        model = torch.compile(model, dynamic=True)
        model(data)
    graphed = capture_cuda_graph(model, data) if cuda_graph else None
    return graphed or model


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
