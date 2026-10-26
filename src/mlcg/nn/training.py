"""
Fast force training with any mlcg model: :func:`optimize_for_training`.

Force matching differentiates the forces ``-dE/dpos`` once more, and
``torch.compile`` can't compile through such a double backward. As in
NequIP, :class:`CompiledTraining` traces the energy *and* the inner
``autograd.grad`` into one graph (``make_fx``, symbolic sizes) and compiles
it: the forces are ordinary operations of the graph, and the loss needs only
a first-order backward. Neighbour lists are built eagerly, outside the graph.
"""

import copy
import warnings
from typing import Callable, Tuple

import torch
from torch.fx.experimental import _config as fx_config
from torch.fx.experimental.proxy_tensor import make_fx
from torch.func import functional_call
from torch.utils import _pytree as pytree

from ..data.atomic_data import AtomicData
from .flash_models.converter import convert_standard_model_to_flash

CHECK_TOLERANCE = 1e-2  # catches missing terms; TF32 differences are ~1e-3
_NOT_INPUTS = ("out", "neighbor_list")
# this rewrite gives the in-place one-hot of e3nn/MACE an integer dtype
INDUCTOR_OPTIONS = {"optimize_scatter_upon_const_tensor": False}


def optimize_for_training(
    model: torch.nn.Module, flash: bool = False, compile: bool = False
) -> Tuple[torch.nn.Module, Callable[[AtomicData], AtomicData]]:
    """Returns ``(model, forward)``: train and checkpoint ``model``
    (flash-converted if ``flash``, with unchanged state-dict keys) and call
    ``forward(data)`` in the training step (``model`` itself, or a
    :class:`CompiledTraining` of it if ``compile``)."""
    if flash:
        model = convert_standard_model_to_flash(model)
    return model, CompiledTraining(model) if compile else model


def precompute_neighbor_lists(model: torch.nn.Module, data: AtomicData):
    """Fills ``data.neighbor_list`` for every network of ``model`` that
    reuses a compatible one, so no neighbour search runs in the graph."""
    neighbor_lists = dict(data.get("neighbor_list") or {})
    for module in model.modules():
        if not hasattr(module, "is_nl_compatible") or module.is_nl_compatible(
            neighbor_lists.get(module.name)
        ):
            continue
        cutoff = getattr(module, "r_max", None)
        if cutoff is None:
            cutoff = module.rbf_layer.cutoff.cutoff_upper
        neighbor_lists.update(
            module.neighbor_list(data, cutoff, module.max_num_neighbors)
        )
    data.neighbor_list = neighbor_lists


def _flatten(data: AtomicData):
    """Tensor inputs of ``data``, and a key for everything else."""
    fields = {k: data[k] for k in data.keys() if k not in _NOT_INPUTS}
    leaves, spec = pytree.tree_flatten((fields, data.neighbor_list))
    tensors = [x for x in leaves if torch.is_tensor(x)]
    constants = tuple(None if torch.is_tensor(x) else x for x in leaves)
    return tensors, (spec, constants)


def _unflatten(key, tensors):
    spec, constants = key
    tensors = iter(tensors)
    leaves = [next(tensors) if c is None else c for c in constants]
    return pytree.tree_unflatten(leaves, spec)


def _strip_detach(graph_module: torch.fx.GraphModule):
    """``make_fx`` detaches the outputs that the inner backward saves (tanh,
    exp, norm, ...); the force loss must differentiate through them."""
    for node in list(graph_module.graph.nodes):
        if node.target is torch.ops.aten.detach.default:
            node.replace_all_uses_with(node.args[0])
            graph_module.graph.erase_node(node)
    graph_module.recompile()


def _relative_error(x, y):
    return float((x - y).norm() / y.norm().clamp_min(1e-30))


class CompiledTraining:
    """``model(data)`` for force training, from one compiled graph of energy
    and forces with the parameters as inputs (sizes of nodes, edges and
    graphs are dynamic). The first call traces, compiles and checks outputs
    and parameter gradients against eager, raising on a mismatch.

    Runs ``model`` eagerly in eval mode or without grad, for a batch the
    graph's size guards reject (e.g. a single graph), and for good if the
    model can't be traced (with a warning).
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.graph = None
        self.traceable = True
        self.compiled_calls = 0

    def _state(self):
        return dict(self.model.named_parameters()) | dict(
            self.model.named_buffers()
        )

    def __call__(self, data: AtomicData) -> AtomicData:
        if not (
            self.traceable and self.model.training and torch.is_grad_enabled()
        ):
            return self.model(data)
        precompute_neighbor_lists(self.model, data)
        tensors, key = _flatten(data)
        state = list(self._state().values())
        if self.graph is None:
            self._compile(data, state, tensors, key)
            if not self.traceable:
                return self.model(data)
        if key != self.key or not self.shape_env.evaluate_guards_expression(
            self.guards, state + tensors
        ):
            return self.model(data)
        data.out = pytree.tree_unflatten(
            self._run(state, tensors), self.out_spec
        )
        return data

    def _run(self, state, tensors):
        self.compiled_calls += 1
        return self.graph(*state, *tensors)

    def _traced(self, data):
        """The traced function; it runs a copy of the model, since tracing
        may leave fake tensors in caches (e.g. e3nn's generated code)."""
        names = list(self._state())
        model = copy.deepcopy(self.model)

        def energy_and_forces(*args):
            state, tensors = args[: len(names)], args[len(names) :]
            fields, neighbor_lists = _unflatten(self.key, tensors)
            sample = copy.copy(data)
            for name, value in fields.items():
                sample[name] = value
            sample.neighbor_list = neighbor_lists
            sample.out = {}
            out = functional_call(model, dict(zip(names, state)), (sample,)).out
            leaves, self.out_spec = pytree.tree_flatten(out)
            return leaves

        return energy_and_forces

    def _compile(self, data, state, tensors, key):
        self.key = key
        try:
            with fx_config.patch(use_duck_shape=False):
                graph = make_fx(
                    self._traced(data),
                    tracing_mode="symbolic",
                    _error_on_data_dependent_ops=True,
                    _allow_non_fake_inputs=True,  # constant tensor attributes
                )(*state, *tensors)
            _strip_detach(graph)
            self._set_guards(graph)
            # only graph sizes vary: static weights keep feature sizes constexpr
            for tensor in state:
                torch._dynamo.mark_static(tensor)
            self.graph = torch.compile(
                graph, dynamic=True, fullgraph=True, options=INDUCTOR_OPTIONS
            )
            actual = self._run(state, tensors)
        except Exception as error:
            warnings.warn(
                f"Can't compile the model ({error}); training eagerly"
            )
            self.traceable, self.graph = False, None
            return
        self._check(data, actual)

    def _set_guards(self, graph):
        """Size conditions the trace assumed, checked on every batch."""
        placeholders = [
            n.meta["val"] for n in graph.graph.nodes if n.op == "placeholder"
        ]
        self.shape_env = graph.shape_env
        self.guards = (
            self.shape_env.produce_guards_expression(
                placeholders, ignore_static=False
            )
            or "True"
        )

    def _check(self, data, actual):
        params = [p for p in self.model.parameters() if p.requires_grad]
        reference = copy.copy(data)
        reference.pos = data.pos.detach().clone()
        reference.out = {}
        expected = pytree.tree_leaves(self.model(reference).out)
        for name, outputs in (
            ("outputs", (actual, expected)),
            (
                "parameter gradients",
                [self._grads(o, params) for o in (actual, expected)],
            ),
        ):
            errors = [_relative_error(x, y) for x, y in zip(*outputs)]
            if max(errors, default=0.0) > CHECK_TOLERANCE:
                raise RuntimeError(
                    f"Compiled {name} differ from eager "
                    f"(relative error {max(errors):.2e})"
                )

    @staticmethod
    def _grads(outputs, params):
        loss = sum(
            x.float().square().sum() for x in outputs if x.is_floating_point()
        )
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        return [
            torch.zeros_like(p) if g is None else g
            for g, p in zip(grads, params)
        ]
