"""
Flash paths (fused kernels, inference tables) as runtime state of the
standard mlcg modules.

A module with a flash path inherits :class:`SupportsFlash`. There are no
separate flash classes: a ``StandardSchNet`` with its flash path on is
still a ``StandardSchNet``, with the same parameters and state dict.

Using the optimizations
-----------------------

The flash path is off by default. The entry points switch it on for every
module of a model that supports it, at any depth (priors and networks
inside ``SumOut``/``GradientsOut``, or inside a user's own container):

- inference (MD): :func:`~mlcg.nn.flash_models.optimize_for_inference`
  returns a fast copy of the model (flash kernels, inference state such as
  SchNet filter tables, eval mode, optionally ``torch.compile`` and a CUDA
  graph). The model passed in is not modified::

      from mlcg.nn.flash_models import optimize_for_inference

      fast = optimize_for_inference(model, data)  # data: a representative batch
      data = fast(data)                           # energies and forces in data.out

  In an mlcg simulation the same is selected with
  ``convert_to_flash=True`` (together with ``specialize_priors=True``).
- training: :func:`enable_flash_kernels` switches on the exact fused
  kernels, which are trainable (double backward for force matching) and
  give the same results as the standard path::

      model = enable_flash_kernels(model)

Saving is unaffected: ``torch.save`` of a model with its flash path on
writes the standard model, which loads in any mlcg version, also one
without flash support.

Adding a flash path to a module
-------------------------------

The state of a flash path is *runtime state*:

- it is off by default, through class-level defaults, so modules created
  or pickled before the flash path existed work unchanged;
- ``copy.deepcopy`` keeps it;
- ``train()`` and ``load_state_dict`` drop the inference state (it was
  built from the old weights);
- pickling (``torch.save``) drops all of it.

To give a module a flash path:

1. Inherit :class:`SupportsFlash` *before* the ``torch.nn.Module`` base,
   so that its ``train`` and pickling hooks run first (a class with the
   wrong order raises ``TypeError`` when it is created).
2. Create the flash state only in :meth:`~SupportsFlash.enable_kernels` and
   :meth:`~SupportsFlash.prepare_inference`, never in ``__init__``, and
   list every name it uses (instance attributes, buffers registered with
   ``persistent=False``, submodules) in ``_runtime_attributes``. Read it
   with a default (a class attribute, or ``self._buffers.get(name)``).
3. Put the flash computation in ``flash_forward`` and call it from
   ``forward`` when the kernels are on; the standard code stays as it is.
   The kernels must reproduce the standard path to float32 accuracy, and
   be differentiable twice if they are used in training.
4. :meth:`~SupportsFlash.enable_kernels` decides which configurations the
   class supports and refuses the others (``_kernels`` stays off, with a
   warning). A model checks its own submodules: e.g.
   :class:`~mlcg.nn.schnet.StandardSchNet` needs every block to have a
   flash path. A subclass with another structure, or another ``forward``,
   overrides ``enable_kernels`` and ``flash_forward`` for its structure
   (e.g. :class:`~mlcg.nn.schnet.RBFRegularizedSchNet`). Overrides call
   ``super().enable_kernels(enabled)``, which refuses a class that
   overrides ``forward`` without its own ``flash_forward``: the inherited
   flash path would compute the parent's model.
5. :meth:`~SupportsFlash.prepare_inference` raises ``ValueError`` for an
   input it can't handle at all, and warns and keeps the exact path when
   only the inference state is unavailable.

mlcg's own modules keep their flash implementation in a companion mixin,
``<Class>FlashPath(SupportsFlash)`` in ``nn/flash_models/``
(:mod:`mlcg.nn.flash_models.schnet`, :mod:`mlcg.nn.flash_models.prior`),
listed first in the module's bases.
The standard file keeps only the flash interface: the ``flash=`` keyword
of blocks, ``has_flash_path``, and, in a class that defines ``forward``
itself, the hand-off of rule 3 (otherwise the companion's ``forward``
dispatches). The companion's ``flash_forward`` counts as the class's own
in the check of rule 4. A single class, as in the example below, works the
same.

A container module needs nothing: the entry points visit all registered
submodules. Blocks take their flash path when the parent passes it as
``flash=`` (:class:`~mlcg.nn.schnet.InteractionBlock`,
:class:`~mlcg.nn.schnet.CFConv`); a model outside mlcg calls them with
``flash=`` on a destination CSR of a symmetric neighbour list
(:func:`~mlcg.nn.kernels.csr_from_edge_index`). Example of a module with
both a training-safe kernel and an inference-only table::

    import torch
    from mlcg.nn.flash_models import SupportsFlash

    class Envelope(SupportsFlash, torch.nn.Module):
        _runtime_attributes = ("_kernels", "_table")

        def __init__(self, width: int):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(width))

        def prepare_inference(self, data, num_knots=None, tolerance=1e-5):
            if self._kernels:
                table = tabulate(self.weight, num_knots, tolerance)
                self.register_buffer("_table", table, persistent=False)

        def drop_inference(self):
            self._buffers.pop("_table", None)

        def forward(self, r):
            if self._kernels:
                return self.flash_forward(r)
            return torch.exp(-self.weight * r[:, None])

        def flash_forward(self, r):
            table = self._buffers.get("_table")
            if table is not None and not self.training:
                return lookup(table, r)              # inference only
            return fused_envelope(self.weight, r)    # exact, trainable

``tabulate``, ``lookup`` and ``fused_envelope`` stand for the module's own
kernels; :class:`~mlcg.nn.schnet.StandardSchNet`,
:class:`~mlcg.nn.schnet.RBFRegularizedSchNet` and
:class:`~mlcg.nn.schnet.CFConv`, with their companions in
:mod:`mlcg.nn.flash_models.schnet`, are complete examples.
"""

import copy
import warnings
from typing import Optional

import torch

TABLE_TOLERANCE = 1e-5
MIN_TOLERANCE = 1e-7  # float32 storage and arithmetic floor
MAX_NUM_KNOTS = 16384
FIRST_NUM_KNOTS = 2048


def check_tolerance(tolerance: float):
    """Raises ``ValueError`` for a table tolerance below what float32
    tables can reach."""
    if tolerance < MIN_TOLERANCE:
        raise ValueError(
            f"Table tolerance {tolerance:.0e} is below the float32 floor "
            f"{MIN_TOLERANCE:.0e}"
        )


class SupportsFlash:
    """Mixin for modules with a flash path (see the module documentation).

    Attributes
    ----------
    _runtime_attributes:
        Names of the instance attributes, buffers and submodules that hold
        the flash state. They are never pickled. Subclasses that add state
        extend it, keeping ``"_kernels"``.
    """

    _kernels = False
    _runtime_attributes = ("_kernels",)

    def enable_kernels(self, enabled: bool = True):
        """Switches the exact fused kernels (trainable, double backward) on
        or off. Refuses, with a warning, a class that overrides ``forward``
        without its own ``flash_forward``. Overrides refuse the
        configurations they don't support, then call this method."""
        if enabled and not self._flash_forward_is_current():
            warnings.warn(
                f"{type(self).__name__} has no flash path: it overrides "
                "forward without a flash_forward"
            )
            return
        self._kernels = enabled

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        mro = cls.__mro__
        if torch.nn.Module in mro and mro.index(torch.nn.Module) < mro.index(
            SupportsFlash
        ):
            raise TypeError(
                f"{cls.__name__}: SupportsFlash must come before "
                "torch.nn.Module in the bases, or its train and state-dict "
                "hooks don't run"
            )

    def _flash_forward_is_current(self) -> bool:
        """Whether ``flash_forward`` (if the class has one) is defined by the
        class that defines ``forward``, by its flash companion (a direct base
        that is not a module) or by a subclass of it."""

        def owner(name):
            return next(c for c in type(self).__mro__ if name in vars(c))

        if not hasattr(type(self), "flash_forward"):
            return True
        forward, flash_forward = owner("forward"), owner("flash_forward")
        companion = flash_forward in forward.__bases__ and not issubclass(
            flash_forward, torch.nn.Module
        )
        return companion or issubclass(flash_forward, forward)

    def prepare_inference(
        self,
        data,
        num_knots: Optional[int] = None,
        tolerance: float = TABLE_TOLERANCE,
    ):
        """Builds the inference-only state (e.g. filter tables) from the
        current weights and a representative ``data``, if the module has
        one; called after :meth:`enable_kernels`. Weights changed afterwards
        need another call."""

    def drop_inference(self):
        """Removes the state of :meth:`prepare_inference`."""

    def train(self, mode: bool = True):
        if mode:
            self.drop_inference()
        return super().train(mode)

    def _load_from_state_dict(self, *args, **kwargs):
        self.drop_inference()
        super()._load_from_state_dict(*args, **kwargs)

    def __getstate__(self):
        state = super().__getstate__()
        names = set(self._runtime_attributes)
        for key in ("_buffers", "_modules"):
            state[key] = type(state[key])(
                (k, v) for k, v in state[key].items() if k not in names
            )
        state["_non_persistent_buffers_set"] = (
            state["_non_persistent_buffers_set"] - names
        )
        for name in names:
            state.pop(name, None)
        return state

    def __deepcopy__(self, memo):
        # bypasses __getstate__, which drops the flash state
        copied = type(self).__new__(type(self))
        memo[id(self)] = copied
        copied.__setstate__(copy.deepcopy(self.__dict__, memo))
        return copied


def enable_flash_kernels(
    model: torch.nn.Module, enabled: bool = True
) -> torch.nn.Module:
    """Switches the flash kernels of every :class:`SupportsFlash` module of
    ``model`` (at any depth) in place, and returns ``model``."""
    for module in model.modules():
        if isinstance(module, SupportsFlash):
            module.enable_kernels(enabled)
    return model
