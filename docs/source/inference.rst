Faster inference
================

``mlcg`` models can run on fused GPU kernels, the *flash* path, for
molecular dynamics and any other inference. The flash path computes the
same model: the same parameters, the same checkpoints, and energies and
forces equal to the standard code within float32 rounding (and, for the
filter tables, a set tolerance). It needs a CUDA GPU and Triton.

Quick start
-----------

**In an mlcg simulation**, set two options of the simulation (e.g. in the
``simulation`` section of the input file of ``mlcg-nvt_langevin``):

.. code:: yaml

   simulation:
     specialize_priors: true   # priors with per-molecule parameters
     convert_to_flash: true    # flash network, tables, CUDA graph

``compile: true`` can be added; with the CUDA graph it adds little.

**Anywhere else**, one call returns a fast copy of the model. ``data`` is a
representative batch (the structures you will evaluate, already on the
GPU); the model passed in is not modified:

.. code:: python

   from mlcg.nn import optimize_for_inference

   fast = optimize_for_inference(model, data)
   data = fast(data)                  # energies and forces in data.out

**With OpenMM** (``mlcg_to_openmm``), where OpenMM runs the priors and the
network is a ``PythonForce``, pass ``flash=True`` (e.g.
``add_network_force(system, network, data, device="cuda", flash=True)``).
It calls :func:`~mlcg.nn.optimize_for_inference`.

**In training**, :func:`~mlcg.nn.enable_flash_kernels` switches on the
exact fused kernels, which are trainable (double backward for force
matching). They save memory; the filter tables, the CUDA graph and the
neighbour search above are inference only.

.. code:: python

   from mlcg.nn import enable_flash_kernels

   model = enable_flash_kernels(model)   # in place

What it does
------------

:func:`~mlcg.nn.optimize_for_inference` deep-copies the model, switches on
the flash path of every module that has one (at any depth: networks and
priors inside ``SumOut``/``GradientsOut`` or your own container), builds
the inference state, freezes the copy in eval mode and captures a CUDA
graph.

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - feature
     - what it does
     - when it doesn't apply
   * - filter tables
     - The SchNet filter network (RBF expansion, MLP, cutoff) is a function
       of the distance only. It is tabulated once, from the current
       weights, as a cubic Hermite table with a relative error below
       ``tolerance`` (default 1e-5), on the fewest knots that reach it. The
       step then reads the table instead of evaluating the filter network
       on every edge, which removes most of the GPU time and memory.
     - A filter that can't meet the tolerance (e.g. ReLU) keeps its filter
       network for that block, with a warning. ``RBFRegularizedSchNet`` has
       no tables: its filters depend on the bead types.
   * - fused message passing
     - The continuous-filter convolution runs as one kernel per row of the
       neighbour list, without per-edge messages in memory. Trainable.
     - ``CFConv`` with ``aggr`` other than ``"add"``.
   * - sync-free neighbour search
     - The radius graph is built on the GPU directly in the kernels'
       format, with a capacity reserved once, so a step never waits for
       the CPU. Periodic cells use the minimum image (see
       :doc:`data/conventions`).
     - A periodic box narrower than twice the cutoff raises when the model
       is prepared.
   * - CUDA graph
     - One energy and force evaluation is captured and replayed at every
       step, which removes the launch and CPU cost that dominates small
       systems. Only positions (and the cell) may change between calls.
     - A model that synchronizes with the host (e.g. per-edge tensors in a
       model without tables, or a very large system in the neighbour
       search's checked mode) runs without the graph, with a warning.
   * - prior kernels
     - Harmonic bonds and angles, repulsion and dihedrals use fused
       kernels. With ``specialize_priors`` (simulations) their parameters
       are also gathered per interaction once, for the molecule simulated.
     - Periodic data runs the standard prior code; other priors keep
       their standard code.

Measured on an RTX 3090 with the production 5-bead and Cα SchNet models:
inside OpenMM, the step with ``flash=True`` is 4.3–6.5× faster than with
the standard network compiled, and the network uses 3–30× less GPU memory,
so correspondingly more copies fit on one GPU.

Which models
------------

- ``StandardSchNet``: all of the above, if every interaction block is an
  :class:`~mlcg.nn.schnet.InteractionBlock` with a ``CFConv``
  (``aggr="add"``); otherwise the model keeps its standard path, with a
  warning.
- ``RBFRegularizedSchNet``: fused message passing on the sync-free
  neighbour search, without tables, and therefore without the CUDA graph
  (its per-edge tensors need one host synchronization per step).
- Priors: as in the table.
- Other modules run their standard code; a model can mix both (e.g. a
  ``SumOut`` of a flash SchNet and priors without kernels).

Accuracy and saving
-------------------

- Forces and energies match the standard model within float32 rounding;
  with tables, within the table tolerance (``tolerance`` argument of
  :func:`~mlcg.nn.optimize_for_inference`).
- Under TF32 matrix products (``torch.set_float32_matmul_precision("high")``),
  two runs of the same model can differ by ~1e-4 relative, flash or not.
- The flash state is never saved. ``torch.save`` of an optimized model
  writes the standard model, which loads in any ``mlcg`` version, also one
  without flash support. ``train()`` and ``load_state_dict`` drop the
  tables, which are built from the weights at the time of the call: after
  changing the weights, optimize again.

Advanced: a flash path for your own model
-----------------------------------------

The flash path of a module is runtime state of the module itself, given by
the :class:`~mlcg.nn.SupportsFlash` mixin; there are no separate flash
classes. mlcg keeps the flash implementation of a model in a companion
mixin, ``<Model>FlashPath(SupportsFlash)`` in ``nn/flash_models/``, which
the model lists first in its bases; the model's own file keeps only the
standard code and the flash interface. A model with a flash path:

1. inherits :class:`~mlcg.nn.SupportsFlash` (through its companion, or
   directly) before ``torch.nn.Module``, or a class that does (e.g.
   ``StandardSchNet``); the wrong order raises ``TypeError``;
2. implements ``enable_kernels``: switch on what it supports, refuse
   (with a warning) what it doesn't, then call
   ``super().enable_kernels(enabled)``;
3. implements ``flash_forward``, and ``forward`` calls it when the kernels
   are on: in the companion if the model inherits its standard
   ``forward``, otherwise as the first two lines of the model's own
   ``forward``;
4. implements ``prepare_inference`` if it has inference state (tables).

A subclass that overrides ``forward`` must also override
``flash_forward``: otherwise the inherited flash path would compute the
parent's model, and ``enable_kernels`` refuses it.
``StandardSchNet`` and ``RBFRegularizedSchNet``, with their companions in
:mod:`mlcg.nn.flash_models.schnet`, are complete examples; the full rules
are in :mod:`mlcg.nn.flash_models.mixin`.

**Using the SchNet blocks outside mlcg.** ``InteractionBlock`` and
``CFConv`` take the flash path when called with ``flash=``: a destination
CSR of a *symmetric* neighbour list (every edge also listed in reverse)
and the edge features in its row order:

.. code:: python

   from mlcg.nn import csr_from_edge_index

   row_ptr, col, order = csr_from_edge_index(edge_index, num_nodes)
   features = (distances[order], rbf[order])
   graph = (row_ptr, col, features, pos, batch, cell, pbc, cutoff)
   x = x + block(x, flash=graph)

``pos``, ``batch``, ``cell``, ``pbc`` and ``cutoff`` are read only by a
block with a table in use (its ``features`` are ``None``); with the exact
kernels they can be ``None``. :class:`~mlcg.nn.RadiusCSR` builds this
graph from positions without host synchronization, as ``StandardSchNet``
does.

API
---

.. autofunction:: mlcg.nn.optimize_for_inference

.. autofunction:: mlcg.nn.enable_flash_kernels

.. autoclass:: mlcg.nn.SupportsFlash
   :members: enable_kernels, prepare_inference, drop_inference

.. autoclass:: mlcg.nn.RadiusCSR

.. autofunction:: mlcg.nn.csr_from_edge_index
