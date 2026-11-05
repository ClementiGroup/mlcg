Data conventions
================

Every model in ``mlcg`` reads its input from an
:class:`~mlcg.data.atomic_data.AtomicData` object. This page lists the
fields a model uses, their shapes and the conventions they follow.

.. warning::

   ``mlcg`` does not verify that the input follows these conventions. In
   particular, it can't tell a cell given in the wrong layout (lattice
   vectors as columns instead of rows) from a valid one in general. The
   user is responsible for providing data in the layout described here.
   The representation of cells may change in future versions, for example
   to support other cell types: check this page when updating ``mlcg``.

Fields
------

A batch of ``B`` structures with ``N`` beads (atoms) in total:

.. list-table::
   :header-rows: 1
   :widths: 18 18 64

   * - field
     - shape, dtype
     - meaning
   * - ``pos``
     - ``[N, 3]`` float
     - Cartesian positions, in the length unit the model was trained with.
       Cells and cutoffs use the same unit.
   * - ``atom_types``
     - ``[N]`` int
     - Atomic number, or the bead type of the coarse-grained mapping.
   * - ``batch``
     - ``[N]`` int
     - Index of the structure each bead belongs to, from ``0`` to
       ``B - 1``, sorted: the beads of a structure are contiguous.
   * - ``ptr``
     - ``[B + 1]`` int
     - Offsets of the structures: the beads of structure ``b`` are
       ``ptr[b]:ptr[b + 1]``.
   * - ``cell``
     - ``[B, 3, 3]`` float, optional
     - Unit cell of each structure, **lattice vectors as rows** (see below).
   * - ``pbc``
     - ``[B, 3]`` bool, optional
     - Whether each structure is periodic along each lattice vector.
   * - ``neighbor_list``
     - dict, optional
     - Precomputed neighbour lists, see
       :func:`~mlcg.neighbor_list.neighbor_list.make_neighbor_list`.
   * - ``out``
     - dict
     - Filled by the models with their predictions (energies, forces).

``batch`` and ``ptr`` are created when structures are collated into a
batch (for example by a PyTorch Geometric ``DataLoader`` or by a
simulation). Structures in a batch never interact: no network or prior term
connects beads of different structures.

Periodic cells
--------------

The rows of ``cell[s]`` are the lattice vectors ``a``, ``b``, ``c`` of
structure ``s``:

.. code:: python

   cell[s] = torch.stack([a, b, c])  # a = cell[s, 0], b = cell[s, 1], c = cell[s, 2]

A periodic image of a bead is its position plus ``n @ cell[s]`` for an
integer vector ``n``. This is the layout of ASE (``Atoms.get_cell()``),
OpenMM (``getPeriodicBoxVectors()``, after conversion to the model's unit)
and MDTraj (``unitcell_vectors``, also after unit conversion). A cell
stored with the lattice vectors as columns must be transposed. For an
orthorhombic (diagonal) box both layouts are the same.

Range of the interactions
~~~~~~~~~~~~~~~~~~~~~~~~~

- The standard neighbour search (:mod:`mlcg.neighbor_list`) includes every
  periodic image within the cutoff.
- The fast inference path (flash models with filter tables, see
  :func:`~mlcg.nn.flash_models.optimize_for_inference`) uses the
  **minimum image** convention: each bead interacts with at most one image
  of every other bead. This is exact only if every periodic width of the
  cell is at least **twice the cutoff**. That path also requires the cell
  in the reduced form used by OpenMM: ``a = (a_x, 0, 0)``,
  ``b = (b_x, b_y, 0)`` and ``c = (c_x, c_y, c_z)``, with
  ``|b_x| <= a_x / 2``, ``|c_x| <= a_x / 2`` and ``|c_y| <= b_y / 2``.
  An unsupported cell raises ``ValueError`` when the model is prepared, and
  a cell that becomes unsupported during a run (for example a box shrunk
  by a barostat) makes the energy NaN.

These checks exist to protect the fast path from wrong results. They are
not a validation of the cell layout: see the warning above.
