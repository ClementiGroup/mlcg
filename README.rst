mlcg
==========

.. start-intro

|Docs badge| |License| |Pipeline|

.. |Docs badge| image:: https://img.shields.io/badge/mlcg-docs-blue.svg
   :target: https://clementigroup.github.io/mlcg/

.. |License| image:: https://img.shields.io/github/license/Naereen/StrapDown.js.svg
   :target: https://opensource.org/licenses/MIT

.. |Pipeline| image:: https://git.imp.fu-berlin.de/ag-clementi/mlcg/badges/main/pipeline.svg
   :target: https://git.imp.fu-berlin.de/ag-clementi/mlcg/-/commits/main
   :alt: pipeline status

This repository collects a set of tools to apply machine learning techniques to coarse grain atomic systems.


Installation
------------
.. start-install

Requires **Python 3.12**. Clone the repo:

.. code:: bash

    git clone git@github.com:ClementiGroup/mlcg.git
    cd mlcg

We recommend `uv <https://docs.astral.sh/uv/>`_ for installation.

``uv sync`` creates a ``.venv`` in the repo root and installs all
dependencies from the correct pre-built wheel pages. If you manage your own
virtual environment instead, install reproducibly from one of the provided
per-platform lock files with ``uv pip``. Pick the row matching your
hardware:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Hardware
     - ``uv sync``
     - Self-managed environment
   * - NVIDIA CUDA 13.0
     - ``uv sync --extra cu130``
     - ``uv pip install -r pylock.cu130.toml``
   * - NVIDIA CUDA 12.8
     - ``uv sync --extra cu128``
     - ``uv pip install -r pylock.cu128.toml``
   * - AMD ROCm 7.2
     - ``uv sync --extra rocm72``
     - ``uv pip install -r pylock.rocm72.toml``
   * - CPU only
     - ``uv sync --extra cpu``
     - ``uv pip install -r pylock.cpu.toml``

In a self-managed environment you can also skip the lock file and install
the package directly with the extra matching your hardware:

.. code:: bash

    uv pip install ".[cu130]"

**For developers**

Add ``--group dev`` to install additional development dependencies
(``black``, ``pytest``, ``coverage``):

.. code:: bash

    uv sync --extra cu130 --group dev

or, for a self-managed environment:

.. code:: bash

    uv pip install ".[cu130]" --group dev

**With pip**

Installing with ``pip`` instead of ``uv`` is also possible, but requires
installing dependencies in a specific order to avoid version conflicts
between optional model backends — see `Installation with pip`_ below.

.. end-install



Examples
--------

Please take a look into the ``examples`` folder of the repository to see how to use this code to train a model over an existing dataset.

CLI
====

The models defined in this library can be conveniently trained using the pytorch-lightning
`CLI <https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html>`_ utilities.

.. end-intro

.. start-doc

Documentation
-------------

Documentation is available `here <https://clementigroup.github.io/mlcg/>`_ and here are some references on how to work with it.

Dependencies
-------------

.. code:: bash

    uv sync --group docs


How to build
------------

.. code:: bash

    cd docs
    sphinx-build -b html source build

How to update the online documentation
---------------------------------------

This update should be done after any update of the `main` branch so that the
documentation is synchronized with the main version of the repository.

.. code:: bash

    git checkout gh-pages
    git rebase main
    cd docs
    sphinx-build -b html source ./
    git commit -a
    git push

.. end-doc

Test Coverage
-------------

The test coverage of this library is monitored with `coverage` for each pull request using `GitHub` actions.  
To produce a report locally, run:

.. code-block:: bash

    coverage run -m pytest
    coverage report

This will run the full set of unit and continuity tests in `mlcg` and the larger integration tests under `tests/integration`, 
including training and simulation of all models described in `examples/input_yamls/README.md`.  

For quick local development testing, it is also possible to exclude the large test by running:

.. code-block:: bash

    coverage run -m pytest --light

Troubleshooting
---------------

If your hardware is not listed above, we recommend installing ``torch``, ``torch-cluster``,
and any desired GPU acceleration libraries (``cuequivariance-torch``,
``cuequivariance-ops-torch``, ``nvalchemi-toolkit-ops``, ``openequivariance``)
manually into a uv environment first, then install the package with:

.. code:: bash

    uv pip install -e .

Installation with pip
----------------------

Some optional model backends declare conflicting sub-dependencies (e.g.
``mace-torch`` and ``nequip`` require incompatible versions of ``e3nn``),
which ``uv`` resolves automatically but ``pip`` does not. Installing with
``pip`` therefore requires following the steps below in order: install the
heavy, platform-specific dependencies first (torch, PyTorch Geometric,
equivariant-operation acceleration), then the remaining pure-Python
dependencies, and finally the model backends **without** letting pip resolve
their sub-dependencies.

1. Install torch following the instructions on the
   `official PyTorch website <https://pytorch.org>`_. We recommend
   **torch 2.11**.

2. Install PyTorch Geometric and its ``torch-cluster`` extension following
   the instructions in the
   `PyTorch Geometric documentation <https://pytorch-geometric.readthedocs.io/en/stable/install/installation.html>`_.

3. Install acceleration for equivariant operations, either via
   ``cuequivariance-torch`` and ``cuequivariance-ops``, or via
   ``openequivariance``.

4. Install the remaining project dependencies from the provided
   ``requirements.txt`` file.

5. Install ``mace-torch`` (supported version ``0.3.16``) **without**
   dependencies.

6. Install ``nequip`` (supported version ``0.12.1``) and ``nequip-allegro``
   (supported version ``0.7.0``) **without** dependencies.

7. Install the package itself.

.. note::
   Steps 5 and 6 must use ``--no-deps``. Both ``mace-torch`` and ``nequip``
   declare conflicting version requirements for ``e3nn``; installing them
   without dependencies keeps the ``e3nn==0.5.3`` version installed in
   step 4 intact, instead of triggering a resolution conflict or a silent
   downgrade/upgrade.

Full example for a machine with CUDA 13.0 (``cu130``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
   pip install torch_geometric
   pip install torch_cluster -f https://data.pyg.org/whl/torch-2.11.0+cu130.html
   pip install cuequivariance-torch
   pip install cuequivariance-ops-torch-cu13
   pip install -r requirements.txt
   pip install --no-deps mace-torch==0.3.16
   pip install --no-deps nequip==0.12.1 nequip-allegro==0.7.0
   pip install --no-deps .

Adapt the index URLs and ``cuequivariance-ops`` package name to your platform
(CPU, ``cu128``, ``cu130``, or ROCm) as needed.
