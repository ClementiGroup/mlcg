mlcg
==========

.. start-intro

|Docs badge| |License| |Circleci|

.. |Docs badge| image:: https://img.shields.io/badge/mlcg-docs-blue.svg
   :target: https://clementigroup.github.io/mlcg/

.. |License| image:: https://img.shields.io/github/license/Naereen/StrapDown.js.svg
   :target: https://opensource.org/licenses/MIT

.. |Circleci| image:: https://dl.circleci.com/status-badge/img/gh/ClementiGroup/mlcg/tree/main.svg?style=shield
    :target: https://dl.circleci.com/status-badge/redirect/gh/ClementiGroup/mlcg/tree/main

This repository collects a set of tools to apply machine learning techniques to coarse grain atomic systems.


Installation
------------
.. start-install

Requires **Python 3.12**. Clone the repo:

.. code:: bash

    git clone git@github.com:ClementiGroup/mlcg.git
    cd mlcg

**With uv (recommended)**

Install `uv <https://docs.astral.sh/uv/>`_, then run ``uv sync`` with the
extra matching your hardware:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Hardware
     - Command
   * - NVIDIA CUDA 12.8
     - ``uv sync --extra cu128``
   * - NVIDIA CUDA 13.0
     - ``uv sync --extra cu130``
   * - CPU only
     - ``uv sync --extra cpu``

This creates a ``.venv`` in the repo root and installs everything, including
``torch 2.11.0`` and ``torch-cluster``, from the correct pre-built wheel pages.
It is also possible to install tools useful for developers by adding ``--dev`` flag.

If you manage your own ``uv``-created virtual environment (e.g. via ``uv venv`` and ``uv pip install``),
use ``uv pip`` in place of ``pip`` in the instructions below.

**With pip (lockfile-based)**

Per-platform lockfiles are provided for reproducible pip installs.
Run both steps — the first installs binary packages with build isolation
disabled, the second installs the remaining dependencies:

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Hardware
     - Step 1
     - Step 2
   * - NVIDIA CUDA 12.8
     - ``pip install --no-deps --no-build-isolation pylock_cu128.toml``
     - ``pip install --no-deps pylock_no_hashes.toml``
   * - NVIDIA CUDA 13.0
     - ``pip install --no-deps --no-build-isolation pylock_cu130.toml``
     - ``pip install --no-deps pylock_no_hashes.toml``
   * - CPU only
     - ``pip install --no-deps --no-build-isolation pylock_cpu.toml``
     - ``pip install --no-deps pylock_no_hashes.toml``

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

    pip install sphinx shibuya sphinx-autodoc-typehints


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