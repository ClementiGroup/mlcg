mlcg
==========

.. start-intro

|Docs badge| |License|

.. |Docs badge| image:: https://img.shields.io/badge/mlcg-docs-blue.svg
   :target: https://clementigroup.github.io/mlcg/

.. |License| image:: https://img.shields.io/github/license/Naereen/StrapDown.js.svg
   :target: https://opensource.org/licenses/MIT


This repository collects a set of tools to apply machine learning techniques to coarse grain atomic systems.


Installation
------------
.. start-install

First we suggest to create a new clean empty virtual environment with **python 3.12**, then clone the repo and 
install the following prerequisites:

.. code:: bash

    git clone git@github.com:ClementiGroup/mlcg.git
    cd mlcg
    pip install -r env_with_hashes.in
    pip install --no-deps git+https://github.com/ACEsuit/mace.git@v0.3.13
    pip install --no-deps nequip==0.12.1 nequip-allegro==0.7.0

Then install this repository with:

.. code:: bash

    pip install .


This will likely rise an error due to some dependency issue about `e3nn` that you can safely ignore.

.. end-install

CLI
---

The models defined in this library can be conveniently trained using the pytorch-lightning
`CLI <https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html>`_ utilities.

Examples
--------

Please take a look into the examples folder to see how to use this code to train a model over an existing dataset.


.. end-intro

.. start-doc

Documentation
-------------

Documentation is available `here <https://clementigroup.github.io/mlcg/>`_ and here are some references on how to work with it.

Dependencies
~~~~~~~~~~~~

.. code:: bash

    pip install sphinx sphinx_rtd_theme sphinx-autodoc-typehints


How to build
~~~~~~~~~~~~

.. code:: bash

    cd docs
    sphinx-build -b html source build

How to update the online documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

This will run the full set of tests in `mlcg` and the larger test under `examples/test_examples.py`, 
including training and simulation of all models described in `examples/input_yamls/README.md`.  

For quick local development testing, it is also possible to exclude the large test by running:

.. code-block:: bash

    coverage run -m pytest --light

Troubleshooting
---------------

If it is not possible to install an environment with `pip install -r env_with_hashes.in`, the
following commands can do a similar job.

.. code:: bash

    pip install --extra-index-url=https://download.pytorch.org/whl/cu124 torch==2.6.0
    pip install torch_geometric
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
    pip install lightning tensorboard torchtnt