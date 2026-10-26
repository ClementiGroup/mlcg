Training
========

``mlcg`` provides some tools to train its models in the ``scripts`` folder and some example input 
files such as ``examples/train_schnet.yaml``. The training is defined 
using the `pytorch-lightning <https://pytorch-lightning.readthedocs.io/en/latest/>`_ package and 
especially its `cli <https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html>`_ utilities.


Extensions for using Pytorch Lightning
----------------------------------

.. autoclass:: mlcg.pl.DataModule
   :members:

.. autoclass:: mlcg.pl.PLModel
   :members:

.. autoclass:: mlcg.pl.LightningCLI
   :members:



Faster training
---------------

Force matching differentiates the forces :math:`-\partial E/\partial
\mathbf{r}` once more, which ``torch.compile`` can't compile through.
:func:`mlcg.nn.optimize_for_training` traces the energy *and* the force
computation into one graph and compiles it, so the loss needs only a
first-order backward. Enable it with the ``optimize`` argument of
:class:`mlcg.pl.PLModel`:

.. code:: yaml

   model:
     optimize:
       compile: true   # compiled energy + forces (training and validation)
       flash: true     # flash SchNet kernels: same speed once compiled, less memory

or on the command line, ``--model.optimize '{compile: true}'``.

- The first training step traces and compiles (tens of seconds) and
  checks the compiled outputs and gradients against eager.
- Batch sizes may change freely: one graph serves every number of beads,
  edges and graphs. Degenerate batches (e.g. a single graph) and models
  that can't be traced run eagerly; the latter with a warning.
- Checkpoints don't change: they restart with any ``optimize`` setting, and
  :func:`mlcg.pl.extract_model_from_checkpoint` returns the model in the
  architecture it was defined with (e.g. ``StandardSchNet``).

.. autofunction:: mlcg.nn.optimize_for_training

.. autoclass:: mlcg.nn.CompiledTraining


Scripts
-------

Scripts that are using ``LightningCLI`` have many convinient built in 
`functionalities <https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html#lightningcli>`_ 
such as a detailed helper

.. code:: bash

   python scripts/mlcg-train.py --help
   python scripts/mlcg-train_h5.py --help
