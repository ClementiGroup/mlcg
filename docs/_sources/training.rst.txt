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


Scripts
-------

Scripts that are using ``LightningCLI`` have many convinient built in 
`functionalities <https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html#lightningcli>`_ 
such as a detailed helper

.. code:: bash

   python scripts/mlcg-train.py --help
   python scripts/mlcg-train_h5.py --help
