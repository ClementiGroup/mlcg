MLIPs
======

`mlcg.nn` currently implements the SchNet graph neural network, as well as several utility classes for computing distance expansions and cutoffs. The `nn` subpackage also contains several useful classes for extracting other properties from energy predictions or aggregating the predictions from several different model types.

SchNet Utilities
----------------

These classes are used to define a SchNet graph neural network. For "typical" SchNet models, users
may find the class `StandardSchNet` to be helpful in getting started quickly.

.. autoclass:: mlcg.nn.StandardSchNet
.. autoclass:: mlcg.nn.schnet.SchNet
.. autoclass:: mlcg.nn.schnet.InteractionBlock
.. autoclass:: mlcg.nn.schnet.CFConv

PaiNN Utilities
----------------

These classes are used to define a PaiNN graph neural network. For "typical" PaiNN models, users
may find the class `StandardPaiNN` to be helpful in getting started quickly.

.. autoclass:: mlcg.nn.StandardPaiNN
.. autoclass:: mlcg.nn.PaiNN

So3krates Utilities
----------------

These classes are used to define a so3krates graph neural network. For "typical" so3krates models, users
may find the class `StandardPaiNN` to be helpful in getting started quickly.

.. autoclass:: mlcg.nn.So3krates
.. autoclass:: mlcg.nn.StandardSo3krates

MACE Utilities
----------------

These classes are used to define a MACE graph neural network, for which the base implementation is required and available `here <https://github.com/ACEsuit/mace.git>`_. For "typical" MACE models, users
may find the class `StandardMACE` to be helpful in getting started quickly.

.. autoclass:: mlcg.nn.StandardMACE
.. autoclass:: mlcg.nn.mace.MACE


Allegro Utilities
----------------

These classes are used to define an Allegro graph neural network, for which the base implementation is required and available `here <https://github.com/ACEsuit/mace.git>`_. For "typical" MACE models, users
may find the class `StandardAllegro` to be helpful in getting started quickly.

.. autoclass:: mlcg.nn.StandardAllegro
.. autoclass:: mlcg.nn.Allegro
