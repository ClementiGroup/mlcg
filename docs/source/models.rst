Models
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
.. autoclass:: mlcg.nn.painn.PaiNN

MACE Utilities
----------------

These classes are used to define a MACE graph neural network, for which the base implementation is required and available `here <https://github.com/ACEsuit/mace.git>`_. For "typical" MACE models, users
may find the class `StandardMACE` to be helpful in getting started quickly.

.. autoclass:: mlcg.nn.StandardMACE
.. autoclass:: mlcg.nn.mace.MACE

Radial Basis Functions
----------------------

Sets of radial basis functions are used to expand the distances (or other molecular features) between atoms on a fixed-sized vector. For instance, this is the main transformation of the distances in the `SchNet` model.

.. autoclass:: mlcg.nn.radial_basis.GaussianBasis
.. autoclass:: mlcg.nn.radial_basis.ExpNormalBasis
.. autoclass:: mlcg.nn.radial_basis.RIGTOBasis
.. autoclass:: mlcg.nn.radial_basis.SpacedExpBasis
.. autoclass:: mlcg.nn.angular_basis.SphericalHarmonics

Cutoff Functions
----------------

Cutoff functions are used to enforce the smoothness of the models w.r.t. neighbor insertion/removal from an atomic environment. Some are also used to damp the signal from a neighbor's displacement that is "far" from the central atom, e.g. `CosineCutoff`. Cutoff functions are also used in the construction of radial basis functions.

.. autoclass:: mlcg.nn.cutoff.IdentityCutoff
.. autoclass:: mlcg.nn.cutoff.CosineCutoff
.. autoclass:: mlcg.nn.cutoff.ShiftedCosineCutoff

Model Building Utilities
------------------------

These classes are used to build more complicated models.

.. autoclass:: mlcg.nn.gradients.GradientsOut
.. autoclass:: mlcg.nn.gradients.SumOut

Loss Functions
--------------

These classes define loss functions for model optimization, as well as generalized losses that combine several losses of different types.

.. autoclass:: mlcg.nn.losses.Loss
.. autoclass:: mlcg.nn.losses.ForceRMSE
.. autoclass:: mlcg.nn.losses.ForceMSE
