Radial Basis Functions
=======================

Sets of radial basis functions are used to expand the distances (or other molecular features) between atoms on a fixed-sized vector. For instance, this is the main transformation of the distances in the `SchNet` model.

.. autoclass:: mlcg.nn.radial_basis.GaussianBasis
.. autoclass:: mlcg.nn.radial_basis.ExpNormalBasis
.. autoclass:: mlcg.nn.radial_basis.RIGTOBasis
.. autoclass:: mlcg.nn.radial_basis.SpacedExpBasis
.. autoclass:: mlcg.nn.angular_basis.SphericalHarmonics

