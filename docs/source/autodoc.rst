API Reference
=============

TODO: split this file into respective files where both API reference and some more high level information is provided to the reader.

Atomic Data
-----------

..
   .. automodule:: mlcg.data._keys
      :members:
      :undoc-members:

TODO: describe the data structure and its use.

.. automodule:: mlcg.data.atomic_data
   :members:


Coarse Graining Utilities
-------------------------

TODO: document the CG mapping format.

.. automodule:: mlcg.cg.projection
   :members:


Models
------

Neural network models for coarse grain property predictions

.. automodule:: mlcg.nn.schnet
   :members:

.. automodule:: mlcg.nn.painn
   :members:

.. automodule:: mlcg.nn.mace
   :members:

Radial basis functions
----------------------

.. automodule:: mlcg.nn.radial_basis.radial_integral_gto
   :members:

.. automodule:: mlcg.nn.radial_basis.exp_normal
   :members:

.. automodule:: mlcg.nn.radial_basis.gaussian
   :members:

.. automodule:: mlcg.nn.radial_basis.exp_spaced
   :members:

.. automodule:: mlcg.nn.angular_basis.spherical_harmonics
   :members:

Cutoff functions
----------------

.. automodule:: mlcg.nn.cutoff
   :members:


Datasets
--------

.. automodule:: mlcg.datasets.chignolin
   :members:

.. automodule:: mlcg.datasets.alanine_dipeptide
   :members:

.. automodule:: mlcg.datasets.h5_dataset
   :members:

Neighbor List
-------------

Main interface to the computation of neighborlists with a finite spherical cutoff

.. automodule:: mlcg.neighbor_list.neighbor_list
   :members:
   :undoc-members:

Torch geometric implementation

.. automodule:: mlcg.neighbor_list.torch_impl
   :members:
   :undoc-members:

Utilities to compute the internal coordinates

.. automodule:: mlcg.geometry.internal_coordinates
   :members:
   :undoc-members:

Simulations
-----------

.. automodule:: mlcg.simulation.LangevinSimulation
   :members:

.. automodule:: mlcg.simulation.OverdampedSimulation
   :members:

.. automodule:: mlcg.simulation.PTSimulation
   :members:
