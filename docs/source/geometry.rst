Geometry
========

`mlcg.geometry` implements geometry and topology tools for CG systems.

Internal Coordinate Functions
-----------------------------

.. autofunction:: mlcg.geometry.internal_coordinates.compute_distances
.. autofunction:: mlcg.geometry.internal_coordinates.compute_distance_vectors
.. autofunction:: mlcg.geometry.internal_coordinates.compute_angles_cos
.. autofunction:: mlcg.geometry.internal_coordinates.compute_angles_raw
.. autofunction:: mlcg.geometry.internal_coordinates.compute_torsions

Topology Utilities
------------------

These classes and functions may be used to define, manipulate, and analyze CG topologies through graph representations.

.. autoclass:: mlcg.geometry.topology.Atom
.. autoclass:: mlcg.geometry.topology.Topology
    :members:
.. autofunction:: mlcg.geometry.topology.add_chain_bonds
.. autofunction:: mlcg.geometry.topology.add_chain_angles
.. autofunction:: mlcg.geometry.topology.get_connectivity_matrix
.. autofunction:: mlcg.geometry.topology.get_n_pairs
.. autofunction:: mlcg.geometry.topology.get_n_paths

Statistics
----------

These functions gather statistics for further analysis or for parametrizing prior models.

.. autofunction:: mlcg.geometry.statistics.compute_statistics
.. autofunction:: mlcg.geometry.statistics.fit_baseline_models

