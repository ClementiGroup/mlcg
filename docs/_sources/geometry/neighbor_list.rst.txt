Neighbor Lists
==============

`mlcg.neighborlist` contains several functions that can be used to create neighbor lists using a finite cutoff with mixed periodic boundary conditions using pytorch. There are also tools to interface with `ASE` tools. 
The neighbor lists are dictionaries containing meta-data (user defined `tag`, `body order`, `cutoff`, `self_interaction`) and the actual indices, `index_mapping`, and `cell_shifts`.

Neighbor List Utilities
-----------------------
These utilities are meant to format and validate neighbor list dictionaries.

.. autofunction:: mlcg.neighbor_list.neighbor_list.make_neighbor_list
.. autofunction:: mlcg.neighbor_list.neighbor_list.atomic_data2neighbor_list
.. autofunction:: mlcg.neighbor_list.neighbor_list.validate_neighborlist


Torch Implementation
--------------------

.. autofunction:: mlcg.neighbor_list.torch_impl.torch_neighbor_list
.. autofunction:: mlcg.neighbor_list.torch_impl.torch_neighbor_list_no_pbc
.. autofunction:: mlcg.neighbor_list.torch_impl.compute_images
.. autofunction:: mlcg.neighbor_list.torch_impl.wrap_positions

Nvalchemi Implementation
--------------------

.. autofunction:: mlcg.neighbor_list.nvalchemi_impl.nvalchemi_naive_neighbor_list
.. autofunction:: mlcg.neighbor_list.nvalchemi_impl.nvalchemi_cell_neighbor_list

ASE Implementation
------------------

.. autofunction:: mlcg.neighbor_list.ase_impl.ase_neighbor_list

