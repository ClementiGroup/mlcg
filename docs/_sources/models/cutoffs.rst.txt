Cutoff Functions
================

Cutoff functions are used to enforce the smoothness of the models w.r.t. neighbor insertion/removal from an atomic environment. Some are also used to damp the signal from a neighbor's displacement that is "far" from the central atom, e.g. `CosineCutoff`. Cutoff functions are also used in the construction of radial basis functions.

.. autoclass:: mlcg.nn.cutoff.IdentityCutoff
.. autoclass:: mlcg.nn.cutoff.CosineCutoff
.. autoclass:: mlcg.nn.cutoff.ShiftedCosineCutoff
