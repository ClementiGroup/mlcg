"""Keys for dictionaries/AtomicData objects.

This is a seperate module to compensate for a TorchScript bug that can only recognize constants when they are accessed as attributes of an imported module.

Adapted from NequIP (https://github.com/mir-group/nequip)
"""

from typing import List, Final
import sys

# == Define allowed keys as constants ==
#: The positions of the atoms in the system
POSITIONS_KEY: Final[str] = "pos"
N_ATOMS_KEY: Final[str] = "n_atoms"
#: The masses of the atoms in the system
MASS_KEY: Final[str] = "masses"
#: conectivity information
NEIGHBOR_LIST_KEY: Final[str] = "neighbor_list"
#: metadata about the structures
TAG_KEY: Final[str] = "tag"

#: A [n_edge, 3] tensor of direction vectors associated to edges
DIRECTION_VECTORS_KEY: Final[str] = "direction_vectors"
#: A [n_edge] tensor of the lengths of EDGE_VECTORS
DISTANCES_KEY: Final[str] = "distances"
#: [n_edge, dim] (possibly equivariant) attributes of each edge
EDGE_ATTRS_KEY: Final[str] = "edge_attrs"
#: [n_edge, dim] invariant embedding of the edges
EDGE_EMBEDDING_KEY: Final[str] = "edge_embedding"
#: [3, 3] unit cell of the atomic structure. Lattice vectors are defined row wise
CELL_KEY: Final[str] = "cell"
#: [3] periodic boundary conditions
PBC_KEY: Final[str] = "pbc"

NODE_FEATURES_KEY: Final[str] = "node_features"
NODE_ATTRS_KEY: Final[str] = "node_attrs"
#: if atoms then it's the atomic number, if it's a CG bead then it's a number
#: defined by the CG mapping
ATOM_TYPE_KEY: Final[str] = "atom_types"

ENERGY_KEY: Final[str] = "energy"
FORCE_KEY: Final[str] = "forces"
VELOCITY_KEY: Final[str] = "velocities"

PROPERTY_KEYS: Final[List[str]] = (ENERGY_KEY, FORCE_KEY)

BATCH_KEY: Final[str] = "batch"

#: Make a list of allowed keys
ALLOWED_KEYS: List[str] = [
    getattr(sys.modules[__name__], k)
    for k in sys.modules[__name__].__dict__.keys()
    if k.endswith("_KEY")
]

SCALAR_KEYS = [ENERGY_KEY]


def validate_keys(keys, graph_required=True):
    pass
    # # Validate combinations
    # if graph_required:
    #     if not (POSITIONS_KEY in keys):
    #         raise KeyError("At least pos and edge_index must be supplied")
    # # for k in keys:
    # #     assert k in ALLOWED_KEYS, f"{k} not in ALLOWED_KEYS={ALLOWED_KEYS}"
