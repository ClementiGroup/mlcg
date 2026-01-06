cimport cython
cimport numpy as np
from libc.string cimport memcpy
import numpy as np

# init numpy
np.import_array()
ctypedef np.npy_intp intp # this is the actual int type in Python

# @cython.wraparound(False)   # Deactivate negative indexing.
# @cython.boundscheck(False)
def noise_and_map_frame(coords, forces, sigma, kbt, rng, fmap=None):
    noise = rng.standard_normal(size=coords.shape, dtype=np.float32)
    aug_coords = coords + sigma * noise
    aug_forces = -kbt * (noise / sigma)
    if fmap is not None:
        real_forces_corrected = forces - aug_forces
        batch_forces = fmap @ real_forces_corrected
        aug_forces += batch_forces
    return aug_coords, aug_forces

class NoiseMapTransformer:
    def __init__(self, noise_level, kbt, random_seed=42):
        self.sigma = np.sqrt(noise_level, dtype=np.float32)
        self.kbt = kbt
        self.rng = np.random.default_rng(random_seed)
    
    def __call__(self, atomic_data):
        fmap = atomic_data["neighbor_list"].get("fmap", None)
        atomic_data["pos"], atomic_data["forces"] = noise_and_map_frame(atomic_data["pos"], atomic_data["forces"], self.sigma, self.kbt, self.rng, fmap)

