import numpy as np


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
    """Add Gaussian noise of variance `noise_level` to the coordinates of a
    frame, correct the forces accordingly, and record `noise_level` as the
    per-bead `noise_levels` of the frame (for noise-conditioned models)."""

    def __init__(self, noise_level, kbt, random_seed=42):
        self.noise_level = np.float32(noise_level)
        self.sigma = np.sqrt(noise_level, dtype=np.float32)
        self.kbt = kbt
        self.rng = np.random.default_rng(random_seed)

    def __call__(self, atomic_data):
        fmap = atomic_data["neighbor_list"].get("fmap", None)
        atomic_data["pos"], atomic_data["forces"] = noise_and_map_frame(
            atomic_data["pos"],
            atomic_data["forces"],
            self.sigma,
            self.kbt,
            self.rng,
            fmap,
        )
        atomic_data["noise_levels"] = np.full(
            len(atomic_data["atom_types"]), self.noise_level, dtype=np.float32
        )
