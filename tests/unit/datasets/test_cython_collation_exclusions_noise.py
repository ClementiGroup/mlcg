"""Tests that the HDF5 "NG" dataset + cython collater feed `excluded_pairs`
and `noise_levels` to the models, with and without periodic boundaries."""

import h5py
import numpy as np
import pytest
import torch

from mlcg.datasets.cython.batch_helper import (
    batch_collate_w_nls,
    collate_index_pairs,
)
from mlcg.datasets.cython.collater import CythonCollater
from mlcg.datasets.cython.transform import NoiseMapTransformer
from mlcg.datasets.h5_dataset_ng import H5DatasetNG
from mlcg.neighbor_list.neighbor_list import atomic_data2neighbor_list
from mlcg.nn.cutoff import CosineCutoff
from mlcg.nn.noise_schnet import NoiseEmbedSchNet
from mlcg.nn.radial_basis import GaussianBasis, SkewedMuGaussianBasis
from mlcg.nn.schnet import StandardSchNet

N_ATOMS = {"a": 8, "b": 5}
N_FRAMES = 6
EXCLUDED_ORDER = 2
BOX = 6.0
BATCH_SIZE = 3

BASE_KEYS = {
    "embeds": "attrs:cg_embeds",
    "coords": "cg_coords",
    "forces": "cg_delta_forces",
}


def chain_bonds(n_atoms):
    return np.stack([np.arange(n_atoms - 1), np.arange(1, n_atoms)])


def make_h5(
    path, pbc=False, noise=False, pairs_as_dataset=False, order=EXCLUDED_ORDER
):
    """Two chain molecules of different sizes, with their bonds as
    `excluded_pairs` and optionally cells/pbc and per-frame noise levels."""
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        if order is not None:
            f.attrs["excluded_order"] = order
        grp = f.create_group("M")
        for name, n_atoms in N_ATOMS.items():
            mol = grp.create_group(name)
            # a chain with ~1.0 spacing, so bonded pairs are within the cutoff
            coords = np.cumsum(
                rng.normal(scale=0.6, size=(N_FRAMES, n_atoms, 3)), axis=1
            ).astype(np.float32)
            mol.create_dataset("cg_coords", data=coords)
            mol.create_dataset(
                "cg_delta_forces",
                data=rng.normal(size=coords.shape).astype(np.float32),
            )
            mol.attrs["cg_embeds"] = np.arange(1, n_atoms + 1)
            mol.attrs["N_frames"] = N_FRAMES
            if pairs_as_dataset:
                mol.create_dataset("excluded_pairs", data=chain_bonds(n_atoms))
            else:
                mol.attrs["excluded_pairs"] = chain_bonds(n_atoms)
            if noise:
                mol.create_dataset(
                    "noise_levels",
                    data=np.linspace(0.1, 0.6, N_FRAMES).astype(np.float32),
                )
            if pbc:
                mol.create_dataset(
                    "cell",
                    data=np.tile(
                        np.eye(3, dtype=np.float32) * BOX, (N_FRAMES, 1, 1)
                    ),
                )
                mol.create_dataset(
                    "pbc", data=np.ones((N_FRAMES, 3), dtype=bool)
                )
    return path


def load(path, pbc=False, noise=False, pairs_key="attrs:excluded_pairs"):
    keys = dict(BASE_KEYS)
    if pairs_key is not None:
        keys["excluded_pairs"] = pairs_key
    if noise:
        keys["noise_levels"] = "noise_levels"
    if pbc:
        keys["cell"] = "cell"
        keys["pbc"] = "pbc"
    part = {
        "train": {
            "metasets": {"M": {"molecules": list(N_ATOMS)}},
            "batch_sizes": {"M": BATCH_SIZE},
        }
    }
    return H5DatasetNG(str(path), part, {"hdf_key_mapping": keys})


def frames_of(ds, idx):
    """Frames by index; indices span both molecules (different sizes)."""
    metaset = ds.partition("train").get_metaset("M")
    return [metaset[i] for i in idx]


MIXED = [0, N_FRAMES, 1]  # molecule a, molecule b, molecule a


def expected_pairs(frames):
    offsets = np.cumsum([0] + [len(f["atom_types"]) for f in frames[:-1]])
    return np.concatenate(
        [
            f["excluded_pairs"]["index_mapping"] + o
            for f, o in zip(frames, offsets)
        ],
        axis=1,
    )


def unordered_keys(pairs, n):
    pairs = torch.as_tensor(pairs)
    return torch.minimum(pairs[0], pairs[1]) * n + torch.maximum(
        pairs[0], pairs[1]
    )


# --------------------------------------------------------------------------- #
# dataset frames
# --------------------------------------------------------------------------- #


def test_frames_carry_excluded_pairs_and_noise(tmp_path):
    ds = load(make_h5(tmp_path / "d.h5", noise=True), noise=True)
    frame = frames_of(ds, [0])[0]
    assert "exc_pair_index" not in frame
    exc = frame["excluded_pairs"]
    assert exc["excluded_order"] == EXCLUDED_ORDER
    assert exc["index_mapping"].dtype == np.int64
    assert exc["index_mapping"].flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(
        exc["index_mapping"], chain_bonds(N_ATOMS["a"])
    )
    assert isinstance(frame["noise_levels"], np.ndarray)
    assert frame["noise_levels"].dtype == np.float32
    assert frame["noise_levels"].shape == (N_ATOMS["a"],)


def test_excluded_pairs_stored_as_dataset(tmp_path):
    ds = load(
        make_h5(tmp_path / "d.h5", pairs_as_dataset=True),
        pairs_key="excluded_pairs",
    )
    frame = frames_of(ds, [0])[0]
    np.testing.assert_array_equal(
        frame["excluded_pairs"]["index_mapping"], chain_bonds(N_ATOMS["a"])
    )


def test_excluded_pairs_require_excluded_order(tmp_path):
    with pytest.raises(ValueError, match="excluded_order"):
        load(make_h5(tmp_path / "d.h5", order=None))


def test_no_excluded_pairs_without_key(tmp_path):
    ds = load(make_h5(tmp_path / "d.h5"), pairs_key=None)
    assert "excluded_pairs" not in frames_of(ds, [0])[0]
    assert "excluded_pairs" not in batch_collate_w_nls(frames_of(ds, MIXED))


# --------------------------------------------------------------------------- #
# batch_helper
# --------------------------------------------------------------------------- #


def test_collate_index_pairs_offsets_and_empty_frames():
    sizes = np.array([4, 3, 6, 2])
    pairs = [
        chain_bonds(4),
        np.empty((2, 0), dtype=np.int64),
        chain_bonds(6),
        chain_bonds(2).astype(np.int32),
    ]  # non-intp input is converted
    out = collate_index_pairs(pairs, sizes)
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    np.testing.assert_array_equal(
        out, np.concatenate([p + o for p, o in zip(pairs, offsets)], axis=1)
    )


@pytest.mark.parametrize("with_pbc", [False, True])
def test_batch_collate_excluded_pairs_and_noise(tmp_path, with_pbc):
    ds = load(
        make_h5(tmp_path / "d.h5", pbc=with_pbc, noise=True),
        pbc=with_pbc,
        noise=True,
    )
    frames = frames_of(ds, MIXED)
    out = batch_collate_w_nls(frames, with_pbc=with_pbc)
    n_atoms = [len(f["atom_types"]) for f in frames]
    np.testing.assert_array_equal(out["n_atoms"], n_atoms)
    np.testing.assert_array_equal(
        out["excluded_pairs"]["excluded_order"], [EXCLUDED_ORDER] * len(frames)
    )
    exc = out["excluded_pairs"]["index_mapping"]
    np.testing.assert_array_equal(exc, expected_pairs(frames))
    np.testing.assert_array_equal(out["batch"][exc[0]], out["batch"][exc[1]])
    np.testing.assert_array_equal(
        out["noise_levels"], np.concatenate([f["noise_levels"] for f in frames])
    )
    assert out["noise_levels"].dtype == np.float32
    assert ("cell" in out) == with_pbc and ("pbc" in out) == with_pbc
    if with_pbc:
        np.testing.assert_array_equal(
            out["cell"], np.stack([f["cell"] for f in frames])
        )


@pytest.mark.parametrize("key", ["excluded_pairs", "noise_levels"])
def test_batch_collate_rejects_partially_present_fields(tmp_path, key):
    ds = load(make_h5(tmp_path / "d.h5", noise=True), noise=True)
    frames = frames_of(ds, MIXED)
    del frames[1][key]
    with pytest.raises(ValueError, match=key):
        batch_collate_w_nls(frames)


def test_batch_collate_rejects_wrong_noise_length(tmp_path):
    ds = load(make_h5(tmp_path / "d.h5", noise=True), noise=True)
    frames = frames_of(ds, MIXED)
    frames[1]["noise_levels"] = frames[1]["noise_levels"][:-1]
    with pytest.raises(ValueError, match="noise levels"):
        batch_collate_w_nls(frames)


def test_with_pbc_requires_cell(tmp_path):
    ds = load(make_h5(tmp_path / "d.h5"))
    with pytest.raises(ValueError, match="cell"):
        batch_collate_w_nls(frames_of(ds, MIXED), with_pbc=True)


# --------------------------------------------------------------------------- #
# collater -> models
# --------------------------------------------------------------------------- #


def schnet(excluded_order):
    return StandardSchNet(
        GaussianBasis(cutoff=2.5, num_rbf=8),
        CosineCutoff(0, 2.5),
        [16],
        hidden_channels=16,
        num_filters=16,
        excluded_order=excluded_order,
    )


@pytest.mark.parametrize("with_pbc", [False, True])
def test_collated_exclusions_reach_the_neighbor_list(tmp_path, with_pbc):
    ds = load(make_h5(tmp_path / "d.h5", pbc=with_pbc), pbc=with_pbc)
    frames = frames_of(ds, MIXED)
    batch = CythonCollater(device="cpu", with_pbc=with_pbc)(frames)
    assert ("cell" in batch) == with_pbc
    excluded = batch.excluded_pairs["index_mapping"]
    assert excluded.dtype == torch.long

    nl = atomic_data2neighbor_list(batch, rcut=2.5)
    n = int(batch.n_atoms.sum())
    assert nl["excluded_order"] == EXCLUDED_ORDER
    assert not torch.isin(
        unordered_keys(nl["index_mapping"], n), unordered_keys(excluded, n)
    ).any()
    # without the exclusions, some bonded pairs would be neighbors
    del batch["excluded_pairs"]
    nl_all = atomic_data2neighbor_list(batch, rcut=2.5)
    assert torch.isin(
        unordered_keys(nl_all["index_mapping"], n), unordered_keys(excluded, n)
    ).any()


def test_schnet_with_excluded_order_runs_on_collated_batch(tmp_path):
    ds = load(make_h5(tmp_path / "d.h5"))
    batch = CythonCollater(device="cpu")(frames_of(ds, MIXED))
    batch.pos.requires_grad_(True)
    out = schnet(EXCLUDED_ORDER)(batch)
    assert out.out["SchNet"]["energy"].shape == (len(MIXED),)


def test_schnet_excluded_order_mismatch_is_caught(tmp_path):
    ds = load(make_h5(tmp_path / "d.h5"), pairs_key=None)
    batch = CythonCollater(device="cpu")(frames_of(ds, MIXED))
    with pytest.raises(ValueError, match="excluded_order"):
        schnet(EXCLUDED_ORDER)(batch)


def test_noise_transform_records_noise_levels(tmp_path):
    ds = load(make_h5(tmp_path / "d.h5"))
    frames = frames_of(ds, MIXED)
    noise_level = 0.3
    batch = CythonCollater(
        device="cpu", transform=NoiseMapTransformer(noise_level, kbt=1.0)
    )(frames)
    assert batch.noise_levels.shape == (int(batch.n_atoms.sum()),)
    assert batch.noise_levels.dtype == torch.float32
    torch.testing.assert_close(
        batch.noise_levels, torch.full_like(batch.noise_levels, noise_level)
    )


def zero_force_transform(frame):
    """Dummy transform that sets the forces to zero. Replaces the array
    instead of writing into it: frames hold views of the dataset."""
    frame["forces"] = np.zeros_like(frame["forces"])


def test_collater_applies_custom_transform(tmp_path):
    # NG counterpart of the old `test_h5_dataset_transform`: the transform is
    # applied per frame by the collater, and the dataset is left untouched
    ds = load(make_h5(tmp_path / "d.h5"))
    frames = frames_of(ds, MIXED)
    original = [f["forces"].copy() for f in frames]
    assert any(np.any(f != 0) for f in original)
    batch = CythonCollater(device="cpu", transform=zero_force_transform)(frames)
    assert batch.forces.shape == (int(batch.n_atoms.sum()), 3)
    assert torch.all(batch.forces == 0)
    metaset = ds.partition("train").get_metaset("M")
    for i, forces in zip(MIXED, original):
        np.testing.assert_array_equal(metaset[i]["forces"], forces)


@pytest.mark.parametrize("with_pbc", [False, True])
def test_noise_embed_schnet_runs_on_collated_batch(tmp_path, with_pbc):
    ds = load(
        make_h5(tmp_path / "d.h5", pbc=with_pbc, noise=True),
        pbc=with_pbc,
        noise=True,
    )
    batch = CythonCollater(device="cpu", with_pbc=with_pbc)(
        frames_of(ds, MIXED)
    )
    batch.pos.requires_grad_(True)
    model = NoiseEmbedSchNet(
        GaussianBasis(cutoff=2.5, num_rbf=8),
        CosineCutoff(0, 2.5),
        SkewedMuGaussianBasis(cutoff=1.0, num_rbf=4),
        [16],
        hidden_channels=16,
        num_filters=16,
        excluded_order=EXCLUDED_ORDER,
    )
    energy = model(batch).out["SchNet"]["energy"]
    forces = torch.autograd.grad(energy.sum(), batch.pos)[0]
    assert energy.shape == (len(MIXED),)
    assert torch.isfinite(forces).all()
