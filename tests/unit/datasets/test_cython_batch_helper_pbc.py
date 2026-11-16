"""Tests for `mlcg.datasets.cython.batch_helper` (and the `CythonCollater`
built on top of it) on periodic systems.

The real-data tests use the POPC bilayer example in
`examples/h5_pl/multiple_molecules_with_pbc/` (`bilayer_with_pbc.h5` and its
companion `bilayer_partition.yaml`).
"""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from ruamel.yaml import YAML

from mlcg.datasets.cython.batch_helper import (
    make_batch,
    collate_bool_array_2d,
    collate_float_array_3d,
    batch_collate_w_nls,
    batch_collate_w_nls_w_pbc,
)
from mlcg.datasets.cython.collater import CythonCollater
from mlcg.datasets.h5_dataset_ng import H5DatasetNG
from mlcg.neighbor_list.neighbor_list import atomic_data2neighbor_list
from mlcg.pl.h5_data_ng import H5DataModule

EXAMPLE_DIR = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "h5_pl"
    / "multiple_molecules_with_pbc"
)
H5_PATH = EXAMPLE_DIR / "bilayer_with_pbc.h5"
PARTITION_YAML = EXAMPLE_DIR / "bilayer_partition.yaml"

METASET = "POPC"
MOLECULE = "bilayer"
BEADS_PER_LIPID = 12
DATA_GRP = "POPC_memb/popc"

HDF_KEY_MAPPING = {
    "embeds": "attrs:cg_embeds",
    "coords": f"{DATA_GRP}/cg_coords",
    "forces": f"{DATA_GRP}/cg_delta_forces",
    "cell": f"{DATA_GRP}/cell",
    "pbc": f"{DATA_GRP}/pbc",
}

BATCH_SIZE = 3
N_BATCHES = 4

pytestmark = pytest.mark.skipif(
    not H5_PATH.exists(), reason=f"example data {H5_PATH} not found"
)


# --------------------------------------------------------------------------- #
# helpers / fixtures
# --------------------------------------------------------------------------- #


def load_partition_options(batch_size=BATCH_SIZE, n_batches=N_BATCHES):
    """Load the companion yaml, normalised for `H5DatasetNG`.

    The example yaml lists molecules under `names`, while `H5DatasetNG`
    reads `molecules`; batch sizes are bumped so batches hold >1 frame.
    """
    with open(PARTITION_YAML) as f:
        options = YAML(typ="safe").load(f)
    for part in options.values():
        for metaset in part["metasets"].values():
            if "molecules" not in metaset:
                metaset["molecules"] = list(metaset.pop("names"))
        part["batch_sizes"] = {k: batch_size for k in part["batch_sizes"]}
        part["max_epoch_samples"] = batch_size * n_batches
    return options


@pytest.fixture(scope="module")
def dataset_pbc():
    return H5DatasetNG(
        str(H5_PATH),
        load_partition_options(),
        {"hdf_key_mapping": dict(HDF_KEY_MAPPING)},
    )


def get_frames(dataset, n=BATCH_SIZE, part="train"):
    metaset = dataset.partition(part).get_metaset(METASET)
    return [metaset[i] for i in range(n)]


def truncate_frame(frame, n_lipids):
    """Keep only the first `n_lipids` lipids of a frame (to get frames of
    different sizes, which exposes offsetting errors)."""
    n = n_lipids * BEADS_PER_LIPID
    out = dict(frame)
    out["pos"] = np.ascontiguousarray(frame["pos"][:n])
    out["forces"] = np.ascontiguousarray(frame["forces"][:n])
    out["atom_types"] = frame["atom_types"][:n]
    return out


# --------------------------------------------------------------------------- #
# low level helpers
# --------------------------------------------------------------------------- #


def test_example_h5_layout():
    """Guard the assumptions the rest of this module makes about the data."""
    with h5py.File(H5_PATH, "r") as f:
        grp = f[METASET][MOLECULE]
        n_frames = int(grp.attrs["N_frames"])
        embeds = grp.attrs["cg_embeds"]
        assert len(embeds) % BEADS_PER_LIPID == 0
        np.testing.assert_array_equal(
            embeds,
            np.tile(embeds[:BEADS_PER_LIPID], len(embeds) // BEADS_PER_LIPID),
        )
        d = grp[DATA_GRP]
        assert d["cg_coords"].shape == (n_frames, len(embeds), 3)
        assert d["cg_delta_forces"].shape == (n_frames, len(embeds), 3)
        assert d["cell"].shape == (n_frames, 3, 3)
        assert d["pbc"].shape == (n_frames, 3)
        assert d["pbc"].dtype == np.bool_
        assert d["pbc"][:].all()


@pytest.mark.parametrize("n_elements", [[5], [3, 1, 4], [1800, 1800, 1800]])
def test_make_batch(n_elements):
    batch, ptr = make_batch(np.asarray(n_elements))
    np.testing.assert_array_equal(
        batch, np.repeat(np.arange(len(n_elements)), n_elements)
    )
    np.testing.assert_array_equal(ptr, np.concatenate([[0], np.cumsum(n_elements)]))


def test_collate_float_array_3d_cells(dataset_pbc):
    frames = get_frames(dataset_pbc)
    cells = [f["cell"] for f in frames]
    out = collate_float_array_3d(cells)
    assert out.shape == (len(frames), 3, 3)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, np.stack(cells))


def test_collate_float_array_3d_distinct_cells():
    rng = np.random.default_rng(0)
    cells = [rng.random((3, 3), dtype=np.float32) for _ in range(5)]
    np.testing.assert_array_equal(collate_float_array_3d(cells), np.stack(cells))
    # also accept a leading singleton axis, (1, 3, 3)
    np.testing.assert_array_equal(
        collate_float_array_3d([c[None] for c in cells]), np.stack(cells)
    )


@pytest.mark.parametrize(
    "pbcs",
    [
        [np.array([True, True, True])] * 3,
        [
            np.array([True, False, True]),
            np.array([False, False, False]),
            np.array([False, True, True]),
        ],
    ],
)
def test_collate_bool_array_2d(pbcs):
    out = collate_bool_array_2d(pbcs)
    assert out.shape == (len(pbcs), 3)
    assert out.dtype == np.bool_
    np.testing.assert_array_equal(out, np.stack(pbcs))


# --------------------------------------------------------------------------- #
# batch_collate_* on the bilayer frames
# --------------------------------------------------------------------------- #


def test_dataset_frames_carry_pbc(dataset_pbc):
    for frame in get_frames(dataset_pbc):
        assert frame["cell"].shape == (3, 3)
        assert frame["pbc"].shape == (3,)
        assert frame["pbc"].all()
        assert frame["pos"].dtype == np.float32


def test_batch_collate_w_nls_w_pbc(dataset_pbc):
    frames = get_frames(dataset_pbc)
    out = batch_collate_w_nls_w_pbc(frames)
    n_atoms = [len(f["atom_types"]) for f in frames]
    np.testing.assert_array_equal(out["n_atoms"], n_atoms)
    np.testing.assert_array_equal(
        out["pos"], np.concatenate([f["pos"] for f in frames])
    )
    np.testing.assert_array_equal(
        out["forces"], np.concatenate([f["forces"] for f in frames])
    )
    np.testing.assert_array_equal(
        out["atom_types"], np.concatenate([f["atom_types"] for f in frames])
    )
    np.testing.assert_array_equal(out["cell"], np.stack([f["cell"] for f in frames]))
    np.testing.assert_array_equal(out["pbc"], np.stack([f["pbc"] for f in frames]))
    np.testing.assert_array_equal(
        out["batch"], np.repeat(np.arange(len(frames)), n_atoms)
    )
    np.testing.assert_array_equal(
        out["ptr"], np.concatenate([[0], np.cumsum(n_atoms)])
    )
    assert out["neighbor_list"] == {}


def test_batch_collate_w_nls_w_pbc_varying_sizes(dataset_pbc):
    frames = [
        truncate_frame(f, n)
        for f, n in zip(get_frames(dataset_pbc), [150, 10, 77])
    ]
    out = batch_collate_w_nls_w_pbc(frames)
    np.testing.assert_array_equal(out["n_atoms"], [1800, 120, 924])
    np.testing.assert_array_equal(
        out["pos"], np.concatenate([f["pos"] for f in frames])
    )
    np.testing.assert_array_equal(out["cell"], np.stack([f["cell"] for f in frames]))
    np.testing.assert_array_equal(out["pbc"], np.stack([f["pbc"] for f in frames]))


def test_batch_collate_w_nls_without_pbc_has_no_cell(dataset_pbc):
    out = batch_collate_w_nls(get_frames(dataset_pbc))
    assert "cell" not in out and "pbc" not in out


# --------------------------------------------------------------------------- #
# CythonCollater
# --------------------------------------------------------------------------- #


def collate(frames, **kwargs):
    return CythonCollater(device="cpu", **kwargs)(frames)


def test_collater_with_pbc(dataset_pbc):
    frames = get_frames(dataset_pbc)
    batch = collate(frames, with_pbc=True)
    assert batch.cell.shape == (len(frames), 3, 3)
    assert batch.cell.dtype == batch.pos.dtype == torch.float32
    assert batch.pbc.shape == (len(frames), 3)
    assert batch.pbc.dtype == torch.bool
    assert batch.batch.dtype == torch.long
    torch.testing.assert_close(
        batch.cell, torch.from_numpy(np.stack([f["cell"] for f in frames]))
    )


def test_collated_pbc_batch_radius_graph_stays_within_frames(dataset_pbc):
    """The collated cell/pbc must be usable by the PBC radius neighbor list,
    and must not produce edges across frames of the batch."""
    frames = get_frames(dataset_pbc, n=2)
    batch = collate(frames, with_pbc=True)
    nl = atomic_data2neighbor_list(batch, rcut=6.0)
    idx = nl["index_mapping"]
    assert idx.shape[1] > 0
    assert torch.all(batch.batch[idx[0]] == batch.batch[idx[1]])
    assert torch.any(nl["cell_shifts"] != 0), "no edge crosses the box"


# --------------------------------------------------------------------------- #
# end to end through the lightning data module
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("part", ["train", "val"])
def test_datamodule_pbc_batches(part):
    dm = H5DataModule(
        h5_file_path=str(H5_PATH),
        partition_options=load_partition_options(),
        loading_options={"hdf_key_mapping": dict(HDF_KEY_MAPPING)},
        with_pbc=True,
    )
    # H5DataModule does not expose the collater device (defaults to cuda)
    dm._collater_fn.device = "cpu"
    dm.setup("fit")
    loader = dm.train_dataloader() if part == "train" else dm.val_dataloader()
    n_seen = 0
    for batch in loader:
        n_frames = len(batch.n_atoms)
        assert n_frames == BATCH_SIZE
        assert batch.pos.shape == (int(batch.n_atoms.sum()), 3)
        assert batch.cell.shape == (n_frames, 3, 3)
        assert batch.pbc.shape == (n_frames, 3)
        assert batch.pbc.all()
        assert torch.all(batch.cell.diagonal(dim1=1, dim2=2) > 0)
        n_seen += 1
    assert n_seen == N_BATCHES
