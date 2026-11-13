import numpy as np
import pytest
import torch
from torch_geometric.loader.dataloader import Collater
from mlcg.data import AtomicData
from mlcg.datasets.h5_dataset import MolData, MetaSet
from mlcg.neighbor_list.neighbor_list import atomic_data2neighbor_list
import tempfile
import torch
import h5py

np.random.seed(54392)

excluded_key_mapping = {
    "embeds": "attrs:cg_embeds",
    "coords": "cg_coords",
    "forces": "cg_delta_forces",
    "excluded_pairs": "attrs:excluded_pairs",
}


def make_hdf5(tdir, detailed_idx=True, excluded_order=None):
    # random 10 molecules of random length

    mol_list = []
    mol_frames = []
    f = h5py.File(tdir + "test_h5.hd5f", "w")
    for i in range(10):
        n_atoms = np.random.randint(2, high=11)
        n_frames = np.random.randint(10, high=25)
        name = str(i)
        mol_list.append(name)
        mol_frames.append(n_frames)
        coords = np.random.randn(n_frames, n_atoms, 3)
        forces = np.random.randn(n_frames, n_atoms, 3)
        noise_levels = np.random.randn(n_frames, n_atoms, 3)
        types = np.random.randn(n_atoms)
        pbcs = np.full((n_frames, 3), False)
        cells = np.random.randn(n_frames, 3, 3)

        grp = f.create_group(name)
        grp.create_dataset("cg_coords", data=coords)
        grp.create_dataset("cg_delta_forces", data=forces)
        grp.create_dataset("noise_levels", data=noise_levels)
        grp.attrs["cg_embeds"] = types
        grp.attrs["N_frames"] = n_frames
        grp.create_dataset("pbc", data=pbcs)
        grp.create_dataset("cell", data=cells)
        bonds = np.stack([np.arange(n_atoms - 1), np.arange(1, n_atoms)])
        grp.attrs["excluded_pairs"] = np.concatenate([bonds, bonds[::-1]], 1)

    if excluded_order is not None:
        f.attrs["excluded_order"] = excluded_order

    if detailed_idx:
        detailed_idx = {
            mol_list[i]: np.random.randint(
                0, high=(mol_frames[i] - 1), size=(10,)
            )
            for i in range(len(mol_list))
        }
    else:
        detailed_idx = None

    return f, mol_list, detailed_idx


def zero_force_transform(inp: AtomicData) -> AtomicData:
    """Dummy transform that set the forces to zero."""
    inp.forces = torch.zeros_like(inp.forces)
    return inp


def test_metaset_props():
    # Test attributes and trimming
    with tempfile.TemporaryDirectory() as tdir:
        f, mol_list, _ = make_hdf5(tdir)
        meta_set = MetaSet.create_from_hdf5_group(f, mol_list)
        assert meta_set.n_mol == 10
        for i, mol in enumerate(mol_list):
            assert meta_set._mol_map[mol] == i
        assert meta_set.n_total_samples == sum(
            [f[mol].attrs["N_frames"] for mol in mol_list]
        )
        # trim 10 frames
        pre_trim_size = meta_set.n_total_samples
        meta_set.trim_down_to(pre_trim_size - 10, verbose=False)
        assert meta_set.n_total_samples == pre_trim_size - 10

    # Test index preselection
    with tempfile.TemporaryDirectory() as tdir:
        f, mol_list, detailed_idx = make_hdf5(tdir, detailed_idx=True)
        sub_coords = [
            f[name]["cg_coords"][:][detailed_idx[name]] for name in mol_list
        ]
        sub_forces = [
            f[name]["cg_delta_forces"][:][detailed_idx[name]]
            for name in mol_list
        ]
        meta_set = MetaSet.create_from_hdf5_group(
            f, mol_list, detailed_indices=detailed_idx
        )
        assert meta_set.n_total_samples == sum(
            [len(detailed_idx[name]) for name in mol_list]
        )
        csum = np.cumsum([len(detailed_idx[name]) for name in mol_list])
        np.testing.assert_equal(meta_set._cumulate_indices, csum)

        # test correct coords/forces indexed
        for i, name in enumerate(mol_list):
            mol_id = meta_set._mol_map[name]
            np.testing.assert_equal(
                sub_coords[i], meta_set._mol_dataset[mol_id]._coords
            )
            np.testing.assert_equal(
                sub_forces[i], meta_set._mol_dataset[mol_id]._forces
            )


def test_excluded_pairs_loading():
    with tempfile.TemporaryDirectory() as tdir:
        f, mol_list, _ = make_hdf5(tdir, excluded_order=2)
        meta_set = MetaSet.create_from_hdf5_group(
            f, mol_list, hdf_key_mapping=excluded_key_mapping
        )
        samples = [meta_set[i] for i in range(0, len(meta_set), 7)]
        for sample in samples:
            assert sample.excluded_pairs["excluded_order"] == 2
            assert sample.excluded_pairs["index_mapping"].dtype == torch.long
        batch = Collater(None, None)(samples)

        num_nodes = batch.pos.shape[0]
        offsets = torch.cumsum(batch.n_atoms, 0) - batch.n_atoms
        expected = torch.cat(
            [
                s.excluded_pairs["index_mapping"] + off
                for s, off in zip(samples, offsets)
            ],
            dim=1,
        )
        torch.testing.assert_close(
            batch.excluded_pairs["index_mapping"], expected
        )

        nl = atomic_data2neighbor_list(batch, rcut=100.0)
        edge_key = nl["index_mapping"][0] * num_nodes + nl["index_mapping"][1]
        excluded_key = expected[0] * num_nodes + expected[1]
        assert nl["excluded_order"] == 2
        assert not torch.isin(excluded_key, edge_key).any()


def test_excluded_pairs_not_requested():
    with tempfile.TemporaryDirectory() as tdir:
        f, mol_list, _ = make_hdf5(tdir, excluded_order=2)
        meta_set = MetaSet.create_from_hdf5_group(f, mol_list)
        assert "excluded_pairs" not in meta_set[0]


def test_excluded_pairs_without_excluded_order():
    with tempfile.TemporaryDirectory() as tdir:
        f, mol_list, _ = make_hdf5(tdir)
        with pytest.raises(ValueError):
            MetaSet.create_from_hdf5_group(
                f, mol_list, hdf_key_mapping=excluded_key_mapping
            )


def test_h5_dataset_transform():
    with tempfile.TemporaryDirectory() as tdir:
        f, mol_list, _ = make_hdf5(tdir)
        meta_set = MetaSet.create_from_hdf5_group(
            f, mol_list, transform=zero_force_transform
        )
        random_indices = np.random.choice(len(meta_set), 10)
        for i in random_indices:
            assert np.allclose(meta_set[i].forces.detach().numpy(), 0.0)
