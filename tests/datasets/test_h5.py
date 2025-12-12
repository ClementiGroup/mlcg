import numpy as np
from mlcg.datasets.h5_dataset import MolData, MetaSet
import tempfile
import h5py

np.random.seed(54392)


def make_hdf5(tdir, detailed_idx=True):
    # random 10 molecules of random length

    mol_list = []
    mol_frames = []
    f = h5py.File(tdir + "test_h5.hd5f", "w")
    for i in range(10):
        n_atoms = np.random.randint(2, high=11)
        n_frames = np.random.randint(10, high=25)
        name = np.random.randint(1, high=1000)
        name = str(name)
        mol_list.append(name)
        mol_frames.append(n_frames)
        coords = np.random.randn(n_frames, n_atoms, 3)
        forces = np.random.randn(n_frames, n_atoms, 3)
        types = np.random.randn(n_atoms)

        grp = f.create_group(name)
        grp.create_dataset("cg_coords", data=coords)
        grp.create_dataset("cg_delta_forces", data=coords)
        grp.attrs["cg_embeds"] = types
        grp.attrs["N_frames"] = n_frames

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
