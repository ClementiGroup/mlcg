r"""Python classes for processing data stored in HDF5 format.

HDF5 format benefits the dataset management for mlcg-tools when training/validation involves multiple molecules of vastly different lengths and when parallelization is used.
The main features are:
1. The internal structure mimics the hierarchy of the dataset itself, such that we don't have to replicate it on filesystem.
2. we don't have to actively open all files in the process
3. we can transparently load only the necessary part of the dataset to the memory

This file contains the python data structures for handling the HDF5 file and its content, i.e., the coordinates, forces and embedding vectors for multiple CG molecules.
An example HDF5 file structure and correponding class types after loading:

.. code-block::

	/ (HDF-group, => `H5Dataset._h5_root`)
	├── OPEP (HDF-group =(part, according to "partition_options")=> `Metaset` in a `Partition`)
        │   ├── opep_0000 (HDF-group, => `MolData`)
	│   │   ├── attrs (HDF-attributes of the molecule "/OPEP/opep_0000")
	│   │   │   ├── cg_embeds (a 1-d numpy.int array)
	│   │   │   ├── N_frames (int, number of frames = size of cg_coords on axis 0)
	│   │   │   ... (optional, e.g., cg_top, cg_pdb, etc that corrsponds to the molecule)
	│   │   ├── cg_coords (HDF-dataset of the molecule "/OPEP/opep_0000", 3-d numpy.float32 array)
	│   │   └── cg(_delta)_forces (HDF-dataset of the molecule "/OPEP/opep_0000", 3-d numpy.float32 array)
	│   ... (other molecules in "/OPEP")
	├── CATH (HDF-group ="partition_options"=> `Metaset` in a `Partition`)
	...


> Data structure `MolData` is the basic brick of the dataset.
It holds embeds, coords and forces of certain number of frames for a single molecule.
.sub_sample: method for performing indexing on the frames.

> Data strcuture `Metaset` holds multiple molecules and joins their frame indices together.
One can access frames of underlying MolData objects with a continuous index.
The idea is that the molecules in a `Metaset` have similar numbers of CG beads, such that the sample requires similar processing time when passing through a neural network.
When this rule is enforced, it will allow automatic balancing of batch composition during training and validation.
- .create_from_hdf5_group: initiate a Metaset by loading data from given HDF-group.
mol_list, detailed_indices, hdf_key_mapping, stride and parallel can control which subset is loaded to the Metaset (details see the description of "H5Dataset")
- .trim_down_to: drop frames of underlying for obtaining a subset with desired number of frames.
The indices of frames to be dropped are controlled by a random number generator. When the parameter "random_seed" and the MolData order and number of frames are the same, the frames to be dropped will be reproducible.
- len() (__len__): return total number of samples
- [] (__get_item__): infer the molecule and get the corresponding frame according to the given unified index. Return value is grouped by an `AtomicData` object.


> Data structure `Partition` can hold multiple Metaset for training and validation purposes.
Its main function is to automatic adjust (subsample) the Metaset(s) and the underlying MolData to form a balanced data source, from which a certain number of batches can be drawn. Each batch will contain a pre-given number of samples from each Metaset.
One or several `Partition`s can be created to load part of the same HDF5 file into the memory.
The exact content inside a `Partition` object is controlled by the `partition_options` as a parameter for initializing a `H5Dataset`.

> Data strcuture `H5Dataset` opens a HDF5 file and establish the partitions.
`partition_options` describe what partitions to create and what content to load into them. (Detailed description and examples are as followed.)
`loading_options` mainly deal with the HDF key mapping (which datasets/attributes corresponds to the coordinates, forces and embeddings) as well as (optionally) the information for correctly split the dataset in a parallel training/validation.

An example "partition_options" (as a Python Mappable (e.g., dict)):

.. code-block::

	{
		"train": {
			"metasets": {
				"OPEP": {
					"molecules": [
						"opep_0000",
						"opep_0001",
						...
					],
					"stride": 1, # optional, default 1
					"detailed_indices": {
                        # optional, providing the indices of frames to work with (before striding and splitting for parallel processes).
						# optional,
                            "opep_0000":
                                val_ratio: 0.1
                                test_ratio: 0.1
                                seed: 12345
                            "opep_0001":
                                val_ratio: 0.1
                                test_ratio: 0.1
                                seed: 12345
                            "filename": ./splits

                        # If detailed_indices are not provided for a given molecule, then it is equivalent to np.arange(N_frames)
						    "opep_0000": [1, 3, 5, 7, 9, ...],


					},
				},
				"CATH": {
					"molecules": [
						"cath_1b43A02",
						...
					],
					"stride": 1, # optional
					"detailed_indices": {}, # optional
				}
			},
			"batch_size": {
				# each batch will contain 24 samples from
				# The two metasets will be trimmed down according to this ratio
				# so optimally it should be proportional to the ratio of numbers of frames in the metasets.
				"OPEP": 24,
				"CATH": 6,
			},
			"subsample_random_seed": 42, # optional, default 42. Random seed for trimming down the frames.
			"max_epoch_samples": None, # optional, default None. For setting a desired dataset size.
			# Works by subsampling the dataset after it is loaded with the given stride.
		},
		"val": { # similar
		}
	}

An example "loading_options" (as a Python Mappable (e.g., dict)):

.. code-block::

	{
		"hdf_key_mapping": {
			# keys for reading CG data from HDF5 groups
			"embeds": "attrs:cg_embeds",
			"coords": "cg_coords",
			"forces": "cg_delta_forces",
		},
		"parallel": { # optional, default rank=0 and world_size=1 (single process).
			# For split the dataset evenly and load only the necessary parts to each process in a parallelized train/val setup
			"rank": 0, # which process is this
			"world_size": 1, # how many processes are there
		}
	}


> Class `H5PartitionDataLoader` mimics the pytorch data loader and can generate training/validation batches with fixed proportion of samples from several Metasets in a Partition.
The proportion and batch size is defined when the partition is initialized.
When the molecules in each Metaset have similar embedding vector lengths, the processing of output batches will require a more or less fixed size of VRAM on GPU, which can benefit the

Note:
1. Usually in a train-val split, each molecule goes to either the train or the test partition.
     In some special cases (e.g., non-transferable training) one molecule can be part of both partitions.
     In this situation, "detailed_indices" can be set to assign the corresponding frames to the desired partitions.
     In addition one can pass in detailed_indices as a dictionary to split frames based on training/test (see partition_options example above)
"""

import h5py
import numpy as np
import torch
import typing
import itertools
import warnings
from torch_geometric.loader.dataloader import Collater as PyGCollater
import torch_geometric.loader.dataloader  # for type hint
from typing import Dict, List, Optional, Sequence, Callable
from mlcg.data import AtomicData
from mlcg.utils import make_splits, calc_num_samples, tqdm
from copy import copy as shallow_copy


class MolData:
    """Data-holder for coordinates, forces and embedding vector of a molecule, e.g., opep_0000.

    Parameters
    ----------
    name:
        Name of the molecule
    embeds:
        Type embeddings for each atom, of shape `(n_atoms,)`
    coords:
        Cartesian coordinates of the molecule, of shape `(n_frames, n_atoms, 3)`
    forces:
        Cartesian forces of the molecule, of shape `(n_frames, n_atoms, 3)`
    """

    def __init__(
        self,
        name: str,
        embeds: np.ndarray,
        coords: np.ndarray,
        forces: np.ndarray,
        weights: np.ndarray = None,
        noise_levels: np.ndarray = None,
        neighbor_list: dict = None,
    ):
        self._name = name
        self._embeds = embeds
        self._coords = coords
        self._forces = forces

        assert (
            len(self._embeds) == self._coords.shape[1] == self._forces.shape[1]
        )
        assert self._coords.shape == self._forces.shape

        self._weights = weights
        if self._weights is not None:
            assert len(self._coords) == len(self._weights)
        self._noise_levels = noise_levels
        if self._noise_levels is not None:
            assert len(self._coords) == len(self._noise_levels)
        self._neighbor_list = neighbor_list

    @property
    def name(self):
        return self._name

    @property
    def embeds(self):
        return self._embeds

    @property
    def coords(self):
        return self._coords

    @property
    def forces(self):
        return self._forces

    @property
    def weights(self):
        return self._weights

    @property
    def noise_levels(self):
        return self._noise_levels

    @property
    def neighbor_list(self):
        return self._neighbor_list

    @property
    def n_frames(self):
        return self.coords.shape[0]

    @property
    def n_beads(self):
        return self.coords.shape[1]

    def __repr__(self):
        return f"""MolData(name={self.name}", N_beads={self.n_beads}, N_frames={self.n_frames})"""

    def sub_sample(self, indices):
        self._coords = self._coords[indices]
        self._forces = self._forces[indices]

        if self.weights is not None:
            self._weights = self._weights[indices]
        if self._noise_levels is not None:
            self._noise_levels = self._noise_levels[indices]


class MetaSet:
    """Set of MolData instances for molecules with similar characterstics

    Parameters
    ----------
    name:
        Name of the metaset

    """

    def __init__(
        self,
        name: str,
        transform: Optional[Callable] = None,
    ):
        self.name = name
        self._mol_dataset = []
        self._mol_map = {}
        self._n_mol_samples = []
        self._cumulate_indices = [0]
        self._transform = transform

    @staticmethod
    def retrieve_hdf(hdf_grp, hdf_key):
        """Unified hdf retriever for attributes and dataset."""

        def is_attr(hdf_key):
            return hdf_key[:6] == "attrs:"

        if is_attr(hdf_key):
            return hdf_grp.attrs[hdf_key[6:]]
        else:
            return hdf_grp[hdf_key]

    @staticmethod
    def grab_n_frames(
        hdf5_group,
        mol_name,
    ):
        """
        Returns number of frames for each mol name
        """

        return MetaSet.retrieve_hdf(hdf5_group[mol_name], "attrs:N_frames")

    @staticmethod
    def create_from_hdf5_group(
        hdf5_group,
        mol_list,
        detailed_indices=None,
        stride=1,
        hdf_key_mapping={
            "embeds": "attrs:cg_embeds",
            "coords": "cg_coords",
            "forces": "cg_delta_forces",
            "weights": "subsampling_weights",
            "noise_levels": "noise_levels",
        },
        parallel={
            "rank": 0,
            "world_size": 1,
        },
        subsample_using_weights=False,
        transform: Optional[Callable] = None,
        mol_neighbor_lists: Optional[dict] = None,
        overriding_offset: Optional[int] = None,
    ):
        def select_for_rank(length_or_indices):
            """Return a slicing for loading the necessary data from the HDF dataset."""
            from numbers import Integral

            if isinstance(length_or_indices, Integral):  # int or np.int*
                indices = None
                length = length_or_indices
            else:
                indices = length_or_indices
                length = len(indices)
            rank = parallel.get("rank", 0)
            world_size = parallel.get("world_size", 1)
            if overriding_offset is not None and world_size > 1:
                raise ValueError(
                    f"overriding_offset ({overriding_offset}) "
                    f"should be left None when world_size is "
                    f"{world_size} > 1"
                )
            if rank < 0 or rank >= world_size:
                raise ValueError(
                    "Rank %d is invalid given the world_size %d."
                    % (rank, world_size)
                )
            if length < world_size:
                raise ValueError(
                    "Length %d is too short given the world_size %d."
                    % (length, world_size)
                )
            residue = length % world_size
            if overriding_offset is not None:
                offset = overriding_offset
            else:
                offset = residue + rank
            if indices is not None:
                return indices[offset :: (world_size * stride)]
            else:
                return np.s_[offset :: (world_size * stride)]

        output = MetaSet(hdf5_group.name.split("/")[-1], transform=transform)
        keys = hdf_key_mapping
        for mol_name in tqdm(mol_list, leave=False, desc="loading molecules"):
            if mol_name not in hdf5_group:
                raise KeyError(
                    "`mol_list` includes moleucle %s,"
                    " which is not in the h5 group at path %s"
                    % (mol_name, hdf5_group.name)
                )
            embeds = MetaSet.retrieve_hdf(hdf5_group[mol_name], keys["embeds"])
            if (
                detailed_indices is not None
                and detailed_indices.get(mol_name) is not None
            ):
                par_range = detailed_indices[mol_name]
                # the indices could be a string pointing to a npy file, which in turn
                # contains the real indices. We need to deal with that.
                if isinstance(par_range, str):
                    # path to an array of detailed_indices?
                    par_range = np.load(par_range)
                split_per_index = True
            else:
                par_range = MetaSet.retrieve_hdf(
                    hdf5_group[mol_name], "attrs:N_frames"
                )
                split_per_index = False
            selection = select_for_rank(par_range)
            if not split_per_index:
                coords = MetaSet.retrieve_hdf(
                    hdf5_group[mol_name], keys["coords"]
                )[selection]
                forces = MetaSet.retrieve_hdf(
                    hdf5_group[mol_name], keys["forces"]
                )[selection]
                if "noise_levels" in keys:
                    noise_levels = MetaSet.retrieve_hdf(
                        hdf5_group[mol_name], keys["noise_levels"]
                    )[selection]
                else:
                    noise_levels = None
                weights = None  # Weights are None by default
                if subsample_using_weights is True:
                    weights = MetaSet.retrieve_hdf(
                        hdf5_group[mol_name], keys["weights"]
                    )[selection]
            else:
                # For large dataset it is usually quicker to first load everything
                # and then perform indexing in memory
                coords = MetaSet.retrieve_hdf(
                    hdf5_group[mol_name], keys["coords"]
                )[:][selection]
                forces = MetaSet.retrieve_hdf(
                    hdf5_group[mol_name], keys["forces"]
                )[:][selection]
                if "noise_levels" in keys:
                    noise_levels = MetaSet.retrieve_hdf(
                        hdf5_group[mol_name], keys["noise_levels"]
                    )[:][selection]
                else:
                    noise_levels = None
                weights = None  # Weights are None by default
                if subsample_using_weights is True:
                    weights = MetaSet.retrieve_hdf(
                        hdf5_group[mol_name], keys["weights"]
                    )[:][selection]
            neighbor_list = None
            if mol_neighbor_lists is not None:
                if mol_name in mol_neighbor_lists:
                    neighbor_list = mol_neighbor_lists[mol_name]
                elif "_default" in mol_neighbor_lists:
                    neighbor_list = mol_neighbor_lists["_default"]
            output.insert_mol(
                MolData(
                    mol_name,
                    embeds,
                    coords,
                    forces,
                    weights=weights,
                    noise_levels=noise_levels,
                    neighbor_list=neighbor_list,
                )
            )
        return output

    def set_transform(
        self,
        transform: Optional[Callable] = None,
    ):
        self._transform = transform

    def insert_mol(self, mol_data):
        self._mol_dataset.append(mol_data)
        self._mol_map[mol_data.name] = self.n_mol - 1
        self._update_info()

    def trim_down_to(self, target_n_samples, random_seed=42, verbose=True):
        """Trimming all datasets randomly to reach the target number of samples.
        MolData attributes of the MetaSet are permanently subsampled by this
        method.
        """
        if target_n_samples > self.n_total_samples or target_n_samples <= 0:
            raise ValueError("Target number of samples invalid")
        elif target_n_samples == self.n_total_samples:
            return
        if verbose:
            warnings.warn(
                f"\nMetaset {self.name}: subsampling to {target_n_samples} samples with random seed {random_seed}. "
                f"\n({self.n_total_samples - target_n_samples}/{self.n_total_samples}) samples will be removed."
            )
        rng = np.random.default_rng(random_seed)
        # get the indices of samples to keep
        indices = rng.choice(
            self.n_total_samples, size=target_n_samples, replace=False
        )
        indices.sort()
        # get the indices for each molecule
        base_index = 0
        left_bound = 0
        for i, next_start_index in enumerate(self._cumulate_indices):
            right_bound = np.searchsorted(indices, next_start_index)
            # subsample the molecular record
            self._mol_dataset[i].sub_sample(
                indices[left_bound:right_bound] - base_index
            )
            base_index = next_start_index
            left_bound = right_bound
        # update the n_mol_samples and cumulate_indices
        self._update_info()

    def _update_info(self):
        self._n_mol_samples = np.array(
            [mol_d.n_frames for mol_d in self._mol_dataset]
        )
        self._cumulate_indices = np.cumsum(self._n_mol_samples)

        if self._weights_exist():
            ## If weights exist for all
            self._make_cumulative_weights()
            print(
                f"cumulative weights have shape {self._cumulative_weights.shape}"
            )

    @property
    def n_mol(self):
        return len(self._mol_dataset)
        return all([mol_d.weights is not None for mol_d in self._mol_dataset])

    @property
    def n_total_samples(self):
        return self._cumulate_indices[-1]

    @property
    def n_mol_samples(self):
        return self._n_mol_samples

    def _weights_exist(self):
        """Checks if _weights is an attribute for all molecules in dataset"""
        return np.all([(mol_d._weights != None) for mol_d in self._mol_dataset])

    def _make_cumulative_weights(self):
        # Concatenate weights from all MolData objects
        self._cumulative_weights = np.concatenate(
            [mol_d._weights for mol_d in self._mol_dataset]
        )
        # Check if length of _cumulative_weights is correct
        assert (
            len(self._cumulative_weights) == self.n_total_samples
        ), "Number of weights does not match number of samples"
        # Set all inf weights to max value
        self._cumulative_weights[self._cumulative_weights == np.inf] = np.max(
            self._cumulative_weights[self._cumulative_weights != np.inf]
        )
        # Check max and min values of weights array
        assert (self._cumulative_weights.max() != np.inf) and (
            (self._cumulative_weights.min() >= 0)
        ), "Smallest weight value is infinite OR some weight is negative"

    def get_mol_data_by_name(self, mol_name):
        index = self._mol_map.get(mol_name, None)
        if index is not None:
            return self._mol_dataset[index]

    def __getitem__(self, idx):
        dataset_id, data_id = self._locate_idx(idx)
        # print(type(self._mol_dataset[dataset_id].forces[data_id]))
        # print(type(self._mol_dataset[dataset_id].noise_levels[data_id:(data_id + 1)]))
        # exit()
        mol_data = self._mol_dataset[dataset_id]
        point_tensors = {
            "pos": mol_data.coords[data_id],
            "forces": mol_data.forces[data_id],
            "atom_types": mol_data.embeds,
        }
        if mol_data.noise_levels is not None:
            point_tensors["noise_levels"] = torch.as_tensor(
                np.tile(
                    self._mol_dataset[dataset_id].noise_levels[data_id],
                    [len(self._mol_dataset[dataset_id].coords[data_id])],
                )
            )
        if mol_data.neighbor_list is not None:
            # AtomicData.from_points uses "neighborlist" in args
            point_tensors["neighborlist"] = mol_data.neighbor_list
        ad_point = AtomicData.from_points(**point_tensors)
        if self._transform is not None:
            ad_point = self._transform(ad_point)
        return ad_point

    def _locate_idx(self, idx):
        full_length = self._cumulate_indices[-1]
        if idx < 0:
            idx += full_length
        if idx < 0 or idx >= full_length:
            raise IndexError("Given idx is outside of the dataset.")
        # find in which dataset it resides
        dataset_id = np.searchsorted(self._cumulate_indices, idx, side="right")
        # locate the position within the dataset
        if dataset_id == 0:
            datapt_id = idx
        else:
            datapt_id = idx - self._cumulate_indices[dataset_id - 1]
        return dataset_id, datapt_id

    def __len__(self):
        return self.n_total_samples

    def __repr__(self):
        return f"Metaset with {self.n_mol} molecules and {self.n_total_samples} samples"


class Partition:
    """Contain several metasets for a certain purpose, e.g., training.

    Parameters
    ----------
    name:
        name of the partition
    """

    def __init__(self, name: str):
        self.name = name
        self._metasets = {}
        self._sample_ready = False

    def add_metaset(self, metaset_name, metaset):
        self._metasets[metaset_name] = metaset
        self._sample_ready = False

    def get_metaset(self, metaset_name):
        assert self._sample_ready, "Sampling setup needs to be done first."
        return self._metasets.get(metaset_name, None)

    def get_metasets(self):
        assert self._sample_ready, "Sampling setup needs to be done first."
        return self._metasets

    def sampling_setup(
        self,
        batch_sizes: typing.Dict[str, MetaSet],
        max_epoch_samples=None,
        random_seed=42,
        verbose=True,
    ):
        """Calculate the number of batches available for an epoch according to
        the batch size for each metaset and optionally the maximum number of
        samples in an epoch. The molecular dataset will be trimmed accordingly.
        """
        if set(batch_sizes) != set(self._metasets):
            raise ValueError(
                "Input `batch_sizes` shall have exactly the same keys"
                " as the metasets in this H5Dataset: "
                + str(list(self._metasets.keys()))
            )
        self._batch_sizes = batch_sizes
        # calculating the number of batches
        sample_sizes = self.sample_sizes
        max_batches = [
            int(sample_sizes[k] // batch_sizes[k]) for k in batch_sizes
        ]
        n_batches = min(max_batches)
        trim_down_is_needed = any(
            [n_batches != desired_n for desired_n in max_batches]
        )
        trim_down_due_to_max_epoch_samples = False
        if max_epoch_samples is not None:
            max_epoch_batch = max_epoch_samples // sum(batch_sizes.values())
            if max_epoch_batch < n_batches:
                trim_down_is_needed = True
                trim_down_due_to_max_epoch_samples = True
                n_batches = max_epoch_batch
        return_info = {
            "total_size": {k: len(self._metasets[k]) for k in self._metasets}
        }
        if trim_down_is_needed:
            # trimming down the number of samples in each metaset accordingly
            for k in self._metasets:
                self._metasets[k].trim_down_to(
                    n_batches * batch_sizes[k],
                    random_seed=random_seed,
                    verbose=False,  # since we already warned above
                )
            # warn about the sampling
            warn_message = f"\nPartition `{self.name}`: The size of samples in given Metasets will be reduced via subsampling "
            if trim_down_due_to_max_epoch_samples:
                warn_message += "according to the `max_epoch_samples`.\n"
            else:
                warn_message += "according to desired batch composition.\n"
            batch_size_message = (
                ":".join(self._metasets.keys())
                + " = "
                + ":".join([str(batch_sizes[k]) for k in self._metasets])
            )
            warn_message += (
                f"This partition now contains {n_batches} mini batches, "
            )
            warn_message += "each containing " + batch_size_message + ".\n"
            warn_message += "Note for reproducibility: the subsampling (trimming down) of each Metasets is controlled by the molecule entries and frames, the `stride`, the ratio of `batch_sizes` as well as the `subsample_random_seed`."
            if verbose:
                warnings.warn(warn_message)
        self._sample_ready = True
        return_info["is_trimmed"] = trim_down_is_needed
        return_info["current_size"] = {
            k: len(self._metasets[k]) for k in self._metasets
        }
        return return_info

    @property
    def batch_sizes(self):
        assert self._sample_ready, "Sampling setup needs to be done first."
        return self._batch_sizes

    @property
    def sample_sizes(self):
        return {k: self._metasets[k].n_total_samples for k in self._metasets}

    def __repr__(self):
        return f"Partition `{self.name}` with Metasets:\n" + "\n".join(
            [
                f'- "{k}": {v} samples'
                for k, v in sorted(self.sample_sizes.items())
            ]
        )


class H5Dataset:
    """The top-level class for handling multiple datasets contained in a HDF5 file.

    Parameters
    ----------
    h5_file_path:
        Path to the hdf5 file containing the dataset(s)
    partition_options:
        Dictionary of partition names mapping to collections of metaset information
    loading_options:
        Dictionary of options specifying how hdf5 attrs/datasets are named
        within hd5 groups
    """

    def __init__(
        self,
        h5_file_path: str,
        partition_options: Dict,
        loading_options: Dict,
        subsample_using_weights: bool = False,
        transform: Optional[Callable] = None,
    ):
        self._h5_path = h5_file_path
        self._h5_root = h5py.File(h5_file_path, "r")
        self._metaset_entries = {}
        # ^ dict containing all metasets in the HDF5 file
        self._partitions = {}  # dict containing the configured metasets
        self._partition_sample_info = {}
        self._detailed_indices = {}
        self._subsample_using_weights = subsample_using_weights
        self._transform = transform
        if self._transform is not None:
            print("Using transform:", self._transform)
        # processing the hdf5 file
        for metaset_name in self._h5_root:
            self._metaset_entries[metaset_name] = self._h5_root[metaset_name]

        self._create_partitions(partition_options, loading_options)

    def _create_partitions(self, partition_options, loading_options):
        ## TODO: sanity check of the given dictionary

        hdf_key_mapping = loading_options.get("hdf_key_mapping")
        parallel = loading_options.get(
            "parallel",
            {
                "rank": 0,
                "world_size": 1,
            },
        )  # if no parallel entry, then target for single process
        overriding_offset = loading_options.get(
            "overriding_offset",
            None,
        )
        neighbor_list_chkpts = loading_options.get(
            "nl_chkpts",
            None,
        )
        if isinstance(neighbor_list_chkpts, dict):
            metaset_mol_neighbor_lists = {}
            for part_name, part_nl_info in neighbor_list_chkpts.items():
                if isinstance(part_nl_info, str):
                    # this is the case when the same neighborlist will be shared for the whole partition
                    metaset_mol_neighbor_lists[part_name] = {
                        "_default": torch.load(part_nl_info)
                    }
                elif isinstance(part_nl_info, dict):
                    # this is the case when different nls are assigned to different molecules in a partition
                    metaset_mol_neighbor_lists[part_name] = {
                        k: torch.load(v) for k, v in part_nl_info.items()
                    }
                else:
                    raise ValueError(
                        f"Expecting neighborlist for partition {part_name} to be either a path string or dictionary"
                    )
        elif isinstance(neighbor_list_chkpts, str):
            metaset_mol_neighbor_lists = {
                "_default": {"_default": torch.load(neighbor_list_chkpts)}
            }
        else:
            metaset_mol_neighbor_lists = None
        for part_name in partition_options:
            ## create a partition, typical names are train/val
            part = Partition(part_name)
            part_info = partition_options[part_name]
            # TODO: check option consistency
            for metaset_name in part_info["metasets"]:
                if metaset_name not in self._metaset_entries:
                    raise KeyError(
                        f"`partition_options` refer to metaset {part},"
                        f" which is not in the h5 dataset at path {self._h5_path}"
                    )
                mol_list = part_info["metasets"][metaset_name]["molecules"]
                input_detailed_indices = part_info["metasets"][
                    metaset_name
                ].get("detailed_indices", None)

                if input_detailed_indices is not None:
                    if isinstance(input_detailed_indices, Dict):
                        is_simple_detailed_indices = False
                        for mol_name in mol_list:
                            if isinstance(
                                input_detailed_indices[mol_name], Sequence
                            ):
                                is_simple_detailed_indices = True
                        if is_simple_detailed_indices:
                            detailed_indices = dict(input_detailed_indices)
                        else:
                            detailed_indices = (
                                self.training_validation_splitting(
                                    input_detailed_indices,
                                    part_name,
                                    metaset_name,
                                    mol_list,
                                )
                            )
                    elif isinstance(input_detailed_indices, Sequence):
                        assert len(mol_list) == 1
                        detailed_indices = {mol_list[0]: input_detailed_indices}
                else:
                    detailed_indices = None

                stride = part_info["metasets"][metaset_name].get("stride", 1)
                if metaset_mol_neighbor_lists is not None:
                    if metaset_name in metaset_mol_neighbor_lists:
                        mol_neighbor_lists = metaset_mol_neighbor_lists[
                            metaset_name
                        ]
                    else:
                        mol_neighbor_lists = metaset_mol_neighbor_lists[
                            "_default"
                        ]
                else:
                    mol_neighbor_lists = None
                part.add_metaset(
                    metaset_name,
                    MetaSet.create_from_hdf5_group(
                        self._metaset_entries[metaset_name],
                        mol_list,
                        detailed_indices,
                        stride=stride,
                        hdf_key_mapping=hdf_key_mapping,
                        parallel=parallel,
                        subsample_using_weights=self._subsample_using_weights,
                        transform=self._transform,
                        mol_neighbor_lists=mol_neighbor_lists,
                        overriding_offset=overriding_offset,
                    ),
                )
            ## trim the metasets to fit the need of sampling
            random_seed = part_info.get("subsample_random_seed", 42)
            max_epoch_samples = part_info.get("max_epoch_samples", None)
            if max_epoch_samples is not None and parallel["world_size"] > 1:
                max_epoch_samples = int(
                    max_epoch_samples // parallel["world_size"]
                )
            self._partition_sample_info[part_name] = part.sampling_setup(
                part_info["batch_sizes"],
                max_epoch_samples,
                random_seed=random_seed,
                verbose=(
                    parallel["rank"] == 0
                ),  # show trimming info only on rank 0
            )
            self._partitions[part_name] = part

    def __del__(self):
        self._h5_root.close()

    @property
    def metaset_hdf_entries(self):
        return self._metaset_entries

    def partition(self, partition_name):
        return self._partitions.get(partition_name, None)

    @property
    def partition_sample_info(self):
        return self._partition_sample_info

    def __repr__(self):
        return (
            'H5Dataset:\nPath: "%s"\nPartitions:\n' % self._h5_path
            + "\n".join(['- "' + str(part) + '"' for part in self._partitions])
        )

    def training_validation_splitting(
        self,
        input_detailed_indices: Dict,
        part_name: str,
        metaset_name: str,
        mol_list: List,
    ):
        """
        Option to split molecule in metaset frame by frame into training or validation
        Inputs:
            input_detailed_indices --
                dictionary read in from yaml file about how data should be split
                    must contain 3 primary keys [val_ratio, test_ratio, seed]
                    additional option to write out to filename if in dict.keys()
            part_name --
                which partition is currently being examined
            metaset_name --
                global name describing class of molecules
            mol_list --
                names of molecules

        Outputs:
            self._detailed_indices[part_name][metaset_name] --
                pass back indices according to queried metaset and partition
        """
        if len(self._detailed_indices) == 0:
            # Split into training and validation at first instance,
            #   write out if specified in yaml and save for later reference
            for partition_key in ["train", "val", "test"]:
                self._detailed_indices[partition_key] = {}
                self._detailed_indices[partition_key][metaset_name] = {}

            for mol_name in mol_list:
                # if isinstance(input_detailed_indices[mol_name], Sequence):

                if (
                    mol_name
                    not in self._detailed_indices["train"][metaset_name]
                ):
                    required_keys = [
                        "val_ratio",
                        "test_ratio",
                        "seed",
                    ]
                    for required_key in required_keys:
                        assert (
                            required_key in input_detailed_indices[mol_name]
                        ), "{} not found in yaml file".format(required_key)
                    dataset_len = MetaSet.grab_n_frames(
                        self._metaset_entries[metaset_name],
                        mol_name,
                    )
                    idx_train, idx_val, idx_test = make_splits(
                        dataset_len, **input_detailed_indices[mol_name]
                    )
                    ## make_splits returns as tensor but can only index with numpy
                    ##   inside create_hdf5_group
                    self._detailed_indices["train"][metaset_name][
                        mol_name
                    ] = idx_train.sort().values.numpy()
                    self._detailed_indices["val"][metaset_name][
                        mol_name
                    ] = idx_val.sort().values.numpy()
                    self._detailed_indices["test"][metaset_name][
                        mol_name
                    ] = idx_test.sort().values.numpy()
            if "filename" in input_detailed_indices:
                np.save(
                    input_detailed_indices["filename"],
                    self._detailed_indices,
                )
            assert (
                part_name in self._detailed_indices
            ), "Parition naming not train, val, or test. Check partition_options yaml file"
        return self._detailed_indices[part_name][metaset_name]


class H5SimpleDataset(H5Dataset):
    """The top-level class for handling a single dataset contained in a HDF5 file.
    Will only load from one single type of molecules (i.e., one Metaset), and do
    not support partition splits.
    Use .get_dataloader for obtaining a dataloader for PyTorch training, etc.

    Parameters
    ----------
    h5_file_path:
        Path to the hdf5 file containing the dataset(s)
    stride: default 1
        Stride for loading the frames
    detailed_indices: default `None`
        Set this to manually define which frames to be included for each molecule.
    metaset_name: default `None`
        the name of the h5 group containing the molecule data. If kept `None` and
        the given file consists of only one metaset, this parameter will be
        inferred from the file.
    mol_list: default `None`
        A list of the molecules to be loaded. When kept `None`, all molecules will
        be loaded.
    hdf_key_mapping: default loading the delta forces
        Key mapping for reading the data from h5 file.
    parallel: default for single process
        For DDP parallelism. Details see the head of this file.
    """

    def __init__(
        self,
        h5_file_path: str,
        stride: int = 1,
        detailed_indices: Optional[Dict[str, List]] = None,
        metaset_name: str = None,
        mol_list: Optional[List[str]] = None,
        hdf_key_mapping: Dict[str, str] = {
            "embeds": "attrs:cg_embeds",
            "coords": "cg_coords",
            "forces": "cg_delta_forces",
        },
        parallel={"rank": 0, "world_size": 1},
        subsample_using_weights: Optional[bool] = False,
        transform: Optional[Callable] = None,
    ):
        # input checking
        if not isinstance(stride, int) and stride > 0:
            raise ValueError("Parameter `stride` is invalid.")
        # load the h5 file
        self._h5_path = h5_file_path
        self._h5_root = h5py.File(h5_file_path, "r")
        self._metaset_entries = {}
        # ^ dict containing all metasets in the HDF5 file
        self._partitions = None
        self._partition_sample_info = None
        self._transform = transform

        # checking metaset
        all_metasets = list(self._h5_root.keys())
        if metaset_name == None:
            if len(all_metasets) != 1:
                raise ValueError(
                    f"Given h5 dataset contains multiple entries:\n"
                    f"{all_metasets}\n"
                    f"Please select one via `metaset_name` or use "
                    f"H5Dataset instead."
                )
            else:
                metaset_name = all_metasets[0]
        else:
            if metaset_name not in all_metasets:
                raise ValueError(
                    f"Metaset {metaset_name} not found in given "
                    f"h5 dataset. The following are available:\n"
                    f"{all_metasets}."
                )
        self.metaset_name = metaset_name
        if mol_list is None:
            mol_list = list(self._h5_root[metaset_name].keys())

        # load molecules
        self._metaset_entries[metaset_name] = self._h5_root[metaset_name]
        self.metaset = MetaSet.create_from_hdf5_group(
            self._h5_root[metaset_name],
            mol_list,
            detailed_indices=detailed_indices,
            stride=stride,
            hdf_key_mapping=hdf_key_mapping,
            parallel=parallel,
            subsample_using_weights=subsample_using_weights,
            transform=self._transform,
        )

    def get_dataloader(
        self,
        batch_size: int,
        collater_fn: torch_geometric.loader.dataloader.Collater = PyGCollater(
            None, None
        ),
        shuffle: bool = True,
        pin_memory: bool = False,
    ):
        """
        Parameters:
        -----------
        batch_size:
        Size of the batches to draw from the metaset
        collater_fn, shuffle, pin_memory:
        See PyTorch documentations for dataloader options.
        """
        data_loader = H5MetasetDataLoader(
            self.metaset,
            batch_size,
            collater_fn=collater_fn,
            shuffle=shuffle,
            pin_memory=pin_memory,
        )
        return data_loader

    @property
    def n_mol(self):
        return self.metaset.n_mol

    @property
    def n_total_samples(self):
        return self.metaset.n_total_samples

    @property
    def n_mol_samples(self):
        return self.metaset.n_mol_samples

    def __len__(self):
        return len(self.metaset)

    def __getitem__(self, idx):
        return self.metaset[idx]

    def __repr__(self):
        return (
            f'H5Dataset:\nPath: "{self._h5_path}"\nMetaset:\n'
            f"- {self.metaset_name}: {self.metaset}"
        )


class H5PartitionDataLoader:
    """Load batches from one or multiple Metasets in a Partition.
    In multiple Metasets scenario, the order of data loaders will be
    alphabetically ascending with respect to the Metaset names."""

    def __init__(
        self,
        data_partition,
        collater_fn=PyGCollater(None, None),
        pin_memory=False,
        subsample_using_weights=False,
        transform=None,
    ):
        self._data_part = data_partition
        self._metasets = []
        self._samplers = []
        self._pin_memory = pin_memory
        for metaset_name, batch_size in sorted(
            data_partition.batch_sizes.items()
        ):
            metaset = data_partition.get_metaset(metaset_name)
            # ^ automatically checks whether the partition is sample_ready()
            self._metasets.append(metaset)
            if subsample_using_weights is False:
                s = torch.utils.data.RandomSampler(metaset)
            elif subsample_using_weights is True:
                metaset._update_info()
                s = torch.utils.data.WeightedRandomSampler(
                    metaset._cumulative_weights,
                    num_samples=calc_num_samples(metaset._cumulative_weights),
                    replacement=False,
                )
            batch_s = torch.utils.data.BatchSampler(s, batch_size, True)
            self._samplers.append(batch_s)
        self._collater_fn = collater_fn
        self._transform = transform

    def __iter__(self):
        self._sampler_iters = []
        for s in self._samplers:
            self._sampler_iters.append(iter(s))
        return self

    def __next__(self):
        # throw "StopIteration" when any of the sampler finishes iteration
        try:
            merged_samples = tuple(
                itertools.chain.from_iterable(
                    (
                        (dts[i] for i in next(si))
                        for dts, si in zip(self._metasets, self._sampler_iters)
                    )
                )
            )
        except (
            RuntimeError
        ) as e:  # catch RuntimeError raised by drained sampler iterater
            raise StopIteration()
        output = self._collater_fn(merged_samples)
        if self._transform is not None:
            self._transform(output)
        if self._pin_memory:
            output = output.pin_memory()
        return output

    def __len__(self):
        return min([len(s) for s in self._samplers])


class H5MetasetDataLoader:
    """Load batches from one Metaset. For kwargs/options, see
    https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset


    Parameters
    -----------
    metaset:
        Dataset object for a single set of molecules
    batch_size:
        Size of the batches to draw from the metaset
    """

    def __init__(
        self,
        metaset: torch.utils.data.Dataset,
        batch_size: int,
        collater_fn: PyGCollater = PyGCollater(None, None),
        shuffle: bool = True,
        pin_memory: bool = False,
        transform=None,
    ):
        self._metaset = metaset
        if shuffle:
            sampler = torch.utils.data.RandomSampler(metaset)
        else:
            sampler = torch.utils.data.SequentialSampler(metaset)
        self._sampler = torch.utils.data.BatchSampler(sampler, batch_size, True)
        self._collater_fn = collater_fn
        self._pin_memory = pin_memory
        self._transform = transform

    def __iter__(self):
        self._sampler_iter = iter(self._sampler)
        return self

    def __next__(self):
        batch_samples = tuple(
            (self._metaset[i] for i in next(self._sampler_iter))
        )
        output = self._collater_fn(batch_samples)
        if self._transform is not None:
            self._transform(output)
        if self._pin_memory:
            output = output.pin_memory()
        return output

    def __len__(self):
        return len(self._sampler)
