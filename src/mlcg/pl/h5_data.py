import copy
from typing import List, Union
from collections.abc import Mapping

import torch.distributed as dist
import pytorch_lightning as pl
from ruamel.yaml import YAML

from ..datasets import H5Dataset, H5PartitionDataLoader, H5MetasetDataLoader

default_key_mapping = {
    "embeds": "attrs:cg_embeds",
    "coords": "cg_coords",
    "forces": "cg_delta_forces",
}


class H5DataModule(pl.LightningDataModule):
    r"""DataModule for datasets stored in HDF5 format

    Parameters
    ----------
    h5_file_path:
        Path to the hdf5 file containing the dataset
    partition_options:
        Mapping that defines which molecules are in the train and
        validation sets. See `mlcg.datasets.h5_dataset.py`.
    loading_options:
        kwarg dictionary. Specifies the dataset organization of the hdf5
        file. Eg, for training on delta forces, one would specify:

        .. code-block::

            loading_options = {"hdf_key_mapping": {
                "embeds": "attrs:cg_embeds",
                "coords": "cg_coords"
                "forces": "cg_delta_forces"
            }

        where the keys are the names values are the attrs/datasets of an hdf5 group.
        See `mlcg.datasets.h5_dataset.py`.
    """

    def __init__(
        self,
        h5_file_path: str = "",
        partition_options: Union[Mapping, str] = {},
        loading_options: Union[Mapping, str] = {
            "hdf_key_mapping": default_key_mapping
        },
        subsample_using_weights: bool = False,
    ):
        super(H5DataModule, self).__init__()
        self.save_hyperparameters()
        self._h5_file_path = h5_file_path
        self._subsample_using_weights = subsample_using_weights

        def get_options(options_or_path):
            if isinstance(options_or_path, Mapping):
                return options_or_path
            elif isinstance(options_or_path, str):
                yaml = YAML()  # automatically supports json :)
                with open(options_or_path, "r") as f:
                    options = yaml.load(f)
                return options

        self._part_options = get_options(partition_options)
        self._load_options = get_options(loading_options)

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        # when DDP, get the rank and world_size for loading the correct subset
        # if not dist.is_available():
        #    raise RuntimeError("Requires distributed package to be available")
        if dist.is_available() and dist.is_initialized():
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
            use_ddp = True
        else:
            # for the DataModule, running without DDP is the same as with DDP and world_size = 1
            num_replicas = 1
            rank = 0
            use_ddp = False
        # write possible parallelization settings to loading_options
        self._process_load_options = copy.deepcopy(self._load_options)
        self._process_load_options["parallel"] = {
            "rank": rank,
            "world_size": num_replicas,
        }
        # load the hdf5 file
        print(
            f"Loading samples for rank ({rank}/{num_replicas})...", flush=True
        )
        self._h5d = H5Dataset(
            self._h5_file_path,
            self._part_options,
            self._process_load_options,
            self._subsample_using_weights,
        )
        if use_ddp:
            sample_info = [None] * num_replicas
            dist.all_gather_object(sample_info, self._h5d.partition_sample_info)
        else:
            sample_info = [self._h5d.partition_sample_info]
        if rank == 0:  # only the main process prints
            info = "\n" + "-" * 79 + "\n"
            info += "Summary of subsampling for batch compsition balancing:\n"
            is_any_trimmed = False
            # merge the subsampling info
            for part_name in sample_info[0]:
                is_trimmed = any(
                    [
                        process_info[part_name]["is_trimmed"]
                        for process_info in sample_info
                    ]
                )
                is_any_trimmed = is_any_trimmed or is_trimmed
                if is_trimmed:
                    info += f"Partition `{part_name}`:\n"
                    for metaset_name in sample_info[0][part_name]["total_size"]:
                        metaset_total_size = 0
                        metaset_current_size = 0
                        for process_info in sample_info:
                            part_info = process_info[part_name]
                            metaset_total_size += part_info["total_size"][
                                metaset_name
                            ]
                            metaset_current_size += part_info["current_size"][
                                metaset_name
                            ]
                        info += f"\tMetaset `{metaset_name}`: {metaset_current_size} / {metaset_total_size} used\n"
            info += "Please check whether the batch size compsition is close to the original sample ratio.\n"
            info += "-" * 79 + "\n"
            if is_any_trimmed:
                print(info)

    def train_dataloader(self):
        # train_split = Dataset(...)
        return self.part_dataloader("train")

    def val_dataloader(self):
        # val_split = Dataset(...)
        return self.part_dataloader("val")

    def test_dataloader(self):
        # test_split = Dataset(...)
        test_part_name = "test"
        if self._h5d.partition(test_part_name) is None:
            # simply use the validation set
            test_part_name = "val"
        return self.part_dataloader(test_part_name)

    def part_dataloader(self, part_name):
        part = self._h5d.partition(part_name)
        if part_name == "train":
            combined_loader = H5PartitionDataLoader(
                part, subsample_using_weights=self._subsample_using_weights
            )
        else:
            loaders = []
            for metaset_name, batch_size in sorted(part.batch_sizes.items()):
                metaset = part.get_metaset(metaset_name)
                loaders.append(
                    H5MetasetDataLoader(metaset, batch_size, shuffle=False)
                )
            if len(loaders) == 1:
                combined_loader = loaders[0]
            else:
                combined_loader = tuple(loaders)
        return combined_loader

    def teardown(self, stage):
        # clean up after fit or test
        # called on every process in DDP
        # del self._h5d # not necessary and lead to exceptions
        pass
