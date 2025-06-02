import os
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset

from ..utils import make_splits


class DataModule(pl.LightningDataModule):
    """PL interface to train with datasets defined in mlcg.datasets.

    Parameters
    ----------

    dataset:
        a dataset from mlcg.datasets (or following the API of `torch_geometric.data.InMemoryDataset`)
    log_dir:
        where to store the data that might be produced during training.
    val_ratio:
        fraction of the dataset used for validation
    test_ratio:
        fraction of the dataset used for testing
    splits:
        filename of a file containing the indices for training, validation, and testing. It should be compatible with `np.load` and contain the fields `'idx_train'`, `'idx_val'`, and `'idx_test'`.
        If None then the dataset is split randomly using the `val_ratio` and `test_ratio`.
    batch_size:
        number of structure to include in each training batches.
    inference_batch_size:
        number of structure to include in each validation/training batches.
    num_workers;
        number of `cpu` used for loading the dataset (see `here <https://pytorch-lightning.readthedocs.io/en/stable/guides/speed.html?highlight=num_workers#num-workers>`_ for more details).
    loading_stride:
        stride used to subselect the dataset. Useful parameter for debugging purposes.
    save_local_copy:
        saves the input dataset in log_dir

    """

    def __init__(
        self,
        dataset: InMemoryDataset,
        log_dir: str,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        splits: str = None,
        batch_size: int = 512,
        inference_batch_size: int = 64,
        num_workers: int = 1,
        loading_stride: int = 1,
        save_local_copy: bool = False,
        pin_memory: bool = True,
    ) -> None:
        super(DataModule, self).__init__()
        # self.save_hyperparameters()

        # assume dataset is similart to torch_geometric.dataset.Dataset
        self.dataset_init_kwargs = {
            "root": dataset.root,
            "transform": dataset.transform,
            "pre_transform": dataset.pre_transform,
            "pre_filter": dataset.pre_filter,
        }
        self.dataset_cls = type(dataset)
        self.save_local_copy = save_local_copy

        if self.save_local_copy:
            self.dataset_root = os.path.join(log_dir, "dataset.pt")
            torch.save(dataset, self.dataset_root)

        self.batch_size = batch_size
        self.inference_batch_size = inference_batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.log_dir = log_dir
        self.splits = splits
        self.loading_stride = loading_stride
        self.splits_fn = os.path.join(self.log_dir, "splits.npz")
        self.pin_memory = pin_memory

    def load_dataset(self):
        if self.save_local_copy:
            dataset = torch.load(self.dataset_root)
        else:
            dataset = self.dataset_cls(**self.dataset_init_kwargs)
        indeces = torch.arange(0, len(dataset), self.loading_stride)
        return dataset[indeces]

    def prepare_data(self):
        """Download, preprocess dataset, etc."""
        pass

    def setup(self, stage=None):
        dataset = self.load_dataset()

        self.idx_train, self.idx_val, self.idx_test = make_splits(
            len(dataset),
            self.val_ratio,
            self.test_ratio,
            filename=self.splits_fn,
            splits=self.splits,
        )

        self.train_dataset = dataset[self.idx_train]
        self.val_dataset = dataset[self.idx_val]
        self.test_dataset = dataset[self.idx_test]

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset, "val")

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, "test")

    def _get_dataloader(self, dataset, stage):
        if stage == "train":
            batch_size = self.batch_size
            shuffle = True
        elif stage in ["val", "test"]:
            batch_size = self.inference_batch_size
            shuffle = False

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
