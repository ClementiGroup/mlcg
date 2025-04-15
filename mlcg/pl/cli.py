import os.path as osp
import torch
import pytorch_lightning.cli as plc
from torch_geometric.data.makedirs import makedirs
import torch_optimizer as optim
import warnings


class LightningCLI(plc.LightningCLI):
    """Command line interface for training a model with pytorch lightning.

    It adds a few functionalities to `pytorch_lightning.utilities.cli.LightningCLI`.

    + register torch optimizers and lr_scheduler so that they can be specified
    in the configuration file. Note that only single (optimizer,lr_scheduler)
    can be specified like that and more complex patterns should be implemented
    in the pytorch_lightning model definition (child of `pytorch_lightning.
    LightningModule`). see `doc <https://pytorch-lightning.readthedocs.io/en/1.
    4.9/common/lightning_cli.html#optimizers-and-learning-rate-schedulers>`_
    for more details.

    + link manually some arguments related to the definition of the work directory. If `default_root_dir` argument of `pytorch_lightning.Trainer` is set and the `save_dir` / `log_dir` / `dirpath` argument of `loggers` / `data` / `callbacks` is set to `default_root_dir` then they will be set to the value of `default_root_dir` / `default_root_dir/data` / `default_root_dir/ckpt`.

    """

    def parse_arguments(
        self, parser: plc.LightningArgumentParser, args: plc.ArgsType
    ) -> None:
        """Parses command line arguments and stores it in self.config"""
        super().parse_arguments(parser, args)
        if "subcommand" in self.config:
            config = self.config[self.config["subcommand"]]
        else:
            config = self.config

        if config["seed_everything"] in (None, False):
            devices = int(config["trainer"]["devices"])
            num_nodes = int(config["trainer"]["num_nodes"])
            if (
                config["trainer"]["gpus"] is not None
            ):  # account for composite/redundant lightning trainer opts
                gpus = config["trainer"]["gpus"]
                if isinstance(gpus, list):
                    gpus = len(gpus)
            if (num_nodes > 1) or (devices > 1) or (gpus > 1):
                warnings.warn(
                    " \n \n ###################################################### \n"
                    "           WARNING: Possible data leakage  \n "
                    "######################################################  \n \n "
                    "`seed_everything` has been set to false/null, but multiple "
                    "devices are being used for distributed training. This can result in "
                    "local ranks/procids making different train/val/test data splits, which "
                    "can further result in train/val/test data leakage. Please follow the "
                    "best practice as outlined by Lightning developers and set "
                    "`seed_everything` to a non-zero integer for multi-device "
                    "computations. Else, make sure that your `PLDataModule` "
                    "avoids this be defining the splits to loaded from "
                    "the disk beforehand. "
                    "\n \n ###################################################### \n \n "
                )

        trainer = config["trainer"]
        default_root_dir = trainer.get("default_root_dir")
        if default_root_dir is not None:
            for ii, logger in enumerate(trainer.get("logger", [])):
                save_dir = logger["init_args"].get("save_dir")
                if save_dir == "default_root_dir":
                    config["trainer"]["logger"][ii]["init_args"][
                        "save_dir"
                    ] = default_root_dir

            log_dir = config["data"].get("log_dir")
            if log_dir == "default_root_dir":
                path = osp.join(default_root_dir, "data")
                makedirs(path)
                config["data"]["log_dir"] = path

            for ii, callback in enumerate(trainer.get("callbacks", [])):
                dirpath = callback["init_args"].get("dirpath")
                if dirpath == "default_root_dir":
                    path = osp.join(default_root_dir, "ckpt")
                    makedirs(path)
                    config["trainer"]["callbacks"][ii]["init_args"][
                        "dirpath"
                    ] = path
