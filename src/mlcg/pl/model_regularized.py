import torch
from typing import List
from pytorch_lightning.cli import OptimizerCallable, LRSchedulerCallable

from mlcg.pl import PLModel
from mlcg.nn import Loss, CustomStepLR


class RegularizedPLModel(PLModel):
    """PL interface to train with models defined in :ref:`mlcg.nn`.
    This interface optionally allows to select different parameter groups
    inside the optimizer.

    Parameters
    ----------

        model:
            instance of a model class from :ref:`mlcg.nn`.
        loss:
            instance of :ref:`mlcg.nn.Loss`.
        optimizer:
            instance of a torch optimizer from :ref:`torch.optim`.
        lr_scheduler:
            instance of learning rate scheduler compatible with optimizer.
        optimizer_groups:
            optional, list of dictionaries containing specification for
            parmeters group different from the main one. Must contain
            a list with partial unique names for specific parameter group under
            key "group_keys" and optionally corresponding optimizer setup
            for the selected groups, for example:

            # Main parameter group setup
            optimizer:
                class_path: torch.optim.AdamW
                init_args:
                lr: 0.0001
                weight_decay: 0.01

            optimizer_groups:
                # Second parameter group setup:
                # all parameters with name containing "regularization.param1"
                # will be selected and will have lr of 0.01
                - group_keys:
                    - regularization.param1
                lr: 0.01
                # Third parameter group setup:
                # all parameters with name containing "regularization.param2"
                # or "regularization.param3" will be selected and will have lr of 0.1
                - group_keys:
                    - regularization.param2
                    - regularization.param3
                lr: 0.1
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss: Loss,
        optimizer: OptimizerCallable = torch.optim.AdamW,
        lr_scheduler: LRSchedulerCallable = CustomStepLR,
        optimizer_groups: List[dict] = [],
    ) -> None:
        """ """

        super().__init__(model, loss)

        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        # Must have group_keys List and lr float and other parameters compatible with optimizer
        self.optimizer_groups = self.check_optimizer_groups(optimizer_groups)

    def configure_optimizers(self):
        # Extract parameters for different parameter groups
        all_extracted_group_names = []
        optimizer_setup = []
        for group in self.optimizer_groups:
            group_keys = group["group_keys"]
            extracted_params = []
            for key in group_keys:
                for name, param in self.named_parameters():
                    if key in name:
                        extracted_params.append(param)
                        all_extracted_group_names.append(name)
            group.pop("group_keys")
            optimizer_setup.append({"params": extracted_params, **group})

        # Extract main parameter group
        extracted_params = [
            param
            for name, param in self.named_parameters()
            if name not in all_extracted_group_names
        ]
        optimizer_setup.insert(0, {"params": extracted_params})

        optimizer = self.optimizer(optimizer_setup)

        scheduler = self.scheduler(optimizer)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @staticmethod
    def check_optimizer_groups(optimizer_groups):
        for group in optimizer_groups:
            if "group_keys" not in list(group.keys()):
                raise KeyError(
                    f"key 'group_keys' was not provided in group {group}"
                )

        return optimizer_groups
