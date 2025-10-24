import torch
from typing import List
from pytorch_lightning.cli import OptimizerCallable, LRSchedulerCallable

from mlcg.pl import PLModel
from mlcg.nn import Loss, CustomStepLR

from torch.optim.lr_scheduler import LRScheduler

from importlib import import_module

class DummyScheduler(LRScheduler):
    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

class MultiLR(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, lambda_factories, last_epoch=-1, verbose=False):
        self.schedulers = []
        values = self._get_optimizer_lr(optimizer)
        for idx, factory in enumerate(lambda_factories):
            if factory is not None:
                self.schedulers.append(factory(optimizer))
            else:
                self.schedulers.append(DummyScheduler(optimizer))
            values[idx] = self._get_optimizer_lr(optimizer)[idx]
            self._set_optimizer_lr(optimizer, values)
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        result = []
        for idx, sched in enumerate(self.schedulers):
            result.append(sched.get_last_lr()[idx])
        return result

    @staticmethod
    def _set_optimizer_lr(optimizer, values):
        for param_group, lr in zip(optimizer.param_groups, values):
            param_group['lr'] = lr

    @staticmethod
    def _get_optimizer_lr(optimizer):
        return [group['lr'] for group in optimizer.param_groups]

    def step(self, epoch=None):
        if self.last_epoch != -1:
            values = self._get_optimizer_lr(self.optimizer)
            for idx, sched in enumerate(self.schedulers):
                sched.step()
                values[idx] = self._get_optimizer_lr(self.optimizer)[idx]
                self._set_optimizer_lr(self.optimizer, values)
        super().step()

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
        lr_scheduler: LRSchedulerCallable = None,
        optimizer_groups: List[dict] = [],
    ) -> None:
        """ """

        super().__init__(model, loss)

        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        # Must have group_keys List and lr float and other parameters compatible with optimizer
        self.optimizer_groups = self.check_optimizer_groups(optimizer_groups)

    # def configure_optimizers(self):
    #     # Extract parameters for different parameter groups
    #     all_extracted_group_names = []
    #     optimizer_setup = []
    #
    #     for group in self.optimizer_groups:
    #         group_keys = group["group_keys"]
    #         extracted_params = []
    #         for key in group_keys:
    #             for name, param in self.named_parameters():
    #                 if key in name:
    #                     extracted_params.append(param)
    #                     all_extracted_group_names.append(name)
    #         group.pop("group_keys")
    #         optimizer_setup.append({"params": extracted_params, **group})
    #
    #
    #     # Extract main parameter group
    #     extracted_params = [
    #         param
    #         for name, param in self.named_parameters()
    #         if name not in all_extracted_group_names
    #     ]
    #     optimizer_setup.insert(0, {"params": extracted_params})
    #
    #     optimizer = self.optimizer(optimizer_setup)
    #
    #     scheduler = self.scheduler(optimizer) # Jacopo's version
    #
    #
    #     return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def _import_class(self, class_path: str):
        """Dynamically import a class from its string path."""
        module_path, class_name = class_path.rsplit('.', 1)
        module = import_module(module_path)
        return getattr(module, class_name)

    def configure_optimizers(self):
        # Extract parameters for different parameter groups
        all_extracted_group_names = []
        optimizer_setup = []
        scheduler_factories = []

        # Process each optimizer group
        for group in self.optimizer_groups:
            group_keys = group["group_keys"]
            extracted_params = []
            for key in group_keys:
                for name, param in self.named_parameters():
                    if key in name:
                        extracted_params.append(param)
                        all_extracted_group_names.append(name)

            # Get scheduler config if exists
            scheduler_config = group.pop("lr_scheduler", None)
            if scheduler_config:
                scheduler_class = self._import_class(scheduler_config["class_path"])
                scheduler_args = scheduler_config.get("init_args", {})
                scheduler_factories.append(
                    lambda opt, sc=scheduler_class, sa=scheduler_args: sc(opt, **sa)
                )
            else:
                # Default to no scheduler for this group
                scheduler_factories.append(
                    None
                )

            group.pop("group_keys", None)
            optimizer_setup.append({"params": extracted_params, **group})

        # Add main scheduler factory if exists
        if self.scheduler:
            scheduler_factories.insert(
                0,
                lambda opt: self.scheduler(opt)
            )
        else:
            scheduler_factories.insert(
                0,
                None
            )

        # Extract main parameter group
        extracted_params = [
            param
            for name, param in self.named_parameters()
            if name not in all_extracted_group_names
        ]
        optimizer_setup.insert(0, {"params": extracted_params})

        # Create optimizer
        optimizer = self.optimizer(optimizer_setup)

        # Create MultiLR scheduler
        scheduler = MultiLR(optimizer, scheduler_factories)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @staticmethod
    def check_optimizer_groups(optimizer_groups):
        for group in optimizer_groups:
            if "group_keys" not in list(group.keys()):
                raise KeyError(
                    f"key 'group_keys' was not provided in group {group}"
                )

        return optimizer_groups
