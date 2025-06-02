import torch
from pytorch_lightning.plugins.environments import (
    ClusterEnvironment,
)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from typing import List, Optional, Union, Any, Dict

from .model import PLModel
from ..nn import SumOut, refresh_module_with_schnet_, fixed_pyg_inspector


def extract_model_from_checkpoint(checkpoint_path, hparams_file):
    with fixed_pyg_inspector():
        plmodel = PLModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path, hparams_file=hparams_file
        )
        model = plmodel.get_model()
        refresh_module_with_schnet_(model)
    return model


def merge_priors_and_checkpoint(
    checkpoint: Union[str, torch.nn.Module],
    priors: Union[str, torch.nn.ModuleDict, SumOut],
    hparams_file: Optional[str] = None,
    use_only_priors: bool = False,
) -> torch.nn.Module:
    """load prior models and trained model from a checkpoint and merge them
    into a :ref:`mlcg.nn.SumOut` module.

    Parameters
    ----------
    checkpoint :
        full path to the checkpoint file OR loaded `torch.nn.Module`
    priors :
        If :obj:`torch.nn.ModuleDict` or `SumOut`, it should be the collection of priors
        used as a baseline for training the ML model. If :obj:`str`, it should
        be a full path to the file holding the priors.
    hparams_file :
        full path to the hyper parameter file associated with the checkpoint file.
        It is typically not necessary to provide it.
    use_only_priors : (bool) (default: False)
        If True, a model consisting only of priors will be returned.

    Returns
    -------
        model :ref:`mlcg.nn.SumOut` module containing :
            - trained model with the priors (if use_only_priors is False)
            - model with only priors (if use_only_priors is True)
    """
    # Check priors being correct type
    assert isinstance(
        priors, (str, SumOut, torch.nn.ModuleDict)
    ), '"priors" has to be either string, SumOut, or ModuleDict'

    # merged_model should be a ModuleDict
    merged_model = torch.nn.ModuleDict()

    # if use_only_priors is not True, then load model from checkpoint file or use loaded model
    if use_only_priors == False:
        # if checkpoint is a path specifying checkpoint file, load model; else make use as model
        if isinstance(checkpoint, str):
            ml_model = extract_model_from_checkpoint(checkpoint, hparams_file)
        else:
            ml_model = checkpoint

        merged_model[ml_model.name] = ml_model

    if isinstance(priors, str):
        prior_model = torch.load(priors)
    else:
        prior_model = priors
    # case where the prior that we are loading is already wrapped in a SumOut layer
    if isinstance(prior_model, SumOut):
        prior_model = prior_model.models

    for key in prior_model.keys():
        merged_model[key] = prior_model[key]

    model = SumOut(models=merged_model)
    return model


class LossScheduler(pl.Callback):
    def __init__(self, epoch_to_update, new_weights):
        """
        epoch_to_update: int, the epoch number to update the weights
        new_weights: list or tensor, the new weights to be used after the epoch
        """
        self.epoch_to_update = epoch_to_update
        self.new_weights = torch.tensor(new_weights)

    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if trainer.current_epoch == self.epoch_to_update:
            pl_module.loss.weights.copy_(self.new_weights)
            print(
                f"Updated loss weights to {self.new_weights.tolist()} at epoch {self.epoch_to_update}"
            )


class OffsetCheckpoint(ModelCheckpoint):
    """Customized checkpoint class used to save checkpoints
    starting from specified epoch."""

    def __init__(self, start_epoch: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch

    def on_train_epoch_end(self, trainer, pl_module):
        # Only save checkpoints if the current epoch >= start_epoch
        if trainer.current_epoch >= self.start_epoch:
            # Call the parent class's save logic
            super().on_train_epoch_end(trainer, pl_module)
