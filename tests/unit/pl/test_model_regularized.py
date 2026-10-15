import pytest

import functools
import torch

from mlcg.nn import Loss, ForceMSE, GradientsOut
from mlcg.pl import RegularizedPLModel


class ToyModel(torch.nn.Module):
    name = "toy"

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)

    def forward(self, data):
        return data


def make_pl_model(model):
    # optimizer/lr_scheduler given as partials, the way LightningCLI passes them
    return RegularizedPLModel(
        model,
        Loss(losses=[ForceMSE()]),
        optimizer=functools.partial(torch.optim.AdamW, lr=1e-3),
        lr_scheduler=functools.partial(
            torch.optim.lr_scheduler.StepLR, step_size=5
        ),
    )


def test_optimizer_and_scheduler_not_in_hparams():
    pl_model = make_pl_model(ToyModel())

    assert "optimizer" not in pl_model.hparams
    assert "lr_scheduler" not in pl_model.hparams
    for key in ["model", "loss", "optimizer_groups"]:
        assert key in pl_model.hparams


def test_configure_optimizers_uses_given_optimizer_and_scheduler():
    pl_model = make_pl_model(ToyModel())
    out = pl_model.configure_optimizers()

    assert type(out["optimizer"]) is torch.optim.AdamW
    assert out["optimizer"].param_groups[0]["lr"] == 1e-3
    assert type(out["lr_scheduler"]) is torch.optim.lr_scheduler.StepLR


@pytest.mark.parametrize(
    "model, derivative",
    [
        (ToyModel(), False),
        (GradientsOut(ToyModel(), targets="forces"), True),
    ],
)
def test_lightning_module_initialized(model, derivative):
    pl_model = make_pl_model(model)

    # attributes set by pl.LightningModule.__init__
    assert pl_model.automatic_optimization is True
    assert pl_model._trainer is None
    assert pl_model.derivative is derivative
