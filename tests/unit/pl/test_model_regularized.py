import pytest

import functools
from copy import deepcopy
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from mlcg.data import AtomicData
from mlcg.data._keys import ENERGY_KEY, FORCE_KEY
from mlcg.nn import Loss, ForceMSE, GradientsOut
from mlcg.pl import RegularizedPLModel, LightningCLI


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


class ToyEnergy(torch.nn.Module):
    name = "toy"

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)
        self.regularization = torch.nn.Linear(3, 1)

    def forward(self, data):
        atom_energy = self.linear(data.pos) + self.regularization(data.pos**2)
        energy = torch.zeros(data.n_atoms.shape[0]).index_add(
            0, data.batch, atom_energy.squeeze(-1)
        )
        data.out[self.name] = {ENERGY_KEY: energy}
        return data


OPTIMIZER_GROUPS = [{"group_keys": ["regularization"], "lr": 1e-2}]


def make_loader():
    generator = torch.Generator().manual_seed(0)
    data_list = [
        AtomicData.from_points(
            pos=torch.randn(4, 3, generator=generator),
            atom_types=torch.arange(1, 5),
            forces=torch.randn(4, 3, generator=generator),
        )
        for _ in range(4)
    ]
    return DataLoader(data_list, batch_size=2)


def make_trainer(max_steps):
    return pl.Trainer(
        max_steps=max_steps,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        limit_val_batches=0,
    )


def build_direct(optimizer_groups):
    return RegularizedPLModel(
        GradientsOut(ToyEnergy(), targets=FORCE_KEY),
        Loss(losses=[ForceMSE()]),
        optimizer=functools.partial(torch.optim.AdamW, lr=1e-3),
        lr_scheduler=functools.partial(
            torch.optim.lr_scheduler.StepLR, step_size=1, gamma=0.5
        ),
        optimizer_groups=optimizer_groups,
    )


def build_cli(optimizer_groups):
    args = {
        "seed_everything": 0,
        "trainer": {"accelerator": "cpu", "devices": 1, "logger": False},
        "model": {
            "model": {
                "class_path": "mlcg.nn.GradientsOut",
                "init_args": {
                    "model": {"class_path": f"{__name__}.ToyEnergy"},
                    "targets": FORCE_KEY,
                },
            },
            "loss": {
                "class_path": "mlcg.nn.Loss",
                "init_args": {"losses": [{"class_path": "mlcg.nn.ForceMSE"}]},
            },
            "optimizer": {
                "class_path": "torch.optim.AdamW",
                "init_args": {"lr": 1e-3},
            },
            "lr_scheduler": {
                "class_path": "torch.optim.lr_scheduler.StepLR",
                "init_args": {"step_size": 1, "gamma": 0.5},
            },
            "optimizer_groups": optimizer_groups,
        },
    }
    cli = LightningCLI(
        model_class=RegularizedPLModel,
        run=False,
        args=args,
        auto_configure_optimizers=False,
        save_config_callback=None,
    )
    return cli.model


def assert_same_model(pl_model, loaded, optimizer_groups):
    assert type(loaded) is RegularizedPLModel
    assert loaded.derivative is True
    assert "optimizer" not in loaded.hparams
    assert "lr_scheduler" not in loaded.hparams
    assert loaded.optimizer_groups == optimizer_groups

    state_dict, loaded_state_dict = pl_model.state_dict(), loaded.state_dict()
    assert state_dict.keys() == loaded_state_dict.keys()
    for key in state_dict:
        torch.testing.assert_close(loaded_state_dict[key], state_dict[key])

    data = next(iter(make_loader()))
    pl_model.eval()
    loaded.eval()
    torch.testing.assert_close(
        loaded.model(data.clone()).out["toy"][FORCE_KEY],
        pl_model.model(data.clone()).out["toy"][FORCE_KEY],
    )


@pytest.mark.parametrize("build", [build_direct, build_cli])
@pytest.mark.parametrize("optimizer_groups", [[], OPTIMIZER_GROUPS])
def test_checkpoint_roundtrip(tmp_path, build, optimizer_groups):
    pl_model = build(deepcopy(optimizer_groups))
    trainer = make_trainer(max_steps=2)
    trainer.fit(pl_model, make_loader())
    ckpt_path = tmp_path / "model.ckpt"
    trainer.save_checkpoint(ckpt_path)

    loaded = RegularizedPLModel.load_from_checkpoint(
        ckpt_path, weights_only=False
    )
    assert_same_model(pl_model, loaded, optimizer_groups)

    # optimizer/lr_scheduler are not in the checkpoint: resuming rebuilds the
    # model from its config, as `fit --ckpt_path` does
    resumed = build(deepcopy(optimizer_groups))
    resume_trainer = make_trainer(max_steps=4)
    resume_trainer.fit(resumed, make_loader(), ckpt_path=ckpt_path)
    assert resume_trainer.global_step == 4
    assert len(resume_trainer.optimizers[0].param_groups) == 1 + len(
        optimizer_groups
    )


@pytest.mark.parametrize("build", [build_direct, build_cli])
@pytest.mark.parametrize("optimizer_groups", [[], OPTIMIZER_GROUPS])
def test_torch_save_roundtrip(tmp_path, build, optimizer_groups):
    pl_model = build(deepcopy(optimizer_groups))
    path = tmp_path / "model.pt"
    torch.save(pl_model, path)

    loaded = torch.load(path, weights_only=False)
    assert_same_model(pl_model, loaded, optimizer_groups)
    loaded.configure_optimizers()


class RecordOptimizationStateOnTrainStart(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        self.lrs = [g["lr"] for g in trainer.optimizers[0].param_groups]
        self.scheduler_state = trainer.lr_scheduler_configs[
            0
        ].scheduler.state_dict()


@pytest.mark.parametrize("build", [build_direct, build_cli])
@pytest.mark.parametrize("optimizer_groups", [[], OPTIMIZER_GROUPS])
def test_resume_from_checkpoint_restores_lr_and_scheduler(
    tmp_path, build, optimizer_groups
):
    interrupted = build(deepcopy(optimizer_groups))
    interrupted_trainer = make_trainer(max_steps=2)
    interrupted_trainer.fit(interrupted, make_loader())
    ckpt_path = tmp_path / "model.ckpt"
    interrupted_trainer.save_checkpoint(ckpt_path)
    ckpt = torch.load(ckpt_path, weights_only=False)

    resumed = build(deepcopy(optimizer_groups))
    recorder = RecordOptimizationStateOnTrainStart()
    resume_trainer = make_trainer(max_steps=6)
    resume_trainer.callbacks.append(recorder)
    resume_trainer.fit(resumed, make_loader(), ckpt_path=ckpt_path)

    initial_lrs = [1e-3] + [g["lr"] for g in optimizer_groups]
    ckpt_lrs = [g["lr"] for g in ckpt["optimizer_states"][0]["param_groups"]]
    assert ckpt_lrs != initial_lrs
    assert recorder.lrs == ckpt_lrs
    assert recorder.scheduler_state == ckpt["lr_schedulers"][0]
    assert type(resume_trainer.lr_scheduler_configs[0].scheduler) is (
        torch.optim.lr_scheduler.StepLR
    )
