import pytest
import pytorch_lightning as pl
import torch
from ase.build import molecule
from torch_geometric.data.collate import collate
from torch_geometric.loader import DataLoader

from mlcg.data.atomic_data import AtomicData
from mlcg.data._keys import FORCE_KEY
from mlcg.nn import (
    CompiledTraining,
    CosineCutoff,
    GaussianBasis,
    GradientsOut,
    Loss,
    ForceMSE,
    StandardPaiNN,
    StandardSchNet,
    StandardSo3krates,
    optimize_for_training,
)
from mlcg.nn.mace import StandardMACE
from mlcg.pl import PLModel, extract_model_from_checkpoint
from mlcg.nn.flash_models import optimize_for_inference
from mlcg.nn.training import _flatten, _unflatten

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOLERANCE = 1e-4
NAMES = ["CH3CH2NH2", "C6H6", "H2CO", "CH3COOH", "butadiene", "cyclobutane"]
TYPES = sorted({int(z) for n in NAMES for z in molecule(n).numbers})


def frames(names, seed=0):
    generator = torch.Generator().manual_seed(seed)
    frames = []
    for name in names:
        mol = molecule(name)
        pos = torch.tensor(mol.get_positions()).float()
        frames.append(
            AtomicData.from_points(
                pos=pos + 0.05 * torch.randn(pos.shape, generator=generator),
                atom_types=torch.tensor(mol.get_atomic_numbers()),
                forces=torch.randn(pos.shape, generator=generator),
            )
        )
    return frames


def batch(names, seed=0):
    data, _, _ = collate(
        AtomicData,
        data_list=frames(names, seed),
        increment=True,
        add_batch=True,
    )
    return data.to(DEVICE)


def schnet():
    return StandardSchNet(
        rbf_layer=GaussianBasis(cutoff=CosineCutoff(0, 5), num_rbf=8),
        cutoff=CosineCutoff(0, 5),
        output_hidden_layer_widths=[8, 4],
        hidden_channels=16,
        embedding_size=20,
        num_filters=16,
        num_interactions=2,
    )


def painn():
    return StandardPaiNN(
        rbf_layer=GaussianBasis(cutoff=CosineCutoff(0, 5), num_rbf=8),
        cutoff=CosineCutoff(0, 5),
        output_hidden_layer_widths=[8, 4],
        hidden_channels=8,
        embedding_size=20,
        num_interactions=2,
    )


def so3krates():
    return StandardSo3krates(
        rbf_layer=GaussianBasis(cutoff=CosineCutoff(0, 5), num_rbf=8),
        cutoff=CosineCutoff(0, 5),
        output_hidden_layer_widths=[8, 4],
        hidden_channels=8,
        embedding_size=20,
        degrees=[1, 2],
        n_heads=2,
        num_interactions=1,
    )


def mace():
    return StandardMACE(
        r_max=5.0,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=1,
        interaction_cls="mace.modules.blocks.RealAgnosticResidualInteractionBlock",
        interaction_cls_first="mace.modules.blocks.RealAgnosticResidualInteractionBlock",
        num_interactions=1,
        hidden_irreps="16x0e",
        MLP_irreps="8x0e",
        avg_num_neighbors=9,
        correlation=2,
        gate=torch.nn.Tanh(),
        max_num_neighbors=1000,
        pair_repulsion=False,
        distance_transform=None,
        radial_MLP=[16, 16],
        radial_type="bessel",
        atomic_numbers=TYPES,
    )


MODELS = [
    ("schnet", schnet, False),
    ("flash_schnet", schnet, True),
    ("painn", painn, False),
    ("so3krates", so3krates, False),
    ("mace", mace, False),
]


def force_loss_grads(forward, model, data):
    data = data.clone()
    data.out = {}
    forces = forward(data).out[model.name][FORCE_KEY]
    loss = (forces - data.forces).square().mean()
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return forces.detach(), grads


def assert_same(actual, expected):
    forces, grads = actual
    forces_ref, grads_ref = expected
    norm = lambda x, y: float((x - y).norm() / y.norm())
    assert norm(forces, forces_ref) < TOLERANCE
    for g, g_ref in zip(grads, grads_ref):
        if g is None and g_ref is None:
            continue
        g = torch.zeros_like(g_ref) if g is None else g
        g_ref = torch.zeros_like(g) if g_ref is None else g_ref
        if g_ref.norm() > 0:
            assert norm(g, g_ref) < TOLERANCE
        else:
            assert g.norm() == 0


@pytest.fixture
def full_float32():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize("name, build, flash", MODELS)
def test_compiled_training_matches_eager(name, build, flash, full_float32):
    if flash and DEVICE == "cpu":
        pytest.skip("flash kernels need CUDA")
    torch.manual_seed(0)
    model = GradientsOut(build(), targets=FORCE_KEY).to(DEVICE).train()
    model, forward = optimize_for_training(model, flash=flash, compile=True)
    for names in (NAMES[:4], NAMES[2:]):
        data = batch(names)
        assert_same(
            force_loss_grads(forward, model, data),
            force_loss_grads(model, model, data),
        )
    assert forward.traceable and forward.graph is not None
    assert forward.compiled_calls == 3  # the check, then both batches

    single = batch(NAMES[:1])
    assert_same(
        force_loss_grads(forward, model, single),
        force_loss_grads(model, model, single),
    )


def test_untraceable_model_trains_eagerly(full_float32):
    class HostSync(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.network = schnet()
            self.name = self.network.name

        def forward(self, data):
            data = self.network(data)
            if data.pos.sum() > 0:
                data.out[self.name]["energy"] = (
                    data.out[self.name]["energy"] * 1
                )
            return data

    model = GradientsOut(HostSync(), targets=FORCE_KEY).to(DEVICE).train()
    forward = CompiledTraining(model)
    data = batch(NAMES[:3])
    with pytest.warns(UserWarning, match="training eagerly"):
        actual = force_loss_grads(forward, model, data)
    assert not forward.traceable
    assert_same(actual, force_loss_grads(model, model, data))


def test_pl_model_optimize_keeps_checkpoint_keys():
    def build(optimize):
        torch.manual_seed(0)
        model = GradientsOut(schnet(), targets=FORCE_KEY)
        loss = Loss(losses=[ForceMSE(force_kwd=FORCE_KEY)])
        return PLModel(model, loss, optimize=optimize)

    plain = build(None)
    optimized = build({"flash": DEVICE == "cuda", "compile": True})
    assert plain.state_dict().keys() == optimized.state_dict().keys()
    assert plain.compiled is None
    assert isinstance(optimized.compiled, CompiledTraining)


def test_flatten_round_trip_keeps_none_leaves():
    data = batch(NAMES[:2])
    data.neighbor_list = {
        "prior": {
            "tag": "prior",
            "index_mapping": torch.zeros(2, 3, dtype=torch.long),
            "cell_shifts": None,
            "rcut": None,
            "extra": [None, torch.ones(3)],
        }
    }
    tensors, key = _flatten(data, {"prior"})
    fields, neighbor_lists = _unflatten(key, tensors)
    assert neighbor_lists["prior"]["cell_shifts"] is None
    assert neighbor_lists["prior"]["extra"][0] is None
    assert neighbor_lists["prior"]["extra"][1] is tensors[-1]
    assert fields["pos"] is data.pos


class AdamPLModel(PLModel):
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def fit(module, epochs, root, ckpt_path=None):
    """Two batches per epoch, so checkpoints fall on epoch boundaries."""
    loader = DataLoader(frames(NAMES[:4]), batch_size=2, shuffle=False)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=epochs,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(root),
    )
    trainer.fit(module, loader, ckpt_path=ckpt_path)
    return trainer


def pl_module(optimize):
    torch.manual_seed(0)
    model = GradientsOut(schnet(), targets=FORCE_KEY)
    loss = Loss(losses=[ForceMSE(force_kwd=FORCE_KEY)])
    return AdamPLModel(model, loss, optimize=optimize)


@pytest.mark.skipif(DEVICE == "cpu", reason="flash kernels need CUDA")
def test_checkpoint_round_trip(tmp_path, full_float32):
    """Trained with flash + compile: training restarts from the checkpoint
    with any ``optimize``, and the extracted model is the standard one,
    ready for simulation."""
    optimize = {"flash": True, "compile": True}
    straight = pl_module(optimize)
    fit(straight, 2, tmp_path / "straight")

    first = pl_module(optimize)
    trainer = fit(first, 1, tmp_path / "first")
    ckpt = tmp_path / "epoch1.ckpt"
    trainer.save_checkpoint(ckpt)
    for resume_optimize in (None, optimize):
        resumed = pl_module(resume_optimize)
        fit(resumed, 2, tmp_path / "resumed", ckpt_path=str(ckpt))
        for (name, p), q in zip(
            resumed.model.named_parameters(), straight.model.parameters()
        ):
            assert float((p - q).norm() / q.norm()) < TOLERANCE, name

    extracted = extract_model_from_checkpoint(str(ckpt), None).to(DEVICE)
    assert type(extracted.model) is StandardSchNet
    trained = PLModel.load_from_checkpoint(str(ckpt)).model.to(DEVICE)
    data = batch(NAMES[:3])
    expected = trained.eval()(data.clone()).out[trained.name][FORCE_KEY]
    actual = extracted.eval()(data.clone()).out[extracted.name][FORCE_KEY]
    assert float((actual - expected).norm() / expected.norm()) < TOLERANCE

    inference = optimize_for_inference(
        extracted, data.clone(), cuda_graph=False
    )
    tabulated = inference(data.clone()).out[extracted.name][FORCE_KEY]
    assert float((tabulated - expected).norm() / expected.norm()) < TOLERANCE


def test_unused_neighbor_lists_keep_the_graph(full_float32):
    """Prior neighbour lists differing between batches don't matter when
    the trained model has no priors."""
    model = GradientsOut(schnet(), targets=FORCE_KEY).to(DEVICE).train()
    forward = CompiledTraining(model)
    prior = {
        "tag": "bonds",
        "index_mapping": torch.zeros(2, 3, dtype=torch.long, device=DEVICE),
    }
    first, second = batch(NAMES[:3]), batch(NAMES[2:])
    first.neighbor_list = {"bonds": prior}
    second.neighbor_list = {"bonds": prior, "angles": dict(prior, tag="a")}
    for data in (first, second):
        assert_same(
            force_loss_grads(forward, model, data),
            force_loss_grads(model, model, data),
        )
    assert forward.compiled_calls == 3  # the check, then both batches
