import pytest
import torch
from ase.build import molecule
from torch_geometric.data.collate import collate

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
from mlcg.pl import PLModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOLERANCE = 1e-4
NAMES = ["CH3CH2NH2", "C6H6", "H2CO", "CH3COOH", "butadiene", "cyclobutane"]
TYPES = sorted({int(z) for n in NAMES for z in molecule(n).numbers})


def batch(names, seed=0):
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
    data, _, _ = collate(
        AtomicData, data_list=frames, increment=True, add_batch=True
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
