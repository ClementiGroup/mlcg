import copy

import pytest
import torch
from torch_geometric.data.collate import collate

from mlcg.data.atomic_data import AtomicData
from mlcg.data._keys import ENERGY_KEY, FORCE_KEY
from mlcg.nn.cutoff import CosineCutoff, IdentityCutoff, ShiftedCosineCutoff
from mlcg.nn.flash_models import (
    StandardFlashSchNet,
    TabulatedSchNet,
    convert_standard_model_to_flash,
)
from mlcg.nn.gradients import GradientsOut
from mlcg.nn.kernels import RadiusCSR, radius
from mlcg.nn.radial_basis import (
    ExpNormalBasis,
    ExtendedExpNormalBasis,
    GaussianBasis,
    RIGTOBasis,
    SpacedExpBasis,
)
from mlcg.nn.schnet import StandardSchNet

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels need CUDA"
)

CUTOFF = 6.0
CUTOFFS = {
    "cosine": lambda: CosineCutoff(0.0, CUTOFF),
    "cosine_lower": lambda: CosineCutoff(1.0, CUTOFF),
    "identity": lambda: IdentityCutoff(0.0, CUTOFF),
    "shifted_cosine": lambda: ShiftedCosineCutoff(CUTOFF, 1.0),
}
BASES = {
    "exp_normal": lambda c: ExpNormalBasis(c, num_rbf=32),
    "extended_exp_normal": lambda c: ExtendedExpNormalBasis(
        c, num_rbf=20, internal_cutoff_lower=0.5
    ),
    "gaussian": lambda c: GaussianBasis(c, num_rbf=25),
    "spaced_exp": lambda c: SpacedExpBasis(c),
    "rigto": lambda c: RIGTOBasis(c, nmax=8, lmax=0),
}
ACTIVATIONS = {
    "tanh": torch.nn.Tanh,
    "silu": torch.nn.SiLU,
    "softplus": torch.nn.Softplus,
}


def random_batch(sizes=(1, 7, 30, 64), seed=0):
    """Molecules of several sizes (one isolated atom) in one batch, beads at
    least ~1 Å apart so the filters are probed over the whole range."""
    gen = torch.Generator().manual_seed(seed)
    frames = []
    for n in sizes:
        pos = torch.randn(n, 3, generator=gen) * (1.5 * n ** (1 / 3))
        types = torch.randint(1, 10, (n,), generator=gen)
        frames.append(AtomicData.from_points(pos=pos, atom_types=types))
    data, _, _ = collate(AtomicData, frames, increment=True, add_batch=True)
    return data


def build_schnet(basis, cutoff, activation, num_interactions=2):
    # ShiftedCosineCutoff breaks the standard SchNet when used inside the
    # bases that broadcast it over the RBF axis, so it is only tested on the
    # convolution there
    unsupported = cutoff == "shifted_cosine" and basis != "rigto"
    rbf_cutoff = CUTOFFS["cosine" if unsupported else cutoff]()
    return StandardSchNet(
        BASES[basis](rbf_cutoff),
        CUTOFFS[cutoff](),
        [40],
        hidden_channels=48,
        num_filters=48,
        num_interactions=num_interactions,
        activation=ACTIVATIONS[activation](),
    )


def energy_forces(model, data, dtype):
    model = GradientsOut(copy.deepcopy(model), targets=FORCE_KEY)
    model = model.to("cuda", dtype).eval()
    data = copy.deepcopy(data).to("cuda")
    data.pos = data.pos.to(dtype)
    data.out = {}
    out = model(data).out[model.name]
    return out[ENERGY_KEY].double(), out[FORCE_KEY].double()


def assert_matches_float64(standard, tabulated, data):
    torch.backends.cuda.matmul.allow_tf32 = False
    e_ref, f_ref = energy_forces(standard, data, torch.float64)
    e, f = energy_forces(tabulated, data, torch.float32)
    assert (e - e_ref).abs().max() <= 1e-4 * e_ref.abs().max() + 1e-5
    assert (f - f_ref).norm() <= 1e-4 * f_ref.norm()


@pytest.mark.parametrize("cutoff", CUTOFFS)
@pytest.mark.parametrize("basis", BASES)
def test_tabulated_bases_and_cutoffs(basis, cutoff):
    standard = build_schnet(basis, cutoff, "tanh")
    tabulated = TabulatedSchNet.flash_from_standard(standard.cuda())
    assert_matches_float64(standard, tabulated, random_batch())


@pytest.mark.parametrize("activation", ACTIVATIONS)
def test_tabulated_activations(activation):
    standard = build_schnet("exp_normal", "cosine", activation, 3)
    tabulated = TabulatedSchNet.flash_from_standard(standard.cuda())
    assert_matches_float64(standard, tabulated, random_batch(seed=1))


def test_converter_falls_back_for_non_smooth_filter():
    standard = GradientsOut(
        build_schnet("exp_normal", "cosine", "tanh"), targets=FORCE_KEY
    )
    for block in standard.model.interaction_blocks:
        block.conv.filter_network.layers[1] = torch.nn.ReLU()
    with pytest.warns(UserWarning, match="Filter table error"):
        flash = convert_standard_model_to_flash(standard.cuda())
    assert isinstance(flash.model, StandardFlashSchNet)


def test_converter_tabulates_by_default():
    standard = GradientsOut(
        build_schnet("exp_normal", "cosine", "tanh"), targets=FORCE_KEY
    ).cuda()
    assert isinstance(
        convert_standard_model_to_flash(standard).model, TabulatedSchNet
    )
    assert isinstance(
        convert_standard_model_to_flash(standard, False).model,
        StandardFlashSchNet,
    )


def test_tabulated_rejects_pbc():
    tabulated = TabulatedSchNet.flash_from_standard(
        build_schnet("exp_normal", "cosine", "tanh").cuda()
    )
    data = random_batch().to("cuda")
    data.cell = torch.eye(3, device="cuda") * 20
    with pytest.raises(NotImplementedError):
        tabulated(data)


def test_tabulated_has_no_double_backward():
    tabulated = TabulatedSchNet.flash_from_standard(
        build_schnet("exp_normal", "cosine", "tanh").cuda()
    )
    data = random_batch().to("cuda")
    data.pos.requires_grad_(True)
    energy = tabulated(data).out[tabulated.name][ENERGY_KEY].sum()
    (forces,) = torch.autograd.grad(energy, data.pos, create_graph=True)
    with pytest.raises(NotImplementedError):
        forces.sum().backward()


def csr_sets(row_ptr, col):
    row_ptr, col = row_ptr.tolist(), col.tolist()
    return [
        sorted(col[row_ptr[i] : row_ptr[i + 1]])
        for i in range(len(row_ptr) - 1)
    ]


def test_radius_csr_matches_radius():
    data = random_batch().to("cuda")
    row_ptr, col, valid = RadiusCSR(CUTOFF)(data.pos, data.batch, data.ptr)
    edge_index, _, _ = radius(data.pos, data.batch, CUTOFF)
    expected = [[] for _ in range(data.pos.shape[0])]
    for src, dst in edge_index.t().tolist():
        expected[dst].append(src)
    assert csr_sets(row_ptr, col) == [sorted(r) for r in expected]
    assert valid.item() == 1.0


def test_radius_csr_regrows_when_checked():
    data = random_batch().to("cuda")
    neighbors = RadiusCSR(CUTOFF, all_pairs_budget=0)
    neighbors.reserve(data.pos, data.batch, data.ptr)
    reference = csr_sets(*neighbors(data.pos, data.batch, data.ptr)[:2])
    neighbors.capacity = 8
    row_ptr, col, valid = neighbors(data.pos, data.batch, data.ptr)
    assert neighbors.capacity > 8
    assert csr_sets(row_ptr, col) == reference and valid.item() == 1.0


def test_tabulated_poisons_truncated_neighbor_lists():
    tabulated = TabulatedSchNet.flash_from_standard(
        build_schnet("exp_normal", "cosine", "tanh").cuda()
    )
    small_graphs = random_batch(sizes=(8, 8, 8, 8)).to("cuda")
    one_graph = random_batch(sizes=(32,)).to("cuda")
    one_graph.pos = one_graph.pos * 0.1
    tabulated(small_graphs)
    energy = tabulated(one_graph).out[tabulated.name][ENERGY_KEY]
    assert torch.isnan(energy).all()


def test_tabulated_step_has_no_host_sync():
    model = GradientsOut(
        TabulatedSchNet.flash_from_standard(
            build_schnet("exp_normal", "cosine", "tanh").cuda()
        ),
        targets=FORCE_KEY,
    ).eval()
    data = random_batch().to("cuda")
    model(data)
    torch.cuda.set_sync_debug_mode("error")
    try:
        data.out = {}
        model(data)
    finally:
        torch.cuda.set_sync_debug_mode("default")
