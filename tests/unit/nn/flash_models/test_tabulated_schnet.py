import copy

import pytest
import torch
from torch_geometric.data.collate import collate

from mlcg.data.atomic_data import AtomicData
from mlcg.data._keys import ENERGY_KEY, FORCE_KEY
from mlcg.nn.cutoff import CosineCutoff, IdentityCutoff, ShiftedCosineCutoff
from mlcg.nn.flash_models import (
    StandardFlashSchNet,
    convert_standard_model_to_flash,
    prepare_for_inference,
)
from mlcg.nn.gradients import GradientsOut
from mlcg.nn.kernels import RadiusCSR, radius
from mlcg.nn.kernels.pbc import periodic_cells
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
NO_CELL = periodic_cells(None, None, 0, "cuda")
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


def tabulate(standard, tolerance=1e-4):
    """The inference model: flash conversion, filter tables, eval mode. The
    accuracy tests were frozen with the 1e-4 table tolerance of 02."""
    flash = StandardFlashSchNet.flash_from_standard(standard)
    return flash.build_tables(tolerance=tolerance).eval()


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
    assert_matches_float64(standard, tabulate(standard.cuda()), random_batch())


@pytest.mark.parametrize("activation", ACTIVATIONS)
def test_tabulated_activations(activation):
    standard = build_schnet("exp_normal", "cosine", activation, 3)
    assert_matches_float64(
        standard, tabulate(standard.cuda()), random_batch(seed=1)
    )


def non_smooth_flash_schnet():
    flash = StandardFlashSchNet.flash_from_standard(
        build_schnet("exp_normal", "cosine", "tanh").cuda()
    )
    for block in flash.interaction_blocks:
        block.conv.filter_network.layers[1] = torch.nn.ReLU()
    return flash


def test_build_tables_rejects_non_smooth_filter():
    with pytest.raises(ValueError, match="Filter table error"):
        non_smooth_flash_schnet().build_tables()


def test_prepare_for_inference_keeps_filters_when_not_tabulable():
    model = GradientsOut(non_smooth_flash_schnet(), targets=FORCE_KEY)
    with pytest.warns(UserWarning, match="Filter table error"):
        model = prepare_for_inference(model, random_batch().to("cuda"))
    assert not model.model.tabulated and not model.training


def test_conversion_keeps_the_filter_activation():
    standard = build_schnet("exp_normal", "cosine", "tanh").cuda()
    for block in standard.interaction_blocks:
        block.conv.filter_network.layers[1] = torch.nn.SiLU()
    flash = StandardFlashSchNet.flash_from_standard(standard)
    for block in flash.interaction_blocks:
        assert isinstance(block.conv.filter_network.layers[1], torch.nn.SiLU)
        assert isinstance(block.activation, torch.nn.Tanh)
    data = random_batch()
    assert_matches_float64(standard, flash, data)
    assert_matches_float64(standard, flash.build_tables().eval(), data)


def test_converter_is_trainable_and_idempotent():
    standard = GradientsOut(
        build_schnet("exp_normal", "cosine", "tanh"), targets=FORCE_KEY
    ).cuda()
    flash = convert_standard_model_to_flash(standard)
    assert isinstance(flash.model, StandardFlashSchNet)
    assert not flash.model.tabulated
    assert convert_standard_model_to_flash(flash).model is flash.model


def test_eval_without_tables_is_the_filter_path():
    model = GradientsOut(
        StandardFlashSchNet.flash_from_standard(
            build_schnet("exp_normal", "cosine", "tanh").cuda()
        ),
        targets=FORCE_KEY,
    )
    data = random_batch().to("cuda")
    outputs = []
    for mode in (True, False):
        model.train(mode)
        batch = copy.deepcopy(data)
        batch.out = {}
        out = model(batch).out[model.name]
        outputs.append((out[ENERGY_KEY].detach(), out[FORCE_KEY].detach()))
    assert not model.model.tabulated
    for train, evaluated in zip(*outputs):
        torch.testing.assert_close(train, evaluated)


def test_train_and_load_state_dict_drop_tables():
    flash = tabulate(build_schnet("exp_normal", "cosine", "tanh").cuda())
    assert all(b.conv.table is not None for b in flash.interaction_blocks)
    flash.train()
    assert not flash.tabulated
    assert all(b.conv.table is None for b in flash.interaction_blocks)
    flash.build_tables().eval()
    flash.load_state_dict(flash.state_dict())
    assert not flash.tabulated


def test_state_dict_is_standard():
    standard = build_schnet("exp_normal", "cosine", "tanh").cuda()
    flash = tabulate(standard)
    assert flash.state_dict().keys() == standard.state_dict().keys()
    standard.load_state_dict(flash.state_dict())


CELLS = {
    "cubic": [[14.0, 0, 0], [0, 14.0, 0], [0, 0, 14.0]],
    "orthorhombic": [[13.0, 0, 0], [0, 15.0, 0], [0, 0, 19.0]],
    "triclinic": [[14.0, 0, 0], [5.0, 15.0, 0], [-4.0, 6.0, 16.0]],
}


def periodic_batch(cells, pbc=(True, True, True), n=48, seed=0):
    """One graph per cell, beads uniform in the cell (so many pairs cross
    the faces), lattice vectors as rows."""
    gen = torch.Generator().manual_seed(seed)
    frames = []
    for cell in cells:
        cell = torch.tensor(cell)
        pos = torch.rand(n, 3, generator=gen) @ cell
        frames.append(
            AtomicData.from_points(
                pos=pos,
                atom_types=torch.randint(1, 10, (n,), generator=gen),
                cell=cell,
                pbc=torch.tensor(pbc),
            )
        )
    data, _, _ = collate(AtomicData, frames, increment=True, add_batch=True)
    return data


@pytest.mark.parametrize(
    "cells, pbc",
    [
        (["cubic", "orthorhombic"], (True, True, True)),
        (["triclinic", "cubic"], (True, True, True)),
        (["orthorhombic", "triclinic"], (True, True, False)),
    ],
)
def test_tabulated_pbc_matches_float64(cells, pbc):
    standard = build_schnet("exp_normal", "cosine", "tanh")
    data = periodic_batch([CELLS[c] for c in cells], pbc)
    assert_matches_float64(standard, tabulate(standard.cuda()), data)


def test_radius_csr_pbc_matches_exact_enumeration():
    """Same pairs as mlcg's exact periodic neighbour list (all images)."""
    from mlcg.neighbor_list.torch_impl import torch_neighbor_list

    data = periodic_batch([CELLS["triclinic"], CELLS["cubic"]]).to("cuda")
    cell, pbc = periodic_cells(data.cell, data.pbc, 2, "cuda")
    row_ptr, col, valid = RadiusCSR(CUTOFF)(
        data.pos, data.batch, data.ptr, cell, pbc
    )
    i, j, _, _ = torch_neighbor_list(data, CUTOFF, self_interaction=False)
    expected = [set() for _ in range(data.pos.shape[0])]
    for a, b in zip(i.tolist(), j.tolist()):
        expected[a].add(b)
    assert csr_sets(row_ptr, col) == [sorted(r) for r in expected]
    assert valid.item() == 1.0


def test_tabulated_pbc_split_molecule_matches_whole():
    """A cluster straddling a corner of a large periodic box gives the same
    energy and forces as the whole cluster without boundaries."""
    tabulated = tabulate(build_schnet("exp_normal", "cosine", "tanh").cuda())
    whole = random_batch(sizes=(40,))
    box = 60.0
    split = AtomicData.from_points(
        pos=torch.remainder(whole.pos - whole.pos.mean(0), box),
        atom_types=whole.atom_types,
        cell=torch.eye(3) * box,
        pbc=torch.tensor([True] * 3),
    )
    split, _, _ = collate(AtomicData, [split], increment=True, add_batch=True)
    for a, b in zip(
        energy_forces(tabulated, whole, torch.float32),
        energy_forces(tabulated, split, torch.float32),
    ):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "cell",
    [
        [[10.0, 0, 0], [0, 14.0, 0], [0, 0, 14.0]],
        [[14.0, 0, 0], [9.0, 15.0, 0], [0, 0, 16.0]],
    ],
    ids=["narrower_than_2_cutoffs", "not_reduced"],
)
def test_unsupported_cell_raises_and_poisons(cell):
    model = GradientsOut(
        build_schnet("exp_normal", "cosine", "tanh"), targets=FORCE_KEY
    ).cuda()
    data = periodic_batch([cell]).to("cuda")
    data.out = {}
    with pytest.raises(ValueError, match="reduced form"):
        prepare_for_inference(model, data)
    tabulated = tabulate(build_schnet("exp_normal", "cosine", "tanh").cuda())
    energy = tabulated(data).out[tabulated.name][ENERGY_KEY]
    assert torch.isnan(energy).all()


def test_tabulated_pbc_step_has_no_host_sync():
    model = GradientsOut(
        tabulate(build_schnet("exp_normal", "cosine", "tanh").cuda()),
        targets=FORCE_KEY,
    ).eval()
    data = periodic_batch([CELLS["triclinic"], CELLS["cubic"]]).to("cuda")
    data.out = {}
    model(data)
    torch.cuda.set_sync_debug_mode("error")
    try:
        model(data)
    finally:
        torch.cuda.set_sync_debug_mode("default")


def test_tabulated_has_no_double_backward():
    tabulated = tabulate(build_schnet("exp_normal", "cosine", "tanh").cuda())
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
    row_ptr, col, valid = RadiusCSR(CUTOFF)(
        data.pos, data.batch, data.ptr, *NO_CELL
    )
    edge_index, _, _ = radius(data.pos, data.batch, CUTOFF)
    expected = [[] for _ in range(data.pos.shape[0])]
    for src, dst in edge_index.t().tolist():
        expected[dst].append(src)
    assert csr_sets(row_ptr, col) == [sorted(r) for r in expected]
    assert valid.item() == 1.0


def test_radius_csr_regrows_when_checked():
    data = random_batch().to("cuda")
    neighbors = RadiusCSR(CUTOFF, all_pairs_budget=0)
    args = (data.pos, data.batch, data.ptr, *NO_CELL)
    neighbors.reserve(*args)
    reference = csr_sets(*neighbors(*args)[:2])
    neighbors.capacity = 8
    row_ptr, col, valid = neighbors(*args)
    assert neighbors.capacity > 8
    assert csr_sets(row_ptr, col) == reference and valid.item() == 1.0


def test_tabulated_poisons_truncated_neighbor_lists():
    tabulated = tabulate(build_schnet("exp_normal", "cosine", "tanh").cuda())
    small_graphs = random_batch(sizes=(8, 8, 8, 8)).to("cuda")
    one_graph = random_batch(sizes=(32,)).to("cuda")
    one_graph.pos = one_graph.pos * 0.1
    tabulated(small_graphs)
    energy = tabulated(one_graph).out[tabulated.name][ENERGY_KEY]
    assert torch.isnan(energy).all()


def test_tabulated_step_has_no_host_sync():
    model = GradientsOut(
        tabulate(build_schnet("exp_normal", "cosine", "tanh").cuda()),
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


def test_optimize_for_inference_graphs_a_network_alone():
    """The network without priors (as in mlcg_to_openmm) keeps its outputs
    under its own name; the replay must return them, equal to the plain
    model."""
    from mlcg.nn.flash_models import GraphedModel, optimize_for_inference

    network = GradientsOut(
        build_schnet("exp_normal", "cosine", "tanh"), targets=FORCE_KEY
    ).cuda()
    data = random_batch().to("cuda")
    data.out = {}
    fast = optimize_for_inference(network, data)
    assert isinstance(fast, GraphedModel)
    graphed = fast(data).out[network.name]
    plain = fast.model(copy.deepcopy(data)).out[network.name]
    for key in (ENERGY_KEY, FORCE_KEY):
        torch.testing.assert_close(graphed[key], plain[key].detach())


@pytest.mark.parametrize(
    "matrix",
    [
        [[1.03, 0, 0], [0, 0.97, 0], [0, 0, 1.01]],
        [[1, 0, 0], [0.03, 1, 0], [0, 0, 1]],
    ],
)
def test_graph_follows_a_new_cell(matrix):
    """A barostat rescales or shears the box: the replay must use the cell
    of the data it is given, like the plain model, and a call without a
    cell must not reuse the captured one."""
    from mlcg.nn.flash_models import GraphedModel, optimize_for_inference

    network = GradientsOut(
        build_schnet("exp_normal", "cosine", "tanh"), targets=FORCE_KEY
    ).cuda()
    data = periodic_batch([CELLS["cubic"], CELLS["triclinic"]]).to("cuda")
    data.out = {}
    fast = optimize_for_inference(network, data)
    assert isinstance(fast, GraphedModel)
    matrix = torch.tensor(matrix, device="cuda")
    deformed = data.clone()
    deformed.pos = data.pos @ matrix
    deformed.cell = data.cell @ matrix
    graphed = fast(deformed.clone()).out[network.name][FORCE_KEY]
    plain = fast.model(deformed.clone()).out[network.name][FORCE_KEY]
    torch.testing.assert_close(graphed, plain.detach())
    deformed.cell = None
    with pytest.raises(ValueError, match="periodic cell"):
        fast(deformed)


def knots_of(flash):
    return [block.conv.table.shape[0] for block in flash.interaction_blocks]


def test_adaptive_grid_uses_fewest_power_of_two_knots():
    flash = StandardFlashSchNet.flash_from_standard(
        build_schnet("exp_normal", "cosine", "tanh").cuda()
    )
    knots = knots_of(flash.build_tables(tolerance=1e-5))
    assert all(k & (k - 1) == 0 for k in knots)
    for k in set(knots):
        if k > 2048:
            with pytest.raises(ValueError, match="Filter table error"):
                flash.build_tables(num_knots=k // 2, tolerance=1e-5)
    tighter = knots_of(flash.build_tables(tolerance=1e-6))
    assert all(t >= k for t, k in zip(tighter, knots))
    assert (
        knots_of(flash.build_tables(num_knots=3000, tolerance=1)) == [3000] * 2
    )


def test_adaptive_grid_refines_sharp_filters():
    """Narrow Gaussians make a sharp but smooth filter: the first grid
    misses the tolerance and the adaptive one refines until it meets it."""
    torch.manual_seed(0)
    standard = StandardSchNet(
        GaussianBasis(CosineCutoff(0.0, CUTOFF), num_rbf=800),
        CosineCutoff(0.0, CUTOFF),
        [40],
        hidden_channels=48,
        num_filters=48,
        num_interactions=1,
        activation=torch.nn.Tanh(),
    ).cuda()
    flash = StandardFlashSchNet.flash_from_standard(standard)
    with pytest.raises(ValueError, match="Filter table error"):
        flash.build_tables(num_knots=2048, tolerance=1e-5)
    assert knots_of(flash.build_tables(tolerance=1e-5))[0] > 2048
    assert_matches_float64(standard, flash.eval(), random_batch())


def test_tolerance_below_float32_floor_raises():
    model = GradientsOut(
        build_schnet("exp_normal", "cosine", "tanh"), targets=FORCE_KEY
    ).cuda()
    with pytest.raises(ValueError, match="float32 floor"):
        StandardFlashSchNet.flash_from_standard(model.model).build_tables(
            tolerance=1e-9
        )
    with pytest.raises(ValueError, match="float32 floor"):
        prepare_for_inference(model, random_batch().to("cuda"), tolerance=1e-9)


@pytest.mark.parametrize("cells", [["triclinic", "cubic"], ["orthorhombic"]])
def test_filter_path_pbc_matches_float64(cells):
    """The non-tabulated (training) path on the sync-free neighbour list,
    with periodic cells, against the standard model with all images."""
    standard = build_schnet("exp_normal", "cosine", "tanh")
    flash = StandardFlashSchNet.flash_from_standard(standard.cuda())
    data = periodic_batch([CELLS[c] for c in cells])
    assert_matches_float64(standard, flash, data)


def test_filter_path_force_matching_gradients():
    """Parameter gradients of a force loss (double backward), with periodic
    cells: the float32 flash model within 1e-4 (relative, by norm, as for
    the forces) of the standard model in float64."""
    torch.backends.cuda.matmul.allow_tf32 = False
    standard = GradientsOut(
        build_schnet("exp_normal", "cosine", "tanh"), targets=FORCE_KEY
    ).cuda()
    flash = convert_standard_model_to_flash(copy.deepcopy(standard))
    data = periodic_batch([CELLS["triclinic"], CELLS["cubic"]]).to("cuda")

    def gradients(model, dtype):
        model = copy.deepcopy(model).to(dtype).train()
        batch = copy.deepcopy(data)
        batch.pos, batch.cell = batch.pos.to(dtype), batch.cell.to(dtype)
        batch.out = {}
        forces = model(batch).out[model.name][FORCE_KEY]
        loss = forces.square().sum()
        return torch.autograd.grad(loss, list(model.parameters()))

    reference = gradients(standard, torch.float64)
    for g, g_ref in zip(gradients(flash, torch.float32), reference):
        assert (g.double() - g_ref).norm() <= 1e-4 * g_ref.norm()


def test_radius_csr_cpu_matches_gpu():
    data = periodic_batch([CELLS["triclinic"], CELLS["cubic"]])
    sets = []
    for device in ("cpu", "cuda"):
        d = data.to(device)
        cell, pbc = periodic_cells(d.cell, d.pbc, 2, device)
        sets.append(
            csr_sets(*RadiusCSR(CUTOFF)(d.pos, d.batch, d.ptr, cell, pbc)[:2])
        )
    assert sets[0] == sets[1]


@pytest.mark.parametrize(
    "basis, cutoff",
    [
        ("gaussian", "shifted_cosine"),
        ("exp_normal", "shifted_cosine"),
        ("extended_exp_normal", "cosine_lower"),
    ],
)
def test_cutoff_breakpoints_get_knots(basis, cutoff):
    """Cutoffs that change formula inside the range declare it; a knot is
    put there, so their filters meet the default tolerance."""
    torch.manual_seed(0)
    standard = build_schnet(basis, cutoff, "tanh").cuda()
    flash = StandardFlashSchNet.flash_from_standard(standard).build_tables()
    for block in flash.interaction_blocks:
        knots = block.conv.table.shape[0] - 1
        breakpoint = block.conv.cutoff.breakpoints()[0]
        offset = breakpoint / CUTOFF * knots
        assert abs(offset - round(offset)) <= 1e-3
    assert_matches_float64(standard, flash.eval(), random_batch())
