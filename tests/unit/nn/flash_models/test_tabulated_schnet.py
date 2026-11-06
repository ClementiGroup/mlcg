import copy
import io

import pytest
import torch
from torch_geometric.data.collate import collate

from mlcg.data.atomic_data import AtomicData
from mlcg.data._keys import ENERGY_KEY, FORCE_KEY
from mlcg.neighbor_list.neighbor_list import atomic_data2neighbor_list
from mlcg.nn.cutoff import CosineCutoff, IdentityCutoff, ShiftedCosineCutoff
from mlcg.nn.flash_models import enable_flash_kernels, prepare_for_inference
from mlcg.nn.gradients import GradientsOut
from mlcg.nn.kernels import RadiusCSR
from mlcg.nn.kernels.pbc import periodic_cells
from mlcg.nn.radial_basis import (
    ExpNormalBasis,
    ExtendedExpNormalBasis,
    GaussianBasis,
    RIGTOBasis,
    SpacedExpBasis,
)
from mlcg.nn.schnet import RBFRegularizedSchNet, StandardSchNet

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


def flash_of(standard):
    """A copy of ``standard`` with its flash kernels on."""
    flash = copy.deepcopy(standard)
    flash.enable_kernels()
    return flash


def build_tables(flash, **kwargs):
    """Filter tables of every block of ``flash``; raises if one fails."""
    for block in flash.interaction_blocks:
        block.conv.build_table(flash.rbf_layer, **kwargs)
    return flash


def tabulated(schnet):
    return all(b.conv.table is not None for b in schnet.interaction_blocks)


def tabulate(standard, tolerance=1e-4):
    """The inference model: flash kernels, filter tables, eval mode. The
    accuracy tests were frozen with the 1e-4 table tolerance of 02."""
    return build_tables(flash_of(standard), tolerance=tolerance).eval()


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
    flash = flash_of(build_schnet("exp_normal", "cosine", "tanh").cuda())
    for block in flash.interaction_blocks:
        block.conv.filter_network.layers[1] = torch.nn.ReLU()
    return flash


def test_build_tables_rejects_non_smooth_filter():
    with pytest.raises(ValueError, match="Filter table error"):
        build_tables(non_smooth_flash_schnet())


def test_prepare_for_inference_keeps_filters_when_not_tabulable():
    model = GradientsOut(non_smooth_flash_schnet(), targets=FORCE_KEY)
    with pytest.warns(UserWarning, match="Filter table error"):
        model = prepare_for_inference(model, random_batch().to("cuda"))
    assert not tabulated(model.model) and not model.training


def test_conversion_keeps_the_filter_activation():
    standard = build_schnet("exp_normal", "cosine", "tanh").cuda()
    for block in standard.interaction_blocks:
        block.conv.filter_network.layers[1] = torch.nn.SiLU()
    flash = flash_of(standard)
    for block in flash.interaction_blocks:
        assert isinstance(block.conv.filter_network.layers[1], torch.nn.SiLU)
        assert isinstance(block.activation, torch.nn.Tanh)
    data = random_batch()
    assert_matches_float64(standard, flash, data)
    assert_matches_float64(standard, build_tables(flash).eval(), data)


def test_enable_flash_kernels_is_trainable_and_idempotent():
    standard = GradientsOut(
        build_schnet("exp_normal", "cosine", "tanh"), targets=FORCE_KEY
    ).cuda()
    flash = enable_flash_kernels(copy.deepcopy(standard))
    assert flash.model._kernels and not tabulated(flash.model)
    assert flash.training == standard.training
    assert enable_flash_kernels(flash) is flash and flash.model._kernels


def test_eval_without_tables_is_the_filter_path():
    model = GradientsOut(
        flash_of(build_schnet("exp_normal", "cosine", "tanh").cuda()),
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
    assert not tabulated(model.model)
    for train, evaluated in zip(*outputs):
        torch.testing.assert_close(train, evaluated)


def test_train_and_load_state_dict_drop_tables():
    flash = tabulate(build_schnet("exp_normal", "cosine", "tanh").cuda())
    assert all(b.conv.table is not None for b in flash.interaction_blocks)
    flash.train()
    assert not tabulated(flash)
    assert all(b.conv.table is None for b in flash.interaction_blocks)
    build_tables(flash).eval()
    flash.load_state_dict(flash.state_dict())
    assert not tabulated(flash)


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


def test_radius_csr_matches_standard_neighbor_list():
    data = random_batch().to("cuda")
    row_ptr, col, valid = RadiusCSR(CUTOFF)(
        data.pos, data.batch, data.ptr, *NO_CELL
    )
    edge_index = atomic_data2neighbor_list(data, CUTOFF)["index_mapping"]
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


COMPRESSIONS = (1.0, 0.5, 0.2, 0.02)


def assert_all_pairs_at_last_compression(data):
    pos = data.pos * COMPRESSIONS[-1]
    for start, end in zip(data.ptr[:-1], data.ptr[1:]):
        assert torch.cdist(pos[start:end], pos[start:end]).max() < CUTOFF


def test_graph_follows_a_compressing_system():
    """A system compressed during the replays, up to every pair within the
    cutoff: the all-pairs capacity holds every edge, so the replay equals
    the plain model at each step."""
    from mlcg.nn.flash_models import GraphedModel, optimize_for_inference

    network = GradientsOut(
        build_schnet("exp_normal", "cosine", "tanh"), targets=FORCE_KEY
    ).cuda()
    data = random_batch().to("cuda")
    assert_all_pairs_at_last_compression(data)
    data.out = {}
    fast = optimize_for_inference(network, data)
    assert isinstance(fast, GraphedModel)
    for scale in COMPRESSIONS:
        step = data.clone()
        step.pos = data.pos * scale
        graphed = fast(step.clone()).out[network.name][FORCE_KEY]
        plain = fast.model(step.clone()).out[network.name][FORCE_KEY]
        assert torch.isfinite(graphed).all()
        torch.testing.assert_close(graphed, plain.detach())


def test_checked_mode_runs_without_graph_and_regrows():
    """Above the all-pairs budget (forced here with a budget of 0), the
    neighbour search checks the edge count on the host: no CUDA graph, and
    a compressed system regrows the capacity and equals the all-pairs
    model."""
    from mlcg.nn.flash_models import GraphedModel, optimize_for_inference

    network = GradientsOut(
        build_schnet("exp_normal", "cosine", "tanh"), targets=FORCE_KEY
    ).cuda()
    data = random_batch().to("cuda")
    data.out = {}
    all_pairs = prepare_for_inference(network, data.clone())
    network.model.neighbors = RadiusCSR(CUTOFF, all_pairs_budget=0)
    with pytest.warns(UserWarning, match="CUDA graph capture failed"):
        fast = optimize_for_inference(network, data.clone())
    assert not isinstance(fast, GraphedModel)
    capacity = fast.model.neighbors.capacity
    for scale in COMPRESSIONS:
        step = data.clone()
        step.pos = data.pos * scale
        forces = fast(step.clone()).out[network.name][FORCE_KEY]
        reference = all_pairs(step.clone()).out[network.name][FORCE_KEY]
        torch.testing.assert_close(forces, reference)
    assert fast.model.neighbors.capacity > capacity


def knots_of(flash):
    return [block.conv.table.shape[0] for block in flash.interaction_blocks]


def test_adaptive_grid_uses_fewest_power_of_two_knots():
    flash = flash_of(build_schnet("exp_normal", "cosine", "tanh").cuda())
    knots = knots_of(build_tables(flash, tolerance=1e-5))
    assert all(k & (k - 1) == 0 for k in knots)
    for k in set(knots):
        if k > 2048:
            with pytest.raises(ValueError, match="Filter table error"):
                build_tables(flash, num_knots=k // 2, tolerance=1e-5)
    tighter = knots_of(build_tables(flash, tolerance=1e-6))
    assert all(t >= k for t, k in zip(tighter, knots))
    assert (
        knots_of(build_tables(flash, num_knots=3000, tolerance=1)) == [3000] * 2
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
    flash = flash_of(standard)
    with pytest.raises(ValueError, match="Filter table error"):
        build_tables(flash, num_knots=2048, tolerance=1e-5)
    assert knots_of(build_tables(flash, tolerance=1e-5))[0] > 2048
    assert_matches_float64(standard, flash.eval(), random_batch())


def test_tolerance_below_float32_floor_raises():
    model = GradientsOut(
        build_schnet("exp_normal", "cosine", "tanh"), targets=FORCE_KEY
    ).cuda()
    with pytest.raises(ValueError, match="float32 floor"):
        build_tables(flash_of(model.model), tolerance=1e-9)
    with pytest.raises(ValueError, match="float32 floor"):
        prepare_for_inference(model, random_batch().to("cuda"), tolerance=1e-9)


@pytest.mark.parametrize("cells", [["triclinic", "cubic"], ["orthorhombic"]])
def test_filter_path_pbc_matches_float64(cells):
    """The non-tabulated (training) path on the sync-free neighbour list,
    with periodic cells, against the standard model with all images."""
    standard = build_schnet("exp_normal", "cosine", "tanh")
    flash = flash_of(standard.cuda())
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
    flash = enable_flash_kernels(copy.deepcopy(standard))
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
    flash = build_tables(flash_of(standard))
    for block in flash.interaction_blocks:
        knots = block.conv.table.shape[0] - 1
        breakpoint = block.conv.cutoff.breakpoints()[0]
        offset = breakpoint / CUTOFF * knots
        assert abs(offset - round(offset)) <= 1e-3
    assert_matches_float64(standard, flash.eval(), random_batch())


def test_saved_model_is_the_standard_model():
    """Pickling drops the flash state (kernels, tables, neighbour search);
    a deep copy keeps it."""
    model = GradientsOut(
        build_schnet("exp_normal", "cosine", "tanh"), targets=FORCE_KEY
    ).cuda()
    data = random_batch().to("cuda")
    fast = prepare_for_inference(model, data)
    assert tabulated(copy.deepcopy(fast).model)
    buffer = io.BytesIO()
    torch.save(fast, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=False)
    for module in loaded.modules():
        state = (vars(module), module._buffers, module._modules)
        for name in getattr(module, "_runtime_attributes", ()):
            assert not any(name in s for s in state)
    torch.testing.assert_close(loaded.state_dict(), model.state_dict())
    torch.testing.assert_close(  # float32 tolerances: atomic-add scatter
        energy_forces(loaded.model, random_batch(), torch.float32),
        energy_forces(model.model, random_batch(), torch.float32),
        rtol=1.3e-6,
        atol=1e-5,
    )


class HalvedBlock(torch.nn.Module):
    """A custom block (not an InteractionBlock) that wraps a standard one."""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x, *args):
        return 0.5 * self.block(x, *args)


def test_standard_schnet_needs_flash_blocks():
    standard = build_schnet("exp_normal", "cosine", "tanh").cuda()
    blocks = standard.interaction_blocks
    blocks[1] = HalvedBlock(blocks[1])
    with pytest.warns(UserWarning, match="blocks are not all InteractionBlock"):
        assert not flash_of(standard)._kernels


class NewForward(StandardSchNet):
    def forward(self, data):
        return super().forward(data)


class NewForwardAndFlash(NewForward):
    def flash_forward(self, data):
        return super().flash_forward(data)


def test_forward_override_needs_flash_forward():
    """A subclass that changes forward would otherwise inherit the flash
    path of its parent's model."""
    standard = build_schnet("exp_normal", "cosine", "tanh")
    standard.__class__ = NewForward
    with pytest.warns(UserWarning, match="without a flash_forward"):
        assert not flash_of(standard)._kernels
    standard.__class__ = NewForwardAndFlash
    assert flash_of(standard)._kernels


def rbf_regularized_schnet():
    torch.manual_seed(0)
    model = RBFRegularizedSchNet(
        BASES["exp_normal"](CUTOFFS["cosine"]()),
        CUTOFFS["cosine"](),
        [40],
        hidden_channels=48,
        embedding_size=10,
        num_filters=48,
        num_interactions=2,
        activation=torch.nn.Tanh(),
        independent_regularizations=True,
    )
    for regularization in model.rbf_layer.regularization:
        regularization.w.data.uniform_(0.0, 1.0)
    return GradientsOut(model, targets=FORCE_KEY).cuda()


def test_rbf_regularized_flash_matches_float64():
    """Exact kernels with a regularized expansion per block (no tables: the
    filters depend on the bead types), through ``prepare_for_inference``
    and in training: energies, forces and the parameter gradients of a force
    loss as close to the standard model in float64 as the standard model in
    float32 is (2x: the float32 error varies between runs, see
    tests_benchmarks/32_flash_blocks)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    standard = rbf_regularized_schnet()
    data = random_batch()
    fast = prepare_for_inference(standard, copy.deepcopy(data).to("cuda"))
    assert fast.model._kernels
    assert not fast.model.interaction_blocks[0].conv.tabulated
    flash = enable_flash_kernels(copy.deepcopy(standard))

    def gradients(model, dtype):
        model = copy.deepcopy(model).to(dtype).train()
        batch = copy.deepcopy(data).to("cuda")
        batch.pos = batch.pos.to(dtype)
        batch.out = {}
        loss = model(batch).out[model.name][FORCE_KEY].square().sum()
        grads = torch.autograd.grad(loss, list(model.parameters()))
        return torch.cat([g.flatten().double() for g in grads])

    def outputs(evaluated, trained, dtype):
        energy, forces = energy_forces(evaluated, data, dtype)
        return energy, forces, gradients(trained, dtype)

    reference = outputs(standard.model, standard, torch.float64)
    standard_32 = outputs(standard.model, standard, torch.float32)
    flash_32 = outputs(fast.model, flash, torch.float32)
    for s32, f32, ref in zip(standard_32, flash_32, reference):
        assert (f32 - ref).abs().max() <= 2 * (s32 - ref).abs().max()
