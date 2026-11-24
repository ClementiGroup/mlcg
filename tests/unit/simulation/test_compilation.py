from typing import Union
import torch
import pytest
from copy import deepcopy

from mlcg.simulation.langevin import (
    LangevinSimulation,
    OverdampedSimulation,
)
from mlcg.simulation.parallel_tempering import PTSimulation
from mlcg.nn.flash_models import GraphedModel
from mlcg.data.atomic_data import AtomicData
from mlcg.data._keys import FORCE_KEY
from mlcg.nn import (
    StandardSchNet,
    StandardPaiNN,
    StandardSo3krates,
    StandardMACE,
    StandardAllegro,
    GradientsOut,
    SumOut,
    CosineCutoff,
    GaussianBasis,
)

from mlcg.mol_utils import _get_initial_data, _ASE_prior_model


@pytest.fixture
def get_initial_data():
    return _get_initial_data()


def ASE_prior_model(**kwargs):
    return _ASE_prior_model(**kwargs)


def ASE_SchNet_model(
    mol: str = "CH3CH2NH2", sum_out: bool = True, device="cpu"
) -> Union[torch.nn.Module, torch.nn.ModuleDict]:
    """Wrapper that returns a simple SchNet model of
    an ASE molecule.

    Parameters
    ----------
    mol:
        Molecule specifying string found in ase.build.molecule
        associated with the g2 organic molecule database
    sum_out:
        If True, the model constituents are wrapped within
        a SumOut instance
    device:
        Device on which to place the model

    Returns
    -------
    model_with_data:
        Dictionary that contains the fitted prior-only model
        under the key "model" and the noisey data, as a collated
        AtomicData instance, used for fitting under the key
        "collated_prior_data"
    """
    base_builder = ASE_prior_model(mol=mol, sum_out=False, device=device)

    nn_model = GradientsOut(
        StandardSchNet(
            rbf_layer=GaussianBasis(cutoff=CosineCutoff(0, 5), num_rbf=8),
            cutoff=CosineCutoff(0, 5),
            output_hidden_layer_widths=[8, 4],
            hidden_channels=8,
            embedding_size=10,
            num_filters=8,
            num_interactions=1,
        ),
        targets=[FORCE_KEY],
    )

    base_builder["model"][nn_model.name] = nn_model.to(device)
    if sum_out:
        base_builder["model"] = SumOut(base_builder["model"])

    return base_builder


def ASE_PaiNN_model(
    mol: str = "CH3CH2NH2", sum_out: bool = True, device="cpu"
) -> Union[torch.nn.Module, torch.nn.ModuleDict]:
    """Wrapper that returns a simple PaiNN model of
    an ASE molecule.

    Parameters
    ----------
    mol:
        Molecule specifying string found in ase.build.molecule
        associated with the g2 organic molecule database
    sum_out:
        If True, the model constituents are wrapped within
        a SumOut instance
    device:
        Device on which to place the model

    Returns
    -------
    model_with_data:
        Dictionary that contains the fitted prior-only model
        under the key "model" and the noisey data, as a collated
        AtomicData instance, used for fitting under the key
        "collated_prior_data"
    """
    base_builder = ASE_prior_model(mol=mol, sum_out=False, device=device)

    nn_model = GradientsOut(
        StandardPaiNN(
            rbf_layer=GaussianBasis(cutoff=CosineCutoff(0, 5), num_rbf=8),
            cutoff=CosineCutoff(0, 5),
            output_hidden_layer_widths=[8, 4],
            hidden_channels=8,
            embedding_size=10,
            num_interactions=1,
        ),
        targets=[FORCE_KEY],
    )

    base_builder["model"][nn_model.name] = nn_model.to(device)
    if sum_out:
        base_builder["model"] = SumOut(base_builder["model"])

    return base_builder


def ASE_SO3krates_model(
    mol: str = "CH3CH2NH2", sum_out: bool = True, device="cpu"
) -> Union[torch.nn.Module, torch.nn.ModuleDict]:
    """Wrapper that returns a simple SO3krates model of
    an ASE molecule.

    Parameters
    ----------
    mol:
        Molecule specifying string found in ase.build.molecule
        associated with the g2 organic molecule database
    sum_out:
        If True, the model constituents are wrapped within
        a SumOut instance
    device:
        Device on which to place the model

    Returns
    -------
    model_with_data:
        Dictionary that contains the fitted prior-only model
        under the key "model" and the noisey data, as a collated
        AtomicData instance, used for fitting under the key
        "collated_prior_data"
    """
    base_builder = ASE_prior_model(mol=mol, sum_out=False, device=device)

    nn_model = GradientsOut(
        StandardSo3krates(
            rbf_layer=GaussianBasis(cutoff=CosineCutoff(0, 5), num_rbf=8),
            cutoff=CosineCutoff(0, 5),
            output_hidden_layer_widths=[8, 4],
            hidden_channels=8,
            embedding_size=10,
            degrees=[1, 2],
            n_heads=2,
            num_interactions=1,
        ),
        targets=[FORCE_KEY],
    )

    base_builder["model"][nn_model.name] = nn_model.to(device)
    if sum_out:
        base_builder["model"] = SumOut(base_builder["model"])

    return base_builder


@pytest.mark.parametrize(
    "model, get_initial_data, add_masses, sim_class, sim_args, betas",
    [
        (
            ASE_prior_model,
            get_initial_data,
            True,
            OverdampedSimulation,
            [],
            1.0,
        ),
        (
            ASE_prior_model,
            get_initial_data,
            True,
            LangevinSimulation,
            [1.0],
            1.0,
        ),
        (
            ASE_prior_model,
            get_initial_data,
            True,
            PTSimulation,
            [1.0, 1],
            [1.67, 1.42, 1.28, 1.00, 0.8, 0.5],
        ),
        (
            ASE_SchNet_model,
            get_initial_data,
            True,
            LangevinSimulation,
            [1.0],
            1.0,
        ),
        (
            ASE_PaiNN_model,
            get_initial_data,
            True,
            LangevinSimulation,
            [1.0],
            1.0,
        ),
        (
            ASE_SO3krates_model,
            get_initial_data,
            True,
            LangevinSimulation,
            [1.0],
            1.0,
        ),
    ],
    indirect=["get_initial_data"],
)
def test_simulation_run(
    model, get_initial_data, add_masses, sim_class, sim_args, betas, tmp_path
):
    """Compare eager vs compiled simulation outputs."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dict = model(device=device)
    torch._functorch.config.donated_buffer = False

    def run_simulation(compile_flag):
        full_model = data_dict["model"]
        mol = data_dict["molecule"]
        neighbor_lists = data_dict["neighbor_lists"]
        initial_data_list = get_initial_data(
            mol, neighbor_lists, corruptor=None, add_masses=add_masses
        )

        sim_kwargs = {
            "filename": tmp_path
            / f"{sim_class.__name__}_{'compiled' if compile_flag else 'eager'}",
            "device": device,
            "optimize": {
                "flash": False,
                "compile": compile_flag,
                "cuda_graph": False,
            },
        }

        sim = sim_class(*sim_args, **sim_kwargs)
        sim.attach_model_and_configurations(
            full_model, initial_data_list, betas
        )
        sim._set_up_simulation(overwrite=False)
        data = deepcopy(sim.initial_data).to(sim.device)
        sim.compile_model(data)
        return sim.calculate_potential_and_forces(data)

    energy_compiled, forces_compiled = run_simulation(compile_flag=True)
    energy_eager, forces_eager = run_simulation(compile_flag=False)

    torch.testing.assert_close(
        energy_eager, energy_compiled, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        forces_eager, forces_compiled, atol=1e-5, rtol=1e-5
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("compile_flag", [False, True])
def test_cuda_graph_matches_plain_model(
    get_initial_data, compile_flag, tmp_path
):
    """The replayed graph gives the plain model's outputs after MD steps."""
    torch._functorch.config.donated_buffer = False
    data_dict = ASE_SchNet_model(device="cuda")
    initial_data_list = get_initial_data(
        data_dict["molecule"],
        data_dict["neighbor_lists"],
        corruptor=None,
        add_masses=True,
    )
    sim = LangevinSimulation(
        1.0,
        filename=tmp_path / "graph",
        device="cuda",
        optimize={"compile": compile_flag},
    )
    sim.attach_model_and_configurations(
        data_dict["model"], initial_data_list, 1.0
    )
    sim._set_up_simulation(overwrite=False)
    data = deepcopy(sim.initial_data).to(sim.device)
    sim.compile_model(data)
    assert sim._graphed_model is not None
    _, forces = sim.calculate_potential_and_forces(data)
    for _ in range(20):
        data, _, forces = sim.timestep(data, forces)

    graphed = sim.calculate_potential_and_forces(data)
    sim._graphed_model = None
    plain = sim.calculate_potential_and_forces(data)
    for g, p in zip(graphed, plain):
        torch.testing.assert_close(g, p, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_graph_rejects_host_sync():
    """A model that synchronizes with the host fails on its first warm-up
    call, before any capture is attempted."""

    class Syncing(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, data):
            self.calls += 1
            energy = data.pos.sum().item() * torch.ones(1, device="cuda")
            data.out = {"energy": energy, "forces": data.pos}
            return data

    data = AtomicData.from_points(
        pos=torch.randn(5, 3), atom_types=torch.ones(5, dtype=torch.long)
    ).to("cuda")
    data.out = {}
    model = Syncing()
    with pytest.raises(RuntimeError):
        GraphedModel(model, data)
    assert model.calls == 1


@pytest.mark.parametrize(
    "model, convert_to_flash",
    [
        (ASE_prior_model, False),
        (ASE_SchNet_model, False),
        (ASE_SchNet_model, True),
        (ASE_PaiNN_model, False),
        (ASE_SO3krates_model, False),
    ],
)
def test_compile_single_structure(
    model, convert_to_flash, get_initial_data, tmp_path
):
    """With one structure in the batch (``batch.max() == 0``) the compiled
    model gives the eager energies and forces."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dict = model(device=device)
    torch._functorch.config.donated_buffer = False
    initial_data_list = get_initial_data(
        data_dict["molecule"],
        data_dict["neighbor_lists"],
        corruptor=None,
        add_masses=True,
    )[:1]

    def potential_and_forces(compile_flag):
        sim = LangevinSimulation(
            1.0,
            filename=tmp_path / f"single_{compile_flag}",
            device=device,
            optimize={
                "flash": convert_to_flash,
                "compile": compile_flag,
                "cuda_graph": False,
            },
        )
        sim.attach_model_and_configurations(
            data_dict["model"], initial_data_list, 1.0
        )
        sim._set_up_simulation(overwrite=False)
        data = deepcopy(sim.initial_data).to(sim.device)
        assert data.batch.max() == 0
        sim.compile_model(data)
        return sim.calculate_potential_and_forces(data)

    energy_eager, forces_eager = potential_and_forces(False)
    energy_compiled, forces_compiled = potential_and_forces(True)
    torch.testing.assert_close(
        energy_eager, energy_compiled, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        forces_eager, forces_compiled, atol=1e-5, rtol=1e-5
    )


def test_optimize_is_off_by_default(tmp_path):
    """A simulation runs the model as given unless ``optimize`` is set."""
    assert LangevinSimulation(1.0, filename=tmp_path / "sim").optimize is None


def test_deprecated_compile_options(tmp_path):
    """The compile options of earlier versions work for one release, with a
    deprecation warning: ``compile`` is ``torch.compile`` alone, the others
    are ignored."""

    def simulation(**kwargs):
        return LangevinSimulation(1.0, filename=tmp_path / "sim", **kwargs)

    with pytest.warns(DeprecationWarning, match="`compile` is deprecated"):
        sim = simulation(compile=True)
    assert sim.optimize == {"flash": False, "cuda_graph": False}
    for name, value in (("compile_mode", "default"), ("force_compile", True)):
        with pytest.warns(DeprecationWarning, match=f"`{name}` is deprecated"):
            sim = simulation(**{name: value})
        assert sim.optimize is None
    with pytest.raises(ValueError, match="without the deprecated"):
        simulation(optimize=True, compile=True)


def test_optimize_rejects_other_values(tmp_path):
    """``optimize`` is ``None``, ``"auto"`` or a dict."""
    with pytest.raises(ValueError, match="`optimize` is None"):
        LangevinSimulation(1.0, filename=tmp_path / "sim", optimize=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("compiled_ms", [0.5, 0.99, None])
def test_optimize_auto_keeps_the_faster(
    get_initial_data, compiled_ms, tmp_path, monkeypatch
):
    """``optimize="auto"`` keeps the compiled model only if it is at least
    3% faster than eager (1 ms here), and runs eager if the model doesn't
    compile (``None``); either way it gives the eager outputs. Compiling and
    timing are stubbed."""
    from mlcg.nn.flash_models import inference

    def compile_stub(model, data):
        if compiled_ms is None:
            raise RuntimeError("torch.compile failed")
        return model

    times = iter([1.0, compiled_ms])
    monkeypatch.setattr(inference, "_compile", compile_stub)
    monkeypatch.setattr(
        inference, "_ms_per_call", lambda model, data: (next(times), 0.0)
    )
    data_dict = ASE_SchNet_model(device="cuda")
    initial_data_list = get_initial_data(
        data_dict["molecule"],
        data_dict["neighbor_lists"],
        corruptor=None,
        add_masses=True,
    )

    def potential_and_forces(name, optimize):
        sim = LangevinSimulation(
            1.0, filename=tmp_path / name, device="cuda", optimize=optimize
        )
        sim.attach_model_and_configurations(
            data_dict["model"], initial_data_list, 1.0
        )
        sim._set_up_simulation(overwrite=False)
        data = deepcopy(sim.initial_data).to(sim.device)
        sim.compile_model(data)
        return sim, sim.calculate_potential_and_forces(data)

    if compiled_ms is None:
        with pytest.warns(UserWarning, match="without torch.compile"):
            sim, auto = potential_and_forces("auto", "auto")
    else:
        sim, auto = potential_and_forces("auto", "auto")
    assert sim.optimize_report["compiled"] == (compiled_ms == 0.5)
    assert sim._graphed_model is not None
    _, eager = potential_and_forces("eager", {"compile": False})
    for a, e in zip(auto, eager):
        torch.testing.assert_close(a, e, atol=1e-5, rtol=1e-5)


def test_optimize_auto_on_cpu(get_initial_data, tmp_path, monkeypatch):
    """``optimize="auto"`` on the CPU: the exact flash kernels, without
    filter tables or a CUDA graph, give the outputs of the model as given.
    Compiling is stubbed."""
    from mlcg.nn.flash_models import inference

    monkeypatch.setattr(inference, "_compile", lambda model, data: model)
    data_dict = ASE_SchNet_model(device="cpu")
    initial_data_list = get_initial_data(
        data_dict["molecule"],
        data_dict["neighbor_lists"],
        corruptor=None,
        add_masses=True,
    )

    def potential_and_forces(name, optimize):
        sim = LangevinSimulation(
            1.0, filename=tmp_path / name, device="cpu", optimize=optimize
        )
        sim.attach_model_and_configurations(
            data_dict["model"], initial_data_list, 1.0
        )
        sim._set_up_simulation(overwrite=False)
        data = deepcopy(sim.initial_data)
        sim.compile_model(data)
        return sim, sim.calculate_potential_and_forces(data)

    sim, auto = potential_and_forces("auto", "auto")
    assert sim._graphed_model is None
    schnets = [m for m in sim.model.modules() if isinstance(m, StandardSchNet)]
    assert schnets[0]._kernels
    assert schnets[0].interaction_blocks[0].conv.table is None
    _, plain = potential_and_forces("plain", None)
    for a, p in zip(auto, plain):
        torch.testing.assert_close(a, p, atol=1e-5, rtol=1e-5)


def test_deprecated_compile_falls_back(get_initial_data, tmp_path, monkeypatch):
    """With the deprecated ``compile``, a model that doesn't compile runs
    with a warning (its failing parts eager, as before ``optimize``); with
    ``optimize`` the failure raises."""
    from mlcg.nn.flash_models import inference

    def compile_fails(model, data):
        raise RuntimeError("torch.compile failed on this model.")

    monkeypatch.setattr(inference, "_compile", compile_fails)
    data_dict = ASE_prior_model()
    initial_data_list = get_initial_data(
        data_dict["molecule"],
        data_dict["neighbor_lists"],
        corruptor=None,
        add_masses=True,
    )

    def compile_model(name, **kwargs):
        sim = LangevinSimulation(1.0, filename=tmp_path / name, **kwargs)
        sim.attach_model_and_configurations(
            data_dict["model"], initial_data_list, 1.0
        )
        sim._set_up_simulation(overwrite=False)
        sim.compile_model(deepcopy(sim.initial_data))

    with pytest.warns(UserWarning, match="falls back"):
        with pytest.warns(DeprecationWarning):
            compile_model("deprecated", compile=True)
    with pytest.raises(RuntimeError, match="failed"):
        compile_model(
            "optimize", optimize={"flash": False, "cuda_graph": False}
        )


def ASE_network_model(network, device):
    """The ASE molecule's priors plus ``network``, as the models above."""
    builder = ASE_prior_model(sum_out=False, device=device)
    nn_model = GradientsOut(network, targets=[FORCE_KEY]).to(device)
    builder["model"][nn_model.name] = nn_model
    builder["model"] = SumOut(builder["model"])
    return builder


MACE_CONFIG = {
    "r_max": 5,
    "num_bessel": 8,
    "num_polynomial_cutoff": 5,
    "max_ell": 1,
    "interaction_cls": "mace.modules.blocks.RealAgnosticResidualInteractionBlock",
    "interaction_cls_first": "mace.modules.blocks.RealAgnosticResidualInteractionBlock",
    "num_interactions": 1,
    "hidden_irreps": "8x0e",
    "MLP_irreps": "8x0e",
    "avg_num_neighbors": 9,
    "correlation": 2,
    "gate": torch.nn.Tanh(),
    "max_num_neighbors": 1000,
    "pair_repulsion": False,
    "distance_transform": None,
    "radial_MLP": [8, 8],
    "radial_type": "bessel",
    "atomic_numbers": [1, 6, 7],
}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize(
    "model",
    [
        ASE_PaiNN_model,
        ASE_SO3krates_model,
        lambda device: ASE_network_model(StandardMACE(**MACE_CONFIG), device),
        lambda device: ASE_network_model(
            StandardAllegro(r_max=5.0, num_layers=1), device
        ),
    ],
    ids=["PaiNN", "So3krates", "MACE", "Allegro"],
)
def test_optimize_auto_runs_other_models(
    get_initial_data, model, tmp_path, monkeypatch
):
    """``optimize="auto"`` runs a model without flash kernels: its neighbour
    search synchronizes with the host, so it gets no CUDA graph (with one
    warning), and it gives the outputs of ``optimize=None``. Compiling and
    timing are stubbed."""
    from mlcg.nn.flash_models import inference

    monkeypatch.setattr(inference, "_compile", lambda model, data: model)
    monkeypatch.setattr(
        inference, "_ms_per_call", lambda model, data: (1.0, 0.0)
    )
    data_dict = model(device="cuda")
    initial_data_list = get_initial_data(
        data_dict["molecule"],
        data_dict["neighbor_lists"],
        corruptor=None,
        add_masses=True,
    )

    def potential_and_forces(name, optimize):
        sim = LangevinSimulation(
            1.0, filename=tmp_path / name, device="cuda", optimize=optimize
        )
        sim.attach_model_and_configurations(
            data_dict["model"], initial_data_list, 1.0
        )
        sim._set_up_simulation(overwrite=False)
        data = deepcopy(sim.initial_data).to(sim.device)
        sim.compile_model(data)
        return sim, sim.calculate_potential_and_forces(data)

    with pytest.warns(UserWarning, match="CUDA graph capture") as record:
        sim, auto = potential_and_forces("auto", "auto")
    assert sum("CUDA graph capture" in str(w.message) for w in record) == 1
    assert sim._graphed_model is None
    _, plain = potential_and_forces("plain", None)
    for a, p in zip(auto, plain):
        torch.testing.assert_close(a, p, atol=1e-5, rtol=1e-5)
