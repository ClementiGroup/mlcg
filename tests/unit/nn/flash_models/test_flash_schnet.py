import copy
import torch
from torch_geometric.data.collate import collate
import pytest

from mlcg.nn.schnet import StandardSchNet
from mlcg.nn.radial_basis import ExpNormalBasis
from mlcg.nn.gradients import GradientsOut, SumOut
from mlcg.nn.prior import HarmonicBonds
from mlcg.nn.flash_models import enable_flash_kernels
from mlcg.nn.cutoff import IdentityCutoff
from mlcg.geometry import Topology
from mlcg.data.atomic_data import AtomicData
from mlcg.data._keys import ENERGY_KEY, FORCE_KEY
from mlcg.mol_utils import MolDatabase
from ase.build import molecule

standard_basis = ExpNormalBasis(cutoff=5)
standard_cutoff = IdentityCutoff(cutoff_lower=0, cutoff_upper=5)

# prepare some data with ASE

database = MolDatabase()


@pytest.mark.parametrize(
    "collated_data, out_keys, expected_shapes",
    [
        (
            database.collated_data,
            [ENERGY_KEY, FORCE_KEY],
            [database.energy_shape, database.force_shape],
        )
    ],
)
def test_prediction(collated_data, out_keys, expected_shapes):
    """Test to make sure that the output dictionary is properly populated
    and that the correspdonding shapes of the outputs are correct given the
    requested gradient targets.
    """
    test_flash_schnet = StandardSchNet(
        standard_basis, standard_cutoff, [128, 128]
    )
    test_flash_schnet.enable_kernels()
    model = GradientsOut(test_flash_schnet, targets=FORCE_KEY).float()
    collated_data = model(collated_data)
    assert len(collated_data.out) != 0
    assert "SchNet" in collated_data.out.keys()
    for key, shape in zip(out_keys, expected_shapes):
        assert key in collated_data.out[model.name].keys()
        assert collated_data.out[model.name][key].shape == shape


@pytest.mark.parametrize(
    "collated_data, out_keys",
    [
        (
            database.collated_data,
            [ENERGY_KEY, FORCE_KEY],
        )
    ],
)
def test_flash_vs_normal(collated_data, out_keys):
    """Test to make sure that the standard schnet and the flash version
    output the same values up to a tolerance
    """
    test_flash_schnet = StandardSchNet(
        standard_basis, standard_cutoff, [128, 128]
    )
    test_flash_schnet.enable_kernels()
    flash_model = GradientsOut(test_flash_schnet, targets=FORCE_KEY).float()
    test_schnet = StandardSchNet(standard_basis, standard_cutoff, [128, 128])
    # make them have the same parameters
    test_schnet.load_state_dict(flash_model.model.state_dict())
    normal_model = GradientsOut(test_schnet, targets=FORCE_KEY).float()
    out_flash = flash_model(collated_data)
    out_normal = normal_model(collated_data)
    for key in out_keys:
        torch.testing.assert_close(
            out_flash.out[flash_model.name][key],
            out_normal.out[normal_model.name][key],
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels")
def test_flash_force_training_gradients():
    """Parameter gradients of a force loss (the double backward, through the
    radius graph used as its own transpose) are as close to the standard
    model in float64 as the standard model in float32 is."""
    models = [
        StandardSchNet(ExpNormalBasis(cutoff=5), standard_cutoff, [32, 32])
        for _ in range(3)
    ]
    for model in models[1:]:
        model.load_state_dict(models[0].state_dict())
    models[0].double()
    models[2].enable_kernels()

    def grads(schnet):
        model = GradientsOut(schnet.cuda(), targets=FORCE_KEY)
        data = database.collated_data.clone().to("cuda")
        data.pos = data.pos.to(next(schnet.parameters()).dtype)
        forces = model(data).out[model.name][FORCE_KEY]
        forces.pow(2).sum().backward()
        return torch.cat(
            [p.grad.flatten().double() for p in schnet.parameters()]
        )

    reference, standard, flash = (grads(model) for model in models)
    # the standard float32 error varies by ~1.4x between runs (atomic sums;
    # tests_benchmarks/30_edges/grad_check.log)
    standard_error = (standard - reference).abs().max()
    assert (flash - reference).abs().max() <= 2 * standard_error


def test_flash_reaches_nested_modules():
    """Every module with a flash path gets its kernels, at any depth and
    inside containers without one (a SumOut of a plain dict in a
    Sequential)."""
    schnet = StandardSchNet(standard_basis, standard_cutoff, [128, 128])
    bonds = HarmonicBonds({(1, 6): {"k": 1.0, "x_0": 1.0}})
    model = torch.nn.Sequential(
        SumOut(
            {
                "SchNet": GradientsOut(schnet, targets=FORCE_KEY),
                "bonds": GradientsOut(bonds, targets=FORCE_KEY),
            }
        )
    )
    enable_flash_kernels(model)
    assert schnet._kernels and bonds._kernels
    assert all(block.conv._kernels for block in schnet.interaction_blocks)


def _excluded_data(excluded_order):
    """Three molecules, their bonds stored once each as excluded pairs."""
    data_list = []
    for name in ["CH3CN", "C6H6", "trans-butane"]:
        mol = molecule(name)
        extra = {}
        if excluded_order > 1:
            extra["excluded_pairs"] = {
                "excluded_order": excluded_order,
                "index_mapping": Topology.from_ase(mol).bonds2torch(),
            }
        data_list.append(
            AtomicData.from_points(
                pos=torch.tensor(mol.get_positions()).float(),
                atom_types=torch.tensor(mol.get_atomic_numbers()),
                **extra,
            )
        )
    data, _, _ = collate(
        AtomicData, data_list=data_list, increment=True, add_batch=True
    )
    return data


def _excluded_models(excluded_order):
    schnet = StandardSchNet(
        standard_basis,
        standard_cutoff,
        [128, 128],
        excluded_order=excluded_order,
    )
    normal = GradientsOut(schnet, targets=FORCE_KEY).float()
    flash = enable_flash_kernels(copy.deepcopy(normal))
    return normal, flash


def test_flash_vs_normal_excluded_pairs():
    """The flash graph leaves out the excluded pairs as the standard
    neighbour list does: same energies and forces, which differ from those
    without exclusions."""
    data = _excluded_data(2)
    normal, flash = _excluded_models(2)
    out_normal = normal(copy.deepcopy(data)).out["SchNet"]
    out_flash = flash(copy.deepcopy(data)).out["SchNet"]
    for key in [ENERGY_KEY, FORCE_KEY]:
        torch.testing.assert_close(out_flash[key], out_normal[key])
    normal.model.excluded_order = 1
    full = normal(_excluded_data(1)).out["SchNet"][ENERGY_KEY]
    assert not torch.allclose(full, out_normal[ENERGY_KEY])


@pytest.mark.parametrize("model_order, data_order", [(2, 1), (1, 2), (3, 2)])
def test_flash_excluded_order_mismatch_raises(model_order, data_order):
    """Preparing checks the order; every call checks that the pairs are
    present exactly when the model excludes some (no host sync)."""
    _, flash = _excluded_models(model_order)
    data = _excluded_data(data_order)
    with pytest.raises(ValueError):
        flash.model.prepare_inference(data)
    if (model_order > 1) != (data_order > 1):
        with pytest.raises(ValueError):
            flash(data)


def test_stored_exclusions():
    """prepare_inference stores the exclusion CSR: same graph as building
    it per call; other pairs of the same shape give NaN; pairs of another
    shape are built per call; train() drops it."""
    _, flash = _excluded_models(2)
    data = _excluded_data(2)
    model = flash.model
    pairs = data.excluded_pairs["index_mapping"]
    args = model._radius_graph(data)[:2]
    model.prepare_inference(data)
    assert model.neighbors.exclusions is not None
    row_ptr, col, valid, _, _ = model._radius_graph(data)
    assert torch.equal(row_ptr, args[0]) and valid.item() == 1.0
    assert torch.equal(col[: int(row_ptr[-1])], args[1][: int(row_ptr[-1])])
    data.excluded_pairs["index_mapping"] = pairs + 1
    assert model._radius_graph(data)[2].isnan()
    data.excluded_pairs["index_mapping"] = pairs[:, 1:]
    assert model._radius_graph(data)[2].item() == 1.0
    flash.train()
    assert model.neighbors.exclusions is None
    _, flash = _excluded_models(1)
    flash.model.prepare_inference(AtomicData())
    assert flash.model.neighbors.exclusions is None
