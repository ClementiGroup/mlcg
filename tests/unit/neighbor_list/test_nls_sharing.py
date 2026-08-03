import os

import pytest
import torch
from torch_geometric.data.collate import collate

from mlcg.data import AtomicData
from mlcg.nn.prior.repulsion import FastCutoffRepulsion,FastCutoffExpRepulsion

try:
    import nvalchemiops  # noqa: F401

    NVALCHEMI_AVAILABLE = True
except ImportError:
    NVALCHEMI_AVAILABLE = False

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and NVALCHEMI_AVAILABLE),
    reason="requires a CUDA device and the nvalchemi-toolkit-ops package",
)


def _load_collated_data(asset_name: str, n_replicas: int) -> AtomicData:
    confs = []
    for _ in range(n_replicas):
        confs += torch.load(
            os.path.join(ASSETS_DIR, f"{asset_name}.pt"),
            weights_only=False,
        )
    col_data, _, _ = collate(AtomicData, confs)
    col_data = col_data.to("cuda")
    col_data.pos = col_data.pos.to(torch.float32)
    if "cell" in col_data:
        col_data.cell = col_data.cell.to(torch.float32)
    return col_data


@pytest.fixture(
    params=[
        "dissolved_with_nonbonded_exclusion",
        "ordered_with_nonbonded_exclusion",
        "circular_with_nonbonded_exclusion",
    ]
)
def col_data(request):
    n_replicas = 3
    return _load_collated_data(request.param, n_replicas)


@pytest.mark.parametrize(
    "nls_distance_method", ["torch", "nvalchemi_naive", "nvalchemi_cell"]
)
def test_fast_cutoff_exp_repulsion_and_network_nls_match(
    nls_distance_method, col_data
):
    """
    A model using `FastCutoffExpRepulsion` (a cutoff-table based
    reimplementation of `CutoffExpRepulsion`) together with the
    `nls_distance_method` neighbor list method should produce the same
    energies and forces as the reference model, which uses the base
    repulsion prior and the default `torch` neighbor list. Consistency
    is checked over many steps of a randomly perturbed trajectory so
    that both implementations are exercised over an evolving set of
    neighbor lists.
    """
    n_steps = 20
    n_replicas = col_data.n_atoms.shape[0]

    model = torch.load(
        os.path.join(ASSETS_DIR, "exp_rep_model.pt"), weights_only=False
    )
    new_model = torch.load(
        os.path.join(ASSETS_DIR, "exp_rep_model.pt"), weights_only=False
    )
    new_model.models["non_bonded"].model = FastCutoffExpRepulsion.from_base(
        model.models["non_bonded"].model
    )
    new_model.models["SchNet"].model.nls_distance_method = nls_distance_method

    model = model.to("cuda")
    new_model = new_model.to("cuda")

    old_eners = torch.zeros(n_steps, n_replicas)
    old_forces = torch.zeros(n_steps, *col_data.pos.shape)
    new_eners = torch.zeros(n_steps, n_replicas)
    new_forces = torch.zeros(n_steps, *col_data.pos.shape)

    for i in range(n_steps):
        model(col_data)
        old_eners[i] = col_data.out["non_bonded"]["energy"].detach().cpu()
        old_forces[i] = col_data.out["non_bonded"]["forces"].detach().cpu()
        col_data.out = {}

        new_model(col_data)
        new_eners[i] = col_data.out["non_bonded"]["energy"].detach().cpu()
        new_forces[i] = col_data.out["non_bonded"]["forces"].detach().cpu()

        # Advance positions along the current forces plus noise so that
        # consistency is checked across an evolving set of neighbor lists,
        # rather than just the initial configuration.
        col_data.pos += 1e-5 * col_data.out["non_bonded"][
            "forces"
        ] + 1e-1 * torch.randn(col_data.pos.shape, device=col_data.pos.device)
        col_data.out = {}

    atol_energy = torch.mean(old_eners) * 1e-5
    torch.testing.assert_close(
        new_eners, old_eners, atol=atol_energy, rtol=1e-5
    )

    atol_forces = torch.mean(torch.abs(old_forces)) * 1e-3
    torch.testing.assert_close(
        new_forces, old_forces, atol=atol_forces, rtol=1e-3
    )

@pytest.fixture(
    params=[
        "dissolved_with_nonbonded_exclusion",
        "ordered_with_nonbonded_exclusion",
        "circular_with_nonbonded_exclusion",
    ]
)
def col_data(request):
    n_replicas = 3
    return _load_collated_data(request.param, n_replicas)


@pytest.mark.parametrize(
    "nls_distance_method", ["torch", "nvalchemi_naive", "nvalchemi_cell"]
)
def test_fast_cutoff_repulsion_and_network_nls_match(
    nls_distance_method, col_data
):
    """
    A model using `FastCutoffRepulsion` (a cutoff-table based
    reimplementation of `CutoffRepulsion`) together with the
    `nls_distance_method` neighbor list method should produce the same
    energies and forces as the reference model, which uses the base
    repulsion prior and the default `torch` neighbor list. Consistency
    is checked over many steps of a randomly perturbed trajectory so
    that both implementations are exercised over an evolving set of
    neighbor lists.
"""
    n_steps = 20
    n_replicas = col_data.n_atoms.shape[0]

    model = torch.load(
        os.path.join(ASSETS_DIR, "rep_model.pt"), weights_only=False
    )
    new_model = torch.load(
        os.path.join(ASSETS_DIR, "rep_model.pt"), weights_only=False
    )
    new_model.models["non_bonded"].model = FastCutoffRepulsion.from_base(
        model.models["non_bonded"].model
    )
    new_model.models["SchNet"].model.nls_distance_method = nls_distance_method

    model = model.to("cuda")
    new_model = new_model.to("cuda")

    old_eners = torch.zeros(n_steps, n_replicas)
    old_forces = torch.zeros(n_steps, *col_data.pos.shape)
    new_eners = torch.zeros(n_steps, n_replicas)
    new_forces = torch.zeros(n_steps, *col_data.pos.shape)

    for i in range(n_steps):
        model(col_data)
        old_eners[i] = col_data.out["non_bonded"]["energy"].detach().cpu()
        old_forces[i] = col_data.out["non_bonded"]["forces"].detach().cpu()
        col_data.out = {}

        new_model(col_data)
        new_eners[i] = col_data.out["non_bonded"]["energy"].detach().cpu()
        new_forces[i] = col_data.out["non_bonded"]["forces"].detach().cpu()

        # Advance positions along the current forces plus noise so that
        # consistency is checked across an evolving set of neighbor lists,
        # rather than just the initial configuration.
        col_data.pos += 1e-5 * col_data.out["non_bonded"][
            "forces"
        ] + 1e-1 * torch.randn(col_data.pos.shape, device=col_data.pos.device)
        col_data.out = {}

    atol_energy = torch.mean(old_eners) * 1e-5
    torch.testing.assert_close(
        new_eners, old_eners, atol=atol_energy, rtol=1e-5
    )

    atol_forces = torch.mean(torch.abs(old_forces)) * 1e-3
    torch.testing.assert_close(
        new_forces, old_forces, atol=atol_forces, rtol=1e-3
    )
