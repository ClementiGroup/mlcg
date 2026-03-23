import pytest
import torch
from torch.autograd import grad
from torch_geometric.utils import scatter

from mlcg.nn.kernels.models.schnet import fused_cfconv
from mlcg.nn.kernels.csr import build_csr_representation_from_edges
from mlcg.nn.schnet import CFConv as StandardCFConv
from mlcg.nn.flash_schnet import CFConv as FlashCFConv
from mlcg.nn.mlp import MLP
from mlcg.nn.cutoff import CosineCutoff

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

# Clear triton cache
import os, shutil

shutil.rmtree(os.path.expanduser("~/.triton/cache"), ignore_errors=True)

# Clear torch.compile cache
torch._dynamo.reset()

# Test data
NUM_NODES = 10
NUM_EDGES = 20
FEATURES = 32
CUTOFF_UPPER = 1.0

# Fixed test graph
SENDERS = torch.tensor(
    [0, 0, 1, 3, 2, 1, 3, 4, 5, 7, 6, 8, 6, 6, 8, 9, 0, 2, 4, 6]
)
RECEIVERS = torch.tensor(
    [2, 1, 3, 4, 0, 0, 1, 3, 6, 6, 8, 9, 5, 7, 6, 8, 1, 3, 5, 7]
)


# Graph generation strategies
@pytest.fixture(params=["fixed", "random", "full_coverage"])
def graph_type(request):
    return request.param


def generate_graph(num_nodes, num_edges, graph_type, device):
    """Generate different types of test graphs."""
    torch.manual_seed(42)

    if graph_type == "fixed":
        # Use the fixed test graph
        senders = SENDERS.to(device)
        receivers = RECEIVERS.to(device)
    elif graph_type == "random":
        # Random edges
        senders = torch.randint(0, num_nodes, (num_edges,), device=device)
        receivers = torch.randint(0, num_nodes, (num_edges,), device=device)
    elif graph_type == "full_coverage":
        # Ensure all nodes appear at least once (like your notebook function)
        senders = sample_with_full_coverage(num_edges, num_nodes, device)
        receivers = sample_with_full_coverage(num_edges, num_nodes, device)

    edge_index = torch.stack([senders, receivers])
    return edge_index


def sample_with_full_coverage(N, M, device=None):
    """Generate indices that include all values from 0 to M-1 at least once.

    Args:
        N: Total number of samples
        M: Number of unique values (0 to M-1) that must appear at least once
        device: Device for the tensor

    Returns:
        Tensor of shape (N,) containing all values 0 to M-1 at least once
    """
    assert M <= N, f"Need at least {M} slots to cover all {M} integers"
    base = torch.arange(M, device=device)
    remaining = torch.randint(0, M, (N - M,), device=device)
    result = torch.cat([base, remaining])
    perm = torch.randperm(N, device=device)
    result = result[perm]
    return result


def create_test_data(device, graph_type="fixed"):
    """Create consistent test data with different graph types."""
    torch.manual_seed(42)

    edge_index = generate_graph(NUM_NODES, NUM_EDGES, graph_type, device)
    distances = torch.rand(NUM_EDGES, device=device) * 0.9 + 0.1  # 0.1 to 1.0
    x = torch.randn(NUM_NODES, FEATURES, device=device)
    filter_out = torch.randn(NUM_EDGES, FEATURES, device=device)

    return x, filter_out, distances, edge_index


def create_cfconv_layers(device):
    """Create standard and flash CFConv layers with same weights."""
    cutoff = CosineCutoff(cutoff_upper=CUTOFF_UPPER)
    filter_network = MLP(
        layer_widths=[FEATURES, FEATURES, FEATURES],
        activation_func=torch.nn.Tanh(),
        last_bias=False,
    ).to(device)

    standard_cfconv = StandardCFConv(
        filter_network=filter_network,
        cutoff=cutoff,
        in_channels=FEATURES,
        out_channels=FEATURES,
        num_filters=FEATURES,
        aggr="add",
    ).to(device)

    flash_cfconv = FlashCFConv(
        filter_network=filter_network,
        cutoff=cutoff,
        in_channels=FEATURES,
        out_channels=FEATURES,
        num_filters=FEATURES,
        aggr="add",
        use_triton=True,
    ).to(device)

    # Copy weights to ensure same initialization
    flash_cfconv.load_state_dict(standard_cfconv.state_dict())

    return standard_cfconv, flash_cfconv


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("graph_type", ["fixed", "random", "full_coverage"])
def test_standard_vs_flash_cfconv_forward(device, graph_type):
    """Test standard CFConv vs flash CFConv forward consistency."""
    x, _, distances, edge_index = create_test_data(device, graph_type)
    edge_attr = torch.randn(NUM_EDGES, FEATURES, device=device)

    standard_cfconv, flash_cfconv = create_cfconv_layers(device)
    csr_data = build_csr_representation_from_edges(edge_index, NUM_NODES)

    with torch.no_grad():
        y_standard = standard_cfconv(x, edge_index, distances, edge_attr)
        y_flash = flash_cfconv(x, edge_index, distances, edge_attr, csr_data)

    torch.testing.assert_close(y_standard, y_flash)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("graph_type", ["fixed", "random", "full_coverage"])
def test_standard_vs_flash_cfconv_gradients(device, graph_type):
    """Test standard CFConv vs flash CFConv gradient consistency."""
    x, _, distances, edge_index = create_test_data(device, graph_type)
    edge_attr = torch.randn(NUM_EDGES, FEATURES, device=device)

    standard_cfconv, flash_cfconv = create_cfconv_layers(device)
    csr_data = build_csr_representation_from_edges(edge_index, NUM_NODES)

    x.requires_grad_(True)
    distances.requires_grad_(True)
    edge_attr.requires_grad_(True)

    # Standard gradients
    y_standard = standard_cfconv(x, edge_index, distances, edge_attr)
    grad_standard = grad(y_standard.sum(), [x, distances, edge_attr]+list(standard_cfconv.parameters()))

    x.grad = None
    distances.grad = None
    edge_attr.grad = None

    # Flash gradients
    y_flash = flash_cfconv(x, edge_index, distances, edge_attr, csr_data)
    grad_flash = grad(y_flash.sum(), [x, distances, edge_attr]+list(flash_cfconv.parameters()))

    for g_std, g_flash in zip(grad_standard, grad_flash):
        torch.testing.assert_close(g_std, g_flash)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("graph_type", ["fixed", "random", "full_coverage"])
def test_standard_vs_flash_cfconv_double_triple_gradients(device, graph_type):
    """Test standard CFConv vs flash CFConv gradient consistency."""
    x, _, distances, edge_index = create_test_data(device, graph_type)
    edge_attr = torch.randn(NUM_EDGES, FEATURES, device=device)

    standard_cfconv, flash_cfconv = create_cfconv_layers(device)
    csr_data = build_csr_representation_from_edges(edge_index, NUM_NODES)

    x.requires_grad_(True)
    distances.requires_grad_(True)
    edge_attr.requires_grad_(True)

    # standard_cfconv.train()
    # Standard gradients
    y_standard = standard_cfconv(x, edge_index, distances, edge_attr)
    forces_standard = -grad(y_standard.sum(), [distances], create_graph=True)[0]
    grad_standard = grad(forces_standard.sum(), [x, distances, edge_attr])
    # grad_standard = grad(forces_standard.sum(), [x, distances, edge_attr]+list(standard_cfconv.parameters()))

    x.grad = None
    distances.grad = None
    edge_attr.grad = None

    # flash_cfconv.train()
    # Flash gradients
    y_flash = flash_cfconv(x, edge_index, distances, edge_attr, csr_data)
    forces_flash = -grad(y_flash.sum(), [distances], create_graph=True)[0]
    grad_flash = grad(forces_flash.sum(), [x, distances, edge_attr])
    # grad_flash = grad(forces_flash.sum(), [x, distances, edge_attr]+list(flash_cfconv.parameters()))

    index = 0
    for g_std, g_flash in zip(grad_standard, grad_flash):
        index += 1
        print(index)
        torch.testing.assert_close(g_std, g_flash)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Compiled kernel tests require CUDA"
)
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("graph_type", ["fixed", "random", "full_coverage"])
def test_fused_cfconv_compiled_forward(device, graph_type):
    """Test compiled fused CFConv forward pass."""
    x, _, distances, edge_index = create_test_data(device, graph_type)
    edge_attr = torch.randn(NUM_EDGES, FEATURES, device=device)

    standard_cfconv, flash_cfconv = create_cfconv_layers(device)
    csr_data = build_csr_representation_from_edges(edge_index, NUM_NODES)

    compiled_standard_cfconv = torch.compile(standard_cfconv)
    compiled_flash_cfconv = torch.compile(flash_cfconv)

    with torch.no_grad():
        y_standard = standard_cfconv(x, edge_index, distances, edge_attr)
        y_standard_compiled = compiled_standard_cfconv(
            x, edge_index, distances, edge_attr
        )
        y_flash_compiled = compiled_flash_cfconv(
            x, edge_index, distances, edge_attr, csr_data
        )

    torch.testing.assert_close(y_standard, y_standard_compiled)
    torch.testing.assert_close(y_standard, y_flash_compiled)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Compiled kernel tests require CUDA"
)
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("graph_type", ["fixed", "random", "full_coverage"])
def test_fused_cfconv_compiled_gradients(device, graph_type):
    """Test compiled fused CFConv gradients."""
    x, _, distances, edge_index = create_test_data(device, graph_type)
    edge_attr = torch.randn(NUM_EDGES, FEATURES, device=device)

    standard_cfconv, flash_cfconv = create_cfconv_layers(device)
    csr_data = build_csr_representation_from_edges(edge_index, NUM_NODES)

    compiled_standard_cfconv = torch.compile(standard_cfconv)
    compiled_flash_cfconv = torch.compile(flash_cfconv)

    x.requires_grad_(True)
    distances.requires_grad_(True)
    edge_attr.requires_grad_(True)

    y_standard = standard_cfconv(x, edge_index, distances, edge_attr)
    y_standard_grad = grad(y_standard.sum(), [x, distances, edge_attr])

    x.grad = None
    distances.grad = None
    edge_attr.grad = None

    y_standard_compiled = compiled_standard_cfconv(
        x, edge_index, distances, edge_attr
    )
    y_standard_compiled_grad = grad(
        y_standard_compiled.sum(), [x, distances, edge_attr]
    )

    x.grad = None
    distances.grad = None
    edge_attr.grad = None

    y_flash_compiled = compiled_flash_cfconv(
        x, edge_index, distances, edge_attr, csr_data
    )
    y_flash_compiled_grad = grad(
        y_flash_compiled.sum(), [x, distances, edge_attr]
    )

    for g_std, g_std_comp, g_flash_comp in zip(
        y_standard_grad, y_standard_compiled_grad, y_flash_compiled_grad
    ):
        torch.testing.assert_close(g_std, g_std_comp)
        torch.testing.assert_close(g_std, g_flash_comp)
