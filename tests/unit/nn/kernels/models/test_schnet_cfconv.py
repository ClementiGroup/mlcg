import pytest
import torch
from torch.autograd import grad

from mlcg.nn import MLP, CosineCutoff, GaussianBasis
from mlcg.nn.kernels.csr import build_csr_representation_from_edges
from mlcg.nn.schnet import CFConv as StandardCFConv
from mlcg.nn.flash_schnet import CFConv as FlashCFConv
from mlcg.neighbor_list.neighbor_list import atomic_data2neighbor_list
from mlcg.geometry.internal_coordinates import compute_distances

from mlcg.mol_utils import MolDatabase

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
CUTOFF_UPPER = 4.0

# Fixed test graph
SENDERS = torch.tensor(
    [0, 0, 1, 3, 2, 1, 3, 4, 5, 7, 6, 8, 6, 6, 8, 9, 0, 2, 4, 6]
)
RECEIVERS = torch.tensor(
    [2, 1, 3, 4, 0, 0, 1, 3, 6, 6, 8, 9, 5, 7, 6, 8, 1, 3, 5, 7]
)

database = MolDatabase()

torch.set_float32_matmul_precision("high")


# Graph generation strategies
@pytest.fixture(params=["fixed", "molecules"])
def graph_type(request):
    return request.param


def generate_graph(graph_type, device):
    """Generate different types of test graphs."""
    torch.manual_seed(420)
    if graph_type == "fixed":
        # Use the fixed test graph
        senders = SENDERS.to(device)
        receivers = RECEIVERS.to(device)
        edge_index = torch.stack([senders, receivers])
        distances = (
            torch.rand(senders.shape[0], device=device) * CUTOFF_UPPER
        )  # 0.1 to 1.0
        x = torch.randn(NUM_NODES, FEATURES, device=device)

    elif graph_type == "molecules":
        database.collated_dat = database.collated_data.to(device)
        neighbor_list = atomic_data2neighbor_list(
            database.collated_data,
            CUTOFF_UPPER,
            self_interaction=False,
            max_num_neighbors=1000,
            nls_distance_method="torch",
        )
        edge_index = neighbor_list["index_mapping"]
        distances = compute_distances(
            database.collated_data.pos, edge_index, neighbor_list["cell_shifts"]
        )
        x = torch.randn(
            database.collated_data.pos.shape[0], FEATURES, device=device
        )

    return edge_index, distances, x


def zero_grad(model: torch.nn.Module):
    """Reset grad in provided model"""
    for param in model.parameters():
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()


def create_models(device):
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

    flash_cfconv.load_state_dict(standard_cfconv.state_dict())

    rbf = GaussianBasis(cutoff=cutoff, num_rbf=FEATURES, trainable=False).to(
        device
    )

    return standard_cfconv, flash_cfconv, rbf


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("graph_type", ["fixed", "molecules"])
def test_standard_vs_flash_cfconv(device, graph_type):
    """Test standard CFConv vs flash CFConv forward consistency."""
    edge_index, distances, x = generate_graph(graph_type, device)
    activation = torch.nn.Tanh()

    standard_cfconv, flash_cfconv, rbf = create_models(device)
    csr_data = build_csr_representation_from_edges(edge_index, x.shape[0])

    x.requires_grad_(True)
    distances.requires_grad_(True)
    edge_attr = rbf(distances)
    zero_grad(standard_cfconv)

    out_torch = activation(standard_cfconv(x, edge_index, distances, edge_attr))
    grad_torch = grad(
        out_torch.sum(),
        [x, distances, edge_attr] + list(standard_cfconv.parameters()),
        create_graph=True,
    )
    double_grad_torch = []
    for gt in grad_torch:
        go = grad(
            gt.sum(),
            [x, edge_attr, distances] + list(standard_cfconv.parameters()),
            create_graph=True,
        )
        double_grad_torch.append(go)

    x = x.detach().requires_grad_(True)
    distances = distances.detach().requires_grad_(True)
    edge_attr = rbf(distances)
    zero_grad(flash_cfconv)

    out_triton = activation(
        flash_cfconv(
            x,
            edge_index,
            distances,
            edge_attr,
            csr_data,
        )
    )
    grad_triton = grad(
        out_triton.sum(),
        [x, distances, edge_attr] + list(flash_cfconv.parameters()),
        create_graph=True,
    )
    double_grad_triton = []
    for gt in grad_triton:
        go = grad(
            gt.sum(),
            [x, edge_attr, distances] + list(flash_cfconv.parameters()),
            create_graph=True,
        )
        double_grad_triton.append(go)

    torch.testing.assert_close(
        out_torch,
        out_triton,
        msg=lambda msg: f"Failed forward check \n\n{msg}",
    )
    input_names = ["x", "distances", "edge_attr"] + [
        f"param_{name}" for name, _ in standard_cfconv.named_parameters()
    ]
    for gto, gtr, name in zip(grad_torch, grad_triton, input_names):
        torch.testing.assert_close(
            gto,
            gtr,
            msg=lambda msg: f"Failed first grad check wrt {name}\n\n{msg}",
        )

    outer_inputs = inner_inputs = input_names

    for gto, gtr, outer_name in zip(
        double_grad_torch, double_grad_triton, outer_inputs
    ):
        for g2to, g2tr, inner_name in zip(gto, gtr, inner_inputs):
            torch.testing.assert_close(
                g2to,
                g2tr,
                atol=5e-5,  # encrease tolerance: also plain torch has error in that bound
                rtol=5e-5,
                msg=lambda msg: f"Failed second grad check wrt inner {inner_name}, outer {outer_name}\n\n{msg}",
            )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Compiled kernel tests require CUDA"
)
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("graph_type", ["fixed", "molecules"])
def test_compiled_standard_vs_flash_cfconv(device, graph_type):
    """Test compiled fused CFConv forward pass."""
    edge_index, distances, x = generate_graph(graph_type, device)
    activation = torch.nn.Tanh()

    standard_cfconv, _flash_cfconv, rbf = create_models(device)
    flash_cfconv = torch.compile(_flash_cfconv)
    csr_data = build_csr_representation_from_edges(edge_index, x.shape[0])

    x.requires_grad_(True)
    distances.requires_grad_(True)
    edge_attr = rbf(distances)
    zero_grad(standard_cfconv)

    out_torch = activation(standard_cfconv(x, edge_index, distances, edge_attr))
    grad_torch = grad(
        out_torch.sum(),
        [x, distances, edge_attr] + list(standard_cfconv.parameters()),
    )

    x = x.detach().requires_grad_(True)
    distances = distances.detach().requires_grad_(True)
    edge_attr = rbf(distances)
    zero_grad(flash_cfconv)

    out_triton = activation(
        flash_cfconv(
            x,
            edge_index,
            distances,
            edge_attr,
            csr_data,
        )
    )
    grad_triton = grad(
        out_triton.sum(),
        [x, distances, edge_attr] + list(flash_cfconv.parameters()),
    )

    torch.testing.assert_close(
        out_torch, out_triton, msg="Failed forward check"
    )
    input_names = ["x", "distances", "edge_attr"] + [
        f"param_{name}" for name, _ in standard_cfconv.named_parameters()
    ]
    for gto, gtr, name in zip(grad_torch, grad_triton, input_names):
        torch.testing.assert_close(
            gto, gtr, msg=f"Failed first grad check wrt {name}"
        )
