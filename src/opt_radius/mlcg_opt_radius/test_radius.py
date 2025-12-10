import pytest
import math
from itertools import product
import scipy.spatial
import torch
from typing import NamedTuple
from .utils import to_set, enforce_mnn, remove_loop, reference_index
from torch.autograd import gradcheck, gradgradcheck

from torch_cluster import radius_graph as rgo

skip_test = False
try:
    from .radius import radius_distance as rgm
except RuntimeError:
    skip_test = True

if skip_test or not torch.cuda.is_available():
    pytest.skip(
        "Cuda device is required for this test. Skipping ...",
        allow_module_level=True,
    )


######################################################################

INT_DTYPE = torch.long
FLOATING_DTYPES = [torch.double]
DEVICES = [torch.device("cuda:0")]

X = [0, 10, 100]
DIM = [1, 2, 3]
X_RANGE = [(-1.0, 1.0), (-10.0, -20.0)]
R = [0.0, 1.0, 10.0]
BATCH_SIZES = [(100,), (50, 100), (10, 20, 100)]
MAX_NUM_NEIGHBORS = [100]
LOOP = [True, False]

TOL = 1e-6

######################################################################


@pytest.mark.parametrize(
    "x_c,\
     dim,\
     x_range,\
     r,\
     batch_size,\
     max_num_neighbors,\
     loop,\
     fdtype,\
     device",
    product(
        X,
        DIM,
        X_RANGE,
        R,
        BATCH_SIZES,
        MAX_NUM_NEIGHBORS,
        LOOP,
        FLOATING_DTYPES,
        DEVICES,
    ),
)
def test_radius(
    x_c, dim, x_range, r, batch_size, max_num_neighbors, loop, fdtype, device
):
    (x_min, x_max) = x_range
    x = (x_max - x_min) * torch.rand(
        (x_c, dim), dtype=fdtype, device=device
    ) + x_min
    batch = []
    p = 0
    for i, b in enumerate(batch_size):
        n = math.ceil((x_c * (b / 100))) - p
        batch.append(torch.ones(n, dtype=int) * i)
        p += n
    batch = torch.cat(batch).to(device)

    o_real_i = rgo(
        x, r, batch, loop, max_num_neighbors, flow="target_to_source"
    )
    o_mine_d, o_mine_i = rgm(x, r, batch, loop, max_num_neighbors)

    # comparing index
    if torch.numel(o_real_i) == 0:
        assert torch.numel(o_mine_i) == 0
    else:
        if not loop:
            o_real_i = remove_loop(o_real_i)
        o_real_i = enforce_mnn(o_real_i, max_num_neighbors)
        print(o_mine_i)
        assert to_set(o_real_i) == to_set(o_mine_i)

        o_real_i = torch.tensor(sorted(list(to_set(o_real_i)))).t()
        o_mine_i = o_mine_i.to("cuda")
        o_real_i = o_real_i.to("cuda")

        # compare distance
        o_real_d = torch.linalg.norm(
            x[o_real_i[0, :], :] - x[o_real_i[1, :], :], axis=-1
        )
        assert torch.all((o_real_d - o_mine_d) < TOL)


######################################################################
@pytest.mark.parametrize(
    "x_c,\
     dim,\
     x_range,\
     r,\
     batch_size,\
     fdtype,\
     device",
    product(
        X[1:],  # skip graphs with 0 nodes
        DIM,
        X_RANGE,
        R,
        BATCH_SIZES,
        FLOATING_DTYPES,
        DEVICES,
    ),
)
def test_with_gradcheck(x_c, dim, x_range, r, batch_size, fdtype, device):
    (x_min, x_max) = x_range
    x = (x_max - x_min) * torch.rand(
        (x_c, dim), dtype=fdtype, device=device
    ) + x_min

    x.requires_grad_(True)
    gradcheck_result = gradcheck(
        lambda x: rgm(x, r)[0],  # Check gradients for `distance`
        x,
        eps=1e-7,
        atol=1e-6,
        nondet_tol=1e-7,
    )
    assert gradcheck_result
    if x_c < 20:
        # otherwise skip the following for speed
        gradgradcheck_result = gradgradcheck(
            lambda x: rgm(x, r)[0],  # Check gradients for `distance`
            x,
            eps=1e-7,
            atol=1e-6,
            nondet_tol=1e-7,
        )
        assert gradgradcheck_result
