import pytest
import torch
import torch.nn.functional as F
from torch.autograd import grad
from typing import List

from mlcg.nn.kernels import radius

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

compiled_radius = torch.compile(radius)


def torch_radius(
    pos: torch.Tensor,
    batch: torch.Tensor,
    cutoff: float,
    pbc: torch.Tensor = None,
    cell: torch.Tensor = None,
    avg_max_num_neigh: int = 32,
) -> List[torch.Tensor]:
    N = pos.shape[0]
    cutoff_sq = cutoff * cutoff
    apply_pbc = (cell is not None) and (pbc is not None)
    if apply_pbc:
        pbc = pbc.to(torch.bool)
        inv_cell = torch.linalg.inv(cell)

    d = pos.unsqueeze(0) - pos.unsqueeze(1)
    if apply_pbc:
        H = cell[batch].unsqueeze(0)
        H_inv = inv_cell[batch].unsqueeze(0)
        pbc_n = pbc[batch].unsqueeze(0)
        f = torch.einsum("ijc,ijcd->ijd", d, H_inv.expand(N, N, 3, 3))
        f = torch.where(pbc_n.expand(N, N, 3), f - torch.round(f), f)
        d = torch.einsum("ijc,ijcd->ijd", f, H.expand(N, N, 3, 3))

    dist_sq = (d * d).sum(dim=-1)
    same_mol = batch.unsqueeze(0) == batch.unsqueeze(1)
    same_mol = same_mol & ~torch.eye(N, dtype=torch.bool, device=pos.device)

    valid = same_mol & (dist_sq < cutoff_sq)
    max_out = pos.shape[0] * avg_max_num_neigh
    E_half = int(valid.triu().sum().item())
    if E_half > max_out:
        raise RuntimeError(
            f"radius_edges: output buffer overflow: {E_half} edges found but "
            f"available are N_atoms*avg_max_num_neigh={max_out}. Increase avg_max_num_neigh."
        )

    src, dst = valid.nonzero(as_tuple=True)
    distances = torch.sqrt(dist_sq[src, dst])
    edge_index = torch.stack([src, dst], dim=0)

    return [edge_index, distances, d[dst, src]]


def sample_data(
    N_atoms,
    N_mol,
    cutoff=4.0,
    spread_pos=10,
    cell_size=8.0,
    device=None,
    periodic=[False, False, False],
    requires_grad_pos: bool = False,
):
    pos = torch.randn(N_atoms, 3, device=device) * spread_pos

    pos_r = pos.clone().requires_grad_(requires_grad_pos)
    pos_k = pos.clone().requires_grad_(requires_grad_pos)

    n_mol = N_mol
    batch = (
        torch.arange(n_mol, device=device)
        .repeat_interleave(int(pos.shape[0] / n_mol))
        .long()
    )

    max_neigh = int(pos.shape[0] / n_mol)

    if not any(periodic):
        cell = None
        pbc = None
    else:
        cell = torch.eye(3, device=device).repeat(n_mol, 1, 1) * cell_size
        pbc = torch.tensor(periodic, device=device).repeat(n_mol, 1)

    return pos_r, pos_k, batch, cutoff, pbc, cell, max_neigh


def sort_edges(edge_index, *tensors):
    stride = edge_index.max().item() + 1
    key = edge_index[0] * stride + edge_index[1]
    order = torch.argsort(key)
    return (edge_index[:, order],) + tuple(t[order] for t in tensors)


@pytest.mark.parametrize("model", [radius, compiled_radius])
@pytest.mark.parametrize(
    "N_atoms, N_mol",
    [
        (100, 5),
        (52, 1),
        (1002, 3),
    ],
)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "periodic",
    [
        [False, False, False],
        [True, True, False],
        [True, True, True],
    ],
)
@pytest.mark.parametrize("test_grad", [True])
def test_radius_kernel(model, N_atoms, N_mol, device, periodic, test_grad):

    pos_r, pos_k, batch, cutoff, pbc, cell, max_neigh = sample_data(
        N_atoms,
        N_mol,
        device=device,
        periodic=periodic,
        requires_grad_pos=test_grad,
    )

    args_r = (pos_r, batch, cutoff, pbc, cell, max_neigh)
    args_k = (pos_k, batch, cutoff, pbc, cell, max_neigh)

    edge_r, dist_r, disp_r = sort_edges(*torch_radius(*args_r))
    edge_k, dist_k, disp_k = sort_edges(*model(*args_k))

    assert torch.equal(edge_k, edge_r)
    torch.testing.assert_close(dist_k, dist_r)
    torch.testing.assert_close(disp_k, disp_r)

    if test_grad:
        ## Silly operation to test autograd
        out_r = 0.01 * (dist_r**2 + (disp_r**2).sum(dim=1)).sum()
        grad_r = grad(out_r, pos_r, create_graph=True)[0]
        double_grad_r = grad(grad_r.sum(), pos_r)[0]

        out_k = 0.01 * (dist_k**2 + (disp_k**2).sum(dim=1)).sum()
        grad_k = grad(out_k, pos_k, create_graph=True)[0]
        double_grad_k = grad(grad_k.sum(), pos_k)[0]

        torch.testing.assert_close(grad_k, grad_r)
        torch.testing.assert_close(double_grad_k, double_grad_r)


@pytest.mark.parametrize(
    "N_atoms, N_mol",
    [
        (100, 5),
        (52, 1),
    ],
)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "periodic",
    [
        [False, False, False],
        [True, True, False],
        [True, True, True],
    ],
)
def test_minimum_avg_max_num_neigh(N_atoms, N_mol, device, periodic):
    with pytest.raises(RuntimeError):
        _, pos_k, batch, cutoff, pbc, cell, _ = sample_data(
            N_atoms,
            N_mol,
            device=device,
            periodic=periodic,
            cutoff=4.0,
            spread_pos=2.0,
            requires_grad_pos=False,
        )
        radius(pos_k, batch, cutoff, pbc, cell, 1)
