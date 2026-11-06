import pytest
import torch

from mlcg.nn.kernels import RadiusCSR
from mlcg.nn.kernels.pbc import periodic_cells

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def torch_radius(
    pos: torch.Tensor,
    batch: torch.Tensor,
    cutoff: float,
    pbc: torch.Tensor = None,
    cell: torch.Tensor = None,
) -> torch.Tensor:
    """``[2, E]`` pairs within ``cutoff`` of each other (minimum image with
    a cell), by brute force."""
    N = pos.shape[0]
    d = pos.unsqueeze(0) - pos.unsqueeze(1)
    if cell is not None:
        H = cell[batch].unsqueeze(0).expand(N, N, 3, 3)
        H_inv = torch.linalg.inv(cell)[batch].unsqueeze(0).expand(N, N, 3, 3)
        f = torch.einsum("ijc,ijcd->ijd", d, H_inv)
        pbc_n = pbc.to(torch.bool)[batch].unsqueeze(0).expand(N, N, 3)
        f = torch.where(pbc_n, f - torch.round(f), f)
        d = torch.einsum("ijc,ijcd->ijd", f, H)
    same_mol = batch.unsqueeze(0) == batch.unsqueeze(1)
    same_mol = same_mol & ~torch.eye(N, dtype=torch.bool, device=pos.device)
    valid = same_mol & ((d * d).sum(dim=-1) < cutoff * cutoff)
    return torch.stack(valid.nonzero(as_tuple=True))


def sample_data(
    N_atoms,
    N_mol,
    cutoff=4.0,
    spread_pos=10,
    cell_size=8.0,
    device=None,
    periodic=[False, False, False],
):
    pos = torch.randn(N_atoms, 3, device=device) * spread_pos
    batch = (
        torch.arange(N_mol, device=device)
        .repeat_interleave(N_atoms // N_mol)
        .long()
    )
    if not any(periodic):
        cell, pbc = None, None
    else:
        cell = torch.eye(3, device=device).repeat(N_mol, 1, 1) * cell_size
        pbc = torch.tensor(periodic, device=device).repeat(N_mol, 1)
    return pos, batch, cutoff, pbc, cell


def neighbor_sets(row_ptr, col):
    row_ptr, col = row_ptr.tolist(), col.tolist()
    return [
        sorted(col[row_ptr[i] : row_ptr[i + 1]])
        for i in range(len(row_ptr) - 1)
    ]


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
def test_radius_csr_matches_brute_force(N_atoms, N_mol, device, periodic):
    pos, batch, cutoff, pbc, cell = sample_data(
        N_atoms, N_mol, device=device, periodic=periodic
    )
    ptr = torch.cat(
        [batch.new_zeros(1), torch.bincount(batch, minlength=N_mol).cumsum(0)]
    )
    row_ptr, col, valid = RadiusCSR(cutoff)(
        pos, batch, ptr, *periodic_cells(cell, pbc, N_mol, device)
    )
    expected = [[] for _ in range(N_atoms)]
    for src, dst in torch_radius(pos, batch, cutoff, pbc, cell).t().tolist():
        expected[dst].append(src)
    assert neighbor_sets(row_ptr, col) == [sorted(e) for e in expected]
    assert valid.item() == 1.0
