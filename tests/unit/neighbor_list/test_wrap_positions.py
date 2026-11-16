import pytest
import torch

from mlcg.data import AtomicData
from nl_testing import DEVICES, TF32_CELLS


@pytest.mark.parametrize(
    "cell",
    [
        torch.eye(3) * 20.0,
        torch.tensor([[20.0, 0, 0], [5.0, 18.0, 0], [3.0, 4.0, 16.0]]),
    ],
)
def test_wrap_positions_keeps_inside_and_wraps_images(cell):
    """Rows of ``cell`` are the lattice vectors (pos = f @ cell): atoms in the
    cell stay put, atoms shifted by lattice vectors come back, and a
    non-periodic axis is left alone."""
    from mlcg.neighbor_list.torch_impl import wrap_positions

    gen = torch.Generator().manual_seed(0)
    frac = torch.rand(8, 3, generator=gen) * 0.9 + 0.05
    inside = frac @ cell
    shifts = torch.randint(-2, 3, (8, 3), generator=gen).float()
    for pbc, moved_axes in (([True] * 3, 3), ([True, True, False], 2)):
        shifts_used = shifts.clone()
        shifts_used[:, moved_axes:] = 0.0
        data = AtomicData.from_points(
            pos=(frac + shifts_used) @ cell,
            atom_types=torch.ones(8, dtype=torch.long),
            cell=cell,
            pbc=torch.tensor(pbc),
        )
        data.batch = torch.zeros(8, dtype=torch.long)
        wrapped = wrap_positions(data, "cpu").pos
        torch.testing.assert_close(wrapped, inside, atol=1e-4, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cell", TF32_CELLS)
def test_wrap_positions_ignores_tf32(cell, device, tf32_matmul):
    """Wrapped positions must not depend on the float32 matmul precision."""
    from mlcg.neighbor_list.torch_impl import wrap_positions

    gen = torch.Generator().manual_seed(0)
    frac = torch.rand(64, 3, generator=gen) + torch.randint(
        -2, 3, (64, 3), generator=gen
    )
    pos = frac @ cell

    def wrapped():
        data = AtomicData.from_points(
            pos=pos,
            atom_types=torch.ones(64, dtype=torch.long),
            cell=cell,
            pbc=torch.tensor([True, True, True]),
        ).to(device)
        data.batch = torch.zeros(64, dtype=torch.long, device=device)
        return wrap_positions(data, device).pos

    reference = wrapped()
    with tf32_matmul():
        result = wrapped()
    torch.testing.assert_close(result, reference, rtol=0, atol=0)
