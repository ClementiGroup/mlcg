"""Neighbour lists must not depend on the float32 matmul precision."""

import ase
import numpy as np
import pytest
import torch

from nl_testing import TF32_CELLS, to_batch


@pytest.mark.parametrize("cell", TF32_CELLS)
def test_pbc_neighbor_list_ignores_tf32(nl_method, cell, device, tf32_matmul):
    """The periodic image shifts must not depend on the float32 matmul
    precision: under TF32 the cell would be rounded and every edge across
    the box boundary would get a wrong length."""
    gen = torch.Generator().manual_seed(0)
    frac = torch.rand(64, 3, generator=gen)
    atoms = ase.Atoms(
        "Ar64", positions=(frac @ cell).numpy(), cell=cell.numpy(), pbc=True
    )
    data = to_batch([atoms], device, torch.float32)

    reference = nl_method.fn(data, 12.0, self_interaction=False)
    with tf32_matmul():
        result = nl_method.fn(data, 12.0, self_interaction=False)
    for a, b in zip(_sorted_by_edge(result), _sorted_by_edge(reference)):
        np.testing.assert_array_equal(a, b)


def _sorted_by_edge(output):
    """The outputs with the edges sorted by ``(i, j, shift)``: some
    implementations return them in no fixed order."""
    i, j, shifts, mask = (
        None if t is None else t.cpu().numpy() for t in output
    )
    order = np.lexsort((*shifts.T[::-1], j, i))
    return [None if t is None else t[order] for t in (i, j, shifts, mask)]
