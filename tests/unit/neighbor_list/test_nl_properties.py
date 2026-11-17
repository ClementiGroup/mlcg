"""Properties every neighbour list must have, checked without a reference."""

import numpy as np
import pytest
import torch

from nl_testing import (
    AR2_ACROSS_BOUNDARY,
    AR_ISOLATED,
    run_nl,
    to_batch,
)


def _edges(method, frames, rc):
    return run_nl(method, to_batch(frames), rc)


def _structure_of(atom, frames):
    return np.searchsorted(np.cumsum([len(f) for f in frames]), atom, "right")


def test_within_cutoff(nl_method, case):
    edges = _edges(nl_method, case.frames, case.rc)
    assert np.all(edges.distances <= case.rc + 1e-6)


def test_symmetric(nl_method, case):
    """Every edge ``(i, j, S)`` comes with ``(j, i, -S)``."""
    rows = _edges(nl_method, case.frames, case.rc).rows
    mirrored = np.column_stack([rows[:, 1], rows[:, 0], -rows[:, 2:]])
    np.testing.assert_array_equal(mirrored[np.lexsort(mirrored.T[::-1])], rows)


def test_no_cross_structure_pairs(nl_method, case):
    rows = _edges(nl_method, case.frames, case.rc).rows
    np.testing.assert_array_equal(
        _structure_of(rows[:, 0], case.frames),
        _structure_of(rows[:, 1], case.frames),
    )


def test_no_shift_along_open_axes(nl_method, case):
    rows = _edges(nl_method, case.frames, case.rc).rows
    pbc = np.array([f.pbc for f in case.frames], dtype=bool)
    open_axes = ~pbc[_structure_of(rows[:, 0], case.frames)]
    assert np.all(rows[:, 2:][open_axes] == 0)


@pytest.mark.nl_methods(lambda method: method.mask)
def test_self_interaction_mask(nl_method, case, self_interaction):
    """The fourth output marks the returned self edges."""
    data = to_batch(case.frames)
    i, j, shifts, mask = nl_method.fn(
        data, case.rc, self_interaction=self_interaction
    )
    is_self = (i == j) & torch.all(shifts == 0, dim=1)
    torch.testing.assert_close(mask, ~is_self, rtol=0, atol=0)


def test_isolated_atom_has_no_neighbours(nl_method):
    assert len(_edges(nl_method, [AR_ISOLATED], 3.0).rows) == 0


def test_minimum_image_across_boundary(nl_method):
    """Two atoms 0.2 Å apart through the boundary of a 10 Å box."""
    edges = _edges(nl_method, [AR2_ACROSS_BOUNDARY], 3.0)
    assert len(edges.rows) > 0
    assert np.all(edges.distances < 1.0)
