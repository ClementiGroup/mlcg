"""Every implementation against the float64 brute-force reference, and the
reference against ASE."""

import contextlib

import numpy as np
import pytest
from ase.build import molecule
from ase.neighborlist import neighbor_list

from nl_testing import (
    MOLECULES,
    assert_same_edges,
    brute_force,
    run_nl,
    to_batch,
)


def test_reference_matches_ase(case, self_interaction):
    data = to_batch(case.frames)
    ref = brute_force(data, case.rc, self_interaction)
    rows, offset = [], 0
    for frame in case.frames:
        i, j, S = neighbor_list(
            "ijS", frame, case.rc, self_interaction=self_interaction
        )
        rows.append(np.column_stack([i + offset, j + offset, S]))
        offset += len(frame)
    rows = np.concatenate(rows)
    np.testing.assert_array_equal(rows[np.lexsort(rows.T[::-1])], ref.rows)


def test_matches_reference(nl_method, case, self_interaction, device, dtype):
    data = to_batch(case.frames, device, dtype)
    got = run_nl(nl_method, data, case.rc, self_interaction)
    assert_same_edges(got, brute_force(data, case.rc, self_interaction))


@pytest.mark.nl_methods(lambda method: method.batched)
def test_molecules_without_pbc_fields(nl_method, device, dtype):
    """Data with no ``pbc`` or ``cell`` (a CG molecule) is not periodic,
    also for the cell-list methods, which then build a box of their own."""
    data = to_batch([molecule(m) for m in MOLECULES], device, dtype)
    reference = brute_force(data, 4.0, self_interaction=False)
    del data.pbc, data.cell
    # expected: without pbc the cell-list methods bin the atoms in a
    # non-periodic box of 70 Å of their own, and warn about it
    expected_warning = (
        pytest.warns(UserWarning, match="box of length 70")
        if nl_method.periodic_only
        else contextlib.nullcontext()
    )
    with expected_warning:
        edges = run_nl(nl_method, data, 4.0)
    assert_same_edges(edges, reference)
