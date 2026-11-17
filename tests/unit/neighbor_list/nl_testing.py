"""Shared pieces of the neighbour-list tests (fixtures in ``conftest.py``).

- ``NL_METHODS``: one record per implementation and what it supports;
  ``supports`` decides which combinations ``conftest.py`` generates.
- ``CASES``: the structure catalogue; each case is one batch of
  ``ase.Atoms`` and one cutoff. The id starts with the group (``molecule``,
  ``orthorhombic``, ``triclinic``, ``mixed_pbc``, ``batched``,
  ``unwrapped``), so ``-k triclinic``
  selects a group.
- ``run_nl`` / ``brute_force``: the edges of a method and of the reference
  as sorted ``(i, j, S)`` rows, ``S`` the integer lattice image of ``j``,
  with the distances.
"""

import itertools
from dataclasses import dataclass
from typing import Callable, List, NamedTuple, Tuple

import ase
import numpy as np
import torch
from ase.build import bulk, molecule
from torch_geometric.data import Batch

from mlcg.neighbor_list.ase_impl import ase_neighbor_list
from mlcg.neighbor_list.torch_impl import torch_neighbor_list
from mlcg.neighbor_list.utils import ase2data

try:
    from mlcg.neighbor_list import nvalchemi_impl
except ImportError:
    nvalchemi_impl = None

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
DTYPES = {"float64": torch.float64, "float32": torch.float32}
# box lengths with more than 10 mantissa bits, so that TF32 would round them
TF32_CELLS = [
    torch.diag(torch.tensor([71.7, 70.3, 75.9])),
    torch.tensor([[71.7, 0.0, 0.0], [5.3, 70.3, 0.0], [3.1, 4.7, 75.9]]),
]


@dataclass(frozen=True)
class NLMethod:
    name: str
    fn: Callable
    integer_shifts: bool = False
    self_interaction: bool = True
    periodic_only: bool = False
    batched: bool = True
    devices: Tuple[str, ...] = tuple(DEVICES)
    mask: bool = True


def _nvalchemi(name, attr, **kwargs) -> NLMethod:
    """``fn`` is None when nvalchemi-toolkit-ops is not installed."""
    fn = None if nvalchemi_impl is None else getattr(nvalchemi_impl, attr)
    return NLMethod(name, fn, self_interaction=False, mask=False, **kwargs)


NL_METHODS = [
    NLMethod("torch", torch_neighbor_list),
    NLMethod(
        "ase", ase_neighbor_list, batched=False, devices=("cpu",), mask=False
    ),
    _nvalchemi("nvalchemi_naive", "nvalchemi_naive_neighbor_list"),
    _nvalchemi(
        "nvalchemi_cell", "nvalchemi_cell_neighbor_list", periodic_only=True
    ),
    _nvalchemi(
        "nvalchemi_raw",
        "nvalchemi_cell_neighbor_list_raw",
        periodic_only=True,
        integer_shifts=True,
    ),
]


@dataclass(frozen=True)
class Case:
    id: str
    frames: List[ase.Atoms]
    rc: float

    @property
    def periodic(self) -> bool:
        return all(f.pbc.any() for f in self.frames)


def random_box(cell, pbc=True, n=24, seed=0, outside=False):
    """``n`` atoms at seeded random positions inside ``cell``, or up to two
    cells away along the periodic axes if ``outside``."""
    rng = np.random.default_rng(seed)
    frac = rng.random((n, 3))
    if outside:
        frac += rng.integers(-2, 3, (n, 3)) * np.broadcast_to(pbc, 3)
    cell = np.asarray(cell, dtype=float)
    return ase.Atoms(f"Ar{n}", positions=frac @ cell, cell=cell, pbc=pbc)


ORTHORHOMBIC = np.diag([9.0, 8.0, 7.0])
SHEARED = [[8.0, 0.0, 0.0], [5.0, 7.0, 0.0], [3.0, 4.0, 9.0]]
STRONGLY_SHEARED = [[8.0, 0.0, 0.0], [7.5, 3.0, 0.0], [6.0, 2.5, 4.0]]
SMALL_TRICLINIC = [[5.0, 0.0, 0.0], [2.0, 5.0, 0.0], [1.0, 1.0, 5.0]]
OPEN_AXIS_PATTERNS = {
    "x_open": [False, True, True],
    "y_open": [True, False, True],
    "z_open": [True, True, False],
}
MOLECULES = ("CH3CH2NH2", "H2O", "methylenecyclopropane")
AG_PRIMITIVE = ase.Atoms(
    "Ag", cell=[(0, 2.0, 2.0), (2.0, 0, 2.0), (2.0, 2.0, 0)], pbc=True
)
AR_ISOLATED = ase.Atoms(
    "Ar", positions=[[0.1, 0.1, 0.1]], cell=[10.0, 10.0, 10.0], pbc=True
)
AR2_ACROSS_BOUNDARY = ase.Atoms(
    "Ar2",
    positions=[[0.1, 0.1, 0.1], [9.9, 0.1, 0.1]],
    cell=[10.0, 10.0, 10.0],
    pbc=True,
)
SLAB = ase.Atoms(
    "C4",
    positions=[[0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 5.0]],
    cell=[5.0, 5.0, 10.0],
    pbc=[True, True, False],
)


def _cases():
    for name, rc in itertools.product(MOLECULES, (2, 4, 6)):
        yield Case(f"molecule-{name}-rc{rc}", [molecule(name)], rc)
    yield Case("orthorhombic-Ar_isolated-rc3", [AR_ISOLATED], 3)
    yield Case("orthorhombic-Ar2_across_boundary-rc3", [AR2_ACROSS_BOUNDARY], 3)
    for rc in (2, 4, 6):
        yield Case(f"triclinic-Ag_primitive-rc{rc}", [AG_PRIMITIVE], rc)
    for rc in (2, 3, 4, 6):
        yield Case(f"triclinic-Cu_primitive-rc{rc}", [bulk("Cu", a=3.6)], rc)
    yield Case("mixed_pbc-slab-rc2", [SLAB], 2)
    yield Case("orthorhombic-Cu_cubic-rc4", [bulk("Cu", cubic=True, a=3.6)], 4)
    yield Case("orthorhombic-random-rc4", [random_box(ORTHORHOMBIC)], 4)
    yield Case("triclinic-Mg_hcp-rc4", [bulk("Mg", "hcp", a=3.2)], 4)
    yield Case("triclinic-sheared-rc4", [random_box(SHEARED)], 4)
    yield Case(
        "triclinic-strongly_sheared-rc3", [random_box(STRONGLY_SHEARED)], 3
    )
    yield Case(
        "triclinic-rc_above_cell_height-rc7",
        [random_box(SMALL_TRICLINIC, n=6)],
        7,
    )
    yield Case("batched-molecules-rc4", [molecule(m) for m in MOLECULES], 4)
    for (open_name, pbc), (cell_name, cell) in itertools.product(
        OPEN_AXIS_PATTERNS.items(),
        (("orthorhombic", ORTHORHOMBIC), ("sheared", SHEARED)),
    ):
        frames = [random_box(cell, pbc)]
        yield Case(f"mixed_pbc-{cell_name}_{open_name}-rc4", frames, 4)
    periodic_boxes = [
        random_box(SHEARED),
        bulk("Mg", "hcp", a=3.2),
        random_box(ORTHORHOMBIC, n=30, seed=1),
    ]
    yield Case("batched-periodic-rc4", periodic_boxes, 4)
    mixed_boxes = [
        random_box(cell, pbc, seed=seed)
        for seed, (cell, pbc) in enumerate(
            zip((SHEARED, ORTHORHOMBIC, SHEARED), OPEN_AXIS_PATTERNS.values())
        )
    ]
    yield Case("batched-mixed_pbc-rc4", mixed_boxes, 4)
    yield Case(
        "batched-molecule_and_crystal-rc4",
        [molecule("CH3CH2NH2"), random_box(SHEARED)],
        4,
    )
    for cell_name, cell in (
        ("orthorhombic", ORTHORHOMBIC),
        ("sheared", SHEARED),
    ):
        for open_name, pbc in (("", True), ("_z_open", [True, True, False])):
            frames = [random_box(cell, pbc, outside=True)]
            yield Case(f"unwrapped-{cell_name}{open_name}-rc4", frames, 4)


CASES = list(_cases())


def supports(
    nl_method, case=None, self_interaction=False, device="cpu", dtype=None
) -> bool:
    return (
        (nl_method.self_interaction or not self_interaction)
        and device in nl_method.devices
        and (
            case is None
            or (case.periodic or not nl_method.periodic_only)
            and (len(case.frames) == 1 or nl_method.batched)
        )
    )


def to_batch(frames, device="cpu", dtype=torch.float64) -> Batch:
    data = Batch.from_data_list([ase2data(f) for f in frames]).to(device)
    data.pos = data.pos.to(dtype)
    data.cell = data.cell.to(dtype)
    return data


class Edges(NamedTuple):
    """Sorted ``(i, j, S0, S1, S2)`` rows and the distance of each."""

    rows: np.ndarray
    distances: np.ndarray


def _sorted_edges(i, j, images, distances) -> Edges:
    rows = np.column_stack([i, j, images]).astype(np.int64)
    order = np.lexsort(rows.T[::-1])
    return Edges(rows[order], np.asarray(distances)[order])


def _edge_cells(data, i, shifts) -> torch.Tensor:
    return data.cell.view(-1, 3, 3)[data.batch[i]].to(shifts.dtype)


def run_nl(method, data, rc, self_interaction=False) -> Edges:
    """Edges of ``method`` on ``data``. Cartesian shifts are turned into
    integer images; a shift that is not a lattice vector fails."""
    i, j, shifts, _ = method.fn(data, rc, self_interaction=self_interaction)
    if "cell" not in data:
        assert torch.all(shifts == 0)
        images = torch.zeros_like(shifts, dtype=torch.float64)
    elif method.integer_shifts:
        images = shifts
        shifts = (images.unsqueeze(-1) * _edge_cells(data, i, shifts)).sum(1)
    else:
        images = torch.zeros_like(shifts, dtype=torch.float64)
        periodic = data.pbc.view(-1, 3)[data.batch[i]].any(dim=1)
        images[periodic] = torch.linalg.solve(
            _edge_cells(data, i, shifts)[periodic].transpose(-1, -2).double(),
            shifts[periodic].double(),
        )
        assert torch.all(shifts[~periodic] == 0)
    rounded = torch.round(images.double())
    assert torch.all((images.double() - rounded).abs() < 1e-3)
    distances = (data.pos[j] - data.pos[i] + shifts).norm(dim=1)
    return _sorted_edges(
        i.cpu().numpy(),
        j.cpu().numpy(),
        rounded.cpu().numpy(),
        distances.double().cpu().numpy(),
    )


def _image_ranges(frac, cell, pbc, rc) -> List[range]:
    """Every lattice image that can hold a neighbour: the cutoff in units
    of the plane spacing, plus the spread of the atoms across the cell."""
    spacing = 1.0 / np.linalg.norm(np.linalg.inv(cell), axis=0)
    span = frac.max(axis=0) - frac.min(axis=0)
    reach = np.ceil(rc / spacing + span).astype(int)
    return [range(-r, r + 1) if p else range(1) for r, p in zip(reach, pbc)]


def brute_force(data, rc, self_interaction=False) -> Edges:
    """Float64 reference: every pair at every lattice image within reach."""
    pos_all = data.pos.double().cpu().numpy()
    cells = data.cell.view(-1, 3, 3).double().cpu().numpy()
    pbcs = data.pbc.view(-1, 3).cpu().numpy()
    edges, offset = [], 0
    for cell, pbc, n in zip(cells, pbcs, data.n_atoms.tolist()):
        pos = pos_all[offset : offset + n]
        ranges = [range(1)] * 3
        if pbc.any():
            frac = np.linalg.solve(cell.T, pos.T).T
            ranges = _image_ranges(frac, cell, pbc, rc)
        images = np.array(list(itertools.product(*ranges)), dtype=float)
        vec = pos[None, :, None] - pos[:, None, None] + images @ cell
        dist = np.linalg.norm(vec, axis=-1)
        keep = dist < rc
        if not self_interaction:
            same_atom = np.eye(n, dtype=bool)[:, :, None]
            keep &= ~(same_atom & (images == 0).all(axis=1))
        i, j, s = np.nonzero(keep)
        edges.append((i + offset, j + offset, images[s], dist[i, j, s]))
        offset += n
    return _sorted_edges(*(np.concatenate(x) for x in zip(*edges)))


def assert_same_edges(got: Edges, ref: Edges):
    np.testing.assert_array_equal(got.rows, ref.rows)
    np.testing.assert_allclose(
        got.distances, ref.distances, rtol=1e-5, atol=1e-8
    )
