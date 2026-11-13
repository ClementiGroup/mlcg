import pytest
import torch
import numpy as np
import mdtraj
from mdtraj.core.element import carbon
from mlcg.geometry.internal_coordinates import compute_torsions
import copy
from mlcg.geometry.statistics import _get_bin_centers
from mlcg.nn.prior import Dihedral

np.random.seed(15149)

# Mock data of ten atoms for 1000 frames
n_frames = 1000
n_atoms = 10
test_coords = np.random.randn(n_frames, n_atoms, 3).astype("float32")
torch_coords = torch.tensor(test_coords.reshape(n_frames * n_atoms, 3))

dihedral_atoms = np.array(
    [[i, i + 1, i + 2, i + 3] for i in range(test_coords.shape[1] - 3)]
)
mapping = torch.tensor(
    np.concatenate(
        [dihedral_atoms.T + (n_atoms * i) for i in range(n_frames)], axis=1
    )
)

topo = mdtraj.core.topology.Topology()
topo.add_chain()
topo.add_residue("ALA1", topo._chains[0])
for _ in range(n_atoms):
    topo.add_atom("CA", carbon, list(topo.residues)[0])
traj = mdtraj.core.trajectory.Trajectory(test_coords, topo)


def test_mdtraj_dihedral_compatibility():
    """Tests to make sure that MLCG torsion calculations
    agree with the dihedral calculations from MDTraj
    """

    mlcg_dihedrals = (
        compute_torsions(torch_coords, mapping)
        .numpy()
        .reshape(n_frames, dihedral_atoms.shape[0])
    )
    mdtraj_dihedrals = mdtraj.compute_dihedrals(traj, dihedral_atoms)
    np.testing.assert_allclose(mlcg_dihedrals, mdtraj_dihedrals, rtol=1e-2)


def test_dihedral_aic_criterion():
    """
    Test to ensure that proper index of fitting is choosen
    Perturb coordinates of first dihedral with a 3-amplitude sine wave
    """
    TargetPrior = Dihedral
    n_atoms = 4
    n_frames = 20000
    test_coords = np.random.randn(n_frames, n_atoms, 3).astype("float32")
    dihedral_atoms = np.array(
        [[i, i + 1, i + 2, i + 3] for i in range(test_coords.shape[1] - 3)]
    )
    mapping = torch.tensor(
        np.concatenate(
            [dihedral_atoms.T + (n_atoms * i) for i in range(n_frames)], axis=1
        )
    )
    test_coords = np.random.randn(n_frames, n_atoms, 3).astype("float32")
    torch_coords = torch.tensor(test_coords.reshape(n_frames * n_atoms, 3))
    torch_coords[1::n_atoms] = torch.tensor((0, 0, 0))
    torch_coords[2::n_atoms] = torch.tensor((1, 0, 0))
    torch_coords[3::n_atoms] = torch.tensor((2, 1, 0))
    pos = torch_coords[:4]
    ind_mapping = torch.tensor((0, 1, 2, 3))
    ind_mapping = torch.reshape(ind_mapping, [4, 1])
    theta_map = theta_mapping(pos, ind_mapping)
    k1s = [0.5, 1, 0, 0, 0, 0]
    k2s = [0, 0, 0, 0, 0, 0]
    v_0 = 0
    samples = sample_from_curve(v_0, k1s, k2s, n_frames)
    for i_s, sample in enumerate(samples):
        for k, v in theta_map.items():
            if np.abs(sample - v) < 1e-2:
                torch_coords[i_s * n_atoms] = torch.stack(k)

    values = TargetPrior.compute_features(torch_coords, mapping)
    b_min = values.min()
    b_max = values.max()
    nbins = 100
    bin_centers = _get_bin_centers(values, nbins, b_min=b_min, b_max=b_max)
    hist = torch.histc(values, bins=nbins, min=b_min, max=b_max)

    mask = hist > 0
    bin_centers_nz = bin_centers[mask]
    ncounts_nz = hist[mask]
    dG_nz = -torch.log(ncounts_nz)
    params = TargetPrior.fit_from_potential_estimates(bin_centers_nz, dG_nz)

    for i_k, (k1, k2) in enumerate(
        zip(params["k1s"].values(), params["k2s"].values())
    ):
        assert np.abs(k1 - k1s[i_k]) < 0.1, f"{k1}, {k1s[i_k]}"
        assert np.abs(k2 - k2s[i_k]) < 0.1, f"{k2}, {k2s[i_k]}"


def sample_from_curve(v_0, k1s, k2s, n_frames, beta=1):
    """
    Sample from a potential
    Inputs:
        v_0
            constant offset
        k1s
            values from sin coefficient
        k2s
            values from cos coefficient
        n_frames
            number of samples to pull
    Outputs:
        theta
            values drawn from potential
    """
    V = v_0
    thetas = torch.from_numpy(np.linspace(-np.pi, np.pi, 100))
    for ik, (k1, k2) in enumerate(zip(k1s, k2s)):
        V += k1 * torch.sin((ik + 1) * thetas) + k2 * torch.cos(
            (ik + 1) * thetas
        )
    pi = torch.exp(-beta * V)
    pi = pi / torch.sum(pi)
    cum_pi = torch.cumsum(pi, dim=0).numpy()
    theta = []
    for _ in range(n_frames):
        sel_ind = np.where(cum_pi > np.random.rand())[0][0]
        theta.append(thetas[sel_ind].numpy())
    return theta


def theta_mapping(pos, mapping, x0=torch.tensor(-1), r=1):
    """
    Given a list of position find the values of x,y,z that give a particular torsion angle
    Inputs:
        pos
            positions
        mapping
            atoms which form torsion
        x0
            position along x dimesion (easiest to keep fixed)
        r
            radius of rotation
    Outputs:
        theta_map
            keys : (x,y,z) position
            values : torsion angles
    """
    theta_map = {}
    thetas = torch.linspace(-np.pi, np.pi, 100)
    for thet in thetas:
        y0 = r * torch.cos(thet)
        z0 = r * torch.sin(thet)
        pos[0] = torch.tensor((x0, y0, z0))
        dr1 = pos[mapping[1]] - pos[mapping[0]]
        dr1 = dr1 / dr1.norm(p=2, dim=1)[:, None]
        dr2 = pos[mapping[2]] - pos[mapping[1]]
        dr2 = dr2 / dr2.norm(p=2, dim=1)[:, None]
        dr3 = pos[mapping[3]] - pos[mapping[2]]
        dr3 = dr3 / dr3.norm(p=2, dim=1)[:, None]
        n1 = torch.cross(dr1, dr2, dim=1)
        n2 = torch.cross(dr2, dr3, dim=1)
        m1 = torch.cross(n1, dr2, dim=1)
        y = torch.sum(m1 * n2, dim=-1)
        x = torch.sum(n1 * n2, dim=-1)
        theta = torch.atan2(-y, x)
        theta_map[(x0, y0, z0)] = theta.numpy()
    return theta_map


# box lengths with more than 10 mantissa bits, so that TF32 would round them
TF32_CELLS = [
    torch.diag(torch.tensor([71.7, 70.3, 75.9])),
    torch.tensor([[71.7, 0.0, 0.0], [5.3, 70.3, 0.0], [3.1, 4.7, 75.9]]),
]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cell", TF32_CELLS)
def test_prior_cell_shifts_ignore_tf32(cell, device, tf32_matmul):
    """The priors' minimum-image shifts must not depend on the float32
    matmul precision."""
    from mlcg.nn.prior.base import compute_cell_shifts

    gen = torch.Generator().manual_seed(0)
    pos = (torch.rand(40, 3, generator=gen) @ cell).to(device)
    mapping = torch.triu_indices(40, 40, 1).to(device)
    args = (
        pos,
        mapping,
        torch.ones(1, 3, dtype=torch.bool, device=device),
        cell[None].to(device),
        torch.zeros(40, dtype=torch.long, device=device),
    )
    reference = compute_cell_shifts(*args)
    with tf32_matmul():
        result = compute_cell_shifts(*args)
    torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.parametrize(
    "cell",
    [
        torch.diag(torch.tensor([20.0, 18.0, 16.0])),
        torch.tensor([[20.0, 0.0, 0.0], [5.0, 18.0, 0.0], [3.0, 4.0, 16.0]]),
    ],
)
def test_prior_cell_shifts_minimum_image(cell):
    """Rows of ``cell`` are the lattice vectors (pos = f @ cell). Pairs
    closer than half the smallest cell height get their minimum image, and
    no shift is applied along a non-periodic lattice vector."""
    from mlcg.nn.prior.base import compute_cell_shifts

    cell = cell.double()
    gen = torch.Generator().manual_seed(0)
    pos = torch.rand(100, 3, generator=gen, dtype=torch.double) @ cell
    i, j = torch.triu_indices(100, 100, 1)
    batch = torch.zeros(100, dtype=torch.long)

    def shifts(pbc):
        return compute_cell_shifts(
            pos, torch.stack([i, j]), torch.tensor([pbc]), cell[None], batch
        ).squeeze(-1)

    images = torch.cartesian_prod(*[torch.arange(-2, 3)] * 3).double() @ cell
    nearest = (pos[j, None] + images - pos[i, None]).norm(dim=-1).min(dim=1)
    volume = torch.linalg.det(cell).abs()
    heights = [
        volume / torch.linalg.cross(cell[(k + 1) % 3], cell[(k + 2) % 3]).norm()
        for k in range(3)
    ]
    short = nearest.values < min(heights) / 2
    distances = (pos[j] + shifts([True] * 3) - pos[i]).norm(dim=1)
    torch.testing.assert_close(distances[short], nearest.values[short])

    lattice_shifts = torch.linalg.solve(cell.T, shifts([True, True, False]).T)
    torch.testing.assert_close(lattice_shifts, lattice_shifts.round())
    assert (lattice_shifts[2] == 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_safe_normalization_has_no_host_sync():
    """No data-dependent shapes: normalizing, forward and backward, needs no
    device-to-host sync (torch 2.13's Inductor fails on two masked
    normalizations in a row); zero vectors stay zero."""
    from mlcg.geometry.internal_coordinates import safe_normalization

    x = torch.tensor(
        [[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]], device="cuda", requires_grad=True
    )
    norms = x.detach().norm(dim=1, keepdim=True)
    torch.cuda.set_sync_debug_mode("error")
    try:
        out = safe_normalization(x, norms)
        out.sum().backward()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    expected = torch.tensor([[0.6, 0.8, 0.0], [0.0, 0.0, 0.0]], device="cuda")
    torch.testing.assert_close(out.detach(), expected)
