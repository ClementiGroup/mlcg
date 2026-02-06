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
