"""TODO: write tests"""

import math
import torch
from typing import List, Optional


@torch.jit.script
def safe_norm(
    input: torch.Tensor,
    dim: Optional[List[int]] = None,
    keepdims: bool = True,
    eps: float = 1e-16,
) -> torch.Tensor:
    """Compute Euclidean norm of input so that 0-norm vectors can be used in
    the backpropagation"""
    if dim is None:
        dim = [0]
    return torch.sqrt(
        torch.square(input).sum(dim=dim, keepdim=keepdims) + eps
    ) - math.sqrt(eps)


@torch.jit.script
def safe_normalization(
    input: torch.Tensor, norms: torch.Tensor
) -> torch.Tensor:
    """Normalizes input using norms avoiding divitions by zero"""
    mask = (norms > 0.0).flatten()
    out = input.clone()
    # out = torch.zeros_like(input)
    out[mask] = input[mask] / norms[mask]
    return out


@torch.jit.script
def compute_distance_vectors(
    pos: torch.Tensor,
    mapping: torch.Tensor,
    cell_shifts: Optional[torch.Tensor] = None,
):
    r"""Compute the distance (or displacement) vectors between the positions in
    :obj:`pos` following the :obj:`mapping` assuming that that mapping indices follow::

     i--j

    such that:

    .. math::

        r_{ij} &= ||\mathbf{r}_j - \mathbf{r}_i||_{2} \\
        \hat{\mathbf{r}}_{ij} &= \frac{\mathbf{r}_{ij}}{r_{ij}}

    In the case of periodic boundary conditions, :obj:`cell_shifts` must be
    provided so that :math:`\mathbf{r}_j` can be outside of the original unit
    cell.
    """
    assert mapping.dim() == 2
    assert mapping.shape[0] == 2

    if cell_shifts is None:
        dr = pos[mapping[1]] - pos[mapping[0]]
    else:
        dr = pos[mapping[1]] - pos[mapping[0]] + cell_shifts

    distances = safe_norm(dr, dim=[1])

    direction_vectors = safe_normalization(dr, distances)
    return distances, direction_vectors


@torch.jit.script
def compute_distances(
    pos: torch.Tensor,
    mapping: torch.Tensor,
    cell_shifts: Optional[torch.Tensor] = None,
):
    r"""Compute the distance between the positions in :obj:`pos` following the
    :obj:`mapping` assuming that mapping indices follow::

     i--j

    such that:

    .. math::

        r_{ij} = ||\mathbf{r}_j - \mathbf{r}_i||_{2}

    In the case of periodic boundary conditions, :obj:`cell_shifts` must be
    provided so that :math:`\mathbf{r}_j` can be outside of the original unit
    cell.
    """
    assert mapping.dim() == 2
    assert mapping.shape[0] == 2

    if cell_shifts is None:
        dr = pos[mapping[1]] - pos[mapping[0]]
    else:
        dr = pos[mapping[1]] - pos[mapping[0]] + cell_shifts

    return dr.norm(p=2, dim=1)


@torch.jit.script
def compute_angles_raw(
    pos: torch.Tensor,
    mapping: torch.Tensor,
    cell_shifts: Optional[torch.Tensor] = None,
):
    r"""Compute the raw angle (in radians) between the positions in :obj:`pos` following the :obj:`mapping` assuming that the mapping indices follow::

       j--k
      /
     i


    .. math::

        \theta_{ijk} = &\text{atan2}(\Vert \hat{\mathbf{n}} \vert, \mathbf{r}_{ij} \cdot \mathbf{r}_{kj} ) \\
        \mathbf{r}_{ij} &= \mathbf{r}_i - \mathbf{r}_j \\
        \mathbf{r}_{kj} &= \mathbf{r}_k - \mathbf{r}_j \\
        \mathbf{\hat{n}} &= \frac{\mathbf{r}_{ij} \times \mathbf{r}_{kj}}{\Vert \mathbf{r}_{ij} \times \mathbf{r}_{kj} \Vert} 

    """
    assert mapping.dim() == 2
    assert mapping.shape[0] == 3

    dr1 = pos[mapping[0]] - pos[mapping[1]]
    dr2 = pos[mapping[2]] - pos[mapping[1]]

    n = torch.cross(dr1, dr2, dim=1)
    n = n.norm(p=2, dim=1)
    d = (dr1 * dr2).sum(dim=1)
    theta = torch.atan2(n, d)

    return theta


@torch.jit.script
def compute_angles_cos(
    pos: torch.Tensor,
    mapping: torch.Tensor,
    cell_shifts: Optional[torch.Tensor] = None,
):
    r"""Compute the cosine of the angle between the positions in :obj:`pos` following the :obj:`mapping` assuming that the mapping indices follow::

       j--k
      /
     i

    .. math::

        \cos{\theta_{ijk}} &= \frac{\mathbf{r}_{ij} \cdot \mathbf{r}_{jk}}{ \Vert \mathbf{r}_{ji}  \Vert \Vert \mathbf{r}_{kj} \Vert}  \\
        \mathbf{r}_{ij} &= \mathbf{r}_i - \mathbf{r}_j \\
        \mathbf{r}_{kj} &= \mathbf{r}_k - \mathbf{r}_j 

    In the case of periodic boundary conditions, :obj:`cell_shifts` must be
    provided so that :math:`\mathbf{r}_j` can be outside of the original unit
    cell.
    """
    assert mapping.dim() == 2
    assert mapping.shape[0] == 3

    dr1 = pos[mapping[0]] - pos[mapping[1]]
    dr2 = pos[mapping[2]] - pos[mapping[1]]
    cos_theta = (
        (dr1 * dr2).sum(dim=1) / dr1.norm(p=2, dim=1) / dr2.norm(p=2, dim=1)
    )
    return cos_theta


@torch.jit.script
def compute_torsions(
    pos: torch.Tensor, mapping: torch.Tensor, eps: float = 1e-6
):
    r"""
    Compute the angle between two planes from positions in :obj:'pos' following the
    :obj:`mapping`::

    For dihedrals: the angle w.r.t. position of i&l is positive if l i rotated clockwise
    when staring down bond jk::

       j--k--l
      /
     i

    For impropers: the angle is positive if when looking in plane ikj, l is rotated clockwise::

     k
      \\
       l--j
      /
     i

    The angle is computed using the formula:

    .. math::

        \phi_{ijkl} &= \text{atan2}(-\mathbf{m} \cdot \mathbf{n}_2, \mathbf{n}_2 \cdot \mathbf{n}_1) \\
        \mathbf{n}_1 &= \mathbf{\hat{r}}_{ji} \times \mathbf{\hat{r}}_{kj} \\
        \mathbf{n}_2 &= \mathbf{\hat{r}}_{kj} \times \mathbf{\hat{r}}_{jl} \\
        \mathbf{m} &= \mathbf{n}_{1} \times \mathbf{\hat{r}}_{kj} 

    Where the :math:`\hat{u}` indicates a normalized vector in direction of :math:`u`. 
    The order and signs in the atan2 arguments are needed to obtain values consistent 
    with mdtraj. In the case of periodic boundary conditions, :obj:`cell_shifts` must be
    provided so that :math:`\mathbf{r}_j` can be outside of the original unit
    cell.
    """
    assert mapping.dim() == 2
    assert mapping.shape[0] == 4
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
    theta = torch.atan2(-y, x)  # -y to match MDTraj convention

    return theta
