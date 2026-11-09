import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch.library import triton_op, wrap_triton

from ...utils import ensure_contiguous
from .....geometry.internal_coordinates import compute_torsions
from .utils import _backnorm


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 64}, num_warps=2),
        triton.Config({"BLOCK": 128}, num_warps=4),
        triton.Config({"BLOCK": 256}, num_warps=4),
        triton.Config({"BLOCK": 512}, num_warps=8),
        triton.Config({"BLOCK": 1024}, num_warps=8),
    ],
    key=[],
)
@triton.jit
def dihedral_fwd_kernel(
    pos_ptr,  # [B*N, 3] float32/float64
    index_mapping_ptr,  # [4, E] int32
    k1_ptr,  # [E, deg] float32/float64
    k2_ptr,  # [E, deg] float32/float64
    v_0_ptr,  # [E] float32/float64
    out_E_ptr,  # [E] float32/float64
    E,
    deg: tl.constexpr,  # runtime passed as tl.constexpr by specialization (<=6 typical)
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    blk_idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = blk_idx < E

    off_i = blk_idx
    off_j = blk_idx + E
    off_k = blk_idx + 2 * E
    off_l = blk_idx + 3 * E

    i = tl.load(index_mapping_ptr + off_i, mask=mask, other=0).to(tl.int64)
    j = tl.load(index_mapping_ptr + off_j, mask=mask, other=0).to(tl.int64)
    k = tl.load(index_mapping_ptr + off_k, mask=mask, other=0).to(tl.int64)
    l = tl.load(index_mapping_ptr + off_l, mask=mask, other=0).to(tl.int64)

    # positions
    i3 = i * 3
    j3 = j * 3
    k3 = k * 3
    l3 = l * 3

    xi = tl.load(pos_ptr + i3 + 0, mask=mask, other=0.0).to(tl.float32)
    yi = tl.load(pos_ptr + i3 + 1, mask=mask, other=0.0).to(tl.float32)
    zi = tl.load(pos_ptr + i3 + 2, mask=mask, other=0.0).to(tl.float32)

    xj = tl.load(pos_ptr + j3 + 0, mask=mask, other=0.0).to(tl.float32)
    yj = tl.load(pos_ptr + j3 + 1, mask=mask, other=0.0).to(tl.float32)
    zj = tl.load(pos_ptr + j3 + 2, mask=mask, other=0.0).to(tl.float32)

    xk = tl.load(pos_ptr + k3 + 0, mask=mask, other=0.0).to(tl.float32)
    yk = tl.load(pos_ptr + k3 + 1, mask=mask, other=0.0).to(tl.float32)
    zk = tl.load(pos_ptr + k3 + 2, mask=mask, other=0.0).to(tl.float32)

    xl = tl.load(pos_ptr + l3 + 0, mask=mask, other=0.0).to(tl.float32)
    yl = tl.load(pos_ptr + l3 + 1, mask=mask, other=0.0).to(tl.float32)
    zl = tl.load(pos_ptr + l3 + 2, mask=mask, other=0.0).to(tl.float32)

    # Unit vectors
    # b1 = r_j - r_i, b2 = r_k - r_j, b3 = r_l - r_k
    r1x = xj - xi
    r1y = yj - yi
    r1z = zj - zi
    r1_len = tl.maximum(tl.sqrt(r1x * r1x + r1y * r1y + r1z * r1z), 1e-12)
    b1x, b1y, b1z = (
        tl.div_rn(r1x, r1_len),
        tl.div_rn(r1y, r1_len),
        tl.div_rn(r1z, r1_len),
    )

    r2x = xk - xj
    r2y = yk - yj
    r2z = zk - zj
    r2_len = tl.maximum(tl.sqrt(r2x * r2x + r2y * r2y + r2z * r2z), 1e-12)
    b2x, b2y, b2z = (
        tl.div_rn(r2x, r2_len),
        tl.div_rn(r2y, r2_len),
        tl.div_rn(r2z, r2_len),
    )

    r3x = xl - xk
    r3y = yl - yk
    r3z = zl - zk
    r3_len = tl.maximum(tl.sqrt(r3x * r3x + r3y * r3y + r3z * r3z), 1e-12)
    b3x, b3y, b3z = (
        tl.div_rn(r3x, r3_len),
        tl.div_rn(r3y, r3_len),
        tl.div_rn(r3z, r3_len),
    )

    # n1 = b1 x b2
    n1x = b1y * b2z - b1z * b2y
    n1y = b1z * b2x - b1x * b2z
    n1z = b1x * b2y - b1y * b2x
    # n2 = b2 x b3
    n2x = b2y * b3z - b2z * b3y
    n2y = b2z * b3x - b2x * b3z
    n2z = b2x * b3y - b2y * b3x
    # m = n1 x b2
    mx = n1y * b2z - n1z * b2y
    my = n1z * b2x - n1x * b2z
    mz = n1x * b2y - n1y * b2x

    x = n1x * n2x + n1y * n2y + n1z * n2z
    y = mx * n2x + my * n2y + mz * n2z
    phi = libdevice.atan2(-y, x)

    v_0_loc = tl.load(v_0_ptr + blk_idx, mask=mask).to(tl.float32)
    ener_edge = tl.zeros((BLOCK,), tl.float32) + v_0_loc

    for mm in range(deg):
        m = mm + 1
        # k1 = tl.load(k1_ptr + mm * E + blk_idx, mask=mask, other=0.0).to(
        k1 = tl.load(k1_ptr + mm + blk_idx * deg, mask=mask, other=0.0).to(
            tl.float32
        )
        # k2 = tl.load(k2_ptr + mm * E + blk_idx, mask=mask, other=0.0).to(
        k2 = tl.load(k2_ptr + mm + blk_idx * deg, mask=mask, other=0.0).to(
            tl.float32
        )
        ang = phi * m
        ener_edge += k1 * tl.sin(ang) + k2 * tl.cos(ang)

    tl.store(out_E_ptr + blk_idx, ener_edge, mask=mask)


@triton_op("mlcg_kernels::flash_dihedral", mutates_args={})
@ensure_contiguous
def flash_dihedral(
    pos: torch.Tensor,
    atom_types: torch.Tensor,
    index_mapping: torch.Tensor,
    mapping_batch: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    v_0: torch.Tensor,
    deg: int,
    num_graphs: int,
) -> torch.Tensor:
    """
    pos:      [N*B, 3] float32
    types:    [N*B] int32
    idx_*:    [T] int32
    batch_id: [T] int32
    k1,k2:    [n_types^4, deg] float32
    """
    assert pos.is_cuda and k1.is_cuda and k2.is_cuda
    E = index_mapping.shape[1]
    out_E = torch.zeros(
        (E,), device=pos.device, dtype=torch.float32
    ).contiguous()
    if not (k1.ndim == 2):
        interaction_types = tuple(
            atom_types[index_mapping[ii]] for ii in range(4)
        )
        # the parameters have shape n_features x n_degs
        k1s = (
            torch.vstack([k1[ii][interaction_types] for ii in range(deg)])
            .t()
            .contiguous()
        )
        k2s = (
            torch.vstack([k2[ii][interaction_types] for ii in range(deg)])
            .t()
            .contiguous()
        )
        v_0s = v_0[interaction_types].view(-1, 1)[:, 0].contiguous()
    else:
        k1s = k1
        k2s = k2
        v_0s = v_0
        if v_0s.ndim > 1:
            v_0s = (v_0s[:, 0]).contiguous()
    wrap_triton(dihedral_fwd_kernel)[
        lambda meta: (triton.cdiv(E, meta["BLOCK"]),)
    ](
        pos,
        index_mapping,
        k1s,
        k2s,
        v_0s,
        out_E_ptr=out_E,
        E=E,
        deg=deg,
    )
    y = torch.zeros((num_graphs,), device=pos.device, dtype=torch.float32)
    return y.index_add_(0, mapping_batch, out_E)


def setup_context_flash_dihedral(ctx, inputs, output):
    (
        pos,
        atom_types,
        index_mapping,
        mapping_batch,
        k1,
        k2,
        v_0,
        deg,
        num_graphs,
    ) = inputs
    ctx.save_for_backward(
        pos,
        atom_types,
        index_mapping,
        mapping_batch,
        k1,
        k2,
    )
    ctx.v_0 = v_0
    ctx.deg = deg
    ctx.num_graphs = num_graphs


def backward_flash_dihedral(ctx, grad_y):
    (
        pos,
        atom_types,
        index_mapping,
        mapping_batch,
        k1,
        k2,
    ) = ctx.saved_tensors

    deg = ctx.deg

    grad_pos = None
    if ctx.needs_input_grad[0]:
        grad_pos = dihedral_pos_bwd(
            pos, atom_types, index_mapping, mapping_batch, k1, k2, deg, grad_y
        )

    # Only grad wrt pos
    return grad_pos, None, None, None, None, None, None, None, None


flash_dihedral.register_autograd(
    backward_flash_dihedral,
    setup_context=setup_context_flash_dihedral,
)


@flash_dihedral.register_kernel("cpu")
def cpu_flash_dihedral(
    pos: torch.Tensor,
    atom_types: torch.Tensor,
    index_mapping: torch.Tensor,
    mapping_batch: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    v_0: torch.Tensor,
    deg: int,
    num_graphs: int,
) -> torch.Tensor:

    if not (k1.ndim == 2):
        interaction_types = tuple(
            atom_types[index_mapping[ii]] for ii in range(4)
        )
        # the parameters have shape n_features x n_degs
        k1s = torch.vstack([k1[ii][interaction_types] for ii in range(deg)]).t()
        k2s = torch.vstack([k2[ii][interaction_types] for ii in range(deg)]).t()
        v_0s = v_0[interaction_types].view(-1, 1)
    else:
        k1s = k1
        k2s = k2
        v_0s = v_0

    phis = compute_torsions(pos, index_mapping).flatten()

    _, n_k = k1s.shape
    n_degs = torch.arange(1, n_k + 1, dtype=phis.dtype, device=phis.device)
    # expand the features w.r.t the mult integer so that it has the
    # shape of k1s and k2s
    angles = phis.view(-1, 1) * n_degs.view(1, -1)
    ey = k1s * torch.sin(angles) + k2s * torch.cos(angles)
    # HOTFIX to avoid shape mismatch when using specialized priors
    # TODO: think of a better fix
    if v_0s.ndim > 1:
        v_0s = v_0s[:, 0]

    ey = ey.sum(dim=1) + v_0s
    y = torch.zeros((num_graphs,), device=pos.device, dtype=pos.dtype)

    return y.index_add_(0, mapping_batch, ey)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 64}, num_warps=2),
        triton.Config({"BLOCK": 128}, num_warps=4),
        triton.Config({"BLOCK": 256}, num_warps=4),
        triton.Config({"BLOCK": 512}, num_warps=8),
        triton.Config({"BLOCK": 1024}, num_warps=8),
    ],
    key=[],  # FIXME: check if we need to zeors grad pos during autotune because of atomics
    reset_to_zero=["grad_pos_ptr"],
)
@triton.jit
def dihedral_pos_bwd_kernel(
    pos_ptr,  # [B*N, 3] float32/float64
    index_mapping_ptr,  # [4, E] int32
    k1_ptr,  # [deg, E] float32/float64
    k2_ptr,  # [deg, E] float32/float64
    E,
    upstream_dE_ptr,  # [T]
    mapping_batch_ptr,  # [T]
    grad_pos_ptr,  # [B, N, 3] (atomic adds)
    deg: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    blk_idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = blk_idx < E

    off_i = blk_idx
    off_j = blk_idx + E
    off_k = blk_idx + 2 * E
    off_l = blk_idx + 3 * E

    i = tl.load(index_mapping_ptr + off_i, mask=mask, other=0).to(tl.int64)
    j = tl.load(index_mapping_ptr + off_j, mask=mask, other=0).to(tl.int64)
    k = tl.load(index_mapping_ptr + off_k, mask=mask, other=0).to(tl.int64)
    l = tl.load(index_mapping_ptr + off_l, mask=mask, other=0).to(tl.int64)

    # positions
    i3 = i * 3
    j3 = j * 3
    k3 = k * 3
    l3 = l * 3

    xi = tl.load(pos_ptr + i3 + 0, mask=mask, other=0.0).to(tl.float32)
    yi = tl.load(pos_ptr + i3 + 1, mask=mask, other=0.0).to(tl.float32)
    zi = tl.load(pos_ptr + i3 + 2, mask=mask, other=0.0).to(tl.float32)

    xj = tl.load(pos_ptr + j3 + 0, mask=mask, other=0.0).to(tl.float32)
    yj = tl.load(pos_ptr + j3 + 1, mask=mask, other=0.0).to(tl.float32)
    zj = tl.load(pos_ptr + j3 + 2, mask=mask, other=0.0).to(tl.float32)

    xk = tl.load(pos_ptr + k3 + 0, mask=mask, other=0.0).to(tl.float32)
    yk = tl.load(pos_ptr + k3 + 1, mask=mask, other=0.0).to(tl.float32)
    zk = tl.load(pos_ptr + k3 + 2, mask=mask, other=0.0).to(tl.float32)

    xl = tl.load(pos_ptr + l3 + 0, mask=mask, other=0.0).to(tl.float32)
    yl = tl.load(pos_ptr + l3 + 1, mask=mask, other=0.0).to(tl.float32)
    zl = tl.load(pos_ptr + l3 + 2, mask=mask, other=0.0).to(tl.float32)

    # Unit vectors
    # b1 = r_j - r_i, b2 = r_k - r_j, b3 = r_l - r_k
    r1x = xj - xi
    r1y = yj - yi
    r1z = zj - zi
    r1_len = tl.maximum(tl.sqrt(r1x * r1x + r1y * r1y + r1z * r1z), 1e-12)
    b1x, b1y, b1z = (
        tl.div_rn(r1x, r1_len),
        tl.div_rn(r1y, r1_len),
        tl.div_rn(r1z, r1_len),
    )

    r2x = xk - xj
    r2y = yk - yj
    r2z = zk - zj
    r2_len = tl.maximum(tl.sqrt(r2x * r2x + r2y * r2y + r2z * r2z), 1e-12)
    b2x, b2y, b2z = (
        tl.div_rn(r2x, r2_len),
        tl.div_rn(r2y, r2_len),
        tl.div_rn(r2z, r2_len),
    )

    r3x = xl - xk
    r3y = yl - yk
    r3z = zl - zk
    r3_len = tl.maximum(tl.sqrt(r3x * r3x + r3y * r3y + r3z * r3z), 1e-12)
    b3x, b3y, b3z = (
        tl.div_rn(r3x, r3_len),
        tl.div_rn(r3y, r3_len),
        tl.div_rn(r3z, r3_len),
    )

    # n1 = b1 x b2
    n1x = b1y * b2z - b1z * b2y
    n1y = b1z * b2x - b1x * b2z
    n1z = b1x * b2y - b1y * b2x
    # n2 = b2 x b3
    n2x = b2y * b3z - b2z * b3y
    n2y = b2z * b3x - b2x * b3z
    n2z = b2x * b3y - b2y * b3x
    # m = n1 x b2
    mx = n1y * b2z - n1z * b2y
    my = n1z * b2x - n1x * b2z
    mz = n1x * b2y - n1y * b2x

    x = n1x * n2x + n1y * n2y + n1z * n2z
    y = mx * n2x + my * n2y + mz * n2z
    phi = libdevice.atan2(-y, x)

    # dE/dphi = sum_m m*(k1_m cos(m phi) - k2_m sin(m phi))
    dEdphi = tl.zeros([BLOCK], tl.float32)
    for mm in range(0, deg):
        m = mm + 1
        # k1 = tl.load(k1_ptr + mm * E + blk_idx, mask=mask, other=0.0).to(
        k1 = tl.load(k1_ptr + mm + blk_idx * deg, mask=mask, other=0.0).to(
            tl.float32
        )
        # k2 = tl.load(k2_ptr + mm * E + blk_idx, mask=mask, other=0.0).to(
        k2 = tl.load(k2_ptr + mm + blk_idx * deg, mask=mask, other=0.0).to(
            tl.float32
        )
        ang = phi * m
        dEdphi += m * (k1 * tl.cos(ang) - k2 * tl.sin(ang))

    batch_idx = tl.load(mapping_batch_ptr + blk_idx, mask=mask, other=0).to(
        tl.int64
    )
    upstream = tl.load(upstream_dE_ptr + batch_idx, mask=mask, other=0.0).to(
        tl.float32
    )
    dEdphi *= upstream

    ## Backprop through torsion angle
    d = tl.maximum(x * x + y * y, 1e-12)
    gx = tl.div_rn(y, d) * dEdphi
    gy = tl.div_rn(-x, d) * dEdphi

    # gradients wrt intermediates
    gn1x = gx * n2x
    gn1y = gx * n2y
    gn1z = gx * n2z

    gmx = gy * n2x
    gmy = gy * n2y
    gmz = gy * n2z

    gn2x = gx * n1x + gy * mx
    gn2y = gx * n1y + gy * my
    gn2z = gx * n1z + gy * mz

    # backprop m = n1 × b2
    gn1x += b2y * gmz - b2z * gmy
    gn1y += b2z * gmx - b2x * gmz
    gn1z += b2x * gmy - b2y * gmx

    gb2x = gmy * n1z - gmz * n1y
    gb2y = gmz * n1x - gmx * n1z
    gb2z = gmx * n1y - gmy * n1x

    # backprop n1 = b1 × b2
    gb1x = b2y * gn1z - b2z * gn1y
    gb1y = b2z * gn1x - b2x * gn1z
    gb1z = b2x * gn1y - b2y * gn1x

    gb2x += gn1y * b1z - gn1z * b1y
    gb2y += gn1z * b1x - gn1x * b1z
    gb2z += gn1x * b1y - gn1y * b1x

    # backprop n2 = b2 × b3
    gb2x += b3y * gn2z - b3z * gn2y
    gb2y += b3z * gn2x - b3x * gn2z
    gb2z += b3x * gn2y - b3y * gn2x

    gb3x = gn2y * b2z - gn2z * b2y
    gb3y = gn2z * b2x - gn2x * b2z
    gb3z = gn2x * b2y - gn2y * b2x

    # backprop normalization of b1, b2, b3
    gr1x, gr1y, gr1z = _backnorm(gb1x, gb1y, gb1z, b1x, b1y, b1z, r1_len)
    gr2x, gr2y, gr2z = _backnorm(gb2x, gb2y, gb2z, b2x, b2y, b2z, r2_len)
    gr3x, gr3y, gr3z = _backnorm(gb3x, gb3y, gb3z, b3x, b3y, b3z, r3_len)

    # backprop r1 = r_j - r_i, r2 = r_k - r_j, r3 = r_l - r_k
    tl.atomic_add(grad_pos_ptr + i3 + 0, -gr1x, mask=mask)
    tl.atomic_add(grad_pos_ptr + i3 + 1, -gr1y, mask=mask)
    tl.atomic_add(grad_pos_ptr + i3 + 2, -gr1z, mask=mask)

    tl.atomic_add(grad_pos_ptr + j3 + 0, gr1x - gr2x, mask=mask)
    tl.atomic_add(grad_pos_ptr + j3 + 1, gr1y - gr2y, mask=mask)
    tl.atomic_add(grad_pos_ptr + j3 + 2, gr1z - gr2z, mask=mask)

    tl.atomic_add(grad_pos_ptr + k3 + 0, gr2x - gr3x, mask=mask)
    tl.atomic_add(grad_pos_ptr + k3 + 1, gr2y - gr3y, mask=mask)
    tl.atomic_add(grad_pos_ptr + k3 + 2, gr2z - gr3z, mask=mask)

    tl.atomic_add(grad_pos_ptr + l3 + 0, gr3x, mask=mask)
    tl.atomic_add(grad_pos_ptr + l3 + 1, gr3y, mask=mask)
    tl.atomic_add(grad_pos_ptr + l3 + 2, gr3z, mask=mask)


@triton_op("mlcg_kernels::dihedral_pos_bwd", mutates_args={})
@ensure_contiguous
def dihedral_pos_bwd(
    pos: torch.Tensor,
    atom_types: torch.Tensor,
    index_mapping: torch.Tensor,
    mapping_batch: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    deg: int,
    grad_y: torch.Tensor,
) -> torch.Tensor:
    grad_pos = torch.zeros_like(pos, dtype=torch.float32).contiguous()
    E = index_mapping.shape[1]
    if not (k1.ndim == 2):
        interaction_types = tuple(
            atom_types[index_mapping[ii]] for ii in range(4)
        )
        # the parameters have shape n_features x n_degs
        k1s = (
            torch.vstack([k1[ii][interaction_types] for ii in range(deg)])
            .t()
            .contiguous()
        )
        k2s = (
            torch.vstack([k2[ii][interaction_types] for ii in range(deg)])
            .t()
            .contiguous()
        )
    else:
        k1s = k1
        k2s = k2

    wrap_triton(dihedral_pos_bwd_kernel)[
        lambda meta: (triton.cdiv(E, meta["BLOCK"]),)
    ](
        pos,
        index_mapping,
        k1s,
        k2s,
        E,
        grad_y,
        mapping_batch,
        grad_pos,
        deg=deg,
    )
    return grad_pos


@dihedral_pos_bwd.register_kernel("cpu")
def cpu_bwd_dihedral(
    pos: torch.Tensor,
    atom_types: torch.Tensor,
    index_mapping: torch.Tensor,
    mapping_batch: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    deg: int,
    grad_y: torch.Tensor,
) -> torch.Tensor:
    i, j, k, l = index_mapping

    # Compute distances and unit vectors
    r1 = pos[j] - pos[i]
    r2 = pos[k] - pos[j]
    r3 = pos[l] - pos[k]
    b1 = F.normalize(r1, dim=1)
    b2 = F.normalize(r2, dim=1)
    b3 = F.normalize(r3, dim=1)

    # Compute torsion angles
    n1 = torch.cross(b1, b2)
    n2 = torch.cross(b2, b3)
    m1 = torch.cross(n1, b2, dim=1)
    y = torch.sum(m1 * n2, dim=-1)
    x = torch.sum(n1 * n2, dim=-1)
    phis = torch.atan2(-y, x)

    # Compute dE/dphi
    if not (k1.ndim == 2):
        interaction_types = tuple(
            atom_types[index_mapping[ii]] for ii in range(4)
        )
        k1s = torch.vstack([k1[ii][interaction_types] for ii in range(deg)]).t()
        k2s = torch.vstack([k2[ii][interaction_types] for ii in range(deg)]).t()
    else:
        k1s = k1
        k2s = k2

    _, n_k = k1s.shape
    n_degs = torch.arange(1, n_k + 1, dtype=phis.dtype, device=phis.device)
    angles = phis.view(-1, 1) * n_degs.view(1, -1)
    de = (
        (k1s * torch.cos(angles) - k2s * torch.sin(angles)) * n_degs.view(1, -1)
    ).sum(dim=1, keepdim=True)

    # Backprop
    b = mapping_batch
    de = de * grad_y[b].unsqueeze(1)

    # Backprob trough the torsion angle:
    d = x**2 + y**2
    gx = (y / d).unsqueeze(-1) * de
    gy = (-x / d).unsqueeze(-1) * de

    # gradients wrt intermediates
    gn1 = gx * n2
    gm = gy * n2
    gn2 = gx * n1 + gy * m1

    # backprop m = n1 × b2
    gn1 = gn1 + torch.cross(b2, gm, dim=1)
    gb2 = torch.cross(gm, n1, dim=1)

    # backprop n1 = b1 × b2
    gb1 = torch.cross(b2, gn1, dim=1)
    gb2 = gb2 + torch.cross(gn1, b1, dim=1)

    # backprop n2 = b2 × b3
    gb2 = gb2 + torch.cross(b3, gn2, dim=1)
    gb3 = torch.cross(gn2, b2, dim=1)

    def backnorm(gb, b, r):
        norm = r.norm(dim=1, keepdim=True)
        return (gb - (gb * b).sum(dim=1, keepdim=True) * b) / norm

    gr1 = backnorm(gb1, b1, r1)
    gr2 = backnorm(gb2, b2, r2)
    gr3 = backnorm(gb3, b3, r3)

    grad_pos = torch.zeros_like(pos)
    grad_pos.index_add_(0, i, -gr1)
    grad_pos.index_add_(0, j, gr1 - gr2)
    grad_pos.index_add_(0, k, gr2 - gr3)
    grad_pos.index_add_(0, l, gr3)

    return grad_pos
