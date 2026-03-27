import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton
from torch_geometric.utils import scatter

from ...utils import ensure_contiguous
from .....geometry import compute_angles_cos


import triton.language as tl

@triton.jit
def atan2_tl(y, x):
    pi      = tl.full((), 3.141592653589793, tl.float32)
    half_pi = pi * 0.5
    eps     = tl.full((), 1e-12, tl.float32)

    y = y.to(tl.float32)
    x = x.to(tl.float32)

    ax = tl.abs(x)
    ay = tl.abs(y)

    # a in [0, pi/2], using a rational/polynomial approx
    # Use the "min/max" trick to keep ratio <= 1
    t0 = tl.minimum(ax, ay)
    t1 = tl.maximum(ax, ay)
    r  = t0 / (t1 + eps)     # in [0,1]

    r2 = r * r

    # Approx for atan(r) on [0,1]
    # One commonly used minimax-ish polynomial:
    # atan(r) ≈ r*(c1 + c3*r^2 + c5*r^4 + c7*r^6)
    c1 = 0.9998660
    c3 = -0.3302995
    c5 = 0.1801410
    c7 = -0.0851330
    a = r * (c1 + r2 * (c3 + r2 * (c5 + r2 * c7)))

    # If ay > ax, atan(ay/ax) = pi/2 - atan(ax/ay)
    a = tl.where(ay > ax, half_pi - a, a)

    # Quadrant fix to get atan2
    # For x < 0: angle = pi - a (if y>=0) else a - pi
    a = tl.where(x < 0, tl.where(y >= 0, pi - a, a - pi), a)

    # Sign for y (when x>=0 the polynomial gave |angle|)
    a = tl.where((x >= 0) & (y < 0), -a, a)

    # Handle x==0 explicitly
    a = tl.where(x == 0, tl.where(y > 0, half_pi,
                         tl.where(y < 0, -half_pi, 0.0)), a)
    return a

def pack_type_key(ti, tj, tk, tl_, n_types: int):
    # ti,tj,tk,tl_ are int tensors
    return (((ti * n_types + tj) * n_types + tk) * n_types + tl_)

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
def flash_dihedral_fwd_kernel(
    pos_ptr,                 # [B*N, 3] float32/float64
    atom_types_ptr,          # [B*N] int32
    index_mapping_ptr,       # [E,4] int32
    k1_ptr,                  # [K, deg] float32/float64 ; K = n_types^4
    k2_ptr,                  # [K, deg] float32/float64 ; K = n_types^4
    v_0_ptr,                 # [K, deg] float32/float64 ; K = n_types^4
    out_E_ptr,               # [E] float32/float64
    DBG_ptr,               # [E] float32/float64
    E: tl.constexpr,
    deg: tl.constexpr,  # runtime passed as tl.constexpr by specialization (<=6 typical)
    n_types: tl.constexpr,
    EDGES_STRIDE0: tl.constexpr,
    EDGES_STRIDE1: tl.constexpr,
    stride_km: tl.constexpr,
    stride_ki: tl.constexpr, 
    stride_kj: tl.constexpr, 
    stride_kk: tl.constexpr, 
    stride_kl: tl.constexpr, 
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    blk_idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = blk_idx < E
    
    off_i = blk_idx * EDGES_STRIDE0 + 0 * EDGES_STRIDE1
    off_j = blk_idx * EDGES_STRIDE0 + 1 * EDGES_STRIDE1
    off_k = blk_idx * EDGES_STRIDE0 + 2 * EDGES_STRIDE1
    off_l = blk_idx * EDGES_STRIDE0 + 3 * EDGES_STRIDE1
    
    i = tl.load(index_mapping_ptr + off_i, mask=mask, other=0).to(tl.int64)
    j = tl.load(index_mapping_ptr + off_j, mask=mask, other=0).to(tl.int64)
    k = tl.load(index_mapping_ptr + off_k, mask=mask, other=0).to(tl.int64)
    l = tl.load(index_mapping_ptr + off_l, mask=mask, other=0).to(tl.int64)
    
    # types -> key
    type_i = tl.load(atom_types_ptr + i , mask=mask, other=-1).to(tl.int64)
    type_j = tl.load(atom_types_ptr + j , mask=mask, other=-2).to(tl.int64)
    type_k = tl.load(atom_types_ptr + k , mask=mask, other=-3).to(tl.int64)
    type_l = tl.load(atom_types_ptr + l , mask=mask, other=-4).to(tl.int64)
    
    # positions
    i3 = i * 3
    j3 = j * 3
    k3 = k * 3
    l3 = l * 3
    
    xi = tl.load(pos_ptr + i3 + 0, mask=mask, other=0.).to(tl.float32)
    yi = tl.load(pos_ptr + i3 + 1, mask=mask, other=0.).to(tl.float32)
    zi = tl.load(pos_ptr + i3 + 2, mask=mask, other=0.).to(tl.float32)
    
    xj = tl.load(pos_ptr + j3 + 0, mask=mask, other=0.).to(tl.float32)
    yj = tl.load(pos_ptr + j3 + 1, mask=mask, other=0.).to(tl.float32)
    zj = tl.load(pos_ptr + j3 + 2, mask=mask, other=0.).to(tl.float32)

    xk = tl.load(pos_ptr + k3 + 0, mask=mask, other=0.).to(tl.float32)
    yk = tl.load(pos_ptr + k3 + 1, mask=mask, other=0.).to(tl.float32)
    zk = tl.load(pos_ptr + k3 + 2, mask=mask, other=0.).to(tl.float32)

    xl = tl.load(pos_ptr + l3 + 0, mask=mask, other=0.).to(tl.float32)
    yl = tl.load(pos_ptr + l3 + 1, mask=mask, other=0.).to(tl.float32)
    zl = tl.load(pos_ptr + l3 + 2, mask=mask, other=0.).to(tl.float32)
    
    # b1 = r_i - r_j, b2 = r_k - r_j, b3 = r_l - r_k
    b1x = xi - xj 
    b1y = yi - yj
    b1z = zi - zj
    
    b2x = xk - xj
    b2y = yk - yj
    b2z = zk - zj
    
    b3x = xl - xk
    b3y = yl - yk
    b3z = zl - zk

    # n1 = b1 x b2
    n1x = b1y * b2z - b1z * b2y
    n1y = b1z * b2x - b1x * b2z
    n1z = b1x * b2y - b1y * b2x
    # n2 = b3 x b2
    n2x = b3y * b2z - b3z * b2y
    n2y = b3z * b2x - b3x * b2z
    n2z = b3x * b2y - b3y * b2x
    
    # squared norms
    b2_sq = b2x*b2x + b2y*b2y + b2z*b2z 
    b2_len = tl.sqrt(b2_sq)

    # phi (same as forward)
    x = n1x*n2x + n1y*n2y + n1z*n2z
    cx = n1y*n2z - n1z*n2y
    cy = n1z*n2x - n1x*n2z
    cz = n1x*n2y - n1y*n2x
    y = (cx*b2x + cy*b2y + cz*b2z) / b2_len
    phi = atan2_tl(y, x)
    

    #key = (((type_i * n_types + type_j) * n_types + type_k) * n_types + type_l).to(tl.int32)
    key = (type_i*stride_ki + type_j*stride_kj + type_k*stride_kk + type_l*stride_kl).to(tl.int32)

    v_0_loc = tl.load(v_0_ptr + key, mask=mask).to(tl.float32)
    # --- energy sum ---
    #ener_edge = tl.full((BLOCK,), n_types, tl.float32)  # accumulate in fp32
    ener_edge = tl.zeros((BLOCK,), tl.float32) + v_0_loc

    tl.store(DBG_ptr + blk_idx, ener_edge, mask=mask)

    for mm in range(deg):
        m = mm + 1
        k1 = tl.load(k1_ptr + key + mm * stride_km, mask=mask, other=0.0).to(tl.float32)
        k2 = tl.load(k2_ptr + key + mm * stride_km, mask=mask, other=0.0).to(tl.float32)
        ang = phi * m
        ener_edge += k1 * tl.sin(ang) + k2 * tl.cos(ang) 
    
    tl.store(out_E_ptr + blk_idx, ener_edge, mask=mask)
    
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
def flash_dihedral_bwd_kernel(
    pos_ptr, 
    atom_types_ptr,
    index_mapping_ptr,
    k1_ptr, 
    k2_ptr,
    E,
    upstream_dE_ptr,     # [T]
    grad_pos_ptr,        # [B, N, 3] (atomic adds)
    deg: tl.constexpr,
    EDGES_STRIDE0: tl.constexpr,
    EDGES_STRIDE1: tl.constexpr,
    stride_km: tl.constexpr,
    stride_ki: tl.constexpr, 
    stride_kj: tl.constexpr, 
    stride_kk: tl.constexpr, 
    stride_kl: tl.constexpr, 
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    blk_idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = blk_idx < E
    
    off_i = blk_idx * EDGES_STRIDE0 + 0 * EDGES_STRIDE1
    off_j = blk_idx * EDGES_STRIDE0 + 1 * EDGES_STRIDE1
    off_k = blk_idx * EDGES_STRIDE0 + 2 * EDGES_STRIDE1
    off_l = blk_idx * EDGES_STRIDE0 + 3 * EDGES_STRIDE1
    
    i = tl.load(index_mapping_ptr + off_i, mask=mask, other=0).to(tl.int64)
    j = tl.load(index_mapping_ptr + off_j, mask=mask, other=0).to(tl.int64)
    k = tl.load(index_mapping_ptr + off_k, mask=mask, other=0).to(tl.int64)
    l = tl.load(index_mapping_ptr + off_l, mask=mask, other=0).to(tl.int64)
    
    # types -> key
    type_i = tl.load(atom_types_ptr + i , mask=mask, other=-1).to(tl.int64)
    type_j = tl.load(atom_types_ptr + j , mask=mask, other=-2).to(tl.int64)
    type_k = tl.load(atom_types_ptr + k , mask=mask, other=-3).to(tl.int64)
    type_l = tl.load(atom_types_ptr + l , mask=mask, other=-4).to(tl.int64)
    
    # positions
    i3 = i * 3
    j3 = j * 3
    k3 = k * 3
    l3 = l * 3
    
    xi = tl.load(pos_ptr + i3 + 0, mask=mask, other=0.).to(tl.float32)
    yi = tl.load(pos_ptr + i3 + 1, mask=mask, other=0.).to(tl.float32)
    zi = tl.load(pos_ptr + i3 + 2, mask=mask, other=0.).to(tl.float32)
    
    xj = tl.load(pos_ptr + j3 + 0, mask=mask, other=0.).to(tl.float32)
    yj = tl.load(pos_ptr + j3 + 1, mask=mask, other=0.).to(tl.float32)
    zj = tl.load(pos_ptr + j3 + 2, mask=mask, other=0.).to(tl.float32)

    xk = tl.load(pos_ptr + k3 + 0, mask=mask, other=0.).to(tl.float32)
    yk = tl.load(pos_ptr + k3 + 1, mask=mask, other=0.).to(tl.float32)
    zk = tl.load(pos_ptr + k3 + 2, mask=mask, other=0.).to(tl.float32)

    xl = tl.load(pos_ptr + l3 + 0, mask=mask, other=0.).to(tl.float32)
    yl = tl.load(pos_ptr + l3 + 1, mask=mask, other=0.).to(tl.float32)
    zl = tl.load(pos_ptr + l3 + 2, mask=mask, other=0.).to(tl.float32)
    
    # b1 = r_i - r_j, b2 = r_k - r_j, b3 = r_l - r_k
    b1x = xi - xj 
    b1y = yi - yj
    b1z = zi - zj
    
    b2x = xk - xj
    b2y = yk - yj
    b2z = zk - zj
    
    b3x = xl - xk
    b3y = yl - yk
    b3z = zl - zk

    # n1 = b1 x b2
    n1x = b1y * b2z - b1z * b2y
    n1y = b1z * b2x - b1x * b2z
    n1z = b1x * b2y - b1y * b2x
    # n2 = b3 x b2
    n2x = b3y * b2z - b3z * b2y
    n2y = b3z * b2x - b3x * b2z
    n2z = b3x * b2y - b3y * b2x
    
    # squared norms
    n1_sq = n1x*n1x + n1y*n1y + n1z*n1z
    n2_sq = n2x*n2x + n2y*n2y + n2z*n2z
    b2_sq = b2x*b2x + b2y*b2y + b2z*b2z 
    b2_len = tl.sqrt(b2_sq)

    # phi (same as forward)
    x = n1x*n2x + n1y*n2y + n1z*n2z
    cx = n1y*n2z - n1z*n2y
    cy = n1z*n2x - n1x*n2z
    cz = n1x*n2y - n1y*n2x
    y = (cx*b2x + cy*b2y + cz*b2z) / b2_len
    phi = atan2_tl(y, x)
    

    #key = (((type_i * n_types + type_j) * n_types + type_k) * n_types + type_l).to(tl.int32)
    key = (type_i*stride_ki + type_j*stride_kj + type_k*stride_kk + type_l*stride_kl).to(tl.int32)


    # dE/dphi = sum_m m*(k1_m cos(m phi) - k2_m sin(m phi))
    dEdphi = tl.zeros([BLOCK], tl.float32)
    for mm in range(0, deg):
        m = mm + 1
        k1 = tl.load(k1_ptr + key + mm * stride_km, mask=mask, other=0.0).to(tl.float32)
        k2 = tl.load(k2_ptr + key + mm * stride_km, mask=mask, other=0.0).to(tl.float32)
        ang = phi * m
        dEdphi += m * (k1 * tl.cos(ang) - k2 * tl.sin(ang))

    upstream = tl.load(upstream_dE_ptr + blk_idx, mask=mask, other=0.0).to(tl.float32)
    dEdphi *= upstream

    # ---- torsion derivative w.r.t. positions (forces) ----
    # Standard dihedral gradient (see e.g. OpenMM / many MD codes):
    # Let:
    #   t1 = n1 / |n1|^2
    #   t2 = n2 / |n2|^2
    # Then:
    #   dphi/di =  (|b2|) * t1
    #   dphi/dl = -(|b2|) * t2
    #   dphi/dj = -dphi/di + ( (b1·b2)/|b2|^2 ) * dphi/di - ( (b3·b2)/|b2|^2 ) * dphi/dl
    #   dphi/dk = -dphi/dl - ( (b1·b2)/|b2|^2 ) * dphi/di + ( (b3·b2)/|b2|^2 ) * dphi/dl
    #
    # Signs depend on the exact phi convention; this matches the atan2(y,x) above with n2=b3xb2.

    inv_n1 = 1.0 / n1_sq
    inv_n2 = 1.0 / n2_sq
    # dphi/di
    dphidx_i =  b2_len * n1x * inv_n1
    dphidy_i =  b2_len * n1y * inv_n1
    dphidz_i =  b2_len * n1z * inv_n1
    # dphi/dl
    dphidx_l = -b2_len * n2x * inv_n2
    dphidy_l = -b2_len * n2y * inv_n2
    dphidz_l = -b2_len * n2z * inv_n2

    b1_dot_b2 = b1x*b2x + b1y*b2y + b1z*b2z
    b3_dot_b2 = b3x*b2x + b3y*b2y + b3z*b2z
    inv_b2sq = 1.0 / b2_sq
    a = b1_dot_b2 * inv_b2sq
    c_ = b3_dot_b2 * inv_b2sq

    # dphi/dj
    dphidx_j = -dphidx_i + a * dphidx_i - c_ * dphidx_l
    dphidy_j = -dphidy_i + a * dphidy_i - c_ * dphidy_l
    dphidz_j = -dphidz_i + a * dphidz_i - c_ * dphidz_l
    # dphi/dk
    dphidx_k = -dphidx_l - a * dphidx_i + c_ * dphidx_l
    dphidy_k = -dphidy_l - a * dphidy_i + c_ * dphidy_l
    dphidz_k = -dphidz_l - a * dphidz_i + c_ * dphidz_l

    # dE/dpos = dE/dphi * dphi/dpos
    gi_x = dEdphi * dphidx_i 
    gi_y = dEdphi * dphidy_i 
    gi_z = dEdphi * dphidz_i
    
    gj_x = dEdphi * dphidx_j 
    gj_y = dEdphi * dphidy_j 
    gj_z = dEdphi * dphidz_j
    
    gk_x = dEdphi * dphidx_k 
    gk_y = dEdphi * dphidy_k 
    gk_z = dEdphi * dphidz_k
    
    gl_x = dEdphi * dphidx_l 
    gl_y = dEdphi * dphidy_l 
    gl_z = dEdphi * dphidz_l


    tl.atomic_add(grad_pos_ptr + i3 + 0 , gi_x, mask=mask)
    tl.atomic_add(grad_pos_ptr + i3 + 1 , gi_y, mask=mask)
    tl.atomic_add(grad_pos_ptr + i3 + 2 , gi_z, mask=mask)

    tl.atomic_add(grad_pos_ptr + j3 + 0 , gj_x, mask=mask)
    tl.atomic_add(grad_pos_ptr + j3 + 1 , gj_y, mask=mask)
    tl.atomic_add(grad_pos_ptr + j3 + 2 , gj_z, mask=mask)

    tl.atomic_add(grad_pos_ptr + k3 + 0 , gk_x, mask=mask)
    tl.atomic_add(grad_pos_ptr + k3 + 1 , gk_y, mask=mask)
    tl.atomic_add(grad_pos_ptr + k3 + 2 , gk_z, mask=mask)
    
    tl.atomic_add(grad_pos_ptr + l3 + 0 , gl_x, mask=mask)
    tl.atomic_add(grad_pos_ptr + l3 + 1 , gl_y, mask=mask)
    tl.atomic_add(grad_pos_ptr + l3 + 2 , gl_z, mask=mask)
    


@triton_op("mlcg_kernels::flash_dihedral", mutates_args={})
@ensure_contiguous
def flash_dihedral(
    pos : torch.Tensor, 
    atom_types : torch.Tensor, 
    index_mapping : torch.Tensor,  
    mapping_batch : torch.Tensor, 
    k1 : torch.Tensor, 
    k2 : torch.Tensor, 
    v_0 : torch.Tensor,
    deg : int,
    num_graphs: int,
)-> torch.Tensor:
    """
    pos:      [B, N, 3] float32
    types:    [B, N] int32
    idx_*:    [T] int32
    batch_id: [T] int32
    k1,k2:    [n_types^4, deg] float32
    """
    assert pos.is_cuda and k1.is_cuda and k2.is_cuda
    E = index_mapping.shape[0]
    out_E = torch.empty((E,), device=pos.device, dtype=torch.float32).contiguous()
    n_types = k1.shape[1]
    
    dbg_ptr = torch.zeros((E,4,), device=pos.device, dtype=torch.int64).contiguous()-6
    wrap_triton(flash_dihedral_fwd_kernel)[lambda meta: (triton.cdiv(E, meta["BLOCK"]),)](
        pos, 
        atom_types,
        index_mapping,
        k1,
        k2,
        v_0,
        out_E_ptr=out_E,
        DBG_ptr=dbg_ptr,
        E=E,
        deg=deg, 
        n_types=n_types, 
        EDGES_STRIDE0=index_mapping.stride(0),
        EDGES_STRIDE1=index_mapping.stride(1),
        stride_km=k1.stride(0),
        stride_ki=k1.stride(1), 
        stride_kj=k1.stride(2), 
        stride_kk=k1.stride(3), 
        stride_kl=k1.stride(4), 
    )
    y = torch.zeros((num_graphs,), device=pos.device, dtype=torch.float32).contiguous()
    y.index_add_(0, mapping_batch, out_E)
    return y

def setup_context_flash_dihedral(ctx,inputs,output):
    (
        pos,
        atom_types,
        index_mapping,
        mapping_batch,
        k1,
        k2,
        v_0,
        deg,
        num_graphs
    ) = inputs
    ctx.save_for_backward(
        pos,
        atom_types,
        index_mapping,
        mapping_batch,
        k1,
        k2,
    )
    ctx.v_0=v_0
    ctx.deg = deg
    ctx.num_graphs = num_graphs
    

def backward_flash_dihedral(ctx, grad_out_E):
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
        grad_pos = torch.zeros_like(pos, dtype=torch.float32)
        E = index_mapping.shape[0]
        wrap_triton(flash_dihedral_bwd_kernel)[lambda meta: (triton.cdiv(E, meta["BLOCK"]),)](
            pos, 
            atom_types,
            index_mapping,
            k1, 
            k2,
            E,
            grad_out_E.contiguous(),
            deg=deg, 
            grad_pos_ptr=grad_pos,
            EDGES_STRIDE0=index_mapping.stride(0),
            EDGES_STRIDE1=index_mapping.stride(1),
            stride_km=k1.stride(0),
            stride_ki=k1.stride(1), 
            stride_kj=k1.stride(2), 
            stride_kk=k1.stride(3), 
            stride_kl=k1.stride(4)
        )

    # Only grad wrt pos
    return grad_pos, None, None, None, None, None, None, None, None

flash_dihedral.register_autograd(
    backward_flash_dihedral,
    setup_context=setup_context_flash_dihedral,
)

@flash_dihedral.register_kernel("cpu")
def cpu_flash_dihedral(
    pos:torch.Tensor, 
    atom_types:torch.Tensor, 
    index_mapping:torch.Tensor, 
    mapping_batch:torch.Tensor, 
    k:torch.Tensor, 
    x_0:torch.Tensor, 
    num_graphs: int,
) -> torch.Tensor:
    interaction_types = tuple(
            atom_types[index_mapping[ii]] for ii in range(2)
        )
    distances = compute_angles_cos(pos,index_mapping)
    xdiff = (distances - x_0[interaction_types])
    y = k[interaction_types]*xdiff*xdiff
    y = scatter(y, mapping_batch, dim=0, reduce="sum", dim_size=num_graphs)
    return y 

