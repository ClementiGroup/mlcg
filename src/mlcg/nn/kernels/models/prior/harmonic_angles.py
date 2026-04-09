import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from ...utils import ensure_contiguous
from .....geometry import compute_angles_cos


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
def harmonic_angles_edge_fwd_kernel(
    pos_ptr,  # *fp16/fp32, [N,3]
    atom_types_ptr,  # *i32/i64,   [N]
    index_mapping_ptr,  # *i32,       [E,2] (may be non-contiguous)
    k_ptr,
    x_ptr,  # *fp16/fp32, [T,T]
    eedge_ptr,  # *fp32,      [E]
    E, 
    TYPE_DTYPE: tl.constexpr,  # 32 or 64
    K_STRIDE0: tl.constexpr,
    K_STRIDE1: tl.constexpr,
    X_0_STRIDE0: tl.constexpr,
    X_0_STRIDE1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    blk_idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = blk_idx < E

    off_i = blk_idx
    off_j = blk_idx + E
    off_k = blk_idx + 2 * E

    i = tl.load(index_mapping_ptr + off_i, mask=mask, other=0).to(tl.int32)
    j = tl.load(index_mapping_ptr + off_j, mask=mask, other=0).to(tl.int32)
    k = tl.load(index_mapping_ptr + off_k, mask=mask, other=0).to(tl.int32)

    if TYPE_DTYPE == 32:
        ti = tl.load(atom_types_ptr + i, mask=mask, other=0).to(tl.int32)
        tj = tl.load(atom_types_ptr + j, mask=mask, other=0).to(tl.int32)
        tk = tl.load(atom_types_ptr + k, mask=mask, other=0).to(tl.int32)
    else:
        ti = tl.load(atom_types_ptr + i, mask=mask, other=0).to(tl.int64)
        tj = tl.load(atom_types_ptr + j, mask=mask, other=0).to(tl.int64)
        tk = tl.load(atom_types_ptr + k, mask=mask, other=0).to(tl.int64)

    k_ener = tl.load(
        k_ptr + ti * K_STRIDE0 + tj * K_STRIDE1 + tk, mask=mask, other=0.0
    ).to(tl.float32)
    x_0 = tl.load(
        x_ptr + ti * X_0_STRIDE0 + tj * X_0_STRIDE1 + tk, mask=mask, other=0.0
    ).to(tl.float32)

    # positions
    i3 = i * 3
    j3 = j * 3
    k3 = k * 3

    xi = tl.load(pos_ptr + i3 + 0, mask=mask, other=0.0).to(tl.float32)
    yi = tl.load(pos_ptr + i3 + 1, mask=mask, other=0.0).to(tl.float32)
    zi = tl.load(pos_ptr + i3 + 2, mask=mask, other=0.0).to(tl.float32)

    xj = tl.load(pos_ptr + j3 + 0, mask=mask, other=0.0).to(tl.float32)
    yj = tl.load(pos_ptr + j3 + 1, mask=mask, other=0.0).to(tl.float32)
    zj = tl.load(pos_ptr + j3 + 2, mask=mask, other=0.0).to(tl.float32)

    xk = tl.load(pos_ptr + k3 + 0, mask=mask, other=0.0).to(tl.float32)
    yk = tl.load(pos_ptr + k3 + 1, mask=mask, other=0.0).to(tl.float32)
    zk = tl.load(pos_ptr + k3 + 2, mask=mask, other=0.0).to(tl.float32)

    dijx = xi - xj
    dijy = yi - yj
    dijz = zi - zj

    dkjx = xk - xj
    dkjy = yk - yj
    dkjz = zk - zj

    # norms
    a2 = dijx * dijx + dijy * dijy + dijz * dijz
    b2 = dkjx * dkjx + dkjy * dkjy + dkjz * dkjz
    A = tl.sqrt(a2)
    B = tl.sqrt(b2)

    invAB = 1.0 / (A * B)
    dot = dijx * dkjx + dijy * dkjy + dijz * dkjz
    cosang = dot * invAB
    xdiff = cosang - x_0
    e = k_ener * xdiff * xdiff
    tl.store(eedge_ptr + blk_idx, e, mask=mask)


@triton_op("mlcg_kernels::flash_harmonic_angles", mutates_args={})
@ensure_contiguous
def flash_harmonic_angles(
    pos: torch.Tensor,
    atom_types: torch.Tensor,
    index_mapping: torch.Tensor,
    mapping_batch: torch.Tensor,
    k: torch.Tensor,
    x_0: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    # assert index_mapping.dtype == torch.int32, "index_mapping must be int32 per your requirement"
    E = index_mapping.shape[1]
    eedge = torch.empty(
        (E,), device=pos.device, dtype=torch.float32
    ).contiguous()
    y = torch.zeros(
        (num_graphs,), device=pos.device, dtype=torch.float32
    ).contiguous()

    wrap_triton(harmonic_angles_edge_fwd_kernel)[
        lambda meta: (triton.cdiv(E, meta["BLOCK"]),)
    ](
        pos,
        atom_types,
        index_mapping,
        k,
        x_0,
        eedge,
        E=E,
        TYPE_DTYPE=32 if atom_types.dtype == torch.int32 else 64,
        K_STRIDE0=k.stride(0),
        K_STRIDE1=k.stride(1),
        X_0_STRIDE0=x_0.stride(0),
        X_0_STRIDE1=x_0.stride(1),
    )
    return y.index_add_(0, mapping_batch, eedge)


def setup_context_flash_harmonic_angles(ctx, inputs, output):
    # Save for backward (only what we need for grad w.r.t pos)
    (
        pos,
        atom_types,
        index_mapping,
        mapping_batch,
        k,
        x_0,
        num_graphs,
    ) = inputs
    ctx.save_for_backward(pos, atom_types, index_mapping, mapping_batch, k, x_0)
    ctx.num_graphs = num_graphs


def backward_flash_harmonic_angles(ctx, grad_y):
    (
        pos,
        atom_types,
        index_mapping,
        mapping_batch,
        k,
        x_0,
    ) = ctx.saved_tensors

    grad_pos = None
    if ctx.needs_input_grad[0]:
        grad_pos = harmonic_angles_pos_bwd(
            pos,
            atom_types,
            index_mapping,
            mapping_batch,
            k,
            x_0,
            grad_y,
            ctx.num_graphs,
        )
    # Return gradients for each forward input
    return grad_pos, None, None, None, None, None, None


flash_harmonic_angles.register_autograd(
    backward_flash_harmonic_angles,
    setup_context=setup_context_flash_harmonic_angles,
)


@flash_harmonic_angles.register_kernel("cpu")
def cpu_flash_harmonic_angles(
    pos: torch.Tensor,
    atom_types: torch.Tensor,
    index_mapping: torch.Tensor,
    mapping_batch: torch.Tensor,
    k: torch.Tensor,
    x_0: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    interaction_types = tuple(atom_types[index_mapping[ii]] for ii in range(3))
    distances = compute_angles_cos(pos, index_mapping)
    xdiff = distances - x_0[interaction_types]
    ey = k[interaction_types] * xdiff * xdiff
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
def harmonic_angles_pos_bwd_kernel(
    pos_ptr,  # *fp16/fp32, [N,3] (read)
    atom_types_ptr,  # *i32/i64,   [N]
    index_mapping_ptr,  # *i32,       [E,3]
    mapping_batch_ptr,  # *i32/i64,   [E]
    k_ptr,
    x_ptr,
    grad_y_ptr,  # *fp32,      [B]
    grad_pos_ptr,  # *fp32,      [N,3] (atomic add)
    E,
    B,
    TYPE_DTYPE: tl.constexpr,  # 32 or 64
    BATCH_DTYPE: tl.constexpr,  # 32 or 64
    K_STRIDE0: tl.constexpr,
    K_STRIDE1: tl.constexpr,
    X_0_STRIDE0: tl.constexpr,
    X_0_STRIDE1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    blk_idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = blk_idx < E

    off_i = blk_idx
    off_j = blk_idx + E
    off_k = blk_idx + 2 * E

    i = tl.load(index_mapping_ptr + off_i, mask=mask, other=0).to(tl.int32)
    j = tl.load(index_mapping_ptr + off_j, mask=mask, other=0).to(tl.int32)
    k = tl.load(index_mapping_ptr + off_k, mask=mask, other=0).to(tl.int32)

    if TYPE_DTYPE == 32:
        ti = tl.load(atom_types_ptr + i, mask=mask, other=0).to(tl.int32)
        tj = tl.load(atom_types_ptr + j, mask=mask, other=0).to(tl.int32)
        tk = tl.load(atom_types_ptr + k, mask=mask, other=0).to(tl.int32)
    else:
        ti = tl.load(atom_types_ptr + i, mask=mask, other=0).to(tl.int64)
        tj = tl.load(atom_types_ptr + j, mask=mask, other=0).to(tl.int64)
        tk = tl.load(atom_types_ptr + k, mask=mask, other=0).to(tl.int64)

    # grad multiplier per edge from upstream grad_y[batch]
    if BATCH_DTYPE == 32:
        b = tl.load(mapping_batch_ptr + blk_idx, mask=mask, other=0).to(
            tl.int32
        )
    else:
        b = tl.load(mapping_batch_ptr + blk_idx, mask=mask, other=0).to(
            tl.int64
        )
    gy = tl.load(grad_y_ptr + b, mask=mask & (b >= 0) & (b < B), other=0.0).to(
        tl.float32
    )

    k_ener = tl.load(
        k_ptr + ti * K_STRIDE0 + tj * K_STRIDE1 + tk, mask=mask, other=0.0
    ).to(tl.float32)
    x_0 = tl.load(
        x_ptr + ti * X_0_STRIDE0 + tj * X_0_STRIDE1 + tk, mask=mask, other=0.0
    ).to(tl.float32)

    # positions
    i3 = i * 3
    j3 = j * 3
    k3 = k * 3

    xi = tl.load(pos_ptr + i3 + 0, mask=mask, other=0.0).to(tl.float32)
    yi = tl.load(pos_ptr + i3 + 1, mask=mask, other=0.0).to(tl.float32)
    zi = tl.load(pos_ptr + i3 + 2, mask=mask, other=0.0).to(tl.float32)

    xj = tl.load(pos_ptr + j3 + 0, mask=mask, other=0.0).to(tl.float32)
    yj = tl.load(pos_ptr + j3 + 1, mask=mask, other=0.0).to(tl.float32)
    zj = tl.load(pos_ptr + j3 + 2, mask=mask, other=0.0).to(tl.float32)

    xk = tl.load(pos_ptr + k3 + 0, mask=mask, other=0.0).to(tl.float32)
    yk = tl.load(pos_ptr + k3 + 1, mask=mask, other=0.0).to(tl.float32)
    zk = tl.load(pos_ptr + k3 + 2, mask=mask, other=0.0).to(tl.float32)

    dijx = xi - xj
    dijy = yi - yj
    dijz = zi - zj

    dkjx = xk - xj
    dkjy = yk - yj
    dkjz = zk - zj

    # norms
    a2 = dijx * dijx + dijy * dijy + dijz * dijz
    b2 = dkjx * dkjx + dkjy * dkjy + dkjz * dkjz
    A = tl.sqrt(a2)
    B = tl.sqrt(b2)

    invAB = 1.0 / (A * B)
    dot = dijx * dkjx + dijy * dkjy + dijz * dkjz
    cosang = dot * invAB
    xdiff = cosang - x_0
    de = 2 * k_ener * xdiff

    invA2 = 1.0 / (a2)  # ~ 1/A^2
    invB2 = 1.0 / (b2)  # ~ 1/B^2

    # dc/da = b/(AB) - c * a/(A^2)
    dcax = dkjx * invAB - cosang * dijx * invA2
    dcay = dkjy * invAB - cosang * dijy * invA2
    dcaz = dkjz * invAB - cosang * dijz * invA2

    # dc/db = a/(AB) - c * b/(B^2)
    dcbx = dijx * invAB - cosang * dkjx * invB2
    dcby = dijy * invAB - cosang * dkjy * invB2
    dcbz = dijz * invAB - cosang * dkjz * invB2

    # Multiply by upstream grad
    gxi = gy * de * dcax
    gyi = gy * de * dcay
    gzi = gy * de * dcaz

    gxk = gy * de * dcbx
    gyk = gy * de * dcby
    gzk = gy * de * dcbz

    gxj = -(gxi + gxk)
    gyj = -(gyi + gyk)
    gzj = -(gzi + gzk)

    # atomic add to grad_pos for i, j and k
    tl.atomic_add(grad_pos_ptr + i3 + 0, gxi, mask=mask)
    tl.atomic_add(grad_pos_ptr + i3 + 1, gyi, mask=mask)
    tl.atomic_add(grad_pos_ptr + i3 + 2, gzi, mask=mask)

    tl.atomic_add(grad_pos_ptr + j3 + 0, gxj, mask=mask)
    tl.atomic_add(grad_pos_ptr + j3 + 1, gyj, mask=mask)
    tl.atomic_add(grad_pos_ptr + j3 + 2, gzj, mask=mask)

    tl.atomic_add(grad_pos_ptr + k3 + 0, gxk, mask=mask)
    tl.atomic_add(grad_pos_ptr + k3 + 1, gyk, mask=mask)
    tl.atomic_add(grad_pos_ptr + k3 + 2, gzk, mask=mask)


@triton_op("mlcg_kernels::harmonic_angles_pos_bwd", mutates_args={})
@ensure_contiguous
def harmonic_angles_pos_bwd(
    pos: torch.Tensor,
    atom_types: torch.Tensor,
    index_mapping: torch.Tensor,
    mapping_batch: torch.Tensor,
    k: torch.Tensor,
    x_0: torch.Tensor,
    grad_y: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:

    grad_pos = torch.zeros(
        (pos.shape[0], 3), device=pos.device, dtype=torch.float32
    ).contiguous()
    E = index_mapping.shape[1]
    wrap_triton(harmonic_angles_pos_bwd_kernel)[
        lambda meta: (triton.cdiv(E, meta["BLOCK"]),)
    ](
        pos,
        atom_types,
        index_mapping,
        mapping_batch,
        k,
        x_0,
        grad_y,
        grad_pos,
        E=E,
        B=num_graphs,
        TYPE_DTYPE=32 if atom_types.dtype == torch.int32 else 64,
        BATCH_DTYPE=32 if mapping_batch.dtype == torch.int32 else 64,
        K_STRIDE0=k.stride(0),
        K_STRIDE1=k.stride(1),
        X_0_STRIDE0=x_0.stride(0),
        X_0_STRIDE1=x_0.stride(1),
    )

    return grad_pos


@harmonic_angles_pos_bwd.register_kernel("cpu")
def cpu_bwd_flash_harmonic_angles(
    pos: torch.Tensor,
    atom_types: torch.Tensor,
    index_mapping: torch.Tensor,
    mapping_batch: torch.Tensor,
    k: torch.Tensor,
    x_0: torch.Tensor,
    grad_y: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    grad_pos = torch.zeros_like(pos)
    i_idx = index_mapping[0]
    j_idx = index_mapping[1]
    k_idx = index_mapping[2]
    b = mapping_batch

    dij = pos[i_idx] - pos[j_idx]
    dkj = pos[k_idx] - pos[j_idx]

    A = dij.norm(p=2, dim=1)
    B = dkj.norm(p=2, dim=1)
    invAB = (A * B).reciprocal_()
    invA2 = ((dij * dij).sum(dim=1)).reciprocal_()
    invB2 = ((dkj * dkj).sum(dim=1)).reciprocal_()
    dot = (dij * dkj).sum(dim=1)
    cosang = dot * invAB

    interaction_types = tuple(atom_types[index_mapping[ii]] for ii in range(3))
    k_ener = k[interaction_types].float()
    x0 = x_0[interaction_types].float()
    xdiff = cosang - x0  # [E]
    de = 2 * k_ener * xdiff

    dca = dkj * invAB.unsqueeze(1) - (
        dij * invA2.unsqueeze(1)
    ) * cosang.unsqueeze(1)
    dcb = dij * invAB.unsqueeze(1) - (
        dkj * invB2.unsqueeze(1)
    ) * cosang.unsqueeze(1)

    gy = grad_y[b]

    gi = (dca * gy.unsqueeze(1)) * de.unsqueeze(1)
    gk = (dcb * gy.unsqueeze(1)) * de.unsqueeze(1)

    gj = -(gi + gk)

    grad_pos.index_add_(0, i_idx, gi)
    grad_pos.index_add_(0, j_idx, gj)
    grad_pos.index_add_(0, k_idx, gk)

    return grad_pos
