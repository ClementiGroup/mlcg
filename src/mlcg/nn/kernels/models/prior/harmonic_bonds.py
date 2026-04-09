import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton
from torch_geometric.utils import scatter

from ...utils import ensure_contiguous
from .....geometry import compute_distances


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
def harmonic_bonds_edge_fwd_kernel(
    pos_ptr,  # *fp16/fp32, [N,3]
    atom_types_ptr,  # *i32/i64,   [N]
    index_mapping_ptr,  # *i32,       [E,2] (may be non-contiguous)
    k_ptr,
    x_ptr,  # *fp16/fp32, [T,T]
    eedge_ptr,  # *fp32,      [E]
    E,
    TYPE_DTYPE: tl.constexpr,  # 32 or 64
    K_STRIDE0: tl.constexpr,
    X_0_STRIDE0: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    blk_idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = blk_idx < E

    off_i = blk_idx
    off_j = blk_idx + E
    i = tl.load(index_mapping_ptr + off_i, mask=mask, other=0).to(
        tl.int32
    )  # FIXME: indeces are int64 do we want them in int32?
    j = tl.load(index_mapping_ptr + off_j, mask=mask, other=0).to(tl.int32)

    if TYPE_DTYPE == 32:
        ti = tl.load(atom_types_ptr + i, mask=mask, other=0).to(tl.int32)
        tj = tl.load(atom_types_ptr + j, mask=mask, other=0).to(tl.int32)
    else:
        ti = tl.load(atom_types_ptr + i, mask=mask, other=0).to(tl.int64)
        tj = tl.load(atom_types_ptr + j, mask=mask, other=0).to(tl.int64)

    k_ener = tl.load(k_ptr + ti * K_STRIDE0 + tj, mask=mask, other=0.0).to(
        tl.float32
    )
    x_0 = tl.load(x_ptr + ti * X_0_STRIDE0 + tj, mask=mask, other=0.0).to(
        tl.float32
    )

    # positions
    i3 = i * 3
    j3 = j * 3
    xi = tl.load(pos_ptr + i3 + 0, mask=mask, other=0.0).to(tl.float32)
    yi = tl.load(pos_ptr + i3 + 1, mask=mask, other=0.0).to(tl.float32)
    zi = tl.load(pos_ptr + i3 + 2, mask=mask, other=0.0).to(tl.float32)
    xj = tl.load(pos_ptr + j3 + 0, mask=mask, other=0.0).to(tl.float32)
    xy = tl.load(pos_ptr + j3 + 1, mask=mask, other=0.0).to(tl.float32)
    zj = tl.load(pos_ptr + j3 + 2, mask=mask, other=0.0).to(tl.float32)

    dx = xi - xj
    gy = yi - xy
    gz = zi - zj

    x = tl.sqrt(dx * dx + gy * gy + gz * gz)
    xdiff = x - x_0
    e = k_ener * xdiff * xdiff
    tl.store(eedge_ptr + blk_idx, e, mask=mask)


@triton_op("mlcg_kernels::flash_harmonic_bonds", mutates_args={})
@ensure_contiguous
def flash_harmonic_bonds(
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
    y = torch.zeros((num_graphs,), device=pos.device, dtype=torch.float32)

    # grid = (triton.cdiv(E, block),)
    wrap_triton(harmonic_bonds_edge_fwd_kernel)[
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
        X_0_STRIDE0=x_0.stride(0),
    )

    return y.index_add_(0, mapping_batch, eedge)


def setup_context_flash_harmonic_bonds(ctx, inputs, output):
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


def backward_flash_harmonic_bonds(ctx, grad_y):
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
        grad_pos = harmonic_bonds_pos_bwd(
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


flash_harmonic_bonds.register_autograd(
    backward_flash_harmonic_bonds,
    setup_context=setup_context_flash_harmonic_bonds,
)


@flash_harmonic_bonds.register_kernel("cpu")
def cpu_flash_harmonic_bonds(
    pos: torch.Tensor,
    atom_types: torch.Tensor,
    index_mapping: torch.Tensor,
    mapping_batch: torch.Tensor,
    k: torch.Tensor,
    x_0: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:

    interaction_types = tuple(atom_types[index_mapping[ii]] for ii in range(2))
    distances = compute_distances(pos, index_mapping)
    xdiff = distances - x_0[interaction_types]
    y_edge = k[interaction_types] * xdiff * xdiff
    y = torch.zeros(num_graphs, device=y_edge.device, dtype=y_edge.dtype)
    return y.index_add_(0, mapping_batch, y_edge)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 64}, num_warps=2),
        triton.Config({"BLOCK": 128}, num_warps=4),
        triton.Config({"BLOCK": 256}, num_warps=4),
        triton.Config({"BLOCK": 512}, num_warps=8),
        triton.Config({"BLOCK": 1024}, num_warps=8),
    ],
    key=[],
    reset_to_zero=["grad_pos_ptr"],
)
@triton.jit
def harmonic_bonds_pos_bwd_kernel(
    pos_ptr,  # *fp16/fp32, [N,3] (read)
    atom_types_ptr,  # *i32/i64,   [N]
    index_mapping_ptr,  # *i32,       [2,E]
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
    X_0_STRIDE0: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    blk_idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = blk_idx < E

    off_i = blk_idx
    off_j = blk_idx + E

    i = tl.load(index_mapping_ptr + off_i, mask=mask, other=0).to(
        tl.int32
    )  # FIXME: indeces are int64 do we want them in int32?
    j = tl.load(index_mapping_ptr + off_j, mask=mask, other=0).to(tl.int32)

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

    if TYPE_DTYPE == 32:
        ti = tl.load(atom_types_ptr + i, mask=mask, other=0).to(tl.int32)
        tj = tl.load(atom_types_ptr + j, mask=mask, other=0).to(tl.int32)
    else:
        ti = tl.load(atom_types_ptr + i, mask=mask, other=0).to(tl.int64)
        tj = tl.load(atom_types_ptr + j, mask=mask, other=0).to(tl.int64)

    k_ener = tl.load(k_ptr + ti * K_STRIDE0 + tj, mask=mask, other=0.0).to(
        tl.float32
    )
    x_0 = tl.load(x_ptr + ti * X_0_STRIDE0 + tj, mask=mask, other=0.0).to(
        tl.float32
    )

    # positions
    i3 = i * 3
    j3 = j * 3
    xi = tl.load(pos_ptr + i3 + 0, mask=mask, other=0.0).to(tl.float32)
    yi = tl.load(pos_ptr + i3 + 1, mask=mask, other=0.0).to(tl.float32)
    zi = tl.load(pos_ptr + i3 + 2, mask=mask, other=0.0).to(tl.float32)

    xj = tl.load(pos_ptr + j3 + 0, mask=mask, other=0.0).to(tl.float32)
    xy = tl.load(pos_ptr + j3 + 1, mask=mask, other=0.0).to(tl.float32)
    zj = tl.load(pos_ptr + j3 + 2, mask=mask, other=0.0).to(tl.float32)

    dx = xi - xj
    dy = yi - xy
    dz = zi - zj

    x = tl.sqrt(dx * dx + dy * dy + dz * dz)
    xdiff = x - x_0
    scale = 2 * k_ener * xdiff / x
    scale = scale * gy

    gx = scale * dx
    g1 = scale * dy
    g2 = scale * dz

    # atomic add to grad_pos for i and j
    tl.atomic_add(grad_pos_ptr + i3 + 0, gx, mask=mask)
    tl.atomic_add(grad_pos_ptr + i3 + 1, g1, mask=mask)
    tl.atomic_add(grad_pos_ptr + i3 + 2, g2, mask=mask)

    tl.atomic_add(grad_pos_ptr + j3 + 0, -gx, mask=mask)
    tl.atomic_add(grad_pos_ptr + j3 + 1, -g1, mask=mask)
    tl.atomic_add(grad_pos_ptr + j3 + 2, -g2, mask=mask)


@triton_op("mlcg_kernels::harmonic_bonds_pos_bwd", mutates_args={})
@ensure_contiguous
def harmonic_bonds_pos_bwd(
    pos: torch.Tensor,
    atom_types: torch.Tensor,
    index_mapping: torch.Tensor,
    mapping_batch: torch.Tensor,
    k: torch.Tensor,
    x_0: torch.Tensor,
    grad_y: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    grad_pos = torch.zeros_like(pos).contiguous()
    E = index_mapping.shape[1]

    wrap_triton(harmonic_bonds_pos_bwd_kernel)[
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
        X_0_STRIDE0=x_0.stride(0),
    )

    return grad_pos


@harmonic_bonds_pos_bwd.register_kernel("cpu")
def cpu_harmonic_bonds_pos_bwd(
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

    i = index_mapping[0].long()
    j = index_mapping[1].long()
    b = mapping_batch.long()

    ti = atom_types[i].long()
    tj = atom_types[j].long()

    k_ener = k[ti, tj].float()
    x0 = x_0[ti, tj].float()

    dx = pos[i] - pos[j]  # [E, 3]
    r = dx.norm(dim=1).clamp(min=1e-12)  # [E]
    xdiff = r - x0  # [E]

    gy = grad_y[b]  # [E]
    scale = (2.0 * k_ener * xdiff / r) * gy  # [E]

    g = scale.unsqueeze(1) * dx  # [E, 3]

    grad_pos.index_add_(0, i, g)
    grad_pos.index_add_(0, j, -g)

    return grad_pos
