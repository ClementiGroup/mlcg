import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton
from torch_geometric.utils import scatter

from ...utils import ensure_contiguous
from .....geometry import compute_distances


@triton.jit
def repulsion_edge_fwd_kernel(
    pos_ptr,  # *fp16/fp32, [N,3]
    types_ptr,  # *i32/i64,   [N]
    edges_ptr,  # *i32,       [E,2] (may be non-contiguous)
    sigma_ptr,  # *fp16/fp32, [T,T]
    eedge_ptr,  # *fp32,      [E]
    E: tl.constexpr,
    eps: tl.constexpr,
    TYPE_DTYPE: tl.constexpr,  # 32 or 64
    SIG_STRIDE0: tl.constexpr,
    EDGES_STRIDE0: tl.constexpr,
    EDGES_STRIDE1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    k = pid * BLOCK + tl.arange(0, BLOCK)
    mask = k < E

    off_i = k * EDGES_STRIDE0 + 0 * EDGES_STRIDE1
    off_j = k * EDGES_STRIDE0 + 1 * EDGES_STRIDE1
    i = tl.load(edges_ptr + off_i, mask=mask, other=0).to(tl.int32)
    j = tl.load(edges_ptr + off_j, mask=mask, other=0).to(tl.int32)

    if TYPE_DTYPE == 32:
        ti = tl.load(types_ptr + i, mask=mask, other=0).to(tl.int32)
        tj = tl.load(types_ptr + j, mask=mask, other=0).to(tl.int32)
    else:
        ti = tl.load(types_ptr + i, mask=mask, other=0).to(tl.int64)
        tj = tl.load(types_ptr + j, mask=mask, other=0).to(tl.int64)

    sig = tl.load(sigma_ptr + ti * SIG_STRIDE0 + tj, mask=mask, other=0.0).to(
        tl.float32
    )

    # positions
    i3 = i * 3
    j3 = j * 3
    xi0 = tl.load(pos_ptr + i3 + 0, mask=mask, other=0.0).to(tl.float32)
    xi1 = tl.load(pos_ptr + i3 + 1, mask=mask, other=0.0).to(tl.float32)
    xi2 = tl.load(pos_ptr + i3 + 2, mask=mask, other=0.0).to(tl.float32)
    xj0 = tl.load(pos_ptr + j3 + 0, mask=mask, other=0.0).to(tl.float32)
    xj1 = tl.load(pos_ptr + j3 + 1, mask=mask, other=0.0).to(tl.float32)
    xj2 = tl.load(pos_ptr + j3 + 2, mask=mask, other=0.0).to(tl.float32)

    dx0 = xi0 - xj0
    dx1 = xi1 - xj1
    dx2 = xi2 - xj2

    r2 = dx0 * dx0 + dx1 * dx1 + dx2 * dx2
    inv_r2 = 1.0 / (r2 + eps)
    inv_r6 = inv_r2 * inv_r2 * inv_r2

    sig6 = sig * sig
    sig6 = sig6 * sig6 * sig  # sig^5
    sig6 = sig6 * sig  # sig^6

    e = sig6 * inv_r6
    tl.store(eedge_ptr + k, e, mask=mask)


@triton.jit
def scatter_sum_edges_kernel(
    eedge_ptr,  # *fp32, [E]
    edge_batch_ptr,  # *i32/i64, [E]
    out_ptr,  # *fp32, [B]
    E: tl.constexpr,
    B: tl.constexpr,
    BATCH_DTYPE: tl.constexpr,  # 32 or 64
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    k = pid * BLOCK + tl.arange(0, BLOCK)
    mask = k < E
    e = tl.load(eedge_ptr + k, mask=mask, other=0.0).to(tl.float32)

    if BATCH_DTYPE == 32:
        b = tl.load(edge_batch_ptr + k, mask=mask, other=0).to(tl.int32)
    else:
        b = tl.load(edge_batch_ptr + k, mask=mask, other=0).to(tl.int64)

    valid = mask & (b >= 0) & (b < B)
    tl.atomic_add(out_ptr + b, e, mask=valid)


@triton.jit
def repulsion_pos_bwd_kernel(
    pos_ptr,  # *fp16/fp32, [N,3] (read)
    types_ptr,  # *i32/i64,   [N]
    edges_ptr,  # *i32,       [E,2]
    sigma_ptr,  # *fp16/fp32, [T,T]
    edge_batch_ptr,  # *i32/i64,   [E]
    grad_y_ptr,  # *fp32,      [B]
    grad_pos_ptr,  # *fp32,      [N,3] (atomic add)
    E: tl.constexpr,
    B: tl.constexpr,
    eps: tl.constexpr,
    TYPE_DTYPE: tl.constexpr,  # 32 or 64
    BATCH_DTYPE: tl.constexpr,  # 32 or 64
    SIG_STRIDE0: tl.constexpr,
    EDGES_STRIDE0: tl.constexpr,
    EDGES_STRIDE1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    k = pid * BLOCK + tl.arange(0, BLOCK)
    mask = k < E

    off_i = k * EDGES_STRIDE0 + 0 * EDGES_STRIDE1
    off_j = k * EDGES_STRIDE0 + 1 * EDGES_STRIDE1
    i = tl.load(edges_ptr + off_i, mask=mask, other=0).to(tl.int64)
    j = tl.load(edges_ptr + off_j, mask=mask, other=0).to(tl.int64)

    # grad multiplier per edge from upstream grad_y[batch]
    if BATCH_DTYPE == 32:
        b = tl.load(edge_batch_ptr + k, mask=mask, other=0).to(tl.int32)
    else:
        b = tl.load(edge_batch_ptr + k, mask=mask, other=0).to(tl.int64)
    gy = tl.load(grad_y_ptr + b, mask=mask & (b >= 0) & (b < B), other=0.0).to(
        tl.float32
    )

    if TYPE_DTYPE == 32:
        ti = tl.load(types_ptr + i, mask=mask, other=0).to(tl.int32)
        tj = tl.load(types_ptr + j, mask=mask, other=0).to(tl.int32)
    else:
        ti = tl.load(types_ptr + i, mask=mask, other=0).to(tl.int64)
        tj = tl.load(types_ptr + j, mask=mask, other=0).to(tl.int64)

    sig = tl.load(sigma_ptr + ti * SIG_STRIDE0 + tj, mask=mask, other=0.0).to(
        tl.float32
    )

    # positions
    i3 = i * 3
    j3 = j * 3
    xi0 = tl.load(pos_ptr + i3 + 0, mask=mask, other=0.0).to(tl.float32)
    xi1 = tl.load(pos_ptr + i3 + 1, mask=mask, other=0.0).to(tl.float32)
    xi2 = tl.load(pos_ptr + i3 + 2, mask=mask, other=0.0).to(tl.float32)
    xj0 = tl.load(pos_ptr + j3 + 0, mask=mask, other=0.0).to(tl.float32)
    xj1 = tl.load(pos_ptr + j3 + 1, mask=mask, other=0.0).to(tl.float32)
    xj2 = tl.load(pos_ptr + j3 + 2, mask=mask, other=0.0).to(tl.float32)

    dx0 = xi0 - xj0
    dx1 = xi1 - xj1
    dx2 = xi2 - xj2

    r2 = dx0 * dx0 + dx1 * dx1 + dx2 * dx2
    inv_r2 = 1.0 / (r2 + eps)
    inv_r4 = inv_r2 * inv_r2
    inv_r8 = inv_r4 * inv_r4  # (r^2)^-4

    # sig^6
    sig6 = sig * sig
    sig6 = sig6 * sig6 * sig
    sig6 = sig6 * sig

    # dE/dxi = (-6 * sig^6 * (r^2)^-4) * dx
    scale = (-6.0) * sig6 * inv_r8
    scale = scale * gy

    g0 = scale * dx0
    g1 = scale * dx1
    g2 = scale * dx2

    # atomic add to grad_pos for i and j
    tl.atomic_add(grad_pos_ptr + i3 + 0, g0, mask=mask)
    tl.atomic_add(grad_pos_ptr + i3 + 1, g1, mask=mask)
    tl.atomic_add(grad_pos_ptr + i3 + 2, g2, mask=mask)

    tl.atomic_add(grad_pos_ptr + j3 + 0, -g0, mask=mask)
    tl.atomic_add(grad_pos_ptr + j3 + 1, -g1, mask=mask)
    tl.atomic_add(grad_pos_ptr + j3 + 2, -g2, mask=mask)


# class RepulsionSigmaOverR6Fn(torch.autograd.Function):
#    @staticmethod
@triton_op("mlcg_kernels::flash_repulsion", mutates_args={})
@ensure_contiguous
def flash_repulsion(
    pos: torch.Tensor,
    atom_types: torch.Tensor,
    index_mapping: torch.Tensor,
    mapping_batch: torch.Tensor,
    sigma: torch.Tensor,
    num_graphs: int,
    eps: float = 1e-12,
    block: int = 256,
    num_warps: int = 4,
) -> torch.Tensor:
    # assert index_mapping.dtype == torch.int32, "index_mapping must be int32 per your requirement"
    E = index_mapping.shape[0]

    eedge = torch.empty(
        (E,), device=pos.device, dtype=torch.float32
    ).contiguous()
    y = torch.zeros(
        (num_graphs,), device=pos.device, dtype=torch.float32
    ).contiguous()

    grid = (triton.cdiv(E, block),)
    wrap_triton(repulsion_edge_fwd_kernel)[grid](
        pos,
        atom_types,
        index_mapping,
        sigma,
        eedge,
        E=E,
        eps=eps,
        TYPE_DTYPE=32 if atom_types.dtype == torch.int32 else 64,
        SIG_STRIDE0=sigma.stride(0),
        EDGES_STRIDE0=index_mapping.stride(0),
        EDGES_STRIDE1=index_mapping.stride(1),
        BLOCK=block,
        num_warps=num_warps,
    )
    wrap_triton(scatter_sum_edges_kernel)[grid](
        eedge,
        mapping_batch,
        y,
        E=E,
        B=num_graphs,
        BATCH_DTYPE=32 if mapping_batch.dtype == torch.int32 else 64,
        BLOCK=block,
        num_warps=num_warps,
    )

    return y


def setup_context_flash_repulsion(ctx, inputs, output):
    # Save for backward (only what we need for grad w.r.t pos)
    (
        pos,
        atom_types,
        index_mapping,
        mapping_batch,
        sigma,
        num_graphs,
        eps,
        block,
        num_warps,
    ) = inputs
    ctx.save_for_backward(pos, atom_types, index_mapping, mapping_batch, sigma)
    ctx.num_graphs = num_graphs
    ctx.eps = eps
    ctx.block = block
    ctx.num_warps = num_warps


def backward_flash_repulsion(ctx, grad_y):
    (
        pos,
        atom_types,
        index_mapping,
        mapping_batch,
        sigma,
    ) = ctx.saved_tensors

    grad_y = grad_y.contiguous().to(torch.float32)

    grad_pos = torch.zeros(
        (pos.shape[0], 3), device=pos.device, dtype=torch.float32
    ).contiguous()

    E = index_mapping.shape[0]
    block = ctx.block
    grid = (triton.cdiv(E, block),)

    wrap_triton(repulsion_pos_bwd_kernel)[grid](
        pos,
        atom_types,
        index_mapping,
        sigma,
        mapping_batch,
        grad_y,
        grad_pos,
        E=E,
        B=ctx.num_graphs,
        eps=ctx.eps,
        TYPE_DTYPE=32 if atom_types.dtype == torch.int32 else 64,
        BATCH_DTYPE=32 if mapping_batch.dtype == torch.int32 else 64,
        SIG_STRIDE0=sigma.stride(0),
        EDGES_STRIDE0=index_mapping.stride(0),
        EDGES_STRIDE1=index_mapping.stride(1),
        BLOCK=block,
        num_warps=ctx.num_warps,
    )

    # Return gradients for each forward input: (pos, atom_types, index_mapping, mapping_batch, sigma, num_graphs, eps, block, num_warps)
    return grad_pos, None, None, None, None, None, None, None, None


flash_repulsion.register_autograd(
    backward_flash_repulsion,
    setup_context=setup_context_flash_repulsion,
)


@flash_repulsion.register_kernel("cpu")
def cpu_flash_repulsion(
    pos: torch.Tensor,
    atom_types: torch.Tensor,
    index_mapping: torch.Tensor,
    mapping_batch: torch.Tensor,
    sigma: torch.Tensor,
    num_graphs: int,
    eps: float = 1e-12,
    block: int = 256,
    num_warps: int = 4,
) -> torch.Tensor:
    interaction_types = tuple(atom_types[index_mapping[ii]] for ii in range(2))
    distances = compute_distances(pos, index_mapping)
    special = sigma[interaction_types] / distances
    y = special * special * special
    y = scatter(y, mapping_batch, dim=0, reduce="sum", dim_size=num_graphs)
    return y
