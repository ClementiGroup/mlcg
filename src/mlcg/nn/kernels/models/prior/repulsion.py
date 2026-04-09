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
def repulsion_edge_fwd_kernel(
    pos_ptr,  # *fp16/fp32, [N,3]
    atom_types_ptr,  # *i32/i64,   [N]
    index_mapping_ptr,  # *i32,       [2, E]
    sigma_ptr,  # *fp16/fp32, [T,T]
    eedge_ptr,  # *fp32,      [E]
    E,
    eps: tl.constexpr,
    TYPE_DTYPE: tl.constexpr,  # 32 or 64
    SIG_STRIDE0: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Compute per-edge repulsion energy from atomic positions and types."""
    pid = tl.program_id(0)
    k = pid * BLOCK + tl.arange(0, BLOCK)
    mask = k < E

    off_i = k
    off_j = k + E
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
    sig6 = sig6 * sig6 * sig6

    e = sig6 * inv_r6
    tl.store(eedge_ptr + k, e, mask=mask)


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
) -> torch.Tensor:
    """Compute repulsion energy per graph using a Triton fused energy kernel.

    The repulsion energy for each pair of atoms i,j is given by:

    $$ E_{ij} = \left( \frac{\sigma_{t_i t_j}}{r_{ij}} \right)^6 $$

    where $t_i$ is the atom type of atom i, $\sigma_{t_i t_j}$ is the repulsion parameter,
    and $r_{ij}$ is the distance between atoms i and j.

    The total energy per graph is the sum over all pairs in that graph.
    """
    E = index_mapping.shape[1]

    eedge = torch.empty(
        (E,), device=pos.device, dtype=torch.float32
    ).contiguous()
    y = torch.zeros((num_graphs,), device=pos.device, dtype=torch.float32)

    wrap_triton(repulsion_edge_fwd_kernel)[
        lambda meta: (triton.cdiv(E, meta["BLOCK"]),)
    ](
        pos,
        atom_types,
        index_mapping,
        sigma,
        eedge,
        E=E,
        eps=eps,
        TYPE_DTYPE=32 if atom_types.dtype == torch.int32 else 64,
        SIG_STRIDE0=sigma.stride(0),
    )

    return y.index_add_(0, mapping_batch, eedge)


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
    ) = inputs
    ctx.save_for_backward(pos, atom_types, index_mapping, mapping_batch, sigma)
    ctx.num_graphs = num_graphs
    ctx.eps = eps


def backward_flash_repulsion(ctx, grad_y):
    (
        pos,
        atom_types,
        index_mapping,
        mapping_batch,
        sigma,
    ) = ctx.saved_tensors

    grad_pos = None

    if ctx.needs_input_grad[0]:
        grad_pos = repulsion_pos_bwd(
            pos,
            atom_types,
            index_mapping,
            mapping_batch,
            sigma,
            grad_y,
            ctx.num_graphs,
            ctx.eps,
        )

    return grad_pos, None, None, None, None, None, None


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
) -> torch.Tensor:
    """CPU reference implementation for repulsion energy aggregation."""
    interaction_types = tuple(atom_types[index_mapping[ii]] for ii in range(2))
    distances = compute_distances(pos, index_mapping)
    special = sigma[interaction_types] / distances
    y_edge = torch.pow(special, 6)
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
def repulsion_pos_bwd_kernel(
    pos_ptr,  # *fp16/fp32, [N,3] (read)
    atom_types_ptr,  # *i32/i64,   [N]
    index_mapping_ptr,  # *i32,       [2,E]
    mapping_batch_ptr,  # *i32/i64,   [E]
    sigma_ptr,  # *fp16/fp32, [T,T]
    grad_y_ptr,  # *fp32,      [B]
    grad_pos_ptr,  # *fp32,      [N,3] (atomic add)
    E,
    B,
    eps: tl.constexpr,
    TYPE_DTYPE: tl.constexpr,  # 32 or 64
    BATCH_DTYPE: tl.constexpr,  # 32 or 64
    SIG_STRIDE0: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Compute gradients of repulsion energy with respect to atomic positions."""
    pid = tl.program_id(0)
    k = pid * BLOCK + tl.arange(0, BLOCK)
    mask = k < E

    off_i = k
    off_j = k + E
    i = tl.load(index_mapping_ptr + off_i, mask=mask, other=0).to(
        tl.int32
    )  # FIXME: indeces are int64 do we want them in int32?
    j = tl.load(index_mapping_ptr + off_j, mask=mask, other=0).to(tl.int32)

    # grad multiplier per edge from upstream grad_y[batch]
    if BATCH_DTYPE == 32:
        b = tl.load(mapping_batch_ptr + k, mask=mask, other=0).to(tl.int32)
    else:
        b = tl.load(mapping_batch_ptr + k, mask=mask, other=0).to(tl.int64)
    gy = tl.load(grad_y_ptr + b, mask=mask & (b >= 0) & (b < B), other=0.0).to(
        tl.float32
    )

    if TYPE_DTYPE == 32:
        ti = tl.load(atom_types_ptr + i, mask=mask, other=0).to(tl.int32)
        tj = tl.load(atom_types_ptr + j, mask=mask, other=0).to(tl.int32)
    else:
        ti = tl.load(atom_types_ptr + i, mask=mask, other=0).to(tl.int64)
        tj = tl.load(atom_types_ptr + j, mask=mask, other=0).to(tl.int64)

    sig = tl.load(sigma_ptr + ti * SIG_STRIDE0 + tj, mask=mask, other=0.0).to(
        tl.float32
    )

    # positions
    i3 = i * 3
    j3 = j * 3
    xi = tl.load(pos_ptr + i3 + 0, mask=mask, other=0.0).to(tl.float32)
    yi = tl.load(pos_ptr + i3 + 1, mask=mask, other=0.0).to(tl.float32)
    zi = tl.load(pos_ptr + i3 + 2, mask=mask, other=0.0).to(tl.float32)
    xj = tl.load(pos_ptr + j3 + 0, mask=mask, other=0.0).to(tl.float32)
    yj = tl.load(pos_ptr + j3 + 1, mask=mask, other=0.0).to(tl.float32)
    zj = tl.load(pos_ptr + j3 + 2, mask=mask, other=0.0).to(tl.float32)

    dx = xi - xj
    dy = yi - yj
    dz = zi - zj

    r2 = dx * dx + dy * dy + dz * dz
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

    gx = scale * dx
    gy = scale * dy
    gz = scale * dz

    # atomic add to grad_pos for i and j
    tl.atomic_add(grad_pos_ptr + i3 + 0, gx, mask=mask)
    tl.atomic_add(grad_pos_ptr + i3 + 1, gy, mask=mask)
    tl.atomic_add(grad_pos_ptr + i3 + 2, gz, mask=mask)

    tl.atomic_add(grad_pos_ptr + j3 + 0, -gx, mask=mask)
    tl.atomic_add(grad_pos_ptr + j3 + 1, -gy, mask=mask)
    tl.atomic_add(grad_pos_ptr + j3 + 2, -gz, mask=mask)


@triton_op("mlcg_kernels::repulsion_pos_bwd", mutates_args={})
@ensure_contiguous
def repulsion_pos_bwd(
    pos: torch.Tensor,
    atom_types: torch.Tensor,
    index_mapping: torch.Tensor,
    mapping_batch: torch.Tensor,
    sigma: torch.Tensor,
    grad_y: torch.Tensor,
    num_graphs: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute position gradients for repulsion energy using Triton.

    The gradient with respect to position is:

    $$ \frac{\partial E_{ij}}{\partial \mathbf{r}_i} = -6 \left( \frac{\sigma_{t_i t_j}}{r_{ij}} \right)^6 \frac{\mathbf{r}_i - \mathbf{r}_j}{r_{ij}^8} $$
    """
    grad_pos = torch.zeros_like(pos).contiguous()

    E = index_mapping.shape[1]

    wrap_triton(repulsion_pos_bwd_kernel)[
        lambda meta: (triton.cdiv(E, meta["BLOCK"]),)
    ](
        pos,
        atom_types,
        index_mapping,
        mapping_batch,
        sigma,
        grad_y,
        grad_pos,
        E=E,
        B=num_graphs,
        eps=eps,
        TYPE_DTYPE=32 if atom_types.dtype == torch.int32 else 64,
        BATCH_DTYPE=32 if mapping_batch.dtype == torch.int32 else 64,
        SIG_STRIDE0=sigma.stride(0),
    )
    return grad_pos


@repulsion_pos_bwd.register_kernel("cpu")
def cpu_repulsion_pos_bwd(
    pos: torch.Tensor,
    atom_types: torch.Tensor,
    index_mapping: torch.Tensor,
    mapping_batch: torch.Tensor,
    sigma: torch.Tensor,
    grad_y: torch.Tensor,
    num_graphs: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    """CPU reference backward implementation for repulsion position gradients."""
    grad_pos = torch.zeros_like(pos)

    i = index_mapping[0].long()
    j = index_mapping[1].long()
    b = mapping_batch.long()

    ti = atom_types[i].long()
    tj = atom_types[j].long()
    sig = sigma[ti, tj].float()

    dx = pos[i] - pos[j]  # [E, 3]
    r2 = (dx * dx).sum(dim=1)  # [E]

    inv_r2 = 1.0 / (r2 + eps)
    inv_r8 = inv_r2 * inv_r2 * inv_r2 * inv_r2  # (r^2+eps)^-4

    sig6 = sig**6
    gy = grad_y[b]  # [E]

    scale = (-6.0) * sig6 * inv_r8 * gy  # [E]

    g = scale.unsqueeze(1) * dx  # [E, 3]

    grad_pos.index_add_(0, i, g)
    grad_pos.index_add_(0, j, -g)

    return grad_pos
