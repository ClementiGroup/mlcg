import torch
from typing import List
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

triton_pi = tl.constexpr(3.141592653589793)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_EDGES": 32, "BLOCK_RBF": 64}, num_warps=4),
        triton.Config({"BLOCK_EDGES": 64, "BLOCK_RBF": 64}, num_warps=4),
        triton.Config({"BLOCK_EDGES": 128, "BLOCK_RBF": 64}, num_warps=4),
        triton.Config({"BLOCK_EDGES": 64, "BLOCK_RBF": 32}, num_warps=4),
        triton.Config({"BLOCK_EDGES": 128, "BLOCK_RBF": 32}, num_warps=4),
    ],
    key=["num_rbf"],
)
@triton.jit
def fused_distance_exp_norm_rbf_cosinecutoff_kernel(
    # Input pointers
    pos_ptr,  # [num_nodes, 3]
    edge_src_ptr,  # [num_edges]
    edge_dst_ptr,  # [num_edges]
    means_ptr,  # [num_rbf]
    betas_ptr,  # [num_rbf]
    alpha_ptr,  # scalar stored in a tensor [1]
    # Output pointers
    dist_output_ptr,  # [num_edges]
    rbf_output_ptr,  # [num_edges, num_rbf]
    # Cutoff parameters
    cutoff_upper,  # TODO: this only works assuming cutoff lower is 0, need to be expanded to consider also other case
    # Sizes
    num_edges,
    num_rbf,
    # Block sizes
    BLOCK_EDGES: tl.constexpr,
    BLOCK_RBF: tl.constexpr,
):
    """
    Fused kernel that combines:
    1. Distance computation: d = ||pos[dst] - pos[src]||
    2. ExpNorm RBF expansion: exp(-beta * (exp(alpha(cutoff_lower - d) - means)^2)
    3. Cosine cutoff: 0.5 * (cos(d * pi / cutoff_upper) + 1) * (d < cutoff_upper)

    Output: distances and RBF expansion with cutoff applied.

    This eliminates intermediate memory traffic by computing distance once
    and reusing it for all RBF centers and cutoff calculation.
    """
    pid_edge = tl.program_id(axis=0)
    pid_rbf = tl.program_id(axis=1)

    # Compute edge and RBF offsets for this block
    edge_start = pid_edge * BLOCK_EDGES
    rbf_start = pid_rbf * BLOCK_RBF

    edge_offsets = edge_start + tl.arange(0, BLOCK_EDGES)
    rbf_offsets = rbf_start + tl.arange(0, BLOCK_RBF)

    edge_mask = edge_offsets < num_edges
    rbf_mask = rbf_offsets < num_rbf

    # Load alpha scalar (once per block)
    alpha = tl.load(alpha_ptr)

    # Load RBF centers for this block (shared across all edges)
    means = tl.load(means_ptr + rbf_offsets, mask=rbf_mask, other=0.0)
    betas = tl.load(betas_ptr + rbf_offsets, mask=rbf_mask, other=0.0)

    # Load edge indices
    src_nodes = tl.load(edge_src_ptr + edge_offsets, mask=edge_mask, other=0)
    dst_nodes = tl.load(edge_dst_ptr + edge_offsets, mask=edge_mask, other=0)

    # Load source positions (strided by 3)
    pos_src_x = tl.load(pos_ptr + src_nodes * 3 + 0, mask=edge_mask, other=0.0)
    pos_src_y = tl.load(pos_ptr + src_nodes * 3 + 1, mask=edge_mask, other=0.0)
    pos_src_z = tl.load(pos_ptr + src_nodes * 3 + 2, mask=edge_mask, other=0.0)

    # Load destination positions
    pos_dst_x = tl.load(pos_ptr + dst_nodes * 3 + 0, mask=edge_mask, other=0.0)
    pos_dst_y = tl.load(pos_ptr + dst_nodes * 3 + 1, mask=edge_mask, other=0.0)
    pos_dst_z = tl.load(pos_ptr + dst_nodes * 3 + 2, mask=edge_mask, other=0.0)

    # Compute distances [BLOCK_EDGES]
    dx = pos_dst_x - pos_src_x
    dy = pos_dst_y - pos_src_y
    dz = pos_dst_z - pos_src_z
    dist = tl.sqrt(dx * dx + dy * dy + dz * dz)

    # Store distances (only on first RBF block to avoid duplicate writes)
    if pid_rbf == 0:
        tl.store(dist_output_ptr + edge_offsets, dist, mask=edge_mask)

    # TODO: this only works assuming cutoff lower is 0, need to be expanded to consider also other case
    # Compute cosine cutoff: 0.5 * (cos(d * pi / cutoff) + 1) * (d < cutoff)
    cos_val = tl.cos(dist * triton_pi / cutoff_upper)
    cutoff_val = 0.5 * (cos_val + 1.0)
    dist_in_range = dist < cutoff_upper
    cutoff_val = tl.where(dist_in_range, cutoff_val, 0.0)

    # Broadcast for 2D computation:
    # dist: [BLOCK_EDGES, 1]
    # centers: [1, BLOCK_RBF]
    dist_2d = dist[:, None]
    means_2d = means[None, :]
    betas_2d = betas[None, :]
    cutoff_2d = cutoff_val[:, None]

    # Compute Gaussian RBF: exp(gamma * (dist - center)^2) * cutoff
    diff = tl.exp(-alpha * dist_2d) - means_2d
    rbf_values = tl.exp(-betas_2d * diff * diff) * cutoff_2d

    # Store output with 2D mask
    output_mask = edge_mask[:, None] & rbf_mask[None, :]
    output_offsets = edge_offsets[:, None] * num_rbf + rbf_offsets[None, :]
    tl.store(rbf_output_ptr + output_offsets, rbf_values, mask=output_mask)


@triton_op(
    "mlcg_kernels::fused_distance_exp_norm_rbf_cosinecutoff", mutates_args={}
)
def fused_distance_exp_norm_rbf_cosinecutoff(
    pos: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    means: torch.Tensor,
    betas: torch.Tensor,
    alpha: float,
    cutoff_upper: float,
) -> List[torch.Tensor]:
    """
    Compute fused distance + ExpNorm RBF expansion + cosine cutoff.

    This kernel fuses three operations that are typically separate:
    1. Distance computation: d = ||pos[dst] - pos[src]||
    2. ExpNorm RBF: exp(-beta * (exp(alpha(cutoff_lower - d) - means)^2)
    3. Cosine cutoff: 0.5 * (cos(d * pi / cutoff) + 1)

    Parameters
    ----------
    pos : torch.Tensor
        Node positions [num_nodes, 3]
    edge_src : torch.Tensor
        Source node indices [num_edges]
    edge_dst : torch.Tensor
        Destination node indices [num_edges]
    means : torch.Tensor
        RBF means [num_rbf]
    betas : torch.Tensor
        RBF betas [num_rbf]
    alpha : float
        RBF alpha factor
    cutoff_upper : float
        Upper cutoff distance for cosine envelope

    Returns
    -------
    distances : torch.Tensor
        Pairwise distances [num_edges]
    rbf_expansion : torch.Tensor
        RBF expansion with cutoff applied [num_edges, num_rbf]
    """
    if not pos.is_contiguous():
        pos = pos.contiguous()
    if not edge_src.is_contiguous():
        edge_src = edge_src.contiguous()
    if not edge_dst.is_contiguous():
        edge_dst = edge_dst.contiguous()
    if not means.is_contiguous():
        means = means.contiguous()
    if not betas.is_contiguous():
        betas = betas.contiguous()

    num_edges = edge_src.shape[0]
    num_rbf = means.shape[0]

    # Allocate outputs
    distances = torch.empty(num_edges, device=pos.device, dtype=pos.dtype)
    rbf_output = torch.empty(
        (num_edges, num_rbf), device=pos.device, dtype=pos.dtype
    )

    if num_edges == 0:
        return distances, rbf_output

    # Convert gamma to tensor for Triton
    alpha_tensor = torch.as_tensor([alpha], device=pos.device, dtype=pos.dtype)

    # 2D grid for the optimized kernel
    def grid(
        META,
    ):  # FIXME: check if the global variable are defined by them or by triton
        return (
            triton.cdiv(num_edges, META["BLOCK_EDGES"]),
            triton.cdiv(num_rbf, META["BLOCK_RBF"]),
        )

    wrap_triton(fused_distance_exp_norm_rbf_cosinecutoff_kernel)[grid](
        pos,
        edge_src,
        edge_dst,
        means,
        betas,
        alpha_tensor,
        distances,
        rbf_output,
        cutoff_upper,
        num_edges,
        num_rbf,
    )

    return [distances, rbf_output]


def setup_context(ctx, inputs, output):
    (
        pos,
        edge_src,
        edge_dst,
        means,
        beta,
        alpha,
        cutoff_upper,
    ) = inputs
    distances, rbf = output[0], output[1]

    ctx.save_for_backward(pos, edge_src, edge_dst, means, beta, distances, rbf)
    ctx.alpha = alpha
    ctx.cutoff_upper = cutoff_upper


def backward(ctx, grads):
    grad_distances, grad_rbf_output = grads[0], grads[1]
    pos, edge_src, edge_dst, means, beta, distances, rbf = ctx.saved_tensors
    alpha = ctx.alpha
    cutoff_upper = ctx.cutoff_upper

    grad_pos = None
    grad_means = None
    grad_betas = None

    need_pos_grad = ctx.needs_input_grad[0]
    need_means_grad = ctx.needs_input_grad[3]
    need_betas_grad = ctx.needs_input_grad[4]

    if need_pos_grad or need_means_grad or need_betas_grad:
        inner_exp = torch.exp(-alpha * distances)
        shifted_inner_exp = inner_exp.unsqueeze(-1) - means

    if need_pos_grad:
        # Direction vectors (normalized)
        dr = pos[edge_dst] - pos[edge_src]  # [num_edges, 3]
        dist_safe = distances.clamp(min=1e-8)
        direction = dr / dist_safe.unsqueeze(-1)  # [num_edges, 3]

        # d(cutoff)/d(dist) = -0.5 * pi/cutoff_upper * sin(dist * pi / cutoff_upper)
        dist_in_range = (distances < cutoff_upper).float()
        sin_val = torch.sin(distances * torch.pi / cutoff_upper)
        d_cutoff_d_dist = (
            -0.5 * (torch.pi / cutoff_upper) * sin_val * dist_in_range
        )

        # drbf_d(dist) = d(cutoff)/d(dist) * exp(-beta*(exp(-alpha*d)-means)^2) + 2 * rbf * beta * (exp(-alpha * d)-means)*exp(-alpha*d)*alpha
        raw_gaussian = torch.exp(
            -beta * shifted_inner_exp**2
        )  # [num_edges, num_rbf]
        d_rbf_d_dist = (
            d_cutoff_d_dist.unsqueeze(-1) * raw_gaussian
            + 2
            * rbf
            * beta
            * shifted_inner_exp
            * inner_exp.unsqueeze(-1)
            * alpha
        )

        # Aggregate gradient from all RBF channels
        grad_dist_from_rbf = (grad_rbf_output * d_rbf_d_dist).sum(
            dim=-1
        )  # [num_edges]

        # Total gradient w.r.t. distance
        total_grad_dist = grad_distances + grad_dist_from_rbf

        # Convert to position gradients
        grad_dr = total_grad_dist.unsqueeze(-1) * direction  # [num_edges, 3]

        # Scatter gradients to positions
        grad_pos = torch.zeros_like(pos)
        grad_pos.index_add_(0, edge_dst, grad_dr)
        grad_pos.index_add_(0, edge_src, -grad_dr)

    if need_means_grad:
        # drbf_dmeans = rbf * 2 * beta * (torch.exp(-alpha*distances) - means)
        drbf_dmeans = rbf * 2 * beta * shifted_inner_exp
        grad_means = (grad_rbf_output * drbf_dmeans).sum(dim=0)

    if need_betas_grad:
        # drbf_dbetas = -rbf * (torch.exp(-alpha*distances) - means)**2
        drbf_dbetas = -rbf * shifted_inner_exp * shifted_inner_exp
        grad_betas = (grad_rbf_output * drbf_dbetas).sum(dim=0)

    return grad_pos, None, None, grad_means, grad_betas, None, None


fused_distance_exp_norm_rbf_cosinecutoff.register_autograd(
    backward, setup_context=setup_context
)


## Adding CPU fallback for the fused kernel
@fused_distance_exp_norm_rbf_cosinecutoff.register_kernel("cpu")
def _(
    pos: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    means: torch.Tensor,
    betas: torch.Tensor,
    alpha: float,
    cutoff_upper: float,
) -> List[torch.Tensor]:
    # Compute distances
    dist = (pos[edge_dst] - pos[edge_src]).norm(p=2, dim=1)  # [num_edges]

    # Compute cosine cutoff
    cutoff_val = 0.5 * (torch.cos(dist * torch.pi / cutoff_upper) + 1.0)
    # remove contributions beyond the cutoff radius
    cutoff_val = torch.where(
        dist < cutoff_upper, cutoff_val, torch.zeros_like(cutoff_val)
    )

    # Compute ExpNorm RBF
    diff = torch.exp(-alpha * dist.unsqueeze(-1)) - means.unsqueeze(
        0
    )  # [num_edges, num_rbf]
    rbf_output = torch.exp(
        -betas.unsqueeze(0) * diff * diff
    ) * cutoff_val.unsqueeze(-1)

    return [dist, rbf_output]
