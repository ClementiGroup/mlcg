import torch
from typing import List
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from ..utils import ensure_contiguous

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
    reset_to_zero=["grad_pos_ptr", "grad_centers_ptr", "grad_gamma_ptr"],
)
@triton.jit
def fused_gaussian_rbf_backward_kernel(
    # Inputs
    pos_ptr,  # [num_nodes, 3]
    edge_src_ptr,  # [num_edges]
    edge_dst_ptr,  # [num_edges]
    centers_ptr,  # [num_rbf]
    distances_ptr,  # [num_edges]
    grad_distances_ptr,  # [num_edges]
    grad_rbf_ptr,  # [num_edges, num_rbf]
    # Outputs
    grad_pos_ptr,  # [num_nodes, 3]
    grad_centers_ptr,  # [num_rbf]
    grad_gamma_ptr,  # [1]
    # Scalars
    gamma_ptr,
    cutoff_upper,
    num_edges,
    num_rbf: tl.constexpr,
    # Flags
    NEED_POS_GRAD: tl.constexpr,
    NEED_CENTERS_GRAD: tl.constexpr,
    NEED_GAMMA_GRAD: tl.constexpr,
    # Block sizes
    BLOCK_EDGES: tl.constexpr,
    BLOCK_RBF: tl.constexpr,
):

    pid_edge = tl.program_id(axis=0)
    edge_start = pid_edge * BLOCK_EDGES
    edge_offsets = edge_start + tl.arange(0, BLOCK_EDGES)
    edge_mask = edge_offsets < num_edges

    gamma = tl.load(gamma_ptr)
    dist = tl.load(distances_ptr + edge_offsets, mask=edge_mask, other=0.0)

    # Compute cosine cutoff: 0.5 * (cos(d * pi / cutoff) + 1) * (d < cutoff)
    cos_val = tl.cos(dist * triton_pi / cutoff_upper)
    cutoff_val = 0.5 * (cos_val + 1.0)
    dist_in_range = dist < cutoff_upper
    cutoff_val = tl.where(dist_in_range, cutoff_val, 0.0)

    acc_gamma_grad = 0.0
    acc_grad_dist_from_rbf = tl.zeros([BLOCK_EDGES], dtype=tl.float32)

    if NEED_POS_GRAD:
        sin_val = tl.sin(dist * triton_pi / cutoff_upper)
        d_cutoff_d_dist = tl.where(
            dist_in_range, -0.5 * sin_val * triton_pi / cutoff_upper, 0.0
        )

    for rbf_start in range(0, num_rbf, BLOCK_RBF):

        rbf_offsets = rbf_start + tl.arange(0, BLOCK_RBF)
        rbf_mask = rbf_offsets < num_rbf

        centers = tl.load(centers_ptr + rbf_offsets, mask=rbf_mask, other=0.0)

        # Compute Gaussian RBF: exp(gamma * (dist - center)^2) * cutoff
        diff = dist[:, None] - centers[None, :]
        rbf_values = tl.exp(gamma * diff * diff) * cutoff_val[:, None]

        # Store output with 2D mask
        rbf_2d_mask = edge_mask[:, None] & rbf_mask[None, :]
        rbf_2d_offsets = edge_offsets[:, None] * num_rbf + rbf_offsets[None, :]

        # Load input grads
        grad_rbf = tl.load(
            grad_rbf_ptr + rbf_2d_offsets, mask=rbf_2d_mask, other=0.0
        ).to(tl.float32)

        if NEED_CENTERS_GRAD:
            drbf_dcenters = -rbf_values * 2 * gamma * diff
            grad_centers_update = tl.sum(
                grad_rbf * drbf_dcenters.to(tl.float32), axis=0
            )
            tl.atomic_add(
                grad_centers_ptr + rbf_offsets,
                grad_centers_update,
                mask=rbf_mask,
            )

        if NEED_GAMMA_GRAD:
            drbf_dgamma = rbf_values * diff * diff
            acc_gamma_grad += tl.sum(grad_rbf * drbf_dgamma)

        if NEED_POS_GRAD:
            d_rbf_d_dist = (
                2 * gamma * diff * rbf_values
                + tl.exp(gamma * diff * diff) * d_cutoff_d_dist[:, None]
            )

            grad_dist_from_rbf = tl.sum(grad_rbf * d_rbf_d_dist, axis=1)

            acc_grad_dist_from_rbf += grad_dist_from_rbf

    if NEED_GAMMA_GRAD:
        tl.atomic_add(grad_gamma_ptr, acc_gamma_grad)

    if NEED_POS_GRAD:
        # grad_pos = tl.load(grad_pos_ptr + edge_offsets, mask=edge_mask, other=0.0)

        src_nodes = tl.load(
            edge_src_ptr + edge_offsets, mask=edge_mask, other=0
        )
        dst_nodes = tl.load(
            edge_dst_ptr + edge_offsets, mask=edge_mask, other=0
        )

        # Load positions to compute direction
        pos_src_x = tl.load(
            pos_ptr + src_nodes * 3 + 0, mask=edge_mask, other=0.0
        )
        pos_src_y = tl.load(
            pos_ptr + src_nodes * 3 + 1, mask=edge_mask, other=0.0
        )
        pos_src_z = tl.load(
            pos_ptr + src_nodes * 3 + 2, mask=edge_mask, other=0.0
        )
        pos_dst_x = tl.load(
            pos_ptr + dst_nodes * 3 + 0, mask=edge_mask, other=0.0
        )
        pos_dst_y = tl.load(
            pos_ptr + dst_nodes * 3 + 1, mask=edge_mask, other=0.0
        )
        pos_dst_z = tl.load(
            pos_ptr + dst_nodes * 3 + 2, mask=edge_mask, other=0.0
        )

        dx = pos_dst_x - pos_src_x
        dy = pos_dst_y - pos_src_y
        dz = pos_dst_z - pos_src_z

        dist_safe = tl.maximum(dist, 1e-8)
        dir_x = dx / dist_safe
        dir_y = dy / dist_safe
        dir_z = dz / dist_safe

        grad_distances = tl.load(
            grad_distances_ptr + edge_offsets, mask=edge_mask, other=0.0
        )

        total_grad_dist = grad_distances + acc_grad_dist_from_rbf

        grad_x = total_grad_dist * dir_x
        grad_y = total_grad_dist * dir_y
        grad_z = total_grad_dist * dir_z

        # Scatter to dst nodes (atomic add, +grad)
        tl.atomic_add(grad_pos_ptr + dst_nodes * 3 + 0, grad_x, mask=edge_mask)
        tl.atomic_add(grad_pos_ptr + dst_nodes * 3 + 1, grad_y, mask=edge_mask)
        tl.atomic_add(grad_pos_ptr + dst_nodes * 3 + 2, grad_z, mask=edge_mask)

        # Scatter to src nodes (atomic add, -grad)
        tl.atomic_add(grad_pos_ptr + src_nodes * 3 + 0, -grad_x, mask=edge_mask)
        tl.atomic_add(grad_pos_ptr + src_nodes * 3 + 1, -grad_y, mask=edge_mask)
        tl.atomic_add(grad_pos_ptr + src_nodes * 3 + 2, -grad_z, mask=edge_mask)


# TODO: register backward for this
@triton_op("mlcg_kernels::fused_gaussian_rbf_backward", mutates_args={})
@ensure_contiguous
def fused_gaussian_rbf_backward(
    pos: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    centers: torch.Tensor,
    gamma: torch.Tensor,
    distances: torch.Tensor,
    cutoff_upper: float,
    grad_distances: torch.Tensor,
    grad_rbf: torch.Tensor,
    need_pos_grad: bool,
    need_centers_grad: bool,
    need_gamma_grad: bool,
) -> List[torch.Tensor]:

    # Initialize with fake if not requested
    grad_pos = torch.zeros_like(pos).contiguous()
    grad_centers = torch.zeros_like(centers).contiguous()
    grad_gamma = torch.zeros(1, dtype=pos.dtype, device=pos.device)

    num_edges = edge_src.shape[0]
    num_rbf = centers.shape[0]

    def grid(META):
        return (triton.cdiv(num_edges, META["BLOCK_EDGES"]),)

    wrap_triton(fused_gaussian_rbf_backward_kernel)[grid](
        # Inputs
        pos,
        edge_src,
        edge_dst,
        centers,
        distances,
        grad_distances,
        grad_rbf,
        # Outputs
        grad_pos,
        grad_centers,
        grad_gamma,
        # Scalars
        gamma,
        cutoff_upper,
        num_edges,
        num_rbf,
        # Flags
        need_pos_grad,
        need_centers_grad,
        need_gamma_grad,
    )

    return [grad_pos, grad_centers, grad_gamma]


@fused_gaussian_rbf_backward.register_kernel("cpu")
def _(
    pos: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    centers: torch.Tensor,
    gamma: torch.Tensor,
    distances: torch.Tensor,
    cutoff_upper: float,
    grad_distances: torch.Tensor,
    grad_rbf: torch.Tensor,
    need_pos_grad: bool,
    need_centers_grad: bool,
    need_gamma_grad: bool,
) -> List[torch.Tensor]:

    grad_pos = torch.zeros_like(pos)
    grad_centers = torch.zeros_like(centers)
    grad_gamma = torch.zeros(1, dtype=pos.dtype, device=pos.device)

    cutoff_val = 0.5 * (torch.cos(distances * torch.pi / cutoff_upper) + 1.0)
    cutoff_val = torch.where(
        distances < cutoff_upper, cutoff_val, torch.zeros_like(cutoff_val)
    )

    diff = distances.unsqueeze(-1) - centers.unsqueeze(
        0
    )  # [num_edges, num_rbf]

    rbf = torch.exp(gamma * torch.pow(diff, 2)) * cutoff_val.unsqueeze(-1)

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

        # d(rbf)/d(dist) = [2*gamma*(dist-center)*exp(...)*cutoff + exp(...)*d_cutoff]
        d_rbf_d_dist = 2 * gamma * diff * rbf + torch.exp(
            gamma * diff**2
        ) * d_cutoff_d_dist.unsqueeze(-1)

        # Aggregate gradient from all RBF channels
        grad_dist_from_rbf = (grad_rbf * d_rbf_d_dist).sum(
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

    if need_centers_grad:
        drbf_dcenters = -rbf * 2 * gamma * diff
        grad_centers = (grad_rbf * drbf_dcenters).sum(dim=0)

    if need_gamma_grad:
        drbf_dgamma = rbf * diff * diff
        grad_gamma = (grad_rbf * drbf_dgamma).sum(dim=0)

    return [grad_pos, grad_centers, grad_gamma]
