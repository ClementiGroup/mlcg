"""
Double backwards kernel for cfconv: useful in training when
double graident is need in force matching
"""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from ....utils import ensure_contiguous

from ...cutoffs import _cosine_cutoff, _d_cosine_cutoff_dd, _d2_cosine_cutoff_dd2

triton_pi = tl.constexpr(3.141592653589793)

# ============================================================================
# grad_grad_out_grad_x_fused_cfconv
# ============================================================================


@triton.jit
def grad_grad_out_grad_x_fused_cfconv_kernel(
    # Input pointers
    grad_grad_x_ptr,  # [num_nodes, feature_dim]
    filters_ptr,  # [num_edges, feature_dim] - filters outputs (FP32 or FP16)
    edge_weight_ptr,  # [num_edges]
    edge_src_ptr,  # [num_edges]
    dst_perm_ptr,  # [num_edges] - dst-CSR permutation
    dst_ptr_ptr,  # [num_nodes + 1] - dst-CSR row pointers
    # Output pointer
    grad_grad_out_ptr,
    # Cutoff parameters
    cutoff_upper,
    # Sizes
    num_nodes,
    feature_dim,
    # Block size #FIXME: check if other casting are needed
    BLOCK_F: tl.constexpr,
    filters_FP16: tl.constexpr,
):
    """
    Fused kernel for grad_grad_out computation in grad_x_fused_cfconv backward pass.

    Computes:
        Computes: grad_grad_out[dst] =  sum_{e: dst[e]=dst} grad_grad_x[src[e]] * filters[e] * cutoff[e]

    """
    target_node = tl.program_id(axis=0)

    if target_node >= num_nodes:
        return

    seg_start_dst = tl.load(dst_ptr_ptr + target_node)
    seg_end_dst = tl.load(dst_ptr_ptr + target_node + 1)

    for f_start in range(0, feature_dim, BLOCK_F):
        f_offsets = f_start + tl.arange(0, BLOCK_F)
        f_mask = f_offsets < feature_dim

        acc = tl.zeros([BLOCK_F], dtype=tl.float32)
        for e_csr in range(seg_start_dst, seg_end_dst):
            edge_idx = tl.load(dst_perm_ptr + e_csr)

            src_node = tl.load(edge_src_ptr + edge_idx)

            grad_grad_x = tl.load(
                grad_grad_x_ptr + src_node * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )
            filters = tl.load(
                filters_ptr + edge_idx * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )
            distances = tl.load(edge_weight_ptr + edge_idx)
            C = _cosine_cutoff(distances, cutoff_upper)

            if filters_FP16:
                filters = filters.to(tl.float32)

            acc += (
                grad_grad_x * filters * C[:, None]
            )  # FIXME: check this broadcast, maybe just C

        tl.store(
            grad_grad_out_ptr + target_node * feature_dim + f_offsets,
            acc,
            mask=f_mask,
        )


@triton_op(
    "mlcg_kernels::grad_grad_out_grad_x_fused_cfconv",
    mutates_args={},
)
@ensure_contiguous
def grad_grad_out_grad_x_fused_cfconv(
    filters: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    grad_grad_x: torch.Tensor,
    dst_perm: torch.Tensor,
    dst_ptr: torch.Tensor,
    cutoff_upper: float,
) -> torch.Tensor:
    num_nodes = grad_grad_x.shape[0]
    num_edges = edge_src.shape[0]
    feature_dim = grad_grad_x.shape[1]

    grad_grad_out = torch.zeros(
        num_nodes,
        feature_dim,
        device=grad_grad_x.device,
        dtype=grad_grad_x.dtype,
    ).contiguous()  # FIXME: check dtype

    if num_edges == 0:
        return grad_grad_out

    filters_fp16 = filters.dtype == torch.float16
    BLOCK_F = 128
    grid = (num_nodes,)

    wrap_triton(grad_grad_out_grad_x_fused_cfconv_kernel)[grid](
        grad_grad_x,
        filters,
        edge_weight,
        edge_src,
        dst_perm,
        dst_ptr,
        grad_grad_out,
        cutoff_upper,
        num_nodes,
        feature_dim,
        BLOCK_F=BLOCK_F,
        filters_FP16=filters_fp16,
    )

    return grad_grad_out


def setup_context_grad_grad_out_grad_x_fused_cfconv(ctx, inputs, output):
    raise NotImplementedError  # TODO: implement setup_context for grad_grad_out_grad_x_fused_cfconv


def backward_grad_grad_out_grad_x_fused_cfconv(ctx, grad_grad_grad_out):
    raise NotImplementedError  # TODO: implement backward for grad_grad_out_grad_x_fused_cfconv


grad_grad_out_grad_x_fused_cfconv.register_autograd(
    backward_grad_grad_out_grad_x_fused_cfconv,
    setup_context=setup_context_grad_grad_out_grad_x_fused_cfconv,
)


@grad_grad_out_grad_x_fused_cfconv.register_kernel("cpu")
def cpu_grad_grad_out_grad_x_fused_cfconv(
    filters: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    grad_grad_x: torch.Tensor,
    dst_perm: torch.Tensor,
    dst_ptr: torch.Tensor,
    cutoff_upper: float,
) -> torch.Tensor:

    C = 0.5 * (torch.cos(edge_weight * torch.pi / cutoff_upper) + 1)
    C = C * (edge_weight < cutoff_upper).to(edge_weight.dtype)

    grad_grad_out = torch.zeros_like(grad_grad_x)
    expanded = grad_grad_x[edge_src] * filters * C.unsqueeze(1)
    grad_grad_out.index_add_(0, edge_dst, expanded)

    return grad_grad_out  # FIXME: check dtype


# ============================================================================
# grad_filters_grad_x_fused_cfconv
# ============================================================================


@triton.jit
def grad_filters_grad_x_fused_cfconv_kernel(
    # Input pointers
    grad_grad_x_ptr,  # [num_nodes, feature_dim]
    grad_out_ptr,  # [num_nodes, feature_dim]
    edge_weight_ptr,  # [num_edges]
    edge_src_ptr,  # [num_edges]
    edge_dst_ptr,  # [num_edges]
    # Output pointer
    grad_filters_ptr,
    # Cutoff parameters
    cutoff_upper,
    # Sizes
    num_edges,
    feature_dim,
    # Block size
    BLOCK_F: tl.constexpr,
):
    """
    Fused kernel for grad_filters computation in grad_x_fused_cfconv backward pass.

    Computes:
        Computes: grad_filters[e] =  grad_grad_x[src[e]] * grad_out[dst[e]] * cutoff[e]

    """
    edge_idx = tl.program_id(axis=0)

    if edge_idx >= num_edges:
        return

    src_node = tl.load(edge_src_ptr + edge_idx)
    dst_node = tl.load(edge_dst_ptr + edge_idx)

    dist = tl.load(edge_weight_ptr + edge_idx)
    C = _cosine_cutoff(dist, cutoff_upper)

    for f_start in range(0, feature_dim, BLOCK_F):
        f_offset = f_start + tl.arange(0, BLOCK_F)
        f_mask = f_offset < feature_dim

        grad_grad_x = tl.load(
            grad_grad_x_ptr + src_node * feature_dim + f_offset,
            mask=f_mask,
            other=0.0,
        )
        grad_out = tl.load(
            grad_out_ptr + dst_node * feature_dim + f_offset,
            mask=f_mask,
            other=0.0,
        )

        tl.store(
            grad_filters_ptr + edge_idx * feature_dim + f_offset,
            grad_grad_x
            * grad_out
            * C[:, None],  # FIXME: check this broadcast, maybe just C
            mask=f_mask,
        )


@triton_op("mlcg_kernels::grad_filters_grad_x_fused_cfconv", mutates_args={})
@ensure_contiguous
def grad_filters_grad_x_fused_cfconv(
    grad_out: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    grad_grad_x: torch.Tensor,
    cutoff_upper: float,
) -> torch.Tensor:
    num_edges = edge_src.shape[0]
    feature_dim = grad_out.shape[1]

    grad_filters = torch.zeros(
        num_edges, feature_dim, device=grad_out.device, dtype=grad_out.dtype
    ).contiguous()  # FIXME: check if dtype is correct

    if num_edges == 0:
        return grad_filters

    BLOCK_F = 128
    grid = (num_edges,)

    wrap_triton(grad_filters_grad_x_fused_cfconv_kernel)[grid](
        grad_grad_x,
        grad_out,
        edge_weight,
        edge_src,
        edge_dst,
        grad_filters,
        cutoff_upper,
        num_edges,
        feature_dim,
        BLOCK_F=BLOCK_F,
    )

    return grad_filters


def setup_context_grad_filters_grad_x_fused_cfconv(ctx, inputs, output):
    raise NotImplementedError  # TODO: implement setup_context for grad_filters_grad_x_fused_cfconv


def backward_grad_filters_grad_x_fused_cfconv(ctx, grad_grad_filters):
    raise NotImplementedError  # TODO: implement backward for grad_filters_grad_x_fused_cfconv


grad_filters_grad_x_fused_cfconv.register_autograd(
    backward_grad_filters_grad_x_fused_cfconv,
    setup_context=setup_context_grad_filters_grad_x_fused_cfconv,
)


@grad_filters_grad_x_fused_cfconv.register_kernel("cpu")
def cpu_grad_filters_grad_x_fused_cfconv(
    grad_out: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    grad_grad_x: torch.Tensor,
    cutoff_upper: float,
) -> torch.Tensor:

    C = 0.5 * (torch.cos(edge_weight * torch.pi / cutoff_upper) + 1)
    C = C * (edge_weight < cutoff_upper).to(edge_weight.dtype)

    grad_filters = grad_grad_x[edge_src] * grad_out[edge_dst] * C.unsqueeze(1)

    return grad_filters  # FIXME: check dtype


# ============================================================================
# grad_edge_weight_grad_x_fused_cfconv
# ============================================================================


@triton.jit
def grad_edge_weight_grad_x_fused_cfconv_kernel(
    # Input pointers
    grad_grad_x_ptr,  # [num_nodes, feature_dim]
    grad_out_ptr,  # [num_nodes, feature_dim]
    filters_ptr,  # [num_edges, feature_dim]
    edge_weight_ptr,  # [num_edges]
    edge_src_ptr,  # [num_edges]
    edge_dst_ptr,  # [num_edges]
    # Output pointer
    grad_edge_weight_ptr,
    # Cutoff parameters
    cutoff_upper,
    # Sizes
    num_edges,
    feature_dim,
    # Block size
    BLOCK_F: tl.constexpr,
    filters_FP16: tl.constexpr,
):
    """
    Fused kernel for grad_edge_weight computation in grad_x_fused_cfconv backward pass.

    Computes:
        Computes: grad_edge_weight[e] =  (grad_grad_x[src[e]] * grad_out[dst[e]] * filters[e]).sum(axs=-1) * d_cutoff_dd[e]

    """
    edge_idx = tl.program_id(axis=0)
    if edge_idx >= num_edges:
        return

    src_node = tl.load(edge_src_ptr + edge_idx)
    dst_node = tl.load(edge_dst_ptr + edge_idx)

    dist = tl.load(edge_weight_ptr + edge_idx)
    dC = _d_cosine_cutoff_dd(dist, cutoff_upper)

    acc = tl.zeros_like(dC)

    for f_start in range(0, feature_dim, BLOCK_F):
        f_offset = f_start + tl.arange(0, BLOCK_F)
        f_mask = f_offset < feature_dim

        grad_grad_x = tl.load(
            grad_grad_x_ptr + src_node * feature_dim + f_offset,
            mask=f_mask,
            other=0.0,
        )
        grad_out = tl.load(
            grad_out_ptr + dst_node * feature_dim + f_offset,
            mask=f_mask,
            other=0.0,
        )
        filters = tl.load(
            filters_ptr + edge_idx * feature_dim + f_offset,
            mask=f_mask,
            other=0.0,
        )

        if filters_FP16:
            filters = filters.to(tl.float32)

        acc += tl.sum(grad_grad_x * grad_out * filters, axis=-1)

    tl.store(grad_edge_weight_ptr + edge_idx, acc * dC)


@triton_op(
    "mlcg_kernels::grad_edge_weight_grad_x_fused_cfconv", mutates_args={}
)
@ensure_contiguous
def grad_edge_weight_grad_x_fused_cfconv(
    grad_out: torch.Tensor,
    filters: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    grad_grad_x: torch.Tensor,
    cutoff_upper: float,
) -> torch.Tensor:

    num_edges = edge_src.shape[0]
    feature_dim = grad_out.shape[1]

    grad_edge_weight = torch.zeros(
        num_edges, device=grad_out.device, dtype=grad_out.dtype
    ).contiguous()

    if num_edges == 0:
        return grad_edge_weight

    filters_fp16 = filters.dtype == torch.float16

    BLOCK_F = 128
    grid = (num_edges,)

    wrap_triton(grad_edge_weight_grad_x_fused_cfconv_kernel)[grid](
        grad_grad_x,
        grad_out,
        filters,
        edge_weight,
        edge_src,
        edge_dst,
        grad_edge_weight,
        cutoff_upper,
        num_edges,
        feature_dim,
        BLOCK_F=BLOCK_F,
        filters_FP16=filters_fp16,
    )

    return grad_edge_weight


def setup_context_grad_edge_weight_grad_x_fused_cfconv(ctx, inputs, output):
    raise NotImplementedError  # TODO: implement setup_context for grad_edge_weight_grad_x_fused_cfconv


def backward_grad_edge_weight_grad_x_fused_cfconv(ctx, grad_grad_edge_weight):
    raise NotImplementedError  # TODO: implement backward for grad_edge_weight_grad_x_fused_cfconv


grad_edge_weight_grad_x_fused_cfconv.register_autograd(
    backward_grad_edge_weight_grad_x_fused_cfconv,
    setup_context=setup_context_grad_edge_weight_grad_x_fused_cfconv,
)


@grad_edge_weight_grad_x_fused_cfconv.register_kernel("cpu")
def cpu_grad_edge_weight_grad_x_fused_cfconv(
    grad_out: torch.Tensor,
    filters: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    grad_grad_x: torch.Tensor,
    cutoff_upper: float,
) -> torch.Tensor:

    dC_dd = (
        -0.5
        * torch.sin(edge_weight * torch.pi / cutoff_upper)
        * (torch.pi / cutoff_upper)
    )
    dC_dd = dC_dd * (edge_weight < cutoff_upper).to(edge_weight.dtype)

    grad_edge_weight = (
        grad_grad_x[edge_src] * grad_out[edge_dst] * filters
    ).sum(dim=1) * dC_dd

    return grad_edge_weight  # FIXME: check dtype

