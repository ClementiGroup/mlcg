"""
Double backwards kernel for cfconv: useful in training when
double graident is need in force matching
"""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from ...utils import ensure_contiguous

from ..cutoffs import _cosine_cutoff, _d_cosine_cutoff_dd, _d2_cosine_cutoff_dd2

triton_pi = tl.constexpr(3.141592653589793)


# ============================================================================
# grad_x_grad_edge_weight_fused_cfconv
# ============================================================================


@triton.jit
def grad_x_grad_edge_weight_fused_cfconv_kernel(
    # Input pointers
    grad_output_ptr,  # [num_nodes, feature_dim]
    filter_ptr,  # [num_edges, feature_dim] - filter outputs (FP32 or FP16)
    edge_weight_ptr,  # [num_edges]
    edge_dst_ptr,  # [num_edges]
    grad_edge_out_ptr,  # [num_edges, feature_dim] - OUTPUT (FP32 or FP16)
    src_perm_ptr,  # [num_edges] - src-CSR permutation
    src_ptr_ptr,  # [num_nodes + 1] - src-CSR row pointers
    # Output pointers
    grad_x_ptr,  # [num_nodes, feature_dim]
    # Cutoff parameters
    cutoff_upper,
    # Sizes
    num_nodes,
    feature_dim,
    # Block size
    BLOCK_F: tl.constexpr,
    FILTER_FP16: tl.constexpr,
    GRAD_EDGE_FP16: tl.constexpr,  # Whether to output FP16
):
    """
    Fused kernel for grad_filter computation in CFConv backward pass.

    Computes:
        Computes: grad_x[src] = sum_{e: src[e]=src} grad_edge[e] * grad_output[dst[e]] * filter[e] * d_cutoff_dd[e]


    """
    target_node = tl.program_id(axis=0)

    if target_node >= num_nodes:
        return

    seg_start_src = tl.load(src_ptr_ptr + target_node)
    seg_end_src = tl.load(src_ptr_ptr + target_node + 1)

    for f_start in range(0, feature_dim, BLOCK_F):
        f_offsets = f_start + tl.arange(0, BLOCK_F)
        f_mask = f_offsets < feature_dim

        acc = tl.zeros([BLOCK_F], dtype=tl.float32)
        for e_csr in range(seg_start_src, seg_end_src):
            edge_idx = tl.load(src_perm_ptr + e_csr)

            dst_node = tl.load(edge_dst_ptr + edge_idx)

            distances = tl.load(edge_weight_ptr + edge_idx)
            C = _d_cosine_cutoff_dd(distances, cutoff_upper)
            grad_output = tl.load(
                grad_output_ptr + dst_node * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )
            filter = tl.load(
                filter_ptr + edge_idx * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )
            if FILTER_FP16:
                filter = filter.to(tl.float32)
            grad_edge = tl.load(
                grad_edge_out_ptr + edge_idx * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )
            if GRAD_EDGE_FP16:
                grad_edge = grad_edge.to(tl.float32)

            acc += grad_edge * filter * grad_output * C[:, None]

        tl.store(
            grad_x_ptr + target_node * feature_dim + f_offsets, acc, mask=f_mask
        )


@triton_op(
    "mlcg_kernels::grad_x_grad_edge_weight_fused_cfconv", mutates_args={}
)
@ensure_contiguous
def grad_x_grad_edge_weight_fused_cfconv(
    grad_output: torch.Tensor,
    filter: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_dst: torch.Tensor,
    grad_edge_out: torch.Tensor,
    src_perm: torch.Tensor,
    src_ptr: torch.Tensor,
    num_nodes: int,
    cutoff_upper: float,
    grad_edge_dtype: torch.dtype = None,
) -> torch.Tensor:
    feature_dim = grad_output.shape[1]

    # Allocate output (zeros for nodes with no outgoing edges)
    grad_x = torch.zeros(
        num_nodes, feature_dim, device=grad_output.device, dtype=torch.float32
    ).contiguous()

    num_edges = edge_dst.shape[0]
    if num_edges == 0:
        return grad_x

    filter_fp16 = filter.dtype == torch.float16
    grad_edge_fp16 = grad_edge_dtype == torch.float16

    BLOCK_F = 128
    grid = (num_nodes,)

    wrap_triton(grad_x_grad_edge_weight_fused_cfconv_kernel)[grid](
        grad_output,
        filter,
        edge_weight,
        edge_dst,
        grad_edge_out,
        src_perm,
        src_ptr,
        grad_x,
        cutoff_upper,
        num_nodes,
        feature_dim,
        BLOCK_F=BLOCK_F,
        FILTER_FP16=filter_fp16,
        GRAD_EDGE_FP16=grad_edge_fp16,
    )

    return grad_x


def setup_context_grad_x_grad_edge_weight_fused_cfconv(ctx, inputs, output):
    raise NotImplementedError  # TODO: implment setup context grad_x_grad_edge_weight_fused_cfconv


def backward_grad_x_grad_edge_weight_fused_cfconv(ctx, grad_output):
    raise NotImplementedError  # TODO: implment backward grad_x_grad_edge_weight_fused_cfconv


grad_x_grad_edge_weight_fused_cfconv.register_autograd(
    backward_grad_x_grad_edge_weight_fused_cfconv,
    setup_context=setup_context_grad_x_grad_edge_weight_fused_cfconv,
)


@grad_x_grad_edge_weight_fused_cfconv.register_kernel("cpu")
def cpu_grad_x_grad_edge_weight_fused_cfconv(
    grad_output: torch.Tensor,
    filter: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_dst: torch.Tensor,
    grad_edge_out: torch.Tensor,
    src_perm: torch.Tensor,
    src_ptr: torch.Tensor,
    num_nodes: int,
    cutoff_upper: float,
    grad_edge_dtype: torch.dtype = None,
) -> torch.Tensor:
    raise NotImplementedError  # FIXME: implement cpu fallback for grad_x_grad_edge_weight_fused_cfconv


# ============================================================================
# grad_grad_out_grad_edge_weight_fused_cfconv
# ============================================================================


@triton.jit
def grad_grad_out_grad_edge_weight_fused_cfconv_kernel(
    # Input pointers
    x_ptr,  # [num_nodes, feature_dim]
    filter_ptr,  # [num_edges, feature_dim] - filter outputs (FP32 or FP16)
    edge_weight_ptr,  # [num_edges]
    edge_src_ptr,  # [num_edges]
    grad_edge_out_ptr,  # [num_edges, feature_dim] - OUTPUT (FP32 or FP16)
    dst_perm_ptr,  # [num_edges] - dst-CSR permutation
    dst_ptr_ptr,  # [num_nodes + 1] - dst-CSR row pointers
    # Output pointers
    grad_grad_out_ptr,  # [num_nodes, feature_dim]
    # Cutoff parameters
    cutoff_upper,
    # Sizes
    num_nodes,
    feature_dim,
    # Block size
    BLOCK_F: tl.constexpr,
    FILTER_FP16: tl.constexpr,
    GRAD_EDGE_FP16: tl.constexpr,  # Whether to output FP16
):
    """
    Fused kernel for grad_filter computation in CFConv backward pass.

    Computes:
        Computes: grad_grad_out[dst] = sum_{e: dst[e]=dst} grad_edge[e] * x[src[e]] * filter[e] * d_cutoff_dd[e]


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

            distances = tl.load(edge_weight_ptr + edge_idx)
            C = _d_cosine_cutoff_dd(distances, cutoff_upper)
            x = tl.load(
                x_ptr + src_node * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )
            filter = tl.load(
                filter_ptr + edge_idx * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )
            if FILTER_FP16:
                filter = filter.to(tl.float32)
            grad_edge = tl.load(
                grad_edge_out_ptr + edge_idx * BLOCK_F + f_offsets,
                mask=f_mask,
                other=0.0,
            )
            if GRAD_EDGE_FP16:
                grad_edge = grad_edge.to(tl.float32)

            acc += grad_edge * x * filter * C[:, None]

        tl.store(
            grad_grad_out_ptr + target_node * feature_dim + f_offsets,
            acc,
            mask=f_mask,
        )


@triton_op(
    "mlcg_kernels::grad_grad_out_grad_edge_weight_fused_cfconv", mutates_args={}
)
@ensure_contiguous
def grad_grad_out_grad_edge_weight_fused_cfconv(
    x: torch.Tensor,
    filter: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    grad_edge_out: torch.Tensor,
    dst_perm: torch.Tensor,
    dst_ptr: torch.Tensor,
    num_nodes: int,
    cutoff_upper: float,
    grad_edge_dtype: torch.dtype = None,
) -> torch.Tensor:
    feature_dim = x.shape[1]

    # Allocate output (zeros for nodes with no outgoing edges)
    grad_grad_out = torch.zeros(
        num_nodes, feature_dim, device=x.device, dtype=torch.float32
    ).contiguous()

    num_edges = edge_src.shape[0]
    if num_edges == 0:
        return grad_grad_out

    filter_fp16 = filter.dtype == torch.float16
    grad_edge_fp16 = grad_edge_dtype == torch.float16

    BLOCK_F = 128
    grid = (num_nodes,)

    wrap_triton(grad_x_grad_edge_weight_fused_cfconv_kernel)[grid](
        x,
        filter,
        edge_weight,
        edge_src,
        grad_edge_out,
        dst_perm,
        dst_ptr,
        grad_grad_out,
        cutoff_upper,
        num_nodes,
        feature_dim,
        BLOCK_F=BLOCK_F,
        FILTER_FP16=filter_fp16,
        GRAD_EDGE_FP16=grad_edge_fp16,
    )

    return grad_grad_out


def setup_context_grad_grad_out_grad_edge_weight_fused_cfconv(
    ctx, inputs, output
):
    raise NotImplementedError  # TODO: implment setup context grad_x_grad_edge_weight_fused_cfconv


def backward_grad_grad_out_grad_edge_weight_fused_cfconv(ctx, grad_output):
    raise NotImplementedError  # TODO: implment backward grad_x_grad_edge_weight_fused_cfconv


grad_grad_out_grad_edge_weight_fused_cfconv.register_autograd(
    backward_grad_grad_out_grad_edge_weight_fused_cfconv,
    setup_context=setup_context_grad_grad_out_grad_edge_weight_fused_cfconv,
)


@grad_grad_out_grad_edge_weight_fused_cfconv.register_kernel("cpu")
def cpu_grad_grad_out_grad_edge_weight_fused_cfconv(
    x: torch.Tensor,
    filter: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    grad_edge_out: torch.Tensor,
    dst_perm: torch.Tensor,
    dst_ptr: torch.Tensor,
    num_nodes: int,
    cutoff_upper: float,
    grad_edge_dtype: torch.dtype = None,
) -> torch.Tensor:
    raise NotImplementedError  # FIXME: implement cpu fallback for grad_grad_out_grad_edge_weight_fused_cfconv


# ============================================================================
# grad_filter_grad_edge_weight_fused_cfconv
# ============================================================================



# ============================================================================
# grad_edge_weight_grad_edge_weight_fused_cfconv
# ============================================================================