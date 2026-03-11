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
# grad_x_grad_filters_fused_cfconv
# ============================================================================


# ============================================================================
# grad_grad_out_grad_filters_fused_cfconv
# ============================================================================


# ============================================================================
# grad_edge_weight_grad_filters_fused_cfconv
# ============================================================================


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

            acc += grad_grad_x * filters * C[:, None]

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
    grad_grad_x: torch.Tensor,
    dst_perm: torch.Tensor,
    dst_ptr: torch.Tensor,
    cutoff_upper: float,
) -> torch.Tensor:
    raise NotImplementedError  # FIXME: implment cpu fallback for grad_grad_out_grad_x_fused_cfconv


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
            grad_grad_x * grad_out * C[:, None],
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
    raise NotImplementedError  # FIXME: implment cpu fallback for grad_filters_grad_x_fused_cfconv


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
        Computes: grad_edge_weight[e] =  (grad_grad_x[src[e]] * grad_out[dst[e]] * filters[e]).sum(axs=-1) * cutoff[e]

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
    raise NotImplementedError  # FIXME: implment cpu fallback for grad_edge_weight_grad_x_fused_cfconv


# ============================================================================
# grad_x_grad_edge_weight_fused_cfconv
# ============================================================================


@triton.jit
def grad_x_grad_edge_weight_fused_cfconv_kernel(
    # Input pointers
    grad_output_ptr,  # [num_nodes, feature_dim]
    filters_ptr,  # [num_edges, feature_dim] - filters outputs (FP32 or FP16)
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
    # Block size #FIXME: check if other casting are needed
    BLOCK_F: tl.constexpr,
    filters_FP16: tl.constexpr,
    GRAD_EDGE_FP16: tl.constexpr,  # Whether to output FP16
):
    """
    Fused kernel for grad_x computation in grad_edge_weight_fused_cfconv backward pass.

    Computes:
        Computes: grad_x[src] = sum_{e: src[e]=src} grad_edge[e] * grad_output[dst[e]] * filters[e] * d_cutoff_dd[e]


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
            filters = tl.load(
                filters_ptr + edge_idx * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )
            if filters_FP16:
                filters = filters.to(tl.float32)
            grad_edge = tl.load(
                grad_edge_out_ptr + edge_idx * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )
            if GRAD_EDGE_FP16:
                grad_edge = grad_edge.to(tl.float32)

            acc += grad_edge * filters * grad_output * C[:, None]

        tl.store(
            grad_x_ptr + target_node * feature_dim + f_offsets, acc, mask=f_mask
        )


@triton_op(
    "mlcg_kernels::grad_x_grad_edge_weight_fused_cfconv", mutates_args={}
)
@ensure_contiguous
def grad_x_grad_edge_weight_fused_cfconv(
    grad_output: torch.Tensor,
    filters: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_dst: torch.Tensor,
    grad_edge_out: torch.Tensor,
    src_perm: torch.Tensor,
    src_ptr: torch.Tensor,
    cutoff_upper: float,
    grad_edge_dtype: torch.dtype = None,
) -> torch.Tensor:

    num_nodes = grad_output.shape[0]
    feature_dim = grad_output.shape[1]
    num_edges = edge_dst.shape[0]

    # Allocate output (zeros for nodes with no outgoing edges)
    grad_x = torch.zeros(
        num_nodes, feature_dim, device=grad_output.device, dtype=torch.float32
    ).contiguous()

    if num_edges == 0:
        return grad_x

    filters_fp16 = filters.dtype == torch.float16
    grad_edge_fp16 = grad_edge_dtype == torch.float16

    BLOCK_F = 128
    grid = (num_nodes,)

    wrap_triton(grad_x_grad_edge_weight_fused_cfconv_kernel)[grid](
        grad_output,
        filters,
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
        filters_FP16=filters_fp16,
        GRAD_EDGE_FP16=grad_edge_fp16,
    )

    return grad_x


def setup_context_grad_x_grad_edge_weight_fused_cfconv(ctx, inputs, output):
    raise NotImplementedError  # TODO: implment setup context grad_x_grad_edge_weight_fused_cfconv


def backward_grad_x_grad_edge_weight_fused_cfconv(ctx, grad_grad_x):
    raise NotImplementedError  # TODO: implment backward grad_x_grad_edge_weight_fused_cfconv


grad_x_grad_edge_weight_fused_cfconv.register_autograd(
    backward_grad_x_grad_edge_weight_fused_cfconv,
    setup_context=setup_context_grad_x_grad_edge_weight_fused_cfconv,
)


@grad_x_grad_edge_weight_fused_cfconv.register_kernel("cpu")
def cpu_grad_x_grad_edge_weight_fused_cfconv(
    grad_output: torch.Tensor,
    filters: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_dst: torch.Tensor,
    grad_edge_out: torch.Tensor,
    src_perm: torch.Tensor,
    src_ptr: torch.Tensor,
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
    filters_ptr,  # [num_edges, feature_dim] - filters outputs (FP32 or FP16)
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
    # Block size #FIXME: check if other casting are needed
    BLOCK_F: tl.constexpr,
    filters_FP16: tl.constexpr,
    GRAD_EDGE_FP16: tl.constexpr,  # Whether to output FP16
):
    """
    Fused kernel for grad_grad_out computation in grad_edge_weight_fused_cfconv backward pass.

    Computes:
        Computes: grad_grad_out[dst] = sum_{e: dst[e]=dst} grad_edge[e] * x[src[e]] * filters[e] * d_cutoff_dd[e]


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
            filters = tl.load(
                filters_ptr + edge_idx * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )
            if filters_FP16:
                filters = filters.to(tl.float32)
            grad_edge = tl.load(
                grad_edge_out_ptr + edge_idx * BLOCK_F + f_offsets,
                mask=f_mask,
                other=0.0,
            )
            if GRAD_EDGE_FP16:
                grad_edge = grad_edge.to(tl.float32)

            acc += grad_edge * x * filters * C[:, None]

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
    filters: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    grad_edge_out: torch.Tensor,
    dst_perm: torch.Tensor,
    dst_ptr: torch.Tensor,
    cutoff_upper: float,
    grad_edge_dtype: torch.dtype = None,
) -> torch.Tensor:

    num_nodes = x.shape[0]
    num_edges = edge_src.shape[0]
    feature_dim = x.shape[1]

    # Allocate output (zeros for nodes with no outgoing edges)
    grad_grad_out = torch.zeros(
        num_nodes, feature_dim, device=x.device, dtype=torch.float32
    ).contiguous()

    if num_edges == 0:
        return grad_grad_out

    filters_fp16 = filters.dtype == torch.float16
    grad_edge_fp16 = grad_edge_dtype == torch.float16

    BLOCK_F = 128
    grid = (num_nodes,)

    wrap_triton(grad_grad_out_grad_edge_weight_fused_cfconv_kernel)[grid](
        x,
        filters,
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
        filters_FP16=filters_fp16,
        GRAD_EDGE_FP16=grad_edge_fp16,
    )

    return grad_grad_out


def setup_context_grad_grad_out_grad_edge_weight_fused_cfconv(
    ctx, inputs, output
):
    raise NotImplementedError  # TODO: implment setup context grad_x_grad_edge_weight_fused_cfconv


def backward_grad_grad_out_grad_edge_weight_fused_cfconv(
    ctx, grad_grad_grad_out
):
    raise NotImplementedError  # TODO: implment backward grad_x_grad_edge_weight_fused_cfconv


grad_grad_out_grad_edge_weight_fused_cfconv.register_autograd(
    backward_grad_grad_out_grad_edge_weight_fused_cfconv,
    setup_context=setup_context_grad_grad_out_grad_edge_weight_fused_cfconv,
)


@grad_grad_out_grad_edge_weight_fused_cfconv.register_kernel("cpu")
def cpu_grad_grad_out_grad_edge_weight_fused_cfconv(
    x: torch.Tensor,
    filters: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    grad_edge_out: torch.Tensor,
    dst_perm: torch.Tensor,
    dst_ptr: torch.Tensor,
    cutoff_upper: float,
    grad_edge_dtype: torch.dtype = None,
) -> torch.Tensor:
    raise NotImplementedError  # FIXME: implement cpu fallback for grad_grad_out_grad_edge_weight_fused_cfconv


# ============================================================================
# grad_filters_grad_edge_weight_fused_cfconv
# ============================================================================


@triton.jit
def grad_filters_grad_edge_weight_fused_cfconv_kernel(
    # Input pointers
    x_ptr,  # [num_nodes, feature_dim]
    grad_output_ptr,  # [num_nodes, feature_dim]
    edge_weight_ptr,  # [num_edges]
    edge_src_ptr,  # [num_edges]
    edge_dst_ptr,  # [num_edges]
    grad_edge_out_ptr,  # [num_edges, feature_dim] - OUTPUT (FP32 or FP16)
    # Output pointers
    grad_filters_ptr,  # [num_nodes, feature_dim]
    # Cutoff parameters
    cutoff_upper,
    # Sizes
    num_edges,
    feature_dim,
    # Block size #FIXME: check if other casting are needed
    BLOCK_F: tl.constexpr,
    GRAD_EDGE_FP16: tl.constexpr,  # Whether to output FP16
):
    """
    Fused kernel for grad_filters computation in grad_edge_weight_fused_cfconv backward pass.

    Computes:
        Computes: grad_filters[e] =  grad_edge[e] * grad_output[dst[e]] * x[src[e]] * d_cutoff_dd[e]

    """
    edge_idx = tl.program_id(axis=0)

    if edge_idx >= num_edges:
        return

    src_node = tl.load(edge_src_ptr + edge_idx)
    dst_node = tl.load(edge_dst_ptr + edge_idx)
    dist = tl.load(edge_weight_ptr + edge_idx)
    C = _d_cosine_cutoff_dd(dist, cutoff_upper)

    for f_start in range(0, feature_dim, BLOCK_F):
        f_offset = f_start + tl.arange(0, BLOCK_F)
        f_mask = f_offset < feature_dim

        grad_out = tl.load(
            grad_output_ptr + dst_node * feature_dim + f_offset,
            mask=f_mask,
            other=0.0,
        )
        x = tl.load(
            x_ptr + src_node * feature_dim + f_offset, mask=f_mask, other=0.0
        )
        grad_edge = tl.load(
            grad_edge_out_ptr + edge_idx * feature_dim + f_offset,
            mask=f_mask,
            other=0.0,
        )
        if GRAD_EDGE_FP16:
            grad_edge = grad_edge.to(tl.float32)
        tl.store(
            grad_filters_ptr + edge_idx * feature_dim + f_offset,
            grad_edge * grad_out * x * C[:, None],
            mask=f_mask,
        )


@triton_op(
    "mlcg_kernels::grad_filters_grad_edge_weight_fused_cfconv", mutates_args={}
)
@ensure_contiguous
def grad_filters_grad_edge_weight_fused_cfconv(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    grad_edge_out: torch.Tensor,
    cutoff_upper: float,
    grad_edge_dtype: torch.dtype = None,
) -> torch.Tensor:

    feature_dim = x.shape[1]
    num_edges = edge_src.shape[0]

    # Allocate output (zeros for nodes with no outgoing edges)
    grad_filters = torch.zeros(
        num_edges, feature_dim, device=x.device, dtype=torch.float32
    ).contiguous()  # FIXME: check dtype is correct

    if num_edges == 0:
        return grad_filters

    grad_edge_fp16 = grad_edge_dtype == torch.float16

    BLOCK_F = 128
    grid = (num_edges,)

    wrap_triton(grad_filters_grad_edge_weight_fused_cfconv_kernel)[grid](
        x,
        grad_output,
        edge_weight,
        edge_src,
        edge_dst,
        grad_edge_out,
        grad_filters,
        cutoff_upper,
        num_edges,
        feature_dim,
        BLOCK_F=BLOCK_F,
        GRAD_EDGE_FP16=grad_edge_fp16,
    )

    return grad_filters


def setup_context_grad_filters_grad_edge_weight_fused_cfconv(
    ctx, inputs, output
):
    raise NotImplementedError  # TODO: implment setup context grad_filters_grad_edge_weight_fused_cfconv


def backward_grad_filters_grad_edge_weight_fused_cfconv(ctx, grad_grad_filters):
    raise NotImplementedError  # TODO: implment backward grad_filters_grad_edge_weight_fused_cfconv


grad_filters_grad_edge_weight_fused_cfconv.register_autograd(
    backward_grad_filters_grad_edge_weight_fused_cfconv,
    setup_context=setup_context_grad_filters_grad_edge_weight_fused_cfconv,
)


@grad_filters_grad_edge_weight_fused_cfconv.register_kernel("cpu")
def cpu_grad_filters_grad_edge_weight_fused_cfconv(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    grad_edge_out: torch.Tensor,
    cutoff_upper: float,
    grad_edge_dtype: torch.dtype = None,
) -> torch.Tensor:
    raise NotImplementedError  # FIXME: implement cpu fallback for grad_filters_grad_edge_weight_fused_cfconv


# ============================================================================
# grad_edge_weight_grad_edge_weight_fused_cfconv
# ============================================================================


@triton.jit
def grad_edge_weight_grad_edge_weight_fused_cfconv_kernel(
    # Input pointers
    x_ptr,  # [num_nodes, feature_dim]
    grad_output_ptr,  # [num_nodes, feature_dim]
    filters_ptr,  # [num_edges, feature_dim]
    edge_weight_ptr,  # [num_edges]
    edge_src_ptr,  # [num_edges]
    edge_dst_ptr,  # [num_edges]
    grad_edge_out_ptr,  # [num_edges, feature_dim] - OUTPUT (FP32 or FP16)
    # Output pointers
    grad_edge_weight_ptr,  # [num_nodes, feature_dim]
    # Cutoff parameters
    cutoff_upper,
    # Sizes
    num_edges,
    feature_dim,
    # Block size #FIXME: check if other casting are needed
    BLOCK_F: tl.constexpr,
    filters_FP16: tl.constexpr,
    GRAD_EDGE_FP16: tl.constexpr,  # Whether to output FP16
):
    """
    Fused kernel for grad_edge_weight computation in grad_edge_weight_fused_cfconv backward pass.

    Computes:
        Computes: grad_edge_weight[e] =  (grad_edge[e] * grad_output[dst[e]] * x[src[e]] * filters[e]).sum(axis=-1) * d2_cutoff_dd2[e]

    """
    edge_idx = tl.program_id(axis=0)

    if edge_idx >= num_edges:
        return

    src_node = tl.load(edge_src_ptr + edge_idx)
    dst_node = tl.load(edge_dst_ptr + edge_idx)
    dist = tl.load(edge_weight_ptr + edge_idx)
    C = _d2_cosine_cutoff_dd2(dist, cutoff_upper)

    acc = tl.zeros_like(C)

    for f_start in range(0, feature_dim, BLOCK_F):
        f_offset = f_start + tl.arange(0, BLOCK_F)
        f_mask = f_offset < feature_dim

        grad_out = tl.load(
            grad_output_ptr + dst_node * feature_dim + f_offset,
            mask=f_mask,
            other=0.0,
        )
        x = tl.load(
            x_ptr + src_node * feature_dim + f_offset, mask=f_mask, other=0.0
        )
        filters = tl.load(
            filters_ptr + edge_idx * feature_dim + f_offset,
            mask=f_mask,
            other=0.0,
        )
        grad_edge = tl.load(
            grad_edge_out_ptr + edge_idx * feature_dim + f_offset,
            mask=f_mask,
            other=0.0,
        )
        if filters_FP16:
            filterss = filterss.to(tl.float32)
        if GRAD_EDGE_FP16:
            grad_edge = grad_edge.to(tl.float32)

        acc += tl.sum(grad_edge * grad_out * x * filters, axis=-1)

    tl.store(grad_edge_weight_ptr + edge_idx, acc * C)


@triton_op(
    "mlcg_kernels::grad_edge_weight_grad_edge_weight_fused_cfconv",
    mutates_args={},
)
@ensure_contiguous
def grad_edge_weight_grad_edge_weight_fused_cfconv(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    filters: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    grad_edge_out: torch.Tensor,
    cutoff_upper: float,
    grad_edge_dtype: torch.dtype = None,
) -> torch.Tensor:

    feature_dim = x.shape[1]
    num_edges = edge_src.shape[0]

    # Allocate output (zeros for nodes with no outgoing edges)
    grad_edge_weight = torch.zeros(
        num_edges, device=x.device, dtype=torch.float32
    ).contiguous()

    if num_edges == 0:
        return grad_edge_weight

    filters_fp16 = filters.dtype == torch.float16
    grad_edge_fp16 = grad_edge_dtype == torch.float16

    BLOCK_F = 128
    grid = (num_edges,)

    wrap_triton(grad_edge_weight_grad_edge_weight_fused_cfconv_kernel)[grid](
        x,
        grad_output,
        filters,
        edge_weight,
        edge_src,
        edge_dst,
        grad_edge_out,
        grad_edge_weight,
        cutoff_upper,
        num_edges,
        feature_dim,
        BLOCK_F=BLOCK_F,
        filters_FP16=filters_fp16,
        GRAD_EDGE_FP16=grad_edge_fp16,
    )

    return grad_edge_weight


def setup_context_grad_edge_weight_grad_edge_weight_fused_cfconv(
    ctx, inputs, output
):
    raise NotImplementedError  # TODO: implment setup context grad_edge_weight_grad_edge_weight_fused_cfconv


def backward_grad_edge_weight_grad_edge_weight_fused_cfconv(
    ctx, grad_grad_edge_weight
):
    raise NotImplementedError  # TODO: implment backward grad_edge_weight_grad_edge_weight_fused_cfconv


grad_edge_weight_grad_edge_weight_fused_cfconv.register_autograd(
    backward_grad_edge_weight_grad_edge_weight_fused_cfconv,
    setup_context=setup_context_grad_edge_weight_grad_edge_weight_fused_cfconv,
)


@grad_edge_weight_grad_edge_weight_fused_cfconv.register_kernel("cpu")
def cpu_grad_edge_weight_grad_edge_weight_fused_cfconv(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    filters: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    grad_edge_out: torch.Tensor,
    cutoff_upper: float,
    grad_edge_dtype: torch.dtype = None,
) -> torch.Tensor:
    raise NotImplementedError  # FIXME: implement cpu fallback for grad_filters_grad_edge_weight_fused_cfconv
