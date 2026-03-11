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

@triton.jit
def grad_x_grad_filters_fused_cfconv_kernel(
    # Input pointers
    grad_output_ptr,  # [num_nodes, feature_dim]
    edge_weight_ptr,  # [num_edges]
    edge_dst_ptr,  # [num_edges]
    grad_x_grad_filters_ptr,  # [num_edges, feature_dim] - OUTPUT (FP32 or FP16)
    # Cutoff parameters
    cutoff_upper,
    # Sizes
    num_edges,
    feature_dim,
    # Block size
    BLOCK_F: tl.constexpr,
    OUTPUT_FP16: tl.constexpr,  # Whether to output FP16
):
    """
    Fused kernel for grad_filters computation in CFConv backward pass.

    Computes:
        grad_x_grad_filters[e] = grad_output[dst[e]] * cutoff(dist[e])

    Fuses:
        1. Gather grad_output[edge_dst]
        2. Cutoff computation
        3. Elementwise multiply

    Supports FP16 output when OUTPUT_FP16=True (matches filters dtype).
    Computation is always done in FP32 for numerical stability.
    """
    edge_idx = tl.program_id(axis=0)

    if edge_idx >= num_edges:
        return

    # Load edge info
    dst_node = tl.load(edge_dst_ptr + edge_idx)
    dist = tl.load(edge_weight_ptr + edge_idx)

    # Compute cutoff inline (CosineCutoff formula)
    C = _cosine_cutoff(dist, cutoff_upper)

    # Process features in blocks
    for f_start in range(0, feature_dim, BLOCK_F):
        f_offsets = f_start + tl.arange(0, BLOCK_F)
        f_mask = f_offsets < feature_dim



        # Gather grad_output[dst]
        grad_j = tl.load(
            grad_output_ptr + dst_node * feature_dim + f_offsets,
            mask=f_mask,
            other=0.0,
        )

        # Fused multiply: grad * C (in FP32)
        grad_filters = grad_j * C

        # Store result (convert to FP16 if needed)
        if OUTPUT_FP16:
            grad_filters = grad_filters.to(tl.float16)
        tl.store(
            grad_x_grad_filters_ptr + edge_idx * feature_dim + f_offsets,
            grad_filters,
            mask=f_mask,
        )


@triton_op("mlcg_kernels::grad_x_grad_filters_fused_cfconv", mutates_args={})
@ensure_contiguous
def grad_x_grad_filters_fused_cfconv(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    cutoff_upper: float,
    out_dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Compute grad_filters in a single fused kernel.

    grad_x_grad_filters[e] = grad_output[dst[e]] * cutoff(dist[e])

    Parameters
    ----------
    x : torch.Tensor
        Node features [num_nodes, feature_dim]
    grad_output : torch.Tensor
        Gradient of output [num_nodes, feature_dim]
    edge_weight : torch.Tensor
        Edge weights (distances) [num_edges]
    edge_src : torch.Tensor
        Source node indices [num_edges]
    edge_dst : torch.Tensor
        Destination node indices [num_edges]
    cutoff_upper : float
        Upper cutoff distance
    out_dtype : torch.dtype, optional
        Output dtype. If None, uses x.dtype. Supports FP32 or FP16.

    Returns
    -------
    torch.Tensor
        grad_filters [num_edges, feature_dim]
    """
    feature_dim = x.shape[1]
    num_edges = edge_src.shape[0]

    # Default output dtype is x.dtype
    if out_dtype is None:
        out_dtype = x.dtype

    grad_x_grad_filters = torch.empty(
        num_edges, feature_dim, device=x.device, dtype=out_dtype
    ).contiguous()

    if num_edges == 0:
        return grad_x_grad_filters

    BLOCK_F = min(128, triton.next_power_of_2(feature_dim))
    grid = (num_edges,)

    # Determine if output should be FP16
    output_fp16 = out_dtype == torch.float16

    wrap_triton(grad_x_grad_filters_fused_cfconv_kernel)[grid](
        grad_output,
        edge_weight,
        edge_dst,
        grad_x_grad_filters,
        cutoff_upper,
        num_edges,
        feature_dim,
        BLOCK_F=BLOCK_F,
        OUTPUT_FP16=output_fp16,
    )

    return grad_x_grad_filters


def setup_context_grad_x_grad_filters_fused_cfconv(ctx, inputs, output):
    raise NotImplementedError

def backward_grad_x_grad_filters_fused_cfconv(ctx, grad_output):
    raise NotImplementedError

grad_x_grad_filters_fused_cfconv.register_autograd(
    backward_grad_x_grad_filters_fused_cfconv, setup_context=setup_context_grad_x_grad_filters_fused_cfconv
)

@grad_x_grad_filters_fused_cfconv.register_kernel("cpu")
def grad_x_grad_filters_fused_cfconv(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    cutoff_upper: float,
    out_dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    CPU fallback for fused_grad_filters
    """
    C = 0.5 * (torch.cos(edge_weight * torch.pi / cutoff_upper) + 1)
    C = C * (edge_weight < cutoff_upper).float()
    grad_filters = grad_output[edge_dst] * C.unsqueeze(-1)

    return grad_filters

# ============================================================================
# grad_grad_out_grad_filters_fused_cfconv
# ============================================================================

@triton.jit
def grad_grad_out_grad_filters_fused_cfconv_kernel(
    # Input pointers
    x_ptr,  # [num_nodes, feature_dim]
    edge_weight_ptr,  # [num_edges]
    edge_src_ptr,  # [num_edges]
    grad_filters_ptr,  # [num_edges, feature_dim] - OUTPUT (FP32 or FP16)
    # Cutoff parameters
    cutoff_upper,
    # Sizes
    num_edges,
    feature_dim,
    # Block size
    BLOCK_F: tl.constexpr,
    OUTPUT_FP16: tl.constexpr,  # Whether to output FP16
):
    """
    Fused kernel for grad_filters computation in CFConv backward pass.

    Computes:
        grad_filters[e] = x[src[e]] * cutoff(dist[e])

    Fuses:
        1. Gather x[edge_src]
        2. Cutoff computation
        3. Elementwise multiply

    Supports FP16 output when OUTPUT_FP16=True (matches filters dtype).
    Computation is always done in FP32 for numerical stability.
    """
    edge_idx = tl.program_id(axis=0)

    if edge_idx >= num_edges:
        return

    # Load edge info
    src_node = tl.load(edge_src_ptr + edge_idx)
    dist = tl.load(edge_weight_ptr + edge_idx)

    # Compute cutoff inline (CosineCutoff formula)
    C = _cosine_cutoff(dist, cutoff_upper)

    # Process features in blocks
    for f_start in range(0, feature_dim, BLOCK_F):
        f_offsets = f_start + tl.arange(0, BLOCK_F)
        f_mask = f_offsets < feature_dim

        # Gather x[src]
        x_j = tl.load(
            x_ptr + src_node * feature_dim + f_offsets, mask=f_mask, other=0.0
        )


        # Fused multiply: x * grad * C (in FP32)
        grad_filters = x_j  * C

        # Store result (convert to FP16 if needed)
        if OUTPUT_FP16:
            grad_filters = grad_filters.to(tl.float16)
        tl.store(
            grad_filters_ptr + edge_idx * feature_dim + f_offsets,
            grad_filters,
            mask=f_mask,
        )


@triton_op("mlcg_kernels::grad_grad_out_grad_filters_fused_cfconv", mutates_args={})
@ensure_contiguous
def grad_grad_out_grad_filters_fused_cfconv(
    x: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    cutoff_upper: float,
    out_dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Compute grad_filters in a single fused kernel.

    grad_filters[e] = x[src[e]] * grad_output[dst[e]] * cutoff(dist[e])

    Parameters
    ----------
    x : torch.Tensor
        Node features [num_nodes, feature_dim]
    grad_output : torch.Tensor
        Gradient of output [num_nodes, feature_dim]
    edge_weight : torch.Tensor
        Edge weights (distances) [num_edges]
    edge_src : torch.Tensor
        Source node indices [num_edges]
    edge_dst : torch.Tensor
        Destination node indices [num_edges]
    cutoff_upper : float
        Upper cutoff distance
    out_dtype : torch.dtype, optional
        Output dtype. If None, uses x.dtype. Supports FP32 or FP16.

    Returns
    -------
    torch.Tensor
        grad_filters [num_edges, feature_dim]
    """
    feature_dim = x.shape[1]
    num_edges = edge_src.shape[0]

    # Default output dtype is x.dtype
    if out_dtype is None:
        out_dtype = x.dtype

    grad_grad_out_grad_filters = torch.empty(
        num_edges, feature_dim, device=x.device, dtype=out_dtype
    ).contiguous()

    if num_edges == 0:
        return grad_grad_out_grad_filters

    BLOCK_F = min(128, triton.next_power_of_2(feature_dim))
    grid = (num_edges,)

    # Determine if output should be FP16
    output_fp16 = out_dtype == torch.float16

    wrap_triton(grad_grad_out_grad_filters_fused_cfconv_kernel)[grid](
        x,
        edge_weight,
        edge_src,
        grad_grad_out_grad_filters,
        cutoff_upper,
        num_edges,
        feature_dim,
        BLOCK_F=BLOCK_F,
        OUTPUT_FP16=output_fp16,
    )

    return grad_grad_out_grad_filters


def setup_context_grad_grad_out_grad_filters_fused_cfconv(ctx, inputs, output):
    raise NotImplementedError

def backward_grad_grad_out_grad_filters_fused_cfconv(ctx, grad_output):
    raise NotImplementedError

grad_grad_out_grad_filters_fused_cfconv.register_autograd(
    backward_grad_grad_out_grad_filters_fused_cfconv, setup_context=setup_context_grad_grad_out_grad_filters_fused_cfconv
)


@grad_grad_out_grad_filters_fused_cfconv.register_kernel("cpu")
def grad_grad_out_grad_filters_fused_cfconv(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    cutoff_upper: float,
    out_dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    CPU fallback for fused_grad_filters
    """
    C = 0.5 * (torch.cos(edge_weight * torch.pi / cutoff_upper) + 1)
    C = C * (edge_weight < cutoff_upper).float()
    grad_grad_out_grad_filters = x[edge_src] * C.unsqueeze(-1)

    return grad_grad_out_grad_filters


# ============================================================================
# grad_edge_weight_grad_filters_fused_cfconv
# ============================================================================


@triton.jit
def grad_edge_weights_grad_filters_fused_cfconv_kernel(
    # Input pointers
    x_ptr,  # [num_nodes, feature_dim]
    grad_output_ptr,  # [num_nodes, feature_dim]
    edge_weight_ptr,  # [num_edges]
    edge_src_ptr,  # [num_edges]
    edge_dst_ptr,  # [num_edges]
    grad_filters_ptr,  # [num_edges, feature_dim] - OUTPUT (FP32 or FP16)
    # Cutoff parameters
    cutoff_upper,
    # Sizes
    num_edges,
    feature_dim,
    # Block size
    BLOCK_F: tl.constexpr,
    OUTPUT_FP16: tl.constexpr,  # Whether to output FP16
):
    """
    Fused kernel for grad_filters computation in CFConv backward pass.

    Computes:
        grad_filters[e] = x[src[e]] * grad_output[dst[e]] * d_cutoff_dd(dist[e])

    Fuses:
        1. Gather x[edge_src]
        2. Gather grad_output[edge_dst]
        3. Cutoff computation
        4. Elementwise multiply

    Memory savings: Eliminates two intermediate tensors (x_gathered, grad_gathered)

    Supports FP16 output when OUTPUT_FP16=True (matches filters dtype).
    Computation is always done in FP32 for numerical stability.
    """
    edge_idx = tl.program_id(axis=0)

    if edge_idx >= num_edges:
        return

    # Load edge info
    src_node = tl.load(edge_src_ptr + edge_idx)
    dst_node = tl.load(edge_dst_ptr + edge_idx)
    dist = tl.load(edge_weight_ptr + edge_idx)

    # Compute cutoff inline (CosineCutoff formula)
    dC = _d_cosine_cutoff_dd(dist, cutoff_upper)

    # Process features in blocks
    for f_start in range(0, feature_dim, BLOCK_F):
        f_offsets = f_start + tl.arange(0, BLOCK_F)
        f_mask = f_offsets < feature_dim

        # Gather x[src]
        x_j = tl.load(
            x_ptr + src_node * feature_dim + f_offsets, mask=f_mask, other=0.0
        )

        # Gather grad_output[dst]
        grad_j = tl.load(
            grad_output_ptr + dst_node * feature_dim + f_offsets,
            mask=f_mask,
            other=0.0,
        )

        # Fused multiply: x * grad * C (in FP32)
        grad_filters = x_j * grad_j * dC

        # Store result (convert to FP16 if needed)
        if OUTPUT_FP16:
            grad_filters = grad_filters.to(tl.float16)
        tl.store(
            grad_filters_ptr + edge_idx * feature_dim + f_offsets,
            grad_filters,
            mask=f_mask,
        )


@triton_op("mlcg_kernels::grad_edge_weights_grad_filters_fused_cfconv", mutates_args={})
@ensure_contiguous
def grad_edge_weights_grad_filters_fused_cfconv(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    cutoff_upper: float,
    out_dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Compute grad_filters in a single fused kernel.

    grad_filters[e] = x[src[e]] * grad_output[dst[e]] * cutoff(dist[e])

    Parameters
    ----------
    x : torch.Tensor
        Node features [num_nodes, feature_dim]
    grad_output : torch.Tensor
        Gradient of output [num_nodes, feature_dim]
    edge_weight : torch.Tensor
        Edge weights (distances) [num_edges]
    edge_src : torch.Tensor
        Source node indices [num_edges]
    edge_dst : torch.Tensor
        Destination node indices [num_edges]
    cutoff_upper : float
        Upper cutoff distance
    out_dtype : torch.dtype, optional
        Output dtype. If None, uses x.dtype. Supports FP32 or FP16.

    Returns
    -------
    torch.Tensor
        grad_filters [num_edges, feature_dim]
    """
    feature_dim = x.shape[1]
    num_edges = edge_src.shape[0]

    # Default output dtype is x.dtype
    if out_dtype is None:
        out_dtype = x.dtype

    grad_filters = torch.empty(
        num_edges, feature_dim, device=x.device, dtype=out_dtype
    ).contiguous()

    if num_edges == 0:
        return grad_filters

    BLOCK_F = min(128, triton.next_power_of_2(feature_dim))
    grid = (num_edges,)

    # Determine if output should be FP16
    output_fp16 = out_dtype == torch.float16

    wrap_triton(grad_edge_weights_grad_filters_fused_cfconv_kernel)[grid](
        x,
        grad_output,
        edge_weight,
        edge_src,
        edge_dst,
        grad_filters,
        cutoff_upper,
        num_edges,
        feature_dim,
        BLOCK_F=BLOCK_F,
        OUTPUT_FP16=output_fp16,
    )

    return grad_filters


def setup_context_grad_edge_weights_grad_filters_fused_cfconv(ctx, inputs, output):
    raise NotImplementedError

def backward_grad_edge_weights_grad_filters_fused_cfconv(ctx, grad_output):
    raise NotImplementedError

grad_edge_weights_grad_filters_fused_cfconv.register_autograd(
    backward_grad_edge_weights_grad_filters_fused_cfconv, setup_context=setup_context_grad_edge_weights_grad_filters_fused_cfconv
)


@grad_edge_weights_grad_filters_fused_cfconv.register_kernel("cpu")
def grad_edge_weights_grad_filters_fused_cfconv(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    cutoff_upper: float,
    out_dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    CPU fallback for fused_grad_filters
    """
    sin_val = torch.sin(edge_weight * torch.pi / cutoff_upper)
    dC = (
        -0.5 * (torch.pi / cutoff_upper) * sin_val 
    )
    dC = dC * (edge_weight < cutoff_upper).float()
    grad_filters = grad_output[edge_dst] * x[edge_src] * dC.unsqueeze(-1)

    return grad_filters


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

            acc += grad_grad_x * filters * C[:, None] #FIXME: check this broadcast, maybe just C

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

    return grad_grad_out #FIXME: check dtype


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
            grad_grad_x * grad_out * C[:, None], #FIXME: check this broadcast, maybe just C
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

    return grad_filters #FIXME: check dtype


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
    
    dC_dd = - 0.5 * torch.sin(edge_weight * torch.pi / cutoff_upper) * (torch.pi / cutoff_upper)
    dC_dd = dC_dd * (edge_weight < cutoff_upper).to(edge_weight.dtype)
    
    grad_edge_weight = (grad_grad_x[edge_src] * grad_out[edge_dst] * filters).sum(dim=1) * dC_dd

    return grad_edge_weight #FIXME: check dtype


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

            acc += grad_edge * filters * grad_output * C[:, None] #FIXME: check this broadcast, maybe just C

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
    edge_src: torch.Tensor,
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
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    grad_edge_out: torch.Tensor,
    src_perm: torch.Tensor,
    src_ptr: torch.Tensor,
    cutoff_upper: float,
    grad_edge_dtype: torch.dtype = None,
) -> torch.Tensor:
    
    dC_dd = - 0.5 * torch.sin(edge_weight * torch.pi / cutoff_upper) * (torch.pi / cutoff_upper)
    dC_dd = dC_dd * (edge_weight < cutoff_upper).to(edge_weight.dtype)

    grad_x = torch.zeros_like(grad_output)
    expanded = grad_edge_out * grad_output[edge_dst] * filters * dC_dd.unsqueeze(1)
    grad_x.index_add_(0, edge_src, expanded)
    return grad_x #FIXME: check dtype


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
                grad_edge_out_ptr + edge_idx * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )
            if GRAD_EDGE_FP16:
                grad_edge = grad_edge.to(tl.float32)

            acc += grad_edge * x * filters * C[:, None] #FIXME: check this broadcast, maybe just C

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
    edge_dst: torch.Tensor,
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
    edge_dst: torch.Tensor,
    grad_edge_out: torch.Tensor,
    dst_perm: torch.Tensor,
    dst_ptr: torch.Tensor,
    cutoff_upper: float,
    grad_edge_dtype: torch.dtype = None,
) -> torch.Tensor:
    
    dC_dd = - 0.5 * torch.sin(edge_weight * torch.pi / cutoff_upper) * (torch.pi / cutoff_upper)
    dC_dd = dC_dd * (edge_weight < cutoff_upper).to(edge_weight.dtype)

    grad_grad_out = torch.zeros_like(x)
    expanded = grad_edge_out * x[edge_src] * filters * dC_dd.unsqueeze(1)
    grad_grad_out.index_add_(0, edge_dst, expanded)
    return grad_grad_out #FIXME: check dtype


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
            grad_edge * grad_out * x * C[:, None], #FIXME: check this broadcast, maybe just C
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
    dC_dd = - 0.5 * torch.sin(edge_weight * torch.pi / cutoff_upper) * (torch.pi / cutoff_upper)
    dC_dd = dC_dd * (edge_weight < cutoff_upper).to(edge_weight.dtype)

    grad_filters = (grad_edge_out * grad_output[edge_dst] * x[edge_src]) * dC_dd.unsqueeze(1)

    return grad_filters #FIXME: check dtype


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
            filters = filters.to(tl.float32)
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
    
    d2C_dd2 = - 0.5 * torch.cos(edge_weight * torch.pi / cutoff_upper) * (torch.pi / cutoff_upper)**2
    d2C_dd2 = d2C_dd2 * (edge_weight < cutoff_upper).to(edge_weight.dtype)

    grad_edge_weight = (grad_edge_out * grad_output[edge_dst] * x[edge_src] * filters).sum(dim=-1) * d2C_dd2

    return grad_edge_weight # FIXME: check dtype
