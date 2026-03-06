"""
Backward kernels meant to be used for kernels in cfconv
"""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

triton_pi = tl.constexpr(3.141592653589793)

# ============================================================================
# Fused Backward Kernel for grad_filter_out
# ============================================================================


@triton.jit
def fused_grad_filter_out_kernel(
    # Input pointers
    x_ptr,  # [num_nodes, feature_dim]
    grad_output_ptr,  # [num_nodes, feature_dim]
    edge_weight_ptr,  # [num_edges]
    edge_src_ptr,  # [num_edges]
    edge_dst_ptr,  # [num_edges]
    grad_filter_out_ptr,  # [num_edges, feature_dim] - OUTPUT (FP32 or FP16)
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
    Fused kernel for grad_filter_out computation in CFConv backward pass.

    Computes:
        grad_filter_out[e] = x[src[e]] * grad_output[dst[e]] * cutoff(dist[e])

    Fuses:
        1. Gather x[edge_src]
        2. Gather grad_output[edge_dst]
        3. Cutoff computation
        4. Elementwise multiply

    Memory savings: Eliminates two intermediate tensors (x_gathered, grad_gathered)

    Supports FP16 output when OUTPUT_FP16=True (matches filter_out dtype).
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
    # C = 0.5 * (cos(dist * pi / cutoff_upper) + 1.0) * (dist < cutoff_upper)
    cos_val = tl.cos(dist * triton_pi / cutoff_upper)
    C = 0.5 * (cos_val + 1.0)
    mask_dist = dist < cutoff_upper
    C = tl.where(mask_dist, C, 0.0)

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
        grad_filter = x_j * grad_j * C

        # Store result (convert to FP16 if needed)
        if OUTPUT_FP16:
            grad_filter = grad_filter.to(tl.float16)
        tl.store(
            grad_filter_out_ptr + edge_idx * feature_dim + f_offsets,
            grad_filter,
            mask=f_mask,
        )


@triton_op("mlcg_kernels::fused_grad_filter_out", mutates_args={})
def fused_grad_filter_out(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    cutoff_upper: float,
    out_dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Compute grad_filter_out in a single fused kernel.

    grad_filter_out[e] = x[src[e]] * grad_output[dst[e]] * cutoff(dist[e])

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
        grad_filter_out [num_edges, feature_dim]
    """
    if not x.is_contiguous():
        x = x.contiguous()
    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()
    if not edge_weight.is_contiguous():
        edge_weight = edge_weight.contiguous()
    if not edge_src.is_contiguous():
        edge_src = edge_src.contiguous()
    if not edge_dst.is_contiguous():
        edge_dst = edge_dst.contiguous()

    feature_dim = x.shape[1]
    num_edges = edge_src.shape[0]

    # Default output dtype is x.dtype
    if out_dtype is None:
        out_dtype = x.dtype

    grad_filter_out = torch.empty(
        num_edges, feature_dim, device=x.device, dtype=out_dtype
    )

    if num_edges == 0:
        return grad_filter_out

    BLOCK_F = min(128, triton.next_power_of_2(feature_dim))
    grid = (num_edges,)

    # Determine if output should be FP16
    output_fp16 = out_dtype == torch.float16

    wrap_triton(fused_grad_filter_out_kernel)[grid](
        x,
        grad_output,
        edge_weight,
        edge_src,
        edge_dst,
        grad_filter_out,
        cutoff_upper,
        num_edges,
        feature_dim,
        BLOCK_F=BLOCK_F,
        OUTPUT_FP16=output_fp16,
    )

    return grad_filter_out


@fused_grad_filter_out.register_kernel("cpu")
def cpu_fused_grad_filter_out(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    cutoff_upper: float,
    out_dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    CPU fallback for fused_grad_filter_out
    """
    raise NotImplementedError  # FIXME: implement CPU compatilbe fallback


# ============================================================================
# Fused Backward Kernel for src_csr_grad_x
# ============================================================================


@triton.jit
def fused_src_csr_grad_x_kernel(
    # Input pointers
    grad_output_ptr,  # [num_nodes, feature_dim] - gradient from output (FP32)
    filter_out_ptr,  # [num_edges, feature_dim] - filter outputs (FP32 or FP16)
    edge_weight_ptr,  # [num_edges] - distances for cutoff
    edge_dst_ptr,  # [num_edges] - destination indices (original order)
    src_perm_ptr,  # [num_edges] - src-CSR permutation
    src_ptr_ptr,  # [num_nodes + 1] - src-CSR row pointers
    # Output pointer
    grad_x_ptr,  # [num_nodes, feature_dim] - gradient w.r.t. x (FP32)
    # Parameters
    cutoff_upper,
    num_nodes,
    feature_dim,
    # Block sizes
    BLOCK_F: tl.constexpr,
    FILTER_FP16: tl.constexpr,  # Whether filter_out is FP16
):
    """
    Fused src-CSR grad_x kernel for CFConv backward pass.

    Computes: grad_x[src] = sum_{e: src[e]=src} grad_output[dst[e]] * filter_out[e] * cutoff[e]

    Key features:
    - One block per SOURCE node (no atomics!)
    - 4 warps covering 128 features (BLOCK_F=128)
    - FP32 accumulation in registers
    - Single store to grad_x per source node
    - Supports FP16 filter_out (loads FP16, promotes to FP32)

    Grid: (num_nodes,) - one block per source node
    """
    src_node = tl.program_id(0)

    if src_node >= num_nodes:
        return

    # Get segment bounds from src-CSR row pointers
    seg_start = tl.load(src_ptr_ptr + src_node)
    seg_end = tl.load(src_ptr_ptr + src_node + 1)

    # Process features in blocks
    for f_start in range(0, feature_dim, BLOCK_F):
        f_offsets = f_start + tl.arange(0, BLOCK_F)
        f_mask = f_offsets < feature_dim

        # Accumulate in FP32 registers (no atomics!)
        acc = tl.zeros([BLOCK_F], dtype=tl.float32)

        # Loop over all edges originating from this source node
        for e_csr in range(seg_start, seg_end):
            # Get original edge index via src-CSR permutation
            edge_idx = tl.load(src_perm_ptr + e_csr)

            # Load destination node index
            dst_node = tl.load(edge_dst_ptr + edge_idx)

            # Load distance and compute cutoff
            dist = tl.load(edge_weight_ptr + edge_idx)
            cos_val = tl.cos(dist * triton_pi / cutoff_upper)
            C = 0.5 * (cos_val + 1.0)
            mask_dist = dist < cutoff_upper
            C = tl.where(mask_dist, C, 0.0)

            # Load filter output (FP16 or FP32)
            filter_val = tl.load(
                filter_out_ptr + edge_idx * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )
            # Promote FP16 to FP32 for computation
            if FILTER_FP16:
                filter_val = filter_val.to(tl.float32)

            # Apply cutoff: W = filter_out * cutoff
            W = filter_val * C

            # Gather grad_output[dst]
            grad_dst = tl.load(
                grad_output_ptr + dst_node * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )

            # Accumulate: grad_x[src] += grad_output[dst] * W
            acc += grad_dst * W

        # Single store per source node - no atomic needed!
        tl.store(
            grad_x_ptr + src_node * feature_dim + f_offsets,
            acc,
            mask=f_mask,
        )


@triton_op("mlcg_kernels::fused_src_csr_grad_x", mutates_args={})
def fused_src_csr_grad_x(
    grad_output: torch.Tensor,
    filter_out: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_dst: torch.Tensor,
    src_ptr: torch.Tensor,
    src_perm: torch.Tensor,
    num_nodes: int,
    cutoff_upper: float,
) -> torch.Tensor:
    """
    Compute grad_x using src-CSR segment reduce (no atomics).

    grad_x[src] = sum_{e: src[e]=src} grad_output[dst[e]] * filter_out[e] * cutoff[e]

    This replaces the atomic scatter used in the backward pass for grad_x
    with a more efficient segment-reduce that has no atomics.

    Parameters
    ----------
    grad_output : torch.Tensor
        Gradient from output [num_nodes, feature_dim], FP32
    filter_out : torch.Tensor
        Filter network output [num_edges, feature_dim], FP32 or FP16
    edge_weight : torch.Tensor
        Edge weights (distances) [num_edges]
    edge_dst : torch.Tensor
        Destination node indices [num_edges]
    src_ptr : torch.Tensor
        Src-CSR row pointers [num_nodes + 1]
    src_perm : torch.Tensor
        Src-CSR permutation [num_edges]
    num_nodes : int
        Number of nodes
    cutoff_upper : float
        Upper cutoff distance

    Returns
    -------
    torch.Tensor
        grad_x [num_nodes, feature_dim] in FP32
    """
    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()
    if not filter_out.is_contiguous():
        filter_out = filter_out.contiguous()
    if not edge_weight.is_contiguous():
        edge_weight = edge_weight.contiguous()
    if not edge_dst.is_contiguous():
        edge_dst = edge_dst.contiguous()
    if not src_ptr.is_contiguous():
        src_ptr = src_ptr.contiguous()
    if not src_perm.is_contiguous():
        src_perm = src_perm.contiguous()

    feature_dim = grad_output.shape[1]

    # Allocate output (zeros for nodes with no outgoing edges)
    grad_x = torch.zeros(
        num_nodes, feature_dim, device=grad_output.device, dtype=torch.float32
    )

    num_edges = edge_dst.shape[0]
    if num_edges == 0:
        return grad_x

    # Block size covers all 128 features (4 warps x 32 threads)
    BLOCK_F = 128

    # Auto-detect filter_out dtype
    filter_fp16 = filter_out.dtype == torch.float16

    # One block per source node
    grid = (num_nodes,)

    wrap_triton(fused_src_csr_grad_x_kernel)[grid](
        grad_output,
        filter_out,
        edge_weight,
        edge_dst,
        src_perm,
        src_ptr,
        grad_x,
        cutoff_upper,
        num_nodes,
        feature_dim,
        BLOCK_F=BLOCK_F,
        FILTER_FP16=filter_fp16,
        num_warps=4,
    )

    return grad_x


@fused_src_csr_grad_x.register_kernel("cpu")
def cpu_fused_src_csr_grad_x(
    grad_output: torch.Tensor,
    filter_out: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_dst: torch.Tensor,
    src_ptr: torch.Tensor,
    src_perm: torch.Tensor,
    num_nodes: int,
    cutoff_upper: float,
) -> torch.Tensor:
    """
    CPU fallback for fused_src_csr_grad_x
    """
    raise NotImplementedError  # FIXME: implement cpu fallback


@triton.jit
def fused_grad_edge_weight_kernel(
    # Input pointers
    x_ptr,  # [num_nodes, feature_dim]
    grad_output_ptr,  # [num_nodes, feature_dim]
    filter_out_ptr,  # [num_edges, feature_dim] - filter outputs (FP32 or FP16)
    edge_weight_ptr,  # [num_edges]
    edge_src_ptr,  # [num_edges]
    edge_dst_ptr,  # [num_edges]
    grad_edge_out_ptr,  # [num_edges, feature_dim] - OUTPUT (FP32 or FP16)
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
    Fused kernel for grad_filter_out computation in CFConv backward pass.

    Computes:
        grad_filter_out[e] = grad_output[dst[e]] * x[src[e]] * filter_out[e] * dcutoff_ddist(dist[e])

    Fuses:
        1. Gather x[edge_src]
        2. Gather grad_output[edge_dst]
        3. Gather filter_out
        4. Cutoff derivative computation
        5. Elementwise multiply

    Memory savings: Eliminates two intermediate tensors (x_gathered, grad_gathered)

    Supports FP16 output when OUTPUT_FP16=True (matches filter_out dtype).
    Computation is always done in FP32 for numerical stability.
    """
    edge_idx = tl.program_id(axis=0)

    if edge_idx >= num_edges:
        return

    # Load edge info
    src_node = tl.load(edge_src_ptr + edge_idx)
    dst_node = tl.load(edge_dst_ptr + edge_idx)
    distances = tl.load(edge_weight_ptr + edge_idx)

    # d(cutoff)/d(dist) = -0.5 * pi/cutoff_upper * sin(dist * pi / cutoff_upper)
    dist_in_range = distances < cutoff_upper
    sin_val = tl.sin(distances * triton_pi / cutoff_upper)
    d_cutoff_d_dist = -0.5 * (triton_pi / cutoff_upper) * sin_val
    d_cutoff_d_dist = tl.where(dist_in_range, d_cutoff_d_dist, 0.0)

    # Process features in blocks

    acc = tl.zeros_like(d_cutoff_d_dist)

    for f_start in range(0, feature_dim, BLOCK_F):
        f_offsets = f_start + tl.arange(0, BLOCK_F)
        f_mask = f_offsets < feature_dim

        # Gather x[src]
        x_j = tl.load(
            x_ptr + src_node * feature_dim + f_offsets, mask=f_mask, other=0.0
        )

        # Gather filter output
        filter_val = tl.load(
            filter_out_ptr + edge_idx * feature_dim + f_offsets,
            mask=f_mask,
            other=0.0,
        )

        # Gather grad_output[dst]
        grad_j = tl.load(
            grad_output_ptr + dst_node * feature_dim + f_offsets,
            mask=f_mask,
            other=0.0,
        )

        # Fused multiply: x * grad * C (in FP32)
        # grad_edge = x_j * filter_val * grad_j * d_cutoff_d_dist
        acc += tl.sum(x_j * filter_val * grad_j, axis=-1)

    grad_edge = acc * d_cutoff_d_dist
    # Store result (convert to FP16 if needed)
    if OUTPUT_FP16:
        grad_edge = grad_edge.to(tl.float16)

    tl.store(
        grad_edge_out_ptr + edge_idx,
        grad_edge,
        # mask=f_mask,
    )


@triton_op("mlcg_kernels::fused_grad_edge_weight", mutates_args={})
def fused_grad_edge_weight(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    filter_out: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    cutoff_upper: float,
    out_dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Compute grad_filter_out in a single fused kernel.

    fused_grad_edge_weight[e] = x[src[e]] * grad_output[dst[e]] * cutoff(dist[e])

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
        grad_filter_out [num_edges, feature_dim]
    """
    if not x.is_contiguous():
        x = x.contiguous()
    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()
    if not filter_out.is_contiguous():
        filter_out = filter_out.contiguous()
    if not edge_weight.is_contiguous():
        edge_weight = edge_weight.contiguous()
    if not edge_src.is_contiguous():
        edge_src = edge_src.contiguous()
    if not edge_dst.is_contiguous():
        edge_dst = edge_dst.contiguous()

    feature_dim = x.shape[1]
    num_edges = edge_src.shape[0]

    # Default output dtype is x.dtype
    if out_dtype is None:
        out_dtype = x.dtype

    grad_edge_out = torch.empty(num_edges, device=x.device, dtype=out_dtype)

    if num_edges == 0:
        return grad_edge_out

    BLOCK_F = min(128, triton.next_power_of_2(feature_dim))
    grid = (num_edges,)

    # Determine if output should be FP16
    output_fp16 = out_dtype == torch.float16
    wrap_triton(fused_grad_edge_weight_kernel)[grid](
        x,
        grad_output,
        filter_out,
        edge_weight,
        edge_src,
        edge_dst,
        grad_edge_out,
        cutoff_upper,
        num_edges,
        feature_dim,
        BLOCK_F=BLOCK_F,
        OUTPUT_FP16=output_fp16,
    )

    return grad_edge_out


@fused_grad_edge_weight.register_kernel("cpu")
def cpu_fused_grad_edge_weight(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    cutoff_upper: float,
    out_dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    CPU fallback for fused_grad_filter_out
    """
    raise NotImplementedError  # FIXME: implement CPU compatilbe fallback
