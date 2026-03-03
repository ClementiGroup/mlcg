"""
CSR-based segment reduce kernels for efficient scatter-add operations.

This module provides:
1. CSR build kernels: Convert COO edge_index to CSR format using bucket sort
2. CSR segment reduce: Perform scatter-add with one block per destination node

The key insight is that CSR build uses O(E) atomics (for counting), while
the original COO scatter uses O(E×F) atomics. This is a massive reduction
when F=128 features.

Usage:
    # Build CSR (once per neighbor list update)
    dst_ptr, csr_perm = build_csr_index(edge_dst, num_nodes)

    # Use CSR for scatter-add (replaces atomic scatter)
    output = csr_segment_reduce(msg, dst_ptr, csr_perm, num_nodes)
"""

import torch
import triton
import triton.language as tl

# ============================================================================
# CSR Build Kernels
# ============================================================================


@triton.jit
def histogram_kernel(
    edge_dst_ptr,
    counts_ptr,
    num_edges,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Step 1: Count edges per destination node (histogram).

    Each thread block processes BLOCK_SIZE edges and atomically increments
    the count for each destination node.

    Total atomics: O(E) - much better than O(E×F) for scatter-add!
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < num_edges

    # Load destination indices
    dst = tl.load(edge_dst_ptr + offset, mask=mask, other=0)

    # Atomically increment counts
    tl.atomic_add(counts_ptr + dst, 1, mask=mask)


@triton.jit
def csr_fill_kernel(
    edge_dst_ptr,
    cursor_ptr,
    csr_perm_ptr,
    num_edges,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Step 3: Fill CSR permutation array using atomic cursors.

    For each edge, atomically get a position in the CSR array and store
    the original edge index there. This effectively sorts edges by
    destination node.

    Total atomics: O(E) - same as histogram.
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < num_edges

    # Load destination indices
    dst = tl.load(edge_dst_ptr + offset, mask=mask, other=0)

    # Atomically get position to write
    # cursor[dst] starts at dst_ptr[dst] and increments
    pos = tl.atomic_add(cursor_ptr + dst, 1, mask=mask)

    # Store original edge index at CSR position
    tl.store(csr_perm_ptr + pos, offset, mask=mask)


def build_csr_index(
    edge_dst: torch.Tensor,
    num_nodes: int,
    edge_src: torch.Tensor = None,
) -> tuple:
    """
    Build CSR index from COO edge_dst using bucket sort.

    This converts edge indices from COO format (unsorted) to CSR format
    (sorted by destination node). The key benefit is that CSR enables
    scatter-add without atomics in the main kernel.

    Parameters
    ----------
    edge_dst : torch.Tensor
        Destination node indices [num_edges], dtype int64
    num_nodes : int
        Number of nodes
    edge_src : torch.Tensor, optional
        Source node indices [num_edges], dtype int64. If provided, also
        returns csr_src (source indices sorted by dst).

    Returns
    -------
    dst_ptr : torch.Tensor
        CSR row pointers [num_nodes + 1], dtype int64
    csr_perm : torch.Tensor
        Permutation from CSR position to original edge index [num_edges]
    csr_src : torch.Tensor (optional)
        Source node indices sorted by destination [num_edges]
        Only returned if edge_src is provided.

    Algorithm
    ---------
    1. Histogram: count edges per destination node
    2. Prefix sum: compute dst_ptr from counts
    3. Fill: scatter edges into CSR order using atomic cursors

    Complexity: O(E) atomics for build, vs O(E×F) atomics for scatter-add.
    For E=1.9M and F=128, this is 128× fewer atomics!
    """
    num_edges = edge_dst.shape[0]
    device = edge_dst.device

    if num_edges == 0:
        dst_ptr = torch.zeros(num_nodes + 1, dtype=torch.int64, device=device)
        csr_perm = torch.empty(0, dtype=torch.int64, device=device)
        if edge_src is not None:
            return (
                dst_ptr,
                csr_perm,
                torch.empty(0, dtype=torch.int64, device=device),
            )
        return dst_ptr, csr_perm

    # Step 1: Histogram - count edges per destination
    counts = torch.zeros(num_nodes, dtype=torch.int32, device=device)

    BLOCK_SIZE = 1024
    grid = ((num_edges + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    histogram_kernel[grid](
        edge_dst,
        counts,
        num_edges,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Step 2: Prefix sum to get dst_ptr
    dst_ptr = torch.zeros(num_nodes + 1, dtype=torch.int64, device=device)
    dst_ptr[1:] = counts.to(torch.int64).cumsum(0)

    # Step 3: Fill CSR using atomic cursors
    # cursor starts at dst_ptr[:-1] and increments
    cursor = dst_ptr[:-1].clone().to(torch.int64)
    csr_perm = torch.empty(num_edges, dtype=torch.int64, device=device)

    csr_fill_kernel[grid](
        edge_dst,
        cursor,
        csr_perm,
        num_edges,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Optionally return source indices sorted by destination
    if edge_src is not None:
        csr_src = edge_src[csr_perm]
        return dst_ptr, csr_perm, csr_src

    return dst_ptr, csr_perm


# ============================================================================
# Src-CSR Build (for backward grad_x computation)
# ============================================================================


@triton.jit
def src_histogram_kernel(
    edge_src_ptr,
    counts_ptr,
    num_edges,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Count edges per SOURCE node (histogram for src-CSR).

    Similar to histogram_kernel but keyed by edge_src instead of edge_dst.
    Used for backward pass grad_x computation.
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < num_edges

    # Load source indices
    src = tl.load(edge_src_ptr + offset, mask=mask, other=0)

    # Atomically increment counts
    tl.atomic_add(counts_ptr + src, 1, mask=mask)


@triton.jit
def src_csr_fill_kernel(
    edge_src_ptr,
    cursor_ptr,
    src_perm_ptr,
    num_edges,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fill src-CSR permutation array using atomic cursors.

    For each edge, atomically get a position in the CSR array and store
    the original edge index. This sorts edges by SOURCE node.
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < num_edges

    # Load source indices
    src = tl.load(edge_src_ptr + offset, mask=mask, other=0)

    # Atomically get position to write
    pos = tl.atomic_add(cursor_ptr + src, 1, mask=mask)

    # Store original edge index at CSR position
    tl.store(src_perm_ptr + pos, offset, mask=mask)


def build_src_csr_index(
    edge_src: torch.Tensor,
    num_nodes: int,
) -> tuple:
    """
    Build src-CSR index from edge_src using bucket sort.

    This converts edge indices to CSR format sorted by SOURCE node.
    Used for backward pass grad_x computation where we need to aggregate
    gradients by source node.

    Parameters
    ----------
    edge_src : torch.Tensor
        Source node indices [num_edges], dtype int64
    num_nodes : int
        Number of nodes

    Returns
    -------
    src_ptr : torch.Tensor
        CSR row pointers [num_nodes + 1], dtype int64
    src_perm : torch.Tensor
        Permutation from CSR position to original edge index [num_edges]

    Algorithm
    ---------
    1. Histogram: count edges per source node
    2. Prefix sum: compute src_ptr from counts
    3. Fill: scatter edges into CSR order using atomic cursors

    Complexity: O(E) atomics for build.
    """
    num_edges = edge_src.shape[0]
    device = edge_src.device

    if num_edges == 0:
        src_ptr = torch.zeros(num_nodes + 1, dtype=torch.int64, device=device)
        src_perm = torch.empty(0, dtype=torch.int64, device=device)
        return src_ptr, src_perm

    # Step 1: Histogram - count edges per source
    counts = torch.zeros(num_nodes, dtype=torch.int32, device=device)

    BLOCK_SIZE = 1024
    grid = ((num_edges + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    src_histogram_kernel[grid](
        edge_src,
        counts,
        num_edges,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Step 2: Prefix sum to get src_ptr
    src_ptr = torch.zeros(num_nodes + 1, dtype=torch.int64, device=device)
    src_ptr[1:] = counts.to(torch.int64).cumsum(0)

    # Step 3: Fill CSR using atomic cursors
    cursor = src_ptr[:-1].clone().to(torch.int64)
    src_perm = torch.empty(num_edges, dtype=torch.int64, device=device)

    src_csr_fill_kernel[grid](
        edge_src,
        cursor,
        src_perm,
        num_edges,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return src_ptr, src_perm


# ============================================================================
# Fused Src-CSR Grad_x Kernel (Backward)
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

    pi = 3.141592653589793

    # Process all features in one block (BLOCK_F=128 covers all 128 features)
    f_offsets = tl.arange(0, BLOCK_F)
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
        cos_val = tl.cos(dist * pi / cutoff_upper)
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
    assert grad_output.is_cuda and grad_output.is_contiguous()
    assert filter_out.is_contiguous()
    assert edge_weight.is_contiguous()
    assert edge_dst.is_contiguous()
    assert src_ptr.is_contiguous()
    assert src_perm.is_contiguous()

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

    fused_src_csr_grad_x_kernel[grid](
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


# ============================================================================
# CSR Segment Reduce Kernels
# ============================================================================


@triton.jit
def csr_segment_reduce_kernel(
    # Input pointers
    msg_ptr,  # [num_edges, feature_dim] - messages to aggregate
    csr_perm_ptr,  # [num_edges] - permutation to original edge indices
    dst_ptr_ptr,  # [num_nodes + 1] - CSR row pointers
    # Output pointer
    output_ptr,  # [num_nodes, feature_dim]
    # Sizes
    num_nodes,
    feature_dim,
    # Block size
    BLOCK_F: tl.constexpr,
):
    """
    CSR-based segment reduce: aggregate messages to destination nodes.

    Key insight: One block per destination node means NO ATOMICS needed!
    Each block has exclusive ownership of its output row.

    Grid: (num_nodes,) - one block per destination

    For each destination node:
      1. Load segment bounds from dst_ptr
      2. Loop over all edges in segment
      3. Accumulate messages in registers
      4. Single store to output (no atomic!)

    This replaces O(E×F) atomic writes with O(N×F) regular writes.
    For E=1.9M, N=34k, F=128: 243M atomics → 4.4M stores (55× reduction)
    """
    node_idx = tl.program_id(0)

    if node_idx >= num_nodes:
        return

    # Get segment bounds from CSR row pointers
    seg_start = tl.load(dst_ptr_ptr + node_idx)
    seg_end = tl.load(dst_ptr_ptr + node_idx + 1)

    # Process features in blocks
    for f_start in range(0, feature_dim, BLOCK_F):
        f_offsets = f_start + tl.arange(0, BLOCK_F)
        f_mask = f_offsets < feature_dim

        # Accumulate in registers (no atomics!)
        acc = tl.zeros([BLOCK_F], dtype=tl.float32)

        # Loop over all edges in this segment
        for e_csr in range(seg_start, seg_end):
            # Get original edge index via permutation
            edge_idx = tl.load(csr_perm_ptr + e_csr)

            # Load message for this edge
            msg = tl.load(
                msg_ptr + edge_idx * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )
            acc += msg

        # Single store per destination - no atomic needed!
        tl.store(
            output_ptr + node_idx * feature_dim + f_offsets,
            acc,
            mask=f_mask,
        )


def csr_segment_reduce(
    msg: torch.Tensor,
    dst_ptr: torch.Tensor,
    csr_perm: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """
    Aggregate messages to destination nodes using CSR segment reduce.

    This replaces atomic scatter-add with a more efficient algorithm:
    - One block per destination node
    - No atomics in the main kernel
    - Each block accumulates in registers and writes once

    Parameters
    ----------
    msg : torch.Tensor
        Messages to aggregate [num_edges, feature_dim]
    dst_ptr : torch.Tensor
        CSR row pointers [num_nodes + 1]
    csr_perm : torch.Tensor
        Permutation from CSR to original edge indices [num_edges]
    num_nodes : int
        Number of nodes

    Returns
    -------
    torch.Tensor
        Aggregated output [num_nodes, feature_dim]
    """
    assert msg.is_cuda and msg.is_contiguous()
    assert dst_ptr.is_cuda and dst_ptr.is_contiguous()
    assert csr_perm.is_cuda and csr_perm.is_contiguous()

    num_edges, feature_dim = msg.shape

    # Allocate output
    output = torch.zeros(
        num_nodes, feature_dim, device=msg.device, dtype=msg.dtype
    )

    if num_edges == 0:
        return output

    # Choose block size for features
    BLOCK_F = min(128, triton.next_power_of_2(feature_dim))

    # One block per destination node
    grid = (num_nodes,)

    csr_segment_reduce_kernel[grid](
        msg,
        csr_perm,
        dst_ptr,
        output,
        num_nodes,
        feature_dim,
        BLOCK_F=BLOCK_F,
    )

    return output


# ============================================================================
# Fused CSR CFConv Kernel
# ============================================================================


@triton.jit
def fused_csr_cfconv_kernel(
    # Input pointers
    x_ptr,  # [num_nodes, feature_dim] - node features
    filter_out_ptr,  # [num_edges, feature_dim] - filter outputs (original order, FP32 or FP16)
    edge_weight_ptr,  # [num_edges] - distances (original order)
    edge_src_ptr,  # [num_edges] - source indices (original order)
    csr_perm_ptr,  # [num_edges] - CSR permutation
    dst_ptr_ptr,  # [num_nodes + 1] - CSR row pointers
    # Output pointer
    output_ptr,  # [num_nodes, feature_dim]
    # Parameters
    cutoff_upper,
    num_nodes,
    feature_dim,
    # Block size
    BLOCK_F: tl.constexpr,
    FILTER_FP16: tl.constexpr,  # Whether filter_out is FP16
):
    """
    Fused CSR-based CFConv: cutoff + gather + multiply + segment-reduce.

    This kernel fuses:
    1. Cutoff calculation: C = 0.5 * (cos(dist * pi / cutoff) + 1)
    2. Filter scaling: W = filter_out * C
    3. Gather: x_j = x[src]
    4. Multiply: msg = x_j * W
    5. Segment reduce: output[dst] = sum(msg) for all edges to dst

    Key difference from atomic scatter: NO ATOMICS!
    Each block processes one destination node and has exclusive write access.

    Supports both FP32 and FP16 filter_out input:
    - If FILTER_FP16=True: loads FP16, promotes to FP32 for computation
    - If FILTER_FP16=False: loads FP32 directly
    Output is always FP32.

    Grid: (num_nodes,) - one block per destination
    """
    node_idx = tl.program_id(0)

    if node_idx >= num_nodes:
        return

    # Get segment bounds from CSR row pointers
    seg_start = tl.load(dst_ptr_ptr + node_idx)
    seg_end = tl.load(dst_ptr_ptr + node_idx + 1)

    pi = 3.141592653589793

    # Process features in blocks
    for f_start in range(0, feature_dim, BLOCK_F):
        f_offsets = f_start + tl.arange(0, BLOCK_F)
        f_mask = f_offsets < feature_dim

        # Accumulate in registers (no atomics!)
        acc = tl.zeros([BLOCK_F], dtype=tl.float32)

        # Loop over all edges targeting this node
        for e_csr in range(seg_start, seg_end):
            # Get original edge index via CSR permutation
            edge_idx = tl.load(csr_perm_ptr + e_csr)

            # Load source node index
            src_node = tl.load(edge_src_ptr + edge_idx)

            # Load distance and compute cutoff
            dist = tl.load(edge_weight_ptr + edge_idx)
            cos_val = tl.cos(dist * pi / cutoff_upper)
            C = 0.5 * (cos_val + 1.0)
            mask_dist = dist < cutoff_upper
            C = tl.where(mask_dist, C, 0.0)

            # Load filter output and apply cutoff (FP16 or FP32)
            filter_val = tl.load(
                filter_out_ptr + edge_idx * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )
            # Promote FP16 to FP32 for computation
            if FILTER_FP16:
                filter_val = filter_val.to(tl.float32)
            W = filter_val * C

            # Gather source features
            x_j = tl.load(
                x_ptr + src_node * feature_dim + f_offsets,
                mask=f_mask,
                other=0.0,
            )

            # Accumulate message
            acc += x_j * W

        # Single store per destination - no atomic needed!
        tl.store(
            output_ptr + node_idx * feature_dim + f_offsets,
            acc,
            mask=f_mask,
        )


def fused_csr_cfconv(
    x: torch.Tensor,
    filter_out: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    dst_ptr: torch.Tensor,
    csr_perm: torch.Tensor,
    num_nodes: int,
    cutoff_upper: float,
) -> torch.Tensor:
    """
    Fused CSR-based CFConv operation.

    This is a drop-in replacement for fused_cutoff_gather_multiply_scatter
    that uses CSR format for efficient aggregation without atomics.

    Supports both FP32 and FP16 filter_out input (auto-detected).
    Output is always FP32.

    Parameters
    ----------
    x : torch.Tensor
        Node features [num_nodes, feature_dim]
    filter_out : torch.Tensor
        Filter outputs [num_edges, feature_dim], can be FP32 or FP16
    edge_weight : torch.Tensor
        Edge weights (distances) [num_edges]
    edge_src : torch.Tensor
        Source node indices [num_edges]
    dst_ptr : torch.Tensor
        CSR row pointers [num_nodes + 1]
    csr_perm : torch.Tensor
        CSR permutation [num_edges]
    num_nodes : int
        Number of nodes
    cutoff_upper : float
        Upper cutoff distance

    Returns
    -------
    torch.Tensor
        Output [num_nodes, feature_dim] in FP32
    """
    assert x.is_cuda and x.is_contiguous()
    assert filter_out.is_contiguous()
    assert edge_weight.is_contiguous()
    assert edge_src.is_contiguous()
    assert dst_ptr.is_contiguous()
    assert csr_perm.is_contiguous()

    feature_dim = x.shape[1]

    # Allocate output (zeros needed for nodes with no incoming edges)
    output = torch.zeros(num_nodes, feature_dim, device=x.device, dtype=x.dtype)

    num_edges = edge_src.shape[0]
    if num_edges == 0:
        return output

    # Choose block size
    BLOCK_F = min(128, triton.next_power_of_2(feature_dim))

    # Auto-detect filter_out dtype
    filter_fp16 = filter_out.dtype == torch.float16

    # One block per destination node
    grid = (num_nodes,)

    fused_csr_cfconv_kernel[grid](
        x,
        filter_out,
        edge_weight,
        edge_src,
        csr_perm,
        dst_ptr,
        output,
        cutoff_upper,
        num_nodes,
        feature_dim,
        BLOCK_F=BLOCK_F,
        FILTER_FP16=filter_fp16,
    )

    return output


# ============================================================================
# Autograd Function with Custom Backward
# ============================================================================


class FusedCSRCFConvFunction(torch.autograd.Function):
    """Autograd function for fused CSR-based CFConv.

    Supports both FP32 and FP16 filter_out:
    - Forward: auto-detects filter_out dtype, always outputs FP32
    - Backward: grad_filter_out matches filter_out dtype

    Optionally uses src-CSR for atomic-free grad_x computation when
    src_ptr and src_perm are provided.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        filter_out,
        edge_weight,
        edge_src,
        edge_dst,
        dst_ptr,
        csr_perm,
        num_nodes,
        cutoff_upper,
        src_ptr,
        src_perm,
    ):
        # Save for backward
        ctx.save_for_backward(
            x,
            filter_out,
            edge_weight,
            edge_src,
            edge_dst,
            dst_ptr,
            csr_perm,
            src_ptr,
            src_perm,
        )
        ctx.num_nodes = num_nodes
        ctx.cutoff_upper = cutoff_upper
        ctx.filter_out_dtype = filter_out.dtype  # Save dtype for backward

        return fused_csr_cfconv(
            x,
            filter_out,
            edge_weight,
            edge_src,
            dst_ptr,
            csr_perm,
            num_nodes,
            cutoff_upper,
        )

    @staticmethod
    def backward(ctx, grad_output):
        (
            x,
            filter_out,
            edge_weight,
            edge_src,
            edge_dst,
            dst_ptr,
            csr_perm,
            src_ptr,
            src_perm,
        ) = ctx.saved_tensors
        num_nodes = ctx.num_nodes
        cutoff_upper = ctx.cutoff_upper
        filter_out_dtype = ctx.filter_out_dtype

        grad_output = grad_output.contiguous()

        grad_x = None
        grad_filter_out = None

        if ctx.needs_input_grad[0]:
            # grad_x[src] += grad_output[dst] * W
            if src_ptr is not None and src_perm is not None:
                # === Src-CSR Path (no atomics!) ===
                grad_x = fused_src_csr_grad_x(
                    grad_output,
                    filter_out,
                    edge_weight,
                    edge_dst,
                    src_ptr,
                    src_perm,
                    num_nodes,
                    cutoff_upper,
                )
            else:
                # === Atomic scatter path (fallback) ===
                from flashmd.kernels.cfconv_kernels import (
                    fused_cutoff_gather_multiply_scatter,
                )

                grad_x = fused_cutoff_gather_multiply_scatter(
                    grad_output,
                    filter_out,
                    edge_weight,
                    edge_dst,  # reversed: dst becomes src
                    edge_src,  # reversed: src becomes dst
                    num_nodes,
                    cutoff_upper,
                )

        if ctx.needs_input_grad[1]:
            # grad_filter_out[e] = x[src[e]] * grad_output[dst[e]] * C[e]
            # Output dtype matches filter_out dtype (FP32 or FP16)
            from flashmd.kernels.cfconv_kernels import fused_grad_filter_out

            grad_filter_out = fused_grad_filter_out(
                x,
                grad_output,
                edge_weight,
                edge_src,
                edge_dst,
                cutoff_upper,
                out_dtype=filter_out_dtype,
            )

        return (
            grad_x,
            grad_filter_out,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def fused_csr_cfconv_autograd(
    x: torch.Tensor,
    filter_out: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    dst_ptr: torch.Tensor,
    csr_perm: torch.Tensor,
    num_nodes: int,
    cutoff_upper: float,
    src_ptr: torch.Tensor = None,
    src_perm: torch.Tensor = None,
) -> torch.Tensor:
    """
    Autograd-compatible fused CSR-based CFConv.

    This is the main entry point for using CSR-based scatter in CFConv.

    Parameters
    ----------
    src_ptr : torch.Tensor, optional
        Src-CSR row pointers [num_nodes + 1]. If provided along with src_perm,
        enables atomic-free grad_x computation in backward.
    src_perm : torch.Tensor, optional
        Src-CSR permutation [num_edges]. Required with src_ptr.
    """
    return FusedCSRCFConvFunction.apply(
        x,
        filter_out,
        edge_weight,
        edge_src,
        edge_dst,
        dst_ptr,
        csr_perm,
        num_nodes,
        cutoff_upper,
        src_ptr,
        src_perm,
    )
