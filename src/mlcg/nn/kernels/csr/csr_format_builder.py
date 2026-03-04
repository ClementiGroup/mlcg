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
from typing import List
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.jit
def histogram_kernel(
    edge_dst_ptr,
    counts_ptr,
    num_edges,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Count edges per destination node (histogram).

    Each thread block processes BLOCK_SIZE edges and atomically increments
    the count for each destination node.

    Total atomics: O(E) - much better than O(E×F) for scatter-add!
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < num_edges

    dst = tl.load(edge_dst_ptr + offset, mask=mask, other=0)
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
    Fill CSR permutation array using atomic cursors.

    For each edge, atomically get a position in the CSR array and store
    the original edge index there. This effectively sorts edges by destination
    node and is the final phase of CSR build. Total atomics: O(E).
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < num_edges

    dst = tl.load(edge_dst_ptr + offset, mask=mask, other=0)
    pos = tl.atomic_add(cursor_ptr + dst, 1, mask=mask)
    tl.store(csr_perm_ptr + pos, offset, mask=mask)


@triton_op("mlcg_kernels::build_csr_index", mutates_args={})
def build_csr_index(
    edge_dst: torch.Tensor,
    num_nodes: int,
) -> List[torch.Tensor]:
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

    Returns
    -------
    dst_ptr : torch.Tensor
        CSR row pointers [num_nodes + 1], dtype int64
    csr_perm : torch.Tensor
        Permutation from CSR position to original edge index [num_edges]

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
        return dst_ptr, csr_perm

    counts = torch.zeros(num_nodes, dtype=torch.int32, device=device)
    BLOCK_SIZE = 1024
    grid = ((num_edges + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    wrap_triton(histogram_kernel)[grid](
        edge_dst,
        counts,
        num_edges,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    dst_ptr = torch.zeros(num_nodes + 1, dtype=torch.int64, device=device)
    dst_ptr[1:] = counts.to(torch.int64).cumsum(0)

    cursor = dst_ptr[:-1].clone().to(torch.int64)
    csr_perm = torch.empty(num_edges, dtype=torch.int64, device=device)

    wrap_triton(csr_fill_kernel)[grid](
        edge_dst,
        cursor,
        csr_perm,
        num_edges,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return [dst_ptr, csr_perm]


@build_csr_index.register_kernel("cpu")
def _(
    edge_dst: torch.Tensor,
    num_nodes: int,
) -> List[torch.Tensor]:
    """
    CPU fallback implementation using pure PyTorch operations.

    Much slower than GPU version - only use for testing or small graphs.
    Uses argsort and bincount which are CPU-optimized PyTorch operations.
    """
    num_edges = edge_dst.shape[0]
    device = edge_dst.device

    if num_edges == 0:
        dst_ptr = torch.zeros(num_nodes + 1, dtype=torch.int64, device=device)
        csr_perm = torch.empty(0, dtype=torch.int64, device=device)
        return dst_ptr, csr_perm

    csr_perm = torch.argsort(edge_dst)
    counts = torch.bincount(edge_dst, minlength=num_nodes)
    dst_ptr = torch.zeros(num_nodes + 1, dtype=edge_dst.dtype, device=device)
    dst_ptr[1:] = counts.to(torch.int64).cumsum(0)

    return [dst_ptr, csr_perm]
