"""
Utilities for CSR-based segment reduction kernels.

This module provides utilities to construct a CSR index from COO-formatted
edge destination indices. The CSR index enables efficient scatter-reduce
kernels by grouping edges by destination node and reducing the need for
feature-level atomic operations.

The CSR construction implemented here uses a bucket-sort-style algorithm
with two GPU kernels:

1. Histogram: count edges per destination node.
2. Fill: scatter edge indices into CSR order using atomic cursors.
"""

import torch
from typing import List
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from ..utils import ensure_contiguous


@triton.jit
def histogram_kernel(
    edge_dst_ptr,
    counts_ptr,
    num_edges,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Build a histogram of destination node degrees.

    Each thread block processes a tile of edge indices and atomically
    increments the count for each destination node in the histogram buffer.
    The output is the number of incoming edges for every destination node.
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
    Scatter COO edges into CSR order using per-node cursors.

    For each edge, atomically reserve a position in the CSR permutation
    array based on the destination node cursor and store the original
    edge index. This produces a permutation that orders edges by destination.
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < num_edges

    dst = tl.load(edge_dst_ptr + offset, mask=mask, other=0)
    pos = tl.atomic_add(cursor_ptr + dst, 1, mask=mask)
    tl.store(csr_perm_ptr + pos, offset, mask=mask)


@triton_op("mlcg_kernels::build_csr_index", mutates_args={})
@ensure_contiguous
def build_csr_index(
    edge_dst: torch.Tensor,
    num_nodes: int,
) -> List[torch.Tensor]:
    """
    Build a CSR index from COO destination indices.

    Parameters
    ----------
    edge_dst : torch.Tensor
        Destination node indices of shape [num_edges], dtype int64.
    num_nodes : int
        Number of destination nodes.

    Returns
    -------
    dst_ptr : torch.Tensor
        CSR row pointers of shape [num_nodes + 1], dtype int64.
    csr_perm : torch.Tensor
        CSR permutation array of shape [num_edges], dtype int64.

    Notes
    -----
    The CSR construction proceeds in three stages:

    1. Histogram: count edges per destination node.
    2. Prefix sum: compute CSR row pointers from node counts.
    3. Fill: place edge indices into CSR order using atomic cursors.

    The result is a CSR index suitable for scatter-reduce kernels that
    avoid per-feature atomic updates.
    """
    num_edges = edge_dst.shape[0]
    device = edge_dst.device

    if num_edges == 0:
        dst_ptr = torch.zeros(num_nodes + 1, dtype=torch.int64, device=device)
        csr_perm = torch.empty(0, dtype=torch.int64, device=device)
        return dst_ptr, csr_perm

    counts = torch.zeros(
        num_nodes, dtype=torch.int32, device=device
    ).contiguous()
    BLOCK_SIZE = 1024
    grid = ((num_edges + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    wrap_triton(histogram_kernel)[grid](
        edge_dst,
        counts,
        num_edges,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    dst_ptr = torch.zeros(
        num_nodes + 1, dtype=torch.int64, device=device
    ).contiguous()
    dst_ptr[1:] = counts.to(torch.int64).cumsum(0)

    cursor = dst_ptr[:-1].clone().to(torch.int64)
    csr_perm = torch.empty(
        num_edges, dtype=torch.int64, device=device
    ).contiguous()

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
    CPU fallback implementation for CSR index construction.

    This implementation uses PyTorch operations and is suitable for
    testing or small graphs only. It sorts edges by destination and
    computes node degrees via `torch.bincount`.
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
