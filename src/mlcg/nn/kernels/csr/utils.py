import torch
from typing import Dict

from .csr_format_builder import build_csr_index


def build_csr_representation_from_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> Dict[str, torch.Tensor]:
    """
    Build complete CSR representation from edge_index matrix.

    Constructs both destination-based and source-based CSR indices from a
    COO edge list. This enables efficient forward pass (scatter-add by
    destination) and backward pass (gradient aggregation by source).

    Parameters
    ----------
    edge_index : torch.Tensor
        COO format edge list [2, num_edges] where row 0 is source indices
        and row 1 is destination indices
    num_nodes : int
        Total number of nodes in the graph

    Returns
    -------
    dict
        Dictionary containing:
        - 'edge_src': Source indices [num_edges]
        - 'edge_dst': Destination indices [num_edges]
        - 'dst_ptr': CSR row pointers for destination [num_nodes + 1]
        - 'csr_perm': Permutation sorting edges by destination [num_edges]
        - 'src_ptr': CSR row pointers for source [num_nodes + 1]
        - 'src_perm': Permutation sorting edges by source [num_edges]
    """
    edge_src = edge_index[0].contiguous()
    edge_dst = edge_index[1].contiguous()

    csr_representation = {
        "edge_src": edge_src,
        "edge_dst": edge_dst,
    }

    dst_ptr, csr_perm = build_csr_index(edge_dst, num_nodes)
    csr_representation["dst_ptr"] = dst_ptr
    csr_representation["csr_perm"] = csr_perm

    src_ptr, src_perm = build_csr_index(edge_src, num_nodes)
    csr_representation["src_ptr"] = src_ptr
    csr_representation["src_perm"] = src_perm

    return csr_representation
