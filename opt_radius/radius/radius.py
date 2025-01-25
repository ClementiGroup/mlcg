from typing import Optional
import torch

from torch.utils.cpp_extension import load
mc = load(name='radius_kernel', 
          sources=['cu/radius.cu'],
          #extra_cuda_cflags=['-arch=compute_89', 
          #                   '-code=sm_89']
         ) 

def radius(
    x: torch.Tensor,
    y: torch.Tensor,
    r: float,
    batch_x: Optional[torch.Tensor] = None,
    batch_y: Optional[torch.Tensor] = None,
    max_num_neighbors: int = 32,
    num_workers: int = 1,
    batch_size: Optional[int] = None,
    ignore_same_index: bool = True 
) -> torch.Tensor:

    if x.numel() == 0 or y.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long, device=x.device)

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y
    x, y = x.contiguous(), y.contiguous()

    if batch_size is None:
        batch_size = 1
        if batch_x is not None:
            assert x.size(0) == batch_x.numel()
            batch_size = int(batch_x.max()) + 1
        if batch_y is not None:
            assert y.size(0) == batch_y.numel()
            batch_size = max(batch_size, int(batch_y.max()) + 1)
    assert batch_size > 0

    ptr_x: Optional[torch.Tensor] = None
    ptr_y: Optional[torch.Tensor] = None

    if batch_size > 1:
        assert batch_x is not None
        assert batch_y is not None
        arange = torch.arange(batch_size + 1, device=x.device)
        ptr_x = torch.bucketize(arange, batch_x)
        ptr_y = torch.bucketize(arange, batch_y)

    return mc.radius_cuda(x, y, ptr_x, ptr_y, r,
                          max_num_neighbors,
                          ignore_same_index)

def radius_graph(
    x: torch.Tensor,
    r: float,
    batch: Optional[torch.Tensor] = None,
    max_num_neighbors: int = 32,
    loop: bool = False,
    flow: str = 'source_to_target',
    num_workers: int = 1,
    batch_size: Optional[int] = None,
) -> torch.Tensor:

    assert flow in ['source_to_target', 'target_to_source']
    edge_index = radius(x, x, r, batch, batch,
                        max_num_neighbors,
                        num_workers, batch_size, not loop)
    if flow == 'source_to_target':
        row, col = edge_index[1], edge_index[0]
    else:
        row, col = edge_index[0], edge_index[1]

    return torch.stack([row, col], dim=0)
