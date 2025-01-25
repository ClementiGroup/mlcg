from typing import Optional
import torch
from os import path

# This means we are compiling every time when loading this module
# if there is no cache. Ideal for debugging but not release
# TODO: set a proper "TORCH_CUDA_ARCH_LIST" to remove the warning
# ref: https://github.com/pytorch/extension-cpp/, Issue #71
# TODO: a proper setup script to install a compiled version
# TODO: check whether the module has already been installed. If so, then skip
from torch.utils.cpp_extension import load
mc = load(name='radius_kernel', 
        sources=[path.join(path.dirname(__file__), '..', 'cu', 'radius_sd.cu')],
              extra_cflags=['-O3'], 
          extra_cuda_cflags=['-O3'] 
          #extra_cuda_cflags=['-arch=compute_89', 
          #                   '-code=sm_89']
         ) 

def radius_graph(
    x: torch.Tensor,
    r: float,
    batch: Optional[torch.Tensor] = None,
    loop: bool = False,
    max_num_neighbors: int = 32,
    num_workers: int = 1,
    batch_size: Optional[int] = None,
) -> torch.Tensor:

    if x.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long, device=x.device)

    if batch_size is None:
        batch_size = 1
        if batch is not None:
            assert x.size(0) == batch.numel()
            batch_size = int(batch.max()) + 1
    assert batch_size > 0
    ptr_x: Optional[torch.Tensor] = None

    if batch_size > 1:
        assert batch is not None
        arange = torch.arange(batch_size + 1, device=x.device)
        ptr_x = torch.bucketize(arange, batch)

    return mc.radius_cuda(x, ptr_x, r,
                          max_num_neighbors,
                          not loop)
