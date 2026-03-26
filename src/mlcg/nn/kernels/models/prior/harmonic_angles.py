import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton
from torch_geometric.utils import scatter

from ...utils import ensure_contiguous
from .....geometry import compute_angles_cos

@triton.jit
def harmonic_angles_edge_fwd_kernel(
    pos_ptr,          # *fp16/fp32, [N,3]
    types_ptr,        # *i32/i64,   [N]
    edges_ptr,        # *i32,       [E,2] (may be non-contiguous)
    k_ptr,
    x_ptr,        # *fp16/fp32, [T,T]
    eedge_ptr,        # *fp32,      [E]
    E: tl.constexpr,
    TYPE_DTYPE: tl.constexpr,     # 32 or 64
    K_STRIDE0: tl.constexpr,
    K_STRIDE1: tl.constexpr,
    X_0_STRIDE0: tl.constexpr,
    X_0_STRIDE1: tl.constexpr,
    EDGES_STRIDE0: tl.constexpr,
    EDGES_STRIDE1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    blk_idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = blk_idx < E

    off_i = blk_idx * EDGES_STRIDE0 + 0 * EDGES_STRIDE1
    off_j = blk_idx * EDGES_STRIDE0 + 1 * EDGES_STRIDE1
    off_k = blk_idx * EDGES_STRIDE0 + 2 * EDGES_STRIDE1

    i = tl.load(edges_ptr + off_i, mask=mask, other=0).to(tl.int32)
    j = tl.load(edges_ptr + off_j, mask=mask, other=0).to(tl.int32)
    k = tl.load(edges_ptr + off_k, mask=mask, other=0).to(tl.int32)

    if TYPE_DTYPE == 32:
        ti = tl.load(types_ptr + i, mask=mask, other=0).to(tl.int32)
        tj = tl.load(types_ptr + j, mask=mask, other=0).to(tl.int32)
        tk = tl.load(types_ptr + k, mask=mask, other=0).to(tl.int32)
    else:
        ti = tl.load(types_ptr + i, mask=mask, other=0).to(tl.int64)
        tj = tl.load(types_ptr + j, mask=mask, other=0).to(tl.int64)
        tk = tl.load(types_ptr + k, mask=mask, other=0).to(tl.int64)

    k_ener = tl.load(k_ptr + ti * K_STRIDE0 + tj*K_STRIDE1 + tk, mask=mask, other=0.).to(tl.float32)
    x_0 = tl.load(x_ptr + ti * X_0_STRIDE0 + tj*X_0_STRIDE1 + tk, mask=mask, other=0.).to(tl.float32)
    

    # positions
    i3 = i * 3
    j3 = j * 3
    k3 = k * 3

    xi0 = tl.load(pos_ptr + i3 + 0, mask=mask, other=0.).to(tl.float32)
    xi1 = tl.load(pos_ptr + i3 + 1, mask=mask, other=0.).to(tl.float32)
    xi2 = tl.load(pos_ptr + i3 + 2, mask=mask, other=0.).to(tl.float32)
    
    xj0 = tl.load(pos_ptr + j3 + 0, mask=mask, other=0.).to(tl.float32)
    xj1 = tl.load(pos_ptr + j3 + 1, mask=mask, other=0.).to(tl.float32)
    xj2 = tl.load(pos_ptr + j3 + 2, mask=mask, other=0.).to(tl.float32)

    xk0 = tl.load(pos_ptr + k3 + 0, mask=mask, other=0.).to(tl.float32)
    xk1 = tl.load(pos_ptr + k3 + 1, mask=mask, other=0.).to(tl.float32)
    xk2 = tl.load(pos_ptr + k3 + 2, mask=mask, other=0.).to(tl.float32)

    """
    dij0 = xi0 - xj0
    dij1 = xi1 - xj1
    dij2 = xi2 - xj2

    dkj0 = xk0 - xj0
    dkj1 = xk1 - xj1
    dkj2 = xk2 - xj2
    
    dot = dij0 * dkj0 + dij1 * dkj1 + dij2 * dkj2  
    norm_dij_dkj = tl.sqrt((dij0 * dij0 + dij1 * dij1 + dij2 * dij2) * (dkj0 * dkj0 + dkj1 * dkj1 + dkj2 * dkj2))
    cos_ang = dot / norm_dij_dkj
    """

       
    dij0 = xi0 - xj0
    dij1 = xi1 - xj1
    dij2 = xi2 - xj2

    dkj0 = xk0 - xj0
    dkj1 = xk1 - xj1
    dkj2 = xk2 - xj2

    # norms
    a2 = dij0*dij0 + dij1*dij1 + dij2*dij2
    b2 = dkj0*dkj0 + dkj1*dkj1 + dkj2*dkj2
    A = tl.sqrt(a2)
    B = tl.sqrt(b2)

    invAB = 1.0 / (A * B)
    dot = dij0*dkj0 + dij1*dkj1 + dij2*dkj2
    cosang = dot * invAB
    xdiff = (cosang-x_0)
    e = k_ener * xdiff * xdiff
    tl.store(eedge_ptr + blk_idx, e, mask=mask)
    
    
    


@triton.jit
def scatter_sum_edges_kernel(
    eedge_ptr,        # *fp32, [E]
    edge_batch_ptr,   # *i32/i64, [E]
    out_ptr,          # *fp32, [B]
    E: tl.constexpr,
    B: tl.constexpr,
    BATCH_DTYPE: tl.constexpr,  # 32 or 64
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    blk_idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = blk_idx < E
    e = tl.load(eedge_ptr + blk_idx, mask=mask, other=0.).to(tl.float32)

    if BATCH_DTYPE == 32:
        b = tl.load(edge_batch_ptr + blk_idx, mask=mask, other=0).to(tl.int32)
    else:
        b = tl.load(edge_batch_ptr + blk_idx, mask=mask, other=0).to(tl.int64)

    valid = mask & (b >= 0) & (b < B)
    tl.atomic_add(out_ptr + b, e, mask=valid)


@triton.jit
def harmonic_angles_pos_bwd_kernel(
    pos_ptr,          # *fp16/fp32, [N,3] (read)
    types_ptr,        # *i32/i64,   [N]
    edges_ptr,        # *i32,       [E,2]
    k_ptr,
    x_ptr,   
    edge_batch_ptr,   # *i32/i64,   [E]
    grad_y_ptr,       # *fp32,      [B]
    grad_pos_ptr,     # *fp32,      [N,3] (atomic add)
    E: tl.constexpr,
    B: tl.constexpr,
    TYPE_DTYPE: tl.constexpr,     # 32 or 64
    BATCH_DTYPE: tl.constexpr,    # 32 or 64
    K_STRIDE0: tl.constexpr,
    K_STRIDE1: tl.constexpr,
    X_0_STRIDE0: tl.constexpr,
    X_0_STRIDE1: tl.constexpr,
    EDGES_STRIDE0: tl.constexpr,
    EDGES_STRIDE1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    blk_idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = blk_idx < E

    off_i = blk_idx * EDGES_STRIDE0 + 0 * EDGES_STRIDE1
    off_j = blk_idx * EDGES_STRIDE0 + 1 * EDGES_STRIDE1
    off_k = blk_idx * EDGES_STRIDE0 + 2 * EDGES_STRIDE1

    i = tl.load(edges_ptr + off_i, mask=mask, other=0).to(tl.int32)
    j = tl.load(edges_ptr + off_j, mask=mask, other=0).to(tl.int32)
    k = tl.load(edges_ptr + off_k, mask=mask, other=0).to(tl.int32)

    if TYPE_DTYPE == 32:
        ti = tl.load(types_ptr + i, mask=mask, other=0).to(tl.int32)
        tj = tl.load(types_ptr + j, mask=mask, other=0).to(tl.int32)
        tk = tl.load(types_ptr + k, mask=mask, other=0).to(tl.int32)
    else:
        ti = tl.load(types_ptr + i, mask=mask, other=0).to(tl.int64)
        tj = tl.load(types_ptr + j, mask=mask, other=0).to(tl.int64)
        tk = tl.load(types_ptr + k, mask=mask, other=0).to(tl.int64)
    
    # grad multiplier per edge from upstream grad_y[batch]
    if BATCH_DTYPE == 32:
        b = tl.load(edge_batch_ptr + blk_idx, mask=mask, other=0).to(tl.int32)
    else:
        b = tl.load(edge_batch_ptr + blk_idx, mask=mask, other=0).to(tl.int64)
    gy = tl.load(grad_y_ptr + b, mask=mask & (b >= 0) & (b < B), other=0.).to(tl.float32)

    k_ener = tl.load(k_ptr + ti * K_STRIDE0 + tj*K_STRIDE1 + tk, mask=mask, other=0.).to(tl.float32)
    x_0 = tl.load(x_ptr + ti * X_0_STRIDE0 + tj*X_0_STRIDE1 + tk, mask=mask, other=0.).to(tl.float32)
    
    

    # positions
    i3 = i * 3
    j3 = j * 3
    k3 = k * 3

    xi0 = tl.load(pos_ptr + i3 + 0, mask=mask, other=0.).to(tl.float32)
    xi1 = tl.load(pos_ptr + i3 + 1, mask=mask, other=0.).to(tl.float32)
    xi2 = tl.load(pos_ptr + i3 + 2, mask=mask, other=0.).to(tl.float32)
    
    xj0 = tl.load(pos_ptr + j3 + 0, mask=mask, other=0.).to(tl.float32)
    xj1 = tl.load(pos_ptr + j3 + 1, mask=mask, other=0.).to(tl.float32)
    xj2 = tl.load(pos_ptr + j3 + 2, mask=mask, other=0.).to(tl.float32)

    xk0 = tl.load(pos_ptr + k3 + 0, mask=mask, other=0.).to(tl.float32)
    xk1 = tl.load(pos_ptr + k3 + 1, mask=mask, other=0.).to(tl.float32)
    xk2 = tl.load(pos_ptr + k3 + 2, mask=mask, other=0.).to(tl.float32)

    
    dij0 = xi0 - xj0
    dij1 = xi1 - xj1
    dij2 = xi2 - xj2

    dkj0 = xk0 - xj0
    dkj1 = xk1 - xj1
    dkj2 = xk2 - xj2

    # norms
    a2 = dij0*dij0 + dij1*dij1 + dij2*dij2
    b2 = dkj0*dkj0 + dkj1*dkj1 + dkj2*dkj2
    A = tl.sqrt(a2)
    B = tl.sqrt(b2)

    invAB = 1.0 / (A * B)
    dot = dij0*dkj0 + dij1*dkj1 + dij2*dkj2
    cosang = dot * invAB

    invA2 = 1.0 / (a2)   # ~ 1/A^2
    invB2 = 1.0 / (b2)   # ~ 1/B^2

    # dc/da = b/(AB) - c * a/(A^2)
    dca0 = dkj0 * invAB - cosang * dij0 * invA2
    dca1 = dkj1 * invAB - cosang * dij1 * invA2
    dca2 = dkj2 * invAB - cosang * dij2 * invA2

    # dc/db = a/(AB) - c * b/(B^2)
    dcb0 = dij0 * invAB - cosang * dkj0 * invB2
    dcb1 = dij1 * invAB - cosang * dkj1 * invB2
    dcb2 = dij2 * invAB - cosang * dkj2 * invB2

    # Multiply by upstream grad
    gi0 = gy * dca0
    gi1 = gy * dca1
    gi2 = gy * dca2

    gk0 = gy * dcb0
    gk1 = gy * dcb1
    gk2 = gy * dcb2
    
    gj0 = -(gi0 + gk0)
    gj1 = -(gi1 + gk1)
    gj2 = -(gi2 + gk2)


    # atomic add to grad_pos for i, j and k
    tl.atomic_add(grad_pos_ptr + i3 + 0, gi0, mask=mask)
    tl.atomic_add(grad_pos_ptr + i3 + 1, gi1, mask=mask)
    tl.atomic_add(grad_pos_ptr + i3 + 2, gi2, mask=mask)

    tl.atomic_add(grad_pos_ptr + j3 + 0, gj0, mask=mask)
    tl.atomic_add(grad_pos_ptr + j3 + 1, gj1, mask=mask)
    tl.atomic_add(grad_pos_ptr + j3 + 2, gj2, mask=mask)

    tl.atomic_add(grad_pos_ptr + k3 + 0, gk0, mask=mask)
    tl.atomic_add(grad_pos_ptr + k3 + 1, gk1, mask=mask)
    tl.atomic_add(grad_pos_ptr + k3 + 2, gk2, mask=mask)


#class RepulsionSigmaOverR6Fn(torch.autograd.Function):
#    @staticmethod
@triton_op("mlcg_kernels::flash_harmonic_angles", mutates_args={})
@ensure_contiguous
def flash_harmonic_angles(
        pos:torch.Tensor, 
        atom_types:torch.Tensor, 
        index_mapping:torch.Tensor, 
        mapping_batch:torch.Tensor, 
        k:torch.Tensor, 
        x_0:torch.Tensor, 
        num_graphs: int,
        block: int =  256, 
        num_warps: int = 4
    ) -> torch.Tensor:
    #assert index_mapping.dtype == torch.int32, "index_mapping must be int32 per your requirement"
    E = index_mapping.shape[0]

    eedge = torch.empty((E,), device=pos.device, dtype=torch.float32).contiguous()
    y = torch.zeros((num_graphs,), device=pos.device, dtype=torch.float32).contiguous()

    grid = (triton.cdiv(E, block),)
    wrap_triton(harmonic_angles_edge_fwd_kernel)[grid](
        pos, atom_types, index_mapping, k, x_0, eedge,
        E=E,
        TYPE_DTYPE=32 if atom_types.dtype == torch.int32 else 64,
        K_STRIDE0=k.stride(0),
        K_STRIDE1=k.stride(1),
        X_0_STRIDE0=x_0.stride(0),
        X_0_STRIDE1=x_0.stride(1),
        EDGES_STRIDE0=index_mapping.stride(0),
        EDGES_STRIDE1=index_mapping.stride(1),
        BLOCK=block,
        num_warps=num_warps,
    )
    wrap_triton(scatter_sum_edges_kernel)[grid](
        eedge, mapping_batch, y,
        E=E, B=num_graphs,
        BATCH_DTYPE=32 if mapping_batch.dtype == torch.int32 else 64,
        BLOCK=block,
        num_warps=num_warps,
    )


    return y





def setup_context_flash_harmonic_angles(ctx,inputs,output):
    # Save for backward (only what we need for grad w.r.t pos)
    (
        pos, 
        atom_types, 
        index_mapping, 
        mapping_batch, 
        k,
        x_0, 
        num_graphs,
        block,
        num_warps
    ) = inputs
    ctx.save_for_backward(pos, atom_types, index_mapping, mapping_batch, k ,x_0)
    ctx.num_graphs = num_graphs
    ctx.block = block
    ctx.num_warps = num_warps

def backward_flash_harmonic_angles(ctx, grad_y):
    (
        pos, 
        atom_types, 
        index_mapping, 
        mapping_batch, 
        k,
        x_0,
    ) = ctx.saved_tensors
    
    
    grad_y = grad_y.contiguous().to(torch.float32)

    grad_pos = torch.zeros((pos.shape[0], 3), device=pos.device, dtype=torch.float32).contiguous()

    E = index_mapping.shape[0]
    block = ctx.block
    grid = (triton.cdiv(E, block),)

    wrap_triton(harmonic_angles_pos_bwd_kernel)[grid](
        pos, 
        atom_types, 
        index_mapping, 
        k,
        x_0, 
        mapping_batch, 
        grad_y, 
        grad_pos,
        E=E, 
        B=ctx.num_graphs, 
        TYPE_DTYPE=32 if atom_types.dtype == torch.int32 else 64,
        BATCH_DTYPE=32 if mapping_batch.dtype == torch.int32 else 64,
        K_STRIDE0=k.stride(0),
        K_STRIDE1=k.stride(1),
        X_0_STRIDE0=x_0.stride(0),
        X_0_STRIDE1=x_0.stride(1),
        EDGES_STRIDE0=index_mapping.stride(0),
        EDGES_STRIDE1=index_mapping.stride(1),
        BLOCK=block,
        num_warps=ctx.num_warps,
    )

    # Return gradients for each forward input: (pos, atom_types, index_mapping, mapping_batch, sigma, num_graphs, eps, block, num_warps)
    return grad_pos, None, None, None, None, None, None, None, None

flash_harmonic_angles.register_autograd(
    backward_flash_harmonic_angles,
    setup_context=setup_context_flash_harmonic_angles,
)

@flash_harmonic_angles.register_kernel("cpu")
def cpu_flash_harmonic_angles(
    pos:torch.Tensor, 
    atom_types:torch.Tensor, 
    index_mapping:torch.Tensor, 
    mapping_batch:torch.Tensor, 
    k:torch.Tensor, 
    x_0:torch.Tensor, 
    num_graphs: int,
    block: int = 
    256, num_warps: int = 4
) -> torch.Tensor:
    interaction_types = tuple(
            atom_types[index_mapping[ii]] for ii in range(2)
        )
    distances = compute_angles_cos(pos,index_mapping)
    xdiff = (distances - x_0[interaction_types])
    y = k[interaction_types]*xdiff*xdiff
    y = scatter(y, mapping_batch, dim=0, reduce="sum", dim_size=num_graphs)
    return y 


