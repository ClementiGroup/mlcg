import triton
import triton.language as tl


@triton.jit
def _backnorm(gbx, gby, gbz, bx, by, bz, r_len):
    """
    Helper function to compute backward for normalization of a vector
    Forward:  b = r / ||r||
    Backward: grad_b = (gb - (gb . b) * b) / ||r||
    Paramenters:
    -----------
        gbx, gby, gbz: gradient components of the output vector b
        bx, by, bz: components of the normalized vector b
        r_len: length of the original vector r
    """
    dot = gbx * bx + gby * by + gbz * bz
    grad_x = tl.div_rn(gbx - dot * bx, r_len)
    grad_y = tl.div_rn(gby - dot * by, r_len)
    grad_z = tl.div_rn(gbz - dot * bz, r_len)
    return grad_x, grad_y, grad_z
