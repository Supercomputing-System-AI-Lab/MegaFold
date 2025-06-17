# This FusedTransition code builds on LayernormLinear -> need to import from fused_layernorm_linear.py

import torch 
import torch.nn as nn 
import triton 
import triton.language as tl
from liger_kernel.ops.utils import calculate_settings, ensure_contiguous
from liger_kernel.utils import infer_device
from megafold.kernels.fused_layernorm_linear import LayernormLinear

@triton.jit
def _swiglu_forward_kernel(x_ptr, y_ptr, stride_y, D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(axis=0).to(tl.int64)

    a_ptr = x_ptr + row_idx * stride_y * 2  # left half of x
    b_ptr = x_ptr + row_idx * stride_y * 2 + stride_y # right half of x 
    y_ptr += row_idx * stride_y

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < stride_y

    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    c_row = b_row * tl.sigmoid(b_row) * a_row # F.silu(b) * a
    
    tl.store(y_ptr + col_offsets, c_row, mask=mask)


@triton.jit
def _swiglu_backward_kernel(dy_ptr, x_ptr, stride, D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(axis=0).to(tl.int64)

    dy_ptr += row_idx * stride
    a_ptr = x_ptr + row_idx * stride * 2 
    b_ptr = x_ptr + row_idx * stride * 2 + stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < D

    dy_row = tl.load(dy_ptr + col_offsets, mask=mask, other=0.) # [(BLOCK_SIZE, )]
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0.) # [(BLOCK_SIZE, )]
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0.).to(tl.float32) # [(BLOCK_SIZE, )]

    sig_b = tl.sigmoid(b_row)
    silu_b = b_row * sig_b
    da_row = dy_row * silu_b # db = dy * silu(a) # [(BLOCK_SIZE, )]
    db_row = dy_row * (silu_b * (1 - sig_b) + sig_b) * a_row # [(BLOCK_SIZE, )]

    # Store da and db back into input tensor as buffer -> do not create new tensors
    # NOTE: triton cannot concat 2 tiles [da_row, db_row] and use only 1 store  -- tl.cat
    tl.store(a_ptr + col_offsets, da_row, mask=mask)
    tl.store(b_ptr + col_offsets, db_row, mask=mask)
    

def swiglu_forward(x):
    """ 
    Input: x = [left, right] -- left half and right half
    Output: y = F.silu(right) * left
    """
    # x: [(..., dim_inner)] -> [(M, dim_inner*2)]:  left half & right half
    ori_shape = x.shape
    D = ori_shape[-1] # D = dim_inner*2
    dim_out = D//2

    x = x.view(-1, D)
    M = x.shape[0]
    
    y = torch.empty((M, dim_out), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE, num_warps = calculate_settings(dim_out)
    _swiglu_forward_kernel[(M,)](
        x, y, y.stride(0),
        D=D, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
    )
    return x, y.view(ori_shape[:-1] + (dim_out, ))


def swiglu_backward(x, dy):
    """ 
    Input: dy
    Output: dx = [dLEFT, dRIGHT] -- left half and right half
    """
    ori_shape = dy.shape
    D = ori_shape[-1] # = dim_out
    dy = dy.view(-1, D)
    M = dy.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(D)
    _swiglu_backward_kernel[(M,)](
        dy, x, dy.stride(0),
        D=D, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
    )
    return x.view(ori_shape[:-1] + (D*2, ))


class FusedSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    @torch.amp.custom_fwd(device_type=infer_device(), cast_inputs=torch.bfloat16)
    def forward(ctx, input):
        input, output = swiglu_forward(input)
        ctx.save_for_backward(input)
        return output 

    @staticmethod
    @ensure_contiguous
    @torch.amp.custom_bwd(device_type=infer_device())
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = swiglu_backward(input, grad_output)
        return grad_input
    

class FusedSwiGLU(nn.Module):
    def forward(self, x):
        return FusedSwiGLUFunction.apply(x)

class FusedTransition(nn.Module):
    def __init__(self, dim, expansion_factor=4, include_ln=True, use_layernormlinear=True, device=None, dtype=None): 
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.dim = dim 
        dim_inner = int(dim * expansion_factor)
        self.dim_inner = dim_inner

        if include_ln: 
            if use_layernormlinear:
                self.ff = nn.Sequential(
                    LayernormLinear(dim, dim_inner * 2, has_linear_bias=False, **factory_kwargs),
                    FusedSwiGLU(),
                    nn.Linear(dim_inner, dim, bias=False, **factory_kwargs),
                )
            else:
                self.ff = nn.Sequential(
                    nn.LayerNorm(dim, **factory_kwargs),
                    nn.Linear(dim, dim_inner * 2, bias=False, **factory_kwargs),
                    FusedSwiGLU(),
                    nn.Linear(dim_inner, dim, bias=False, **factory_kwargs),
                )
        else:
            self.ff = nn.Sequential(
                nn.Linear(dim, dim_inner * 2, bias=False, **factory_kwargs),
                FusedSwiGLU(),
                nn.Linear(dim_inner, dim, bias=False, **factory_kwargs),
            )
    
    def forward(self, input):
        return self.ff(input)
    
    def extra_repr(self):
        return f"dim={self.dim}, dim_inner={self.dim_inner}"


if __name__ == "__main__":
    M, d = 384, 3072
    device = 'cuda'
    dtype = torch.bfloat16  
    x = torch.randn((M, d), device=device, dtype=dtype, requires_grad=True)
    do = torch.randn((M, d//2),  device=device, dtype=dtype)
    triton_swiglu = FusedSwiGLU()
    
    # NCU CODE: 
    o = triton_swiglu(x)
    o.backward(do, retain_graph=True)
