# USAGE: replace: [LayerNorm(K, ); Linear(K, N)] with LayernormLinear(K, N)

import torch
import torch.nn as nn
from torch.nn import functional as F
import triton
import triton.language as tl
# from triton.language.math import rsqrt
from liger_kernel.ops.utils import calculate_settings
from liger_kernel.utils import infer_device
from liger_kernel.ops.utils import ensure_contiguous
from helper import calculate_config_layernorm_linear
import math 
import sys

DEFAULT_CONFIG = (16, 16, 16, 8, 1, 8, 255)

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def get_cuda_autotune_forward_config():   
    # massive search:
    return [
        triton.Config({'BLOCK_SIZE_M': BLOCK_SIZE_M, 'BLOCK_SIZE_N': BLOCK_SIZE_N, 'BLOCK_SIZE_K': BLOCK_SIZE_K, 'GROUP_SIZE_M': GROUP_SIZE_M}, num_stages=num_stages, num_warps=num_warps, maxnreg=maxnreg)
        for BLOCK_SIZE_M in [32, 64, 128]   
        for BLOCK_SIZE_N in [32, 64, 128, 256]
        for BLOCK_SIZE_K in [16, 32]
        for GROUP_SIZE_M in [4, 8]
        for num_stages in ([2, 4])
        for num_warps in [4, 8]
        for maxnreg in [96, 128, 168, 255]
    ]


def get_hip_autotune_forward_config():
    return [
        triton.Config({'BLOCK_SIZE_M': BLOCK_SIZE_M, 'BLOCK_SIZE_N': BLOCK_SIZE_N, 'BLOCK_SIZE_K': BLOCK_SIZE_K, 'GROUP_SIZE_M': GROUP_SIZE_M, 'waves_per_eu': waves_per_eu}, num_stages=num_stages, num_warps=num_warps)
        for BLOCK_SIZE_M in [64, 128, 256]   
        for BLOCK_SIZE_N in [32, 64, 128, 256]
        for BLOCK_SIZE_K in [16, 32]
        for GROUP_SIZE_M in [1, 4, 8]
        for num_stages in [2, 3, 4]
        for num_warps in [4, 8]
        for waves_per_eu in [2, 3]
    ]


def get_autotune_forward_config():
    if is_cuda():
        return get_cuda_autotune_forward_config()
    else:
        return get_hip_autotune_forward_config()


def get_cuda_autotune_backward_config():
    # massive search (M, K) parallel:
    return [
        triton.Config({'BLOCK_SIZE_M': BLOCK_SIZE_M, 'BLOCK_SIZE_K': BLOCK_SIZE_K, 'BLOCK_SIZE_N': BLOCK_SIZE_N, 'GROUP_SIZE_M': GROUP_SIZE_M}, num_stages=num_stages, num_warps=num_warps, maxnreg=maxnreg)
        for BLOCK_SIZE_M in [16, 32, 64, 128]   
        for BLOCK_SIZE_K in [16, 32, 64, 128, 256]
        for BLOCK_SIZE_N in [16, 32]
        for GROUP_SIZE_M in [4, 8]
        for num_stages in [2, 4]
        for num_warps in [4, 8]
        for maxnreg in [96, 128, 168, 196, 255]
    ]


# FOR LARGE N: -- change the autotune space to allow larger BLOCK_SIZE_N
# def get_cuda_autotune_backward_config():
#     # massive search (M, K) parallel:
#     return [
#         triton.Config({'BLOCK_SIZE_M': BLOCK_SIZE_M, 'BLOCK_SIZE_K': BLOCK_SIZE_K, 'BLOCK_SIZE_N': BLOCK_SIZE_N, 'GROUP_SIZE_M': GROUP_SIZE_M}, num_stages=num_stages, num_warps=num_warps, maxnreg=maxnreg)
#         for BLOCK_SIZE_M in [128, 256]   
#         for BLOCK_SIZE_K in [16, 32, 64, 128, 256]
#         for BLOCK_SIZE_N in [16, 32, 64]
#         for GROUP_SIZE_M in [4, 8]
#         for num_stages in [2, 4]
#         for num_warps in [4, 8]
#         for maxnreg in [128, 168, 196, 255]
#     ]

def get_hip_autotune_backward_config():
    return [
        triton.Config({'BLOCK_SIZE_M': BLOCK_SIZE_M, 'BLOCK_SIZE_N': BLOCK_SIZE_N, 'BLOCK_SIZE_K': BLOCK_SIZE_K, 'GROUP_SIZE_M': GROUP_SIZE_M, 'waves_per_eu': waves_per_eu}, num_stages=num_stages, num_warps=num_warps)
        for BLOCK_SIZE_M in [64, 128, 256]   
        for BLOCK_SIZE_N in [32, 64, 128, 256]
        for BLOCK_SIZE_K in [16, 32]
        for GROUP_SIZE_M in [1, 4, 8]
        for num_stages in [2, 3, 4]
        for num_warps in [4, 8]
        for waves_per_eu in [2, 3]
    ]

def get_autotune_backward_config():
    if is_cuda():
        return get_cuda_autotune_backward_config()
    else:
        return get_hip_autotune_backward_config()


@triton.jit
def _first_forward_kernel(
    X_ptr,  # pointer to input, shape (M, K)
    X_row_stride,  # stride of each row in input
    Mean_ptr,  # pointer to mean, shape (M,)
    Mean_row_stride,  # stride of each row in mean
    RSTD_ptr,  # pointer to rstd, shape (M,)
    RSTD_row_stride,  # stride of each row in rstd
    K,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    First kernel just calculates the mean and variance along the whole row and store back 
    """
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < K

    X_ptr += row_idx * X_row_stride
    Mean_ptr += row_idx * Mean_row_stride
    RSTD_ptr += row_idx * RSTD_row_stride

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0.0)

    mean = tl.sum(X_row, axis=0) / K
    var = tl.sum((X_row - mean) * (X_row - mean), axis=0) / K
    # rstd = rsqrt(var + eps)
    rstd = 1 / tl.sqrt(var + eps)

    tl.store(Mean_ptr, mean)
    tl.store(RSTD_ptr, rstd)
    

@triton.autotune(
    configs=get_autotune_forward_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def _second_forward_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, linear_bias_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    WEIGHT, BIAS,
    Mean_ptr, RSTD_ptr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    Mean_row_stride, RSTD_row_stride,
    has_layernorm_bias: tl.constexpr, has_linear_bias: tl.constexpr, 
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """
    Second kernel: calculate matmul. Normalize each input block loaded  
    a: [M, K] 
    b: [K, N]
    c: [M, N]
    linear_bias: [N]
    """
    pid = tl.program_id(axis=0).to(tl.int64)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    mean = tl.load(Mean_ptr + offs_m * Mean_row_stride, mask= offs_m < M, other=0.0) # (BLOCK_SIZE_M, )
    rstd = tl.load(RSTD_ptr + offs_m * RSTD_row_stride, mask= offs_m < M, other=0.0) # (BLOCK_SIZE_M, )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32) # (BLOCK_SIZE_M, BLOCK_SIZE_N)

    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)): # , num_stages=1, loop_unroll_factor=1
        # Load the next block of A and B, generate a mask by checking the K dimension.
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0) # [(BLOCK_SIZE_M, BLOCK_SIZE_K)]
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_n[None, :] < N), other=0.0) # [(BLOCK_SIZE_K, BLOCK_SIZE_N)]

        weight = tl.load(WEIGHT + k * BLOCK_SIZE_K + offs_k, mask=offs_k < K - k * BLOCK_SIZE_K, other=0.0) # [(BLOCK_SIZE_K, )]
        
        if has_layernorm_bias:
            bias = tl.load(BIAS + k * BLOCK_SIZE_K + offs_k, mask=offs_k < K - k * BLOCK_SIZE_K, other=0.0) # [(BLOCK_SIZE_K, )]
            a = ((a - mean[:, None]) * rstd[:, None]) * weight[None, :] + bias[None, :] # [(BLOCK_SIZE_M, BLOCK_SIZE_K)]
        else:
            a = ((a - mean[:, None]) * rstd[:, None]) * weight[None, :]
        
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator) # [(BLOCK_SIZE_M, BLOCK_SIZE_K)] x [(BLOCK_SIZE_K, BLOCK_SIZE_N)]

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(DTYPE) # [(BLOCK_SIZE_M, BLOCK_SIZE_N)]

    if has_linear_bias:
        linear_bias = tl.load(linear_bias_ptr + offs_n, mask=offs_n < N, other=0.0) # [(BLOCK_SIZE_N, )]
        c = c + linear_bias[None, :]
    
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.autotune(
    configs=get_autotune_backward_config(),
    key=['M', 'K', 'N'],
    # reset_to_zero=['dB', 'c1', 'c2', 'dBIAS', 'dWEIGHT'] # reset to zero after each autotune, so the values don't get accumulated across atomic_adds
)
@triton.jit
def _first_backward_kernel(
    dOUT_ptr, B, dX, dB, X, Mean, RSTD, WEIGHT, BIAS, dWEIGHT, dBIAS, c1, c2,
    M, K, N,
    stride_am, stride_an,
    stride_bk, stride_bn,
    stride_cm, stride_ck,
    stride_weight, stride_c1, stride_c2, stride_mean,
    has_layernorm_bias: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,  BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
):
    """ 
    Fuse dY = dOUT @ B^T and store temporary wdy into dX
    Fuse in dB = Y^T @ dOUT via atomic adds
    Calculate dWEIGHT, dBIAS, c1, c2
    """
    pid = tl.program_id(axis=0).to(tl.int64)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_k
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_k = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    dOUT_ptrs = dOUT_ptr + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_an) # [BLOCK_SIZE_M, BLOCK_SIZE_N]
    B_T_ptrs = B + (offs_k[None, :] * stride_bk + offs_n[:, None] * stride_bn) # trick to point to B transposed

    # Recompute x_hat
    x = (tl.load(X + (offs_m[:, None] * stride_cm + offs_k[None, :] * stride_ck), mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)) # [(BLOCK_SIZE_M, BLOCK_SIZE_K)]
    mean = tl.load(Mean + (offs_m * stride_mean), mask= offs_m < M, other=0.0) # [(BLOCK_SIZE_M, )]
    rstd = tl.load(RSTD + (offs_m * stride_mean), mask= offs_m < M, other=0.0) # [(BLOCK_SIZE_M, )] 
    x_hat = (x - mean[:, None]) * rstd[:, None]  # [(BLOCK_SIZE_M, BLOCK_SIZE_K)]
    weight = tl.load(WEIGHT + offs_k, mask = offs_k < K, other= 0.0) # [(BLOCK_SIZE_K, )]
    
    if has_layernorm_bias: 
        bias = tl.load(BIAS + offs_k, mask = offs_k < K, other= 0.0) # [(BLOCK_SIZE_K, )]
        y_T = tl.trans((x_hat * weight[None, :]) + bias[None, :]) # [(BLOCK_SIZE_K, BLOCK_SIZE_M)]
    else:
        y_T = tl.trans((x_hat * weight[None, :]))

    dy = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    
    for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)): # , num_stages=1, loop_unroll_factor=1
        dOUT = tl.load(dOUT_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
        
        # add-ons here to atomic add on dB: contention across M-blocks to write to the same address
        db = tl.dot(y_T, dOUT)
        tl.atomic_add(dB + offs_k[:, None] * N + offs_n[None, :], db.to(dB.type.element_ty), mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), sem="relaxed")
        
        b_T = tl.load(B_T_ptrs, mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0)
        dy = tl.dot(dOUT, b_T, dy)

        # Advance pointers 
        dOUT_ptrs += BLOCK_SIZE_N * stride_an
        B_T_ptrs += BLOCK_SIZE_N * stride_bn
        offs_n += BLOCK_SIZE_N  # offs_n is also changing here as well 

    wdy = dy * weight[None, :] # [(BLOCK_SIZE_M, BLOCK_SIZE_K)]

    # Accumulate c1 and c2
    c1_vals = (tl.sum(x_hat * wdy, axis=1) / K) # [(BLOCK_SIZE_M, )]
    c2_vals = (tl.sum(wdy, axis=1) / K)  # [(BLOCK_SIZE_M, )]

    # atomic add: contention across column-blocks because blocks on different columns are trying to write to the same c1 column
    tl.atomic_add(c1 + offs_m * stride_c1, c1_vals.to(c1.type.element_ty), mask=offs_m < M, sem="relaxed")
    tl.atomic_add(c2 + offs_m * stride_c2, c2_vals.to(c2.type.element_ty), mask=offs_m < M, sem="relaxed")

    dWEIGHT_vals = tl.sum(dy * x_hat, axis=0) # [(BLOCK_SIZE_K, )]
    if has_layernorm_bias:
        dBIAS_vals = tl.sum(dy, axis=0)

    # atomic add: contention across row-blocks because blocks on different rows are trying to write to the same dWEIGHT row -> this will be huge given X= [everything_else, hidden_dimension]
    # Because dY and x_hat is ready here, we can include dWEIGHT and dBIAS computation here:
    tl.atomic_add(dWEIGHT + offs_k * stride_weight, dWEIGHT_vals.to(dWEIGHT.type.element_ty), mask=offs_k < K, sem="relaxed")  # [(BLOCK_SIZE_K, )]
    if has_layernorm_bias:
        tl.atomic_add(dBIAS + offs_k * stride_weight, dBIAS_vals.to(dBIAS.type.element_ty), mask=offs_k < K, sem="relaxed") # [(BLOCK_SIZE_K, )]

    tl.store(dX + offs_m[:, None] * stride_cm + offs_k[None, :] * stride_ck, wdy.to(dX.type.element_ty), mask=((offs_m[:, None] < M) & (offs_k[None, :] < K))) # for now, dX will only store wdy


@triton.jit
def _second_backward_kernel(
    dX, X, c1, c2, Mean, RSTD,
    M, K, N,
    stride_x, stride_mean,
    BLOCK_SIZE: tl.constexpr,
):
    """ 
    Calculate dX by wdy, x_hat, c1, c2
    """
    # on each row: load current wdy in dX, load c1 and c2, perhaps recompute x_hat again 
    # then perform: real_dX = wdy - c1 * x_hat - c2
    row = tl.program_id(0).to(tl.int64)
    dX += row * stride_x
    X += row * stride_x
    c1 += row * stride_mean 
    c2 += row * stride_mean
    Mean += row * stride_mean
    RSTD += row * stride_mean
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < K

    # Recompute x_hat
    x = tl.load(X + col_offsets, mask=mask, other=0.0) # [(K, )]
    mean = tl.load(Mean) # (1, )
    rstd = tl.load(RSTD) # (1, )
    x_hat = (x - mean) * rstd # [(K, )]

    # Load c1 and c2 vals
    c1_val = tl.load(c1) # (1, )
    c2_val = tl.load(c2) # (1, )

    # compute real dx
    wdy = tl.load(dX + col_offsets, mask=mask, other=0.0) # [(K, )]
    dx = (wdy - (c1_val * x_hat + c2_val)) * rstd  # [(K, )]

    # store back dX
    tl.store(dX + col_offsets, dx, mask= mask)


def layernorm_linear_forward(a, b, linear_bias, WEIGHT, BIAS, has_layernorm_bias=True, has_linear_bias=False, forward_config=DEFAULT_CONFIG):
    shape = a.shape 
    a = a.view(-1, shape[-1]) # change [*, seq_len, dim] into [M, K]
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    
    # Allocates output, Mean, RSTD
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    Mean = torch.empty((M, ), dtype=a.dtype, device=a.device)
    RSTD = torch.empty((M, ), dtype=a.dtype, device=a.device)

    # First kernel: calc mean and var 
    BLOCK_SIZE, num_warps = calculate_settings(K)
    _first_forward_kernel[(M, )](
        a, a.stride(0), Mean, Mean.stride(0), RSTD, RSTD.stride(0),
        K, 1e-5,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
    )

    # Second kernel: calculate matmul. Normalize each input block loaded  
    # BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, num_stages, num_warps, maxnreg = forward_config # if forward_config is not None else calculate_config_layernorm_linear(M, N, K, 0)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    _second_forward_kernel[grid](
        a, b, linear_bias, c,
        M, N, K, 
        WEIGHT, BIAS,
        Mean, RSTD,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        Mean.stride(0), RSTD.stride(0),
        has_layernorm_bias=has_layernorm_bias, has_linear_bias=has_linear_bias,
        DTYPE=tl.float16 if a.dtype == torch.float16 else tl.bfloat16 if a.dtype == torch.bfloat16 else tl.float32, 
        # BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, GROUP_SIZE_M=GROUP_SIZE_M, num_stages=num_stages, num_warps=num_warps, maxnreg=maxnreg        
    )
    c = c.view(shape[:-1] + (N, ))
    return c, a, b, Mean, RSTD


def layernorm_linear_backward(dOUT, X, B, Mean, RSTD, WEIGHT, BIAS, has_layernorm_bias=True, has_linear_bias=False, backward_config=DEFAULT_CONFIG):
    # forward pass is: X -> X_hat -> Y -> Y @ B = output
    # view trick
    dOUT_shape = dOUT.shape
    dOUT = dOUT.view(-1, dOUT_shape[-1])
    M, K = X.shape
    N = dOUT.shape[-1]
    
    dX = torch.empty((M, K), dtype=X.dtype, device=X.device) 
    dB = torch.zeros((K, N), dtype=torch.float32, device=X.device) # NOTE: change this to be zeros and dtype = fp32 for atomic-adds  
    
    dWEIGHT = torch.zeros((K, ), dtype = torch.float32, device=WEIGHT.device)
    dBIAS = None 
    if has_layernorm_bias:
        dBIAS = torch.zeros((K, ), dtype = torch.float32, device=X.device) 

    c1 = torch.zeros((M, ), dtype=torch.float32, device=X.device)
    c2 = torch.zeros((M, ), dtype=torch.float32, device=X.device)

    # compute dX: fuse matmul with backward-LN: dY = dOUT @ B^T & backward dY->dX
    # Must do two parts like this because c1 and c2 are constants created from the WHOLE ROW and c1 * x_hat is element-wise
    # BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, GROUP_SIZE_M, num_stages, num_warps, maxnreg = backward_config # if backward_config is not None else calculate_config_layernorm_linear(M, N, K, 1)
    _first_backward_kernel[lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(K, META['BLOCK_SIZE_K']), )](
        dOUT, B, dX, dB, X, Mean, RSTD, WEIGHT, BIAS, dWEIGHT, dBIAS, c1, c2, 
        M, K, N,
        dOUT.stride(0), dOUT.stride(1),
        B.stride(0), B.stride(1),
        dX.stride(0), dX.stride(1),
        WEIGHT.stride(0), c1.stride(0), c2.stride(0), Mean.stride(0),
        has_layernorm_bias=has_layernorm_bias,
        # BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_N=BLOCK_SIZE_N, GROUP_SIZE_M=GROUP_SIZE_M, num_stages=num_stages, num_warps=num_warps, maxnreg=maxnreg
    )
    
    # now, c1 and c2 are ready to be computed in dX -> do in row parallel
    BLOCK_SIZE, num_warps = calculate_settings(K)
    _second_backward_kernel[(M, )](
        dX, X, c1, c2, Mean, RSTD,
        M, K, N,
        X.stride(0), Mean.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    
    # cannot fuse in, so best to separate it out 
    dLinearBias = None 
    if has_linear_bias:
        dLinearBias = dOUT.sum(axis=0) # [(N, )]

    dX = dX.view(dOUT_shape[:-1] + (K, ))
    dB, dWEIGHT = dB.to(X.dtype), dWEIGHT.to(X.dtype) 
    if has_layernorm_bias:
        dBIAS = dBIAS.to(X.dtype)

    return dX, dB, dLinearBias, dWEIGHT, dBIAS


class LayernormLinearFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    @torch.amp.custom_fwd(device_type=infer_device(), cast_inputs=torch.bfloat16)
    def forward(ctx, X, linear_weight, linear_bias, WEIGHT, BIAS, has_layernorm_bias=True, has_linear_bias=False, supported_configs=(DEFAULT_CONFIG, DEFAULT_CONFIG)):
        c, X, linear_weight, Mean, RSTD = layernorm_linear_forward(X, linear_weight, linear_bias, WEIGHT, BIAS, has_layernorm_bias, has_linear_bias, supported_configs[0])
        ctx.save_for_backward(X, linear_weight, Mean, RSTD, WEIGHT, BIAS)
        ctx.backward_config = supported_configs[1]
        ctx.has_layernorm_bias = has_layernorm_bias
        ctx.has_linear_bias = has_linear_bias 
        return c

    @staticmethod
    @ensure_contiguous
    @torch.amp.custom_bwd(device_type=infer_device())
    def backward(ctx, dOUT):
        X, linear_weight, Mean, RSTD, WEIGHT, BIAS = ctx.saved_tensors
        dX, dB, dLinearBias, dWEIGHT, dBIAS = layernorm_linear_backward(dOUT, X, linear_weight, Mean, RSTD, WEIGHT, BIAS, ctx.has_layernorm_bias, ctx.has_linear_bias, ctx.backward_config)
        return dX, dB, dLinearBias, dWEIGHT, dBIAS, None, None, None


class LayernormLinear(nn.Module):
    def __init__(self, K, N, has_layernorm_bias=True, has_linear_bias=True, device=None, dtype=None): 
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.K = K 
        self.N = N
        self.has_layernorm_bias = has_layernorm_bias
        self.has_linear_bias = has_linear_bias

        self.WEIGHT = nn.Parameter(torch.empty(K, **factory_kwargs)) # Layernorm weight 
        self.BIAS = None
        if self.has_layernorm_bias: 
            self.BIAS = nn.Parameter(torch.empty(K, **factory_kwargs))  # Layernorm bias 
        self.linear_weight = nn.Parameter(torch.empty((K, N), **factory_kwargs)) 
        self.linear_bias = None
        if self.has_linear_bias:
            self.linear_bias = nn.Parameter(torch.empty((N,), **factory_kwargs))
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        # init Layernorm params 
        torch.nn.init.ones_(self.WEIGHT)
        if self.has_layernorm_bias:
            torch.nn.init.zeros_(self.BIAS)

        # init Linear params
        # torch.nn.init.kaiming_uniform_(self.linear_weight.T, a=math.sqrt(5))
        temp =  torch.empty((self.N, self.K), device=self.linear_weight.device, dtype=self.linear_weight.dtype)
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(temp, a=math.sqrt(5))
            self.linear_weight.data.copy_(temp.T)
        del temp
        
        if self.has_linear_bias:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.linear_weight.T)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.linear_bias, -bound, bound)

    def forward(self, X):
        M = math.prod(X.shape[:-1])
        supported_configs = calculate_config_layernorm_linear(M, self.N, self.K, 2)
        if supported_configs is None: # revert back to standard torch 
            return F.linear(F.layer_norm(X, (self.K, ), self.WEIGHT, self.BIAS), self.linear_weight.T, self.linear_bias)
        return LayernormLinearFunction.apply(X, self.linear_weight, self.linear_bias, self.WEIGHT, self.BIAS, self.has_layernorm_bias, self.has_linear_bias, supported_configs)

    def extra_repr(self):
        return f"K={self.K}, N={self.N}, has_linear_bias={self.has_linear_bias}"


if __name__ == "__main__":
    M, N, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    print("Running: autotune for sizes: (M, N, K): ", M, N, K)
    dtype=torch.bfloat16
    device= 'cuda'
    a = torch.randn((M, K), dtype=dtype, device=device)
    b = torch.randn((K, N), dtype=dtype, device=device)
    WEIGHT = nn.Parameter(torch.ones((K,), device=a.device, dtype=a.dtype))
    BIAS = nn.Parameter(torch.zeros((K,), device=a.device, dtype=a.dtype))
    
    # call fwd 
    fn = lambda: layernorm_linear_forward(a, b, None, WEIGHT, BIAS, True, False)
    fn()
    
    # call bwd
    o = torch.randn((M, N), dtype=dtype, device = device)
    do = torch.randn_like(o)
    Mean = torch.randn((M, ), dtype=dtype, device=device)
    RSTD = torch.randn((M, ), dtype=dtype, device=device)
    fn = lambda: layernorm_linear_backward(do, a, b, Mean, RSTD, WEIGHT, BIAS, True, False)
    fn()