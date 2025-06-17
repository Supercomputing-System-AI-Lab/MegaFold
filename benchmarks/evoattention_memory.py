import torch 
from deepspeed.utils.timer import SynchronizedWallClockTimer 
from benchmarks.evoattention_speed import full_attention, full_deepspeed_evoformer_attention, full_evoformer_attention


device = 'cuda'
dtype = torch.bfloat16
BATCH, H, HEAD_DIM, N_SEQ = 4, 16, 64, 1
N_CTX = 128
provider = "triton"

if provider == "triton":
    q = torch.randn((BATCH, N_SEQ, N_CTX, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True) # 4 * 1 * 384 * 32 * 64 * 2(bfloat16)
    k = torch.randn((BATCH, N_SEQ, N_CTX, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, N_SEQ, N_CTX, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    res_mask = torch.randint(0, 2, (BATCH, N_SEQ, 1, 1, N_CTX), dtype=torch.bool, device=device) 
    pair_bias = torch.randn((BATCH, 1, H, N_CTX, N_CTX), dtype=dtype, device=device, requires_grad=True)
    full_evoformer_attention(q, k, v, res_mask, pair_bias)
elif provider == "deepspeed":
    q = torch.randn((BATCH, N_SEQ, N_CTX, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, N_SEQ, N_CTX, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, N_SEQ, N_CTX, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    res_mask = torch.randint(0, 2, (BATCH, N_SEQ, 1, 1, N_CTX), dtype=torch.bool, device=device).bfloat16() # deepspeed only works with bfloat16
    pair_bias = torch.randn((BATCH, 1, H, N_CTX, N_CTX), dtype=dtype, device=device, requires_grad=True)
    full_deepspeed_evoformer_attention(q, k, v, res_mask, pair_bias)
elif provider == "torch":   
    q = torch.randn((BATCH, H, N_SEQ, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, H, N_SEQ, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, H, N_SEQ, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    res_mask = torch.randint(0, 2, (BATCH, 1, N_SEQ, 1, N_CTX), dtype=torch.bool, device=device)
    pair_bias = torch.randn((BATCH, H, 1, N_CTX, N_CTX), dtype=dtype, device=device, requires_grad=True)
    full_attention(q, k, v, res_mask, pair_bias)

mem_usage = SynchronizedWallClockTimer.memory_usage()
print("Memory usage after all: ", mem_usage)

# Memory benchmark full (msa row wise) -- max mem allocated
# batch4-head32-dim64-nseq1
#     N_CTX  Triton [FP16]  deepspeed  torch 
# 0   128.0       0.0314GB  0.0314GB   0.0686
# 1   256.0       0.0940GB  0.0940GB   0.1995
# 2   384.0       0.1879GB  0.1879GB   0.4085
# 3   512.0       0.3130GB  0.3130GB   0.6956
# 4   640.0       0.4694GB  0.4694GB   1.0608
# 5   768.0       0.6570GB  0.6570GB   1.5042
# 6  1024.0       1.1260GB  1.1260GB   2.6253
# 7  2048.0       4.2520GB  4.2520GB   10.2346

