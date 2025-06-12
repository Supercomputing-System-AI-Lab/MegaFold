import torch
import triton
import triton.language as tl
from evoformer import TritonEvoformer
from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def _attention(q, k, v, res_mask, pair_bias):
    softmax_scale = 1 / (q.shape[-1]**0.5)
    ref_P = (torch.matmul(q * softmax_scale, k.transpose(3, 4)) + pair_bias)
    ref_P = ref_P.masked_fill(~res_mask, max_neg_value(ref_P)) 
    ref_P = torch.softmax(ref_P.float(), dim=-1).to(q.dtype)
    ref_O = torch.matmul(ref_P, v)
    return ref_O
    
def full_evoformer_attention(q, k, v, res_mask, pair_bias):
    o = TritonEvoformer(q, k, v, res_mask, pair_bias)
    do = torch.randn_like(o)
    o.backward(do, retain_graph=True)

def full_deepspeed_evoformer_attention(q, k, v, res_mask, pair_bias):
    o = DS4Sci_EvoformerAttention(q, k, v, [res_mask, pair_bias])
    do = torch.randn_like(o)
    o.backward(do, retain_graph=True)

def full_attention(q, k, v, res_mask, pair_bias):
    BATCH, H, N_SEQ, N_CTX, HEAD_DIM = q.shape
    o = _attention(q, k, v, res_mask, pair_bias)
    o= o.reshape((BATCH, N_SEQ, N_CTX, H, HEAD_DIM))
    do = torch.randn_like(o)
    o.backward(do, retain_graph=True)


BATCH, N_HEADS, HEAD_DIM, N_SEQ = 4, 16, 64, 1
configs = []
for mode in ["full"]:
    configs.append(
        triton.testing.Benchmark(
            x_names=["N_CTX"],
            x_vals=[128, 256, 384, 512, 640, 768, 1024],
            line_arg="provider",
            line_vals=["triton"]  + ["torch"],
            line_names=["Triton"] + ["torch"],
            styles=[("red", "-"), ("blue", "-")],
            ylabel="time (seconds)",
            plot_name=f"evo-attention-batch{BATCH}-head{N_HEADS}-dim{HEAD_DIM}-nseq{N_SEQ}-{mode}",
            args={
                "H": N_HEADS,
                "BATCH": BATCH,
                "HEAD_DIM": HEAD_DIM,
                "N_SEQ": N_SEQ,
                "mode": mode,
            },
        ))

@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, N_SEQ, mode, provider, device='cuda'):
    assert mode in ["fwd", "bwd", "full"]
    dtype = torch.bfloat16
    print("benching: ", BATCH, H, N_CTX, HEAD_DIM, N_SEQ, mode, provider)
    rep, warmup = 5000, 200
    if "triton" in provider:
        q = torch.randn((BATCH, N_SEQ, N_CTX, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, N_SEQ, N_CTX, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, N_SEQ, N_CTX, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        res_mask = torch.randint(0, 2, (BATCH, N_SEQ, 1, 1, N_CTX), dtype=torch.bool, device=device) 
        pair_bias = torch.randn((BATCH, 1, H, N_CTX, N_CTX), dtype=torch.float32, device=device, requires_grad=True)
        fn = lambda: TritonEvoformer(q, k, v, res_mask, pair_bias)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        if mode == "full":
            fn = lambda: full_evoformer_attention(q, k, v, res_mask, pair_bias)
        ms = triton.testing.do_bench(fn, rep=rep, warmup=warmup)
        
    if "deepspeed" in provider: 
        q = torch.randn((BATCH, N_SEQ, N_CTX, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, N_SEQ, N_CTX, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, N_SEQ, N_CTX, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        res_mask = torch.randint(0, 2, (BATCH, N_SEQ, 1, 1, N_CTX), dtype=torch.bool, device=device).bfloat16() # deepspeed only works with bfloat16
        pair_bias = torch.randn((BATCH, 1, H, N_CTX, N_CTX), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: DS4Sci_EvoformerAttention(q, k, v, [res_mask, pair_bias])
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        if mode == "full":
            fn = lambda: full_deepspeed_evoformer_attention(q, k, v, res_mask, pair_bias)
        ms = triton.testing.do_bench(fn, rep=rep, warmup=warmup)
        
    if provider == "torch":
        q = torch.randn((BATCH, H, N_SEQ, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_SEQ, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_SEQ, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        res_mask = torch.randint(0, 2, (BATCH, 1, N_SEQ, 1, N_CTX), dtype=torch.bool, device=device)
        pair_bias = torch.randn((BATCH, H, 1, N_CTX, N_CTX), dtype=torch.float32, device=device, requires_grad=True)

        fn = lambda: _attention(q, k, v, res_mask, pair_bias)
        if mode == "bwd":
            o = fn()
            o= o.reshape((BATCH, N_SEQ, N_CTX, H, HEAD_DIM))
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        if mode == "full":
            fn = lambda: full_attention(q, k, v, res_mask, pair_bias)
        ms = triton.testing.do_bench(fn, rep=rep, warmup=warmup)
    # flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    # total_flops = 2 * flops_per_matmul
    # if mode == "bwd":
    #     total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    # if mode == "full":
    #     total_flops *= 3.5 # 1.0 (forward) + 2.5 (backward)
    # return total_flops * 1e-12 / (ms * 1e-3)
    return ms * 1e-3

if __name__ == "__main__":
    bench_flash_attention.run(print_data=True)


# TFLOPS all 3 with every kernel autotuned
# evoformer-attention-batch4-head32-dim64-nseq1-fwd:
#     N_CTX  Triton [FP16]  deepspeed     torch
# 0   128.0      17.084237  12.486875  4.876049
# 1   256.0      34.550276  20.387909  6.122749
# 2   384.0      40.743243  24.168290  6.683937
# 3   512.0      50.743337  30.456071  6.991134
# 4   640.0      58.568942  35.272233  7.088304
# 5   768.0      59.247306  34.472820  7.199189
# 6  1024.0      64.525117  36.512337  7.283911
# 7  2048.0      73.356383  38.834419  6.660764
# evoformer-attention-batch4-head32-dim64-nseq1-bwd:
#     N_CTX  Triton [FP16]  deepspeed      torch
# 0   128.0       4.701080   6.828865   7.048776
# 1   256.0       9.985486   9.903225   8.931376
# 2   384.0      11.158058  10.507144   9.843291
# 3   512.0      11.791622  10.847500  10.452394
# 4   640.0      12.137659  10.908849  10.659470
# 5   768.0      12.254756  11.021939  10.880776
# 6  1024.0      12.598075  11.186054  11.100313
# 7  2048.0      12.943386  11.258994  11.428881
# evoformer-attention-batch4-head32-dim64-nseq1-full:
#     N_CTX  Triton [FP16]  deepspeed     torch
# 0   128.0       3.830861   9.151733  4.296454
# 1   256.0      12.330365  11.428805  7.935360
# 2   384.0      14.036798  12.701977  8.662607
# 3   512.0      14.986963  13.200310  9.132943
# 4   640.0      15.584658  13.494470  9.290238
# 5   768.0      15.759853  13.607241  9.460576
# 6  1024.0      16.284042  13.889170  9.634959
# 7  2048.0      16.887387  14.098500  9.477697