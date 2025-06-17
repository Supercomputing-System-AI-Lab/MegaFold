import torch
import torch.nn as nn 
import triton
from FusedLayernormLinear.fused_layernorm_linear import LayernormLinear, layernorm_linear_forward, layernorm_linear_backward

def full_triton_layernorm_linear(do, a, fused_kernel):
    o = fused_kernel(a)
    o.backward(do, retain_graph=True)

def full_torch_layernorm_linear(do, a, ln, linear):
    o = linear(ln(a))
    o.backward(do, retain_graph=True)


MNK = [(60000, 16, 128), (80000, 16, 128), (100000, 16, 128), (120000, 16, 128), (147456, 16, 128), (160000, 16, 128), (180000, 16, 128), (200000, 16, 128)]
configs = []
for mode in ["full"]:
    configs.append(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],
            x_vals=MNK, 
            line_arg="provider",
            line_vals=["triton"] + ["torch"],
            line_names=["triton"]+ ["torch"],
            styles=[("red", "-"), ("blue", "-")],
            ylabel="seconds",
            plot_name=f"LayernormLinear Time - {mode} pass",
            args={"mode": mode},
        )
    )
@triton.testing.perf_report(configs)
def bench_layernorm_linear(M, N, K, mode, provider, device='cuda'):
    assert mode in ["fwd", "bwd", "full"]
    dtype = torch.float16 
    device = "cuda"
    rep, warmup = 5000, 200
    if provider == "triton":
        a = torch.randn((M, K), dtype=dtype, device=device)
        fused_kernel = LayernormLinear(K, N, has_layernorm_bias=True, has_linear_bias=False, dtype=dtype, device=device)
        fn = lambda: fused_kernel(a) 
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        if mode == "full":
            o = fn() 
            do = torch.randn((M, N), dtype=dtype, device=device)
            fn = lambda: full_triton_layernorm_linear(do, a, fused_kernel)
        ms = triton.testing.do_bench(fn, rep=rep, warmup = warmup)
  
    if provider == "torch":
        a = torch.randn((M, K), dtype=dtype, device=device)
        ln = nn.LayerNorm(K, dtype=dtype, device=device)
        linear = nn.Linear(K, N, bias=False, dtype =dtype, device=device)
        fn = lambda: linear(ln(a))
        if mode == "bwd":
            o = fn() 
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        if mode == "full":
            do = torch.randn((M, N), dtype=dtype, device=device)
            fn = lambda: full_torch_layernorm_linear(do, a, ln, linear)         
        ms = triton.testing.do_bench(fn, rep=rep, warmup=warmup)
    return ms * 1e-3


if __name__ == "__main__":
    bench_layernorm_linear.run(print_data=True)
