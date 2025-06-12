import torch
import triton
from fused_transition import FusedSwiGLU
from torch_transition import SwiGLU


torch.manual_seed(42)

def full_transition(do, x, layer):
    o = layer(x)
    o.backward(do, retain_graph=True)


Md = [(147456, 512), (384, 512), (1536, 3204), (384, 3204), (147456, 512), (384, 3072), (147456, 1024), (11772, 512), (2943, 512), (1536, 3072), (384, 3072)]
configs = []
for mode in ["full"]: # , "bwd", "full"
    configs.append(
        triton.testing.Benchmark(
            x_names=["M", "d"],
            x_vals=Md, 
            line_arg="provider",
            line_vals=["triton"] + ["torch"],
            line_names=["triton"]+ ["torch"],
            styles=[("red", "-"), ("blue", "-")],
            ylabel="seconds",
            plot_name=f"SwiGLU Time - {mode} pass",
            args={"mode": mode},
        )
    )
@triton.testing.perf_report(configs)
def bench_transition(M, d, mode, provider, device='cuda'):
    assert mode in ["fwd", "bwd", "full"]
    dtype = torch.bfloat16 
    device = "cuda"
    rep, warmup = 5000, 200
    
    if provider == "triton":
        x = torch.randn((M, d), device=device, dtype=dtype, requires_grad=True)
        do = torch.randn((M, d//2),  device=device, dtype=dtype)
        triton_swiglu = FusedSwiGLU()
        fn = lambda: triton_swiglu(x)
        if mode == "bwd":
            o = fn()
            fn = lambda: o.backward(do, retain_graph=True)
        if mode == "full":
            fn = lambda: full_transition(do, x, triton_swiglu)
        ms = triton.testing.do_bench(fn, rep=rep, warmup=warmup, grad_to_none=[x])
  
    if provider == "torch":
        x = torch.randn((M, d), device=device, dtype=dtype, requires_grad=True)
        do = torch.randn((M, d//2),  device=device, dtype=dtype)
        torch_swiglu = SwiGLU()
        fn = lambda: torch_swiglu(x)
        if mode == "bwd":
            o = fn()
            fn = lambda: o.backward(do, retain_graph=True)
        if mode == "full":
            fn = lambda: full_transition(do, x, torch_swiglu)         
        ms = triton.testing.do_bench(fn, rep=rep, warmup=warmup, grad_to_none=[x])
        
    return ms * 1e-3    # seconds

if __name__ == "__main__":
    bench_transition.run(print_data=True)
