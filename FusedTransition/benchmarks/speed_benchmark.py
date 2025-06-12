import torch
import triton
from fused_transition import FusedTransition
from torch_transition import Transition

torch.manual_seed(42)

def full_transition(do, x, layer):
    o = layer(x)
    o.backward(do, retain_graph=True)

# (M, d, expansion_factor)
MDE = [(60000, 128, 4), (80000, 128, 4), (100000, 128, 4), (120000, 128, 4), (147456, 128, 4), (160000, 128, 4), (180000, 128, 4), (200000, 128, 4)]

configs = []
for mode in ["full"]: # , "bwd", "full"
    configs.append(
        triton.testing.Benchmark(
            x_names=["M", "d", "ef"],
            x_vals=MDE, 
            line_arg="provider",
            line_vals=["triton"] + ["torch"],
            line_names=["triton"]+ ["torch"],
            styles=[("red", "-"), ("blue", "-")],
            ylabel="seconds",
            plot_name=f"Transition Time - {mode} pass",
            args={"mode": mode},
        )
    )
@triton.testing.perf_report(configs)
def bench_transition(M, d, ef, mode, provider, device='cuda'):
    assert mode in ["fwd", "bwd", "full"]
    dtype = torch.bfloat16 
    device = "cuda"
    rep, warmup = 5000, 200
    
    if provider == "triton":
        x = torch.randn((M, d), device=device, dtype=dtype, requires_grad=True)
        do = torch.randn((M, d),  device=device, dtype=dtype)
        triton_transition = FusedTransition(dim=d, expansion_factor=ef, device=device, dtype=dtype)
        fn = lambda: triton_transition(x)
        if mode == "bwd":
            o = fn()
            fn = lambda: o.backward(do, retain_graph=True)
        if mode == "full":
            fn = lambda: full_transition(do, x, triton_transition)          
        ms = triton.testing.do_bench(fn, rep=rep, warmup=warmup, grad_to_none=[x])
  
    if provider == "torch":
        x = torch.randn((M, d), device=device, dtype=dtype, requires_grad=True)
        do = torch.randn((M, d),  device=device, dtype=dtype)
        torch_transition = Transition(dim=d, expansion_factor=ef, device=device, dtype=dtype)
        fn = lambda: torch_transition(x)
        if mode == "bwd":
            o = fn()
            fn = lambda: o.backward(do, retain_graph=True)
        if mode == "full":
            fn = lambda: full_transition(do, x, torch_transition)         
        ms = triton.testing.do_bench(fn, rep=rep, warmup=warmup, grad_to_none=[x])
        
    return ms * 1e-3    # seconds

if __name__ == "__main__":
    bench_transition.run(print_data=True)
