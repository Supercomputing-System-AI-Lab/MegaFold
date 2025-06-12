import torch 
from deepspeed.utils.timer import SynchronizedWallClockTimer 
from fused_transition import FusedSwiGLU
from torch_transition import SwiGLU

device = 'cuda'
dtype = torch.bfloat16
M, d = 147456, 512
provider = "triton"


if provider == "triton":
    x = torch.randn((M, d), device=device, dtype=dtype, requires_grad=True)
    triton_swiglu = FusedSwiGLU()
    o = triton_swiglu(x)
    do = torch.randn((M, d//2),  device=device, dtype=dtype)
    o.backward(do, retain_graph=True)

elif provider == "torch":
    x = torch.randn((M, d), device=device, dtype=dtype, requires_grad=True)
    torch_swiglu = SwiGLU()
    o = torch_swiglu(x)
    do = torch.randn(o.shape, device=device, dtype=dtype)
    o.backward(do, retain_graph=True)


mem_usage = SynchronizedWallClockTimer.memory_usage()
print("Memory usage after pass: ", mem_usage)
