import torch 
import torch.nn as nn 
from torch_transition import Transition, SwiGLU
from fused_transition import FusedTransition

torch.manual_seed(42)

def test_module(M, d, expansion_factor=4):
    dtype = torch.float16 
    device = "cuda"
    atol = 1e-1
    rtol = 1e-1
    
    x = torch.randn((M, d), device=device, dtype=dtype, requires_grad=True)
    x2 = x.clone().detach().requires_grad_(True)

    torch.manual_seed(42)
    torch_transition = Transition(dim=d, expansion_factor=expansion_factor, device=device, dtype=dtype)
    torch.manual_seed(42)
    fused_transition = FusedTransition(dim=d, expansion_factor=expansion_factor, device=device, dtype=dtype)
    
    with torch.no_grad():
        fused_transition.ff[0].WEIGHT.copy_(torch_transition.ff[0].weight)
        fused_transition.ff[0].BIAS.copy_(torch_transition.ff[0].bias)
        fused_transition.ff[0].linear_weight.copy_(torch_transition.ff[1].weight.T)
        fused_transition.ff[2].weight.copy_(torch_transition.ff[3].weight)
    
    torch_out = torch_transition(x)
    triton_out = fused_transition(x2)

    dY = torch.randn((M, d), device=device, dtype=dtype)    
    
    torch_out.backward(dY)
    triton_out.backward(dY)

    try:
        # fwd 
        assert torch.allclose(torch_out, triton_out, atol, rtol), f"torch_out: {torch_out}, triton_out: {triton_out}, max err: {torch.max(torch.abs(torch_out - triton_out))}"
        
        # bwd
        assert torch.allclose(x.grad, x2.grad, atol, rtol), f"x.grad: {x.grad}, dX: {x2.grad}, max err: {torch.max(torch.abs(x.grad - x2.grad))}"
        assert torch.allclose(torch_transition.ff[0].weight.grad, fused_transition.layernorm_weight.grad, atol, rtol), f"torch_transition.ff[0].weight.grad: {torch_transition.ff[0].weight.grad}, fused_transition.layernorm_weight.grad: {fused_transition.layernorm_weight.grad}, max err: {torch.max(torch.abs(torch_transition.ff[0].weight.grad - fused_transition.layernorm_weight.grad))}"
        assert torch.allclose(torch_transition.ff[0].bias.grad, fused_transition.layernorm_bias.grad, atol, rtol), f"torch_transition.ff[0].bias.grad: {torch_transition.ff[0].bias.grad}, fused_transition.layernorm_bias.grad: {fused_transition.layernorm_bias.grad}, max err: {torch.max(torch.abs(torch_transition.ff[0].bias.grad - fused_transition.layernorm_bias.grad))}"
        assert torch.allclose(torch_transition.ff[1].weight.grad, fused_transition.first_linear_weight.grad.T, atol, rtol), f"torch_transition.ff[1].weight.grad: {torch_transition.ff[1].weight.grad}, fused_transition.first_linear_weight.grad.T: {fused_transition.first_linear_weight.grad.T}, max err: {torch.max(torch.abs(torch_transition.ff[1].weight.grad - fused_transition.first_linear_weight.grad.T))}"
        assert torch.allclose(torch_transition.ff[3].weight.grad, fused_transition.second_linear_weight.grad.T, atol, rtol), f"torch_transition.ff[3].weight.grad: {torch_transition.ff[3].weight.grad}, fused_transition.second_linear_weight.grad.T: {fused_transition.second_linear_weight.grad.T}, max err: {torch.max(torch.abs(torch_transition.ff[3].weight.grad - fused_transition.second_linear_weight.grad.T))}"
    
    except AssertionError as e:
        print(e)


if __name__ == "__main__":
    # covered by autotune 
    test_module(147456, 128, 0.5)
    test_module(147456, 64, 0.5)
    test_module(120000, 128, 0.25)
    test_module(100000, 128, 0.25)
    test_module(80000, 128, 0.25)

    # not covered by autotune 
    test_module(3002, 128)
    test_module(9003, 128)
    test_module(3005, 128)
    test_module(384, 768)
    test_module(1536, 768)
    test_module(3000, 768)

    test_module(64, 64)
    test_module(128, 128)
    test_module(384, 384)

    test_module(5000, 384)
    test_module(5000, 128)
    test_module(3849, 384)
    test_module(6655, 64)
    test_module(38402, 384)
    test_module(3844, 768)
    test_module(8424, 128)
    test_module(9000, 64)
    test_module(4999, 384)
    
    test_module(20, 50)
    test_module(4999, 56)
    