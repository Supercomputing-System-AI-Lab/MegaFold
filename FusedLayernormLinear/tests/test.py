import torch 
import torch.nn as nn 
from autotuned import LayernormLinear

torch.manual_seed(42)

def full_test(M, N, K):
    dtype = torch.float32
    device = "cuda"
    forward_atol, forward_rtol = 1e-2, 1e-2
    backward_atol, backward_rtol = 1e-1, 1e-1
    
    input = torch.randn((M, K), dtype=dtype, device=device, requires_grad=True)
    input2 = input.clone().detach().requires_grad_(True)

    torch.manual_seed(42)
    ln = nn.LayerNorm(K, dtype=dtype, device=device)
    linear = nn.Linear(K, N, bias=False, dtype=dtype, device=device)
    torch.manual_seed(42) # reset seed for random init
    fused_kernel = LayernormLinear(K, N, has_linear_bias=False, dtype=dtype, device=device)
    
    ref_out = linear(ln(input))
    triton_out = fused_kernel(input2)
    
    dOUT = torch.randn((M, N), dtype=dtype, device=device)
    
    ref_out.backward(dOUT)
    triton_out.backward(dOUT)
    
    try:
        # check fwd
        assert torch.allclose(ref_out, triton_out, atol=forward_atol, rtol=forward_rtol), f"ref_out: {ref_out}, triton_out: {triton_out}, max err: {torch.max(torch.abs(ref_out - triton_out))}"    
        # check bwd
        assert torch.allclose(input.grad, input2.grad, atol=backward_atol, rtol=backward_rtol), f"input.grad: {input.grad}, input2.grad: {input2.grad}, max err: {torch.max(torch.abs(input.grad - input2.grad))}"
        assert torch.allclose(linear.weight.grad.T, fused_kernel.linear_weight.grad, atol=backward_atol, rtol=backward_rtol), f"linear.weight.grad.T: {linear.weight.grad.T}, fused_kernel.linear_weight.grad: {fused_kernel.linear_weight.grad}, max err: {torch.max(torch.abs(linear.weight.grad.T - fused_kernel.linear_weight.grad))}"
        assert torch.allclose(ln.weight.grad, fused_kernel.WEIGHT.grad, atol=backward_atol, rtol=backward_rtol), f"Layernorm weight: {ln.weight.grad}, fused_kernel.WEIGHT.grad: {fused_kernel.WEIGHT.grad}, max err: {torch.max(torch.abs(ln.weight.grad - fused_kernel.WEIGHT.grad))}"
        assert torch.allclose(ln.bias.grad, fused_kernel.BIAS.grad, atol=backward_atol, rtol=backward_rtol), f"Layernorm bias: {ln.bias.grad}, fused_kernel.BIAS.grad: {fused_kernel.BIAS.grad}, max err: {torch.max(torch.abs(ln.bias.grad - fused_kernel.BIAS.grad))}"
    except AssertionError as e:
        print(e)
    
    print("Passed full test for (M, N, K) = ", (M, N, K))


if __name__ == "__main__":
    full_test(16, 16, 16)
    full_test(64, 64, 128)
    full_test(16, 256, 64)
    full_test(256, 256, 512)
    full_test(512, 16, 512)
    full_test(18, 31, 99)
    full_test(27, 59, 64)
    full_test(20, 50, 60)
    full_test(200, 384, 400)
    full_test(100, 768, 400)
    full_test(1879, 18, 666)

    for M in [250, 500, 1000, 2000, 3000,4000, 6000, 8000]:
        for N in [16, 32, 64, 128, 256, 384, 512, 768]:
            for K in [16, 32, 64, 128, 256, 384, 512, 768]:
                full_test(M, N, K) 
