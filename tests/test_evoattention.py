import torch 
from FusedEvoAttention.autotuned import TritonEvoformer

def max_neg_value(t):
    """Get the maximum negative value of Tensor based on its `dtype`.

    :param t: The Tensor.
    :return: The maximum negative value of its `dtype`.
    """
    return -torch.finfo(t.dtype).max


def test_full_step(BATCH_SIZE, HEAD, N_SEQ, SEQ_LEN, DIM, dtype=torch.float16):
    Q = (
        torch.empty(
            (BATCH_SIZE, N_SEQ, SEQ_LEN, HEAD, DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    K = (
        torch.empty(
            (BATCH_SIZE, N_SEQ, SEQ_LEN, HEAD, DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    V = (
        torch.empty(
            (BATCH_SIZE, N_SEQ, SEQ_LEN, HEAD, DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    mask = torch.randint(0, 2, (BATCH_SIZE, N_SEQ, 1, 1, SEQ_LEN), device="cuda")
    res_mask = 1e9 * (mask -1)
    pair_bias = (
        torch.empty(
            (BATCH_SIZE, 1, HEAD, SEQ_LEN, SEQ_LEN), dtype=torch.float32, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    softmax_scale = 1 / (DIM**0.5)
    dO = torch.randn_like(Q)

    # reference implementation
    ref_Q = Q.transpose(-2, -3)
    ref_K = K.transpose(-2, -3)
    ref_V = V.transpose(-2, -3)
    
    ref_P = torch.matmul(ref_Q * softmax_scale, ref_K.transpose(-1, -2)) + pair_bias + res_mask
    ref_P = torch.softmax(ref_P.float(), dim=-1).to(dtype)
    ref_O = torch.matmul(ref_P, ref_V)
    ref_out = ref_O.transpose(-2, -3)
    
    ref_out.backward(dO)
    ref_dV, V.grad = V.grad.clone(), None
    ref_dK, K.grad = K.grad.clone(), None
    ref_dQ, Q.grad = Q.grad.clone(), None
    ref_d_pair_bias, pair_bias.grad = pair_bias.grad.clone(), None

    # triton implementation
    tri_out = TritonEvoformer(Q, K, V, res_mask, pair_bias).to(dtype)
    tri_out.backward(dO)
    tri_dV, V.grad = V.grad.clone(), None
    tri_dK, K.grad = K.grad.clone(), None
    tri_dQ, Q.grad = Q.grad.clone(), None
    tri_d_pair_bias, pair_bias.grad = pair_bias.grad.clone(), None

    # compare
    rtol = 0.0 if dtype == torch.float16 else 2e-2 # allow error for bfloat16
    atol = 1e-2
    
    assert torch.allclose(ref_out, tri_out, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dQ, tri_dQ, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dK, tri_dK, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dV, tri_dV, atol=atol, rtol=rtol)
    assert torch.allclose(ref_d_pair_bias, tri_d_pair_bias, atol=atol, rtol=rtol), f'ref_d_pair_bias: {ref_d_pair_bias}, tri_d_pair_bias: {tri_d_pair_bias}'

 
def tests_full_step(dtype):
    # arbitrary seq-len & dim
    test_full_step(BATCH_SIZE=1, HEAD=16, N_SEQ=1, SEQ_LEN=23, DIM=8, dtype=dtype)    
    test_full_step(BATCH_SIZE=1, HEAD=16, N_SEQ=1, SEQ_LEN=99, DIM=8, dtype=dtype)
    test_full_step(BATCH_SIZE=1, HEAD=16, N_SEQ=1, SEQ_LEN=136, DIM=8, dtype=dtype)  
    test_full_step(BATCH_SIZE=1, HEAD=16, N_SEQ=1, SEQ_LEN=190, DIM=8, dtype=dtype)  
    test_full_step(BATCH_SIZE=1, HEAD=16, N_SEQ=1, SEQ_LEN=237, DIM=8, dtype=dtype)  
    test_full_step(BATCH_SIZE=1, HEAD=16, N_SEQ=1, SEQ_LEN=278, DIM=8, dtype=dtype)  
    test_full_step(BATCH_SIZE=1, HEAD=16, N_SEQ=1, SEQ_LEN=341, DIM=8, dtype=dtype)  
    test_full_step(BATCH_SIZE=1, HEAD=16, N_SEQ=1, SEQ_LEN=389, DIM=8, dtype=dtype)  
    test_full_step(BATCH_SIZE=1, HEAD=16, N_SEQ=1, SEQ_LEN=422, DIM=8, dtype=dtype)  
    test_full_step(BATCH_SIZE=1, HEAD=16, N_SEQ=1, SEQ_LEN=499, DIM=8, dtype=dtype)  
    
    test_full_step(BATCH_SIZE=1, HEAD=4, N_SEQ=128, SEQ_LEN=23, DIM=16, dtype=dtype)
    test_full_step(BATCH_SIZE=1, HEAD=4, N_SEQ=128, SEQ_LEN=97, DIM=16, dtype=dtype)
    test_full_step(BATCH_SIZE=1, HEAD=4, N_SEQ=128, SEQ_LEN=131, DIM=16, dtype=dtype)
    test_full_step(BATCH_SIZE=1, HEAD=4, N_SEQ=128, SEQ_LEN=189, DIM=16, dtype=dtype)
    test_full_step(BATCH_SIZE=1, HEAD=4, N_SEQ=128, SEQ_LEN=241, DIM=32, dtype=dtype)
    test_full_step(BATCH_SIZE=1, HEAD=4, N_SEQ=128, SEQ_LEN=298, DIM=32, dtype=dtype)
    test_full_step(BATCH_SIZE=1, HEAD=4, N_SEQ=128, SEQ_LEN=304, DIM=32, dtype=dtype)
    test_full_step(BATCH_SIZE=1, HEAD=4, N_SEQ=128, SEQ_LEN=383, DIM=32, dtype=dtype)
    test_full_step(BATCH_SIZE=1, HEAD=4, N_SEQ=128, SEQ_LEN=409, DIM=32, dtype=dtype)
    test_full_step(BATCH_SIZE=1, HEAD=4, N_SEQ=128, SEQ_LEN=476, DIM=32, dtype=dtype)
    
    test_full_step(BATCH_SIZE=1, HEAD=16, N_SEQ=1, SEQ_LEN=23, DIM=64, dtype=dtype)    
    test_full_step(BATCH_SIZE=1, HEAD=16, N_SEQ=1, SEQ_LEN=99, DIM=64, dtype=dtype)
    test_full_step(BATCH_SIZE=1, HEAD=16, N_SEQ=1, SEQ_LEN=136, DIM=64, dtype=dtype)  
    test_full_step(BATCH_SIZE=1, HEAD=16, N_SEQ=1, SEQ_LEN=190, DIM=64, dtype=dtype)  
    test_full_step(BATCH_SIZE=1, HEAD=16, N_SEQ=1, SEQ_LEN=237, DIM=64, dtype=dtype)  
    test_full_step(BATCH_SIZE=1, HEAD=16, N_SEQ=1, SEQ_LEN=278, DIM=64, dtype=dtype)  
    test_full_step(BATCH_SIZE=1, HEAD=16, N_SEQ=1, SEQ_LEN=341, DIM=64, dtype=dtype)  
    test_full_step(BATCH_SIZE=1, HEAD=16, N_SEQ=1, SEQ_LEN=389, DIM=64, dtype=dtype)  
    test_full_step(BATCH_SIZE=1, HEAD=16, N_SEQ=1, SEQ_LEN=422, DIM=64, dtype=dtype)  
    test_full_step(BATCH_SIZE=1, HEAD=16, N_SEQ=1, SEQ_LEN=499, DIM=64, dtype=dtype)  
    
    test_full_step(BATCH_SIZE=1, HEAD=16, N_SEQ=1, SEQ_LEN=128, DIM=64, dtype=dtype)    
    test_full_step(BATCH_SIZE=1, HEAD=4, N_SEQ=128, SEQ_LEN=128, DIM=32, dtype=dtype)    
    
    test_full_step(BATCH_SIZE=5, HEAD=10, N_SEQ=200, SEQ_LEN=384, DIM=64, dtype=dtype)
    test_full_step(BATCH_SIZE=5, HEAD=10, N_SEQ=200, SEQ_LEN=384, DIM=64, dtype=dtype)
    test_full_step(BATCH_SIZE=10, HEAD=5, N_SEQ=200, SEQ_LEN=384, DIM=64, dtype=dtype)
    
    test_full_step(BATCH_SIZE=1, HEAD=10, N_SEQ=200, SEQ_LEN=640, DIM=64, dtype=dtype)
    test_full_step(BATCH_SIZE=1, HEAD=10, N_SEQ=200, SEQ_LEN=640, DIM=64, dtype=dtype)
    test_full_step(BATCH_SIZE=1, HEAD=15, N_SEQ=200, SEQ_LEN=640, DIM=64, dtype=dtype)
    
    test_full_step(BATCH_SIZE=1, HEAD=5, N_SEQ=100, SEQ_LEN=768, DIM=64, dtype=dtype)
    test_full_step(BATCH_SIZE=1, HEAD=5, N_SEQ=100, SEQ_LEN=768, DIM=64, dtype=dtype)
    test_full_step(BATCH_SIZE=1, HEAD=10, N_SEQ=100, SEQ_LEN=768, DIM=64, dtype=dtype)
    
    test_full_step(BATCH_SIZE=1, HEAD=1, N_SEQ=100, SEQ_LEN=1280, DIM=64, dtype=dtype)
    test_full_step(BATCH_SIZE=1, HEAD=5, N_SEQ=300, SEQ_LEN=512, DIM=64, dtype=dtype)
    test_full_step(BATCH_SIZE=1, HEAD=5, N_SEQ=300, SEQ_LEN=384, DIM=64, dtype=dtype)
    print("PASSED")

if __name__ == "__main__":
    tests_full_step(dtype=torch.float16)
    print("DONE")