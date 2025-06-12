# Reference implementation of Transition 

import torch 
import torch.nn as nn 

class Transition(nn.Module):
    """A Transition module."""

    def __init__(self, *, dim, expansion_factor=4, device=None, dtype=None):
        super().__init__()
        dim_inner = int(dim * expansion_factor)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim, device=device, dtype=dtype), ## in code: this can be pre_ln 
            nn.Linear(dim, dim_inner * 2, bias=False, device=device, dtype=dtype),
            SwiGLU(),
            nn.Linear(dim_inner, dim, bias=False, device=device, dtype=dtype),
        )

    def forward(self, x):
        """Perform the forward pass.
        :param x: The input tensor.
        :return: The output tensor.
        """
        return self.ff(x)
    
    
class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit."""

    def forward(self, x):
        """Perform the forward pass.

        :param x: The input tensor.
        :return: The output tensor.
        """
        x, gates = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gates) * x
