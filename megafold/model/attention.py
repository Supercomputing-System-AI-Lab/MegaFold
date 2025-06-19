from __future__ import annotations

from functools import partial

# import einx
import torch
import torch.nn.functional as F
from beartype.typing import Literal

try:
    from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention
except Exception:
    DS4Sci_EvoformerAttention = None
import einops
from einops import einsum, rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn import Module

try:
    from megafold.model.FusedEvoAttention.evoattention import TritonEvoformer
except Exception:
    TritonEvoformer = None
from megafold.tensor_typing import Bool, Float, typecheck
from megafold.utils.model_utils import (
    concat_previous_window,
    max_neg_value,
    pad_at_dim,
    softclamp,
)
from megafold.utils.utils import default, exists, not_exists

# alias

LinearNoBias = partial(nn.Linear, bias=False)

# for changing full attention bias matrix to a local windowed one for atom attention


@typecheck
def full_pairwise_repr_to_windowed(
    pairwise_repr: Shaped["... m m dp"], window_size: int  # type: ignore
) -> Shaped["... n w (w*2) dp"]:  # type: ignore
    """Convert a full pairwise representation matrix to a local windowed one.

    :param pairwise_repr: The full pairwise representation matrix.
    :param window_size: The window size.
    :return: The local windowed pairwise representation matrix.
    """
    seq_len, device = pairwise_repr.shape[-2], pairwise_repr.device

    padding_needed = (window_size - (seq_len % window_size)) % window_size
    pairwise_repr = F.pad(pairwise_repr, (0, 0, 0, padding_needed, 0, padding_needed), value=0.0)
    pairwise_repr = rearrange(
        pairwise_repr, "... (i w1) (j w2) d -> ... i j w1 w2 d", w1=window_size, w2=window_size
    )
    pairwise_repr = concat_previous_window(pairwise_repr, dim_seq=-4, dim_window=-2)

    # get the diagonal

    n = torch.arange(pairwise_repr.shape[-4], device=device)

    # pairwise_repr = einx.get_at('... [i j] w1 w2 d, n, n -> ... n w1 w2 d', pairwise_repr, n, n)

    pairwise_repr = pairwise_repr[..., n, n, :, :, :]

    return pairwise_repr


@typecheck
def full_attn_bias_to_windowed(
    attn_bias: Shaped["... m m"], window_size: int  # type: ignore
) -> Shaped["... n w (w*2)"]:  # type: ignore
    """Convert a full attention bias matrix to a local windowed one.

    :param attn_bias: The full attention bias matrix.
    :param window_size: The window size.
    :return: The local windowed attention bias matrix.
    """
    attn_bias = rearrange(attn_bias, "... -> ... 1")
    attn_bias = full_pairwise_repr_to_windowed(attn_bias, window_size=window_size)
    return rearrange(attn_bias, "... 1 -> ...")


# multi-head attention


class Attention(Module):
    """Attention model."""

    @typecheck
    def __init__(
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
        dropout=0.0,
        gate_output=True,
        query_bias=True,
        window_size=None,
        num_memory_kv: int = 0,
        enable_attn_softclamp=False,
        attn_softclamp_value=50.0,
        softmax_full_precision=False,
        init_gate_bias=-2.0,
    ):
        super().__init__()
        """
        ein notation:

        b - batch
        h - heads
        n - sequence
        d - dimension
        e - dimension (pairwise rep)
        i - source sequence
        j - context sequence
        m - memory key / value seq
        """

        dim_inner = dim_head * heads

        self.attend = Attend(
            dropout=dropout,
            window_size=window_size,
            enable_attn_softclamp=enable_attn_softclamp,
            attn_softclamp_value=attn_softclamp_value,
            softmax_full_precision=softmax_full_precision,
        )

        self.split_heads = Rearrange("b n (h d) -> b h n d", h=heads)
        self.merge_heads = Rearrange("b h n d -> b n (h d)")

        self.to_q = nn.Linear(dim, dim_inner, bias=query_bias)
        self.to_kv = LinearNoBias(dim, dim_inner * 2)
        self.to_out = LinearNoBias(dim_inner, dim)

        self.memory_kv = None

        if num_memory_kv > 0:
            self.memory_kv = nn.Parameter(torch.zeros(2, heads, num_memory_kv, dim_head))
            nn.init.normal_(self.memory_kv, std=0.02)

        # gating of value
        # allows attention to attend to nothing

        self.to_gates = None

        if gate_output:
            gate_linear = nn.Linear(dim, dim_inner)
            nn.init.zeros_(gate_linear.weight)
            nn.init.constant_(gate_linear.bias, init_gate_bias)

            self.to_gates = gate_linear

    @typecheck
    def forward(
        self,
        seq: Float["b i d"],  # type: ignore
        mask: Bool["b n"] | None = None,  # type: ignore
        context: Float["b j d"] | None = None,  # type: ignore
        windowed_mask: Bool["b nw w (w*2)"] | None = None,  # type: ignore
        attn_bias: Float["... i j"] | Float["... nw w (w*2)"] | None = None,  # type: ignore
        **kwargs,
    ) -> Float["b i d"]:  # type: ignore
        """Run multi-head attention on a sequence.

        :param seq: The input sequence.
        :param mask: The mask to apply to the sequence.
        :param context: The context sequence to reference.
        :param attn_bias: The attention bias to apply.
        :param kwargs: Additional keyword arguments for the attention computation.
        :return: The output sequence.
        """

        q = self.to_q(seq)

        context_seq = default(context, seq)
        k, v = self.to_kv(context_seq).chunk(2, dim=-1)

        q, k, v = tuple(self.split_heads(t) for t in (q, k, v))

        # attention
        out = self.attend(
            q,
            k,
            v,
            attn_bias=attn_bias,
            mask=mask,
            windowed_mask=windowed_mask,
            memory_kv=self.memory_kv,
            **kwargs,
        )                        


        # merge heads

        out = self.merge_heads(out)

        # gate output

        if exists(self.to_gates):
            gates = self.to_gates(seq)
            #print('hi', out.is_contiguous(), gates.is_contiguous())
            out = out * gates.sigmoid()

        # combine heads

        return self.to_out(out)


# the main attention function


class Attend(Module):
    """Attention module."""

    def __init__(
        self,
        dropout=0.0,
        window_size=None,
        scale: float | None = None,
        enable_attn_softclamp=False,
        attn_softclamp_value=50.0,
        softmax_full_precision=False,
    ):
        super().__init__()
        """
        ein notation:

        b - batch
        h - heads
        n - sequence
        d - dimension
        e - dimension (pairwise rep)
        i - source sequence
        j - context sequence
        w - local attention windows
        """

        self.scale = scale
        self.dropout = dropout

        self.is_local_attn = exists(window_size)
        self.window_size = window_size

        self.attn_dropout = nn.Dropout(dropout)

        # softclamp attention logits
        # being adopted by a number of recent llms (gemma, grok)

        self.enable_attn_softclamp = enable_attn_softclamp
        self.attn_softclamp_value = attn_softclamp_value

        # whether to use full precision for softmax
        self.softmax_full_precision = softmax_full_precision

    @typecheck
    def local_attn(
        self,
        q: Float["b h n d"],  # type: ignore
        k: Float["b h n d"],  # type: ignore
        v: Float["b h n d"],  # type: ignore
        mask: Bool["b n"] | None = None,  # type: ignore
        windowed_mask: Bool["b nw w (w*2)"] | None = None,  # type: ignore
        attn_bias: Float["... n n"] | Float["... nw w (w*2)"] | None = None,  # type: ignore
        memory_kv: Float["2 h m d"] | None = None,  # type: ignore
    ) -> Float["b h n d"]:  # type: ignore
        """Run simple local attention with a radius of 1 window size.

        :param q: The query tensor.
        :param k: The key tensor.
        :param v: The value tensor.
        :param mask: The mask to apply to the sequence.
        :param attn_bias: The attention bias to apply.
        :param memory_kv: The memory key and value tensors.
        :return: The output tensor.
        """

        window_size, batch, seq_len, device = (
            self.window_size,
            q.shape[0],
            q.shape[-2],
            q.device,
        )

        # constitute mask if not given

        if not_exists(mask):
            mask = torch.ones((batch, seq_len), device=device, dtype=torch.bool)

        # pad to multiple of window size if needed

        padding_needed = (window_size - (seq_len % window_size)) % window_size

        if padding_needed > 0:
            q, k, v = tuple(
                pad_at_dim(t, (0, padding_needed), value=0.0, dim=-2) for t in (q, k, v)
            )
            mask = F.pad(mask, (0, padding_needed), value=False)

        # break into windows

        q, k, v = tuple(rearrange(t, "b h (n w) d -> b h n w d", w=window_size) for t in (q, k, v))
        mask = rearrange(mask, "b (n w) -> b n w", w=window_size)

        # just do radius of 1 for now
        # perhaps not even necessary, and could try shifted windows (a la Swin)

        k, v = tuple(pad_at_dim(t, (1, 0), dim=-3) for t in (k, v))
        mask = pad_at_dim(mask, (1, 0), dim=-2, value=False)

        k, v = tuple(torch.cat((t[..., :-1, :, :], t[..., 1:, :, :]), dim=-2) for t in (k, v))
        mask = torch.cat((mask[..., :-1, :], mask[..., 1:, :]), dim=-1)

        # handle attention bias (inefficiently)

        is_full_attn_bias = attn_bias.shape[-1] == attn_bias.shape[-2]

        if exists(attn_bias) and is_full_attn_bias:
            attn_bias = full_attn_bias_to_windowed(attn_bias, window_size=window_size)

        # carry out attention as usual

        scale = q.shape[-1] ** -0.5

        q = q * scale

        # append memory key / values for local attention windows

        if exists(memory_kv):
            batch, seq, num_mem_kv = k.shape[0], k.shape[2], memory_kv.shape[-2]

            mk, mv = memory_kv
            mk, mv = tuple(repeat(t, "h m d -> b h n m d", b=batch, n=seq) for t in (mk, mv))
            k = torch.cat((mk, k), dim=-2)
            v = torch.cat((mv, v), dim=-2)

            if exists(attn_bias):
                attn_bias = pad_at_dim(attn_bias, (num_mem_kv, 0), value=0.0)

            if exists(windowed_mask):
                windowed_mask = pad_at_dim(windowed_mask, (num_mem_kv, 0), value=True)

            if exists(mask):
                mask = pad_at_dim(mask, (num_mem_kv, 0), value=True)

        # similarity

        sim = einops.einsum(q, k, "... i d, ... j d -> ... i j")

        if exists(attn_bias):
            if attn_bias.ndim == 4:
                attn_bias = rearrange(attn_bias, "b ... -> b 1 ...")

            assert attn_bias.ndim == sim.ndim
            sim = sim + attn_bias

        # maybe softclamp

        if self.enable_attn_softclamp:
            sim = softclamp(sim, self.attn_softclamp_value)

        # windowed masking - for masking out atoms not belonging to the same molecule / polypeptide / nucleic acid in sequence-local attention

        if exists(windowed_mask):
            # sim = einx.where(
            #     "b n i j, b h n i j, -> b h n i j", windowed_mask, sim, max_neg_value(sim)
            # )
            sim = sim.masked_fill(~windowed_mask[:, None, ...], max_neg_value(sim))

        # mask out buckets of padding

        # sim = einx.where("b n j, b h n i j, -> b h n i j", mask, sim, max_neg_value(sim))
        sim = sim.masked_fill(~mask[:, None, :, None, :], max_neg_value(sim))

        # local attention

        attn = sim.softmax(dim=-1)

        # aggregate

        out = einops.einsum(attn, v, "... i j, ... j d -> ... i d")

        # un-window the output

        out = rearrange(out, "b h n w d -> b h (n w) d")

        # excise the padding for windowing

        out = out[..., :seq_len, :]

        return out

    @typecheck
    def forward(
        self,
        q: Float["b h i d"],  # type: ignore
        k: Float["b h j d"],  # type: ignore
        v: Float["b h j d"],  # type: ignore
        mask: Bool["b j"] | None = None,  # type: ignore
        windowed_mask: Bool["b nw w (w*2)"] | None = None,  # type: ignore
        attn_bias: Float["... i j"] | Float["... nw w (w*2)"] | None = None,  # type: ignore
        memory_kv: Float["2 h m d"] | None = None,  # type: ignore
        batch_size: int = 1,
        use_optimized_evo: Literal["deepspeed", "triton"] | None = None,
    ) -> Float["b h i d"]:  # type: ignore
        """Run attention.

        :param q: The query tensor.
        :param k: The key tensor.
        :param v: The value tensor.
        :param mask: The mask to apply to the sequence.
        :param attn_bias: The attention bias to apply.
        :param memory_kv: The memory key and value tensors.
        :param batch_size: The original (unpadded) input batch size.
        :param use_optimized_evo: Whether to use an optimized Evoformer kernel.
        :return: The output tensor.
        """
        dtype = q.dtype
        seq_len = q.shape[-2]

        is_windowed_attn_bias = None

        if exists(attn_bias):
            is_windowed_attn_bias = attn_bias.shape[-1] != attn_bias.shape[-2]

        # local windowed attention

        if self.is_local_attn:
            return self.local_attn(
                q,
                k,
                v,
                mask=mask,
                windowed_mask=windowed_mask,
                attn_bias=attn_bias,
                memory_kv=memory_kv,
            )

        assert (
            not_exists(is_windowed_attn_bias) or not is_windowed_attn_bias
        ), "Windowed attention bias is not supported with full attention."

        # append memory key / values

        if exists(memory_kv):
            batch, num_mem_kv = q.shape[0], memory_kv.shape[-2]

            mk, mv = memory_kv
            mk, mv = tuple(repeat(t, "h m d -> b h m d", b=batch) for t in (mk, mv))
            k = torch.cat((mk, k), dim=-2)
            v = torch.cat((mv, v), dim=-2)

            if exists(attn_bias):
                attn_bias = pad_at_dim(attn_bias, (num_mem_kv, 0), value=0.0)

            if exists(mask):
                mask = pad_at_dim(mask, (num_mem_kv, 0), value=True)

        if exists(use_optimized_evo) and seq_len > 16:
            # when possible, perform attention using an optimized Evoformer kernel
            # NOTE: optimized Evoformer attention requires inputs to be in `bfloat16` precision
            orig_shape = q.shape
            b = batch_size
            h, n = q.shape[-3:-1]

            q_ = rearrange(q, "(b n1) h n2 d -> b n1 n2 h d", b=b, h=h, n2=n).bfloat16()
            k_ = rearrange(k, "(b n1) h n2 d -> b n1 n2 h d", b=b, h=h, n2=n).bfloat16()
            v_ = rearrange(v, "(b n1) h n2 d -> b n1 n2 h d", b=b, h=h, n2=n).bfloat16()

            mask_ = rearrange(mask, "(b n1) n2 -> b n1 1 1 n2", b=b, n2=n).bfloat16()
            biases = [mask_]

            if exists(attn_bias):
                attn_bias_ = rearrange(
                    attn_bias, "(b n1) h n2 n3 -> b n1 h n2 n3", b=b, h=h, n2=n, n3=n
                )[:, 0:1, ...].bfloat16()
                biases.append(attn_bias_)

            if use_optimized_evo == "deepspeed":
                assert exists(
                    DS4Sci_EvoformerAttention
                ), "DeepSpeed's EvoformerAttention not found."
                out = DS4Sci_EvoformerAttention(q_, k_, v_, biases).reshape(orig_shape).type(dtype)
            elif use_optimized_evo == "triton":
                assert exists(TritonEvoformer), "Triton's EvoformerAttention not found."
                out = (
                    TritonEvoformer(q_, k_, v_, biases[0], biases[1])
                    .reshape(orig_shape)
                    .type(dtype)
                )
            else:
                raise ValueError(
                    f"Invalid optimized Evoformer kernel selection: {use_optimized_evo}"
                )

        else:
            # default attention

            scale = default(self.scale, q.shape[-1] ** -0.5)

            q = q * scale

            # similarity

            sim = einops.einsum(q, k, "b h i d, b h j d -> b h i j")

            # attn bias

            if exists(attn_bias):
                sim = sim + attn_bias

            # maybe softclamp

            if self.enable_attn_softclamp:
                sim = softclamp(sim, self.attn_softclamp_value)

            # masking

            if exists(mask):
                # sim = einx.where("b j, b h i j, -> b h i j", mask, sim, max_neg_value(sim))
                sim = sim.masked_fill(~mask[:, None, None, :], max_neg_value(sim))

            # attention cast float32 - in case there are instabilities with lower precision

            softmax_kwargs = dict()

            if self.softmax_full_precision:
                softmax_kwargs.update(dtype=torch.float32)

            # attention

            attn = sim.softmax(dim=-1, **softmax_kwargs).to(dtype)

            # aggregate values

            out = einops.einsum(attn, v, "b h i j, b h j d -> b h i d")

        # dropout - NOTE: this is applied post-aggregation for compatibility with optimized EvoformerAttention

        out = self.attn_dropout(out)

        return out
