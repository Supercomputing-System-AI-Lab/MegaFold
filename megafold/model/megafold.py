"""
global ein notation:
a - number of tokens in a given chain (asym_id)
b - batch
ba - batch with augmentation
bt - batch with templates dimension merged
h - heads
n - molecule sequence length
i - molecule sequence length (source)
j - molecule sequence length (target)
l - number of distogram bins
f - number of input ligand fragments
m - atom sequence length
nw - windowed sequence length
d - feature dimension
ds - feature dimension (single)
dsi - feature dimension (single input)
dp - feature dimension (pairwise)
dap - feature dimension (atompair)
dapi - feature dimension (atompair input)
da - feature dimension (atom)
dai - feature dimension (atom input)
dmi - feature dimension (msa input)
dmf - additional msa feats derived from msa (has_deletion and deletion_value)
dtf - additional token feats derived from msa (profile and deletion_mean)
dac - additional pairwise token constraint embeddings
dpe - additional protein language model embeddings
dne - additional nucleotide language model embeddings
t - templates
s - msa
r - registers
ts - diffusion timesteps

additional_msa_feats: [*, 2]:
- concatted to the msa single rep
0: has_deletion
1: deletion_value

additional_token_feats: [*, 33]:
- concatted to the single rep
0: profile
1: deletion_mean

additional_molecule_feats: [*, 5]:
- used for deriving relative positions
0: molecule_index
1: token_index
2: asym_id
3: entity_id
4: sym_id

is_molecule_types: [*, 5]:
0: is_protein
1: is_rna
2: is_dna
3: is_ligand
4: is_metal_ions_or_misc
"""

from __future__ import annotations

import random
from contextlib import nullcontext
from functools import partial, wraps
from importlib.metadata import version
from itertools import zip_longest
from math import pi, sqrt
from pathlib import Path

# import einx
import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype.typing import Any, Dict, List, Literal, NamedTuple, Tuple
from Bio.PDB.Structure import Structure
from Bio.PDB.StructureBuilder import StructureBuilder
from colt5_attention import ConditionalRoutedAttention
from einops import einsum, pack, rearrange, reduce, repeat, unpack
from einops.layers.torch import Rearrange
from frame_averaging_pytorch import FrameAverage
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from loguru import logger
from taylor_series_linear_attention import TaylorSeriesLinearAttn
from torch import Tensor, tensor
from torch.nn import Linear, Module, ModuleList, Sequential
from tqdm import tqdm

from megafold.model.attention import (
    Attention,
    full_attn_bias_to_windowed,
    full_pairwise_repr_to_windowed,
)
from megafold.common.biomolecule import get_residue_constants
from megafold.inputs import (
    ADDITIONAL_MOLECULE_FEATS,
    CONSTRAINT_DIMS,
    CONSTRAINTS,
    CONSTRAINTS_MASK_VALUE,
    DEFAULT_NUM_MOLECULE_MODS,
    IS_BIOMOLECULE_INDICES,
    IS_DNA,
    IS_DNA_INDEX,
    IS_LIGAND,
    IS_LIGAND_INDEX,
    IS_METAL_ION,
    IS_METAL_ION_INDEX,
    IS_MOLECULE_TYPES,
    IS_NON_NA_INDICES,
    IS_NON_PROTEIN_INDICES,
    IS_PROTEIN,
    IS_PROTEIN_INDEX,
    IS_RNA,
    IS_RNA_INDEX,
    MAX_DNA_NUCLEOTIDE_ID,
    MIN_RNA_NUCLEOTIDE_ID,
    MISSING_RNA_NUCLEOTIDE_ID,
    NUM_HUMAN_AMINO_ACIDS,
    NUM_MOLECULE_IDS,
    NUM_MSA_ONE_HOT,
    BatchedAtomInput,
    MegaFoldInput,
    PDBInput,
    hard_validate_atom_indices_ascending,
    megafold_inputs_to_batched_atom_input,
)
from megafold.life import ATOMS
from megafold.nlm import NLMEmbedding, NLMRegistry, remove_nlms
from megafold.plm import PLMEmbedding, PLMRegistry, remove_plms
from megafold.tensor_typing import IS_DEBUGGING, Bool, Float, Int, checkpoint, typecheck
from megafold.utils.model_utils import (
    ExpressCoordinatesInFrame,
    LossBreakdown,
    MegaFoldLoss,
    RigidFrom3Points,
    RigidFromReference3Points,
    SumPooling,
    autocasting_disable_decorator,
    batch_compute_rigid_alignment,
    batch_compute_rmsd,
    batch_repeat_interleave,
    batch_repeat_interleave_pairwise,
    calculate_weighted_rigid_align_weights,
    cast_tuple,
    clamp_tensor,
    compact,
    concat_previous_window,
    create_uid_tensor,
    dict_to_device,
    dict_to_float_dtype,
    distance_to_dgram,
    exclusive_cumsum,
    freeze_,
    l2norm,
    lens_to_mask,
    log,
    masked_average,
    max_neg_value,
    maybe,
    mean_pool_fixed_windows_with_mask,
    mean_pool_with_lens,
    pack_one,
    pad_and_window,
    pad_or_slice_to,
    sample_harmonic_prior,
    save_args_and_kwargs,
    should_checkpoint,
    sum_pool_with_lens,
    symmetrize,
    to_pairwise_mask,
    weighted_rigid_align,
)
from megafold.utils.utils import (
    apply_function_to_ordered_dict_keys,
    default,
    exists,
    identity,
    not_exists,
)
from megafold.kernels.fused_layernorm_linear import LayernormLinear
from megafold.kernels.fused_transition import FusedTransition

# constants

LinearNoBias = partial(Linear, bias=False)
LayerNorm = nn.LayerNorm


# linear and outer sum
# for single repr -> pairwise pattern throughout this architecture


class LinearNoBiasThenOuterSum(Module):
    """LinearNoBias module followed by outer sum."""

    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.proj = LinearNoBias(dim, dim_out * 2)

    @typecheck
    def forward(
        self, t: Float["b n ds"]  # type: ignore
    ) -> Float["b n n dp"]:  # type: ignore
        """Perform the forward pass.

        :param t: The input tensor.
        :return: The output tensor.
        """
        single_i, single_j = self.proj(t).chunk(2, dim=-1)
        # out = einx.add("b i d, b j d -> b i j d", single_i, single_j)
        out = single_i[..., None, :] + single_j[:, None, ...]
        return out


# classic feedforward, SwiGLU variant
# they name this "transition" in their paper
# Algorithm 11


# class SwiGLU(Module):
#     """Swish-Gated Linear Unit."""

#     @typecheck
#     def forward(
#         self, x: Float["... d"]  # type: ignore
#     ) -> Float[" ... (d//2)"]:  # type: ignore
#         """Perform the forward pass.

#         :param x: The input tensor.
#         :return: The output tensor.
#         """
#         x, gates = x.chunk(2, dim=-1)
#         return F.silu(gates) * x

class Transition(Module):
    """A Transition module."""

    def __init__(self, *, dim, expansion_factor=4):
        super().__init__()
        dim_inner = int(dim * expansion_factor)

        self.ff = Sequential(
            LinearNoBias(dim, dim_inner * 2),
            SwiGLU(),
            LinearNoBias(dim_inner, dim),
        )

    @typecheck
    def forward(
        self, x: Float["... d"]  # type: ignore
    ) -> Float["... d"]:  # type: ignore
        """Perform the forward pass.

        :param x: The input tensor.
        :return: The output tensor.
        """
        return self.ff(x)


# dropout
# they seem to be using structured dropout - row / col wise in triangle modules


class Dropout(Module):
    """A Dropout module."""

    @typecheck
    def __init__(self, prob: float, *, dropout_type: Literal["row", "col"] | None = None):
        super().__init__()
        self.dropout = nn.Dropout(prob)
        self.dropout_type = dropout_type

    @typecheck
    def forward(self, t: Tensor) -> Tensor:
        """Perform the forward pass.

        :param t: The input tensor.
        :return: The output tensor.
        """
        if self.dropout_type in {"row", "col"}:
            assert (
                t.ndim == 4
            ), "Tensor `t` must consist of 4 dimensions for row/col structured dropout."

        if not_exists(self.dropout_type):
            return self.dropout(t)

        if self.dropout_type == "row":
            batch, _, col, dim = t.shape
            ones_shape = (batch, 1, col, dim)

        elif self.dropout_type == "col":
            batch, row, _, dim = t.shape
            ones_shape = (batch, row, 1, dim)

        ones = t.new_ones(ones_shape)
        dropped = self.dropout(ones)
        return t * dropped


# normalization
# both pre layernorm as well as adaptive layernorm wrappers


class PreLayerNorm(Module):
    """A Pre-LayerNorm module."""

    @typecheck
    def __init__(
        self,
        fn: (
            Attention | Transition | TriangleAttention | TriangleMultiplication | AttentionPairBias
        ),
        *,
        dim,
    ):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    @typecheck
    def forward(
        self, x: Float["... n d"], **kwargs  # type: ignore
    ) -> Float["... n d"]:  # type: ignore
        """Perform the forward pass.

        :param x: The input tensor.
        :return: The output tensor.
        """
        x = self.norm(x)
        return self.fn(x, **kwargs)


class AdaptiveLayerNorm(Module):
    """Algorithm 26."""

    def __init__(self, *, dim, dim_cond):
        super().__init__()
        self.norm = (
            # NOTE: `elementwise_affine=False` excludes the `weight` and `bias` terms
            nn.LayerNorm(dim, elementwise_affine=False)
        )
        self.norm_cond = nn.LayerNorm(dim_cond, bias=False)

        self.to_gamma = nn.Sequential(Linear(dim_cond, dim), nn.Sigmoid())

        self.to_beta = LinearNoBias(dim_cond, dim)

    @typecheck
    def forward(
        self,
        x: Float["b n d"],  # type: ignore
        cond: Float["b n dc"],  # type: ignore
    ) -> Float["b n d"]:  # type: ignore
        """Perform the forward pass.

        :param x: The input tensor.
        :param cond: The conditional tensor.
        :return: The output tensor.
        """
        normed = self.norm(x)
        normed_cond = self.norm_cond(cond)

        gamma = self.to_gamma(normed_cond)
        beta = self.to_beta(normed_cond)
        return normed * gamma + beta


class ConditionWrapper(Module):
    """Algorithm 25."""

    @typecheck
    def __init__(
        self,
        fn: Attention | Transition | TriangleAttention | AttentionPairBias,
        *,
        dim,
        dim_cond,
        adaln_zero_bias_init_value=-2.0,
    ):
        super().__init__()
        self.fn = fn
        self.adaptive_norm = AdaptiveLayerNorm(dim=dim, dim_cond=dim_cond)

        adaln_zero_gamma_linear = Linear(dim_cond, dim)
        nn.init.zeros_(adaln_zero_gamma_linear.weight)
        nn.init.constant_(adaln_zero_gamma_linear.bias, adaln_zero_bias_init_value)

        self.to_adaln_zero_gamma = nn.Sequential(adaln_zero_gamma_linear, nn.Sigmoid())

    @typecheck
    def forward(
        self,
        x: Float["b n d"],  # type: ignore
        *,
        cond: Float["b n dc"],  # type: ignore
        **kwargs,
    ) -> Float["b n d"]:  # type: ignore
        """Perform the forward pass.

        :param x: The input tensor.
        :param cond: The conditional tensor.
        :return: The output tensor.
        """
        x = self.adaptive_norm(x, cond=cond)

        out = self.fn(x, **kwargs)

        gamma = self.to_adaln_zero_gamma(cond)
        return out * gamma


# triangle multiplicative module
# seems to be unchanged from alphafold2


class TriangleMultiplication(Module):
    """A TriangleMultiplication module from AlphaFold 2."""

    @typecheck
    def __init__(
        self,
        *,
        dim,
        dim_hidden=None,
        mix: Literal["incoming", "outgoing"] = "incoming",
        dropout=0.0,
        dropout_type: Literal["row", "col"] | None = None,
    ):
        super().__init__()

        dim_hidden = default(dim_hidden, dim)
        self.dim_hidden = dim_hidden

        # self.pre_ln = LayerNorm(dim)
        # self.left_right_proj = nn.Sequential(LinearNoBias(dim, dim_hidden * 4), nn.GLU(dim=-1))

        # self.out_gate = LinearNoBias(dim, dim_hidden)
        
        # NOTE: check correctness here + check if cause up new memory        
        self.combined = LayernormLinear(dim, dim_hidden*5)
        self.glu = nn.GLU(dim=-1)

        if mix == "outgoing":
            self.mix_einsum_eq = "... i k d, ... j k d -> ... i j d"
        elif mix == "incoming":
            self.mix_einsum_eq = "... k j d, ... k i d -> ... i j d"

        # self.to_out_norm = LayerNorm(dim_hidden)

        # self.to_out = Sequential(
        #     LinearNoBias(dim_hidden, dim),
        #     Dropout(dropout, dropout_type=dropout_type),
        # )
        self.to_out = Sequential(
            LayernormLinear(dim_hidden, dim, has_linear_bias=False), 
            Dropout(dropout, dropout_type=dropout_type),
        )

    @typecheck
    def forward(
        self,
        x: Float["b n n d"],  # type: ignore
        mask: Bool["b n"] | None = None,  # type: ignore
    ) -> Float["b n n d"]:  # type: ignore
        """Perform the forward pass.

        :param x: The input tensor.
        :param mask: The mask tensor.
        :return: The output tensor.
        """
        if exists(mask):
            mask = to_pairwise_mask(mask)
            mask = rearrange(mask, "... -> ... 1")

        # left, right = self.left_right_proj(x).chunk(2, dim=-1)
        combined_out = self.combined(x)
        left_and_right, out_gate = combined_out[..., :self.dim_hidden*4], combined_out[..., self.dim_hidden*4:].sigmoid()
        left, right = self.glu(left_and_right).chunk(2, dim=-1)

        if exists(mask):
            left = left * mask
            right = right * mask

        out = einsum(left, right, self.mix_einsum_eq)

        # out = self.to_out_norm(out)

        # out_gate = self.out_gate(x).sigmoid()

        # print("out: (K,N)=(128, 128)", out.shape)
        
        return self.to_out(out) * out_gate


# there are two types of attention in this paper, triangle and attention-pair-bias
# they differ by how the attention bias is computed
# triangle is axial attention w/ itself projected for bias


class AttentionPairBias(Module):
    """An Attention module with pair bias computation."""

    def __init__(
        self,
        *,
        heads,
        dim_pairwise,
        window_size=None,
        num_memory_kv=0,
        **attn_kwargs,
    ):
        super().__init__()

        self.window_size = window_size

        self.attn = Attention(
            heads=heads, window_size=window_size, num_memory_kv=num_memory_kv, **attn_kwargs
        )

        # line 8 of Algorithm 24

        # to_attn_bias_linear = LinearNoBias(dim_pairwise, heads)
        # nn.init.zeros_(to_attn_bias_linear.weight)

        # self.to_attn_bias = nn.Sequential(
        #     nn.LayerNorm(dim_pairwise), to_attn_bias_linear, Rearrange("b ... h -> b h ...")
        # )
        
        ln_linear = LayernormLinear(dim_pairwise, heads, has_linear_bias=False)
        nn.init.zeros_(ln_linear.linear_weight)
        self.to_attn_bias = nn.Sequential(
            ln_linear, Rearrange("b ... h -> b h ...")
        )

    @typecheck
    def forward(
        self,
        single_repr: Float["b n ds"],  # type: ignore
        *,
        pairwise_repr: Float["b n n dp"] | Float["b nw w (w*2) dp"],  # type: ignore
        attn_bias: Float["b n n"] | Float["b nw w (w*2)"] | None = None,  # type: ignore
        verbose: bool = True,
        **kwargs,
    ) -> Float["b n ds"]:  # type: ignore
        """Perform the forward pass.

        :param single_repr: The single representation tensor.
        :param pairwise_repr: The pairwise representation tensor.
        :param attn_bias: The attention bias tensor.
        :param verbose: Whether to print verbose output.
        :return: The output tensor.
        """
        b = pairwise_repr.shape[0]
        w, has_window_size = self.window_size, exists(self.window_size)

        # take care of windowing logic
        # for sequence-local atom transformer

        windowed_pairwise = pairwise_repr.ndim == 5

        windowed_attn_bias = None

        if exists(attn_bias):
            windowed_attn_bias = attn_bias.shape[-1] != attn_bias.shape[-2]

        if has_window_size:
            if not windowed_pairwise:
                pairwise_repr = full_pairwise_repr_to_windowed(pairwise_repr, window_size=w)
            if exists(attn_bias):
                attn_bias = full_attn_bias_to_windowed(attn_bias, window_size=w)
        else:
            assert (
                not windowed_pairwise
            ), "Cannot pass in windowed pairwise representation if no `window_size` given to `AttentionPairBias`."
            assert (
                not_exists(windowed_attn_bias) or not windowed_attn_bias
            ), "Cannot pass in windowed attention bias if no `window_size` is set for `AttentionPairBias`."

        # attention bias preparation with further addition from pairwise repr

        if exists(attn_bias):
            attn_bias = rearrange(attn_bias, "b ... -> b 1 ...")
        else:
            attn_bias = 0.0

        # print("pairwise_repr: (K,N)=(128, 16)", pairwise_repr.shape)
        attn_bias = self.to_attn_bias(pairwise_repr) + attn_bias

        out = self.attn(single_repr, attn_bias=attn_bias, batch_size=b, **kwargs)
        return out


class TriangleAttention(Module):
    """An Attention module with triangular bias computation."""

    def __init__(
        self,
        *,
        dim,
        heads,
        node_type: Literal["starting", "ending"],
        dropout=0.0,
        dropout_type: Literal["row", "col"] | None = None,
        **attn_kwargs,
    ):
        super().__init__()
        self.need_transpose = node_type == "ending"

        self.to_attn_bias = nn.Sequential(
            LinearNoBias(dim, heads), Rearrange("... i j h -> ... h i j")
        )
        self.dropout = Dropout(dropout, dropout_type=dropout_type)

        self.attn = Attention(dim=dim, heads=heads, **attn_kwargs)

    @typecheck
    def forward(
        self,
        pairwise_repr: Float["b n n d"],  # type: ignore
        mask: Bool["b n"] | None = None,  # type: ignore
        **kwargs,
    ) -> Float["b n n d"]:  # type: ignore
        """Perform the forward pass.

        :param pairwise_repr: The pairwise representation tensor.
        :param mask: The mask tensor.
        :return: The output tensor.
        """
        b = len(pairwise_repr)

        if self.need_transpose:
            pairwise_repr = rearrange(pairwise_repr, "b i j d -> b j i d")

        attn_bias = self.to_attn_bias(pairwise_repr)
        batch_repeat = pairwise_repr.shape[1]
        if exists(mask):
            mask = repeat(mask, "b ... -> (b repeat) ...", repeat=batch_repeat)

        attn_bias = repeat(attn_bias, "b ... -> (b repeat) ...", repeat=batch_repeat)
        pairwise_repr, unpack_one = pack_one(pairwise_repr, "* n d")  # noqa: F811
        out = self.attn(pairwise_repr, mask=mask, attn_bias=attn_bias, batch_size=b, **kwargs)
        out = unpack_one(out)

        if self.need_transpose:
            out = rearrange(out, "b j i d -> b i j d")

        return self.dropout(out)


# PairwiseBlock
# used in both MSAModule and Pairformer
# consists of all the "Triangle" modules + Transition


class PairwiseBlock(Module):
    """A PairwiseBlock module."""

    def __init__(
        self,
        *,
        dim_pairwise=128,
        tri_mult_dim_hidden=None,
        tri_attn_dim_head=32,
        tri_attn_heads=4,
        dropout_row_prob=0.25,
        dropout_col_prob=0.25,
    ):
        super().__init__()

        pre_ln = partial(PreLayerNorm, dim=dim_pairwise)

        tri_mult_kwargs = dict(dim=dim_pairwise, dim_hidden=tri_mult_dim_hidden)

        tri_attn_kwargs = dict(dim=dim_pairwise, heads=tri_attn_heads, dim_head=tri_attn_dim_head)

        # self.tri_mult_outgoing = pre_ln(
        #     TriangleMultiplication(
        #         mix="outgoing",
        #         dropout=dropout_row_prob,
        #         dropout_type="row",
        #         **tri_mult_kwargs,
        #     )
        # )
        self.tri_mult_outgoing = TriangleMultiplication(
            mix="outgoing",
            dropout=dropout_row_prob,
            dropout_type="row",
            **tri_mult_kwargs,
        )
        
        # self.tri_mult_incoming = pre_ln(
        #     TriangleMultiplication(
        #         mix="incoming",
        #         dropout=dropout_row_prob,
        #         dropout_type="row",
        #         **tri_mult_kwargs,
        #     )
        # )
        self.tri_mult_incoming = TriangleMultiplication(
            mix="incoming",
            dropout=dropout_row_prob,
            dropout_type="row",
            **tri_mult_kwargs,
        )
        
        self.tri_attn_starting = pre_ln(
            TriangleAttention(
                node_type="starting",
                dropout=dropout_row_prob,
                dropout_type="row",
                **tri_attn_kwargs,
            )
        )
        self.tri_attn_ending = pre_ln(
            TriangleAttention(
                node_type="ending",
                dropout=dropout_col_prob,
                dropout_type="col",
                **tri_attn_kwargs,
            )
        )
        self.pairwise_transition = FusedTransition(dim=dim_pairwise) # pre_ln(Transition(dim=dim_pairwise))

    @typecheck
    def forward(
        self,
        *,
        pairwise_repr: Float["b n n d"],  # type: ignore
        mask: Bool["b n"] | None = None,  # type: ignore
        **kwargs,
    ) -> Float["b n n d"]:  # type: ignore
        """Perform the forward pass.

        :param pairwise_repr: The pairwise representation tensor.
        :param mask: The mask tensor.
        :param kwargs: Additional keyword arguments for the attention computation.
        :return: The output tensor.
        """
        pairwise_repr = self.tri_mult_outgoing(pairwise_repr, mask=mask) + pairwise_repr
        pairwise_repr = self.tri_mult_incoming(pairwise_repr, mask=mask) + pairwise_repr
        pairwise_repr = self.tri_attn_starting(pairwise_repr, mask=mask, **kwargs) + pairwise_repr
        pairwise_repr = self.tri_attn_ending(pairwise_repr, mask=mask, **kwargs) + pairwise_repr

        pairwise_repr = self.pairwise_transition(pairwise_repr) + pairwise_repr
        return pairwise_repr


# msa module


class OuterProductMean(Module):
    """Algorithm 9."""

    def __init__(self, *, dim_msa=64, dim_pairwise=128, dim_hidden=32, eps=1e-5):
        super().__init__()
        self.eps = eps
        # self.norm = LayerNorm(dim_msa)
        # self.to_hidden = LinearNoBias(dim_msa, dim_hidden * 2)
        self.to_hidden = LayernormLinear(dim_msa, dim_hidden * 2, has_linear_bias=False)
        self.to_pairwise_repr = nn.Linear(dim_hidden**2, dim_pairwise)

    @typecheck
    def forward(
        self,
        msa: Float["b s n d"],  # type: ignore
        *,
        mask: Bool["b n"] | None = None,  # type: ignore
        msa_mask: Bool["b s"] | None = None,  # type: ignore
    ) -> Float["b n n dp"]:  # type: ignore
        """Perform the forward pass.

        :param msa: The MSA tensor.
        :param mask: The mask tensor.
        :param msa_mask: The MSA mask tensor.
        :return: The output tensor.
        """
        dtype = msa.dtype
        # print("msa: (K,N)=(64,64)", msa.shape)
        # msa = self.norm(msa)

        # line 2

        a, b = self.to_hidden(msa).chunk(2, dim=-1)
        

        # maybe masked mean for outer product

        if exists(msa_mask):
            # a = einx.multiply("b s i d, b s -> b s i d", a, msa_mask.type(dtype))
            # b = einx.multiply("b s j e, b s -> b s j e", b, msa_mask.type(dtype))
            a = a * msa_mask[..., None, None].type(dtype)
            b = b * msa_mask[..., None, None].type(dtype)

            outer_product = einsum(a, b, "b s i d, b s j e -> b i j d e")

            num_msa = reduce(msa_mask.type(dtype), "... s -> ...", "sum")

            # outer_product_mean = einx.divide(
            #     "b i j d e, b", outer_product, num_msa.clamp(min=self.eps)
            # )
            outer_product_mean = outer_product / num_msa[..., None, None, None, None].clamp(
                min=self.eps
            )
        else:
            num_msa = msa.shape[1]
            outer_product = einsum(a, b, "b s i d, b s j e -> b i j d e")
            outer_product_mean = outer_product / num_msa

        # flatten

        outer_product_mean = rearrange(outer_product_mean, "... d e -> ... (d e)")

        # masking for pairwise repr

        if exists(mask):
            mask = to_pairwise_mask(mask)
            # outer_product_mean = einx.multiply(
            #     "b i j d, b i j", outer_product_mean, mask.type(dtype)
            # )
            outer_product_mean = outer_product_mean * mask[..., None].type(dtype)

        pairwise_repr = self.to_pairwise_repr(outer_product_mean)
        return pairwise_repr

class MSAPairWeightedAveraging(Module):
    """Algorithm 10."""

    def __init__(
        self,
        *,
        dim_msa=64,
        dim_pairwise=128,
        dim_head=32,
        heads=8,
        dropout=0.0,
        dropout_type: Literal["row", "col"] | None = None,
    ):
        super().__init__()
        dim_inner = dim_head * heads

        self.msa_to_values_and_gates = nn.Sequential(
            #LayerNorm(dim_msa),
            #LinearNoBias(dim_msa, dim_inner * 2),
            LayernormLinear(dim_msa, dim_inner * 2, has_linear_bias=False), 
            Rearrange("b s n (gv h d) -> gv b h s n d", gv=2, h=heads),
        )

        self.pairwise_repr_to_attn = nn.Sequential(
            # LayerNorm(dim_pairwise),
            # LinearNoBias(dim_pairwise, heads),
            LayernormLinear(dim_pairwise, heads, has_linear_bias=False),
            Rearrange("b i j h -> b h i j"),
        )

        self.to_out = nn.Sequential(
            Rearrange("b h s n d -> b s n (h d)"),
            LinearNoBias(dim_inner, dim_msa),
            Dropout(dropout, dropout_type=dropout_type),
        )

    @typecheck
    def forward(
        self,
        *,
        msa: Float["b s n d"],  # type: ignore
        pairwise_repr: Float["b n n dp"],  # type: ignore
        mask: Bool["b n"] | None = None,  # type: ignore
    ) -> Float["b s n d"]:  # type: ignore
        """Perform the forward pass.

        :param msa: The MSA tensor.
        :param pairwise_repr: The pairwise representation tensor.
        :param mask: The mask tensor.
        :return: The output tensor.
        """
        # print("msa: (K,N)=(64,512)", msa.shape)
        
        values, gates = self.msa_to_values_and_gates(msa)
        gates = gates.sigmoid()

        # line 3
        # print("pairwise_repr:  (K, N) = (128, 8)", pairwise_repr.shape)
        b = self.pairwise_repr_to_attn(pairwise_repr)

        if exists(mask):
            mask = rearrange(mask, "b j -> b 1 1 j")
            b = b.masked_fill(~mask, max_neg_value(b))

        # line 5

        weights = b.softmax(dim=-1)

        # line 6

        out = einsum(weights, values, "b h i j, b h s j d -> b h s i d")

        out = out * gates

        # combine heads

        return self.to_out(out)


class MSAModule(Module):
    """Algorithm 8."""

    def __init__(
        self,
        *,
        dim_single=384,
        dim_pairwise=128,
        depth=4,
        dim_msa=64,
        dim_msa_input=NUM_MSA_ONE_HOT,
        dim_additional_msa_feats=2,
        outer_product_mean_dim_hidden=32,
        msa_pwa_dropout_row_prob=0.15,
        msa_pwa_heads=8,
        msa_pwa_dim_head=32,
        checkpoint=False,
        pairwise_block_kwargs: dict = dict(),
        max_num_msa: (
            int | None
        ) = 16_000,  # NOTE: here, we improvise since the AF3 paper does not specify this
        layerscale_output: bool = True,
    ):
        super().__init__()

        self.max_num_msa = default(
            max_num_msa, float("inf")
        )  # cap the number of MSAs, will do sample without replacement if exceeds

        self.msa_init_proj = LinearNoBias(dim_msa_input + dim_additional_msa_feats, dim_msa)

        self.single_to_msa_feats = LinearNoBias(dim_single, dim_msa)

        layers = ModuleList([])

        for _ in range(depth):
            msa_pre_ln = partial(PreLayerNorm, dim=dim_msa)

            outer_product_mean = OuterProductMean(
                dim_msa=dim_msa,
                dim_pairwise=dim_pairwise,
                dim_hidden=outer_product_mean_dim_hidden,
            )

            msa_pair_weighted_avg = MSAPairWeightedAveraging(
                dim_msa=dim_msa,
                dim_pairwise=dim_pairwise,
                heads=msa_pwa_heads,
                dim_head=msa_pwa_dim_head,
                dropout=msa_pwa_dropout_row_prob,
                dropout_type="row",
            )

            msa_transition = FusedTransition(dim=dim_msa)

            pairwise_block = PairwiseBlock(
                dim_pairwise=dim_pairwise,
                **pairwise_block_kwargs,
            )

            layers.append(
                ModuleList(
                    [
                        outer_product_mean,
                        msa_pair_weighted_avg,
                        msa_transition, # msa_pre_ln(msa_transition),
                        pairwise_block,
                    ]
                )
            )

        self.checkpoint = checkpoint

        self.layers = layers

        self.layerscale_output = (
            nn.Parameter(torch.zeros(dim_pairwise)) if layerscale_output else 1.0
        )

        # msa related

        self.dmi = dim_additional_msa_feats

    @typecheck
    def to_layers(
        self,
        *,
        pairwise_repr: Float["b n n dp"],  # type: ignore
        msa: Float["b s n dm"],  # type: ignore
        mask: Bool["b n"] | None = None,  # type: ignore
        msa_mask: Bool["b s"] | None = None,  # type: ignore
        **kwargs,
    ) -> Float["b n n dp"]:  # type: ignore
        """Perform the forward pass with individual layers.

        :param single_repr: The single representation tensor.
        :param pairwise_repr: The pairwise representation tensor.
        :param msa: The MSA tensor.
        :param mask: The mask tensor.
        :param msa_mask: The MSA mask tensor.
        :param kwargs: Additional keyword arguments for the attention computation.
        :return: The output tensor.
        """
        for (
            outer_product_mean,
            msa_pair_weighted_avg,
            msa_transition,
            pairwise_block,
        ) in self.layers:
            # communication between msa and pairwise rep

            pairwise_repr = outer_product_mean(msa, mask=mask, msa_mask=msa_mask) + pairwise_repr

            msa = msa_pair_weighted_avg(msa=msa, pairwise_repr=pairwise_repr, mask=mask) + msa
            msa = msa_transition(msa) + msa

            # pairwise block

            pairwise_repr = pairwise_block(pairwise_repr=pairwise_repr, mask=mask, **kwargs)

        # ensure the MSA weights are always in the computational graph

        pairwise_repr = pairwise_repr + (0.0 * msa.mean((-3, -1))[..., None, None])

        return pairwise_repr

    @typecheck
    def to_checkpointed_layers(
        self,
        *,
        pairwise_repr: Float["b n n dp"],  # type: ignore
        msa: Float["b s n dm"],  # type: ignore
        mask: Bool["b n"] | None = None,  # type: ignore
        msa_mask: Bool["b s"] | None = None,  # type: ignore
        **kwargs,
    ) -> Float["b n n dp"]:  # type: ignore
        """Perform the forward pass with checkpointed layers.

        :param single_repr: The single representation tensor.
        :param pairwise_repr: The pairwise representation tensor.
        :param msa: The MSA tensor.
        :param mask: The mask tensor.
        :param msa_mask: The MSA mask tensor.
        :param kwargs: Additional keyword arguments for the attention computation.
        :return: The output tensor.
        """
        inputs = (pairwise_repr, mask, msa, msa_mask, kwargs)

        wrapped_layers = []

        def outer_product_mean_wrapper(fn):
            @wraps(fn)
            def inner(inputs):
                pairwise_repr, mask, msa, msa_mask, kwargs = inputs
                pairwise_repr = fn(msa=msa, mask=mask, msa_mask=msa_mask) + pairwise_repr
                return pairwise_repr, mask, msa, msa_mask, kwargs

            return inner

        def msa_pair_weighted_avg_wrapper(fn):
            @wraps(fn)
            def inner(inputs):
                pairwise_repr, mask, msa, msa_mask, kwargs = inputs
                msa = fn(msa=msa, pairwise_repr=pairwise_repr, mask=mask) + msa
                return pairwise_repr, mask, msa, msa_mask, kwargs

            return inner

        def msa_transition_wrapper(fn):
            @wraps(fn)
            def inner(inputs):
                pairwise_repr, mask, msa, msa_mask, kwargs = inputs
                msa = fn(msa) + msa
                return pairwise_repr, mask, msa, msa_mask, kwargs

            return inner

        def pairwise_block_wrapper(fn):
            @wraps(fn)
            def inner(inputs):
                pairwise_repr, mask, msa, msa_mask, kwargs = inputs
                pairwise_repr = fn(pairwise_repr=pairwise_repr, mask=mask, **kwargs)
                return pairwise_repr, mask, msa, msa_mask, kwargs

            return inner

        for (
            outer_product_mean,
            msa_pair_weighted_avg,
            msa_transition,
            pairwise_block,
        ) in self.layers:
            wrapped_layers.append(outer_product_mean_wrapper(outer_product_mean))
            wrapped_layers.append(msa_pair_weighted_avg_wrapper(msa_pair_weighted_avg))
            wrapped_layers.append(msa_transition_wrapper(msa_transition))
            wrapped_layers.append(pairwise_block_wrapper(pairwise_block))

        for layer in wrapped_layers:
            inputs = checkpoint(layer, inputs)

        pairwise_repr, _, msa, *_ = inputs

        # ensure the MSA weights are always in the computational graph

        pairwise_repr = pairwise_repr + (0.0 * msa.mean((-3, -1))[..., None, None])

        return pairwise_repr

    @typecheck
    def forward(
        self,
        *,
        single_repr: Float["b n ds"],  # type: ignore
        pairwise_repr: Float["b n n dp"],  # type: ignore
        msa: Float["b s n dm"],  # type: ignore
        mask: Bool["b n"] | None = None,  # type: ignore
        msa_mask: Bool["b s"] | None = None,  # type: ignore
        additional_msa_feats: Float["b s n {self.dmi}"] | None = None,  # type: ignore
        **kwargs,
    ) -> Float["b n n dp"]:  # type: ignore
        """Perform the forward pass.

        :param single_repr: The single representation tensor.
        :param pairwise_repr: The pairwise representation tensor.
        :param msa: The MSA tensor.
        :param mask: The mask tensor.
        :param msa_mask: The MSA mask tensor.
        :param additional_msa_feats: The additional MSA features tensor.
        :param kwargs: Additional keyword arguments for the attention computation.
        :return: The output tensor.
        """
        batch, num_msa, device = *msa.shape[:2], msa.device

        # sample without replacement

        if num_msa > self.max_num_msa:
            rand = torch.randn((batch, num_msa), device=device)

            if exists(msa_mask):
                rand.masked_fill_(~msa_mask, max_neg_value(msa))

            indices = rand.topk(self.max_num_msa, dim=-1).indices

            # msa = einx.get_at('b [s] n dm, b sampled -> b sampled n dm', msa, indices)

            msa, unpack_one = pack_one(msa, "b s *")
            msa_indices = repeat(indices, "b sampled -> b sampled d", d=msa.shape[-1])
            msa = msa.gather(1, msa_indices)
            msa = unpack_one(msa)

            if exists(msa_mask):
                # msa_mask = einx.get_at('b [s], b sampled -> b sampled', msa_mask, indices)

                msa_mask = msa_mask.gather(1, indices)

            if exists(additional_msa_feats):
                # additional_msa_feats = einx.get_at('b s 2, b sampled -> b sampled 2', additional_msa_feats, indices)

                additional_msa_feats, unpack_one = pack_one(additional_msa_feats, "b s *")
                additional_msa_indices = repeat(
                    indices, "b sampled -> b sampled d", d=additional_msa_feats.shape[-1]
                )
                additional_msa_feats = additional_msa_feats.gather(1, additional_msa_indices)
                additional_msa_feats = unpack_one(additional_msa_feats)

        # account for no msa

        if exists(msa_mask):
            has_msa = reduce(msa_mask, "b s -> b", "any")

        # account for additional msa features

        if exists(additional_msa_feats):
            msa = torch.cat((msa, additional_msa_feats), dim=-1)

        # process msa

        msa = self.msa_init_proj(msa)

        single_msa_feats = self.single_to_msa_feats(single_repr)

        msa = rearrange(single_msa_feats, "b n d -> b 1 n d") + msa

        # going through the layers

        if should_checkpoint(self, (pairwise_repr, msa)):
            to_layers_fn = self.to_checkpointed_layers
        else:
            to_layers_fn = self.to_layers

        pairwise_repr = to_layers_fn(
            msa=msa, mask=mask, pairwise_repr=pairwise_repr, msa_mask=msa_mask, **kwargs
        )

        # final masking and then layer scale

        if exists(msa_mask):
            # pairwise_repr = einx.where("b, b ..., -> b ...", has_msa, pairwise_repr, 0.0)
            pairwise_repr = pairwise_repr * has_msa[..., None, None, None]

        return pairwise_repr * self.layerscale_output


# pairformer stack


class PairformerStack(Module):
    """Algorithm 17."""

    def __init__(
        self,
        *,
        dim_single=384,
        dim_pairwise=128,
        depth=48,
        recurrent_depth=1,  # effective depth will be `depth` * `recurrent_depth`
        pair_bias_attn_dim_head=64,
        pair_bias_attn_heads=16,
        dropout_row_prob=0.25,
        num_register_tokens=0,
        checkpoint=False,
        pairwise_block_kwargs: dict = dict(),
        pair_bias_attn_kwargs: dict = dict(),
    ):
        super().__init__()
        layers = ModuleList([])

        pair_bias_attn_kwargs = dict(
            dim=dim_single,
            dim_pairwise=dim_pairwise,
            heads=pair_bias_attn_heads,
            dim_head=pair_bias_attn_dim_head,
            dropout=dropout_row_prob,
            **pair_bias_attn_kwargs,
        )

        for _ in range(depth):
            single_pre_ln = partial(PreLayerNorm, dim=dim_single)

            pairwise_block = PairwiseBlock(
                dim_pairwise=dim_pairwise,
                **pairwise_block_kwargs,
            )

            pair_bias_attn = AttentionPairBias(**pair_bias_attn_kwargs)
            single_transition = FusedTransition(dim=dim_single)

            layers.append(
                ModuleList(
                    [
                        pairwise_block,
                        single_pre_ln(pair_bias_attn),
                        single_transition, # single_pre_ln(single_transition),
                    ]
                )
            )

        self.layers = layers

        # checkpointing

        self.checkpoint = checkpoint

        # https://arxiv.org/abs/2405.16039 and https://arxiv.org/abs/2405.15071
        # although possibly recycling already takes care of this

        assert recurrent_depth > 0
        self.recurrent_depth = recurrent_depth

        self.num_registers = num_register_tokens
        self.has_registers = num_register_tokens > 0

        if self.has_registers:
            self.single_registers = nn.Parameter(torch.zeros(num_register_tokens, dim_single))
            self.pairwise_row_registers = nn.Parameter(
                torch.zeros(num_register_tokens, dim_pairwise)
            )
            self.pairwise_col_registers = nn.Parameter(
                torch.zeros(num_register_tokens, dim_pairwise)
            )

    @typecheck
    def to_layers(
        self,
        *,
        single_repr: Float["b n ds"],  # type: ignore
        pairwise_repr: Float["b n n dp"],  # type: ignore
        mask: Bool["b n"] | None = None,  # type: ignore
        **kwargs,
    ) -> Tuple[Float["b n ds"], Float["b n n dp"]]:  # type: ignore
        """Convert the module to a non-checkpointed version.

        :param single_repr: The single representation tensor.
        :param pairwise_repr: The pairwise representation tensor.
        :param mask: The mask tensor. :return The output tensors.
        :param kwargs: Additional keyword arguments for the attention computation.
        :return: The output tensors.
        """
        for _ in range(self.recurrent_depth):
            for pairwise_block, pair_bias_attn, single_transition in self.layers:
                pairwise_repr = pairwise_block(pairwise_repr=pairwise_repr, mask=mask, **kwargs)

                single_repr = (
                    pair_bias_attn(single_repr, pairwise_repr=pairwise_repr, mask=mask, **kwargs)
                    + single_repr
                )
                single_repr = single_transition(single_repr) + single_repr

        return single_repr, pairwise_repr

    @typecheck
    def to_checkpointed_layers(
        self,
        *,
        single_repr: Float["b n ds"],  # type: ignore
        pairwise_repr: Float["b n n dp"],  # type: ignore
        mask: Bool["b n"] | None = None,  # type: ignore
        **kwargs,
    ) -> Tuple[Float["b n ds"], Float["b n n dp"]]:  # type: ignore
        """Convert the module to a checkpointed version.

        :param single_repr: The single representation tensor.
        :param pairwise_repr: The pairwise representation tensor.
        :param mask: The mask tensor.
        :param kwargs: Additional keyword arguments for the attention computation.
        :return: The output tensors.
        """
        inputs = (single_repr, pairwise_repr, mask, kwargs)

        def pairwise_block_wrapper(layer):
            def inner(inputs, *args, **kwargs):
                single_repr, pairwise_repr, mask, kwargs = inputs
                pairwise_repr = layer(pairwise_repr=pairwise_repr, mask=mask, **kwargs)
                return single_repr, pairwise_repr, mask, kwargs

            return inner

        def pair_bias_attn_wrapper(layer):
            def inner(inputs, *args, **kwargs):
                single_repr, pairwise_repr, mask, kwargs = inputs
                single_repr = (
                    layer(single_repr, pairwise_repr=pairwise_repr, mask=mask, **kwargs)
                    + single_repr
                )
                return single_repr, pairwise_repr, mask, kwargs

            return inner

        def single_transition_wrapper(layer):
            def inner(inputs, *args, **kwargs):
                single_repr, pairwise_repr, mask, kwargs = inputs
                single_repr = layer(single_repr) + single_repr
                return single_repr, pairwise_repr, mask, kwargs

            return inner

        wrapped_layers = []

        for _ in range(self.recurrent_depth):
            for pairwise_block, pair_bias_attn, single_transition in self.layers:
                wrapped_layers.append(pairwise_block_wrapper(pairwise_block))
                wrapped_layers.append(pair_bias_attn_wrapper(pair_bias_attn))
                wrapped_layers.append(single_transition_wrapper(single_transition))

        for layer in wrapped_layers:
            inputs = checkpoint(layer, inputs)

        single_repr, pairwise_repr, *_ = inputs
        return single_repr, pairwise_repr

    @typecheck
    def forward(
        self,
        *,
        single_repr: Float["b n ds"],  # type: ignore
        pairwise_repr: Float["b n n dp"],  # type: ignore
        mask: Bool["b n"] | None = None,  # type: ignore
        **kwargs,
    ) -> Tuple[Float["b n ds"], Float["b n n dp"]]:  # type: ignore
        """Perform the forward pass.

        :param single_repr: The single representation tensor.
        :param pairwise_repr: The pairwise representation tensor.
        :param mask: The mask tensor.
        :param kwargs: Additional keyword arguments for the attention computation.
        :return: The output tensors.
        """
        # prepend register tokens

        if self.has_registers:
            batch_size, num_registers = (
                single_repr.shape[0],
                self.num_registers,
            )
            single_registers = repeat(self.single_registers, "r d -> b r d", b=batch_size)
            single_repr = torch.cat((single_registers, single_repr), dim=1)

            row_registers = repeat(
                self.pairwise_row_registers,
                "r d -> b r n d",
                b=batch_size,
                n=pairwise_repr.shape[-2],
            )
            pairwise_repr = torch.cat((row_registers, pairwise_repr), dim=1)
            col_registers = repeat(
                self.pairwise_col_registers,
                "r d -> b n r d",
                b=batch_size,
                n=pairwise_repr.shape[1],
            )
            pairwise_repr = torch.cat((col_registers, pairwise_repr), dim=2)

            if exists(mask):
                mask = F.pad(mask, (num_registers, 0), value=True)

        # maybe checkpoint

        if should_checkpoint(self, (single_repr, pairwise_repr)):
            to_layers_fn = self.to_checkpointed_layers
        else:
            to_layers_fn = self.to_layers

        # main transformer block layers

        single_repr, pairwise_repr = to_layers_fn(
            single_repr=single_repr, pairwise_repr=pairwise_repr, mask=mask, **kwargs
        )

        # splice out registers

        if self.has_registers:
            single_repr = single_repr[:, num_registers:]
            pairwise_repr = pairwise_repr[:, num_registers:, num_registers:]

        return single_repr, pairwise_repr


# embedding related


class RelativePositionEncoding(Module):
    """Algorithm 3."""

    def __init__(self, *, r_max=32, s_max=2, dim_out=128):
        super().__init__()
        self.r_max = r_max
        self.s_max = s_max

        dim_input = (2 * r_max + 2) + (2 * r_max + 2) + 1 + (2 * s_max + 2)
        self.out_embedder = LinearNoBias(dim_input, dim_out)

    @typecheck
    def forward(
        self, *, additional_molecule_feats: Int[f"b n {ADDITIONAL_MOLECULE_FEATS}"]  # type: ignore
    ) -> Float["b n n dp"]:  # type: ignore
        """Perform the forward pass.

        :param additional_molecule_feats: The additional molecule features tensor.
        :return: The output tensor.
        """

        dtype = self.out_embedder.weight.dtype
        device = additional_molecule_feats.device
        assert (
            additional_molecule_feats.shape[-1] >= 5
        ), "Additional molecule features must have at least 5 dimensions."

        res_idx, token_idx, asym_id, entity_id, sym_id = additional_molecule_feats.unbind(dim=-1) # each is (b, n) 

        diff_res_idx = res_idx.unsqueeze(-1) - res_idx.unsqueeze(-2) # (b, n, n)
        diff_token_idx = token_idx.unsqueeze(-1) - token_idx.unsqueeze(-2) # (b, n, n)
        diff_sym_id = sym_id.unsqueeze(-1) - sym_id.unsqueeze(-2) # (b, n, n)

        # mask_same_chain = einx.subtract("b i, b j -> b i j", asym_id, asym_id) == 0
        mask_same_chain = (asym_id.unsqueeze(-1) - asym_id.unsqueeze(-2)) == 0 # (b, n, n)
        mask_same_res = diff_res_idx == 0 # (b, n, n)
        mask_same_entity = (entity_id.unsqueeze(-1) - entity_id.unsqueeze(-2)).unsqueeze(-1) == 0 # (b, n, n, 1)

        d_res = torch.where(
            mask_same_chain,
            torch.clip(diff_res_idx + self.r_max, 0, 2 * self.r_max),
            2 * self.r_max + 1,
        ) # (b, n, n)

        d_token = torch.where(
            mask_same_chain * mask_same_res,
            torch.clip(diff_token_idx + self.r_max, 0, 2 * self.r_max),
            2 * self.r_max + 1,
        ) # (b, n, n)

        d_chain = torch.where(
            mask_same_entity.squeeze(-1),  # NOTE: Protenix (and now AF3's source code) reports this is the correct implementation
            torch.clip(diff_sym_id + self.s_max, 0, 2 * self.s_max),
            2 * self.s_max + 1,
        ) # (b, n, n)

        def onehot(x, bins): # Goal: onehot for x values s.t. x_values get map to the closest bucket of the bins
            dist_from_bins = x.unsqueeze(-1) - bins
            indices = dist_from_bins.abs().min(dim=-1, keepdim=True).indices
            one_hots = F.one_hot(indices.long(), num_classes=len(bins))
            return one_hots.type(dtype)

        r_arange = torch.arange(2 * self.r_max + 2, device=device)
        s_arange = torch.arange(2 * self.s_max + 2, device=device)

        a_rel_pos = onehot(d_res, r_arange) # (b, n, n, len(r_arange))
        a_rel_token = onehot(d_token, r_arange) # (b, n, n, len(r_arange))
        a_rel_chain = onehot(d_chain, s_arange) # (b, n, n, len(s_arange))

        out, _ = pack((a_rel_pos, a_rel_token, mask_same_entity, a_rel_chain), "b i j *")

        return self.out_embedder(out)


class TemplateEmbedder(Module):
    """Algorithm 16."""

    def __init__(
        self,
        *,
        dim_template_feats,
        dim=64,
        dim_pairwise=128,
        pairformer_stack_depth=2,
        pairwise_block_kwargs: dict = dict(),
        eps=1e-5,
        checkpoint=False,
        layerscale_output=True,
    ):
        super().__init__()
        self.eps = eps

        self.template_feats_to_embed_input = LinearNoBias(dim_template_feats, dim)

        # self.pairwise_to_embed_input = nn.Sequential(
        #     LayerNorm(dim_pairwise), LinearNoBias(dim_pairwise, dim)
        # )
        self.pairwise_to_embed_input = LayernormLinear(dim_pairwise, dim, has_linear_bias=False)

        layers = ModuleList([])
        for _ in range(pairformer_stack_depth):
            block = PairwiseBlock(dim_pairwise=dim, **pairwise_block_kwargs)

            layers.append(block)

        self.pairformer_stack = layers

        self.checkpoint = checkpoint

        self.final_norm = LayerNorm(dim)

        # final projection of mean pooled repr -> out

        self.to_out = nn.Sequential(nn.ReLU(), LinearNoBias(dim, dim_pairwise))

        self.layerscale = nn.Parameter(torch.zeros(dim_pairwise)) if layerscale_output else 1.0

    @typecheck
    def to_layers(
        self,
        templates: Float["bt n n dt"],  # type: ignore
        *,
        mask: Bool["bt n"] | None = None,  # type: ignore
        **kwargs,
    ) -> Float["bt n n dt"]:  # type: ignore
        """Perform the forward pass with individual layers.

        :param templates: The templates tensor.
        :param mask: The mask tensor.
        :param kwargs: Additional keyword arguments for the attention computation.
        :return: The output tensor.
        """
        for block in self.pairformer_stack:
            templates = block(pairwise_repr=templates, mask=mask, **kwargs) + templates

        return templates

    @typecheck
    def to_checkpointed_layers(
        self,
        templates: Float["bt n n dt"],  # type: ignore
        *,
        mask: Bool["bt n"] | None = None,  # type: ignore
        **kwargs,
    ) -> Float["bt n n dt"]:  # type: ignore
        """Perform the forward pass with checkpointed layers.

        :param templates: The templates tensor.
        :param mask: The mask tensor.
        :param kwargs: Additional keyword arguments for the attention computation.
        :return: The output tensor.
        """
        wrapped_layers = []
        inputs = (templates, mask, kwargs)

        def block_wrapper(fn):
            @wraps(fn)
            def inner(inputs):
                templates, mask, kwargs = inputs
                templates = fn(pairwise_repr=templates, mask=mask, **kwargs)
                return templates, mask, kwargs

            return inner

        for block in self.pairformer_stack:
            wrapped_layers.append(block_wrapper(block))

        for layer in wrapped_layers:
            inputs = checkpoint(layer, inputs)

        templates, *_ = inputs
        return templates

    @typecheck
    def forward(
        self,
        *,
        templates: Float["b t n n dt"],  # type: ignore
        template_mask: Bool["b t"],  # type: ignore
        pairwise_repr: Float["b n n dp"],  # type: ignore
        mask: Bool["b n"] | None = None,  # type: ignore
        **kwargs,
    ) -> Float["b n n dp"]:  # type: ignore
        """Perform the forward pass.

        :param templates: The templates tensor.
        :param template_mask: The template mask tensor.
        :param pairwise_repr: The pairwise representation tensor.
        :param mask: The mask tensor.
        :param kwargs: Additional keyword arguments for the attention computation.
        :return: The output tensor.
        """
        dtype = templates.dtype
        num_templates = templates.shape[1]

        # print("pairwise_repr: (K,N)= (128,64)", pairwise_repr.shape)
        pairwise_repr = self.pairwise_to_embed_input(pairwise_repr)
        pairwise_repr = rearrange(pairwise_repr, "b i j d -> b 1 i j d")

        templates = self.template_feats_to_embed_input(templates) + pairwise_repr

        templates, unpack_one = pack_one(templates, "* i j d")

        has_templates = reduce(template_mask, "b t -> b", "any")

        if exists(mask):
            mask = repeat(mask, "b n -> (b t) n", t=num_templates)

        # going through the pairformer stack

        if should_checkpoint(self, templates):
            to_layers_fn = self.to_checkpointed_layers
        else:
            to_layers_fn = self.to_layers

        # layers

        templates = to_layers_fn(templates, mask=mask, **kwargs)

        # final norm

        templates = self.final_norm(templates)

        templates = unpack_one(templates)

        # masked mean pool template repr

        # templates = einx.where("b t, b t ..., -> b t ...", template_mask, templates, 0.0)
        templates = templates * template_mask[..., None, None, None]

        num = reduce(templates, "b t i j d -> b i j d", "sum")
        den = reduce(template_mask.type(dtype), "b t -> b", "sum")

        # avg_template_repr = einx.divide("b i j d, b -> b i j d", num, den.clamp(min=self.eps))
        avg_template_repr = num / den[..., None, None, None].clamp(min=self.eps)

        out = self.to_out(avg_template_repr)

        # out = einx.where("b, b ..., -> b ...", has_templates, out, 0.0)
        out = out * has_templates[..., None, None, None]

        return out * self.layerscale


# diffusion related
# both diffusion transformer as well as atom encoder / decoder


class FourierEmbedding(Module):
    """Algorithm 22."""

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(1, dim)
        self.proj.requires_grad_(False)

    @typecheck
    def forward(
        self,
        times: Float[" b"],  # type: ignore
    ) -> Float["b d"]:  # type: ignore
        """Perform the forward pass.

        :param times: The times tensor.
        :return: The output tensor.
        """
        times = rearrange(times, "b -> b 1")
        rand_proj = self.proj(times)
        return torch.cos(2 * pi * rand_proj)


class PairwiseConditioning(Module):
    """Algorithm 21."""

    def __init__(
        self,
        *,
        dim_pairwise_trunk,
        dim_pairwise_rel_pos_feats,
        dim_pairwise=128,
        num_transitions=2,
        transition_expansion_factor=2,
    ):
        super().__init__()

        self.dim_pairwise_init_proj = nn.Sequential(
            LinearNoBias(dim_pairwise_trunk + dim_pairwise_rel_pos_feats, dim_pairwise),
            LayerNorm(dim_pairwise),
        )

        transitions = ModuleList([])
        for _ in range(num_transitions):
            transition = FusedTransition(dim=dim_pairwise, expansion_factor=transition_expansion_factor)
            # PreLayerNorm(
            #     Transition(
            #         dim=dim_pairwise,
            #         expansion_factor=transition_expansion_factor,
            #     ),
            #     dim=dim_pairwise,
            # )
            transitions.append(transition)

        self.transitions = transitions

    @typecheck
    def forward(
        self,
        *,
        pairwise_trunk: Float["b n n dpt"],  # type: ignore
        pairwise_rel_pos_feats: Float["b n n dpr"],  # type: ignore
    ) -> Float["b n n dp"]:  # type: ignore
        """Perform the forward pass.

        :param pairwise_trunk: The pairwise trunk tensor.
        :param pairwise_rel_pos_feats: The pairwise relative position features tensor.
        :return: The output tensor.
        """
        pairwise_repr = torch.cat((pairwise_trunk, pairwise_rel_pos_feats), dim=-1)

        pairwise_repr = self.dim_pairwise_init_proj(pairwise_repr)

        for transition in self.transitions:
            pairwise_repr = transition(pairwise_repr) + pairwise_repr

        return pairwise_repr


class SingleConditioning(Module):
    """Algorithm 21."""

    def __init__(
        self,
        *,
        sigma_data: float,
        dim_single=384,
        dim_fourier=256,
        num_transitions=2,
        transition_expansion_factor=2,
        eps=1e-20,
    ):
        super().__init__()
        self.eps = eps

        self.dim_single = dim_single
        self.sigma_data = sigma_data

        self.norm_single = LayerNorm(dim_single)

        self.fourier_embed = FourierEmbedding(dim_fourier)
        self.norm_fourier = LayerNorm(dim_fourier)
        self.fourier_to_single = LinearNoBias(dim_fourier, dim_single)

        transitions = ModuleList([])
        for _ in range(num_transitions):
            transition = FusedTransition(dim=dim_single, expansion_factor=transition_expansion_factor)
            # PreLayerNorm(
            #     Transition(
            #         dim=dim_single,
            #         expansion_factor=transition_expansion_factor,
            #     ),
            #     dim=dim_single,
            # )
            transitions.append(transition)

        self.transitions = transitions

    @typecheck
    def forward(
        self,
        *,
        times: Float[" b"],  # type: ignore
        single_trunk_repr: Float["b n dst"],  # type: ignore
        single_inputs_repr: Float["b n dsi"],  # type: ignore
    ) -> Float["b n (dst+dsi)"]:  # type: ignore
        """Perform the forward pass.

        :param times: The times tensor.
        :param single_trunk_repr: The single trunk representation tensor.
        :param single_inputs_repr: The single inputs representation tensor.
        :return: The output tensor.
        """
        single_repr = torch.cat((single_trunk_repr, single_inputs_repr), dim=-1)

        assert (
            single_repr.shape[-1] == self.dim_single
        ), "Single representation must have the correct dimension."

        single_repr = self.norm_single(single_repr)

        fourier_embed = self.fourier_embed(
            0.25 * log(times / self.sigma_data, eps=self.eps)
        )

        normed_fourier = self.norm_fourier(fourier_embed)
        fourier_to_single = self.fourier_to_single(normed_fourier)

        single_repr = rearrange(fourier_to_single, "b d -> b 1 d") + single_repr

        for transition in self.transitions:
            single_repr = transition(single_repr) + single_repr

        return single_repr


class DiffusionTransformer(Module):
    """Algorithm 23."""

    def __init__(
        self,
        *,
        depth,
        heads,
        dim=384,
        dim_single_cond=None,
        dim_pairwise=128,
        attn_window_size=None,
        attn_pair_bias_kwargs: dict = dict(),
        attn_num_memory_kv=False,
        trans_expansion_factor=2,
        num_register_tokens=0,
        add_residual=True,
        use_linear_attn=False,
        checkpoint=False,
        linear_attn_kwargs=dict(heads=8, dim_head=16),
        use_colt5_attn=False,
        colt5_attn_kwargs=dict(
            heavy_dim_head=64, heavy_heads=8, num_heavy_tokens_q=512, num_heavy_tokens_kv=512
        ),
    ):
        super().__init__()
        self.attn_window_size = attn_window_size

        dim_single_cond = default(dim_single_cond, dim)

        layers = ModuleList([])

        for _ in range(depth):
            linear_attn = None

            if use_linear_attn:
                linear_attn = TaylorSeriesLinearAttn(
                    dim=dim,
                    prenorm=True,
                    gate_value_heads=True,
                    remove_even_power_dups=True,
                    **linear_attn_kwargs,
                )

            colt5_attn = None

            if use_colt5_attn:
                colt5_attn = ConditionalRoutedAttention(
                    dim=dim, has_light_attn=False, **colt5_attn_kwargs
                )

            pair_bias_attn = AttentionPairBias(
                dim=dim,
                dim_pairwise=dim_pairwise,
                heads=heads,
                window_size=attn_window_size,
                num_memory_kv=attn_num_memory_kv,
                **attn_pair_bias_kwargs,
            )

            transition = FusedTransition(dim=dim, expansion_factor=trans_expansion_factor, include_ln=False)

            conditionable_pair_bias = ConditionWrapper(
                pair_bias_attn, dim=dim, dim_cond=dim_single_cond
            )

            conditionable_transition = ConditionWrapper(
                transition, dim=dim, dim_cond=dim_single_cond
            )

            layers.append(
                ModuleList(
                    [linear_attn, colt5_attn, conditionable_pair_bias, conditionable_transition]
                )
            )

        self.checkpoint = checkpoint

        self.layers = layers

        self.add_residual = add_residual

        self.has_registers = num_register_tokens > 0
        self.num_registers = num_register_tokens

        if self.has_registers:
            assert not_exists(
                attn_window_size
            ), "Register tokens are disabled for windowed attention."
            self.registers = nn.Parameter(torch.zeros(num_register_tokens, dim))

    @typecheck
    def to_checkpointed_serial_layers(
        self,
        noised_repr: Float["b n d"],  # type: ignore
        *,
        single_repr: Float["b n ds"],  # type: ignore
        pairwise_repr: Float["b n n dp"] | Float["b nw w (w*2) dp"],  # type: ignore
        mask: Bool["b n"] | None = None,  # type: ignore
        windowed_mask: Bool["b nw w (w*2)"] | None = None,  # type: ignore
        **kwargs,
    ) -> Float["b n d"]:  # type: ignore
        """Perform the forward pass with checkpointed serial layers.

        :param noised_repr: The noised representation tensor.
        :param single_repr: The single representation tensor.
        :param pairwise_repr: The pairwise representation tensor.
        :param mask: The mask tensor.
        :param windowed_mask: The windowed mask tensor.
        :param kwargs: Additional keyword arguments for the attention computation.
        :return: The output tensor.
        """
        inputs = (noised_repr, single_repr, pairwise_repr, mask, windowed_mask, kwargs)

        wrapped_layers = []

        def efficient_attn_wrapper(fn):
            def inner(inputs):
                noised_repr, single_repr, pairwise_repr, mask, windowed_mask, kwargs = inputs
                noised_repr = fn(noised_repr, mask=mask) + noised_repr
                return noised_repr, single_repr, pairwise_repr, mask, windowed_mask, kwargs

            return inner

        def attn_wrapper(fn):
            def inner(inputs):
                noised_repr, single_repr, pairwise_repr, mask, windowed_mask, kwargs = inputs
                noised_repr = (
                    fn(
                        noised_repr,
                        cond=single_repr,
                        pairwise_repr=pairwise_repr,
                        mask=mask,
                        windowed_mask=windowed_mask,
                        **kwargs,
                    )
                    + noised_repr
                )
                return noised_repr, single_repr, pairwise_repr, mask, windowed_mask, kwargs

            return inner

        def transition_wrapper(fn):
            def inner(inputs):
                noised_repr, single_repr, pairwise_repr, mask, windowed_mask, kwargs = inputs
                noised_repr = fn(noised_repr, cond=single_repr) + noised_repr
                return noised_repr, single_repr, pairwise_repr, mask, windowed_mask, kwargs

            return inner

        for linear_attn, colt5_attn, attn, transition in self.layers:
            if exists(linear_attn):
                wrapped_layers.append(efficient_attn_wrapper(linear_attn))

            if exists(colt5_attn):
                wrapped_layers.append(efficient_attn_wrapper(colt5_attn))

            wrapped_layers.append(attn_wrapper(attn))
            wrapped_layers.append(transition_wrapper(transition))

        for layer in wrapped_layers:
            inputs = checkpoint(layer, inputs)

        noised_repr, *_ = inputs
        return noised_repr

    @typecheck
    def to_serial_layers(
        self,
        noised_repr: Float["b n d"],  # type: ignore
        *,
        single_repr: Float["b n ds"],  # type: ignore
        pairwise_repr: Float["b n n dp"] | Float["b nw w (w*2) dp"],  # type: ignore
        mask: Bool["b n"] | None = None,  # type: ignore
        windowed_mask: Bool["b nw w (w*2)"] | None = None,  # type: ignore
        **kwargs,
    ) -> Float["b n d"]:  # type: ignore
        """Perform the forward pass with serial layers.

        :param noised_repr: The noised representation tensor.
        :param single_repr: The single representation tensor.
        :param pairwise_repr: The pairwise representation tensor.
        :param mask: The mask tensor.
        :param windowed_mask: The windowed mask tensor.
        :param kwargs: Additional keyword arguments for the attention computation.
        :return: The output tensor.
        """
        for linear_attn, colt5_attn, attn, transition in self.layers:
            if exists(linear_attn):
                noised_repr = linear_attn(noised_repr, mask=mask) + noised_repr

            if exists(colt5_attn):
                noised_repr = colt5_attn(noised_repr, mask=mask) + noised_repr

            noised_repr = (
                attn(
                    noised_repr,
                    cond=single_repr,
                    pairwise_repr=pairwise_repr,
                    mask=mask,
                    windowed_mask=windowed_mask,
                    **kwargs,
                )
                + noised_repr
            )

            noised_repr = transition(noised_repr, cond=single_repr) + noised_repr

        return noised_repr

    @typecheck
    def forward(
        self,
        noised_repr: Float["b n d"],  # type: ignore
        *,
        single_repr: Float["b n ds"],  # type: ignore
        pairwise_repr: Float["b n n dp"] | Float["b nw w (w*2) dp"],  # type: ignore
        mask: Bool["b n"] | None = None,  # type: ignore
        windowed_mask: Bool["b nw w (w*2)"] | None = None,  # type: ignore
        **kwargs,
    ) -> Float["b n d"]:  # type: ignore
        """Perform the forward pass.

        :param noised_repr: The noised representation tensor.
        :param single_repr: The single representation tensor.
        :param pairwise_repr: The pairwise representation tensor.
        :param mask: The mask tensor.
        :param windowed_mask: The windowed mask tensor.
        :param kwargs: Additional keyword arguments for the attention computation.
        :return: The output tensor.
        """
        w = self.attn_window_size
        has_windows = exists(w)

        # handle windowing

        pairwise_is_windowed = pairwise_repr.ndim == 5

        if has_windows and not pairwise_is_windowed:
            pairwise_repr = full_pairwise_repr_to_windowed(pairwise_repr, window_size=w)

        # register tokens

        if self.has_registers:
            num_registers = self.num_registers
            registers = repeat(self.registers, "r d -> b r d", b=noised_repr.shape[0])
            noised_repr, registers_ps = pack((registers, noised_repr), "b * d")

            single_repr = F.pad(single_repr, (0, 0, num_registers, 0), value=0.0)
            pairwise_repr = F.pad(
                pairwise_repr, (0, 0, num_registers, 0, num_registers, 0), value=0.0
            )

            if exists(mask):
                mask = F.pad(mask, (num_registers, 0), value=True)

        # main transformer

        if should_checkpoint(self, (noised_repr, single_repr, pairwise_repr)):
            to_layers_fn = self.to_checkpointed_serial_layers
        else:
            to_layers_fn = self.to_serial_layers

        noised_repr = to_layers_fn(
            noised_repr,
            single_repr=single_repr,
            pairwise_repr=pairwise_repr,
            mask=mask,
            windowed_mask=windowed_mask,
            **kwargs,
        )

        # splice out registers

        if self.has_registers:
            _, noised_repr = unpack(noised_repr, registers_ps, "b * d")

        return noised_repr


class AtomToTokenPooler(Module):
    """Algorithm 24."""

    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.proj = nn.Sequential(LinearNoBias(dim, dim_out), nn.ReLU())

    @typecheck
    def forward(
        self,
        *,
        atom_feats: Float["b m da"],  # type: ignore
        molecule_atom_lens: Int["b n"],  # type: ignore
    ) -> Float["b n ds"]:  # type: ignore
        """Perform the forward pass.

        :param atom_feats: The atom features tensor.
        :param molecule_atom_lens: The molecule atom lengths tensor.
        :return: The output tensor.
        """
        atom_feats = self.proj(atom_feats)
        tokens = mean_pool_with_lens(atom_feats, molecule_atom_lens)
        return tokens


class DiffusionModule(Module):
    """Algorithm 20."""

    @typecheck
    def __init__(
        self,
        *,
        dim_pairwise_trunk,
        dim_pairwise_rel_pos_feats,
        atoms_per_window=27,  # for atom sequence, take the approach of (batch, seq, atoms, ..), where atom dimension is set to the molecule or molecule with greatest number of atoms, the rest padded. atom_mask must be passed in - default to 27 for proteins, with tryptophan having 27 atoms
        dim_pairwise=128,
        sigma_data=16.0,
        dim_atom=128,
        dim_atompair=16,
        dim_token=768,
        dim_single=384,
        dim_fourier=256,
        single_cond_kwargs: dict = dict(
            num_transitions=2,
            transition_expansion_factor=2,
        ),
        pairwise_cond_kwargs: dict = dict(
            num_transitions=2,
            transition_expansion_factor=2,
        ),
        atom_encoder_depth=3,
        atom_encoder_heads=4,
        token_transformer_depth=24,
        token_transformer_heads=16,
        atom_decoder_depth=3,
        atom_decoder_heads=4,
        atom_encoder_kwargs: dict = dict(),
        atom_decoder_kwargs: dict = dict(),
        token_transformer_kwargs: dict = dict(),
        use_linear_attn=False,
        checkpoint=False,
        linear_attn_kwargs: dict = dict(heads=8, dim_head=16),
    ):
        super().__init__()

        self.atoms_per_window = atoms_per_window

        # conditioning

        self.single_conditioner = SingleConditioning(
            sigma_data=sigma_data,
            dim_single=dim_single,
            dim_fourier=dim_fourier,
            **single_cond_kwargs,
        )

        self.pairwise_conditioner = PairwiseConditioning(
            dim_pairwise_trunk=dim_pairwise_trunk,
            dim_pairwise_rel_pos_feats=dim_pairwise_rel_pos_feats,
            dim_pairwise=dim_pairwise,
            **pairwise_cond_kwargs,
        )

        # atom attention encoding related modules

        self.atom_pos_to_atom_feat = LinearNoBias(3, dim_atom)

        self.missing_atom_feat = nn.Parameter(torch.zeros(dim_atom))

        # self.single_repr_to_atom_feat_cond = nn.Sequential(
        #     LayerNorm(dim_single), LinearNoBias(dim_single, dim_atom)
        # )
        self.single_repr_to_atom_feat_cond = LayernormLinear(dim_single, dim_atom, has_linear_bias=False)

        # self.pairwise_repr_to_atompair_feat_cond = nn.Sequential(
        #     LayerNorm(dim_pairwise),
        #     LinearNoBias(dim_pairwise, dim_atompair),
        # )
        self.pairwise_repr_to_atompair_feat_cond = LayernormLinear(dim_pairwise, dim_atompair, has_linear_bias=False)

        self.atom_repr_to_atompair_feat_cond = nn.Sequential(
            # LayerNorm(dim_atom),
            # LinearNoBias(dim_atom, dim_atompair * 2),
            LayernormLinear(dim_atom, dim_atompair * 2, has_linear_bias=False),
            nn.ReLU(),
        )

        self.atompair_feats_mlp = nn.Sequential(
            LinearNoBias(dim_atompair, dim_atompair),
            nn.ReLU(),
            LinearNoBias(dim_atompair, dim_atompair),
            nn.ReLU(),
            LinearNoBias(dim_atompair, dim_atompair),
        )

        self.atom_encoder = DiffusionTransformer(
            dim=dim_atom,
            dim_single_cond=dim_atom,
            dim_pairwise=dim_atompair,
            attn_window_size=atoms_per_window,
            depth=atom_encoder_depth,
            heads=atom_encoder_heads,
            use_linear_attn=use_linear_attn,
            linear_attn_kwargs=linear_attn_kwargs,
            checkpoint=checkpoint,
            **atom_encoder_kwargs,
        )

        self.atom_feats_to_pooled_token = AtomToTokenPooler(
            dim=dim_atom,
            dim_out=dim_token,
        )

        # token attention related modules

        # self.cond_tokens_with_cond_single = nn.Sequential(
        #     LayerNorm(dim_single), LinearNoBias(dim_single, dim_token)
        # )
        self.cond_tokens_with_cond_single = LayernormLinear(dim_single, dim_token, has_linear_bias=False)

        self.token_transformer = DiffusionTransformer(
            dim=dim_token,
            dim_single_cond=dim_single,
            dim_pairwise=dim_pairwise,
            depth=token_transformer_depth,
            heads=token_transformer_heads,
            checkpoint=checkpoint,
            **token_transformer_kwargs,
        )

        # self.attended_token_norm = LayerNorm(dim_token)

        # atom attention decoding related modules

        # self.tokens_to_atom_decoder_input_cond = LinearNoBias(dim_token, dim_atom)
        self.tokens_to_atom_decoder_input_cond = LayernormLinear(dim_token, dim_atom, has_linear_bias=False)
        

        self.atom_decoder = DiffusionTransformer(
            dim=dim_atom,
            dim_single_cond=dim_atom,
            dim_pairwise=dim_atompair,
            attn_window_size=atoms_per_window,
            depth=atom_decoder_depth,
            heads=atom_decoder_heads,
            use_linear_attn=use_linear_attn,
            linear_attn_kwargs=linear_attn_kwargs,
            checkpoint=checkpoint,
            **atom_decoder_kwargs,
        )

        # self.atom_feat_to_atom_pos_update = nn.Sequential(
        #     LayerNorm(dim_atom), LinearNoBias(dim_atom, 3)
        # )
        self.atom_feat_to_atom_pos_update = LayernormLinear(dim_atom, 3, has_linear_bias=False)

    @typecheck
    def forward(
        self,
        noised_atom_pos: Float["b m 3"],  # type: ignore
        *,
        atom_feats: Float["b m da"],  # type: ignore
        atompair_feats: Float["b m m dap"] | Float["b nw w (w*2) dap"],  # type: ignore
        atom_mask: Bool["b m"],  # type: ignore
        times: Float[" b"],  # type: ignore
        mask: Bool["b n"],  # type: ignore
        single_trunk_repr: Float["b n dst"],  # type: ignore
        single_inputs_repr: Float["b n dsi"],  # type: ignore
        pairwise_trunk: Float["b n n dpt"],  # type: ignore
        pairwise_rel_pos_feats: Float["b n n dpr"],  # type: ignore
        molecule_atom_lens: Int["b n"],  # type: ignore
        atom_parent_ids: Int["b m"] | None = None,  # type: ignore
        missing_atom_mask: Bool["b m"] | None = None,  # type: ignore
        **kwargs,
    ) -> Float["b m 3"]:  # type: ignore
        """Perform the forward pass.

        :param noised_atom_pos: The noised atom position tensor.
        :param atom_feats: The atom features tensor.
        :param atompair_feats: The atom pair features tensor.
        :param atom_mask: The atom mask tensor.
        :param times: The times tensor.
        :param mask: The mask tensor.
        :param single_trunk_repr: The single trunk representation tensor.
        :param single_inputs_repr: The single inputs representation tensor.
        :param pairwise_trunk: The pairwise trunk tensor.
        :param pairwise_rel_pos_feats: The pairwise relative position features tensor.
        :param molecule_atom_lens: The molecule atom lengths tensor.
        :param atom_parent_ids: The atom parent IDs tensor.
        :param missing_atom_mask: The missing atom mask tensor.
        :param kwargs: Additional keyword arguments for the attention computation.
        :return: The output tensor.
        """
        w = self.atoms_per_window
        device = noised_atom_pos.device

        batch_size, seq_len = single_trunk_repr.shape[:2]
        atom_seq_len = atom_feats.shape[1]

        conditioned_single_repr = self.single_conditioner(
            times=times,
            single_trunk_repr=single_trunk_repr,
            single_inputs_repr=single_inputs_repr,
        )

        conditioned_pairwise_repr = self.pairwise_conditioner(
            pairwise_trunk=pairwise_trunk,
            pairwise_rel_pos_feats=pairwise_rel_pos_feats,
        )

        # remove unused keyword arguments if present

        kwargs.pop("is_molecule_types", None)
        kwargs.pop("additional_molecule_feats", None)

        # lines 7-14 in Algorithm 5

        atom_feats_cond = atom_feats

        # the most surprising part of the paper; no geometric biases!

        noised_atom_pos_feats = self.atom_pos_to_atom_feat(noised_atom_pos)

        # for missing atoms, replace the noise atom pos features with a missing embedding

        if exists(missing_atom_mask):
            # noised_atom_pos_feats = einx.where(
            #     "b m, d, b m d -> b m d",
            #     missing_atom_mask,
            #     self.missing_atom_feat,
            #     noised_atom_pos_feats,
            # )
            noised_atom_pos_feats = torch.where(
                missing_atom_mask[..., None], self.missing_atom_feat, noised_atom_pos_feats
            )

        # sum the noised atom position features to the atom features

        atom_feats = noised_atom_pos_feats + atom_feats

        # condition atom feats cond (cl) with single repr
        # print("conditioned_single_repr: (K, N)=(384, 128)", conditioned_single_repr.shape)
        single_repr_cond = self.single_repr_to_atom_feat_cond(conditioned_single_repr)

        single_repr_cond = batch_repeat_interleave(single_repr_cond, molecule_atom_lens)
        single_repr_cond = pad_or_slice_to(
            single_repr_cond, length=atom_feats_cond.shape[1], dim=1
        )

        atom_feats_cond = single_repr_cond + atom_feats_cond

        # window the atom pair features before passing to atom encoder and decoder if necessary

        atompair_is_windowed = atompair_feats.ndim == 5

        if not atompair_is_windowed:
            atompair_feats = full_pairwise_repr_to_windowed(
                atompair_feats, window_size=self.atoms_per_window
            )

        # condition atompair feats with pairwise repr
        # print("conditioned_pairwise_repr: (K, N)=(128, 16)", conditioned_pairwise_repr.shape)
        pairwise_repr_cond = self.pairwise_repr_to_atompair_feat_cond(conditioned_pairwise_repr)

        indices = torch.arange(seq_len, device=device)
        indices = repeat(indices, "n -> b n", b=batch_size)

        indices = batch_repeat_interleave(indices, molecule_atom_lens)
        indices = pad_or_slice_to(indices, atom_seq_len, dim=-1)
        indices = pad_and_window(indices, w)

        row_indices = col_indices = indices
        row_indices = rearrange(row_indices, "b n w -> b n w 1", w=w)
        col_indices = rearrange(col_indices, "b n w -> b n 1 w", w=w)

        col_indices = concat_previous_window(col_indices, dim_seq=1, dim_window=-1)
        row_indices, col_indices = torch.broadcast_tensors(row_indices, col_indices)

        # pairwise_repr_cond = einx.get_at('b [i j] dap, b nw w1 w2, b nw w1 w2 -> b nw w1 w2 dap', pairwise_repr_cond, row_indices, col_indices)

        row_indices, unpack_one = pack_one(row_indices, "b *")
        col_indices, _ = pack_one(col_indices, "b *")

        rowcol_indices = col_indices + row_indices * pairwise_repr_cond.shape[2]
        rowcol_indices = repeat(
            rowcol_indices, "b rc -> b rc dap", dap=pairwise_repr_cond.shape[-1]
        )
        pairwise_repr_cond, _ = pack_one(pairwise_repr_cond, "b * dap")

        pairwise_repr_cond = pairwise_repr_cond.gather(1, rowcol_indices)
        pairwise_repr_cond = unpack_one(pairwise_repr_cond, "b * dap")

        atompair_feats = pairwise_repr_cond + atompair_feats

        # condition atompair feats further with single atom repr
        # print("atom_feats: (K, N)=(128, 32)", atom_feats.shape)
        atom_repr_cond = self.atom_repr_to_atompair_feat_cond(atom_feats)
        atom_repr_cond = pad_and_window(atom_repr_cond, w)

        atom_repr_cond_row, atom_repr_cond_col = atom_repr_cond.chunk(2, dim=-1)

        atom_repr_cond_col = concat_previous_window(atom_repr_cond_col, dim_seq=1, dim_window=2)

        # atompair_feats = einx.add(
        #     "b nw w1 w2 dap, b nw w1 dap -> b nw w1 w2 dap", atompair_feats, atom_repr_cond_row
        # )
        # atompair_feats = einx.add(
        #     "b nw w1 w2 dap, b nw w2 dap -> b nw w1 w2 dap", atompair_feats, atom_repr_cond_col
        # )
        atompair_feats = atompair_feats + atom_repr_cond_row[..., None, :]
        atompair_feats = atompair_feats + atom_repr_cond_col[..., None, :, :]

        # furthermore, they did one more MLP on the atompair feats for attention biasing in atom transformer

        atompair_feats = self.atompair_feats_mlp(atompair_feats) + atompair_feats

        # take care of restricting atom attention to be intra molecular, if the atom_parent_ids were passed in

        windowed_mask = None

        if exists(atom_parent_ids):
            atom_parent_ids_rows = pad_and_window(atom_parent_ids, w)
            atom_parent_ids_columns = concat_previous_window(
                atom_parent_ids_rows, dim_seq=1, dim_window=2
            )

            # windowed_mask = einx.equal(
            #     "b n i, b n j -> b n i j", atom_parent_ids_rows, atom_parent_ids_columns
            # )
            windowed_mask = torch.eq(
                atom_parent_ids_rows[..., None], atom_parent_ids_columns[..., None, :]
            )

        # atom encoder

        atom_feats = self.atom_encoder(
            atom_feats,
            mask=atom_mask,
            windowed_mask=windowed_mask,
            single_repr=atom_feats_cond,
            pairwise_repr=atompair_feats,
            # NOTE: Optimized Evoformer kernels cannot be used with atomic attention yet,
            # but atomic attention is currently windowed, so it's probably not a big deal
            use_optimized_evo=None,
        )

        atom_feats_skip = atom_feats

        tokens = self.atom_feats_to_pooled_token(
            atom_feats=atom_feats,
            molecule_atom_lens=molecule_atom_lens,
        )

        # token transformer
        # print("conditioned_single_repr: (K,N)=(384,768)", conditioned_single_repr.shape)
        tokens = self.cond_tokens_with_cond_single(conditioned_single_repr) + tokens

        tokens = self.token_transformer(
            tokens,
            mask=mask,
            single_repr=conditioned_single_repr,
            pairwise_repr=conditioned_pairwise_repr,
            **kwargs,
        )

        # tokens = self.attended_token_norm(tokens)

        # atom decoder
        # print("tokens: (K,N)=(768,128)", tokens.shape)
        atom_decoder_input = self.tokens_to_atom_decoder_input_cond(tokens)

        atom_decoder_input = batch_repeat_interleave(atom_decoder_input, molecule_atom_lens)
        atom_decoder_input = pad_or_slice_to(
            atom_decoder_input, length=atom_feats_skip.shape[1], dim=1
        )

        atom_decoder_input = atom_decoder_input + atom_feats_skip

        atom_feats = self.atom_decoder(
            atom_decoder_input,
            mask=atom_mask,
            windowed_mask=windowed_mask,
            single_repr=atom_feats_cond,
            pairwise_repr=atompair_feats,
            use_optimized_evo=None,
        )
        
        # print("atom_feats: (K,N)=(128,3)", atom_feats.shape)
        atom_pos_update = self.atom_feat_to_atom_pos_update(atom_feats)

        return atom_pos_update


# elucidated diffusion model adapted for atom position diffusing
# from Karras et al.
# https://arxiv.org/abs/2206.00364


class ElucidatedAtomDiffusion(Module):
    """An ElucidatedAtomDiffusion module."""

    @typecheck
    def __init__(
        self,
        net: DiffusionModule,
        *,
        num_sample_steps=200,  # number of sampling steps
        sigma_min=0.002,  # min noise level
        sigma_max=80,  # max noise level
        sigma_data=16.0,  # standard deviation of data distribution
        rho=7,  # controls the sampling schedule
        P_mean=-1.2,  # mean of log-normal distribution from which noise is drawn for training
        P_std=1.5,  # standard deviation of log-normal distribution from which noise is drawn for training
        S_churn=80,  # parameters for stochastic sampling - depends on dataset, Table 5 in paper
        S_tmin=0.05,
        S_tmax=50,
        S_noise=1.003,
        step_scale=1.5,
        diffusion_num_augmentations=48,
        diffusion_chunk_size=4,
        stochastic_frame_average=False,
        augment_during_sampling=True,
        atom_permutation_alignment_kwargs: dict = dict(
            run_checker=False,
            eps=1e-8,
        ),
        centre_random_augmentation_kwargs: dict = dict(),
        karras_formulation=True,  # use the original EDM formulation from Karras et al. Table 1 in https://arxiv.org/abs/2206.00364 - differences are that the noise and sampling schedules are scaled by sigma data, as well as loss weight adds the sigma data instead of multiply in denominator
        atom_permutation_alignment=True,
        verbose=False,
    ):
        super().__init__()

        self.verbose = verbose
        self.net = net

        # parameters

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.rho = rho

        self.P_mean = P_mean
        self.P_std = P_std

        self.num_sample_steps = num_sample_steps  # otherwise known as N in the paper
        self.step_scale = step_scale

        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

        # augmentation

        self.diffusion_num_augmentations = diffusion_num_augmentations
        self.diffusion_chunk_size = diffusion_chunk_size
        self.augment_during_sampling = augment_during_sampling

        self.centre_random_augmenter = CentreRandomAugmentation(
            **centre_random_augmentation_kwargs
        )

        # stochastic frame averaging
        # https://arxiv.org/abs/2305.05577

        self.stochastic_frame_average = stochastic_frame_average

        if stochastic_frame_average:
            self.frame_average = FrameAverage(
                dim=3, stochastic=True, return_stochastic_as_augmented_pos=True
            )

        # permutation alignment

        self.atom_permutation_alignment = None

        if atom_permutation_alignment:
            self.atom_permutation_alignment = AtomPermutationAlignment(
                **atom_permutation_alignment_kwargs,
            )

        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

        # whether to use original karras formulation or not

        self.karras_formulation = karras_formulation

    @property
    def device(self):
        """Return the device of the module.

        :return: The device of the module.
        """
        return next(self.net.parameters()).device

    @property
    def dtype(self):
        """Return the dtype of the module.

        :return: The dtype of the module.
        """
        return next(self.net.parameters()).dtype

    # derived preconditioning params - Table 1

    def c_skip(self, sigma):
        """Return the c_skip value.

        :param sigma: The sigma value.
        :return: The c_skip value.
        """
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        """Return the c_out value.

        :param sigma: The sigma value.
        :return: The c_out value.
        """
        return sigma * self.sigma_data * (self.sigma_data**2 + sigma**2) ** -0.5

    def c_in(self, sigma):
        """Return the c_in value.

        :param sigma: The sigma value.
        :return: The c_in value.
        """
        return 1 * (sigma**2 + self.sigma_data**2) ** -0.5

    def c_noise(self, sigma):
        """Return the c_noise value.

        :param sigma: The sigma value.
        :return: The c_noise value.
        """
        return log(sigma) * 0.25

    # preconditioned network output

    @typecheck
    def preconditioned_network_forward(
        self,
        noised_atom_pos: Float["b m 3"],  # type: ignore
        sigma: Float[" b"] | Float[" "] | float,  # type: ignore
        network_condition_kwargs: dict,
        clamp=False,
        **kwargs,
    ):
        """Run a network forward pass, with the preconditioned inputs.

        :param noised_atom_pos: The noised atom position tensor.
        :param sigma: The sigma value.
        :param network_condition_kwargs: The network condition keyword arguments.
        :param clamp: Whether to clamp the output.
        :param kwargs: Additional keyword arguments for the attention computation.
        :return: The output tensor.
        """
        batch, dtype, device = (
            noised_atom_pos.shape[0],
            noised_atom_pos.dtype,
            noised_atom_pos.device,
        )

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, dtype=dtype, device=device)

        padded_sigma = rearrange(sigma, "b -> b 1 1")

        net_out = self.net(
            self.c_in(padded_sigma) * noised_atom_pos,
            times=sigma,
            **network_condition_kwargs,
            **kwargs,
        )

        out = self.c_skip(padded_sigma) * noised_atom_pos + self.c_out(padded_sigma) * net_out

        if clamp:
            out = out.clamp(-1.0, 1.0)

        return out

    # sampling

    # sample schedule
    # equation (7) in the paper

    def sample_schedule(self, num_sample_steps=None):
        """Return the schedule of sigmas for sampling. Algorithm (7) in the paper.

        :param num_sample_steps: The number of sample steps.
        :return: The schedule of sigmas for sampling.
        """
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        N = num_sample_steps
        inv_rho = 1 / self.rho

        # NOTE: this differs in notation from the paper slightly
        steps = torch.arange(num_sample_steps, device=self.device, dtype=self.dtype)
        sigmas = (
            self.sigma_max**inv_rho
            + steps / (N - 1) * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

        sigmas = F.pad(sigmas, (0, 1), value=0.0)  # last step is sigma value of 0.

        return sigmas * self.sigma_data

    @torch.no_grad()
    def sample(
        self,
        atom_mask: Bool["b m"] | None = None,  # type: ignore
        num_sample_steps=None,
        clamp=False,
        use_tqdm_pbar=True,
        tqdm_pbar_title="Sampling time step",
        return_all_timesteps=False,
        use_optimized_evo=None,
        umeyama_correction=True,
        **network_condition_kwargs,
    ) -> Float["b m 3"] | Float["ts b m 3"]:  # type: ignore
        """Sample clean atom positions.

        :param atom_mask: The atom mask tensor.
        :param num_sample_steps: The number of sample steps.
        :param clamp: Whether to clamp the output.
        :param network_condition_kwargs: The network condition keyword arguments.
        :param use_tqdm_pbar: Whether to use tqdm progress bar.
        :param tqdm_pbar_title: The tqdm progress bar title.
        :param return_all_timesteps: Whether to return all timesteps.
        :param use_optimized_evo: Whether to use an optimized Evoformer kernel.
        :param umeyama_correction: Whether to employ Kabsch-Umeyama correction to align denoised
            structures to their previous (input) timestep versions to get consistent sampling
            trajectories.
        :return: The clean atom positions.
        """
        dtype = self.dtype

        step_scale, num_sample_steps = self.step_scale, default(
            num_sample_steps, self.num_sample_steps
        )

        shape = (*atom_mask.shape, 3)

        network_condition_kwargs.update(atom_mask=atom_mask)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma

        sigmas = self.sample_schedule(num_sample_steps)

        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / num_sample_steps, sqrt(2) - 1),
            0.0,
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        # atom position is noise at the beginning

        init_sigma = sigmas[0]

        atom_pos = init_sigma * torch.randn(shape, dtype=dtype, device=self.device)

        # gradually denoise

        maybe_tqdm_wrapper = tqdm if use_tqdm_pbar else identity

        maybe_augment_fn = (
            self.centre_random_augmenter if self.augment_during_sampling else identity
        )

        all_atom_pos = [atom_pos]

        for sigma, sigma_next, gamma in maybe_tqdm_wrapper(
            sigmas_and_gammas, desc=tqdm_pbar_title
        ):
            sigma, sigma_next, gamma = tuple(t.item() for t in (sigma, sigma_next, gamma))

            atom_pos = maybe_augment_fn(atom_pos.float()).type(dtype)

            eps = self.S_noise * torch.randn(
                shape, dtype=dtype, device=self.device
            )  # stochastic sampling

            sigma_hat = sigma + gamma * sigma
            atom_pos_hat = atom_pos + sqrt(sigma_hat**2 - sigma**2) * eps

            model_output = self.preconditioned_network_forward(
                atom_pos_hat,
                sigma_hat,
                clamp=clamp,
                network_condition_kwargs=network_condition_kwargs,
                use_optimized_evo=use_optimized_evo,
            )

            if umeyama_correction:
                try:
                    model_output = weighted_rigid_align(
                        # NOTE: `weighted_rigid_align` returns the input-aligned model output
                        atom_pos_hat,
                        model_output,
                        mask=atom_mask,
                    )
                except Exception as e:
                    logger.warning(f"Umeyama correction failed with error {e}. Skipping...")

            # NOTE: `atom_pos_hat` here matches what Protenix (and now AF3's source code) also reports using
            denoised_over_sigma = (atom_pos_hat - model_output) / sigma_hat

            atom_pos_next = (
                atom_pos_hat + (sigma_next - sigma_hat) * denoised_over_sigma * step_scale
            )

            # second order correction, if not the last timestep

            if self.karras_formulation and sigma_next != 0:
                model_output_next = self.preconditioned_network_forward(
                    atom_pos_next,
                    sigma_next,
                    clamp=clamp,
                    network_condition_kwargs=network_condition_kwargs,
                    use_optimized_evo=use_optimized_evo,
                )
                denoised_prime_over_sigma = (atom_pos_next - model_output_next) / sigma_next
                atom_pos_next = (
                    atom_pos_hat
                    + 0.5
                    * (sigma_next - sigma_hat)
                    * (denoised_over_sigma + denoised_prime_over_sigma)
                    * step_scale
                )

            atom_pos = atom_pos_next

            all_atom_pos.append(atom_pos)

        # if returning atom positions across all timesteps for visualization
        # then stack the `all_atom_pos`

        if return_all_timesteps:
            atom_pos = torch.stack(all_atom_pos)

        if clamp:
            atom_pos = atom_pos.clamp(-1.0, 1.0)

        return atom_pos

    # training

    @typecheck
    def karras_loss_weight(self, sigma: Float["b 1 1"]) -> Float["b 1 1"]:  # type: ignore
        """Return the loss weight for training.

        :param sigma: The sigma value.
        :return: The loss weight for training.
        """
        return (sigma**2 + self.sigma_data**2) * (sigma * self.sigma_data) ** -2

    @typecheck
    def loss_weight(self, sigma: Float["b 1 1"]) -> Float["b 1 1"]:  # type: ignore
        """Return the loss weight for training. For some reason, in paper they add instead of
        multiply as in original paper.

        :param sigma: The sigma value.
        :return: The loss weight for training.
        """
        return (sigma**2 + self.sigma_data**2) * (sigma + self.sigma_data) ** -2

    def noise_distribution(self, batch_size):
        """Sample Gaussian-distributed noise.

        :param batch_size: The batch size.
        :return: Sampled Gaussian noise.
        """
        return (
            self.P_mean + self.P_std * torch.randn((batch_size,), device=self.device)
        ).exp() * self.sigma_data

    @typecheck
    def forward(
        self,
        atom_pos_ground_truth: Float["b m 3"],  # type: ignore
        atom_mask: Bool["b m"],  # type: ignore
        atom_feats: Float["b m da"],  # type: ignore
        atompair_feats: Float["b m m dap"] | Float["b nw w (w*2) dap"],  # type: ignore
        mask: Bool["b n"],  # type: ignore
        single_trunk_repr: Float["b n dst"],  # type: ignore
        single_inputs_repr: Float["b n dsi"],  # type: ignore
        pairwise_trunk: Float["b n n dpt"],  # type: ignore
        pairwise_rel_pos_feats: Float["b n n dpr"],  # type: ignore
        molecule_atom_lens: Int["b n"],  # type: ignore
        missing_atom_mask: Bool["b m"] | None = None,  # type: ignore
        atom_parent_ids: Int["b m"] | None = None,  # type: ignore
        is_molecule_types: Bool[f"b n {IS_MOLECULE_TYPES}"] | None = None,  # type: ignore
        additional_molecule_feats: Int[f"b n {ADDITIONAL_MOLECULE_FEATS}"] | None = None,  # type: ignore
        nucleotide_loss_weight=5.0,
        ligand_loss_weight=10.0,
        single_structure_input=False,
        use_optimized_evo=None,
        verbose=None,
        filepath: List[str] | Tuple[str, ...] | None = None,
        molecule_atom_perms: List[List[List[int]]] | None = None,
    ) -> Tuple[
        Float["ba m 3"],  # type: ignore
        Float["ba m 3"],  # type: ignore
        Float["ba m"],  # type: ignore
        Float["ba 1 1"],  # type: ignore
    ]:
        """Perform the forward pass.

        :param atom_pos_ground_truth: The ground truth atom position tensor.
        :param atom_mask: The atom mask tensor.
        :param atom_feats: The atom features tensor.
        :param atompair_feats: The atom pair features tensor.
        :param mask: The mask tensor.
        :param single_trunk_repr: The single trunk representation tensor.
        :param single_inputs_repr: The single inputs representation tensor.
        :param pairwise_trunk: The pairwise trunk tensor.
        :param pairwise_rel_pos_feats: The pairwise relative position features tensor.
        :param molecule_atom_lens: The molecule atom lengths tensor.
        :param token_bonds: The token bonds tensor.
        :param molecule_atom_indices: The molecule atom indices tensor.
        :param missing_atom_mask: The missing atom mask tensor.
        :param atom_parent_ids: The atom parent IDs tensor.
        :param is_molecule_types: The molecule types tensor.
        :param additional_molecule_feats: The additional molecule features tensor.
        :param add_smooth_lddt_loss: Whether to add the smooth lddt loss.
        :param add_bond_loss: Whether to add the bond loss.
        :param nucleotide_loss_weight: The nucleotide loss weight.
        :param ligand_loss_weight: The ligand loss weight.
        :param single_structure_input: Whether to the input(s) represent a single structure.
        :param use_optimized_evo: Whether to use an optimized Evoformer kernel.
        :param verbose: Whether to be verbose.
        :param filepath: The input filepath(s).
        :param molecule_atom_perms: The molecule atom permutations.
        :return: The denoised atom positions, augmented atom positions, alignment weights, and loss
            weights.
        """
        verbose = default(verbose, self.verbose)

        dtype = atom_pos_ground_truth.dtype
        batch_size = atom_pos_ground_truth.shape[0]

        # augmentations

        with torch.no_grad():
            num_augs = self.diffusion_num_augmentations + int(self.stochastic_frame_average)
            diffusion_chunk_size = default(self.diffusion_chunk_size, num_augs)
            aug_atom_pos = repeat(atom_pos_ground_truth, "b ... -> (b a) ...", a=num_augs)

            # center the ground truth coordinates while accounting for masking

            num = reduce(aug_atom_pos * atom_mask[..., None], "b n c -> b c", "sum")
            den = reduce(atom_mask.float(), "b n -> b", "sum")
            aug_atom_pos_mean = num[..., None, :] / den[..., None, None].clamp(min=1.0)

            aug_atom_pos = (aug_atom_pos - aug_atom_pos_mean) * atom_mask[..., None]

            # handle stochastic frame averaging

            if self.stochastic_frame_average:
                if verbose:
                    logger.info("Applying stochastic frame averaging...")

                fa_atom_pos, aug_atom_pos = aug_atom_pos[:1], aug_atom_pos[1:]

                fa_atom_pos = self.frame_average(
                    fa_atom_pos.float(), frame_average_mask=atom_mask
                ).type(dtype)

                fa_atom_pos = fa_atom_pos * atom_mask[..., None]

            # normal random augmentations, 48 times in the AF3 paper

            if verbose:
                logger.info("Applying random augmentations...")

            aug_atom_pos = self.centre_random_augmenter(aug_atom_pos.float(), mask=atom_mask).type(
                dtype
            )

            # concat back the stochastic frame averaged position

            if self.stochastic_frame_average:
                aug_atom_pos = torch.cat((fa_atom_pos, aug_atom_pos), dim=0)

        # diffusion loss

        sigmas = self.noise_distribution(batch_size * num_augs).type(dtype)
        padded_sigmas = rearrange(sigmas, "b -> b 1 1")

        noise = torch.randn_like(aug_atom_pos) * padded_sigmas

        # regular loss weight as defined in EDM paper

        loss_weight_fn = self.karras_loss_weight if self.karras_formulation else self.loss_weight
        loss_weights = loss_weight_fn(sigmas)

        if verbose:
            logger.info("Running preconditioned network forward pass within EDM")

        denoised_atom_pos = []
        num_aug_chunks = num_augs // diffusion_chunk_size + (num_augs % diffusion_chunk_size != 0)
        for i in range(num_aug_chunks):
            # NOTE: `alphas` are `1.0` in the AF3 paper
            noised_atom_pos_i = (aug_atom_pos + noise)[
                ..., i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size, :, :
            ]
            sigmas_i = sigmas[..., i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size]

            diffusion_chunk_size_ = len(sigmas_i)
            mask_i = mask.expand(diffusion_chunk_size // batch_size, -1)[
                :diffusion_chunk_size_, ...
            ]

            denoised_atom_pos_i = self.preconditioned_network_forward(
                noised_atom_pos_i,
                sigmas_i,
                network_condition_kwargs=dict(
                    atom_feats=atom_feats,
                    atom_mask=atom_mask,
                    missing_atom_mask=missing_atom_mask,
                    atompair_feats=atompair_feats,
                    atom_parent_ids=atom_parent_ids,
                    mask=mask_i,
                    single_trunk_repr=single_trunk_repr,
                    single_inputs_repr=single_inputs_repr,
                    pairwise_trunk=pairwise_trunk,
                    pairwise_rel_pos_feats=pairwise_rel_pos_feats,
                    molecule_atom_lens=molecule_atom_lens,
                    use_optimized_evo=use_optimized_evo,
                ),
            )
            denoised_atom_pos.append(denoised_atom_pos_i)
        denoised_atom_pos = torch.cat(denoised_atom_pos, dim=-3)

        # section 3.7.1 equation 2 - weighted rigid aligned ground truth

        amp_context = (
            torch.autocast(device_type="cuda", enabled=False, cache_enabled=False)
            if torch.cuda.is_available()
            else nullcontext()
        )

        with torch.no_grad(), amp_context:
            if verbose:
                logger.info("Calculating weighted rigid aligned ground truth within EDM")

            align_weights = calculate_weighted_rigid_align_weights(
                atom_pos=denoised_atom_pos,
                molecule_atom_lens=molecule_atom_lens,
                is_molecule_types=is_molecule_types,
                nucleotide_loss_weight=nucleotide_loss_weight,
                ligand_loss_weight=ligand_loss_weight,
            )

            aug_atom_pos_aligned = weighted_rigid_align(
                pred_coords=denoised_atom_pos.float(),
                true_coords=aug_atom_pos.float(),
                weights=align_weights.float(),
                mask=atom_mask,
            ).type(dtype)

        # section 4.2 - multi-chain permutation alignment

        # NOTE: since the ground-truth chains are cropped during diffusion training,
        # we do not perform multi-chain permutation alignment for diffusion training

        # section 4.2 - atom permutation alignment

        # NOTE: to simplify the implementation, we optimally permute the atoms in the
        # predicted structure to match the ground truth, in constrast to matching the
        # ground truth atoms to the predicted atoms as done in the AF3 paper

        if (
            exists(self.atom_permutation_alignment)
            and exists(additional_molecule_feats)
            and exists(molecule_atom_perms)
            and single_structure_input
        ):
            if verbose:
                logger.info("Running atom permutation alignment within EDM")

            try:
                denoised_atom_pos, _ = self.atom_permutation_alignment(
                    pred_coords=denoised_atom_pos,
                    true_coords=atom_pos_ground_truth,  # NOTE: we intentionally use the unaugmented ground truth here
                    additional_molecule_feats=additional_molecule_feats,
                    molecule_atom_lens=molecule_atom_lens,
                    molecule_atom_perms=molecule_atom_perms,
                    mask=atom_mask,
                    permute_labels=False,
                )
            except Exception as e:
                # NOTE: For many (random) unit test inputs, permutation alignment can be unstable
                logger.warning(
                    f"Skipping atom permutation alignment {f'for {filepath}' if exists(filepath) else ''} due to: {e}"
                )

        return denoised_atom_pos, aug_atom_pos_aligned, align_weights, loss_weights


# modules


class MultiChainPermutationAlignment(Module):
    """Section 4.2 of the AlphaFold 3 Supplement."""

    @typecheck
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()

    @staticmethod
    @typecheck
    def split_ground_truth_labels(gt_features: Dict[str, Tensor]) -> List[Dict[str, Tensor]]:
        """Split ground truth features according to chains.

        Adapted from:
        https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/multi_chain_permutation.py

        :param gt_features: A dictionary within a PyTorch Dataset iteration, which is returned by
            the upstream DataLoader.iter() method. In the DataLoader pipeline, all tensors
            belonging to all the ground truth chains are concatenated. This function is needed to
            1) detect the number of chains, i.e., unique(asym_id) and 2) split the concatenated
            tensors back to individual ones that correspond to individual asym_ids.
        :return: A list of feature dictionaries with only necessary ground truth features required
            to finish multi-chain permutation. E.g., it will be a list of 5 elements if there are 5
            chains in total.
        """
        _, asym_id_counts = torch.unique_consecutive(
            gt_features["asym_id"], return_counts=True, dim=-1
        )
        n_res = gt_features["asym_id"].shape[-1]

        def split_dim(shape):
            """Return the dimension index where the size is n_res."""
            return next(iter(i for i, size in enumerate(shape) if size == n_res), None)

        labels = list(
            map(
                dict,
                zip(
                    *[
                        [
                            (k, v)
                            for v in torch.split(
                                v_all, asym_id_counts.tolist(), dim=split_dim(v_all.shape)
                            )
                        ]
                        for k, v_all in gt_features.items()
                        if n_res in v_all.shape
                    ]
                ),
            )
        )
        return labels

    @staticmethod
    @typecheck
    def get_per_asym_token_index(features: Dict[str, Tensor], padding_value: int = -1) -> Dict[int, Int["b ..."]]:  # type: ignore
        """A function that retrieves a mapping denoting which token belongs to which `asym_id`.

        Adapted from:
        https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/multi_chain_permutation.py

        :param features: A dictionary that contains input features after cropping.
        :return: A dictionary that records which region of the sequence belongs to which `asym_id`.
        """
        batch_size = features["token_index"].shape[0]

        unique_asym_ids = list(
            dict.fromkeys(
                [i for i in torch.unique_consecutive(features["asym_id"]) if i != padding_value]
            )
        )
        per_asym_token_index = {}
        for cur_asym_id in unique_asym_ids:
            asym_mask = (features["asym_id"] == cur_asym_id).bool()
            per_asym_token_index[int(cur_asym_id)] = rearrange(
                features["token_index"][asym_mask], "(b a) -> b a", b=batch_size
            )

        return per_asym_token_index

    @staticmethod
    @typecheck
    def get_entity_to_asym_list(features: Dict[str, Tensor]) -> Dict[int, Tensor]:
        """Generate a dictionary mapping unique entity IDs to lists of unique asymmetry IDs
        (asym_id) for each entity.

        Adapted from:
        https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/multi_chain_permutation.py

        :param features: A dictionary containing data features, including `entity_id` and `asym_id` tensors.
        :return: A dictionary where keys are unique entity IDs, and values are tensors of unique asymmetry IDs
            associated with each entity.
        """
        entity_to_asym_list = {}
        unique_entity_ids = list(
            dict.fromkeys(torch.unique_consecutive(features["entity_id"]).tolist())
        )

        # First pass: Collect all unique `cur_asym_id` values across all entities
        for cur_ent_id in unique_entity_ids:
            ent_mask = features["entity_id"] == cur_ent_id
            cur_asym_id = torch.unique_consecutive(features["asym_id"][ent_mask])
            entity_to_asym_list[int(cur_ent_id)] = cur_asym_id

        return entity_to_asym_list

    @typecheck
    def get_least_asym_entity_or_longest_length(
        self, batch: Dict[str, Tensor], input_asym_id: List[int], padding_value: int = -1
    ) -> Tuple[Tensor, List[Tensor]]:
        """Check how many subunit(s) one sequence has. Select the subunit that is less common,
        e.g., if the protein was AABBB then select one of the As as an anchor.

        If there is a tie, e.g. AABB, first check which sequence is the longest,
        then choose one of the corresponding subunits as an anchor.

        Adapted from:
        https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/multi_chain_permutation.py

        :param batch: In this function, `batch` is the full ground truth features.
        :param input_asym_id: A list of `asym_ids` that are in the cropped input features.
        :param padding_value: The padding value used in the input features.
        :return: Selected ground truth `asym_ids` and a list of
            integer tensors denoting of all possible pred anchor candidates.
        """
        entity_to_asym_list = self.get_entity_to_asym_list(features=batch)
        unique_entity_ids = list(
            dict.fromkeys(
                [
                    i.item()
                    for i in torch.unique_consecutive(batch["entity_id"])
                    if i != padding_value
                ]
            )
        )

        entity_asym_count = {}
        entity_length = {}

        # NOTE: we must rely on the chain ID ordering of `input_asym_id` for correctness
        all_asym_ids = input_asym_id.copy()

        for entity_id in unique_entity_ids:
            asym_ids = torch.unique_consecutive(batch["asym_id"][batch["entity_id"] == entity_id])

            # Make sure some asym IDs associated with ground truth entity ID exist in cropped prediction
            asym_ids_in_pred = [a for a in asym_ids if a in input_asym_id]
            if not asym_ids_in_pred:
                continue

            entity_asym_count[int(entity_id)] = len(asym_ids)

            # Calculate entity length
            entity_mask = batch["entity_id"] == entity_id
            entity_length[int(entity_id)] = entity_mask.sum(-1).mode().values.item()

        min_asym_count = min(entity_asym_count.values())
        least_asym_entities = [
            entity for entity, count in entity_asym_count.items() if count == min_asym_count
        ]

        # If multiple entities have the least asym_id count, return those with the longest length
        if len(least_asym_entities) > 1:
            max_length = max([entity_length[entity] for entity in least_asym_entities])
            least_asym_entities = [
                entity for entity in least_asym_entities if entity_length[entity] == max_length
            ]

        # If there are still multiple entities, return a random one
        if len(least_asym_entities) > 1:
            least_asym_entities = [random.choice(least_asym_entities)]  # nosec

        assert (
            len(least_asym_entities) == 1
        ), "There should be only one entity with the least `asym_id` count."
        least_asym_entities = least_asym_entities[0]

        anchor_gt_asym_id = random.choice(entity_to_asym_list[least_asym_entities])  # nosec
        anchor_pred_asym_ids = [
            asym_id
            for asym_id in entity_to_asym_list[least_asym_entities]
            if asym_id in input_asym_id
        ]

        # Since the entity ID to asym ID mapping is many-to-many, we need to select only
        # prediction asym IDs with equal length w.r.t. the sampled ground truth asym ID
        anchor_gt_asym_id_length = (
            (batch["asym_id"] == anchor_gt_asym_id).sum(-1).mode().values.item()
        )
        anchor_pred_asym_ids = [
            asym_id
            for asym_id in anchor_pred_asym_ids
            if (batch["asym_id"] == asym_id).sum(-1).mode().values.item()
            == anchor_gt_asym_id_length
        ]

        # Remap `asym_id` values to handle for ground-truth asym IDs not in alphabetical order,
        # but leave the prediction asym IDs as is since they are used for masking
        remap_dict = {old_id: new_id for new_id, old_id in enumerate(all_asym_ids)}

        remapped_anchor_gt_asym_id = torch.tensor([remap_dict[anchor_gt_asym_id.item()]])

        return remapped_anchor_gt_asym_id, anchor_pred_asym_ids

    @staticmethod
    @typecheck
    def calculate_input_mask(
        true_masks: List[Int["b ..."]],  # type: ignore
        anchor_gt_idx: int,
        asym_mask: Bool["b n"],  # type: ignore
        pred_mask: Float["b n"],  # type: ignore
    ) -> Bool["b a"]:  # type: ignore
        """Calculate an input mask for downstream optimal transformation computation.

        :param true_masks: A list of masks from the ground truth chains. E.g., it will be a length
            of 5 if there are 5 chains in ground truth structure.
        :param anchor_gt_idx: A tensor with one integer in it (i.e., the index of selected ground
            truth anchor).
        :param asym_mask: A boolean tensor with which to mask out other elements in a tensor if
            they do not belong to a specific asym ID.
        :param pred_mask: A boolean tensor corresponding to the mask with which to mask the
            predicted features.
        :return: A boolean mask.
        """
        batch_size = pred_mask.shape[0]
        anchor_pred_mask = rearrange(
            pred_mask[asym_mask],
            "(b a) -> b a",
            b=batch_size,
        )
        anchor_true_mask = true_masks[anchor_gt_idx]
        input_mask = (anchor_true_mask * anchor_pred_mask).bool()
        return input_mask

    @typecheck
    def calculate_optimal_transform(
        self,
        true_poses: List[Float["b ... 3"]],  # type: ignore
        anchor_gt_idx: int,
        true_masks: List[Int["b ..."]],  # type: ignore
        pred_mask: Float["b n"],  # type: ignore
        asym_mask: Bool["b n"],  # type: ignore
        pred_pos: Float["b n 3"],  # type: ignore
    ) -> Tuple[Float["b 3 3"], Float["b 1 3"]]:  # type: ignore
        """Take the selected anchor ground truth token center atom positions and the selected
        predicted anchor token center atom position and then calculate the optimal rotation matrix
        to align the ground-truth anchor and predicted anchor.

        Process:
        1) Select an anchor chain from ground truth, denoted by anchor_gt_idx, and an anchor chain from the predicted structure.
            Both anchor_gt and anchor_pred have exactly the same sequence.
        2) Obtain the token center atom positions corresponding to the selected anchor_gt,
            done be slicing the true_pose according to anchor_gt_token
        3) Calculate the optimal transformation that can best align the token center atoms of anchor_pred to those of anchor_gt
            via the Kabsch algorithm (https://en.wikipedia.org/wiki/Kabsch_algorithm).

        Adapted from:
        https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/multi_chain_permutation.py

        :param true_poses: A list of tensors, corresponding to the token center atom positions of the ground truth structure.
            E.g., If there are 5 chains, this list will have a length of 5.
        :param anchor_gt_idx: A tensor with one integer in it (i.e., the index of selected ground truth anchor).
        :param true_masks: list of masks from the ground truth chains. E.g., it will be a length of 5 if there are
            5 chains in ground truth structure.
        :param pred_mask: A boolean tensor corresponding to the mask with which to mask the predicted features.
        :param asym_mask: A boolean tensor with which to mask out other elements in a tensor if they do not belong
            to a specific asym ID.
        :param pred_pos: A tensor of predicted token center atom positions.
        :return: A rotation matrix that records the optimal rotation that will best align the selected anchor prediction to the
            selected anchor truth as well as a matrix that records how the atoms should be shifted after applying `r`.
            N.b., Optimal alignment requires 1) a rotation and 2) a shift of the positions.
        """
        dtype = pred_pos.dtype
        batch_size = pred_pos.shape[0]

        input_mask = self.calculate_input_mask(
            true_masks=true_masks,
            anchor_gt_idx=anchor_gt_idx,
            asym_mask=asym_mask,
            pred_mask=pred_mask,
        )
        anchor_true_pos = true_poses[anchor_gt_idx]
        anchor_pred_pos = rearrange(
            pred_pos[asym_mask],
            "(b a) ... -> b a ...",
            b=batch_size,
        )

        amp_context = (
            torch.autocast(device_type="cuda", enabled=False, cache_enabled=False)
            if torch.cuda.is_available()
            else nullcontext()
        )

        with torch.no_grad(), amp_context:
            _, r, x = weighted_rigid_align(
                pred_coords=anchor_pred_pos.float(),
                true_coords=anchor_true_pos.float(),
                mask=input_mask,
                return_transforms=True,
            )

        return r.type(dtype), x.type(dtype)

    @staticmethod
    @typecheck
    def apply_transform(pose: Float["b a 3"], r: Float["b 3 3"], x: Float["b 1 3"]) -> Float["b a 3"]:  # type: ignore
        """Apply the optimal transformation to the predicted token center atom positions.

        :param pose: A tensor of predicted token center atom positions.
        :param r: A rotation matrix that records the optimal rotation that will best align the selected anchor prediction to the
            selected anchor truth.
        :param x: A matrix that records how the atoms should be shifted after applying `r`.
        :return: A tensor of predicted token center atom positions after applying the optimal transformation.
        """
        aligned_pose = einsum(r, pose - pose.mean(1, keepdim=True), "b i j, b n j -> b n i") + x
        aligned_pose.detach_()
        return aligned_pose

    @typecheck
    def greedy_align(
        self,
        batch: Dict[str, Tensor],
        entity_to_asym_list: Dict[int, Tensor],
        pred_pos: Float["b n 3"],  # type: ignore
        pred_mask: Float["b n"],  # type: ignore
        true_poses: List[Float["b ... 3"]],  # type: ignore
        true_masks: List[Int["b ..."]],  # type: ignore
        padding_value: int = -1,
    ) -> List[Tuple[int, int]]:
        """
        Implement Algorithm 4 in the Supplementary Information of the AlphaFold-Multimer paper:
            Evans, R et al., 2022 Protein complex prediction with AlphaFold-Multimer,
            bioRxiv 2021.10.04.463034; doi: https://doi.org/10.1101/2021.10.04.463034

        NOTE: The tuples in the returned list are zero-indexed.

        Adapted from:
        https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/multi_chain_permutation.py

        :param batch: A dictionary of ground truth features.
        :param entity_to_asym_list: A dictionary recording which asym ID(s) belong to which entity ID.
        :param pred_pos: Predicted positions of token center atoms from the results of model.forward().
        :param pred_mask: A boolean tensor that masks `pred_pos`.
        :param true_poses: A list of tensors, corresponding to the token center atom positions of the ground truth structure.
            E.g., if there are 5 chains, this list will have a length of 5.
        :param true_masks: A list of tensors, corresponding to the masks of the token center atom positions of the ground truth structure.
            E.g., if there are 5 chains, this list will have a length of 5.
        :param padding_value: The padding value used in the input features.
        :return: A list of tuple(int, int) that provides instructions for how the ground truth chains should be permuted.
            E.g., if 3 chains in the input structure have the same sequences, an example return value would be:
            `[(0, 2), (1, 1), (2, 0)]`, meaning the first chain in the predicted structure should be aligned
            to the third chain in the ground truth and the second chain in the predicted structure is fine
            to stay with the second chain in the ground truth.
        """
        batch_size = pred_pos.shape[0]

        used = [
            # NOTE: This is a list the keeps a record of whether a ground truth chain has been used.
            False
            for _ in range(len(true_poses))
        ]
        alignments = []

        unique_asym_ids = list(
            dict.fromkeys(
                [i for i in torch.unique_consecutive(batch["asym_id"]) if i != padding_value]
            )
        )

        # Remap `asym_id` values to handle for ground-truth asym IDs not in alphabetical order yet listed first,
        # since the `labels` extracted for each chain are extracted in the order of the input batch's chain IDs
        remap_dict = {old_id.item(): new_id for new_id, old_id in enumerate(unique_asym_ids)}

        for cur_asym_id in unique_asym_ids:
            i = remap_dict[int(cur_asym_id)]

            asym_mask = batch["asym_id"] == cur_asym_id
            cur_entity_ids = rearrange(
                batch["entity_id"][asym_mask],
                "(b a) -> b a",
                b=batch_size,
            )

            # NOTE: Here, we assume there can be multiple unique entity IDs associated
            # with a given asym ID. This is a valid assumption when the original batch
            # contains a single unique structure that has one or more chains spread
            # across multiple entities (e.g., in the case of ligands residing in a
            # protein-majority chain).

            best_rmsd = torch.inf
            best_idx = None

            unique_cur_entity_ids = torch.unique_consecutive(cur_entity_ids, dim=-1).unbind(dim=-1)

            for batch_cur_entity_id in unique_cur_entity_ids:
                cur_pred_pos = rearrange(
                    pred_pos[asym_mask],
                    "(b a) ... -> b a ...",
                    b=batch_size,
                )
                cur_pred_mask = rearrange(
                    pred_mask[asym_mask],
                    "(b a) -> b a",
                    b=batch_size,
                )

                # NOTE: Here, we assume there is only one unique entity ID per batch,
                # which is a valid assumption only when the original batch size is 1
                # (meaning only a single unique structure is represented in the batch).

                unique_cur_entity_id = torch.unique_consecutive(batch_cur_entity_id)
                assert (
                    len(unique_cur_entity_id) == 1
                ), "There should be only one unique entity ID per batch."
                cur_asym_list = entity_to_asym_list[int(unique_cur_entity_id)]

                for next_asym_id in cur_asym_list:
                    j = remap_dict[int(next_asym_id)]

                    if not used[j]:  # NOTE: This is a possible candidate.
                        cropped_pos = true_poses[j]
                        mask = true_masks[j]

                        rmsd = batch_compute_rmsd(
                            true_pos=cropped_pos.mean(1, keepdim=True),
                            pred_pos=cur_pred_pos.mean(1, keepdim=True),
                            mask=(
                                cur_pred_mask.any(-1, keepdim=True) * mask.any(-1, keepdim=True)
                            ),
                        ).mean()

                        if rmsd < best_rmsd:
                            # NOTE: We choose the permutation that minimizes the batch-wise
                            # average RMSD of the predicted token center atom centroid coordinates
                            # with respect to the ground truth token center atom centroid coordinates.
                            best_rmsd = rmsd
                            best_idx = j

            if exists(best_idx):
                # NOTE: E.g., for ligands within a protein-majority chain, we may have
                # multiple unique entity IDs associated with a given asym ID. In this case,
                # we need to ensure that we do not reuse a chain that has already been used
                # in the permutation alignment process.
                used[best_idx] = True
                alignments.append((i, best_idx))

        assert all(used), "All chains should be used in the permutation alignment process."
        return alignments

    @staticmethod
    @typecheck
    def pad_features(feature_tensor: Tensor, num_tokens_pad: int, pad_dim: int) -> Tensor:
        """Pad an input feature tensor. Padding values will be 0 and put behind the true feature
        values.

        Adapted from:
        https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/multi_chain_permutation.py

        :param feature_tensor: A feature tensor to pad.
        :param num_tokens_pad: The number of tokens to pad.
        :param pad_dim: Along which dimension of `feature_tensor` to pad.
        :return: A padded feature tensor.
        """
        pad_shape = list(feature_tensor.shape)
        pad_shape[pad_dim] = num_tokens_pad
        padding_tensor = feature_tensor.new_zeros(pad_shape, device=feature_tensor.device)
        return torch.concat((feature_tensor, padding_tensor), dim=pad_dim)

    @typecheck
    def merge_labels(
        self,
        labels: List[Dict[str, Tensor]],
        alignments: List[Tuple[int, int]],
        original_num_tokens: int,
        dimension_to_merge: int = 1,
    ) -> Dict[str, Tensor]:
        """Merge ground truth labels according to permutation results.

        Adapted from:
        https://github.com/dptech-corp/Uni-Fold/blob/b1c89a2cebd4e4ee4c47b4e443f92beeb9138fbb/unifold/losses/chain_align.py#L176C1-L176C1
        and
        https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/multi_chain_permutation.py

        :param labels: A list of original ground truth feats. E.g., if there are 5 chains,
            `labels` will have a length of 5.
        :param alignments: A list of tuples, each entry specifying the corresponding label of the asym ID.
        :param original_num_tokens: An integer corresponding to the number of tokens specified
            by one's (e.g., training-time) crop size.
        :param dimension_to_merge: The dimension along which to merge the labels.
        :return: A new dictionary of permuted ground truth features.
        """
        outs = {}
        for k in labels[0].keys():
            cur_out = {}
            for i, j in alignments:
                label = labels[j][k]
                cur_out[i] = label

            cur_out = [x[1] for x in cur_out.items()]
            if len(cur_out) > 0:
                new_v = torch.concat(cur_out, dim=dimension_to_merge)

                # Check whether padding is needed.
                if new_v.shape[dimension_to_merge] != original_num_tokens:
                    num_tokens_pad = original_num_tokens - new_v.shape[dimension_to_merge]
                    new_v = self.pad_features(new_v, num_tokens_pad, pad_dim=dimension_to_merge)

                outs[k] = new_v

        return outs

    @typecheck
    def compute_permutation_alignment(
        self,
        out: Dict[str, Tensor],
        features: Dict[str, Tensor],
        ground_truth: Dict[str, Tensor],
        padding_value: int = -1,
    ) -> List[Tuple[int, int]]:
        """A method that permutes chains in ground truth before calculating the loss because the
        mapping between the predicted and ground truth will become arbitrary. The model cannot be
        assumed to predict chains in the same order as the ground truth. Thus, this function picks
        the optimal permutation of predicted chains that best matches the ground truth, by
        minimising the RMSD (i.e., the best permutation of ground truth chains is selected based on
        which permutation has the lowest RMSD calculation).

        Details are described in Section 7.3 in the Supplementary of AlphaFold-Multimer paper:
        https://www.biorxiv.org/content/10.1101/2021.10.04.463034v2

        Adapted from:
        https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/multi_chain_permutation.py

        :param out: A dictionary of output tensors from model.forward().
        :param features: A dictionary of feature tensors that are used as input for model.forward().
        :param ground_truth: A list of dictionaries of features corresponding to chains in ground truth structure.
            E.g., it will be a length of 5 if there are 5 chains in ground truth structure.
        :param padding_value: The padding value used in the input features.
        :return: A list of tuple(int, int) that instructs how ground truth chains should be permutated.
        """
        num_tokens = features["token_index"].shape[-1]

        unique_asym_ids = [
            i for i in dict.fromkeys(features["asym_id"][0].tolist()) if i != padding_value
        ]
        is_monomer = len(unique_asym_ids) == 1

        per_asym_token_index = self.get_per_asym_token_index(
            features=features, padding_value=padding_value
        )

        if is_monomer:
            best_alignments = list(enumerate(range(len(per_asym_token_index))))
            return best_alignments

        best_rmsd = torch.inf
        best_alignments = None

        # 1. Choose the least ambiguous ground truth "anchor" chain.
        # For example, in an A3B2 complex an arbitrary B chain is chosen.
        # In the event of a tie e.g., A2B2 stoichiometry, the longest chain
        # is chosen, with the hope that in general the longer chains are
        # likely to have higher confidence predictions.

        # 2. Select the prediction anchor chain from the set of all prediction
        # chains with the same sequence as the ground truth anchor chain.

        anchor_gt_asym, anchor_pred_asym_ids = self.get_least_asym_entity_or_longest_length(
            batch=ground_truth,
            input_asym_id=unique_asym_ids,
        )
        entity_to_asym_list = self.get_entity_to_asym_list(features=ground_truth)
        labels = self.split_ground_truth_labels(gt_features=ground_truth)
        anchor_gt_idx = int(anchor_gt_asym)

        # 3. Optimally align the ground truth anchor chain to the prediction
        # anchor chain using a rigid alignment algorithm.

        pred_pos = out["pred_coords"]
        pred_mask = out["mask"].to(dtype=pred_pos.dtype)

        true_poses = [label["true_coords"] for label in labels]
        true_masks = [label["mask"].long() for label in labels]

        # Assignment Stage - Section 7.3.2 of the AlphaFold-Multimer Paper

        # 1. Greedily assign each of the predicted chains to their nearest
        # neighbour of the same sequence in the ground truth. These assignments
        # define the optimal permutation to apply to the ground truth chains.
        # Nearest neighbours are defined as the chains with the smallest distance
        # between the average of their token center atom coordinates.

        # 2. Repeat the above alignment and assignment stages for all valid choices
        # of the prediction anchor chain given the ground truth anchor chain.

        # 3. Finally, we pick the permutation that minimizes the RMSD between the
        # token center atom coordinate averages of the predicted and ground truth chains.

        for candidate_pred_anchor in anchor_pred_asym_ids:
            asym_mask = (features["asym_id"] == candidate_pred_anchor).bool()

            r, x = self.calculate_optimal_transform(
                true_poses=true_poses,
                anchor_gt_idx=anchor_gt_idx,
                true_masks=true_masks,
                pred_mask=pred_mask,
                asym_mask=asym_mask,
                pred_pos=pred_pos,
            )

            # Apply transforms.
            aligned_true_poses = [
                self.apply_transform(pose.to(r.dtype), r, x) for pose in true_poses
            ]

            alignments = self.greedy_align(
                batch=features,
                entity_to_asym_list=entity_to_asym_list,
                pred_pos=pred_pos,
                pred_mask=pred_mask,
                true_poses=aligned_true_poses,
                true_masks=true_masks,
            )

            merged_labels = self.merge_labels(
                labels=labels,
                alignments=alignments,
                original_num_tokens=num_tokens,
            )

            aligned_true_pos = self.apply_transform(merged_labels["true_coords"].to(r.dtype), r, x)

            rmsd = batch_compute_rmsd(
                true_pos=aligned_true_pos.mean(1, keepdim=True),
                pred_pos=pred_pos.mean(1, keepdim=True),
                mask=(
                    pred_mask.any(-1, keepdim=True) * merged_labels["mask"].any(-1, keepdim=True)
                ),
            ).mean()

            if rmsd < best_rmsd:
                # NOTE: We choose the permutation that minimizes the batch-wise
                # average RMSD of the predicted token center atom centroid coordinates
                # with respect to the ground truth token center atom centroid coordinates.
                best_rmsd = rmsd
                best_alignments = alignments

        # NOTE: The above algorithm naturally generalizes to both training and inference
        # contexts (i.e., with and without cropping) by, where applicable, pre-applying
        # cropping to the (ground truth) input coordinates and features.

        assert exists(best_alignments), "Best alignments must be found."
        return best_alignments

    @typecheck
    def forward(
        self,
        pred_coords: Float["b m 3"],  # type: ignore - predicted coordinates
        true_coords: Float["b m 3"],  # type: ignore - true coordinates
        molecule_atom_lens: Int["b n"],  # type: ignore - molecule atom lengths
        molecule_atom_indices: Int["b n"],  # type: ignore - molecule atom indices
        token_bonds: Bool["b n n"],  # type: ignore - token bonds
        additional_molecule_feats: Int[f"b n {ADDITIONAL_MOLECULE_FEATS}"] | None = None,  # type: ignore - additional molecule features
        is_molecule_types: Bool[f"b n {IS_MOLECULE_TYPES}"] | None = None,  # type: ignore - molecule types
        mask: Bool["b m"] | None = None,  # type: ignore - mask for variable lengths
        verbose: bool = False,
    ) -> Tuple[Float["b m 3"], Bool["b m"]]:  # type: ignore
        """Compute the multi-chain permutation alignment.

        NOTE: This function assumes that the ground truth features are batched yet only contain
        features for the same structure. This is the case after performing data augmentation
        with a batch size of 1 in the `MegaFold` module's forward pass. If the batched
        ground truth features represent multiple different structures, this function will not
        return correct results.

        NOTE: We use `torch.unique_consecutive` to ensure that the order of unique chain values
        is preserved. This is important for the `get_entity_to_asym_list` function, which
        relies on the order of unique entity IDs to be consistent with the order of unique
        asymmetric unit IDs for chain permutations.

        :param pred_coords: Predicted coordinates.
        :param true_coords: True coordinates.
        :param molecule_atom_lens: The molecule atom lengths.
        :param molecule_atom_indices: The molecule atom indices.
        :param token_bonds: The token bonds.
        :param is_molecule_types: Molecule type of each atom.
        :param mask: The mask for variable lengths.
        :param verbose: Whether to print verbose logs.
        :return: The optimally chain-permuted aligned true coordinates and mask.
        """
        batch_size, num_atoms = pred_coords.shape[:2]

        if not_exists(additional_molecule_feats) or not_exists(is_molecule_types):
            # NOTE: If no chains or no molecule types are specified,
            # we cannot perform multi-chain permutation alignment.
            true_coords.detach_()
            return true_coords

        if exists(mask):
            # Zero out all predicted and true coordinates where not an atom.
            # pred_coords = einx.where("b n, b n c, -> b n c", mask, pred_coords, 0.0)
            # true_coords = einx.where("b n, b n c, -> b n c", mask, true_coords, 0.0)
            pred_coords = pred_coords * mask[..., None]
            true_coords = true_coords * mask[..., None]

        # Alignment Stage - Section 7.3.1 of the AlphaFold-Multimer Paper

        _, token_index, token_asym_id, token_entity_id, _ = additional_molecule_feats.unbind(
            dim=-1
        )

        # NOTE: Ligands covalently bonded to polymer chains are to be permuted
        # in sync with the corresponding chains by assigning them the same
        # entity ID (entity_id) to group all covalently bonded components together.
        polymer_indices = [IS_PROTEIN_INDEX, IS_RNA_INDEX, IS_DNA_INDEX]
        ligand_indices = [IS_LIGAND_INDEX, IS_METAL_ION_INDEX]

        is_polymer_types = is_molecule_types[..., polymer_indices].any(-1)
        is_ligand_types = is_molecule_types[..., ligand_indices].any(-1)

        polymer_ligand_pair_mask = is_polymer_types[..., None] & is_ligand_types[..., None, :]
        polymer_ligand_pair_mask = polymer_ligand_pair_mask | polymer_ligand_pair_mask.transpose(
            -1, -2
        )

        covalent_bond_mask = polymer_ligand_pair_mask & token_bonds

        is_covalent_residue_mask = covalent_bond_mask.any(-1)
        is_covalent_polymer_mask = is_polymer_types & is_covalent_residue_mask
        is_covalent_ligand_mask = is_ligand_types & is_covalent_residue_mask

        # NOTE: Covalent ligand-polymer bond pairs may be many-to-many, so
        # we need to group them together by assigning covalent ligands the same
        # entity IDs as the polymer chains to which they are most frequently bonded.
        mapped_token_entity_id = token_entity_id.clone()
        for i in torch.where(is_covalent_ligand_mask)[-1]:
            mapped_token_entity_id[:, i] = (
                token_entity_id[covalent_bond_mask[:, i] & is_covalent_polymer_mask]
                .mode(-1)
                .values
            )

        # Compute the optimal permutation alignment for each batch element.
        permuted_true_coords_list, permuted_mask_list = [], []

        for batch_idx in range(batch_size):
            i = slice(batch_idx, batch_idx + 1)

            # Record initial batch-wise RMSD as a reference point for sanity-checking.
            initial_rmsd = batch_compute_rmsd(
                true_pos=true_coords[i],
                pred_pos=pred_coords[i],
                mask=mask[i],
            ).mean()

            # Segment the ground truth coordinates into chains.
            asym_id = batch_repeat_interleave(token_asym_id[i], molecule_atom_lens[i])
            labels = self.split_ground_truth_labels(
                dict(asym_id=asym_id, true_coords=true_coords[i], mask=mask[i])
            )

            # Pool atom-level features into token-level features.
            mol_atom_indices = repeat(
                molecule_atom_indices[i], "b m -> b m d", d=true_coords[i].shape[-1]
            )

            token_pred_coords = torch.gather(pred_coords[i], 1, mol_atom_indices)
            token_true_coords = torch.gather(true_coords[i], 1, mol_atom_indices)
            token_mask = torch.gather(mask[i], 1, molecule_atom_indices[i])

            # Permute ground truth chains.
            out = {"pred_coords": token_pred_coords, "mask": token_mask}
            features = {
                "asym_id": token_asym_id[i],
                "entity_id": mapped_token_entity_id[i],
                "token_index": token_index[i],
            }
            ground_truth = {
                "true_coords": token_true_coords,
                "mask": token_mask,
                "asym_id": token_asym_id[i],
                "entity_id": mapped_token_entity_id[i],
            }

            alignments = self.compute_permutation_alignment(
                out=out,
                features=features,
                ground_truth=ground_truth,
            )

            # Reorder ground truth coordinates according to permutation results.
            labels = self.merge_labels(
                labels=labels,
                alignments=alignments,
                original_num_tokens=num_atoms,
            )
            permuted_true_coords = labels["true_coords"]
            permuted_mask = labels["mask"]

            # Sanity-check new batch-wise RMSD.
            new_rmsd = batch_compute_rmsd(
                true_pos=permuted_true_coords,
                pred_pos=pred_coords[i],
                mask=permuted_mask,
            ).mean()

            if new_rmsd >= initial_rmsd:
                # NOTE: If the new batch-wise RMSD is greater than or equal to the initial batch-wise RMSD,
                # we revert to the original ground truth coordinates.
                if verbose:
                    logger.warning(
                        f"Multi-chain permutation alignment failed to reduce batch-wise RMSD (new: {new_rmsd} vs. initial: {initial_rmsd}). "
                        "Reverting to original ground truth coordinates."
                    )
                permuted_true_coords = true_coords[i]
                permuted_mask = mask[i]
            elif verbose:
                logger.info(
                    f"Multi-chain permutation alignment successfully reduced batch-wise RMSD (new: {new_rmsd} vs. initial: {initial_rmsd})."
                )

            permuted_true_coords_list.append(permuted_true_coords.squeeze(0))
            permuted_mask_list.append(permuted_mask.squeeze(0))

        permuted_true_coords = torch.stack(permuted_true_coords_list)
        permuted_mask = torch.stack(permuted_mask_list)

        return permuted_true_coords, permuted_mask


class AtomPermutationAlignment(Module):
    """Section 4.2 of the AlphaFold 3 Supplement.

    Implementation adapted from:
    https://github.com/bytedance/Protenix
    """

    @typecheck
    def __init__(
        self,
        run_checker: bool = False,
        eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__()

        self.run_checker = run_checker
        self.eps = eps

    @typecheck
    @staticmethod
    def get_identity_permutation(batch_size: int, num_atoms: int, device: str | torch.device) -> Int["b m"]:  # type: ignore
        """Return identity permutation indices if no permutations exist for each residue.

        :param batch_size: The batch size.
        :param num_atoms: The number of atoms.
        :param device: The device.
        :return: The identity permutation indices.
        """
        identity = torch.arange(num_atoms, device=device)
        return torch.stack([identity for _ in range(batch_size)], dim=0)

    @typecheck
    @staticmethod
    def is_permutation(x: Int[" n"]):  # type: ignore
        """
        Check if the input tensor `x` is a permutation of integers from 0 to N - 1.

        :param x: The input tensor.
        """
        assert x.dim() == 1, "Input tensor must be 1D."
        N = x.size(0)
        assert torch.equal(
            torch.sort(x)[0], torch.arange(N, device=x.device)
        ), f"Input tensor must be a permutation of integers from 0 to N - 1. "

    @typecheck
    @staticmethod
    def are_permutations(x: Int["... n"], dim: int = -1):  # type: ignore
        """
        Check if slices along the specified dimension in `x` are permutations of integers from 0 to N - 1.

        :param x: A tensor with any number of dimensions, containing slices of size `n` along `dim`.
        :param dim: The dimension along which to check for permutations. Defaults to `-1`.
        """
        assert x.dim() > 0, "Input tensor must have at least one dimension."

        # Create a view of `x` that moves the specified dimension to `-1`.
        N = x.size(dim)
        x = x.transpose(dim, -1).contiguous()
        x = x.reshape(-1, N)
        for i in range(x.size(0)):
            AtomPermutationAlignment.is_permutation(x[i])

    @typecheck
    @staticmethod
    def contains_identity(x: Int["... n"], dim: int = -1):  # type: ignore
        """Check if `x` contains the identity permutation.

        :param x: A tensor with any number of dimensions, containing slices of size `n` along `dim`.
        :param dim: The dimension along which to check for permutations. Defaults to `-1`.
        """
        assert x.dim() > 0, (
            f"Input tensor must have at least one dimension. " f"Found {x.dim()} dimensions."
        )

        # Create a view of `x` that moves the specified dimension to `-1`.
        N = x.size(dim)
        x = x.transpose(dim, -1).contiguous()
        x = x.reshape(-1, N)

        expected = torch.arange(N, device=x.device).unsqueeze(dim=0)
        assert (x == expected).all(dim=-1).any(), (
            "Input tensor must contain the identity permutation. "
            "Found no identity permutation in the input tensor."
        )

    @typecheck
    @staticmethod
    def does_not_contain_identity(x: Int["... n"], dim: int = -1):  # type: ignore
        """Check if `x` does not contain the identity permutation.

        :param x: A tensor with any number of dimensions, containing slices of size `n` along `dim`.
        :param dim: The dimension along which to check for permutations. Defaults to `-1`.
        """
        assert x.dim() > 0, (
            f"Input tensor must have at least one dimension. " f"Found {x.dim()} dimensions."
        )

        # Create a view of `x` that moves the specified dimension to `-1`.
        N = x.size(dim)
        x = x.transpose(dim, -1).contiguous()
        x = x.reshape(-1, N)
        expected = torch.arange(N, device=x.device).unsqueeze(dim=0)
        assert not (x == expected).all(dim=-1).any(), (
            "Input tensor must not contain the identity permutation. "
            "Found the identity permutation in the input tensor."
        )

    @typecheck
    @staticmethod
    def batch_permute(perm: Int[" n"], x: Float["n batch_dims_x"] | Int["n batch_dims_x"], x_permuted: Float["... n batch_dims_x"] | Int["... n batch_dims_x"]):  # type: ignore
        """Permute the input tensor `x` according to the permutation `perm`.

        :param perm: The permutation tensor.
        :param x: The input tensor.
        :param x_permuted: The permuted input tensor.
        """
        batch_shape = perm.shape[:-1]
        N = perm.size(-1)
        assert x.size(0) == N, (
            f"Input tensor must have the same size as the permutation tensor. "
            f"Found {x.size(0)} elements in the input tensor and {N} elements in the permutation tensor."
        )

        perm = perm.view(-1, N)
        permuted_x = [x[perm[i]] for i in range(len(perm))]
        permuted_x = torch.stack(permuted_x, dim=0)  # NOTE: Has shape [-1, n, batch_dims_x]
        target_shape = batch_shape + (N,) + x.shape[1:]
        assert torch.allclose(permuted_x.reshape(target_shape), x_permuted), (
            f"Permutated input tensor must match the permuted input tensor. "
            f"Found {permuted_x.size()} permuted elements and {x_permuted.size()} permuted elements."
        )

    @typecheck
    @staticmethod
    def pad_at_dim(
        x: torch.Tensor,
        dim: int,
        pad_length: Tuple[int, int] | List[int],
        value: float = 0.0,
    ) -> torch.Tensor:
        """Pad input `x` at dimension `dim` with length `pad_length[0]` to the left and
        `pad_length[1]` to the right.

        :param x: The input tensor.
        :param dim: The dimension along which to pad.
        :param pad_length: The length of padding to apply to the left and right.
        :param value: The value to use for padding.
        :return: The padded tensor.
        """
        num_dims = len(x.shape)
        if dim < 0:
            dim = num_dims + dim

        pad = (pad_length[0], pad_length[1])
        if pad == (0, 0):
            return x

        k = num_dims - (dim + 1)
        if k > 0:
            pad_skip = (0, 0) * k
            pad = (*pad_skip, *pad)

        return nn.functional.pad(x, pad=pad, value=value)

    @typecheck
    @staticmethod
    def collect_permuted_coords(
        coords_list: List[Float["num_res_atoms 3"]],  # type: ignore
        mask_list: List[Bool["num_res_atoms"]],  # type: ignore
        perms_list: List[Int["* num_res_atoms"]],  # type: ignore
        run_checker: bool = False,
    ) -> Tuple[Int["num_total_perms max_num_res_atoms"], Int["num_total_perms max_num_res_atoms"]]:  # type: ignore
        """Apply permutations to coordinates and masks.

        :param coords_list: The list of coordinates, each of shape [num_res_atoms, 3].
        :param mask_list: The list of masks, each of shape [num_res_atoms].
        :param perms_list: The list of permutations, each of shape [num_perms, num_res_atoms] where `num_perms` can vary per residue.
        :param run_checker: Whether to run the checker.
        :return: The permuted coordinates and masks.
        """
        max_num_res_atoms = max(perm.size(-1) for perm in perms_list)
        perm_coords = []  # NOTE: Has shape [num_total_perms, num_res_atoms, 3]
        perm_masks = []  # NOTE: Has shape [num_total_perms, num_res_atoms]

        num_total_perms = 0
        for perm, res_coords, res_mask in zip(perms_list, coords_list, mask_list):
            # Perform basic shape checks.
            num_perms, num_res_atoms = perm.size()
            assert res_coords.size(-1) == 3, (
                f"Coordinates must have shape [num_res_atoms, 3]. "
                f"Found {res_coords.size()} coordinates."
            )
            assert res_coords.size(0) == res_mask.size(0) == perm.size(-1), (
                f"Coordinates and masks must have the same number of atoms. "
                f"Found {res_coords.size(0)} atoms in the coordinates and {res_mask.size(0)} atoms in the mask."
            )

            # Permute coordinates and masks.
            res_coords_permuted = res_coords[perm]  # NOTE: Has shape [num_perms, num_res_atoms, 3]
            res_mask_permuted = res_mask[perm]  # NOTE: Has shape [num_perms, num_res_atoms]

            assert res_coords_permuted.size() == (num_perms, num_res_atoms, 3), (
                f"Permutated coordinates must have shape [num_perms, num_res_atoms, 3]. "
                f"Found {res_coords_permuted.size()} coordinates."
            )
            assert res_mask_permuted.size() == (num_perms, num_res_atoms), (
                f"Permutated masks must have shape [num_perms, num_res_atoms]. "
                f"Found {res_mask_permuted.size()} masks."
            )

            if run_checker:
                AtomPermutationAlignment.are_permutations(perm, dim=-1)
                AtomPermutationAlignment.batch_permute(perm, res_coords, res_coords_permuted)
                AtomPermutationAlignment.batch_permute(perm, res_mask, res_mask_permuted)

            # Pad to `max_num_res_atoms`.
            num_res_atoms = perm.size(dim=-1)
            if num_res_atoms < max_num_res_atoms:
                pad_length = (0, max_num_res_atoms - num_res_atoms)
                res_coords_permuted = AtomPermutationAlignment.pad_at_dim(
                    res_coords_permuted, dim=-2, pad_length=pad_length
                )  # NOTE: Has shape [num_perms, max_num_res_atoms, 3]
                res_mask_permuted = AtomPermutationAlignment.pad_at_dim(
                    res_mask_permuted, dim=-1, pad_length=pad_length
                )

            num_total_perms += num_perms
            perm_coords.append(res_coords_permuted)
            perm_masks.append(res_mask_permuted)

        perm_coords = torch.cat(perm_coords, dim=0)
        perm_masks = torch.cat(perm_masks, dim=0)

        # Perform final shape checks.
        assert perm_coords.size() == (num_total_perms, max_num_res_atoms, 3), (
            f"Permutated coordinates must have shape [num_total_perms, max_num_res_atoms, 3]. "
            f"Found {perm_coords.size()} coordinates."
        )
        assert perm_masks.size() == (num_total_perms, max_num_res_atoms), (
            f"Permutated masks must have shape [num_total_perms, max_num_res_atoms]. "
            f"Found {perm_masks.size()} masks."
        )

        return perm_coords, perm_masks

    @typecheck
    @staticmethod
    def expand_at_dim(x: torch.Tensor, dim: int, n: int) -> torch.Tensor:
        """Expand a tensor `x` at dimension `dim` `n` times.

        :param x: The input tensor.
        :param dim: The dimension along which to expand.
        :param n: The number of times to expand.
        :return: The expanded tensor.
        """
        x = x.unsqueeze(dim=dim)

        if dim < 0:
            dim = x.dim() + dim

        before_shape = x.shape[:dim]
        after_shape = x.shape[dim + 1 :]

        return x.expand(*before_shape, n, *after_shape)

    @typecheck
    @staticmethod
    def check_identity(j_value: int, perm: Int["... n"]):  # type: ignore
        """Check if the identity permutation is contained in the best permutation.

        :param j_value: The value of the best permutation.
        :param perm: The best permutation.
        """
        if j_value > 0:
            AtomPermutationAlignment.does_not_contain_identity(perm)
        else:
            AtomPermutationAlignment.contains_identity(perm)

    @typecheck
    @staticmethod
    def optimize_per_residue_permutation_by_rmsd(
        per_residue_pred_coord_list: List[torch.Tensor],
        per_residue_coord_list: List[torch.Tensor],
        per_residue_coord_mask_list: List[torch.Tensor],
        per_residue_perm_list: List[torch.Tensor],
        run_checker: bool = False,
        eps: float = 1e-8,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        """Find the optimal permutations of true coordinates and masks to minimize the RMSD between
        the true and predicted coordinates.

        :param per_residue_pred_coord_list: The predicted coordinates for each residue.
        :param per_residue_coord_list: The true coordinates for each residue.
        :param per_residue_coord_mask_list: The mask for each residue.
        :param per_residue_perm_list: The molecule atom permutations for each residue.
        :param run_checker: Whether to run the checker.
        :param eps: The epsilon value.
        :return: The best permutation, whether the residue is permuted, the optimized RMSD, and the
            original RMSD.
        """
        # Find max number of per-residue atoms.
        per_residue_num_perms = [perm.size(0) for perm in per_residue_perm_list]
        per_residue_num_atoms = [perm.size(1) for perm in per_residue_perm_list]
        num_max_atoms = max(per_residue_num_atoms)

        # Permute true coordinates and masks according to the permutations in `per_residue_perm_list`.
        permuted_coords, permuted_mask = AtomPermutationAlignment.collect_permuted_coords(
            coords_list=per_residue_coord_list,
            mask_list=per_residue_coord_mask_list,
            perms_list=per_residue_perm_list,
            run_checker=run_checker,
        )  # NOTE: Have shapes ([num_total_perms, num_max_atoms, 3], [num_total_perms, num_max_atoms])

        assert permuted_coords.size(-2) == permuted_mask.size(-1) == num_max_atoms, (
            f"Permutated coordinates and masks must have the same number of atoms. "
            f"Found {permuted_coords.size(-2)} atoms in the coordinates and {permuted_mask.size(-1)} atoms in the mask."
        )
        num_total_perms = permuted_coords.size(0)

        # Pad `pred_coord` to the same shape as `permuted_coord`.
        per_residue_pred_coord_list = [
            AtomPermutationAlignment.pad_at_dim(
                p_coords, dim=-2, pad_length=(0, num_max_atoms - p_coords.size(dim=-2))
            )
            for p_coords in per_residue_pred_coord_list
        ]

        # Repeat `num_perms` times for each residue.
        pred_coords = torch.stack(
            sum(
                [
                    [p_coords] * n_perms
                    for n_perms, p_coords in zip(
                        per_residue_num_perms, per_residue_pred_coord_list
                    )
                ],
                [],
            ),
            dim=-3,
        )  # NOTE: Has shape [num_total_perms, num_max_atoms, 3] or [batch_size, num_total_perms, num_max_atoms, 3]
        assert pred_coords.shape[-3:] == (num_total_perms, num_max_atoms, 3), (
            f"Predicted coordinates must have shape [num_total_perms, num_max_atoms, 3]. "
            f"Found {pred_coords.shape[-3:]} coordinates."
        )

        batch_shape = pred_coords.shape[:-3]
        assert len(batch_shape) in (0, 1), (
            f"Batch shape must be either empty or have a single dimension. "
            f"Found {len(batch_shape)} dimensions."
        )
        if len(batch_shape) == 1:
            # Expand true coordinates and mask to have the same batch size as the predicted coordinates.
            batch_size = pred_coords.size(0)
            permuted_coords = AtomPermutationAlignment.expand_at_dim(
                permuted_coords, dim=0, n=batch_size
            )
            permuted_mask = AtomPermutationAlignment.expand_at_dim(
                permuted_mask, dim=0, n=batch_size
            )

        # Compute the per-residue RMSD.
        amp_context = (
            torch.autocast(device_type="cuda", enabled=False, cache_enabled=False)
            if torch.cuda.is_available()
            else nullcontext()
        )
        with amp_context:
            per_res_rmsd = batch_compute_rmsd(
                true_pos=permuted_coords.float(),
                pred_pos=pred_coords.float(),
                mask=permuted_mask,
                eps=eps,
            )  # NOTE: Has shape [num_total_perms] or [batch_size, num_total_perms]
            assert per_res_rmsd.size() == batch_shape + (num_total_perms,), (
                f"Per-residue RMSD must have shape [batch_size, num_total_perms]. "
                f"Found {per_res_rmsd.size()} RMSD values."
            )

        # Find the best atom permutation.
        best_permutation_list = []
        is_permuted_list = []
        optimized_rmsd_list = []
        original_rmsd_list = []

        # Enumerate over all residues - NOTE: This could later be improved by instead using `scatter()`.
        perm_idx = 0
        for num_perms, num_res_atoms, perm in zip(
            per_residue_num_perms, per_residue_num_atoms, per_residue_perm_list
        ):
            cur_res_rmsd = per_res_rmsd[
                ..., perm_idx : perm_idx + num_perms
            ]  # NOTE: Has shape [batch_shape, num_perms]
            best_rmsd, best_j = torch.min(cur_res_rmsd, dim=-1)  # NOTE: Has shape [batch_shape]
            best_perm = perm[best_j]  # NOTE: Has shape [batch_shape, num_res_atoms]
            best_permutation_list.append(best_perm)

            is_permuted_list.append(
                best_j > 0
            )  # NOTE: The first item of the permutation lists is the identity permutation.

            optimized_rmsd_list.append(best_rmsd)
            original_rmsd_list.append(cur_res_rmsd[..., 0])

            perm_idx += num_perms

            # Perform development checks.
            if run_checker:
                assert perm.size() == (
                    num_perms,
                    num_res_atoms,
                ), "Permutation must have shape [num_perms, num_res_atoms]."
                assert cur_res_rmsd.size() == batch_shape + (
                    num_perms,
                ), "Per-residue RMSD must have shape [batch_size, num_perms]."
                assert best_rmsd.size() == batch_shape, "Best RMSD must have shape [batch_size]."
                assert (
                    best_j.size() == batch_shape
                ), "Best permutation index must have shape [batch_size]."
                assert best_perm.size() == batch_shape + (
                    num_res_atoms,
                ), "Best permutation must have shape [batch_size, num_res_atoms]."
                AtomPermutationAlignment.are_permutations(
                    best_perm, dim=-1
                ), "Best permutation must be a permutation."

                if best_j.dim() == 0:
                    AtomPermutationAlignment.check_identity(best_j, best_perm)
                else:
                    for j_value, perm_j in zip(best_j, best_perm):
                        AtomPermutationAlignment.check_identity(j_value, perm_j)

        return (
            best_permutation_list,
            is_permuted_list,
            optimized_rmsd_list,
            original_rmsd_list,
        )

    @typecheck
    def identify_residues_with_symmetric_atoms(
        self,
        coords: Float["m 3"],  # type: ignore
        chain_residue_index: Int[" m"],  # type: ignore
        molecule_atom_perms: List[List[int]],
        mask: Bool[" m"] | None = None,  # type: ignore
        min_num_atoms_required: int = 3,
    ) -> Tuple[List[Tuple[int, int]], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], int]:  # type: ignore
        """Convert atom-level permutation attributes to residue-level attributes. Only residues
        that require symmetric corrections are returned.

        :param coords: The coordinates.
        :param chain_residue_index: The joint chain-residue index UID of each atom.
        :param molecule_atom_perms: The molecule atom permutations.
        :param mask: The mask for variable lengths.
        :param min_num_atoms_required: The minimum number of atoms required for alignment.
        :return: The residues with symmetric atoms.
        """
        device = mask.device

        # Find the start and end positions of each residue.
        diff = torch.tensor(
            [True] + (chain_residue_index[1:] != chain_residue_index[:-1]).tolist()
        )
        res_start_pos = torch.cat(
            (torch.nonzero(diff, as_tuple=True)[0], torch.tensor([len(chain_residue_index)]))
        )
        res_start_end = list(
            zip(res_start_pos[:-1].tolist(), res_start_pos[1:].tolist())
        )  # NOTE: Has shape [num_res, 2]

        num_res = len(res_start_end)
        assert num_res == len(
            torch.unique(chain_residue_index)
        ), f"Number of residues must match the number of unique chain-residue indices. Found {num_res} residues and {len(torch.unique(chain_residue_index))} unique indices."

        pos_list = []
        coords_list = []
        mask_list = []
        perms_list = []

        # Traverse residues and store the corresponding data.
        for start, end in res_start_end:
            assert len(torch.unique(chain_residue_index[start:end])) == 1, (
                f"Residue indices must be unique within each residue. "
                f"Found {len(torch.unique(chain_residue_index[start:end]))} unique indices "
                f"within residue {start}:{end}."
            )

            # Skip if this residue contains less than `min_num_atoms_required` resolved atoms.
            # NOTE: Atom permutation alignment requires at least `min_num_atoms_required` atoms to obtain a reasonable result.
            res_mask = mask[start:end].bool()  # NOTE: Has shape [num_res_atom]
            if res_mask.sum() < min_num_atoms_required:
                continue

            # Drop duplicated permutations.
            perm = torch.tensor(molecule_atom_perms[start:end], device=device, dtype=torch.long)
            perm = torch.unique(perm, dim=-1)  # NOTE: Has shape [num_res_atom, num_perms]
            num_res_atom = perm.shape[0]

            # Perform basic checks.
            assert perm.min().item() == 0, f"Minimum permutation index must be 0."
            assert (
                perm.max().item() == num_res_atom - 1
            ), f"Maximum permutation index must be {num_res_atom - 1}."

            # Maybe perform development checks.
            if self.run_checker:
                self.are_permutations(perm, dim=0)
                self.contains_identity(perm, dim=0)

            # If all symmetric atoms are unresolved, drop the permutation.
            identity_perm = torch.arange(len(perm), device=device).unsqueeze(
                dim=-1
            )  # NOTE: Has shape [num_res_atom, 1]

            is_sym_atom = perm != identity_perm  # [num_res_atom, N_perm]
            is_sym_atom_resolved = is_sym_atom * res_mask.unsqueeze(dim=-1)
            is_valid_perm = is_sym_atom_resolved.any(dim=0)
            if not is_valid_perm.any():
                # Skip if no valid permutation (other than the identity) exists.
                continue

            perm = perm[..., is_valid_perm]
            perm = torch.cat([identity_perm, perm], dim=-1)  # Put identity at the first position.
            perm = perm.transpose(-1, -2)  # NOTE: Has shape [num_perms, num_res_atom]

            pos_list.append((start, end))
            coords_list.append(coords[start:end, :])
            mask_list.append(res_mask)
            perms_list.append(perm)

        return pos_list, coords_list, mask_list, perms_list, num_res

    @typecheck
    @torch.no_grad()
    def permute_atoms(
        self,
        pred_coords: Float["b m 3"],  # type: ignore
        true_coords: Float["m 3"],  # type: ignore
        chain_residue_index: Int[" m"],  # type: ignore
        molecule_atom_perms: List[List[int]],  # type: ignore
        mask: Bool[" m"] | None = None,  # type: ignore
        alignment_mask: Bool[" m"] | None = None,  # type: ignore
        min_num_atoms_required: int = 3,
        res_atom_cutoff: List[int] = [15, 30, 50, 100, 100000],
        verbose: bool = False,
    ) -> Tuple[Int["b m"], Dict[str, Any]]:  # type: ignore
        """Permute atoms.

        :param pred_coords: Predicted coordinates.
        :param true_coords: True coordinates.
        :param chain_residue_index: The joint chain-residue index UID of each atom.
        :param molecule_atom_perms: The molecule atom permutations.
        :param mask: The mask for variable lengths.
        :param alignment_mask: The alignment mask.
        :param min_num_atoms_required: The minimum number of atoms required for alignment.
        :param res_atom_cutoff: The residue atom cutoffs for optimizing runtime.
        :param verbose: Whether to print verbose logs.
        :return: Permutation indices and the alignment metadata dictionary.
        """
        device = pred_coords.device
        batch_size, num_atoms = pred_coords.shape[:2]

        # Prepare to collect alignment metadata.
        alignment_info = {}

        # Initialize an identity permutation.
        permutation = self.get_identity_permutation(
            batch_size=batch_size, num_atoms=num_atoms, device=device
        )

        # Identify residues that require permutations.
        (
            per_residue_pos_list,
            per_residue_coords_list,
            per_residue_mask_list,
            per_residue_perms_list,
            num_res,
        ) = self.identify_residues_with_symmetric_atoms(
            coords=true_coords,
            chain_residue_index=chain_residue_index,
            molecule_atom_perms=molecule_atom_perms,
            mask=mask,
            min_num_atoms_required=min_num_atoms_required,
        )

        alignment_info["num_res"] = num_res
        alignment_info["num_res_with_symmetry"] = len(per_residue_coords_list)
        alignment_info["num_res_permuted"] = 0.0
        alignment_info["has_res_permuted"] = 0

        # If no residues contain symmetries, exit early.
        if not per_residue_perms_list:
            if verbose:
                logger.info("No atom permutations are needed. Returning the identity permutation.")
            return permutation, alignment_info

        # Perform a global alignment of predicted to true coordinates.
        alignment_mask = alignment_mask * mask if exists(alignment_mask) else mask

        if alignment_mask.sum().item() < min_num_atoms_required:
            if verbose:
                logger.info(
                    f"Fewer than {min_num_atoms_required} atoms are aligned. No atom permutations are needed. "
                    "Returning the identity permutation."
                )
            return permutation, alignment_info

        aligned_pred_coords = batch_compute_rigid_alignment(
            true_pos=true_coords,
            pred_pos=pred_coords,
            mask=alignment_mask,
            eps=self.eps,
        )[-1]

        # To efficiently optimize the residues parallely, group the
        # residues according to the number of atoms in each residue.
        per_residue_num_atoms = [coord.size(0) for coord in per_residue_coords_list]

        grouped_indices = {}
        for i, n in enumerate(per_residue_num_atoms):
            for atom_cutoff in res_atom_cutoff:
                if n <= atom_cutoff:
                    break
            grouped_indices.setdefault(atom_cutoff, []).append(i)

        assert len(sum(list(grouped_indices.values()), [])) == len(per_residue_perms_list), (
            f"Grouped indices must have the same length as the number of residues with symmetry. "
            f"Found {len(sum(list(grouped_indices.values()), []))} grouped indices and "
            f"{len(per_residue_perms_list)} residues with symmetry."
        )

        residue_pos_list = []
        residue_best_permutation_list = []
        residue_is_permuted_list = []
        residue_optimized_rmsd_list = []
        residue_original_rmsd_list = []
        for atom_cutoff, residue_group in grouped_indices.items():
            if verbose:
                logger.info(f"{len(residue_group)} residues have <= {atom_cutoff} atoms.")

            # Enumerate permutations within each residue to minimize per-residue RMSD.
            per_res_pos_list = [per_residue_pos_list[i] for i in residue_group]
            (
                per_res_best_permutation,
                per_res_is_permuted,
                per_res_optimized_rmsd,
                per_res_orig_rmsd,
            ) = self.optimize_per_residue_permutation_by_rmsd(
                per_residue_pred_coord_list=[
                    aligned_pred_coords[..., pos[0] : pos[1], :] for pos in per_res_pos_list
                ],
                per_residue_coord_list=[per_residue_coords_list[i] for i in residue_group],
                per_residue_coord_mask_list=[per_residue_mask_list[i] for i in residue_group],
                per_residue_perm_list=[per_residue_perms_list[i] for i in residue_group],
                run_checker=self.run_checker,
                eps=self.eps,
            )
            residue_pos_list.extend(per_res_pos_list)
            residue_best_permutation_list.extend(per_res_best_permutation)
            residue_is_permuted_list.extend(per_res_is_permuted)
            residue_optimized_rmsd_list.extend(per_res_optimized_rmsd)
            residue_original_rmsd_list.extend(per_res_orig_rmsd)

        # Aggregate per-residue results and statistics.
        indices_list = [torch.arange(pos[0], pos[1], device=device) for pos in residue_pos_list]
        residue_atom_indices = torch.cat(indices_list, dim=-1)  # NOTE: Has shape [num_perm_atoms]
        residue_best_permutation = torch.cat(
            [ind[perm] for ind, perm in zip(indices_list, residue_best_permutation_list)],
            dim=-1,
        )  # NOTE: Has shape [batch_size, num_perm_atoms] or [num_perm_atoms]
        permutation[..., residue_atom_indices] = residue_best_permutation

        is_res_permuted = torch.stack(residue_is_permuted_list, dim=-1).float()
        alignment_info["num_res_permuted"] = is_res_permuted.sum(dim=-1).mean().item()
        alignment_info["has_res_permuted"] = (
            (is_res_permuted.sum(dim=-1) > 0).float().mean().item()
        )

        return permutation, alignment_info

    @typecheck
    def optimally_permute_symmetric_atoms(
        self,
        pred_coords: Float["b m 3"],  # type: ignore
        true_coords: Float["m 3"],  # type: ignore
        chain_residue_index: Int[" m"],  # type: ignore
        molecule_atom_perms: List[List[int]],  # type: ignore
        mask: Bool[" m"] | None = None,  # type: ignore
        alignment_mask: Bool[" m"] | None = None,  # type: ignore
        permute_labels: bool = True,
        verbose: bool = False,
    ) -> Tuple[Float["b m 3"], Bool["b m"], Dict[str, Any], Int["b m"]]:  # type: ignore
        """Optimally permute atoms.

        :param pred_coords: Predicted coordinates.
        :param true_coords: True coordinates.
        :param chain_residue_index: The joint chain-residue index UID of each atom.
        :param molecule_atom_perms: The molecule atom permutations.
        :param mask: The mask for variable lengths.
        :param alignment_mask: The alignment mask.
        :param permute_labels: Whether to permute labels rather than predictions.
        :param verbose: Whether to print verbose logs.
        :return: The optimally atom-permuted aligned true coordinates along with the corresponding
            alignment mask, alignment metadata dictionary, and permutation indices.
        """
        alignment_mask = alignment_mask * mask if exists(alignment_mask) else mask

        indices_permutation, alignment_info = self.permute_atoms(
            pred_coords=pred_coords,
            true_coords=true_coords,
            chain_residue_index=chain_residue_index,
            molecule_atom_perms=molecule_atom_perms,
            mask=mask,
            alignment_mask=alignment_mask,
            verbose=verbose,
        )

        if permute_labels:
            # Return the permutation of the true coordinates.
            return (
                true_coords[indices_permutation],
                mask[indices_permutation],
                alignment_info,
                indices_permutation,
            )
        else:
            # Return the permutation of the predicted coordinates.
            indices_permutation = torch.argsort(indices_permutation, dim=1)
            indices_permutation_expanded = indices_permutation[..., None].expand(-1, -1, 3)
            pred_coord_permuted = pred_coords.gather(1, indices_permutation_expanded)

            return (
                pred_coord_permuted,
                mask[indices_permutation],
                alignment_info,
                indices_permutation,
            )

    @typecheck
    def forward(
        self,
        pred_coords: Float["b m 3"],  # type: ignore
        true_coords: Float["b m 3"],  # type: ignore
        additional_molecule_feats: Int[f"b n {ADDITIONAL_MOLECULE_FEATS}"],  # type: ignore
        molecule_atom_lens: Int["b n"],  # type: ignore
        molecule_atom_perms: List[List[List[int]]],  # type: ignore
        mask: Bool["b m"] | None = None,  # type: ignore
        alignment_mask: Bool["b m"] | None = None,  # type: ignore
        permute_labels: bool = True,
        only_accept_improved_global_rmsd: bool = False,
        verbose: bool = False,
    ) -> Tuple[Float["b m 3"], Bool["b m"]]:  # type: ignore
        """Compute the atom permutation alignment.

        NOTE: This function assumes that the ground truth features are batched yet only contain
        features for the same structure. This is the case after performing data augmentation
        with a batch size of 1 in the `MegaFold` module's forward pass. If the batched
        ground truth features represent multiple different structures, this function will not
        return correct results.

        :param pred_coords: Predicted coordinates.
        :param true_coords: True coordinates.
        :param additional_molecule_feats: Additional molecule features.
        :param molecule_atom_lens: The molecule atom lengths.
        :param molecule_atom_perms: The molecule atom permutations.
        :param mask: The mask for variable lengths.
        :param alignment_mask: The alignment mask.
        :param permute_labels: Whether to permute labels rather than predictions.
        :param only_accept_improved_global_rmsd: Whether to only accept alignments with improved global RMSD.
        :param verbose: Whether to print verbose logs.
        :return: The optimally atom-permuted aligned (true if `permute_labels`, else predicted) coordinates and mask.
        """
        aligned_coords = true_coords if permute_labels else pred_coords
        unaligned_coords = pred_coords if permute_labels else true_coords

        residue_index, _, chain_index, _, _ = additional_molecule_feats.unbind(dim=-1)
        chain_residue_index = create_uid_tensor(chain_index, residue_index)
        atom_chain_residue_index = batch_repeat_interleave(chain_residue_index, molecule_atom_lens)

        # Record initial batch-wise RMSD as a reference point for sanity-checking.
        initial_rmsd = batch_compute_rigid_alignment(
            true_pos=unaligned_coords,
            pred_pos=aligned_coords,
            mask=mask,
        )[0].mean()

        try:
            (
                permuted_aligned_coords,
                permuted_aligned_mask,
                _,
                _,
            ) = self.optimally_permute_symmetric_atoms(
                pred_coords=pred_coords,
                # NOTE: We need only reference one copy of the true features.
                true_coords=true_coords[0],
                chain_residue_index=atom_chain_residue_index[0],
                molecule_atom_perms=molecule_atom_perms[0],
                mask=mask[0],
                alignment_mask=alignment_mask[0] if exists(alignment_mask) else None,
                permute_labels=permute_labels,
                verbose=verbose,
            )
        except Exception as e:
            if verbose:
                logger.warning(
                    f"Atom permutation alignment failed with error: {e}. Reverting to original aligned coordinates."
                )
            permuted_aligned_coords = aligned_coords
            permuted_aligned_mask = mask

        new_rmsd = batch_compute_rigid_alignment(
            true_pos=unaligned_coords,
            pred_pos=permuted_aligned_coords,
            mask=permuted_aligned_mask,
        )[0].mean()

        if only_accept_improved_global_rmsd and new_rmsd >= initial_rmsd:
            # NOTE: If the new batch-wise RMSD is greater than or equal to the initial batch-wise RMSD, we revert
            # to the original aligned coordinates. For chain-wise permutations, we almost always want this behavior.
            # However, here for atom-level permutations, we accept any optimal permutation by default.
            if verbose:
                logger.warning(
                    f"Atom permutation alignment failed to reduce batch-wise RMSD (new: {new_rmsd} vs. initial: {initial_rmsd}). "
                    "Reverting to original aligned coordinates."
                )
            permuted_aligned_coords = aligned_coords
            permuted_aligned_mask = mask
        elif verbose:
            logger.info(
                f"Atom permutation alignment successfully reduced batch-wise RMSD (new: {new_rmsd} vs. initial: {initial_rmsd})."
            )

        return permuted_aligned_coords, permuted_aligned_mask


class CentreRandomAugmentation(Module):
    """Algorithm 19."""

    @typecheck
    def __init__(self, trans_scale: float = 1.0):
        super().__init__()
        self.trans_scale = trans_scale
        self.register_buffer("dummy", torch.tensor(0), persistent=False)

    @property
    def device(self):
        """Return the device of the module.

        :return: The device of the module.
        """
        return self.dummy.device

    @typecheck
    def forward(
        self,
        coords: Float["b n 3"],  # type: ignore
        mask: Bool["b n"] | None = None,  # type: ignore
    ) -> Float["b n 3"]:  # type: ignore
        """Compute the augmented coordinates.

        :param coords: The coordinates to be augmented by a random rotation and translation.
        :param mask: The mask for variable lengths.
        :return: The augmented coordinates.
        """
        batch_size = coords.shape[0]

        # Center the coordinates
        # Accounting for masking

        if exists(mask):
            num = reduce(coords * mask[..., None], "b n c -> b c", "sum")
            den = reduce(mask.float(), "b n -> b", "sum")
            # coords_mean = einx.divide("b c, b -> b 1 c", num, den.clamp(min=1.0))
            coords_mean = num[..., None, :] / den[..., None, None].clamp(min=1.0)
        else:
            coords_mean = coords.mean(dim=1, keepdim=True)

        centered_coords = (
            (coords - coords_mean) * mask[..., None] if exists(mask) else coords - coords_mean
        )

        # Generate random rotation matrix
        rotation_matrix = self._random_rotation_matrix(batch_size)

        # Generate random translation vector
        translation_vector = self._random_translation_vector(batch_size)
        translation_vector = rearrange(translation_vector, "b c -> b 1 c")

        # Apply rotation and translation
        augmented_coords = (
            einsum(centered_coords, rotation_matrix, "b n i, b j i -> b n j") + translation_vector
        )

        return augmented_coords

    @typecheck
    def _random_rotation_matrix(self, batch_size: int) -> Float["b 3 3"]:  # type: ignore
        """Generate a random rotation matrix.

        :param batch_size: The batch size.
        :return: The random rotation matrix.
        """
        # Generate random rotation angles
        angles = torch.rand((batch_size, 3), device=self.device) * 2 * torch.pi

        # Compute sine and cosine of angles
        sin_angles = torch.sin(angles)
        cos_angles = torch.cos(angles)

        # Construct rotation matrix
        eye = torch.eye(3, device=self.device)
        rotation_matrix = repeat(eye, "i j -> b i j", b=batch_size).clone()

        rotation_matrix[:, 0, 0] = cos_angles[:, 0] * cos_angles[:, 1]
        rotation_matrix[:, 0, 1] = (
            cos_angles[:, 0] * sin_angles[:, 1] * sin_angles[:, 2]
            - sin_angles[:, 0] * cos_angles[:, 2]
        )
        rotation_matrix[:, 0, 2] = (
            cos_angles[:, 0] * sin_angles[:, 1] * cos_angles[:, 2]
            + sin_angles[:, 0] * sin_angles[:, 2]
        )
        rotation_matrix[:, 1, 0] = sin_angles[:, 0] * cos_angles[:, 1]
        rotation_matrix[:, 1, 1] = (
            sin_angles[:, 0] * sin_angles[:, 1] * sin_angles[:, 2]
            + cos_angles[:, 0] * cos_angles[:, 2]
        )
        rotation_matrix[:, 1, 2] = (
            sin_angles[:, 0] * sin_angles[:, 1] * cos_angles[:, 2]
            - cos_angles[:, 0] * sin_angles[:, 2]
        )
        rotation_matrix[:, 2, 0] = -sin_angles[:, 1]
        rotation_matrix[:, 2, 1] = cos_angles[:, 1] * sin_angles[:, 2]
        rotation_matrix[:, 2, 2] = cos_angles[:, 1] * cos_angles[:, 2]

        return rotation_matrix

    @typecheck
    def _random_translation_vector(self, batch_size: int) -> Float["b 3"]:  # type: ignore
        """Generate a random translation vector.

        :param batch_size: The batch size.
        :return: The random translation vector.
        """
        # Generate random translation vector
        translation_vector = torch.randn((batch_size, 3), device=self.device) * self.trans_scale
        return translation_vector


# input embedder


class EmbeddedInputs(NamedTuple):
    """The EmbeddedInputs class."""

    single_inputs: Float["b n ds"]  # type: ignore
    single_init: Float["b n ds"]  # type: ignore
    pairwise_init: Float["b n n dp"]  # type: ignore
    atom_feats: Float["b m da"]  # type: ignore
    atompair_feats: Float["b m m dap"]  # type: ignore


class InputFeatureEmbedder(Module):
    """Algorithm 2."""

    def __init__(
        self,
        *,
        dim_atom_inputs,
        dim_atompair_inputs=5,
        atoms_per_window=27,
        dim_atom=128,
        dim_atompair=16,
        dim_token=384,
        dim_single=384,
        dim_pairwise=128,
        dim_additional_token_feats=33,
        num_molecule_types=NUM_MOLECULE_IDS,
        atom_transformer_blocks=3,
        atom_transformer_heads=4,
        atom_transformer_kwargs: dict = dict(),
    ):
        super().__init__()
        self.atoms_per_window = atoms_per_window

        self.to_atom_feats = LinearNoBias(dim_atom_inputs, dim_atom)

        self.to_atompair_feats = LinearNoBias(dim_atompair_inputs, dim_atompair)

        self.atom_repr_to_atompair_feat_cond = nn.Sequential(
            # LayerNorm(dim_atom),
            # LinearNoBias(dim_atom, dim_atompair * 2),
            LayernormLinear(dim_atom, dim_atompair * 2, has_linear_bias=False), 
            nn.ReLU(),
        )

        self.atompair_feats_mlp = nn.Sequential(
            LinearNoBias(dim_atompair, dim_atompair),
            nn.ReLU(),
            LinearNoBias(dim_atompair, dim_atompair),
            nn.ReLU(),
            LinearNoBias(dim_atompair, dim_atompair),
        )

        self.atom_transformer = DiffusionTransformer(
            depth=atom_transformer_blocks,
            heads=atom_transformer_heads,
            dim=dim_atom,
            dim_single_cond=dim_atom,
            dim_pairwise=dim_atompair,
            attn_window_size=atoms_per_window,
            **atom_transformer_kwargs,
        )

        self.atom_feats_to_pooled_token = AtomToTokenPooler(
            dim=dim_atom,
            dim_out=dim_token,
        )

        dim_single_input = dim_token + dim_additional_token_feats

        self.dim_additional_token_feats = dim_additional_token_feats

        self.single_input_to_single_init = LinearNoBias(dim_single_input, dim_single)
        self.single_input_to_pairwise_init = LinearNoBiasThenOuterSum(
            dim_single_input, dim_pairwise
        )

        # this accounts for the `restypes` in the additional molecule features

        self.single_molecule_embed = nn.Embedding(num_molecule_types, dim_single)
        self.pairwise_molecule_embed = nn.Embedding(num_molecule_types, dim_pairwise)

    @typecheck
    def forward(
        self,
        *,
        atom_inputs: Float["b m dai"],  # type: ignore
        atompair_inputs: Float["b m m dapi"] | Float["b nw w1 w2 dapi"],  # type: ignore
        molecule_atom_lens: Int["b n"],  # type: ignore
        molecule_ids: Int["b n"],  # type: ignore
        additional_token_feats: Float["b n {self.dim_additional_token_feats}"] | None = None,  # type: ignore
    ) -> EmbeddedInputs:
        """Compute the embedded inputs.

        :param atom_inputs: The atom inputs tensor.
        :param atompair_inputs: The atom pair inputs tensor.
        :param molecule_atom_lens: The molecule atom lengths tensor.
        :param molecule_ids: The molecule ids tensor.
        :param additional_token_feats: The additional token features tensor.
        :return: The embedded inputs.
        """

        w = self.atoms_per_window

        atom_feats = self.to_atom_feats(atom_inputs)
        atompair_feats = self.to_atompair_feats(atompair_inputs)

        # window the atom pair features before passing to atom encoder and decoder

        is_windowed = atompair_inputs.ndim == 5

        if not is_windowed:
            atompair_feats = full_pairwise_repr_to_windowed(atompair_feats, window_size=w)

        # condition atompair with atom repr
        # print("atom_feats: (K,N)=(128,32)", atom_feats.shape)
        atom_feats_cond = self.atom_repr_to_atompair_feat_cond(atom_feats)

        atom_feats_cond = pad_and_window(atom_feats_cond, w)

        atom_feats_cond_row, atom_feats_cond_col = atom_feats_cond.chunk(2, dim=-1)
        atom_feats_cond_col = concat_previous_window(atom_feats_cond_col, dim_seq=1, dim_window=-2)

        # atompair_feats = einx.add(
        #     "b nw w1 w2 dap, b nw w1 dap", atompair_feats, atom_feats_cond_row
        # )
        # atompair_feats = einx.add(
        #     "b nw w1 w2 dap, b nw w2 dap", atompair_feats, atom_feats_cond_col
        # )
        atompair_feats = atompair_feats + atom_feats_cond_row[..., None, :]
        atompair_feats = atompair_feats + atom_feats_cond_col[..., None, :, :]

        # initial atom transformer

        atom_feats = self.atom_transformer(
            atom_feats,
            single_repr=atom_feats,
            pairwise_repr=atompair_feats,
            use_optimized_evo=None,
        )

        atompair_feats = self.atompair_feats_mlp(atompair_feats) + atompair_feats

        single_inputs = self.atom_feats_to_pooled_token(
            atom_feats=atom_feats,
            molecule_atom_lens=molecule_atom_lens,
        )

        if exists(additional_token_feats):
            single_inputs = torch.cat((single_inputs, additional_token_feats), dim=-1)

        single_init = self.single_input_to_single_init(single_inputs)
        pairwise_init = self.single_input_to_pairwise_init(single_inputs)

        # account for molecule id (restypes), where padding tokens are treated as unknown amino acids

        molecule_ids = torch.where(
            molecule_ids >= 0, molecule_ids, NUM_HUMAN_AMINO_ACIDS
        )  # account for padding

        single_molecule_embed = self.single_molecule_embed(molecule_ids)

        pairwise_molecule_embed = self.pairwise_molecule_embed(molecule_ids)
        # pairwise_molecule_embed = einx.add(
        #     "b i dp, b j dp -> b i j dp", pairwise_molecule_embed, pairwise_molecule_embed
        # )
        pairwise_molecule_embed = (
            pairwise_molecule_embed[..., None, :] + pairwise_molecule_embed[..., None, :, :]
        )

        # sum to single init and pairwise init, equivalent to one-hot in additional residue features

        single_init = single_init + single_molecule_embed
        pairwise_init = pairwise_init + pairwise_molecule_embed

        return EmbeddedInputs(
            single_inputs,
            single_init,
            pairwise_init,
            atom_feats,
            atompair_feats,
        )


# distogram head


class DistogramHead(Module):
    """A module for the distogram head."""

    @typecheck
    def __init__(
        self,
        *,
        dim_pairwise: int = 128,
        num_dist_bins: int = 64,
        dim_atom: int = 128,
        atom_resolution: bool = False,
        checkpoint: bool = False,
    ):
        super().__init__()

        self.to_distogram_logits = nn.Sequential(
            LinearNoBias(dim_pairwise, num_dist_bins),
            Rearrange("b ... l -> b l ..."),
        )

        # atom resolution
        # for now, just embed per atom distances, sum to atom features, project to pairwise dimension

        self.atom_resolution = atom_resolution

        if atom_resolution:
            self.atom_feats_to_pairwise = LinearNoBiasThenOuterSum(dim_atom, dim_pairwise)

        # checkpointing

        self.checkpoint = checkpoint

        # tensor typing

        self.da = dim_atom

    @typecheck
    def to_layers(
        self,
        pairwise_repr: Float["b n n d"],  # type: ignore
        molecule_atom_lens: Int["b n"] | None = None,  # type: ignore
        atom_feats: Float["b m {self.da}"] | None = None,  # type: ignore
    ) -> Float["b l n n"] | Float["b l m m"]:  # type: ignore
        """Compute the distogram logits.

        :param pairwise_repr: The pairwise representation tensor.
        :param molecule_atom_lens: The molecule atom lengths tensor.
        :param atom_feats: The atom features tensor.
        :return: The distogram logits.
        """
        if self.atom_resolution:
            assert exists(molecule_atom_lens)
            assert exists(atom_feats)

            pairwise_repr = batch_repeat_interleave_pairwise(pairwise_repr, molecule_atom_lens)

            pairwise_repr = pairwise_repr + self.atom_feats_to_pairwise(atom_feats)

        logits = self.to_distogram_logits(symmetrize(pairwise_repr))

        return logits

    @typecheck
    def to_checkpointed_layers(
        self,
        pairwise_repr: Float["b n n d"],  # type: ignore
        molecule_atom_lens: Int["b n"] | None = None,  # type: ignore
        atom_feats: Float["b m {self.da}"] | None = None,  # type: ignore
    ) -> Float["b l n n"] | Float["b l m m"]:  # type: ignore
        """Compute the checkpointed distogram logits.

        :param pairwise_repr: The pairwise representation tensor.
        :param molecule_atom_lens: The molecule atom lengths tensor.
        :param atom_feats: The atom features tensor.
        :return: The checkpointed distogram logits.
        """
        wrapped_layers = []
        inputs = (pairwise_repr, molecule_atom_lens, atom_feats)

        def atom_resolution_wrapper(fn):
            @wraps(fn)
            def inner(inputs):
                pairwise_repr, molecule_atom_lens, atom_feats = inputs

                assert exists(molecule_atom_lens)
                assert exists(atom_feats)

                pairwise_repr = batch_repeat_interleave_pairwise(pairwise_repr, molecule_atom_lens)

                pairwise_repr = pairwise_repr + fn(atom_feats)
                return pairwise_repr, molecule_atom_lens, atom_feats

            return inner

        def distogram_wrapper(fn):
            @wraps(fn)
            def inner(inputs):
                pairwise_repr, molecule_atom_lens, atom_feats = inputs
                pairwise_repr = fn(symmetrize(pairwise_repr))
                return pairwise_repr, molecule_atom_lens, atom_feats

            return inner

        if self.atom_resolution:
            wrapped_layers.append(atom_resolution_wrapper(self.atom_feats_to_pairwise))
        wrapped_layers.append(distogram_wrapper(self.to_distogram_logits))

        for layer in wrapped_layers:
            inputs = checkpoint(layer, inputs)

        logits, *_ = inputs
        return logits

    @typecheck
    def forward(
        self,
        pairwise_repr: Float["b n n d"],  # type: ignore
        molecule_atom_lens: Int["b n"] | None = None,  # type: ignore
        atom_feats: Float["b m {self.da}"] | None = None,  # type: ignore
    ) -> Float["b l n n"] | Float["b l m m"]:  # type: ignore
        """Compute the distogram logits.

        :param pairwise_repr: The pairwise representation tensor.
        :param molecule_atom_lens: The molecule atom lengths tensor.
        :param atom_feats: The atom features tensor.
        :return: The distogram logits.
        """
        # going through the layers

        if should_checkpoint(self, pairwise_repr):
            to_layers_fn = self.to_checkpointed_layers
        else:
            to_layers_fn = self.to_layers

        logits = to_layers_fn(
            pairwise_repr=pairwise_repr,
            molecule_atom_lens=molecule_atom_lens,
            atom_feats=atom_feats,
        )

        return logits


# confidence head


class ConfidenceHeadLogits(NamedTuple):
    """The ConfidenceHeadLogits class."""

    pae: Float["b pae n n"] | None  # type: ignore
    pde: Float["b pde n n"]  # type: ignore
    plddt: Float["b plddt m"]  # type: ignore
    resolved: Float["b 2 m"]  # type: ignore
    affinity: List[Float[" *"]] | None  # type: ignore

    @typecheck
    def to(self, device: str) -> ConfidenceHeadLogits:
        """Move the tensors to the specified device.

        :param device: The device to move the tensors to.
        :return: A new ConfidenceHeadLogits instance with tensors moved to the specified device.
        """
        pae = self.pae.to(device) if exists(self.pae) else None
        pde = self.pde.to(device)
        plddt = self.plddt.to(device)
        resolved = self.resolved.to(device)
        affinity = [aff.to(device) for aff in self.affinity] if exists(self.affinity) else None

        return ConfidenceHeadLogits(
            pae=pae, pde=pde, plddt=plddt, resolved=resolved, affinity=affinity
        )

    @typecheck
    def repeat(self, n: int) -> ConfidenceHeadLogits:
        """Repeat the tensors along the `b` dimension.

        :param n: The number of times to repeat the tensors.
        :return: A new ConfidenceHeadLogits instance with tensors repeated along the `b` dimension.
        """
        pae = self.pae.repeat(n, 1, 1, 1) if exists(self.pae) else None
        pde = self.pde.repeat(n, 1, 1, 1)
        plddt = self.plddt.repeat(n, 1, 1)
        resolved = self.resolved.repeat(n, 1, 1)
        affinity = self.affinity * n if exists(self.affinity) else None

        return ConfidenceHeadLogits(
            pae=pae, pde=pde, plddt=plddt, resolved=resolved, affinity=affinity
        )

    @typecheck
    def rank_order_confidence_head_logits(logits: ConfidenceHeadLogits, sorted_indices: Int[" b"]) -> ConfidenceHeadLogits:  # type: ignore
        """Rank-order the tensors in ConfidenceHeadLogits along the `b` dimension using the sorted
        index list.

        :param logits (ConfidenceHeadLogits): The original ConfidenceHeadLogits instance.
        :param sorted_indices (torch.tensor): The sorted index list.

        :return: A new ConfidenceHeadLogits instance with tensors reordered along the `b` dimension.
        """
        pae = logits.pae[sorted_indices] if exists(logits.pae) else None
        pde = logits.pde[sorted_indices]
        plddt = logits.plddt[sorted_indices]
        resolved = logits.resolved[sorted_indices]
        affinity = (
            [logits.affinity[sorted_id] for sorted_id in sorted_indices]
            if exists(logits.affinity)
            else None
        )

        return ConfidenceHeadLogits(
            pae=pae, pde=pde, plddt=plddt, resolved=resolved, affinity=affinity
        )


class MegaFoldLogits(NamedTuple):
    """The MegaFoldLogits class."""

    pae: Float["b pae n n"] | None  # type: ignore
    pde: Float["b pde n n"]  # type: ignore
    plddt: Float["b plddt m"]  # type: ignore
    resolved: Float["b 2 m"]  # type: ignore
    affinity: List[Float[" *"]] | None  # type: ignore
    distance: Float["b dist m m"] | Float["b dist n n"] | None  # type: ignore


class ConfidenceHead(Module):
    """Algorithm 31."""

    @typecheck
    def __init__(
        self,
        *,
        dim_single_inputs,
        dim_atom=128,
        atompair_dist_bins: List[float],
        dim_single=384,
        dim_pairwise=128,
        num_plddt_bins=50,
        num_pde_bins=64,
        num_pae_bins=64,
        pairformer_depth=4,
        pairformer_kwargs: dict = dict(),
        affinity_dropout: float = 0.01,
        checkpoint=False,
    ):  # type: ignore
        super().__init__()

        atompair_dist_bins = tensor(atompair_dist_bins)

        self.register_buffer("atompair_dist_bins", atompair_dist_bins)

        num_dist_bins = atompair_dist_bins.shape[-1]
        self.num_dist_bins = num_dist_bins

        self.dist_bin_pairwise_embed = nn.Embedding(num_dist_bins, dim_pairwise)
        self.single_inputs_to_pairwise = LinearNoBiasThenOuterSum(dim_single_inputs, dim_pairwise)

        # pairformer stack

        self.pairformer_stack = PairformerStack(
            dim_single=dim_single,
            dim_pairwise=dim_pairwise,
            depth=pairformer_depth,
            checkpoint=checkpoint,
            **pairformer_kwargs,
        )

        # to predictions

        self.to_pae_logits = nn.Sequential(
            LinearNoBias(dim_pairwise, num_pae_bins),
            Rearrange("b ... l -> b l ..."),
        )

        self.to_pde_logits = nn.Sequential(
            LinearNoBias(dim_pairwise, num_pde_bins),
            Rearrange("b ... l -> b l ..."),
        )

        self.to_plddt_logits = nn.Sequential(
            LinearNoBias(dim_single, num_plddt_bins),
            Rearrange("b ... l -> b l ..."),
        )

        self.to_resolved_logits = nn.Sequential(
            LinearNoBias(dim_single, 2), Rearrange("b ... l -> b l ...")
        )

        self.ligand_affinity_pooling = SumPooling(learnable=True, hidden_dim=dim_single)
        self.to_affinity_logits = nn.Sequential(
            nn.Dropout(affinity_dropout),
            LinearNoBias(dim_single, 1),
        )

        # atom resolution

        self.atom_feats_to_single = LinearNoBias(dim_atom, dim_single)

        # tensor typing

        self.da = dim_atom

        # NOTE: Protenix reports new weights and initializations as follows:

        self.single_inputs_embed = LinearNoBias(dim_single_inputs, dim_single)
        # self.single_repr_embed = LinearNoBias(dim_single, dim_single)
        # self.single_repr_norm = LayerNorm(dim_single)
        self.single_repr_embed = LayernormLinear(dim_single, dim_single, has_linear_bias=False)
        nn.init.zeros_(self.single_repr_embed.linear_weight)
        
        # self.pairwise_repr_embed = LinearNoBias(dim_pairwise, dim_pairwise)
        # self.pairwise_repr_norm = LayerNorm(dim_pairwise)
        self.pairwise_repr_embed = LayernormLinear(dim_pairwise, dim_pairwise, has_linear_bias=False)
        nn.init.zeros_(self.pairwise_repr_embed.linear_weight)

        # zero init for output layer (before softmax)
        nn.init.zeros_(self.to_pae_logits[0].weight)
        nn.init.zeros_(self.to_pde_logits[0].weight)
        nn.init.zeros_(self.to_plddt_logits[0].weight)
        nn.init.zeros_(self.to_resolved_logits[0].weight)
        nn.init.zeros_(self.to_affinity_logits[1].weight)

        # zero init for trunk embedding input layer
        # nn.init.zeros_(self.single_repr_embed.weight)
        # nn.init.zeros_(self.pairwise_repr_embed.weight)

    @typecheck
    def forward(
        self,
        *,
        single_inputs_repr: Float["b n dsi"],  # type: ignore
        single_repr: Float["b n ds"],  # type: ignore
        pairwise_repr: Float["b n n dp"],  # type: ignore
        pred_atom_pos: Float["b m 3"],  # type: ignore
        atom_feats: Float["b m {self.da}"],  # type: ignore
        molecule_atom_indices: Int["b n"],  # type: ignore
        molecule_atom_lens: Int["b n"],  # type: ignore
        is_ligand_atom_res_idx: Int["b m"],  # type: ignore
        mask: Bool["b n"] | None = None,  # type: ignore
        num_ligands: List[int] | None = None,
        return_pae_logits: bool = True,
        return_affinity_logits: bool = True,
        **kwargs,
    ) -> ConfidenceHeadLogits:
        """Compute the confidence head logits.

        :param single_inputs_repr: The single inputs representation tensor.
        :param single_repr: The single representation tensor.
        :param pairwise_repr: The pairwise representation tensor.
        :param pred_atom_pos: The predicted atom positions tensor.
        :param atom_feats: The atom features tensor.
        :param molecule_atom_indices: The molecule atom indices tensor.
        :param molecule_atom_lens: The molecule atom lengths tensor.
        :param is_ligand_atom_res_idx: The ligand atom residue indices tensor (i.e., unique chain-
            residue IDs for ligand atoms and -1 for all other atoms).
        :param mask: The mask tensor.
        :param num_ligands: The number of (fragment) ligands in each batch element.
        :param return_pae_logits: Whether to return the predicted aligned error (PAE) logits.
        :param return_affinity_logits: Whether to return the predicted binding affinity logits.
        :param kwargs: Additional keyword arguments for the attention computation.
        :return: The confidence head logits.
        """
        dtype = single_repr.dtype
        batch_size = len(single_repr)

        # pairwise_repr = pairwise_repr + self.single_inputs_to_pairwise(single_inputs_repr)

        # NOTE: Protenix introduces new weights instead as follows:

        # print("single_repr: (K,N)=(384,384)", single_repr.shape)
        # single_repr = self.single_repr_embed(self.single_repr_norm(single_repr)) + self.single_inputs_embed(single_inputs_repr)
        single_repr = self.single_repr_embed(single_repr) + self.single_inputs_embed(single_inputs_repr)
        
        # print("pairwise_repr: (K,N)=(128,128)", pairwise_repr.shape)
        # pairwise_repr = self.pairwise_repr_embed(self.pairwise_repr_norm(pairwise_repr)) + self.single_inputs_to_pairwise(single_inputs_repr)
        pairwise_repr = self.pairwise_repr_embed(pairwise_repr) + self.single_inputs_to_pairwise(single_inputs_repr)


        # pluck out the representative atoms for non-atomic resolution confidence head outputs

        # pred_molecule_pos = einx.get_at('b [m] c, b n -> b n c', pred_atom_pos, molecule_atom_indices)

        molecule_atom_indices = repeat(
            molecule_atom_indices, "b n -> b n c", c=pred_atom_pos.shape[-1]
        )
        pred_molecule_pos = pred_atom_pos.gather(1, molecule_atom_indices)

        # interatomic distances - embed and add to pairwise

        intermolecule_dist = torch.cdist(
            pred_molecule_pos.float(), pred_molecule_pos.float(), p=2
        ).type(dtype)

        dist_bin_indices = distance_to_dgram(
            intermolecule_dist, self.atompair_dist_bins, return_labels=True
        )
        pairwise_repr = pairwise_repr + self.dist_bin_pairwise_embed(dist_bin_indices)

        # pairformer stack

        single_repr, pairwise_repr = self.pairformer_stack(
            single_repr=single_repr,
            pairwise_repr=pairwise_repr,
            mask=mask,
            **kwargs,
        )

        # handle atom level resolution

        atom_single_repr = batch_repeat_interleave(single_repr, molecule_atom_lens)

        atom_single_repr = atom_single_repr + self.atom_feats_to_single(atom_feats)

        # to logits

        pde_logits = self.to_pde_logits(symmetrize(pairwise_repr))

        plddt_logits = self.to_plddt_logits(atom_single_repr)
        resolved_logits = self.to_resolved_logits(atom_single_repr)

        # they only incorporate pae at some stage of training

        pae_logits = None

        if return_pae_logits:
            pae_logits = self.to_pae_logits(pairwise_repr)

        # maybe predict binding affinity values for each ligand

        affinity_logits = None
        num_ligands = default(num_ligands, [1] * batch_size)

        if return_affinity_logits:
            is_ligand_atom = is_ligand_atom_res_idx != -1

            # aggregate the ligand atom affinity logits for each (fragment) ligand

            affinity_logits = []

            for batch_idx in range(batch_size):
                is_ligand_atom_ = is_ligand_atom[batch_idx]

                ligand_atom_single_repr_ = atom_single_repr[batch_idx][is_ligand_atom_]
                ligand_atom_res_idx_ = is_ligand_atom_res_idx[batch_idx][is_ligand_atom_]
                num_ligands_ = num_ligands[batch_idx]

                # NOTE: we need to convert arbitrary chain-residue ID hashes into unique consecutive IDs (unsorted) starting from 0
                remapped_ligand_atom_res_idx_ = torch.unique_consecutive(
                    ligand_atom_res_idx_, return_inverse=True
                )[-1]

                pooled_affinity_single_repr = self.ligand_affinity_pooling(
                    ligand_atom_single_repr_,
                    remapped_ligand_atom_res_idx_,
                    num_ligands_,
                )
                pooled_affinity_logits = self.to_affinity_logits(
                    pooled_affinity_single_repr
                ).squeeze(-1)

                affinity_logits.append(pooled_affinity_logits)

        # return all logits

        return ConfidenceHeadLogits(
            pae_logits, pde_logits, plddt_logits, resolved_logits, affinity_logits
        )


class ConfidenceScore(NamedTuple):
    """The ConfidenceScore class."""

    plddt: Float["b m"]  # type: ignore
    ptm: Float[" b"]  # type: ignore
    iptm: Float[" b"] | None  # type: ignore


class ComputeConfidenceScore(Module):
    """Compute confidence score."""

    @typecheck
    def __init__(
        self,
        pae_breaks: Float[" pae_break"] = torch.arange(0, 31.5, 0.5),  # type: ignore
        pde_breaks: Float[" pde_break"] = torch.arange(0, 31.5, 0.5),  # type: ignore
        eps: float = 1e-8,
    ):
        super().__init__()
        self.eps = eps
        self.register_buffer("pae_breaks", pae_breaks)
        self.register_buffer("pde_breaks", pde_breaks)

    @typecheck
    def _calculate_bin_centers(
        self,
        breaks: Float[" breaks"],  # type: ignore
    ) -> Float[" breaks+1"]:  # type: ignore
        """Calculate bin centers from bin edges.

        :param breaks: [num_bins -1] bin edges
        :return: bin_centers: [num_bins] bin centers
        """

        step = breaks[1] - breaks[0]

        bin_centers = breaks + step / 2
        last_bin_center = breaks[-1] + step

        bin_centers = torch.concat([bin_centers, last_bin_center.unsqueeze(0)])

        return bin_centers

    @typecheck
    def forward(
        self,
        confidence_head_logits: ConfidenceHeadLogits,
        asym_id: Int["b n"],  # type: ignore
        has_frame: Bool["b n"],  # type: ignore
        ptm_residue_weight: Float["b n"] | None = None,  # type: ignore
        multimer_mode: bool = True,
    ) -> ConfidenceScore:
        """Main function to compute confidence score.

        :param confidence_head_logits: ConfidenceHeadLogits
        :param asym_id: [b n] asym_id of each residue
        :param has_frame: [b n] has_frame of each residue
        :param ptm_residue_weight: [b n] weight of each residue
        :param multimer_mode: bool
        :return: Confidence score
        """
        plddt = self.compute_plddt(confidence_head_logits.plddt)

        # Section 5.9.1 equation 17
        ptm = self.compute_ptm(
            confidence_head_logits.pae,
            asym_id,
            has_frame,
            ptm_residue_weight,
            interface=False,
        )

        iptm = None

        if multimer_mode:
            # Section 5.9.2 equation 18
            iptm = self.compute_ptm(
                confidence_head_logits.pae,
                asym_id,
                has_frame,
                ptm_residue_weight,
                interface=True,
            )

        confidence_score = ConfidenceScore(plddt=plddt, ptm=ptm, iptm=iptm)
        return confidence_score

    @staticmethod
    @typecheck
    def compute_plddt(
        logits: Float["b plddt m"],  # type: ignore
    ) -> Float["b m"]:  # type: ignore
        """Compute plDDT from logits.

        :param logits: [b c m] logits
        :return: [b m] plDDT
        """
        logits = rearrange(logits, "b plddt m -> b m plddt")
        num_bins = logits.shape[-1]
        bin_width = 1.0 / num_bins
        bin_centers = torch.arange(
            0.5 * bin_width, 1.0, bin_width, dtype=logits.dtype, device=logits.device
        )
        probs = F.softmax(logits, dim=-1)

        predicted_lddt = einsum(probs, bin_centers, "b m plddt, plddt -> b m")
        return predicted_lddt * 100

    @typecheck
    def compute_ptm(
        self,
        pae_logits: Float["b pae n n"],  # type: ignore
        asym_id: Int["b n"],  # type: ignore
        has_frame: Bool["b n"],  # type: ignore
        residue_weights: Float["b n"] | None = None,  # type: ignore
        interface: bool = False,
        compute_chain_wise_iptm: bool = False,
    ) -> Float[" b"] | Tuple[Float["b chains chains"], Bool["b chains chains"], Int["b chains"]]:  # type: ignore
        """Compute pTM from logits.

        :param pae_logits: [b c n n] logits
        :param asym_id: [b n] asym_id of each residue
        :param has_frame: [b n] has_frame of each residue
        :param residue_weights: [b n] weight of each residue
        :param interface: bool
        :param compute_chain_wise_iptm: bool
        :return: pTM
        """

        num_batch, *_, num_res, device, dtype = (
            *pae_logits.shape,
            pae_logits.device,
            pae_logits.dtype,
        )

        if not_exists(residue_weights):
            residue_weights = torch.ones_like(has_frame)

        residue_weights = (residue_weights * has_frame).type(dtype)

        pae_logits = rearrange(pae_logits, "b c i j -> b i j c")

        bin_centers = self._calculate_bin_centers(self.pae_breaks).to(device).type(dtype)

        num_frame = torch.sum(has_frame, dim=-1)
        # Clip num_frame to avoid negative/undefined d0.
        clipped_num_frame = torch.clamp(num_frame, min=19)

        # Compute d_0(num_frame) as defined by TM-score, eqn. (5) in Yang & Skolnick
        # "Scoring function for automated assessment of protein structure template
        # quality", 2004: http://zhanglab.ccmb.med.umich.edu/papers/2004_3.pdf
        d0 = (1.24 * (clipped_num_frame - 15) ** (1.0 / 3) - 1.8).type(dtype)

        # TM-Score term for every bin. [num_batch, num_bins]
        tm_per_bin = 1.0 / (1 + torch.square(bin_centers[None, :]) / torch.square(d0[..., None]))

        # Convert logits to probs.
        probs = F.softmax(pae_logits, dim=-1)

        # E_distances tm(distance).
        predicted_tm_term = einsum(probs, tm_per_bin, "b i j pae, b pae -> b i j ")

        if compute_chain_wise_iptm:
            # chain_wise_iptm[b, i, j]: iptm of chain i and chain j in batch b

            # get the max num_chains across batch
            unique_chains = [list(dict.fromkeys(asym.tolist())) for asym in asym_id]
            max_chains = max(len(chains) for chains in unique_chains)

            chain_wise_iptm = torch.zeros((num_batch, max_chains, max_chains), device=device)
            chain_wise_iptm_mask = torch.zeros_like(chain_wise_iptm).bool()

            for b in range(num_batch):
                for i, chain_i in enumerate(unique_chains[b]):
                    for j, chain_j in enumerate(unique_chains[b]):
                        if chain_i != chain_j:
                            mask_i = asym_id[b] == chain_i
                            mask_j = asym_id[b] == chain_j

                            # pair_mask = einx.multiply("i, j -> i j", mask_i, mask_j)
                            pair_mask = mask_i[:, None] * mask_j[None, :]

                            # pair_residue_weights = pair_mask * einx.multiply(
                            #     "... i, ... j -> ... i j", residue_weights[b], residue_weights[b]
                            # )
                            pair_residue_weights = pair_mask * (
                                residue_weights[b][..., None] * residue_weights[b][..., None, :]
                            )

                            if pair_residue_weights.sum() == 0:
                                # chain i or chain j does not have any valid frame
                                continue

                            normed_residue_mask = pair_residue_weights / (
                                self.eps + torch.sum(pair_residue_weights, dim=-1, keepdims=True)
                            )

                            masked_predicted_tm_term = predicted_tm_term[b] * pair_mask

                            per_alignment = torch.sum(
                                masked_predicted_tm_term * normed_residue_mask, dim=-1
                            )
                            weighted_argmax = (residue_weights[b] * per_alignment).argmax()
                            chain_wise_iptm[b, i, j] = per_alignment[weighted_argmax]
                            chain_wise_iptm_mask[b, i, j] = True

            return chain_wise_iptm, chain_wise_iptm_mask, torch.tensor(unique_chains)

        else:
            pair_mask = torch.ones(size=(num_batch, num_res, num_res), device=device).bool()
            if interface:
                # pair_mask *= einx.not_equal("b i, b j -> b i j", asym_id, asym_id)
                pair_mask *= asym_id[..., None] != asym_id[..., None, :]

            predicted_tm_term *= pair_mask

            # pair_residue_weights = pair_mask * einx.multiply(
            #     "b i, b j -> b i j", residue_weights, residue_weights
            # )
            pair_residue_weights = pair_mask * (
                residue_weights[..., None] * residue_weights[..., None, :]
            )
            normed_residue_mask = pair_residue_weights / (
                self.eps + torch.sum(pair_residue_weights, dim=-1, keepdims=True)
            )

            per_alignment = torch.sum(predicted_tm_term * normed_residue_mask, dim=-1)
            weighted_argmax = (residue_weights * per_alignment).argmax(dim=-1)
            return per_alignment[torch.arange(num_batch), weighted_argmax]

    @typecheck
    def compute_pde(
        self,
        pde_logits: Float["b pde n n"],  # type: ignore
        tok_repr_atm_mask: Bool["b n"],  # type: ignore
    ) -> Float["b n n"]:  # type: ignore
        """Compute PDE from logits."""

        pde_logits = rearrange(pde_logits, "b pde i j -> b i j pde")
        bin_centers = self._calculate_bin_centers(self.pde_breaks)
        probs = F.softmax(pde_logits, dim=-1)

        pde = einsum(probs, bin_centers, "b i j pde, pde -> b i j")

        mask = to_pairwise_mask(tok_repr_atm_mask)

        pde = pde * mask
        return pde


class ComputeClash(Module):
    """Compute clash score."""

    def __init__(
        self,
        atom_clash_dist: float = 1.1,
        chain_clash_count: int = 100,
        chain_clash_ratio: float = 0.5,
    ):
        super().__init__()
        self.atom_clash_dist = atom_clash_dist
        self.chain_clash_count = chain_clash_count
        self.chain_clash_ratio = chain_clash_ratio

    def compute_has_clash(
        self,
        atom_pos: Float["m 3"],  # type: ignore
        asym_id: Int[" n"],  # type: ignore
        indices: Int[" m"],  # type: ignore
        valid_indices: Bool[" m"],  # type: ignore
        atom_is_molecule_types: Bool[f"m {IS_MOLECULE_TYPES}"],  # type: ignore
    ) -> Bool[""]:  # type: ignore
        """Compute if there is a clash in the chain.

        :param atom_pos: [m 3] atom positions
        :param asym_id: [n] asym_id of each residue
        :param indices: [m] indices
        :param valid_indices: [m] valid indices
        :param atom_is_molecule_types: [m 5] atom is molecule types
        :return: [1] has_clash
        """
        dtype = atom_pos.dtype

        # Section 5.9.2

        atom_pos = atom_pos[valid_indices]
        atom_asym_id = asym_id[indices][valid_indices]

        unique_chains = list(dict.fromkeys(atom_asym_id.tolist()))
        for i in range(len(unique_chains)):
            for j in range(i + 1, len(unique_chains)):
                chain_i, chain_j = unique_chains[i], unique_chains[j]

                mask_i = (atom_asym_id == chain_i) & (
                    atom_is_molecule_types[..., IS_BIOMOLECULE_INDICES].any(dim=-1)
                )
                mask_j = (atom_asym_id == chain_j) & (
                    atom_is_molecule_types[..., IS_BIOMOLECULE_INDICES].any(dim=-1)
                )

                if not (mask_i.any() and mask_j.any()):
                    continue

                chain_i_len = mask_i.sum()
                chain_j_len = mask_j.sum()
                assert min(chain_i_len, chain_j_len) > 0

                chain_pair_dist = torch.cdist(
                    atom_pos[mask_i].float(), atom_pos[mask_j].float(), p=2
                ).type(dtype)
                chain_pair_clash = chain_pair_dist < self.atom_clash_dist
                clashes = chain_pair_clash.sum()
                has_clash = (clashes > self.chain_clash_count) or (
                    clashes / min(chain_i_len, chain_j_len) > self.chain_clash_ratio
                )

                if has_clash:
                    return torch.tensor(True, dtype=torch.bool, device=atom_pos.device)

        return torch.tensor(False, dtype=torch.bool, device=atom_pos.device)

    def forward(
        self,
        atom_pos: Float["b m 3"] | Float["m 3"],  # type: ignore
        atom_mask: Bool["b m"] | Bool[" m"],  # type: ignore
        molecule_atom_lens: Int["b n"] | Int[" n"],  # type: ignore
        asym_id: Int["b n"] | Int[" n"],  # type: ignore
        is_molecule_types: Bool[f"b n {IS_MOLECULE_TYPES}"],  # type: ignore
    ) -> Bool[" b"]:  # type: ignore
        """Compute if there is a clash in the chain.

        :param atom_pos: [b m 3] atom positions
        :param atom_mask: [b m] atom mask
        :param molecule_atom_lens: [b n] molecule atom lens
        :param asym_id: [b n] asym_id of each residue
        :param is_molecule_types: [b n 5] is_molecule_types
        :return: [b] has_clash
        """

        if atom_pos.ndim == 2:
            atom_pos = atom_pos.unsqueeze(0)
            molecule_atom_lens = molecule_atom_lens.unsqueeze(0)
            asym_id = asym_id.unsqueeze(0)
            atom_mask = atom_mask.unsqueeze(0)

        device = atom_pos.device
        batch_size, seq_len = asym_id.shape

        indices = torch.arange(seq_len, device=device)

        indices = repeat(indices, "n -> b n", b=batch_size)
        valid_indices = torch.ones_like(indices).bool()

        # valid_indices at padding position has value False
        indices = batch_repeat_interleave(indices, molecule_atom_lens)
        valid_indices = batch_repeat_interleave(valid_indices, molecule_atom_lens)

        if exists(atom_mask):
            valid_indices = valid_indices * atom_mask

        atom_is_molecule_types = batch_repeat_interleave(is_molecule_types, molecule_atom_lens)

        has_clash = []
        for b in range(batch_size):
            has_clash.append(
                self.compute_has_clash(
                    atom_pos[b],
                    asym_id[b],
                    indices[b],
                    valid_indices[b],
                    atom_is_molecule_types[b],
                )
            )

        has_clash = torch.stack(has_clash)
        return has_clash


class ComputeRankingScore(Module):
    """Compute ranking score."""

    def __init__(
        self,
        eps: float = 1e-8,
        score_iptm_weight: float = 0.8,
        score_ptm_weight: float = 0.2,
        score_disorder_weight: float = 0.5,
        score_clash_weight: float = 100.0,
        score_full_complex_weight: float = 0.9,
        score_modified_residue_weight: float = 0.1,
        rasa_norm_constant: float = 200.0,  # NOTE: must match `fibonacci_sphere_n` in `ComputeModelSelectionScore`
    ):
        super().__init__()
        self.eps = eps
        self.compute_clash = ComputeClash()
        self.compute_confidence_score = ComputeConfidenceScore(eps=eps)
        self.compute_model_selection_score = ComputeModelSelectionScore(eps=eps)

        self.score_iptm_weight = score_iptm_weight
        self.score_ptm_weight = score_ptm_weight
        self.score_disorder_weight = score_disorder_weight
        self.score_clash_weight = score_clash_weight
        self.score_full_complex_weight = score_full_complex_weight
        self.score_modified_residue_weight = score_modified_residue_weight

        self.rasa_norm_constant = rasa_norm_constant

    @typecheck
    def compute_disorder(
        self,
        atom_rasa: Float["b m"],  # type: ignore
        atom_mask: Bool["b m"],  # type: ignore
        atom_is_molecule_types: Bool[f"b m {IS_MOLECULE_TYPES}"],  # type: ignore
        rasa_threshold: float = 0.581,
    ) -> Float[" b"]:  # type: ignore
        """Compute disorder score.

        :param atom_rasa: [b m] atom rasa
        :param atom_mask: [b m] atom mask
        :param atom_is_molecule_types: [b m 2] atom is molecule types
        :param rasa_threshold: float
        :return: [b] disorder
        """
        is_protein_mask = atom_is_molecule_types[..., IS_PROTEIN_INDEX]
        mask = atom_mask * is_protein_mask

        disorder = ((atom_rasa > rasa_threshold) * mask).sum(dim=-1) / (self.eps + mask.sum(dim=1))
        return disorder

    @typecheck
    def compute_full_complex_metric(
        self,
        confidence_head_logits: ConfidenceHeadLogits,
        asym_id: Int["b n"],  # type: ignore
        has_frame: Bool["b n"],  # type: ignore
        molecule_atom_lens: Int["b n"],  # type: ignore
        molecule_ids: Int["b n"],  # type: ignore
        atom_pos: Float["b m 3"],  # type: ignore
        atom_mask: Bool["b m"],  # type: ignore
        is_molecule_types: Bool[f"b n {IS_MOLECULE_TYPES}"],  # type: ignore
        return_confidence_score: bool = False,
    ) -> Float[" b"] | Tuple[Float[" b"], Tuple[ConfidenceScore, Bool[" b"]]]:  # type: ignore
        """Compute full complex metric.

        :param confidence_head_logits: ConfidenceHeadLogits
        :param asym_id: [b n] asym_id of each residue
        :param has_frame: [b n] has_frame of each residue
        :param molecule_atom_lens: [b n] molecule atom lens
        :param molecule_ids: [b n] molecule ids
        :param atom_pos: [b m 3] atom positions
        :param atom_mask: [b m] atom mask
        :param is_molecule_types: [b n 2] is_molecule_types
        :return: [b] score
        """

        # Section 5.9.3.1

        device, dtype = atom_pos.device, atom_pos.dtype
        batch_size, seq_len = asym_id.shape

        indices = torch.arange(seq_len, device=device)

        indices = repeat(indices, "n -> b n", b=batch_size)
        valid_indices = torch.ones_like(indices).bool()

        # valid_indices at padding position has value False
        indices = batch_repeat_interleave(indices, molecule_atom_lens)
        valid_indices = batch_repeat_interleave(valid_indices, molecule_atom_lens)

        # broadcast is_molecule_types to atom

        # einx.get_at('b [n] is_type, b m -> b m is_type', is_molecule_types, indices)

        indices = repeat(indices, "b m -> b m is_type", is_type=is_molecule_types.shape[-1])
        atom_is_molecule_types = is_molecule_types.gather(1, indices) * valid_indices[..., None]

        confidence_score = self.compute_confidence_score(
            confidence_head_logits, asym_id, has_frame, multimer_mode=True
        )

        rasa = (
            torch.stack(
                [
                    self.compute_model_selection_score._compute_unresolved_rasa(*args)
                    for args in zip(
                        [None] * len(asym_id),
                        [None] * len(asym_id),
                        asym_id,
                        molecule_ids,
                        molecule_atom_lens,
                        atom_pos,
                        atom_mask,
                        [False] * len(asym_id),
                    )
                ]
            )
            / self.rasa_norm_constant
        )

        atom_rasa = batch_repeat_interleave(rasa, molecule_atom_lens)
        disorder = self.compute_disorder(atom_rasa, atom_mask, atom_is_molecule_types).type(dtype)

        has_clash = self.compute_clash(
            atom_pos,
            atom_mask,
            molecule_atom_lens,
            asym_id,
            is_molecule_types,
        ).type(dtype)

        # Section 5.9.3 equation 19
        weighted_score = (
            confidence_score.iptm * self.score_iptm_weight
            + confidence_score.ptm * self.score_ptm_weight
            + disorder * self.score_disorder_weight
            - has_clash * self.score_clash_weight
        )

        if not return_confidence_score:
            return weighted_score

        return weighted_score, (confidence_score, has_clash)

    @typecheck
    def compute_single_chain_metric(
        self,
        confidence_head_logits: ConfidenceHeadLogits,
        asym_id: Int["b n"],  # type: ignore
        has_frame: Bool["b n"],  # type: ignore
    ) -> Float[" b"]:  # type: ignore
        """Compute single chain metric.

        :param confidence_head_logits: ConfidenceHeadLogits
        :param asym_id: [b n] asym_id of each residue
        :param has_frame: [b n] has_frame of each residue
        :return: [b] score
        """

        # Section 5.9.3.2

        confidence_score = self.compute_confidence_score(
            confidence_head_logits, asym_id, has_frame, multimer_mode=False
        )

        score = confidence_score.ptm
        return score

    @typecheck
    def compute_interface_metric(
        self,
        confidence_head_logits: ConfidenceHeadLogits,
        asym_id: Int["b n"],  # type: ignore
        has_frame: Bool["b n"],  # type: ignore
        interface_chains: List,
    ) -> Float[" b"]:  # type: ignore
        """Compute interface metric.

        :param confidence_head_logits: ConfidenceHeadLogits
        :param asym_id: [b n] asym_id of each residue
        :param has_frame: [b n] has_frame of each residue
        :param interface_chains: List
        :return: [b] score
        """

        batch = asym_id.shape[0]

        # Section 5.9.3.3

        # interface_chains: List[chain_id_tuple]
        # chain_id_tuple:
        #  - correspond to the asym_id of one or two chain
        #  - compute R(C) for one chain
        #  - compute 1/2 [R(A) + R(b)] for two chain

        (
            chain_wise_iptm,
            chain_wise_iptm_mask,
            unique_chains,
        ) = self.compute_confidence_score.compute_ptm(
            confidence_head_logits.pae, asym_id, has_frame, compute_chain_wise_iptm=True
        )

        # Section 5.9.3 equation 20
        interface_metric = torch.zeros(batch).type_as(chain_wise_iptm)

        # R(c) = mean(Mij) restricted to i = c or j = c
        masked_chain_wise_iptm = chain_wise_iptm * chain_wise_iptm_mask
        iptm_sum = masked_chain_wise_iptm + rearrange(masked_chain_wise_iptm, "b i j -> b j i")
        iptm_count = chain_wise_iptm_mask.int() + rearrange(
            chain_wise_iptm_mask.int(), "b i j -> b j i"
        )

        for b, chains in enumerate(interface_chains):
            for chain in chains:
                idx = unique_chains[b].tolist().index(chain)
                interface_metric[b] += iptm_sum[b, idx].sum() / iptm_count[b, idx].sum().clamp(
                    min=1
                )
            interface_metric[b] /= len(chains)
        return interface_metric

    @typecheck
    def compute_modified_residue_score(
        self,
        confidence_head_logits: ConfidenceHeadLogits,
        atom_mask: Bool["b m"],  # type: ignore
        atom_is_modified_residue: Bool["b m"],  # type: ignore
    ) -> Float[" b"]:  # type: ignore
        """Compute modified residue score.

        :param confidence_head_logits: ConfidenceHeadLogits
        :param atom_mask: [b m] atom mask
        :param atom_is_modified_residue: [b m] atom is modified residue
        :return: [b] score
        """

        # Section 5.9.3.4

        plddt = self.compute_confidence_score.compute_plddt(confidence_head_logits.plddt)

        mask = atom_is_modified_residue * atom_mask
        plddt_mean = masked_average(plddt, mask, dim=-1, eps=self.eps)

        return plddt_mean

    @typecheck
    def compute_score(
        self,
        confidence_head_logits: ConfidenceHeadLogits,
        asym_id: Int["b n"],  # type: ignore
        has_frame: Bool["b n"],  # type: ignore
        molecule_atom_lens: Int["b n"],  # type: ignore
        molecule_ids: Int["b n"],  # type: ignore
        is_molecule_types: Bool[f"b n {IS_MOLECULE_TYPES}"],  # type: ignore
        atom_pos: Float["b m 3"],  # type: ignore
        atom_mask: Bool["b m"],  # type: ignore
        atom_is_modified_residue: Bool["b m"],  # type: ignore
    ) -> Float[" b"]:  # type: ignore
        """Compute ranking score.

        :param confidence_head_logits: ConfidenceHeadLogits
        :param asym_id: [b n] asym_id of each residue
        :param has_frame: [b n] has_frame of each residue
        :param molecule_atom_lens: [b n] molecule atom lens
        :param is_molecule_types: [b n 2] is_molecule_types
        :param molecule_ids: [b n] molecule ids
        :param atom_pos: [b m 3] atom positions
        :param atom_mask: [b m] atom mask
        :param atom_is_modified_residue: [b m] atom is modified residue
        :return: [b] ranking score
        """
        full_complex_metric = self.compute_full_complex_metric(
            confidence_head_logits,
            asym_id,
            has_frame,
            molecule_atom_lens,
            molecule_ids,
            atom_pos,
            atom_mask,
            is_molecule_types,
        )

        modified_residue_score = self.compute_modified_residue_score(
            confidence_head_logits,
            atom_mask,
            atom_is_modified_residue,
        )

        # NOTE: Here, we improvise by incorporating the modified residue score in a weighted average,
        # since the AF3 supplement does not provide the exact formula for the final ranking score.
        score = (
            full_complex_metric * self.score_full_complex_weight
            + modified_residue_score * self.score_modified_residue_weight
        )

        return score


# model selection


@typecheck
def get_cid_molecule_type(
    cid: int,
    asym_id: Int[" n"],  # type: ignore
    is_molecule_types: Bool[f"n {IS_MOLECULE_TYPES}"],  # type: ignore
    return_one_hot: bool = False,
) -> int | Bool[f" {IS_MOLECULE_TYPES}"]:  # type: ignore
    """Get the (majority) molecule type for where `asym_id == cid`.

    NOTE: Several PDB chains contain multiple molecule types, so
    we must choose a single molecule type for the chain. We choose
    the molecule type that is most common (i.e., the mode) in the chain.

    :param cid: chain id
    :param asym_id: [n] asym_id of each residue
    :param is_molecule_types: [n 2] is_molecule_types
    :param return_one_hot: return one hot
    :return: molecule type
    """

    cid_is_molecule_types = is_molecule_types[asym_id == cid]

    molecule_types = cid_is_molecule_types.int().argmax(1)
    molecule_type_mode = molecule_types.mode()
    molecule_type = cid_is_molecule_types[molecule_type_mode.indices.item()]

    if not return_one_hot:
        molecule_type = molecule_type_mode.values.item()
    return molecule_type


@typecheck
def get_cid_is_modified(
    cid: int,
    asym_id: Int[" n"],  # type: ignore
    is_modified_residue: Bool[" n"],  # type: ignore
    is_modified_chain_threshold: float = 0.5,
) -> bool:  # type: ignore
    """Get the (majority) "is modified residue" status for where `asym_id == cid`.

    NOTE: Several PDB chains contain a mixture of modified and canonical residue
    types, so we must choose a single modified status for the chain. We choose
    the modified residue status that is most common (i.e., the mode) in the chain.

    :param cid: chain id
    :param asym_id: [n] asym_id of each residue
    :param is_modified_residue: [n] is modified status for each residue
    :param is_modified_chain_threshold: threshold for modified chain status
    :return: is modified residue status for the entire chain
    """

    cid_is_modified_residue = is_modified_residue[asym_id == cid]
    is_modified_chain = (
        cid_is_modified_residue.sum().item() / len(cid_is_modified_residue)
        > is_modified_chain_threshold
    )

    return is_modified_chain


@typecheck
def protein_structure_from_feature(
    asym_id: Int[" n"],  # type: ignore
    molecule_ids: Int[" n"],  # type: ignore
    molecule_atom_lens: Int[" n"],  # type: ignore
    atom_pos: Float["m 3"],  # type: ignore
    atom_mask: Bool[" m"],  # type: ignore
) -> Structure:
    """Create structure for unresolved proteins.

    :param atom_mask: True for valid atoms, False for missing/padding atoms
    return: A Biopython Structure object
    """

    num_atom = atom_pos.shape[0]
    num_res = molecule_ids.shape[0]

    residue_constants = get_residue_constants(res_chem_index=IS_PROTEIN)

    molecule_atom_indices = exclusive_cumsum(molecule_atom_lens)

    builder = StructureBuilder()
    builder.init_structure("structure")
    builder.init_model(0)

    cur_cid = None
    cur_res_id = None

    for res_idx in range(num_res):
        num_atom = molecule_atom_lens[res_idx]
        cid = str(asym_id[res_idx].detach().cpu().item())

        if cid != cur_cid:
            builder.init_chain(cid)
            builder.init_seg(segid=" ")
            cur_cid = cid
            cur_res_id = 0

        restype = (
            "X"
            if molecule_ids[res_idx] >= len(residue_constants.restypes)
            else residue_constants.restypes[molecule_ids[res_idx]]
        )
        resname = residue_constants.restype_1to3[restype]
        atom_names = residue_constants.restype_name_to_compact_atom_names[resname]
        atom_names = list(filter(lambda x: x, atom_names))
        # assume residues for unresolved protein are standard
        assert (
            len(atom_names) == num_atom
        ), f"Molecule atom lens {num_atom} doesn't match with residue constant {len(atom_names)}"

        # skip if all atom of the residue is missing
        atom_idx_offset = molecule_atom_indices[res_idx]
        if not torch.any(atom_mask[atom_idx_offset : atom_idx_offset + num_atom]):
            continue

        builder.init_residue(resname, " ", cur_res_id + 1, " ")
        cur_res_id += 1

        for atom_idx in range(num_atom):
            if not atom_mask[atom_idx]:
                continue

            atom_coord = atom_pos[atom_idx + atom_idx_offset].float().detach().cpu().numpy()
            atom_name = atom_names[atom_idx]
            builder.init_atom(
                name=atom_name,
                coord=atom_coord,
                b_factor=1.0,
                occupancy=1.0,
                fullname=atom_name,
                altloc=" ",
                # only N, C, O in restype_name_to_compact_atom_names for protein
                # so just take the first char
                element=atom_name[0],
            )

    return builder.get_structure()


Sample = Tuple[Float["b m 3"], Float["b pde n n"], Float["b m"], Float["b dist n n"]]  # type: ignore
ScoredSample = Tuple[int, Float["b m 3"], Float["b m"], Float[" b"], Float[" b"]]  # type: ignore


class ScoreDetails(NamedTuple):
    """The ScoreDetails class."""

    best_gpde_index: int
    best_lddt_index: int
    score: Float[" b"]  # type: ignore
    scored_samples: List[ScoredSample]


class ComputeModelSelectionScore(Module):
    """Compute model selection score."""

    INITIAL_TRAINING_DICT = {
        "protein-protein": {"interface": 20, "intra-chain": 20},
        "DNA-protein": {"interface": 10},
        "RNA-protein": {"interface": 10},
        "ligand-protein": {"interface": 10},
        "DNA-ligand": {"interface": 5},
        "RNA-ligand": {"interface": 5},
        "DNA-DNA": {"interface": 4, "intra-chain": 4},
        "RNA-RNA": {"interface": 16, "intra-chain": 16},
        "DNA-RNA": {"interface": 4, "intra-chain": 4},
        "ligand-ligand": {"interface": 20, "intra-chain": 20},
        "mod_protein-mod_protein": {"interface": 10, "intra-chain": 10},
        "mod_RNA-mod_RNA": {"interface": 10, "intra-chain": 10},
        "mod_DNA-mod_DNA": {"interface": 10, "intra-chain": 10},
        "unresolved": {"unresolved": 10},
        "metal_ion-DNA": {"interface": 10, "intra-chain": 10},
        "metal_ion-ligand": {"interface": 10, "intra-chain": 10},
        "metal_ion-metal_ion": {"interface": 10, "intra-chain": 10},
        "metal_ion-protein": {"interface": 10, "intra-chain": 10},
        "metal_ion-RNA": {"interface": 10, "intra-chain": 10},
    }

    FINETUNING_DICT = {
        "protein-protein": {"interface": 20, "intra-chain": 20},
        "DNA-protein": {"interface": 10},
        "RNA-protein": {"interface": 2},
        "ligand-protein": {"interface": 10},
        "DNA-ligand": {"interface": 5},
        "RNA-ligand": {"interface": 2},
        "DNA-DNA": {"interface": 4, "intra-chain": 4},
        "RNA-RNA": {"interface": 16, "intra-chain": 16},
        "DNA-RNA": {"interface": 4, "intra-chain": 4},
        "ligand-ligand": {"interface": 20, "intra-chain": 20},
        "mod_protein-mod_protein": {"interface": 0, "intra-chain": 0},
        "mod_RNA-mod_RNA": {"interface": 0, "intra-chain": 0},
        "mod_DNA-mod_DNA": {"interface": 0, "intra-chain": 0},
        "unresolved": {"unresolved": 10},
        "metal_ion-DNA": {"interface": 0, "intra-chain": 0},
        "metal_ion-ligand": {"interface": 0, "intra-chain": 0},
        "metal_ion-metal_ion": {"interface": 0, "intra-chain": 0},
        "metal_ion-protein": {"interface": 0, "intra-chain": 0},
        "metal_ion-RNA": {"interface": 0, "intra-chain": 0},
    }

    TYPE_MAPPING = {
        IS_PROTEIN: "protein",
        IS_DNA: "DNA",
        IS_RNA: "RNA",
        IS_LIGAND: "ligand",
        IS_METAL_ION: "metal_ion",
    }

    @typecheck
    def __init__(
        self,
        eps: float = 1e-8,
        # NOTE: Protenix reports using 64 bins over the range [2.3125, 21.6875]
        dist_breaks: Float[" dist_break"] = torch.linspace(2.3125, 21.6875, 63),  # type: ignore
        # dist_breaks: Float[" dist_break"] = torch.linspace(2, 22, 63),  # type: ignore
        nucleic_acid_cutoff: float = 30.0,
        other_cutoff: float = 15.0,
        contact_mask_threshold: float = 8.0,
        is_fine_tuning: bool = False,
        weight_dict_config: dict = None,
        fibonacci_sphere_n: int = 200,  # NOTE: more points equal better approximation at cost of compute
    ):
        super().__init__()
        self.compute_confidence_score = ComputeConfidenceScore(eps=eps)
        self.eps = eps
        self.nucleic_acid_cutoff = nucleic_acid_cutoff
        self.other_cutoff = other_cutoff
        self.contact_mask_threshold = contact_mask_threshold
        self.is_fine_tuning = is_fine_tuning
        self.weight_dict_config = weight_dict_config

        self.register_buffer("dist_breaks", dist_breaks)
        self.register_buffer("lddt_thresholds", torch.tensor([0.5, 1.0, 2.0, 4.0]))

        # for rsa calculation

        atom_type_radii = tensor(
            [
                1.65,  # 0 - nitrogen
                1.87,  # 1 - carbon alpha
                1.76,  # 2 - carbon
                1.4,  # 3 - oxygen
                1.8,  # 4 - side atoms
                1.4,  # 5 - water
            ]
        )

        self.atom_type_index = dict(N=0, CA=1, C=2, O=3)  # rest go to 4 (side chain atom)

        self.register_buffer("atom_radii", atom_type_radii, persistent=False)

        # constitute the fibonacci sphere

        num_surface_dots = fibonacci_sphere_n * 2 + 1
        golden_ratio = 1.0 + sqrt(5.0) / 2
        weight = (4.0 * pi) / num_surface_dots

        arange = torch.arange(
            -fibonacci_sphere_n, fibonacci_sphere_n + 1, device=self.device
        )  # for example, N = 3 -> [-3, -2, -1, 0, 1, 2, 3]

        lat = torch.asin((2.0 * arange) / num_surface_dots)
        lon = torch.fmod(arange, golden_ratio) * 2 * pi / golden_ratio

        # ein:
        # sd - surface dots
        # c - coordinate (3)
        # i, j - source and target atom

        unit_surface_dots: Float["sd 3"] = torch.stack(  # type: ignore
            (lon.sin() * lat.cos(), lon.cos() * lat.cos(), lat.sin()), dim=-1
        )

        self.register_buffer("unit_surface_dots", unit_surface_dots)
        self.surface_weight = weight

    @property
    def device(self):
        """Return the device of the atom radii buffer."""
        return self.atom_radii.device

    @typecheck
    def compute_gpde(
        self,
        pde_logits: Float["b pde n n"],  # type: ignore
        dist_logits: Float["b dist n n"],  # type: ignore
        dist_breaks: Float[" dist_break"],  # type: ignore
        tok_repr_atm_mask: Bool["b n"],  # type: ignore
    ) -> Float[" b"]:  # type: ignore
        """Compute global PDE following Section 5.7 of the AF3 supplement.

        :param pde_logits: [b pde n n] PDE logits
        :param dist_logits: [b dist n n] distance logits
        :param dist_breaks: [dist_break] distance breaks
        :param tok_repr_atm_mask: [b n] true if token representation atoms exists
        :return: [b] global PDE
        """

        dtype = pde_logits.dtype

        pde = self.compute_confidence_score.compute_pde(pde_logits, tok_repr_atm_mask)

        dist_logits = rearrange(dist_logits, "b dist i j -> b i j dist")
        dist_probs = F.softmax(dist_logits, dim=-1)

        # for distances greater than the last breaks
        dist_breaks = F.pad(dist_breaks.float(), (0, 1), value=1e6).type(dtype)
        contact_mask = dist_breaks < self.contact_mask_threshold

        # contact_prob = einx.where(
        #     " dist, b i j dist, -> b i j dist", contact_mask, dist_probs, 0.0
        # ).sum(dim=-1)
        contact_prob = (dist_probs * contact_mask[None, None, None, :]).sum(dim=-1)

        mask = to_pairwise_mask(tok_repr_atm_mask)
        contact_prob = contact_prob * mask

        # Section 5.7 equation 16
        gpde = masked_average(pde, contact_prob, dim=(-1, -2))

        return gpde

    @typecheck
    def compute_lddt(
        self,
        pred_coords: Float["b m 3"],  # type: ignore
        true_coords: Float["b m 3"],  # type: ignore
        is_dna: Bool["b m"],  # type: ignore
        is_rna: Bool["b m"],  # type: ignore
        pairwise_mask: Bool["b m m"],  # type: ignore
        coords_mask: Bool["b m"] | None = None,  # type: ignore
    ) -> Float[" b"]:  # type: ignore
        """Compute lDDT.

        :param pred_coords: predicted coordinates
        :param true_coords: true coordinates
        :param is_dna: boolean tensor indicating DNA atoms
        :param is_rna: boolean tensor indicating RNA atoms
        :param pairwise_mask: boolean tensor indicating atompair for which LDDT is computed
        :param coords_mask: boolean tensor indicating valid atoms
        :return: lDDT
        """
        dtype = pred_coords.dtype
        atom_seq_len, device = pred_coords.shape[1], pred_coords.device

        # Compute distances between all pairs of atoms
        pred_dists = torch.cdist(pred_coords.float(), pred_coords.float(), p=2).type(dtype)
        true_dists = torch.cdist(true_coords.float(), true_coords.float(), p=2).type(dtype)

        # Compute distance difference for all pairs of atoms
        dist_diff = torch.abs(true_dists - pred_dists)

        # lddt = einx.subtract("thresholds, ... -> ... thresholds", self.lddt_thresholds, dist_diff)
        lddt = self.lddt_thresholds[None, None, None, :] - dist_diff[..., None]
        lddt = (lddt >= 0).type(dtype).mean(dim=-1)

        # Restrict to bespoke inclusion radius
        is_nucleotide = is_dna | is_rna
        is_nucleotide_pair = to_pairwise_mask(is_nucleotide)

        inclusion_radius = torch.where(
            is_nucleotide_pair,
            true_dists < self.nucleic_acid_cutoff,
            true_dists < self.other_cutoff,
        )

        # Compute mean, avoiding self term
        mask = inclusion_radius & ~torch.eye(atom_seq_len, dtype=torch.bool, device=device)

        # Take into account variable lengthed atoms in batch
        if exists(coords_mask):
            paired_coords_mask = to_pairwise_mask(coords_mask)
            mask = mask & paired_coords_mask

        mask = mask * pairwise_mask

        # Calculate masked averaging
        lddt_mean = masked_average(lddt, mask, dim=(-1, -2))

        return lddt_mean

    @typecheck
    def compute_chain_pair_lddt(
        self,
        asym_mask_a: Bool["b m"] | Bool[" m"],  # type: ignore
        asym_mask_b: Bool["b m"] | Bool[" m"],  # type: ignore
        pred_coords: Float["b m 3"] | Float["m 3"],  # type: ignore
        true_coords: Float["b m 3"] | Float["m 3"],  # type: ignore
        is_molecule_types: Bool[f"b m {IS_MOLECULE_TYPES}"] | Bool[f"m {IS_MOLECULE_TYPES}"],  # type: ignore
        coords_mask: Bool["b m"] | Bool[" m"] | None = None,  # type: ignore
    ) -> Float[" b"]:  # type: ignore
        """Compute the plDDT between atoms marked by `asym_mask_a` and `asym_mask_b`.

        :param asym_mask_a: [b m] asym_mask_a
        :param asym_mask_b: [b m] asym_mask_b
        :param pred_coords: [b m 3] predicted coordinates
        :param true_coords: [b m 3] true coordinates
        :param is_molecule_types: [b m 2] is_molecule_types
        :param coords_mask: [b m] coords_mask
        :return: [b] lddt
        """

        if not_exists(coords_mask):
            coords_mask = torch.ones_like(asym_mask_a)

        if asym_mask_a.ndim == 1:
            (
                asym_mask_a,
                asym_mask_b,
                pred_coords,
                true_coords,
                is_molecule_types,
                coords_mask,
            ) = map(
                lambda t: rearrange(t, "... -> 1 ..."),
                (
                    asym_mask_a,
                    asym_mask_b,
                    pred_coords,
                    true_coords,
                    is_molecule_types,
                    coords_mask,
                ),
            )

        is_dna = is_molecule_types[..., IS_DNA_INDEX]
        is_rna = is_molecule_types[..., IS_RNA_INDEX]
        pairwise_mask = to_pairwise_mask(asym_mask_a)

        lddt = self.compute_lddt(
            pred_coords, true_coords, is_dna, is_rna, pairwise_mask, coords_mask
        )

        return lddt

    @typecheck
    def get_lddt_weight(
        self,
        type_chain_a: int,
        type_chain_b: int,
        type_a_is_modified: bool,
        type_b_is_modified: bool,
        lddt_type: Literal["interface", "intra-chain", "unresolved"],
        is_fine_tuning: bool = None,
    ) -> int:
        """Get a specified lDDT weight.

        :param type_chain_a: type of chain a
        :param type_chain_b: type of chain b
        :param type_a_is_modified: is chain a mostly modified residues?
        :param type_b_is_modified: is chain b mostly modified residues?
        :param lddt_type: lDDT type
        :param is_fine_tuning: is fine tuning
        :return: lDDT weight
        """
        is_fine_tuning = default(is_fine_tuning, self.is_fine_tuning)

        weight_dict = default(
            self.weight_dict_config,
            self.FINETUNING_DICT if is_fine_tuning else self.INITIAL_TRAINING_DICT,
        )

        if lddt_type == "unresolved":
            weight = weight_dict.get(lddt_type, {}).get(lddt_type, None)
            assert weight
            return weight

        type_mapping_a = self.TYPE_MAPPING[type_chain_a]
        type_mapping_b = self.TYPE_MAPPING[type_chain_b]

        type_a_prefix = "mod_" if type_a_is_modified else ""
        type_b_prefix = "mod_" if type_b_is_modified else ""

        interface_type = sorted([type_a_prefix + type_mapping_a, type_b_prefix + type_mapping_b])
        interface_type = "-".join(interface_type)
        weight = weight_dict.get(interface_type, {}).get(lddt_type, None)

        # NOTE: For modified-unmodified chain pairs, we fall back to the unmodified chain pair weights
        if not_exists(weight) and (type_a_is_modified or type_b_is_modified):
            interface_type = sorted([type_mapping_a, type_mapping_b])
            interface_type = "-".join(interface_type)
            weight = weight_dict.get(interface_type, {}).get(lddt_type, None)

        assert (
            weight
        ), f"Weight not found for {interface_type} {lddt_type} (a_mod={type_a_is_modified}, b_mod={type_b_is_modified}) with fine_tuning={is_fine_tuning}"
        return weight

    @typecheck
    def compute_weighted_lddt(
        self,
        # atom level input
        pred_coords: Float["b m 3"],  # type: ignore
        true_coords: Float["b m 3"],  # type: ignore
        atom_mask: Bool["b m"],  # type: ignore
        # token level input
        asym_id: Int["b n"],  # type: ignore
        is_molecule_types: Bool[f"b n {IS_MOLECULE_TYPES}"],  # type: ignore
        is_modified_residue: Bool["b n"],  # type: ignore
        molecule_atom_lens: Int["b n"],  # type: ignore
        # additional input
        chains_list: List[Tuple[int, int] | Tuple[int]],
        is_fine_tuning: bool = None,
        unweighted: bool = False,
        # RASA input
        compute_rasa: bool = False,
        unresolved_cid: List[int] | None = None,
        unresolved_residue_mask: Bool["b n"] | None = None,  # type: ignore
        molecule_ids: Int["b n"] | None = None,  # type: ignore
    ) -> Float[" b"]:  # type: ignore
        """Compute the weighted lDDT.

        :param pred_coords: [b m 3] predicted coordinates
        :param true_coords: [b m 3] true coordinates
        :param atom_mask: [b m] atom mask
        :param asym_id: [b n] asym_id of each residue
        :param is_molecule_types: [b n 2] is_molecule_types
        :param is_modified_residue: [b n] is modified residue status
        :param molecule_atom_lens: [b n] molecule atom lens
        :param chains_list: List of chains
        :param is_fine_tuning: is fine tuning
        :param unweighted: unweighted lddt
        :param compute_rasa: compute RASA
        :param unresolved_cid: unresolved chain ids
        :param unresolved_residue_mask: unresolved residue mask
        :return: [b] weighted lddt
        """
        is_fine_tuning = default(is_fine_tuning, self.is_fine_tuning)

        device = pred_coords.device
        batch_size = pred_coords.shape[0]

        # Broadcast asym_id and is_molecule_types to atom level
        atom_asym_id = batch_repeat_interleave(
            asym_id, molecule_atom_lens, output_padding_value=-1
        )
        atom_is_molecule_types = batch_repeat_interleave(is_molecule_types, molecule_atom_lens)

        weighted_lddt = torch.zeros(batch_size, device=device)

        for b in range(batch_size):
            chains = chains_list[b]
            if len(chains) == 2:
                asym_id_a = chains[0]
                asym_id_b = chains[1]
                lddt_type = "interface"
            elif len(chains) == 1:
                asym_id_a = asym_id_b = chains[0]
                lddt_type = "intra-chain"
            else:
                raise Exception(f"Invalid chain list {chains}")

            type_chain_a = get_cid_molecule_type(
                asym_id_a, atom_asym_id[b], atom_is_molecule_types[b], return_one_hot=False
            )
            type_chain_b = get_cid_molecule_type(
                asym_id_b, atom_asym_id[b], atom_is_molecule_types[b], return_one_hot=False
            )

            type_a_is_modified = get_cid_is_modified(
                asym_id_a,
                asym_id[b],
                is_modified_residue[b],
            )

            type_b_is_modified = get_cid_is_modified(
                asym_id_b,
                asym_id[b],
                is_modified_residue[b],
            )

            lddt_weight = self.get_lddt_weight(
                type_chain_a,
                type_chain_b,
                type_a_is_modified,
                type_b_is_modified,
                lddt_type,
                is_fine_tuning,
            )

            asym_mask_a = atom_asym_id[b] == asym_id_a
            asym_mask_b = atom_asym_id[b] == asym_id_b

            lddt = self.compute_chain_pair_lddt(
                asym_mask_a,
                asym_mask_b,
                pred_coords[b],
                true_coords[b],
                atom_is_molecule_types[b],
                atom_mask[b],
            )

            weighted_lddt[b] = (1.0 if unweighted else lddt_weight) * lddt

        # Average the lDDT with the relative solvent accessible surface area (RASA) for unresolved proteins
        # NOTE: This differs from the AF3 Section 5.7 slightly, as here we compute the algebraic mean of the (batched) lDDT and RASA
        if compute_rasa:
            assert (
                exists(unresolved_cid) and exists(unresolved_residue_mask) and exists(molecule_ids)
            ), "RASA computation requires `unresolved_cid`, `unresolved_residue_mask`, and `molecule_ids` to be provided."
            weighted_rasa = self.compute_unresolved_rasa(
                unresolved_cid,
                unresolved_residue_mask,
                asym_id,
                molecule_ids,
                molecule_atom_lens,
                true_coords,
                atom_mask,
                is_fine_tuning=is_fine_tuning,
            )
            weighted_lddt = (weighted_lddt + weighted_rasa) / 2

        return weighted_lddt

    @typecheck
    def calc_atom_access_surface_score_from_structure(
        self, structure: Structure, **kwargs
    ) -> Float[" n"]:  # type: ignore
        """Calculate the atom access surface score for a given structure.

        :param structure: Biopython Structure object
        :return: [n] atom access surface score
        """
        # use the structure as source of truth, matching what xluo did

        structure_atom_pos = []
        structure_atom_type_for_radii = []
        side_atom_index = len(self.atom_type_index)

        for atom in structure.get_atoms():
            one_atom_pos = list(atom.get_vector())
            one_atom_type = self.atom_type_index.get(atom.name, side_atom_index)

            structure_atom_pos.append(one_atom_pos)
            structure_atom_type_for_radii.append(one_atom_type)

        structure_atom_pos: Float["m 3"] = tensor(structure_atom_pos, device=self.device)  # type: ignore
        structure_atom_type_for_radii: Int["m"] = tensor(structure_atom_type_for_radii, device=self.device)  # type: ignore

        structure_atoms_per_residue: Int["n"] = tensor([len([*residue.get_atoms()]) for residue in structure.get_residues()], device=self.device).long()  # type: ignore

        return self.calc_atom_access_surface_score(
            atom_pos=structure_atom_pos,
            atom_type=structure_atom_type_for_radii,
            molecule_atom_lens=structure_atoms_per_residue,
            **kwargs,
        )

    @typecheck
    def calc_atom_access_surface_score(
        self,
        atom_pos: Float["m 3"],  # type: ignore
        atom_type: Int[" m"],  # type: ignore
        molecule_atom_lens: Int[" n"] | None = None,  # type: ignore
        atom_distance_min_thres: float = 1e-4,
    ) -> Float[" m"] | Float[" n"]:  # type: ignore
        """Calculate the atom access surface score for a given set of atoms.

        :param atom_pos: [m 3] atom positions
        :param atom_type: [m] atom types
        :param molecule_atom_lens: [n] number of atoms for each residue
        :param atom_distance_min_thres: minimum distance threshold for atom pairs
        :return: [m] or [n] atom access surface score
        """
        atom_radii: Float["m"] = self.atom_radii[atom_type]  # type: ignore

        water_radii = self.atom_radii[-1]

        # atom radii is always summed with water radii

        atom_radii += water_radii
        atom_radii_sq = atom_radii.pow(
            2
        )  # always use square of radii or distance for comparison - save on sqrt

        # write custom RSA function here

        # get atom relative positions + distance
        # for determining whether to include pairs of atom in calculation for the `free` adjective

        # atom_rel_pos = einx.subtract("j c, i c -> i j c", atom_pos, atom_pos)
        atom_rel_pos = atom_pos.unsqueeze(0) - atom_pos.unsqueeze(1)
        atom_rel_dist_sq = atom_rel_pos.pow(2).sum(dim=-1)

        # max_distance_include = einx.add("i, j -> i j", atom_radii, atom_radii).pow(2)
        max_distance_include = (atom_radii.unsqueeze(1) + atom_radii.unsqueeze(0)).pow(2)

        include_in_free_calc = (atom_rel_dist_sq < max_distance_include) & (
            atom_rel_dist_sq > atom_distance_min_thres
        )

        # max included in calculation per row

        max_included = include_in_free_calc.long().sum(dim=-1).amax()

        include_in_free_calc, include_indices = include_in_free_calc.long().topk(
            max_included, dim=-1
        )

        # atom_rel_pos = einx.get_at('i [m] c, i j -> i j c', atom_rel_pos, include_indices)

        include_in_free_calc = include_in_free_calc.bool()
        atom_rel_pos = atom_rel_pos.gather(1, repeat(include_indices, "i j -> i j c", c=3))
        target_atom_radii_sq = atom_radii_sq[include_indices]

        # overall logic

        # surface_dots = einx.multiply("m, sd c -> m sd c", atom_radii, self.unit_surface_dots)
        surface_dots = atom_radii[..., None, None] * self.unit_surface_dots[None, ...]

        dist_from_surface_dots_sq = (
            # einx.subtract("i j c, i sd c -> i sd j c", atom_rel_pos, surface_dots)
            (atom_rel_pos[..., None, :, :] - surface_dots[..., None, :])
            .pow(2)
            .sum(dim=-1)
        )

        # target_atom_close_to_surface_dots = einx.less(
        #     "i j, i sd j -> i sd j", target_atom_radii_sq, dist_from_surface_dots_sq
        # )
        target_atom_close_to_surface_dots = (
            target_atom_radii_sq[..., None, :] < dist_from_surface_dots_sq
        )

        # target_atom_close_or_not_included = einx.logical_or(
        #     "i sd j, i j -> i sd j", target_atom_close_to_surface_dots, ~include_in_free_calc
        # )
        target_atom_close_or_not_included = (
            target_atom_close_to_surface_dots | ~include_in_free_calc[..., None, :]
        )

        is_free = reduce(
            target_atom_close_or_not_included, "i sd j -> i sd", "all"
        )  # basically the most important line, calculating whether an atom is free by some distance measure

        score = reduce(is_free.float() * self.surface_weight, "m sd -> m", "sum")

        per_atom_access_surface_score = score * atom_radii_sq

        if not exists(molecule_atom_lens):
            return per_atom_access_surface_score

        # sum up all surface scores for atoms per residue
        # the final score seems to be the average of the rsa across all residues (selected by `chain_unresolved_residue_mask`)

        rasa, mask = sum_pool_with_lens(
            rearrange(per_atom_access_surface_score, "... -> 1 ... 1"),
            rearrange(molecule_atom_lens, "... -> 1 ..."),
        )

        # rasa = einx.where("b n, b n d, -> b n d", mask, rasa, 0.0)
        rasa = rasa * mask[..., None]

        rasa = rearrange(rasa, "1 n 1 -> n")

        return rasa

    @typecheck
    def _compute_unresolved_rasa(
        self,
        unresolved_cid: int | None,
        unresolved_residue_mask: Bool[" n"] | None,  # type: ignore
        asym_id: Int[" n"],  # type: ignore
        molecule_ids: Int[" n"],  # type: ignore
        molecule_atom_lens: Int[" n"],  # type: ignore
        atom_pos: Float["m 3"],  # type: ignore
        atom_mask: Bool[" m"],  # type: ignore
        return_mean: bool = True,
        **rsa_calc_kwargs,
    ) -> Float[""] | Float[" m"] | Float[" n"]:  # type: ignore
        """Compute the unresolved relative solvent accessible surface area (RASA) for proteins.
        using in-house (i.e., PyTorch-native) RSA calculation.

        unresolved_cid: asym_id for protein chains with unresolved residues
        unresolved_residue_mask: True for unresolved residues, False for resolved residues
        asym_id: asym_id for each residue
        molecule_ids: molecule_ids for each residue
        molecule_atom_lens: number of atoms for each residue
        atom_pos: [m 3] atom positions
        atom_mask: True for valid atoms, False for missing/padding atoms
        return_mean: return the mean RASA
        :return: unresolved RASA
        """

        num_atom = atom_pos.shape[0]

        chain_mask = (
            (
                (asym_id == unresolved_cid)
                if exists(unresolved_cid)
                else torch.ones_like(asym_id, dtype=torch.bool)
            )
            & (molecule_ids < get_residue_constants("peptide").restype_num)
            & (
                # NOTE: for now, we only consider unmodified amino acids
                molecule_atom_lens
                > 1
            )
        )
        chain_unresolved_residue_mask = (
            unresolved_residue_mask
            if exists(unresolved_residue_mask)
            else torch.ones_like(asym_id, dtype=torch.bool)
        )
        chain_asym_id = asym_id[chain_mask]
        chain_molecule_ids = molecule_ids[chain_mask]
        chain_molecule_atom_lens = molecule_atom_lens[chain_mask]

        chain_mask_to_atom = torch.repeat_interleave(chain_mask, molecule_atom_lens)

        # if there's padding in num atom
        num_pad = num_atom - molecule_atom_lens.sum()
        if num_pad > 0:
            chain_mask_to_atom = F.pad(chain_mask_to_atom, (0, num_pad), value=False)

        chain_atom_pos = atom_pos[chain_mask_to_atom]
        chain_atom_mask = atom_mask[chain_mask_to_atom]

        rasa = torch.zeros_like(asym_id, dtype=torch.float)

        if chain_mask.any():
            structure = protein_structure_from_feature(
                chain_asym_id,
                chain_molecule_ids,
                chain_molecule_atom_lens,
                chain_atom_pos,
                chain_atom_mask,
            )

            # per atom rsa calculation

            rasa[chain_mask] = self.calc_atom_access_surface_score_from_structure(
                structure, **rsa_calc_kwargs
            )

        unresolved_rasa = rasa[chain_unresolved_residue_mask]

        return unresolved_rasa.mean() if return_mean else unresolved_rasa

    @typecheck
    def compute_unresolved_rasa(
        self,
        unresolved_cid: List[int],
        unresolved_residue_mask: Bool["b n"],  # type: ignore
        asym_id: Int["b n"],  # type: ignore
        molecule_ids: Int["b n"],  # type: ignore
        molecule_atom_lens: Int["b n"],  # type: ignore
        atom_pos: Float["b m 3"],  # type: ignore
        atom_mask: Bool["b m"],  # type: ignore
        is_fine_tuning: bool = None,
    ) -> Float[" b"] | Float["b m"] | Float["b n"]:  # type: ignore
        """Compute the unresolved relative solvent accessible surface area (RASA) for (batched)
        proteins.

        unresolved_cid: asym_id for protein chains with unresolved residues
        unresolved_residue_mask: True for unresolved residues, False for resolved residues
        asym_id: [b n] asym_id of each residue
        molecule_ids: [b n] molecule_ids of each residue
        molecule_atom_lens: [b n] molecule atom lens
        atom_pos: [b m 3] atom positions
        atom_mask: [b m] atom mask
        :return: [b] or [b m] or [b n] unresolved RASA
        """
        is_fine_tuning = default(is_fine_tuning, self.is_fine_tuning)

        weight_dict = default(
            self.weight_dict_config,
            self.FINETUNING_DICT if is_fine_tuning else self.INITIAL_TRAINING_DICT,
        )

        weight = weight_dict.get("unresolved", {}).get("unresolved", None)
        assert weight, "Weight not found for unresolved"

        unresolved_rasa = [
            self._compute_unresolved_rasa(*args)
            for args in zip(
                unresolved_cid,
                unresolved_residue_mask,
                asym_id,
                molecule_ids,
                molecule_atom_lens,
                atom_pos,
                atom_mask,
            )
        ]
        return torch.stack(unresolved_rasa) * weight

    @typecheck
    def compute_model_selection_score(
        self,
        batch: BatchedAtomInput,
        samples: List[Sample],
        is_fine_tuning: bool = None,
        return_details: bool = False,
        return_unweighted_scores: bool = False,
        compute_rasa: bool = False,
        unresolved_cid: List[int] | None = None,
        unresolved_residue_mask: Bool["b n"] | None = None,  # type: ignore
        missing_chain_index: int = -1,
        device: str | torch.device | None = None,
    ) -> Float[" b"] | ScoreDetails:  # type: ignore
        """Compute the model selection score for an input batch and corresponding (sampled) atom
        positions.

        :param batch: A batch of `AtomInput` data.
        :param samples: A list of sampled atom positions along with their predicted distance errors and labels.
        :param is_fine_tuning: is fine tuning
        :param return_details: return the top model and its score
        :param return_unweighted_scores: return the unweighted scores (i.e., lDDT)
        :param compute_rasa: compute the relative solvent accessible surface area (RASA) for unresolved proteins
        :param unresolved_cid: unresolved chain ids
        :param unresolved_residue_mask: unresolved residue mask
        :param missing_chain_index: missing chain index
        :param device: device
        :return: [b] model selection score and optionally the top model
        """
        device = default(device, samples[0][0].device)

        is_fine_tuning = default(is_fine_tuning, self.is_fine_tuning)

        if compute_rasa:
            if not (exists(unresolved_cid) and exists(unresolved_residue_mask)):
                logger.warning(
                    "RASA computation requires `unresolved_cid` and `unresolved_residue_mask` to be provided. Skipping RASA computation."
                )
                compute_rasa = False

        # collect required features

        batch_dict = batch.cpu_dict() if device == "cpu" else batch.dict()

        atom_pos_true = batch_dict["atom_pos"]
        atom_mask = ~batch_dict["missing_atom_mask"]

        asym_id = batch_dict["additional_molecule_feats"].unbind(dim=-1)[2]
        is_molecule_types = batch_dict["is_molecule_types"]

        chains = [
            tuple(chain for chain in chains_list if chain != missing_chain_index)
            for chains_list in batch_dict["chains"].tolist()
        ]
        molecule_atom_lens = batch_dict["molecule_atom_lens"]
        molecule_ids = batch_dict["molecule_ids"]

        valid_atom_len_mask = batch_dict["molecule_atom_lens"] >= 0
        tok_repr_atm_mask = batch_dict["distogram_atom_indices"] >= 0 & valid_atom_len_mask

        is_modified_residue = batch_dict["is_molecule_mod"].any(dim=-1)

        # score samples

        scored_samples: List[ScoredSample] = []

        for sample_idx, sample in enumerate(samples):
            atom_pos_pred, pde_logits, plddt, dist_logits = sample

            weighted_lddt = self.compute_weighted_lddt(
                atom_pos_pred,
                atom_pos_true,
                atom_mask,
                asym_id,
                is_molecule_types,
                is_modified_residue,
                molecule_atom_lens,
                chains_list=chains,
                is_fine_tuning=is_fine_tuning,
                compute_rasa=compute_rasa,
                unresolved_cid=unresolved_cid,
                unresolved_residue_mask=unresolved_residue_mask,
                molecule_ids=molecule_ids,
                unweighted=return_unweighted_scores,
            )

            gpde = self.compute_gpde(
                pde_logits,
                dist_logits,
                self.dist_breaks,
                tok_repr_atm_mask,
            )

            scored_samples.append((sample_idx, atom_pos_pred, plddt, weighted_lddt, gpde))

        # quick collate

        *_, all_weighted_lddt, all_gpde = zip(*scored_samples)

        # rank by batch-averaged minimum gPDE

        best_gpde_index = torch.stack(all_gpde).mean(dim=-1).argmin().item()

        # rank by batch-averaged maximum lDDT

        best_lddt_index = torch.stack(all_weighted_lddt).mean(dim=-1).argmax().item()

        # some weighted score

        model_selection_score = (
            scored_samples[best_gpde_index][-2] + scored_samples[best_lddt_index][-2]
        ) / 2

        if not return_details:
            return model_selection_score

        score_details = ScoreDetails(
            best_gpde_index=best_gpde_index,
            best_lddt_index=best_lddt_index,
            score=model_selection_score,
            scored_samples=scored_samples,
        )

        return score_details

    @typecheck
    def forward(
        self, megafolds: Tuple[MegaFold], batched_atom_inputs: BatchedAtomInput, **kwargs
    ) -> Float[" b"] | ScoreDetails:  # type: ignore
        """Make model selections by computing the model selection score.

        NOTE: Give this function a tuple of `MegaFold` modules and a batch of atomic inputs, and it will
        select the best module via the model selection score by returning the index of the corresponding tuple.

        :param megafolds: Tuple of `MegaFold` modules
        :param batched_atom_inputs: A batch of `AtomInput` data
        :param kwargs: Additional keyword arguments
        :return: Model selection score
        """

        samples = []

        with torch.no_grad():
            for megafold in megafolds:
                megafold.eval()

                pred_atom_pos, logits = megafold(
                    **batched_atom_inputs.dict(),
                    return_loss=False,
                    return_confidence_head_logits=True,
                    return_distogram_head_logits=True,
                )
                plddt = self.compute_confidence_score.compute_plddt(logits.plddt)

                samples.append((pred_atom_pos, logits.pde, plddt, logits.distance))

        scores = self.compute_model_selection_score(batched_atom_inputs, samples=samples, **kwargs)

        return scores


# main class


class MegaFold(Module):
    """Algorithm 1."""

    @save_args_and_kwargs
    @typecheck
    def __init__(
        self,
        *,
        dim_atom_inputs,
        dim_template_feats,
        dim_template_model=64,
        atoms_per_window=27,
        dim_atom=128,
        dim_atompair_inputs=5,
        dim_atompair=16,
        dim_input_embedder_token=384,
        dim_single=384,
        dim_pairwise=128,
        dim_token=768,
        dim_msa_inputs=NUM_MSA_ONE_HOT,
        dim_additional_msa_feats=2,  # in paper, they include two meta information per msa-token pair (has_deletion w/ dim=1, deletion_value w/ dim=1)
        dim_additional_token_feats=33,  # in paper, they include two meta information per token (profile w/ dim=32, deletion_mean w/ dim=1)
        num_molecule_types: int = NUM_MOLECULE_IDS,  # restype in additional residue information, apparently 32. will do 33 to account for metal ions
        num_atom_embeds: int | None = None,
        num_atompair_embeds: int | None = None,
        num_molecule_mods: int | None = DEFAULT_NUM_MOLECULE_MODS,
        # NOTE: Protenix reports using 64 bins over the range [2.3125, 21.6875]
        distance_bins: List[float] = torch.linspace(2.3125, 21.6875, 64).float().tolist(),
        # distance_bins: List[float] = torch.linspace(2, 22, 64).float().tolist(),  # NOTE: in paper, DM seems to reuse AF2's setup of having 64 bins from 2 to 22
        pae_bins: List[float] = torch.linspace(0.5, 32, 64).float().tolist(),
        pde_bins: List[float] = torch.linspace(0.5, 32, 64).float().tolist(),
        ignore_index=-1,
        num_dist_bins: int | None = None,
        num_plddt_bins=50,
        num_pae_bins: int | None = None,
        num_pde_bins: int | None = None,
        sigma_data=16.0,
        num_rollout_steps=20,
        loss_confidence_weight=1e-4,
        loss_distogram_weight=1e-2,
        loss_diffusion_weight=4.0,
        prior_type: Literal["diffusion"] = "diffusion",
        multi_chain_permutation_alignment: bool = True,
        atom_permutation_alignment: bool = True,
        input_embedder_kwargs: dict = dict(
            atom_transformer_blocks=3,
            atom_transformer_heads=4,
            atom_transformer_kwargs=dict(),
        ),
        confidence_head_kwargs: dict = dict(pairformer_depth=4),
        template_embedder_kwargs: dict = dict(
            pairformer_stack_depth=2,
            pairwise_block_kwargs=dict(),
            layerscale_output=True,
        ),
        msa_module_kwargs: dict = dict(
            depth=4,
            dim_msa=64,
            outer_product_mean_dim_hidden=32,
            msa_pwa_dropout_row_prob=0.15,
            msa_pwa_heads=8,
            msa_pwa_dim_head=32,
            pairwise_block_kwargs=dict(),
            layerscale_output=True,
        ),
        pairformer_stack: dict = dict(
            depth=48,
            pair_bias_attn_dim_head=64,
            pair_bias_attn_heads=16,
            dropout_row_prob=0.25,
            pairwise_block_kwargs=dict(),
        ),
        relative_position_encoding_kwargs: dict = dict(
            r_max=32,
            s_max=2,
        ),
        diffusion_module_kwargs: dict = dict(
            single_cond_kwargs=dict(
                num_transitions=2,
                transition_expansion_factor=2,
            ),
            pairwise_cond_kwargs=dict(
                num_transitions=2,
                transition_expansion_factor=2,
            ),
            atom_encoder_depth=3,
            atom_encoder_heads=4,
            token_transformer_depth=24,
            token_transformer_heads=16,
            atom_decoder_depth=3,
            atom_decoder_heads=4,
        ),
        edm_kwargs: dict = dict(
            sigma_min=0.002,
            sigma_max=80,
            rho=7,
            P_mean=-1.2,
            P_std=1.2,
            S_churn=80,
            S_tmin=0.05,
            S_tmax=50,
            S_noise=1.003,
        ),
        multi_chain_permutation_alignment_kwargs: dict = dict(),
        atom_permutation_alignment_kwargs: dict = dict(
            run_checker=False,
            eps=1e-8,
        ),
        lddt_mask_nucleic_acid_cutoff=30.0,
        lddt_mask_other_cutoff=15.0,
        nucleotide_loss_weight: float = 5.0,
        ligand_loss_weight: float = 10.0,
        min_conf_resolution: float = 0.1,
        max_conf_resolution: float = 4.0,
        stochastic_frame_average=False,
        distogram_atom_resolution=False,
        checkpoint_input_embedding=False,
        checkpoint_trunk_pairformer=False,
        checkpoint_distogram_head=False,
        checkpoint_confidence_head=False,
        checkpoint_diffusion_module=False,
        detach_when_recycling=True,
        use_optimized_evo: Literal["deepspeed", "triton"] | None = None,
        globally_enable_autocasting: bool = True,
        use_tempo_layernorm=False,
        plm_embeddings: PLMEmbedding | tuple[PLMEmbedding, ...] | None = None,
        nlm_embeddings: NLMEmbedding | tuple[NLMEmbedding, ...] | None = None,
        plm_kwargs: dict | tuple[dict, ...] | None = None,
        nlm_kwargs: dict | tuple[dict, ...] | None = None,
        constraints: List[CONSTRAINTS] | None = None,
        diffusion_num_augmentations: int = 48,
        diffusion_chunk_size: int = 4,
        diffusion_add_smooth_lddt_loss: bool = True,
        diffusion_add_bond_loss: bool = False,
        train_structure_and_distogram: bool = True,
        train_pae: bool = True,
        karras_formulation: bool = True,
        disable_distogram_casting: bool = True,
        disable_edm_casting: bool = True,
        disable_sampling_casting: bool = True,
        disable_confidence_casting: bool = True,
        disable_loss_casting: bool = True,
        input_independent_baseline: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.verbose = verbose

        if verbose:
            logger.info("Initializing MegaFold...")

        if verbose and exists(use_optimized_evo) and use_optimized_evo == "deepspeed":
            logger.info("Using DeepSpeed's optimized Evoformer kernel.")

        if verbose and exists(use_optimized_evo) and use_optimized_evo == "triton":
            logger.info("Using Triton's optimized Evoformer kernel.")

        # select attention implementation

        self.use_optimized_evo = use_optimized_evo

        # enable or disable autocasting globally

        self.globally_enable_autocasting = globally_enable_autocasting

        # choose layer normalization type globally

        global LayerNorm
        LayerNorm = nn.LayerNorm

        # store atom and atompair input dimensions for shape validation

        self.dim_atom_inputs = dim_atom_inputs
        self.dim_template_feats = dim_template_feats
        self.dim_atompair_inputs = dim_atompair_inputs

        # optional atom and atom bond embeddings

        num_atom_embeds = default(num_atom_embeds, 0)
        num_atompair_embeds = default(num_atompair_embeds, 0)

        has_atom_embeds = num_atom_embeds > 0
        has_atompair_embeds = num_atompair_embeds > 0

        if has_atom_embeds:
            self.atom_embeds = nn.Embedding(num_atom_embeds, dim_atom)

        if has_atompair_embeds:
            self.atompair_embeds = nn.Embedding(num_atompair_embeds, dim_atompair)

        self.has_atom_embeds = has_atom_embeds
        self.has_atompair_embeds = has_atompair_embeds

        # optional pairwise token constraint embeddings

        self.constraints = constraints

        if exists(constraints):
            self.constraint_embeds = nn.ModuleList(
                [
                    LinearNoBias(CONSTRAINT_DIMS[constraint], dim_pairwise)
                    for constraint in constraints
                ]
            )
            self.learnable_constraint_masks = nn.ParameterList(
                [nn.Parameter(torch.zeros(1)) for _ in constraints]
            )

        # residue or nucleotide modifications

        num_molecule_mods = default(num_molecule_mods, 0)
        has_molecule_mod_embeds = num_molecule_mods > 0

        if has_molecule_mod_embeds:
            self.molecule_mod_embeds = nn.Embedding(num_molecule_mods, dim_single)

        self.has_molecule_mod_embeds = has_molecule_mod_embeds

        # optional protein language model(s) (PLM) embeddings

        self.plms = None

        if exists(plm_embeddings):
            self.plms = []  # purposefully keep PLM weights from being registered as parameters

            for one_plm_embedding, one_plm_kwargs in zip_longest(
                cast_tuple(plm_embeddings), cast_tuple(plm_kwargs)
            ):
                assert (
                    one_plm_embedding in PLMRegistry
                ), f"Received invalid PLM embedding name: {one_plm_embedding}. Acceptable ones are {list(PLMRegistry.keys())}."

                constructor = PLMRegistry.get(one_plm_embedding)

                one_plm_kwargs = default(one_plm_kwargs, {})
                plm = constructor(**one_plm_kwargs)

                freeze_(plm)

                self.plms.append(plm.half())

        if exists(self.plms):
            concatted_plm_embed_dim = sum([plm.embed_dim for plm in self.plms])

            self.to_plm_embeds = LinearNoBias(concatted_plm_embed_dim, dim_single)

        # optional nucleotide language model(s) (NLM) embeddings

        self.nlms = None

        if exists(nlm_embeddings):
            self.nlms = []  # purposefully keep NLM weights from being registered as parameters

            for one_nlm_embedding, one_nlm_kwargs in zip_longest(
                cast_tuple(nlm_embeddings), cast_tuple(nlm_kwargs)
            ):
                assert (
                    one_nlm_embedding in NLMRegistry
                ), f"Received invalid NLM embedding name: {one_nlm_embedding}. Acceptable ones are {list(NLMRegistry.keys())}."

                constructor = NLMRegistry.get(one_nlm_embedding)

                one_nlm_kwargs = default(one_nlm_kwargs, {})
                nlm = constructor(**one_nlm_kwargs)

                freeze_(nlm)

                self.nlms.append(nlm.half())

        if exists(self.nlms):
            concatted_nlm_embed_dim = sum([nlm.embed_dim for nlm in self.nlms])

            self.to_nlm_embeds = LinearNoBias(concatted_nlm_embed_dim, dim_single)

        # atoms per window

        self.atoms_per_window = atoms_per_window

        # input feature embedder

        self.input_embedder = InputFeatureEmbedder(
            num_molecule_types=num_molecule_types,
            dim_atom_inputs=dim_atom_inputs,
            dim_atompair_inputs=dim_atompair_inputs,
            atoms_per_window=atoms_per_window,
            dim_atom=dim_atom,
            dim_atompair=dim_atompair,
            dim_token=dim_input_embedder_token,
            dim_single=dim_single,
            dim_pairwise=dim_pairwise,
            dim_additional_token_feats=dim_additional_token_feats,
            **input_embedder_kwargs,
        )

        # they concat some MSA related information per token (`profile` w/ dim=32, `deletion_mean` w/ dim=1)
        # line 2 of Algorithm 2
        # the `f_restypes` is handled elsewhere

        dim_single_inputs = dim_input_embedder_token + dim_additional_token_feats

        # relative positional encoding
        # used by pairwise in main alphafold2 trunk
        # and also in the diffusion module separately from alphafold3

        self.relative_position_encoding = RelativePositionEncoding(
            dim_out=dim_pairwise, **relative_position_encoding_kwargs
        )

        # token bonds
        # Algorithm 1 - line 5

        self.token_bond_to_pairwise_feat = nn.Sequential(
            Rearrange("... -> ... 1"), LinearNoBias(1, dim_pairwise)
        )

        # templates

        self.template_embedder = TemplateEmbedder(
            dim_template_feats=dim_template_feats,
            dim=dim_template_model,
            dim_pairwise=dim_pairwise,
            checkpoint=checkpoint_input_embedding,
            **template_embedder_kwargs,
        )

        # msa

        # they concat some MSA related information per MSA-token pair (`has_deletion` w/ dim=1, `deletion_value` w/ dim=1)

        self.msa_module = MSAModule(
            dim_single=dim_single_inputs,
            dim_pairwise=dim_pairwise,
            dim_msa_input=dim_msa_inputs,
            dim_additional_msa_feats=dim_additional_msa_feats,
            checkpoint=checkpoint_input_embedding,
            **msa_module_kwargs,
        )

        # main pairformer trunk, 48 layers

        self.pairformer = PairformerStack(
            dim_single=dim_single,
            dim_pairwise=dim_pairwise,
            checkpoint=checkpoint_trunk_pairformer,
            **pairformer_stack,
        )

        # recycling related

        self.detach_when_recycling = detach_when_recycling

        # self.recycle_single = nn.Sequential(
        #     LayerNorm(dim_single), LinearNoBias(dim_single, dim_single)
        # )
        self.recycle_single = LayernormLinear(dim_single, dim_single, has_linear_bias=False)

        # self.recycle_pairwise = nn.Sequential(
        #     LayerNorm(dim_pairwise),
        #     LinearNoBias(dim_pairwise, dim_pairwise),
        # )
        self.recycle_pairwise = LayernormLinear(dim_pairwise, dim_pairwise, has_linear_bias=False)

        # diffusion 
        
        edm_class = ElucidatedAtomDiffusion

        self.diffusion_module = DiffusionModule(
            dim_pairwise_trunk=dim_pairwise,
            dim_pairwise_rel_pos_feats=dim_pairwise,
            atoms_per_window=atoms_per_window,
            dim_pairwise=dim_pairwise,
            sigma_data=sigma_data,
            dim_atom=dim_atom,
            dim_atompair=dim_atompair,
            dim_token=dim_token,
            dim_single=dim_single + dim_single_inputs,
            checkpoint=checkpoint_diffusion_module,
            **diffusion_module_kwargs,
        )

        self.edm = edm_class(
            self.diffusion_module,
            sigma_data=sigma_data,
            diffusion_num_augmentations=diffusion_num_augmentations,
            diffusion_chunk_size=diffusion_chunk_size,
            stochastic_frame_average=stochastic_frame_average,
            karras_formulation=karras_formulation,
            atom_permutation_alignment=atom_permutation_alignment,
            **edm_kwargs,
        )

        self.num_rollout_steps = num_rollout_steps

        # logit heads

        distance_bins_tensor = tensor(distance_bins)
        num_dist_bins = default(num_dist_bins, len(distance_bins_tensor))

        assert (
            len(distance_bins_tensor) == num_dist_bins
        ), "The argument `distance_bins_tensor` must have a length equal to the `num_dist_bins` passed in."

        self.distogram_atom_resolution = distogram_atom_resolution

        self.distogram_head = DistogramHead(
            dim_pairwise=dim_pairwise,
            dim_atom=dim_atom,
            num_dist_bins=num_dist_bins,
            atom_resolution=distogram_atom_resolution,
            checkpoint=checkpoint_distogram_head,
        )

        # lddt related

        self.lddt_mask_nucleic_acid_cutoff = lddt_mask_nucleic_acid_cutoff
        self.lddt_mask_other_cutoff = lddt_mask_other_cutoff

        # pae related bins and modules

        pae_bins_tensor = tensor(pae_bins)
        num_pae_bins = len(pae_bins)

        # pde related bins

        pde_bins_tensor = tensor(pde_bins)
        self.register_buffer("pde_bins", pde_bins_tensor)
        num_pde_bins = len(pde_bins)

        # confidence head

        self.confidence_head = ConfidenceHead(
            dim_single_inputs=dim_single_inputs,
            dim_atom=dim_atom,
            # NOTE: Protenix reports using the following confidence head bins instead (I believe they meant 1.2 width?):
            atompair_dist_bins=torch.linspace(3.375, 21.375, 16).float().tolist(),
            # atompair_dist_bins=distance_bins,
            dim_single=dim_single,
            dim_pairwise=dim_pairwise,
            num_plddt_bins=num_plddt_bins,
            num_pde_bins=num_pde_bins,
            num_pae_bins=num_pae_bins,
            checkpoint=checkpoint_confidence_head,
            **confidence_head_kwargs,
        )

        # loss related

        self.diffusion_add_smooth_lddt_loss = diffusion_add_smooth_lddt_loss
        self.diffusion_add_bond_loss = diffusion_add_bond_loss
        self.train_structure_and_distogram = train_structure_and_distogram
        self.train_pae = train_pae

        self.ignore_index = ignore_index
        self.loss_distogram_weight = loss_distogram_weight
        self.loss_confidence_weight = loss_confidence_weight
        self.loss_diffusion_weight = loss_diffusion_weight
        self.nucleotide_loss_weight = nucleotide_loss_weight
        self.ligand_loss_weight = ligand_loss_weight

        self.prior_type = prior_type

        self.disable_distogram_casting = disable_distogram_casting
        self.disable_edm_casting = disable_edm_casting
        self.disable_sampling_casting = disable_sampling_casting
        self.disable_confidence_casting = disable_confidence_casting
        self.disable_loss_casting = disable_loss_casting

        self.multi_chain_permutation_alignment = None
        self.atom_permutation_alignment = None

        if multi_chain_permutation_alignment:
            self.multi_chain_permutation_alignment = MultiChainPermutationAlignment(
                **multi_chain_permutation_alignment_kwargs,
            )

        if atom_permutation_alignment:
            self.atom_permutation_alignment = AtomPermutationAlignment(
                **atom_permutation_alignment_kwargs,
            )

        self.loss = MegaFoldLoss(
            distogram_weight=loss_distogram_weight,
            diffusion_weight=loss_diffusion_weight,
            confidence_weight=loss_confidence_weight,
            distance_bins=distance_bins_tensor,
            pae_bins=pae_bins_tensor,
            pde_bins=pde_bins_tensor,
            num_plddt_bins=num_plddt_bins,
            diffusion_chunk_size=diffusion_chunk_size,
            lddt_mask_nucleic_acid_cutoff=lddt_mask_nucleic_acid_cutoff,
            lddt_mask_other_cutoff=lddt_mask_other_cutoff,
            min_conf_resolution=min_conf_resolution,
            max_conf_resolution=max_conf_resolution,
            diffusion_add_smooth_lddt_loss=diffusion_add_smooth_lddt_loss,
            diffusion_add_bond_loss=diffusion_add_bond_loss,
            train_pae=train_pae,
            distogram_atom_resolution=distogram_atom_resolution,
            karras_formulation=karras_formulation,
            ignore_index=ignore_index,
            smooth_lddt_loss_kwargs=dict(
                nucleic_acid_cutoff=lddt_mask_nucleic_acid_cutoff,
                other_cutoff=lddt_mask_other_cutoff,
            ),
        )

        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

        # input-independent baseline

        self.input_independent_baseline = input_independent_baseline

        # some shorthand for jaxtyping

        self.w = atoms_per_window
        self.dapi = self.dim_atompair_inputs
        self.dai = self.dim_atom_inputs
        self.dmf = dim_additional_msa_feats
        self.dtf = dim_additional_token_feats
        self.dmi = dim_msa_inputs
        self.num_mods = num_molecule_mods

    @property
    def device(self):
        """Device of the model."""
        return self.zero.device

    @remove_plms
    @remove_nlms
    def state_dict(self, *args, **kwargs):
        """State dict without PLMs or NLMs."""
        return super().state_dict(*args, **kwargs)

    @remove_plms
    @remove_nlms
    def load_state_dict(self, *args, **kwargs):
        """Load state dict without PLMs or NLMs."""
        return super().load_state_dict(*args, **kwargs)

    @property
    def state_dict_with_init_args(self):
        """State dict with the initialization arguments."""
        return dict(
            version=self._version,
            init_args_and_kwargs=self._args_and_kwargs,
            state_dict=self.state_dict(),
        )

    @typecheck
    def save(self, path: str | Path, overwrite=False):
        """Save the model to a file.

        :param path: The path to save the model.
        :param overwrite: Whether to overwrite an existing file.
        """
        if isinstance(path, str):
            path = Path(path)

        assert not path.is_dir() and (not path.exists() or overwrite)

        path.parent.mkdir(exist_ok=True, parents=True)

        package = dict(model=self.state_dict_with_init_args)

        torch.save(package, str(path))  # nosec

    @typecheck
    def load(self, path: str | Path, strict=False, map_location="cpu", load_ema_weights=True):
        """Load a saved model.

        :param path: The path to the saved model.
        :param strict: Whether to strictly load the model.
        :param map_location: The device to map the model to.
        """
        if isinstance(path, str):
            path = Path(path)

        assert path.exists() and path.is_file()

        package = torch.load(str(path), map_location=map_location, weights_only=True)  # nosec

        weights_type = "ema_model" if load_ema_weights and "ema_model" in package else "model"
        model_state_dict = apply_function_to_ordered_dict_keys(
            package[weights_type],
            func=lambda k: k.removeprefix(f"{weights_type}."),
        )

        current_version = version("megafold")
        if package["version"] != current_version:
            logger.info(
                f'Loading a saved model from version {package["version"]} but you are on version {current_version}'
            )

        self.load_state_dict(model_state_dict, strict=strict)

        return package.get("id", None)

    @staticmethod
    @typecheck
    def init_and_load(path: str | Path, map_location="cpu", load_ema_weights=True):
        """Initialize and load a saved model.

        :param path: The path to the saved model.
        :param map_location: The device to map the model to.
        """
        if isinstance(path, str):
            path = Path(path)

        assert path.is_file()

        package = torch.load(str(path), map_location=map_location, weights_only=True)  # nosec

        args, kwargs = package["init_args_and_kwargs"]
        megafold = MegaFold(*args, **kwargs)

        plms, nlms = megafold.plms, megafold.nlms

        megafold.load(
            path, map_location=map_location, load_ema_weights=load_ema_weights
        )  # NOTE: checkpoints do not store PLMs or NLMs

        # # inspect the model weights and compare them
        # del megafold.plms, megafold.nlms

        # import copy

        # untrained_megafold = copy.deepcopy(megafold)
        # trained_megafold = copy.deepcopy(megafold)
        # ema_trained_megafold = copy.deepcopy(megafold)

        # trained_megafold.load("checkpoints/collated_(qke0nhjy.1690-qke0nhjy)_megafold.ckpt.1930.pt")  # NOTE: checkpoints do not store PLMs or NLMs
        # ema_trained_megafold.load("checkpoints/collated_(qke0nhjy.1690-qke0nhjy)_megafold.ckpt.1930_EMA.pt")  # NOTE: checkpoints do not store PLMs or NLMs

        # same_untrained_trained_params = []
        # same_untrained_ema_trained_params = []
        # same_trained_ema_trained_params = []
        # for (n1, p1), (n2, p2), (n3, p3) in zip(untrained_megafold.named_parameters(), trained_megafold.named_parameters(), ema_trained_megafold.named_parameters()):
        #     if torch.allclose(p1, p2):
        #         same_untrained_trained_params.append(n1)
        #     if torch.allclose(p1, p3):
        #         same_untrained_ema_trained_params.append(n1)
        #     if torch.allclose(p2, p3):
        #         same_trained_ema_trained_params.append(n2)

        # print(f"Same params between untrained and trained: {len(same_untrained_trained_params)}")
        # print(f"Same params between untrained and EMA trained: {len(same_untrained_ema_trained_params)}")
        # print(f"Same params between trained and EMA trained: {len(same_trained_ema_trained_params)}")

        megafold.plms, megafold.nlms = plms, nlms

        return megafold

    @typecheck
    def shrink_and_perturb_(
        self, shrink_factor: float = 0.5, perturb_factor: float = 0.01
    ) -> "MegaFold":
        """
        Implement Shrink & Perturb.
        By: Ash et al. (https://arxiv.org/abs/1910.08475)

        :param shrink_factor: The shrink factor.
        :param perturb_factor: The perturb factor.
        :return: The perturbed model.
        """
        assert 0.0 <= shrink_factor <= 1.0, "Shrink factor must be between 0 and 1."

        for p in self.parameters():
            noise = torch.randn_like(p.data)
            p.data.mul_(1.0 - shrink_factor).add_(noise * perturb_factor)

        return self

    @typecheck
    def forward_with_megafold_inputs(
        self,
        megafold_inputs: MegaFoldInput | PDBInput | list[MegaFoldInput | PDBInput],
        dtype: str | torch.dtype = torch.float32,
        return_atom_dict: bool = False,
        **kwargs,
    ):
        """Run the forward pass of MegaFold with MegaFoldInputs."""
        if not isinstance(megafold_inputs, list):
            megafold_inputs = [megafold_inputs]

        batched_atom_inputs = megafold_inputs_to_batched_atom_input(
            megafold_inputs, atoms_per_window=self.w
        )

        atom_dict = batched_atom_inputs.dict()
        atom_dict = dict_to_device(atom_dict, device=self.device)
        atom_dict = dict_to_float_dtype(atom_dict, dtype=dtype)

        outputs = self.forward(**atom_dict, **kwargs)

        if return_atom_dict:
            return outputs, atom_dict

        return outputs

    @typecheck
    def run_trunk(
        self,
        *,
        dtype: str | torch.dtype,
        atom_inputs: Float["b m {self.dai}"],  # type: ignore
        atompair_inputs: Float["b m m {self.dapi}"] | Float["b nw {self.w} {self.w*2} {self.dapi}"],  # type: ignore
        additional_molecule_feats: Int[f"b n {ADDITIONAL_MOLECULE_FEATS}"],  # type: ignore
        is_molecule_types: Bool[f"b n {IS_MOLECULE_TYPES}"],  # type: ignore
        molecule_atom_lens: Int["b n"],  # type: ignore
        molecule_ids: Int["b n"],  # type: ignore
        additional_msa_feats: Float["b s n {self.dmf}"] | None = None,  # type: ignore
        additional_token_feats: Float["b n {self.dtf}"] | None = None,  # type: ignore
        atom_ids: Int["b m"] | None = None,  # type: ignore
        atompair_ids: Int["b m m"] | Int["b nw {self.w} {self.w*2}"] | None = None,  # type: ignore
        is_molecule_mod: Bool["b n {self.num_mods}"] | None = None,  # type: ignore
        token_bonds: Bool["b n n"] | None = None,  # type: ignore
        msa: Float["b s n {self.dmi}"] | None = None,  # type: ignore
        msa_mask: Bool["b s"] | None = None,  # type: ignore
        templates: Float["b t n n dt"] | None = None,  # type: ignore
        template_mask: Bool["b t"] | None = None,  # type: ignore
        num_recycling_steps: int = 4,  # NOTE: this value is from the AlphaFold 2 paper, since the AlphaFold 3 paper doesn't list it
        token_constraints: Float["b n n dac"] | None = None,  # type: ignore
        detach_when_recycling: bool = None,
        verbose: bool = False,
        input_independent_baseline: bool = False,
        use_optimized_evo: Literal["deepspeed", "triton"] | None = None,
        chains: Int["b 2"] | None = None,  # type: ignore
        filepath: List[str] | Tuple[str, ...] | None = None,
    ) -> Tuple[
        Float["b m da"],  # type: ignore
        Float["b nw {self.w} {self.w*2} dap"],  # type: ignore
        Float["b n dp"],  # type: ignore
        Float["b n dsi"],  # type: ignore
        Float["b n ds"] | None,  # type: ignore
        Float["b n n ds"] | None,  # type: ignore
        Bool["b n"],  # type: ignore
        Bool["b m m"] | None,  # type: ignore
    ]:
        """Run the main trunk of the model.

        :param dtype: The floating point data type.
        :param atom_inputs: The atom inputs tensor.
        :param atompair_inputs: The atom pair inputs tensor.
        :param additional_molecule_feats: The additional molecule features tensor.
        :param is_molecule_types: The is molecule types tensor.
        :param molecule_atom_lens: The molecule atom lengths tensor.
        :param molecule_ids: The molecule IDs tensor.
        :param additional_msa_feats: The additional multiple sequence alignment features tensor.
        :param additional_token_feats: The additional token features tensor.
        :param atom_ids: The atom IDs tensor.
        :param atompair_ids: The atom pair IDs tensor.
        :param is_molecule_mod: The is molecule modification tensor.
        :param token_bonds: The token bonds tensor.
        :param msa: The multiple sequence alignment tensor.
        :param msa_mask: The multiple sequence alignment mask tensor.
        :param templates: The templates tensor.
        :param template_mask: The template mask tensor.
        :param num_recycling_steps: The number of recycling steps.
        :param token_constraints: The pairwise token constraints tensor.
        :param detach_when_recycling: Whether to detach gradients when recycling.
        :param verbose: Whether to print verbose output.
        :param input_independent_baseline: Whether to use an input-independent baseline.
        :param use_optimized_evo: Whether to use an optimized Evoformer kernel.
        :param chains: The chains tensor.
        :param filepath: The input filepath(s).
        :return: The trunk output tensors.
        """

        # get metadata

        batch_size = molecule_atom_lens.shape[0]
        seq_len = molecule_atom_lens.shape[-1]
        atom_seq_len = atom_inputs.shape[-2]

        if verbose:
            logger.info(
                f"Model input (atom) sequence length ({atom_seq_len}) {seq_len} with precision: {dtype}"
            )

        # embed inputs

        if verbose:
            logger.info("Embedding inputs...")

        (
            single_inputs,
            single_init,
            pairwise_init,
            atom_feats,
            atompair_feats,
        ) = self.input_embedder(
            atom_inputs=atom_inputs,
            atompair_inputs=atompair_inputs,
            additional_token_feats=additional_token_feats,
            molecule_atom_lens=molecule_atom_lens,
            molecule_ids=molecule_ids,
        )

        # handle maybe atom and atompair embeddings

        assert not (
            exists(atom_ids) ^ self.has_atom_embeds
        ), "You either set `num_atom_embeds` and did not pass in `atom_ids` or vice versa"
        assert not (
            exists(atompair_ids) ^ self.has_atompair_embeds
        ), "You either set `num_atompair_embeds` and did not pass in `atompair_ids` or vice versa"

        if self.has_atom_embeds:
            if verbose:
                unique_atom_id, unique_atom_count = atom_ids.unique(return_counts=True)
                atom_types_and_counts = [
                    (ATOMS[unique_atom_id[i]], unique_atom_count[i].item())
                    for i in range(len(unique_atom_id))
                ]
                logger.info(
                    f"Handling atom embeddings from atom types and counts {atom_types_and_counts}..."
                )

            atom_embeds = self.atom_embeds(atom_ids)
            atom_feats = atom_feats + atom_embeds

        bond_mask = None

        if self.has_atompair_embeds:
            atompair_embeds = self.atompair_embeds(atompair_ids)

            bond_mask = atompair_ids == 1

            if atompair_embeds.ndim == 4:
                atompair_embeds = full_pairwise_repr_to_windowed(
                    atompair_embeds, window_size=self.atoms_per_window
                )

            atompair_feats = atompair_feats + atompair_embeds

        # handle maybe molecule modifications

        if verbose:
            logger.info("Handling molecule modifications...")

        assert not (
            exists(is_molecule_mod) ^ self.has_molecule_mod_embeds
        ), "You either set `num_molecule_mods` and did not pass in `is_molecule_mod` or vice versa"

        if self.has_molecule_mod_embeds:
            single_init, seq_unpack_one = pack_one(single_init, "* ds")

            is_molecule_mod, _ = pack_one(is_molecule_mod, "* mods")

            if not is_molecule_mod.is_sparse:
                is_molecule_mod = is_molecule_mod.to_sparse()

            seq_indices, mod_id = is_molecule_mod.indices()
            scatter_values = self.molecule_mod_embeds(mod_id)

            seq_indices = repeat(seq_indices, "n -> n ds", ds=single_init.shape[-1])
            single_init = single_init.scatter_add(0, seq_indices, scatter_values)

            single_init = seq_unpack_one(single_init)

        # handle maybe pairwise token constraint embeddings

        if verbose:
            logger.info("Handling pairwise token constraint embeddings...")

        if exists(self.constraints):
            assert exists(
                token_constraints
            ), "`token_constraints` must be provided to use constraint embeddings."

            for i, constraint in enumerate(self.constraints):
                constraint_slice = slice(i, i + CONSTRAINT_DIMS[constraint])

                token_constraint = torch.where(
                    # replace fixed constraint mask values with learnable mask
                    token_constraints[..., constraint_slice] == CONSTRAINTS_MASK_VALUE,
                    self.learnable_constraint_masks[i],
                    token_constraints[..., constraint_slice],
                )

                pairwise_init = pairwise_init + self.constraint_embeds[i](token_constraint)

        # handle maybe protein language model (PLM) embeddings

        if verbose:
            logger.info("Handling protein language model embeddings...")

        if exists(self.plms):
            # NOTE: PLM embeddings must be created on CPU to reduce GPU memory usage (by ~5 GB)
            aa_ids = torch.where(
                (molecule_ids < 0) | (molecule_ids > NUM_HUMAN_AMINO_ACIDS),
                NUM_HUMAN_AMINO_ACIDS,
                molecule_ids,
            )
            molecule_aa_ids = torch.where(
                is_molecule_types[..., IS_NON_PROTEIN_INDICES].any(dim=-1),
                -1,
                aa_ids,
            )

            with torch.no_grad():
                plm_embeds = []
                for plm in self.plms:
                    plm = plm.to(self.device)  # NOTE: this is a no-op if already on the device
                    plm_embeds.append(plm(molecule_aa_ids).type(dtype))

            # concat all PLM embeddings and project and add to single init

            all_plm_embeds = torch.cat(plm_embeds, dim=-1)
            single_plm_init = self.to_plm_embeds(all_plm_embeds)

            single_init = single_init + single_plm_init

        # handle maybe nucleotide language model (NLM) embeddings

        if verbose:
            logger.info("Handling nucleotide language model embeddings...")

        if exists(self.nlms):
            # NOTE: NLM embeddings must be created on CPU to reduce GPU memory usage (by ~5 GB)
            na_ids = torch.where(
                (molecule_ids < MIN_RNA_NUCLEOTIDE_ID) | (molecule_ids > MAX_DNA_NUCLEOTIDE_ID),
                MISSING_RNA_NUCLEOTIDE_ID,
                molecule_ids,
            )
            molecule_na_ids = torch.where(
                is_molecule_types[..., IS_NON_NA_INDICES].any(dim=-1),
                -1,
                na_ids,
            )

            with torch.no_grad():
                nlm_embeds = []
                for nlm in self.nlms:
                    nlm = nlm.to(self.device)  # NOTE: this is a no-op if already on the device
                    nlm_embeds.append(nlm(molecule_na_ids).type(dtype))

            # concat all NLM embeddings and project and add to single init

            all_nlm_embeds = torch.cat(nlm_embeds, dim=-1)
            single_nlm_init = self.to_nlm_embeds(all_nlm_embeds)

            single_init = single_init + single_nlm_init

        # relative positional encoding

        if verbose:
            logger.info("Applying relative positional encoding...")

        relative_position_encoding = self.relative_position_encoding(
            additional_molecule_feats=additional_molecule_feats
        )

        # only apply relative positional encodings to biomolecules that are chained
        # not to ligands + metal ions

        is_chained_biomol = is_molecule_types[..., IS_BIOMOLECULE_INDICES].any(
            dim=-1
        )  # first three types are chained biomolecules (protein, rna, dna)
        paired_is_chained_biomol = to_pairwise_mask(is_chained_biomol)

        # relative_position_encoding = einx.where(
        #     "b i j, b i j d, -> b i j d", paired_is_chained_biomol, relative_position_encoding, 0.0
        # )
        relative_position_encoding = relative_position_encoding.masked_fill(
            ~paired_is_chained_biomol.unsqueeze(-1), 0.0
        )

        if input_independent_baseline:
            relative_position_encoding *= 0.0

        # add relative positional encoding to pairwise init

        pairwise_init = pairwise_init + relative_position_encoding

        # token bond features

        if verbose:
            logger.info("Applying token bond features...")

        if exists(token_bonds):
            # well do some precautionary standardization
            # (1) mask out diagonal - token to itself does not count as a bond
            # (2) symmetrize, in case it is not already symmetrical (could also throw an error)

            token_bonds = token_bonds | rearrange(token_bonds, "b i j -> b j i")
            diagonal = torch.eye(seq_len, device=self.device, dtype=torch.bool)
            token_bonds = token_bonds.masked_fill(diagonal, False)
        else:
            seq_arange = torch.arange(seq_len, device=self.device)
            # token_bonds = einx.subtract("i, j -> i j", seq_arange, seq_arange).abs() == 1
            token_bonds = (seq_arange[:, None] - seq_arange[None, :]).abs() == 1
            token_bonds = repeat(token_bonds, "i j -> b i j", b=batch_size)

        if input_independent_baseline:
            token_bonds.fill_(False)

        token_bonds_feats = self.token_bond_to_pairwise_feat(token_bonds.type(dtype))

        pairwise_init = pairwise_init + token_bonds_feats

        # molecule mask

        mask = molecule_atom_lens > 0

        # init recycled single and pairwise

        detach_when_recycling = default(detach_when_recycling, self.detach_when_recycling)
        maybe_recycling_detach = torch.detach if detach_when_recycling else identity

        recycled_pairwise = recycled_single = None
        single = pairwise = None

        # for each recycling step

        if verbose:
            logger.info("Starting recycling steps...")

        num_recycling_steps = (
            random.randint(1, num_recycling_steps)  # nosec
            if self.training
            else num_recycling_steps
        )

        for i in range(num_recycling_steps):
            # handle recycled single and pairwise if not first step

            recycled_single = recycled_pairwise = 0.0

            if exists(single):
                single = maybe_recycling_detach(single) 
                # print("single: (K,N)=(384, 384)", single.shape)
                recycled_single = self.recycle_single(single)

            if exists(pairwise):
                pairwise = maybe_recycling_detach(pairwise)
                # print("pairwise: (K,N)=(128,128)", pairwise.shape)
                recycled_pairwise = self.recycle_pairwise(pairwise)

            single = single_init + recycled_single
            pairwise = pairwise_init + recycled_pairwise

            # else go through main transformer trunk from alphafold2

            # templates

            if verbose:
                logger.info(f"Applying template embeddings in recycling step {i}...")

            if not_exists(templates):
                templates = torch.zeros(
                    (batch_size, 1, seq_len, seq_len, self.dim_template_feats),
                    dtype=dtype,
                    device=self.device,
                )
                template_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=self.device)

            if templates.shape[-3] != pairwise.shape[-3]:
                logger.warning(
                    f"Expected templates to have a sequence length of {pairwise.shape[-3]}, but got {templates.shape[-3]} for {filepath} with chains {chains}. Nullifying template features for this example."
                )
                templates = torch.zeros(
                    (batch_size, 1, seq_len, seq_len, self.dim_template_feats),
                    dtype=dtype,
                    device=self.device,
                )
                template_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=self.device)

            # ensure template embedder always contributes to the loss

            embedded_template = self.template_embedder(
                templates=templates,
                template_mask=template_mask,
                pairwise_repr=pairwise,
                mask=mask,
                use_optimized_evo=use_optimized_evo,
            )

            pairwise = embedded_template + pairwise

            # msa

            if verbose:
                logger.info(f"Applying MSA embeddings in recycling step {i}...")

            if exists(msa):
                if msa.shape[-2] != pairwise.shape[-3]:
                    logger.warning(
                        f"Expected MSA to have a sequence length of {pairwise.shape[-3]}, but got {msa.shape[-2]} for {filepath} with chains {chains}. Nullifying MSA features for this example."
                    )
                    msa = torch.zeros(
                        (batch_size, 1, seq_len, self.dmi),
                        dtype=dtype,
                        device=self.device,
                    )
                    msa_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=self.device)

            embedded_msa = self.msa_module(
                msa=msa,
                single_repr=single_inputs,
                pairwise_repr=pairwise,
                msa_mask=msa_mask,
                additional_msa_feats=additional_msa_feats,
                mask=mask,
                use_optimized_evo=use_optimized_evo,
            )

            pairwise = embedded_msa + pairwise

            # main attention trunk (pairformer)

            if verbose:
                logger.info(f"Applying pairformer in recycling step {i}...")

            single, pairwise = self.pairformer(
                single_repr=single,
                pairwise_repr=pairwise,
                mask=mask,
                use_optimized_evo=use_optimized_evo,
            )

            # ensure the recycling weights are always in the computational graph

            if num_recycling_steps == 1:
                single = single + (0.0 * self.recycle_single(torch.zeros_like(single)))
                pairwise = pairwise + (0.0 * self.recycle_pairwise(torch.zeros_like(pairwise)))

        return (
            atom_feats,
            atompair_feats,
            relative_position_encoding,
            single_inputs,
            single,
            pairwise,
            mask,
            bond_mask,
        )

    @typecheck
    def forward(
        self,
        *,
        atom_inputs: Float["b m {self.dai}"],  # type: ignore
        atompair_inputs: Float["b m m {self.dapi}"] | Float["b nw {self.w} {self.w*2} {self.dapi}"],  # type: ignore
        additional_molecule_feats: Int[f"b n {ADDITIONAL_MOLECULE_FEATS}"],  # type: ignore
        is_molecule_types: Bool[f"b n {IS_MOLECULE_TYPES}"],  # type: ignore
        molecule_atom_lens: Int["b n"],  # type: ignore
        molecule_ids: Int["b n"],  # type: ignore
        additional_msa_feats: Float["b s n {self.dmf}"] | None = None,  # type: ignore
        additional_token_feats: Float["b n {self.dtf}"] | None = None,  # type: ignore
        atom_ids: Int["b m"] | None = None,  # type: ignore
        atompair_ids: Int["b m m"] | Int["b nw {self.w} {self.w*2}"] | None = None,  # type: ignore
        is_molecule_mod: Bool["b n {self.num_mods}"] | None = None,  # type: ignore
        atom_mask: Bool["b m"] | None = None,  # type: ignore
        missing_atom_mask: Bool["b m"] | None = None,  # type: ignore
        atom_indices_for_frame: Int["b n 3"] | None = None,  # type: ignore
        valid_atom_indices_for_frame: Bool["b n"] | None = None,  # type: ignore
        atom_parent_ids: Int["b m"] | None = None,  # type: ignore
        token_bonds: Bool["b n n"] | None = None,  # type: ignore
        msa: Float["b s n {self.dmi}"] | None = None,  # type: ignore
        msa_mask: Bool["b s"] | None = None,  # type: ignore
        templates: Float["b t n n dt"] | None = None,  # type: ignore
        template_mask: Bool["b t"] | None = None,  # type: ignore
        num_recycling_steps: int = 4,  # NOTE: this value is from the AlphaFold 2 paper, since the AlphaFold 3 paper doesn't list it
        distogram_atom_indices: Int["b n"] | None = None,  # type: ignore
        molecule_atom_indices: Int["b n"] | None = None,  # type: ignore - the 'token centre atoms' mentioned in the paper, unsure where it is used in the architecture
        num_sample_steps: int | None = None,
        atom_pos: Float["b m 3"] | None = None,  # type: ignore
        resolved_labels: Int["b m"] | None = None,  # type: ignore
        resolution: Float[" b"] | None = None,  # type: ignore
        token_constraints: Float["b n n dac"] | None = None,  # type: ignore
        affinities: List[Float[" *"]] | Tuple[Float[" *"], ...] | None = None,  # type: ignore
        return_loss_breakdown=False,
        return_loss: bool = None,
        return_all_diffused_atom_pos: bool = False,
        return_confidence_head_logits: bool = False,
        return_distogram_head_logits: bool = False,
        return_bio_pdb_structures: bool = False,
        num_rollout_steps: int | None = None,
        rollout_show_tqdm_pbar: bool = False,
        detach_when_recycling: bool = None,
        hard_validate: bool = False,
        call_confidence_head: bool = True,
        input_independent_baseline: bool | None = None,
        use_optimized_evo: Literal["deepspeed", "triton"] | None = None,
        globally_enable_autocasting: bool | None = None,
        verbose: bool | None = None,
        dtype: str | torch.dtype | None = None,
        chains: Int["b 2"] | None = None,  # type: ignore
        num_ligands: List[int] | Tuple[int, ...] | None = None,
        filepath: List[str] | Tuple[str, ...] | None = None,
        example_source: List[str] | Tuple[str, ...] | None = None,
        molecule_atom_perms: List[List[List[int]]] | Tuple[List[List[int]]] | None = None,
    ) -> (
        Float["b m 3"]  # type: ignore
        | List[Structure]
        | Float["ts b m 3"]  # type: ignore
        | Tuple[Float["b m 3"] | List[Structure] | Float["ts b m 3"], ConfidenceHeadLogits | MegaFoldLogits]  # type: ignore
        | Float[""]  # type: ignore
        | Tuple[Float[""], LossBreakdown]  # type: ignore
    ):
        """Run the forward pass of MegaFold.

        :param atom_inputs: The atom inputs tensor.
        :param atompair_inputs: The atom pair inputs tensor.
        :param additional_molecule_feats: The additional molecule features tensor.
        :param is_molecule_types: The is molecule types tensor.
        :param molecule_atom_lens: The molecule atom lengths tensor.
        :param molecule_ids: The molecule IDs tensor.
        :param additional_msa_feats: The additional multiple sequence alignment features tensor.
        :param additional_token_feats: The additional token features tensor.
        :param atom_ids: The atom IDs tensor.
        :param atompair_ids: The atom pair IDs tensor.
        :param is_molecule_mod: The is molecule modification tensor.
        :param atom_mask: The atom mask tensor.
        :param missing_atom_mask: The missing atom mask tensor.
        :param atom_indices_for_frame: The atom indices for frame tensor.
        :param valid_atom_indices_for_frame: The valid atom indices for frame tensor.
        :param atom_parent_ids: The atom parent IDs tensor.
        :param token_bonds: The token bonds tensor.
        :param msa: The multiple sequence alignment tensor.
        :param msa_mask: The multiple sequence alignment mask tensor.
        :param templates: The templates tensor.
        :param template_mask: The template mask tensor.
        :param num_recycling_steps: The number of recycling steps.
        :param distogram_atom_indices: The distogram atom indices tensor.
        :param molecule_atom_indices: The molecule atom indices tensor.
        :param num_sample_steps: The number of sample steps.
        :param atom_pos: The atom positions tensor.
        :param resolved_labels: The resolved labels tensor.
        :param resolution: The resolution tensor.
        :param token_constraints: The pairwise token constraints tensor.
        :param affinities: The optional (fragment) ligand binding affinity values.
        :param return_loss_breakdown: Whether to return the loss breakdown.
        :param return_loss: Whether to return the loss.
        :param return_confidence_head_logits: Whether to return the confidence head logits.
        :param return_distogram_head_logits: Whether to return the distogram head logits.
        :param num_rollout_steps: The number of rollout steps.
        :param rollout_show_tqdm_pbar: Whether to show a tqdm progress bar during rollout.
        :param detach_when_recycling: Whether to detach gradients when recycling.
        :param hard_validate: Whether to hard validate the input tensors.
        :param call_confidence_head: Whether to call the confidence head.
        :param input_independent_baseline: Whether to nullify input features for an input-independent baseline.
        :param use_optimized_evo: Whether to use an optimized Evoformer kernel.
        :param globally_enable_autocasting: Whether to globally enable PyTorch's autocasting for mixed precision.
        :param verbose: Whether to print verbose output.
        :param dtype: The floating point data type.
        :param chains: The chains tensor.
        :param num_ligands: The number of ligands in each batch element.
        :param filepath: The input filepath(s).
        :param example_source: The source dataset of the input example(s).
        :param molecule_atom_perms: The molecule atom permutations.
        :return: The atomic coordinates, the confidence head logits, the distogram head logits, the
            loss, or the loss breakdown.
        """

        # nullify input features for an input-independent baseline

        input_independent_baseline = default(
            input_independent_baseline, self.input_independent_baseline
        )

        if input_independent_baseline:
            logger.warning("Nullifying input features for an input-independent baseline.")
            atom_inputs *= 0.0
            atompair_inputs *= 0.0
            additional_molecule_feats *= 0
            molecule_ids.fill_(-1)
            if exists(additional_msa_feats):
                additional_msa_feats *= 0.0
            if exists(additional_token_feats):
                additional_token_feats *= 0.0
            if exists(atom_ids):
                atom_ids.fill_(len(ATOMS) - 1)
            if exists(atompair_ids):
                atompair_ids.fill_(0)
            if exists(is_molecule_mod):
                is_molecule_mod.fill_(False)
            if exists(token_bonds):
                token_bonds.fill_(False)
            if exists(msa):
                msa *= 0.0
            if exists(templates):
                templates *= 0.0
            if exists(token_constraints):
                token_constraints.fill_(CONSTRAINTS_MASK_VALUE)

        # set up mixed-precision context

        dtype = default(dtype, atom_inputs.dtype)
        globally_enable_autocasting = default(
            globally_enable_autocasting, self.globally_enable_autocasting
        )

        amp_context = (
            torch.autocast(
                device_type="cuda",
                dtype=dtype,
                enabled=globally_enable_autocasting,
                cache_enabled=False,
            )
            if torch.cuda.is_available()
            else nullcontext()
        )

        with amp_context:
            # set defaults

            verbose = default(verbose, self.verbose)
            use_optimized_evo = default(use_optimized_evo, self.use_optimized_evo)

            molecule_atom_perms = (
                list(molecule_atom_perms) if exists(molecule_atom_perms) else None
            )

            # get metadata

            batch_size = molecule_atom_lens.shape[0]
            atom_seq_len = atom_inputs.shape[-2]

            single_structure_input = atom_inputs.shape[0] == 1

            # validate atom and atompair input dimensions

            assert (
                atom_inputs.shape[-1] == self.dim_atom_inputs
            ), f"Expected {self.dim_atom_inputs} for atom_inputs feature dimension, but received {atom_inputs.shape[-1]}"
            assert (
                atompair_inputs.shape[-1] == self.dim_atompair_inputs
            ), f"Expected {self.dim_atompair_inputs} for atompair_inputs feature dimension, but received {atompair_inputs.shape[-1]}"

            # hard validate when debug env variable is turned on

            hard_debug = hard_validate or IS_DEBUGGING

            if hard_debug:
                if verbose:
                    logger.info("Hard validating inputs...")

                maybe(hard_validate_atom_indices_ascending)(
                    distogram_atom_indices, "distogram_atom_indices"
                )
                maybe(hard_validate_atom_indices_ascending)(
                    molecule_atom_indices, "molecule_atom_indices"
                )

                is_biomolecule = ~(
                    (~is_molecule_types[..., IS_BIOMOLECULE_INDICES].any(dim=-1))
                    | (exists(is_molecule_mod) and is_molecule_mod.any(dim=-1))
                )
                maybe(hard_validate_atom_indices_ascending)(
                    atom_indices_for_frame,
                    "atom_indices_for_frame",
                    mask=is_biomolecule,
                )

            # soft validate

            if verbose:
                logger.info("Soft validating inputs...")

            valid_molecule_atom_mask = valid_atom_len_mask = molecule_atom_lens >= 0

            molecule_atom_lens = molecule_atom_lens.masked_fill(~valid_atom_len_mask, 0)

            if exists(molecule_atom_indices):
                valid_molecule_atom_mask = molecule_atom_indices >= 0 & valid_atom_len_mask
                molecule_atom_indices = molecule_atom_indices.masked_fill(
                    ~valid_molecule_atom_mask, 0
                )

            if exists(distogram_atom_indices):
                valid_distogram_mask = distogram_atom_indices >= 0 & valid_atom_len_mask
                distogram_atom_indices = distogram_atom_indices.masked_fill(
                    ~valid_distogram_mask, 0
                )

            if exists(atom_indices_for_frame):
                valid_atom_indices_for_frame = default(
                    valid_atom_indices_for_frame, torch.ones_like(molecule_atom_lens).bool()
                )

                valid_atom_indices_for_frame = (
                    valid_atom_indices_for_frame
                    & (atom_indices_for_frame >= 0).all(dim=-1)
                    & valid_atom_len_mask
                )
                # atom_indices_for_frame = einx.where(
                #     "b n, b n three, -> b n three",
                #     valid_atom_indices_for_frame,
                #     atom_indices_for_frame,
                #     0,
                # )
                atom_indices_for_frame = atom_indices_for_frame.masked_fill(
                    ~valid_atom_indices_for_frame.unsqueeze(-1), 0
                )

            assert exists(molecule_atom_lens) or exists(
                atom_mask
            ), "Either `molecule_atom_lens` or `atom_mask` must be provided."

            if hard_debug:
                assert (
                    molecule_atom_lens >= 0
                ).all(), "molecule_atom_lens must be greater or equal to 0"

            # if atompair inputs are not windowed, window it

            if verbose:
                logger.info("Windowing atompair inputs...")

            is_atompair_inputs_windowed = atompair_inputs.ndim == 5

            if not is_atompair_inputs_windowed:
                atompair_inputs = full_pairwise_repr_to_windowed(
                    atompair_inputs, window_size=self.atoms_per_window
                )

            # handle atom mask

            if not_exists(atom_mask):
                total_atoms = molecule_atom_lens.sum(dim=-1)
                atom_mask = lens_to_mask(total_atoms, max_len=atom_seq_len)

            if exists(missing_atom_mask):
                atom_mask = atom_mask & ~missing_atom_mask

            # run the main trunk

            if verbose:
                logger.info("Running the main trunk...")

            (
                atom_feats,
                atompair_feats,
                relative_position_encoding,
                single_inputs,
                single,
                pairwise,
                mask,
                bond_mask,
            ) = self.run_trunk(
                dtype=dtype,
                atom_inputs=atom_inputs,
                atompair_inputs=atompair_inputs,
                additional_molecule_feats=additional_molecule_feats,
                is_molecule_types=is_molecule_types,
                molecule_atom_lens=molecule_atom_lens,
                molecule_ids=molecule_ids,
                additional_msa_feats=additional_msa_feats,
                additional_token_feats=additional_token_feats,
                atom_ids=atom_ids,
                atompair_ids=atompair_ids,
                is_molecule_mod=is_molecule_mod,
                token_bonds=token_bonds,
                msa=msa,
                msa_mask=msa_mask,
                templates=templates,
                template_mask=template_mask,
                num_recycling_steps=num_recycling_steps,
                token_constraints=token_constraints,
                detach_when_recycling=detach_when_recycling,
                verbose=verbose,
                input_independent_baseline=input_independent_baseline,
                use_optimized_evo=use_optimized_evo,
                chains=chains,
                filepath=filepath,
            )

            # prepare to maybe predict ligand binding affinities

            # NOTE: only ligand atoms are considered for affinity prediction
            res_idx, _, chain_idx, _, _ = additional_molecule_feats.unbind(dim=-1)
            chain_res_idx = create_uid_tensor(chain_idx, res_idx)
            atom_chain_res_idx = batch_repeat_interleave(chain_res_idx, molecule_atom_lens)

            is_ligand_res_atom = batch_repeat_interleave(
                is_molecule_types[..., IS_LIGAND], molecule_atom_lens
            )
            is_ligand_atom_res_idx = torch.full_like(atom_chain_res_idx, -1, dtype=torch.long)
            is_ligand_atom_res_idx[is_ligand_res_atom] = atom_chain_res_idx[is_ligand_res_atom]

            # NOTE: if no ligand atoms are present, randomly select a residue to treat as a surrogate ligand
            num_ligands = default(
                list(num_ligands) if exists(num_ligands) else num_ligands, [1] * batch_size
            )
            for batch_idx in range(batch_size):
                if (is_ligand_atom_res_idx[batch_idx] == -1).all():
                    num_ligands_ = 1
                    num_ligands[batch_idx] = num_ligands_

                    random_chain_res_id = random.sample(
                        chain_res_idx[batch_idx].unique().tolist(), num_ligands_
                    ).pop()  # nosec
                    is_ligand_atom_res_idx[batch_idx][
                        atom_chain_res_idx[batch_idx] == random_chain_res_id
                    ] = random_chain_res_id

            # determine whether to return loss if any labels were to be passed in
            # otherwise will sample the atomic coordinates

            atom_pos_given = exists(atom_pos)

            confidence_head_labels = (
                atom_indices_for_frame,
                resolved_labels,
            )

            can_return_loss = atom_pos_given or exists(resolved_labels)

            # default whether to return loss by whether labels or atom positions are given

            return_loss = default(return_loss, can_return_loss)

            # if neither atom positions or any labels are passed in, sample a structure and return

            if verbose:
                logger.info("Sampling atomic coordinates...")

            if not return_loss:
                sampled_atom_pos = autocasting_disable_decorator(self.disable_sampling_casting)(
                    self.edm.sample
                )(
                    num_sample_steps=num_sample_steps,
                    atom_feats=atom_feats,
                    atompair_feats=atompair_feats,
                    atom_parent_ids=atom_parent_ids,
                    atom_mask=atom_mask,
                    mask=mask,
                    single_trunk_repr=single,
                    single_inputs_repr=single_inputs,
                    pairwise_trunk=pairwise,
                    pairwise_rel_pos_feats=relative_position_encoding,
                    molecule_atom_lens=molecule_atom_lens,
                    is_molecule_types=is_molecule_types,
                    additional_molecule_feats=additional_molecule_feats,
                    return_all_timesteps=return_all_diffused_atom_pos,
                    use_optimized_evo=use_optimized_evo,
                )

                if exists(atom_mask):
                    # sampled_atom_pos = einx.where(
                    #     "b m, ... b m c, -> ... b m c", atom_mask, sampled_atom_pos, 0.0
                    # )
                    sampled_atom_pos = sampled_atom_pos.masked_fill(~atom_mask.unsqueeze(-1), 0.0)

                if return_confidence_head_logits:
                    confidence_head_atom_pos_input = sampled_atom_pos.clone()

                # convert sampled atom positions to Biopython PDB structures

                if return_bio_pdb_structures:
                    assert (
                        not return_all_diffused_atom_pos
                    ), "Cannot return Biopython PDB structures when `return_all_diffused_atom_pos` is set to True."

                    sampled_atom_pos = [
                        protein_structure_from_feature(*args)
                        for args in zip(
                            additional_molecule_feats[..., 2],
                            molecule_ids,
                            molecule_atom_lens,
                            sampled_atom_pos,
                            atom_mask,
                        )
                    ]

                if not return_confidence_head_logits:
                    return sampled_atom_pos

                # ensure confidence head inputs are repeated as necessary

                ch_atom_pos_input = confidence_head_atom_pos_input.detach()

                ch_single = single.detach()
                ch_single_inputs = single_inputs.detach()
                ch_pairwise = pairwise.detach()
                ch_molecule_atom_indices = molecule_atom_indices
                ch_molecule_atom_lens = molecule_atom_lens
                ch_is_ligand_atom_res_idx = is_ligand_atom_res_idx
                ch_atom_feats = atom_feats
                ch_mask = mask
                ch_num_ligands = num_ligands

                if return_all_diffused_atom_pos:
                    # NOTE: for the confidence head, only the last
                    # diffused atom positions are used for sake of memory
                    ch_atom_pos_input = ch_atom_pos_input[0]

                confidence_head_logits = autocasting_disable_decorator(
                    self.disable_confidence_casting
                )(self.confidence_head.__call__)(
                    single_repr=ch_single,
                    single_inputs_repr=ch_single_inputs,
                    pairwise_repr=ch_pairwise,
                    pred_atom_pos=ch_atom_pos_input,
                    molecule_atom_indices=ch_molecule_atom_indices,
                    molecule_atom_lens=ch_molecule_atom_lens,
                    is_ligand_atom_res_idx=ch_is_ligand_atom_res_idx,
                    atom_feats=ch_atom_feats,
                    mask=ch_mask,
                    num_ligands=ch_num_ligands,
                    return_pae_logits=True,
                    use_optimized_evo=use_optimized_evo,
                )

                returned_logits = confidence_head_logits

                if return_distogram_head_logits:
                    distogram_head_logits = autocasting_disable_decorator(
                        self.disable_distogram_casting
                    )(self.distogram_head.__call__)(
                        ch_pairwise.clone(),
                        molecule_atom_lens=ch_molecule_atom_lens,
                        atom_feats=ch_atom_feats,
                    )

                    returned_logits = MegaFoldLogits(
                        **confidence_head_logits._asdict(), distance=distogram_head_logits
                    )

                return sampled_atom_pos, returned_logits

            # if being forced to return loss, but do not have sufficient information to return losses, just return 0

            if return_loss and not can_return_loss:
                zero = self.zero.requires_grad_()

                if not return_loss_breakdown:
                    return zero

                return zero, LossBreakdown(*((zero,) * 11))

            # collect model predictions and labels iteratively

            model_preds = {}
            model_labels = {}

            # distogram head

            if self.train_structure_and_distogram:
                if verbose:
                    logger.info("Calculating distogram logits...")

                model_preds["distogram"] = autocasting_disable_decorator(
                    self.disable_distogram_casting
                )(self.distogram_head.__call__)(
                    pairwise, molecule_atom_lens=molecule_atom_lens, atom_feats=atom_feats
                )
                model_labels["atom_pos"] = (
                    atom_pos if self.disable_distogram_casting else atom_pos.type(dtype)
                )

            # diffusion module

            if self.train_structure_and_distogram:
                if verbose:
                    logger.info("Calculating diffusion predictions...")

                (
                    model_preds["denoised_atom_pos"],
                    model_labels["diffusion_atom_pos_aligned"],
                    model_labels["diffusion_align_weights"],
                    model_labels["diffusion_loss_weights"],
                ) = autocasting_disable_decorator(self.disable_edm_casting)(self.edm.__call__)(
                    atom_pos_ground_truth=atom_pos,
                    additional_molecule_feats=additional_molecule_feats,
                    is_molecule_types=is_molecule_types,
                    atom_feats=atom_feats,
                    atompair_feats=atompair_feats,
                    atom_parent_ids=atom_parent_ids,
                    missing_atom_mask=missing_atom_mask,
                    atom_mask=atom_mask,
                    mask=mask,
                    single_trunk_repr=single,
                    single_inputs_repr=single_inputs,
                    pairwise_trunk=pairwise,
                    pairwise_rel_pos_feats=relative_position_encoding,
                    molecule_atom_lens=molecule_atom_lens,
                    molecule_atom_perms=molecule_atom_perms,
                    nucleotide_loss_weight=self.nucleotide_loss_weight,
                    ligand_loss_weight=self.ligand_loss_weight,
                    filepath=filepath,
                    single_structure_input=single_structure_input,
                    use_optimized_evo=use_optimized_evo,
                    verbose=verbose,
                )

            # confidence head

            pdb_example = exists(example_source) and all(
                source == "pdb" for source in example_source
            )

            sampling_dtype = torch.float32 if self.disable_sampling_casting else dtype

            should_call_confidence_head = any([*map(exists, confidence_head_labels)]) and exists(
                molecule_atom_indices
            )

            if pdb_example and should_call_confidence_head and call_confidence_head:
                if verbose:
                    logger.info("Performing diffusion mini-rollout...")

                # diffusion mini-rollout

                num_rollout_steps = default(num_rollout_steps, self.num_rollout_steps)

                with torch.no_grad():
                    denoised_atom_pos = autocasting_disable_decorator(
                        self.disable_sampling_casting
                    )(self.edm.sample)(
                        num_sample_steps=num_rollout_steps,
                        atom_feats=atom_feats.detach(),
                        atompair_feats=atompair_feats.detach(),
                        atom_mask=atom_mask,
                        mask=mask,
                        single_trunk_repr=single.detach(),
                        single_inputs_repr=single_inputs.detach(),
                        pairwise_trunk=pairwise.detach(),
                        pairwise_rel_pos_feats=relative_position_encoding.detach(),
                        molecule_atom_lens=molecule_atom_lens,
                        is_molecule_types=is_molecule_types,
                        additional_molecule_feats=additional_molecule_feats,
                        use_tqdm_pbar=rollout_show_tqdm_pbar,
                        tqdm_pbar_title="Training rollout",
                        use_optimized_evo=use_optimized_evo,
                    ).detach()
                    model_preds["mini_denoised_atom_pos"] = denoised_atom_pos

                    # structurally align and optimally-permute ground truth structure to match predicted structure

                    if atom_pos_given:
                        # section 3.7.1 equation 2 - weighted rigid aligned ground truth

                        if verbose:
                            logger.info("Calculating weighted rigid aligned ground truth...")

                        align_weights = calculate_weighted_rigid_align_weights(
                            atom_pos=denoised_atom_pos,
                            molecule_atom_lens=molecule_atom_lens,
                            is_molecule_types=is_molecule_types,
                            nucleotide_loss_weight=self.nucleotide_loss_weight,
                            ligand_loss_weight=self.ligand_loss_weight,
                        )

                        try:
                            mini_aligned_atom_pos = weighted_rigid_align(
                                pred_coords=denoised_atom_pos.float(),
                                true_coords=atom_pos.float(),
                                weights=align_weights.float(),
                                mask=atom_mask,
                            ).type(sampling_dtype)
                        except Exception as e:
                            # NOTE: For many (random) unit test inputs, weighted rigid alignment can be unstable
                            logger.warning(f"Skipping weighted rigid alignment due to: {e}")

                        # section 4.2 - multi-chain permutation alignment

                        if (
                            exists(self.multi_chain_permutation_alignment)
                            and single_structure_input
                        ):
                            if verbose:
                                logger.info("Calculating multi-chain permutation alignment...")

                            try:
                                mini_aligned_atom_pos, atom_mask = autocasting_disable_decorator(
                                    self.disable_sampling_casting
                                )(self.multi_chain_permutation_alignment.__call__)(
                                    pred_coords=denoised_atom_pos,
                                    true_coords=mini_aligned_atom_pos,
                                    molecule_atom_lens=molecule_atom_lens,
                                    molecule_atom_indices=molecule_atom_indices,
                                    token_bonds=token_bonds,
                                    additional_molecule_feats=additional_molecule_feats,
                                    is_molecule_types=is_molecule_types,
                                    mask=atom_mask,
                                    verbose=verbose,
                                )
                            except Exception as e:
                                # NOTE: For many (random) unit test inputs, permutation alignment can be unstable
                                logger.warning(
                                    f"Skipping multi-chain permutation alignment {f'for {filepath}' if exists(filepath) else ''} due to: {e}"
                                )

                        # section 4.2 - atom permutation alignment

                        if (
                            exists(self.atom_permutation_alignment)
                            and exists(additional_molecule_feats)
                            and exists(molecule_atom_perms)
                            and single_structure_input
                        ):
                            if verbose:
                                logger.info("Calculating atom permutation alignment...")

                            try:
                                mini_aligned_atom_pos, atom_mask = autocasting_disable_decorator(
                                    self.disable_sampling_casting
                                )(self.atom_permutation_alignment.__call__)(
                                    pred_coords=denoised_atom_pos,
                                    true_coords=mini_aligned_atom_pos,
                                    additional_molecule_feats=additional_molecule_feats,
                                    molecule_atom_lens=molecule_atom_lens,
                                    molecule_atom_perms=molecule_atom_perms,
                                    mask=atom_mask,
                                    verbose=verbose,
                                )
                            except Exception as e:
                                # NOTE: For many (random) unit test inputs, permutation alignment can be unstable
                                logger.warning(
                                    f"Skipping atom permutation alignment {f'for {filepath}' if exists(filepath) else ''} due to: {e}"
                                )

                        model_labels["mini_aligned_atom_pos"] = mini_aligned_atom_pos
                        model_labels["resolved_labels"] = resolved_labels
                        model_labels["resolution"] = resolution
                        model_labels["affinities"] = affinities

            else:
                # skip mini-rollout if the confidence head is not to be learned (i.e., called) in this step

                model_preds["mini_denoised_atom_pos"] = torch.zeros(
                    (batch_size, atom_seq_len, 3), dtype=sampling_dtype, device=self.device
                )
                model_labels["mini_aligned_atom_pos"] = None
                model_labels["resolved_labels"] = None
                model_labels["resolution"] = None
                model_labels["affinities"] = None

            # confidence head logits

            if verbose:
                logger.info("Calculating confidence head logits...")

            ch_logits = autocasting_disable_decorator(self.disable_confidence_casting)(
                self.confidence_head.__call__
            )(
                single_repr=single.detach(),
                single_inputs_repr=single_inputs.detach(),
                pairwise_repr=pairwise.detach(),
                pred_atom_pos=model_preds["mini_denoised_atom_pos"].detach(),
                molecule_atom_indices=molecule_atom_indices,
                molecule_atom_lens=molecule_atom_lens,
                is_ligand_atom_res_idx=is_ligand_atom_res_idx,
                mask=mask,
                atom_feats=atom_feats.detach(),
                num_ligands=num_ligands,
                return_pae_logits=True,
                use_optimized_evo=use_optimized_evo,
            )
            model_preds.update(
                {
                    "pde": ch_logits.pde,
                    "pae": ch_logits.pae,
                    "plddt": ch_logits.plddt,
                    "resolved": ch_logits.resolved,
                    "affinity": ch_logits.affinity,
                }
            )

            # calculate the loss

            loss, loss_breakdown = autocasting_disable_decorator(self.disable_loss_casting)(
                self.loss.__call__
            )(
                model_preds,
                model_labels,
                molecule_atom_lens=molecule_atom_lens,
                is_molecule_types=is_molecule_types,
                atom_mask=atom_mask,
                valid_atom_indices_for_frame=valid_atom_indices_for_frame,
                atom_indices_for_frame=atom_indices_for_frame,
                bond_mask=bond_mask,
                missing_atom_mask=missing_atom_mask,
                distogram_atom_indices=distogram_atom_indices,
                molecule_atom_indices=molecule_atom_indices,
                valid_distogram_mask=valid_distogram_mask,
                valid_molecule_atom_mask=valid_molecule_atom_mask,
            )

            if not return_loss_breakdown:
                return loss

            return loss, loss_breakdown


# an megafold that can download pretrained weights from huggingface


class MegaFoldWithHubMixin(MegaFold, PyTorchModelHubMixin):
    """An MegaFold model that can be loaded from the HuggingFace Hub."""

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str | None,
        cache_dir: str | Path | None,
        force_download: bool,
        proxies: Dict | None,
        resume_download: bool,
        local_files_only: bool,
        token: str | bool | None,
        map_location: str = "cpu",
        strict: bool = False,
        model_filename: str = "megafold.bin",
        **model_kwargs,
    ):
        """Load a pretrained model from the HuggingFace Hub.

        :param model_id: The model ID.
        :param revision: The revision.
        :param cache_dir: The cache directory.
        :param force_download: Whether to force download.
        :param proxies: The proxies.
        :param resume_download: Whether to resume download.
        :param local_files_only: Whether to use local files only.
        :param token: The token.
        :param map_location: The device to map the model to.
        :param strict: Whether to strictly load the model.
        :param model_kwargs: The model keyword arguments.
        """
        model_file = Path(model_id) / model_filename

        if not model_file.exists():
            model_file = hf_hub_download(
                repo_id=model_id,
                filename=model_filename,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )

        model = cls.init_and_load(model_file, strict=strict, map_location=map_location)

        return model

