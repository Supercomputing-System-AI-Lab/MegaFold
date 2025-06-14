import importlib.metadata
from collections import defaultdict
from contextlib import nullcontext
from functools import partial, wraps

# import einx
import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype.typing import Any, Callable, Dict, List, NamedTuple, Tuple, Union
from einops import einsum, pack, rearrange, reduce, repeat, unpack
from loguru import logger
from torch import Tensor, autocast, is_floating_point, is_tensor, tensor
from torch.nn import Identity, Linear, Module
from torch.utils._pytree import tree_map

from omnifold.noise import HarmonicSDE
from omnifold.tensor_typing import Bool, Float, Int, Shaped, checkpoint, typecheck
from omnifold.utils.utils import default, exists, not_exists

# constants

Shape = Union[Tuple[int, ...], List[int]]

IS_PROTEIN_INDEX = 0
IS_RNA_INDEX = 1
IS_DNA_INDEX = 2

ADDITIONAL_MOLECULE_FEATS = 5
IS_MOLECULE_TYPES = 5

# helper functions

# default scheduler used in paper w/ warmup


@typecheck
def default_lambda_lr_fn(steps: int) -> float:
    """Default lambda learning rate function.

    :param steps: The number of steps taken.
    :return: The learning rate.
    """
    # 1000 step warmup
    if steps < 1000:
        return steps / 1000

    # decay 0.95 every 5e4 steps
    steps -= 1000
    return 0.95 ** (steps / 5e4)


@typecheck
def distance_to_dgram(
    distance: Float["... dist"],  # type: ignore
    bins: Float[" bins"],  # type: ignore
    return_labels: bool = False,
) -> Int["... dist"] | Int["... dist bins"]:  # type: ignore
    """Converting from distance to discrete bins, e.g., for distance_labels and pae_labels using
    the same logic as OpenFold.

    :param distance: The distance tensor.
    :param bins: The bins tensor.
    :param return_labels: Whether to return the labels.
    :return: The one-hot bins tensor or the bin labels.
    """

    distance = distance.abs()

    bins = F.pad(bins, (0, 1), value=float("inf"))
    low, high = bins[:-1], bins[1:]

    # one_hot = (
    #     einx.greater_equal("..., bin_low -> ... bin_low", distance, low)
    #     & einx.less("..., bin_high -> ... bin_high", distance, high)
    # ).long()
    one_hot = ((distance.unsqueeze(-1) >= low) & (distance.unsqueeze(-1) < high)).long()

    if return_labels:
        return one_hot.argmax(dim=-1)

    return one_hot


@typecheck
def offset_only_positive(t: Tensor, offset: Tensor) -> Tensor:
    """Offset a Tensor only if it is positive."""
    is_positive = t >= 0
    t_offsetted = t + offset
    return torch.where(is_positive, t_offsetted, t)


@typecheck
def l2norm(t: Tensor, eps: float = 1e-20, dim: int = -1) -> Tensor:
    """Perform an L2 normalization on a Tensor.

    :param t: The Tensor.
    :param eps: The epsilon value.
    :param dim: The dimension to normalize over.
    :return: The L2 normalized Tensor.
    """
    return F.normalize(t, p=2, eps=eps, dim=dim)


@typecheck
def max_neg_value(t: Tensor) -> float:
    """Get the maximum negative value of Tensor based on its `dtype`.

    :param t: The Tensor.
    :return: The maximum negative value of its `dtype`.
    """
    return -torch.finfo(t.dtype).max


def dict_to_device(d: dict, device: str | torch.device) -> dict:
    """Move a dictionary of tensors to a device.

    :param d: The dictionary of tensors.
    :param device: The device to move to.
    :return: The dictionary of tensors on the device.
    """
    return tree_map(lambda t: t.to(device) if is_tensor(t) else t, d)


def dict_to_float_dtype(d: dict, dtype: str | torch.dtype) -> dict:
    """Cast a dictionary of tensors to a new float type.

    :param d: The dictionary of tensors.
    :param dtype: The dtype to cast to.
    :return: The dictionary of tensors with the new float dtype.
    """
    return tree_map(lambda t: t.type(dtype) if is_tensor(t) and is_floating_point(t) else t, d)


@typecheck
def log(t: Tensor, eps=1e-20) -> Tensor:
    """Run a safe log function that clamps the input to be above `eps` to avoid `log(0)`.

    :param t: The input tensor.
    :param eps: The epsilon value.
    :return: Tensor in the log domain.
    """
    return torch.log(t.clamp(min=eps))


@typecheck
def divisible_by(num: int, den: int) -> bool:
    """Check if a number is divisible by another number.

    :param num: The numerator.
    :param den: The denominator.
    :return: True if `num` is divisible by `den`, False otherwise.
    """
    return (num % den) == 0


@typecheck
def at_most_one_of(*flags: bool) -> bool:
    """Check if at most one of the flags is True.

    :param flags: The flags to check.
    :return: True if at most one of the flags is True, False otherwise.
    """
    return sum([*map(int, flags)]) <= 1


@typecheck
def compact(*args):
    """Compact a tuple of objects by removing any `None` values.

    :param args: The objects to compact.
    :return: The compacted objects.
    """
    return tuple(filter(exists, args))


@typecheck
def cast_tuple(t: Any, length: int = 1) -> Tuple[Any, ...]:
    """Cast an object to a tuple of objects with the given length.

    :param t: The object to cast.
    :param length: The length of the tuple.
    :return: The casted tuple.
    """
    return t if isinstance(t, tuple) else ((t,) * length)


@typecheck
def pack_one(t: Tensor, pattern: str) -> Tuple[Tensor, Callable]:
    """Pack a single tensor into a tuple of tensors with the given pattern.

    :param t: The tensor to pack.
    :param pattern: The pattern with which to pack.
    :return: The packed tensor and the unpack function.
    """
    packed, ps = pack([t], pattern)

    def unpack_one(to_unpack, unpack_pattern=None):
        """Unpack a single tensor.

        :param to_unpack: The tensor to unpack.
        :param pattern: The pattern with which to unpack.
        :return: The unpacked tensor.
        """
        (unpacked,) = unpack(to_unpack, ps, default(unpack_pattern, pattern))
        return unpacked

    return packed, unpack_one


@typecheck
def softclamp(t: Tensor, value: float) -> Tensor:
    """Perform a soft clamp on a Tensor.

    :param t: The Tensor.
    :param value: The value to clamp to.
    :return: The soft clamped Tensor
    """
    return (t / value).tanh() * value


@typecheck
def exclusive_cumsum(t: Tensor, dim: int = -1) -> Tensor:
    """Perform an exclusive cumulative summation on a Tensor.

    :param t: The Tensor.
    :param dim: The dimension to sum over.
    :return: The exclusive cumulative sum Tensor.
    """
    return t.cumsum(dim=dim) - t


@typecheck
def symmetrize(t: Float["b n n ..."]) -> Float["b n n ..."]:  # type: ignore
    """Symmetrize a Tensor.

    :param t: The Tensor.
    :return: The symmetrized Tensor.
    """
    return t + rearrange(t, "b i j ... -> b j i ...")


@typecheck
def freeze_(m: Module):
    """Freeze a module.

    :param m: The module to freeze.
    """
    for p in m.parameters():
        p.requires_grad = False


@typecheck
def clamp_tensor(value: torch.Tensor, min: float = 1e-6, max: float = 1 - 1e-6) -> torch.Tensor:
    """Set the upper and lower bounds of a tensor via clamping.

    :param value: The tensor to clamp.
    :param min: The minimum value to clamp to. Default is `1e-6`.
    :param max: The maximum value to clamp to. Default is `1 - 1e-6`.
    :return: The clamped tensor.
    """
    return value.clamp(min=min, max=max)


@typecheck
def get_model_params_repr(model: nn.Module) -> float | torch.Tensor:
    """Get an average representation of the model parameters.

    :param model: The model.
    :return: The average representation of the model parameters.
    """
    return sum(p.mean() for p in model.parameters()) / len(list(model.parameters()))


# decorators


@typecheck
def maybe(fn):
    """Decorator to check if a Tensor exists before running a function on it."""

    @wraps(fn)
    def inner(t, *args, **kwargs):
        """Inner function to check if a Tensor exists before running a function on it."""
        if not_exists(t):
            return None
        return fn(t, *args, **kwargs)

    return inner


@typecheck
def pad_at_dim(t, pad: Tuple[int, int], *, dim=-1, value=0.0) -> Tensor:
    """Pad a Tensor at a specific dimension.

    :param t: The Tensor.
    :param pad: The padding.
    :param dim: The dimension to pad.
    :param value: The value to pad with.
    :return: The padded Tensor.
    """
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), value=value)


# padding and slicing


@typecheck
def slice_at_dim(t: Tensor, dim_slice: slice, *, dim: int) -> Tensor:
    """Slice a Tensor at a specific dimension.

    :param t: The Tensor.
    :param dim_slice: The slice object.
    :param dim: The dimension to slice.
    :return: The sliced Tensor.
    """
    dim += t.ndim if dim < 0 else 0
    colons = [slice(None)] * t.ndim
    colons[dim] = dim_slice
    return t[tuple(colons)]


@typecheck
def pad_to_length(t: Tensor, length: int, *, dim: int = -1, value=0) -> Tensor:
    """Pad a Tensor to a specific length at a specific dimension.

    :param t: The Tensor.
    :param length: The length to pad to.
    :param dim: The dimension to pad.
    :param value: The value to pad with.
    :return: The padded Tensor.
    """
    padding = max(length - t.shape[dim], 0)

    if padding == 0:
        return t

    return pad_at_dim(t, (0, padding), dim=dim, value=value)


@typecheck
def pad_or_slice_to(t: Tensor, length: int, *, dim: int, pad_value=0) -> Tensor:
    """Pad or slice a Tensor to a specific length at a specific dimension.

    :param t: The Tensor.
    :param length: The length to pad or slice to.
    :param dim: The dimension to pad or slice.
    :param pad_value: The value to pad with.
    :return: The padded or sliced Tensor.
    """
    curr_length = t.shape[dim]

    if curr_length < length:
        t = pad_to_length(t, length, dim=dim, value=pad_value)
    elif curr_length > length:
        t = slice_at_dim(t, slice(0, length), dim=dim)

    return t


@typecheck
def pad_to_multiple(t: Tensor, multiple: int, *, dim=-1, value=0.0) -> Tensor:
    """Pad a Tensor to a multiple of a specific number at a specific dimension.

    :param t: The Tensor.
    :param multiple: The multiple to pad to.
    :param dim: The dimension to pad.
    :param value: The value to pad with.
    :return: The padded Tensor.
    """
    seq_len = t.shape[dim]
    padding_needed = (multiple - (seq_len % multiple)) % multiple

    if padding_needed == 0:
        return t

    return pad_at_dim(t, (0, padding_needed), dim=dim, value=value)


@typecheck
def concat_previous_window(t: Tensor, *, dim_seq: int, dim_window: int) -> Tensor:
    """Concatenate the previous window of a Tensor.

    :param t: The Tensor.
    :param dim_seq: The sequence dimension.
    :param dim_window: The window dimension.
    :return: The concatenated Tensor.
    """
    t = pad_at_dim(t, (1, 0), dim=dim_seq, value=0.0)

    t = torch.cat(
        (
            slice_at_dim(t, slice(None, -1), dim=dim_seq),
            slice_at_dim(t, slice(1, None), dim=dim_seq),
        ),
        dim=dim_window,
    )

    return t


@typecheck
def pad_and_window(t: Float["b n ..."] | Int["b n ..."], window_size: int) -> Tensor:  # type: ignore
    """Pad and window a Tensor.

    :param t: The Tensor.
    :param window_size: The window size.
    :return: The padded and windowed Tensor.
    """
    t = pad_to_multiple(t, window_size, dim=1)
    t = rearrange(t, "b (n w) ... -> b n w ...", w=window_size)
    return t


# packed atom representation functions


@typecheck
def lens_to_mask(
    lens: Int["b ..."], max_len: int | None = None  # type: ignore
) -> Bool["... m"]:  # type: ignore
    """Convert a Tensor of lengths to a mask Tensor.

    :param lens: The lengths Tensor.
    :param max_len: The maximum length.
    :return: The mask Tensor.
    """
    device = lens.device
    if not_exists(max_len):
        max_len = lens.amax()
    arange = torch.arange(max_len, device=device)
    # return einx.less("m, ... -> ... m", arange, lens)
    return arange.unsqueeze(0) < lens.unsqueeze(-1)


@typecheck
def mean_pool_with_lens(
    feats: Float["b m d"],  # type: ignore
    lens: Int["b n"],  # type: ignore
) -> Float["b n d"]:  # type: ignore
    """Mean pool with lens.

    :param feats: The features tensor.
    :param lens: The lengths tensor.
    :return: The mean pooled features tensor.
    """
    summed, mask = sum_pool_with_lens(feats, lens)
    # avg = einx.divide("b n d, b n", summed, lens.clamp(min=1))
    # avg = einx.where("b n, b n d, -> b n d", mask, avg, 0.0)
    avg = summed / lens.clamp(min=1).unsqueeze(-1)
    avg = avg.masked_fill(~mask.unsqueeze(-1), 0.0)
    return avg


@typecheck
def sum_pool_with_lens(
    feats: Float["b m d"],  # type: ignore
    lens: Int["b n"],  # type: ignore
) -> tuple[Float["b n d"], Bool["b n"]]:  # type: ignore
    """Sum pool with lens.

    :param feats: The features tensor.
    :param lens: The lengths tensor.
    :return: The summed features tensor and the mask tensor.
    """
    seq_len = feats.shape[1]

    mask = lens > 0
    assert (
        lens.sum(dim=-1) <= seq_len
    ).all(), (
        "One of the lengths given exceeds the total sequence length of the features passed in."
    )

    cumsum_feats = feats.cumsum(dim=1)
    cumsum_feats = F.pad(cumsum_feats, (0, 0, 1, 0), value=0.0)

    cumsum_indices = lens.cumsum(dim=1)
    cumsum_indices = F.pad(cumsum_indices, (1, 0), value=0)

    # sel_cumsum = einx.get_at('b [m] d, b n -> b n d', cumsum_feats, cumsum_indices)

    cumsum_indices = repeat(cumsum_indices, "b n -> b n d", d=cumsum_feats.shape[-1])
    sel_cumsum = cumsum_feats.gather(-2, cumsum_indices)

    # subtract cumsum at one index from the previous one
    summed = sel_cumsum[:, 1:] - sel_cumsum[:, :-1]

    return summed, mask


@typecheck
def mean_pool_fixed_windows_with_mask(
    feats: Float["b m d"],  # type: ignore
    mask: Bool["b m"],  # type: ignore
    window_size: int,
    return_mask_and_inverse: bool = False,
) -> Float["b n d"] | Tuple[Float["b n d"], Bool["b n"], Callable[[Float["b m d"]], Float["b n d"]]]:  # type: ignore
    """Mean pool fixed windows with a mask.

    :param feats: The features tensor.
    :param mask: The mask tensor.
    :param window_size: The window size.
    :param return_mask_and_inverse: Whether to return the mask and inverse function.
    :return: The mean pooled features tensor.
    """
    seq_len = feats.shape[-2]
    assert divisible_by(seq_len, window_size)

    # feats = einx.where("b m, b m d, -> b m d", mask, feats, 0.0)
    feats = feats.masked_fill(~mask[..., None], 0.0)

    num = reduce(feats, "b (n w) d -> b n d", "sum", w=window_size)
    den = reduce(mask.float(), "b (n w) -> b n 1", "sum", w=window_size)

    avg = num / den.clamp(min=1.0)

    if not return_mask_and_inverse:
        return avg

    pooled_mask = reduce(mask, "b (n w) -> b n", "any", w=window_size)

    @typecheck
    def inverse_fn(pooled: Float["b n d"]) -> Float["b m d"]:  # type: ignore
        """Unpool the pooled features tensor."""
        unpooled = repeat(pooled, "b n d -> b (n w) d", w=window_size)
        # unpooled = einx.where("b m, b m d, -> b m d", mask, unpooled, 0.0)
        unpooled = unpooled.masked_fill(~mask[..., None], 0.0)
        return unpooled

    return avg, pooled_mask, inverse_fn


@typecheck
def batch_repeat_interleave(
    feats: Float["b n ..."] | Bool["b n ..."] | Bool["b n"] | Int["b n"],  # type: ignore
    lens: Int["b n"],  # type: ignore
    output_padding_value: (
        float | int | bool | None
    ) = None,  # NOTE: this value determines what the output padding value will be
) -> Float["b m ..."] | Bool["b m ..."] | Bool["b m"] | Int["b m"]:  # type: ignore
    """Batch repeat and interleave a sequence of features.

    :param feats: The features tensor.
    :param lens: The lengths tensor.
    :param output_padding_value: The output padding value.
    :return: The batch repeated and interleaved features tensor.
    """
    device, dtype = feats.device, feats.dtype

    batch, seq, *dims = feats.shape

    # get mask from lens

    mask = lens_to_mask(lens)

    # derive arange

    window_size = mask.shape[-1]
    arange = torch.arange(window_size, device=device)

    offsets = exclusive_cumsum(lens)
    # indices = einx.add("w, b n -> b n w", arange, offsets)
    indices = arange[None, None, :] + offsets[..., None]

    # create output tensor + a sink position on the very right (index max_len)

    total_lens = lens.clamp(min=0).sum(dim=-1)
    output_mask = lens_to_mask(total_lens)

    max_len = total_lens.amax()

    output_indices = torch.zeros((batch, max_len + 1), device=device, dtype=torch.long)

    indices = indices.masked_fill(~mask, max_len)  # scatter to sink position for padding
    indices = rearrange(indices, "b n w -> b (n w)")

    # scatter

    seq_arange = torch.arange(seq, device=device)
    seq_arange = repeat(seq_arange, "n -> b (n w)", b=batch, w=window_size)

    # output_indices = einx.set_at('b [m], b nw, b nw -> b [m]', output_indices, indices, seq_arange)

    output_indices = output_indices.scatter(1, indices, seq_arange)

    # remove sink

    output_indices = output_indices[:, :-1]

    # gather

    # output = einx.get_at('b [n] ..., b m -> b m ...', feats, output_indices)

    feats, unpack_one = pack_one(feats, "b n *")
    output_indices = repeat(output_indices, "b m -> b m d", d=feats.shape[-1])
    output = feats.gather(1, output_indices)
    output = unpack_one(output)

    # set output padding value

    output_padding_value = default(output_padding_value, False if dtype == torch.bool else 0)

    # output = einx.where("b n, b n ..., -> b n ...", output_mask, output, output_padding_value)
    output = output.masked_fill(
        ~output_mask[(...,) + (None,) * (output.dim() - output_mask.dim())],
        output_padding_value,
    )

    return output


@typecheck
def batch_repeat_interleave_pairwise(
    pairwise: Float["b n n d"],  # type: ignore
    molecule_atom_lens: Int["b n"],  # type: ignore
) -> Float["b m m d"]:  # type: ignore
    """Batch repeat and interleave a sequence of pairwise features."""
    pairwise = batch_repeat_interleave(pairwise, molecule_atom_lens)

    molecule_atom_lens = repeat(molecule_atom_lens, "b ... -> (b r) ...", r=pairwise.shape[1])
    pairwise, unpack_one = pack_one(pairwise, "* n d")
    pairwise = batch_repeat_interleave(pairwise, molecule_atom_lens)
    return unpack_one(pairwise)


@typecheck
def to_pairwise_mask(
    mask_i: Bool["... n"],  # type: ignore
    mask_j: Bool["... n"] | None = None,  # type: ignore
) -> Bool["... n n"]:  # type: ignore
    """Convert two masks into a pairwise mask.

    :param mask_i: The first mask.
    :param mask_j: The second mask.
    :return: The pairwise mask.
    """
    mask_j = default(mask_j, mask_i)
    assert mask_i.shape == mask_j.shape
    # return einx.logical_and("... i, ... j -> ... i j", mask_i, mask_j)
    return mask_i.unsqueeze(-1) & mask_j.unsqueeze(-2)


@typecheck
def masked_average(
    t: Shaped["..."],  # type: ignore
    mask: Shaped["..."],  # type: ignore
    *,
    dim: int | Tuple[int, ...],
    eps=1.0,
) -> Float["..."]:  # type: ignore
    """Compute the masked average of a Tensor.

    :param t: The Tensor.
    :param mask: The mask.
    :param dim: The dimension(s) to average over.
    :param eps: The epsilon value.
    :return: The masked average.
    """
    num = (t * mask).sum(dim=dim)
    den = mask.sum(dim=dim)
    return num / den.clamp(min=eps)


@typecheck
def remove_consecutive_duplicate(
    t: Int["n ..."], remove_to_value: int = -1  # type: ignore
) -> Int["n ..."]:  # type: ignore
    """Remove consecutive duplicates from a Tensor.

    :param t: The Tensor.
    :param remove_to_value: The value to remove to.
    :return: The Tensor with consecutive duplicates removed.
    """
    is_duplicate = t[1:] == t[:-1]

    if is_duplicate.ndim == 2:
        is_duplicate = is_duplicate.all(dim=-1)

    is_duplicate = F.pad(is_duplicate, (1, 0), value=False)
    # return einx.where("n, n ..., -> n ... ", ~is_duplicate, t, remove_to_value)
    return torch.where(~is_duplicate[..., None], t, remove_to_value)


@typecheck
def batch_compute_rmsd(
    true_pos: Float["b a 3"],  # type: ignore
    pred_pos: Float["b a 3"],  # type: ignore
    mask: Bool["b a"] | None = None,  # type: ignore
    eps: float = 1e-6,
) -> Float["b"]:  # type: ignore
    """Calculate the root-mean-square deviation (RMSD) between predicted and ground truth
    coordinates.

    :param true_pos: The ground truth coordinates.
    :param pred_pos: The predicted coordinates.
    :param mask: The mask tensor.
    :param eps: A small value to prevent division by zero.
    :return: The RMSD.
    """
    # Apply mask if provided
    if exists(mask):
        # true_coords = einx.where("b a, b a c, -> b a c", mask, true_pos, 0.0)
        # pred_coords = einx.where("b a, b a c, -> b a c", mask, pred_pos, 0.0)
        true_coords = true_pos.masked_fill(~mask[..., None], 0.0)
        pred_coords = pred_pos.masked_fill(~mask[..., None], 0.0)

    # Compute squared differences across the last dimension (which is of size 3)
    sq_diff = torch.square(true_coords - pred_coords).sum(dim=(-1, -2))  # [b]

    # Compute mean squared deviation per batch
    msd = sq_diff / mask.sum(-1)  # [b]

    # Replace NaN values with a large number to avoid issues
    msd = torch.nan_to_num(msd, nan=1e8)

    # Return the root mean square deviation per batch
    return torch.sqrt(msd + eps)  # [b]


@typecheck
@autocast(device_type="cuda", enabled=False, cache_enabled=False)
def batch_compute_rigid_alignment(
    true_pos: Float["a 3"] | Float["b a 3"],  # type: ignore
    pred_pos: Float["b a 3"],  # type: ignore
    mask: Bool["a"] | Bool["b a"] | None = None,  # type: ignore
    eps: float = 1e-6,
) -> Tuple[Float["b"], Float["b a 3"]]:  # type: ignore
    """Optimally aligns predicted coordinates to ground truth coordinates.

    :param true_pos: The ground truth coordinates.
    :param pred_pos: The predicted coordinates.
    :param mask: The mask tensor.
    :param eps: A small value to prevent division by zero.
    :return: The RMSD and its corresponding (aligned) predicted coordinates.
    """
    batch_size, dtype = pred_pos.shape[0], pred_pos.dtype

    # Expand batch size of true features if necessary
    if true_pos.ndim < pred_pos.ndim:
        true_pos = repeat(true_pos, "a three -> b a three", b=batch_size)
    elif true_pos.shape[0] != batch_size:
        true_pos = repeat(true_pos, "n a three -> (b n) a three", b=batch_size)
        assert (
            true_pos.shape[0] == pred_pos.shape[0]
        ), "Batch size mismatch between true and predicted coordinates in `batch_compute_rigid_alignment()`."

    if exists(mask) and mask.ndim < pred_pos.ndim - 1:
        mask = repeat(mask, "a -> b a", b=batch_size)
    elif exists(mask) and mask.shape[0] != batch_size:
        mask = repeat(mask, "n a -> (b n) a", b=batch_size)
        assert (
            mask.shape[0] == pred_pos.shape[0]
        ), "Batch size mismatch between mask and predicted coordinates in `batch_compute_rigid_alignment()`."

    # Apply mask if provided
    if exists(mask):
        # true_coords = einx.where("b a, b a c, -> b a c", mask, true_pos, 0.0)
        # pred_coords = einx.where("b a, b a c, -> b a c", mask, pred_pos, 0.0)
        true_coords = true_pos.masked_fill(~mask[..., None], 0.0)
        pred_coords = pred_pos.masked_fill(~mask[..., None], 0.0)

    # Optimally align predicted coordinates to ground truth coordinates
    aligned_pred_coords = weighted_rigid_align(
        # NOTE: `weighted_rigid_align` returns the aligned version of `true_coords`
        pred_coords=true_coords.float(),
        true_coords=pred_coords.float(),
        mask=mask,
    ).type(dtype)

    # Return the root mean square deviation per batch and its associated (aligned) predicted coordinates
    return batch_compute_rmsd(true_coords, aligned_pred_coords, mask, eps), aligned_pred_coords


@typecheck
def calculate_weighted_rigid_align_weights(
    atom_pos: Float["b m 3"],  # type: ignore
    molecule_atom_lens: Int["b n"],  # type: ignore
    is_molecule_types: Bool["b n ..."] | None = None,  # type: ignore
    nucleotide_loss_weight: float = 5.0,
    ligand_loss_weight: float = 10.0,
) -> Float["b m"]:  # type: ignore
    """Calculate the weighted rigid alignment weights.

    :param atom_pos: Reference atom positions.
    :param molecule_atom_lens: The molecule atom lengths.
    :param is_molecule_types: The molecule types.
    :param nucleotide_loss_weight: The nucleotide loss weight.
    :param ligand_loss_weight: The ligand loss weight.
    :return: The weighted rigid alignment weights.
    """

    align_weights = atom_pos.new_ones(atom_pos.shape[:2])

    if exists(is_molecule_types):
        is_nucleotide_or_ligand_fields = is_molecule_types.unbind(dim=-1)

        is_nucleotide_or_ligand_fields = tuple(
            batch_repeat_interleave(t, molecule_atom_lens) for t in is_nucleotide_or_ligand_fields
        )
        is_nucleotide_or_ligand_fields = tuple(
            pad_or_slice_to(t, length=align_weights.shape[-1], dim=-1)
            for t in is_nucleotide_or_ligand_fields
        )

        _, atom_is_dna, atom_is_rna, atom_is_ligand, _ = is_nucleotide_or_ligand_fields

        # section 3.7.1 equation 4

        # upweighting of nucleotide and ligand atoms is additive per equation 4

        align_weights = torch.where(
            atom_is_dna | atom_is_rna,
            1 + nucleotide_loss_weight,
            align_weights,
        )
        align_weights = torch.where(atom_is_ligand, 1 + ligand_loss_weight, align_weights)

    return align_weights


@typecheck
@autocast(device_type="cuda", enabled=False, cache_enabled=False)
def weighted_rigid_align(
    pred_coords: Float["b m 3"],  # type: ignore - predicted coordinates
    true_coords: Float["b m 3"],  # type: ignore - true coordinates
    weights: Float["b m"] | None = None,  # type: ignore - weights for each atom
    mask: Bool["b m"] | None = None,  # type: ignore - mask for variable lengths
    return_transforms: bool = False,
) -> Union[Float["b m 3"], Tuple[Float["b m 3"], Float["b 3 3"], Float["b 1 3"]]]:  # type: ignore
    """Compute the weighted rigid alignment following Algorithm 28 of the AlphaFold 3 supplement.

    The check for ambiguous rotation and low rank of cross-correlation between aligned point clouds
    is inspired by
    https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/points_alignment.html.

    :param pred_coords: Predicted coordinates.
    :param true_coords: True coordinates.
    :param weights: Weights for each atom.
    :param mask: The mask for variable lengths.
    :param return_transform: Whether to return the transformation matrix.
    :return: The optimally aligned true coordinates.
    """

    batch_size, num_points, dim = pred_coords.shape

    if not_exists(weights):
        # if no weights are provided, assume uniform weights
        weights = torch.ones_like(pred_coords[..., 0])

    if exists(mask):
        # zero out all predicted and true coordinates where not an atom
        # pred_coords = einx.where("b n, b n c, -> b n c", mask, pred_coords, 0.0)
        # true_coords = einx.where("b n, b n c, -> b n c", mask, true_coords, 0.0)
        # weights = einx.where("b n, b n, -> b n", mask, weights, 0.0)
        pred_coords = pred_coords * mask[..., None]
        true_coords = true_coords * mask[..., None]
        weights = weights * mask

    # Take care of weights broadcasting for coordinate dimension
    weights = rearrange(weights, "b n -> b n 1")

    # Compute weighted centroids
    true_centroid = (true_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    )
    pred_centroid = (pred_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    )

    # Center the coordinates
    true_coords_centered = true_coords - true_centroid
    pred_coords_centered = pred_coords - pred_centroid

    if num_points < (dim + 1):
        logger.warning(
            "Warning: The size of one of the point clouds is <= dim+1. "
            + "`weighted_rigid_align()` cannot return a unique rotation."
        )

    # Compute the weighted covariance matrix
    cov_matrix = einsum(
        weights * true_coords_centered, pred_coords_centered, "b n i, b n j -> b i j"
    )

    # Compute the SVD of the covariance matrix
    U, S, V = torch.svd(cov_matrix)
    U_T = U.transpose(-2, -1)

    # Catch ambiguous rotation by checking the magnitude of singular values
    if (S.abs() <= 1e-15).any() and not (num_points < (dim + 1)):
        logger.warning(
            "Warning: Excessively low rank of "
            + "cross-correlation between aligned point clouds. "
            + "`weighted_rigid_align()` cannot return a unique rotation."
        )

    det = torch.det(einsum(V, U_T, "b i j, b j k -> b i k"))

    # Ensure proper rotation matrix with determinant 1
    diag = torch.eye(dim, dtype=det.dtype, device=det.device)
    diag = repeat(diag, "i j -> b i j", b=batch_size).clone()

    diag[:, -1, -1] = det
    rot_matrix = einsum(V, diag, U_T, "b i j, b j k, b k l -> b i l")

    # Apply the rotation and translation
    true_aligned_coords = (
        einsum(rot_matrix, true_coords_centered, "b i j, b n j -> b n i") + pred_centroid
    )
    true_aligned_coords.detach_()

    if return_transforms:
        translation = pred_centroid
        return true_aligned_coords, rot_matrix, translation

    return true_aligned_coords


# checkpointing utils


@typecheck
def should_checkpoint(
    self: Module,
    inputs: Tensor | Tuple[Tensor, ...],
    check_instance_variable: str | None = "checkpoint",
) -> bool:
    """Determine if activation checkpointing should be used.

    :param self: The module.
    :param inputs: The inputs.
    :param check_instance_variable: The instance variable to check.
    :return: True if activation checkpointing should be used, False otherwise.
    """
    if is_tensor(inputs):
        inputs = (inputs,)

    return (
        self.training
        and any([i.requires_grad for i in inputs])
        and (not_exists(check_instance_variable) or getattr(self, check_instance_variable, False))
    )


@typecheck
def save_args_and_kwargs(fn):
    """Save the arguments and keyword arguments of a function as instance variables.

    :param fn: The function to wrap.
    :return: The wrapped function.
    """

    @wraps(fn)
    def inner(self, *args, **kwargs):
        self._args_and_kwargs = (args, kwargs)
        self._version = "1.0" # importlib.metadata.version("omnifold")

        return fn(self, *args, **kwargs)

    return inner


# functions for deriving the frames for ligands
# this follows the logic from OmniFold Supplementary section 4.3.2


@typecheck
def get_indices_three_closest_atom_pos(
    atom_pos: Float["... n d"],  # type: ignore
    mask: Bool["... n"] | None = None,  # type: ignore
) -> Int["... n 3"]:  # type: ignore
    """Get the indices of the three closest atoms to each atom.

    :param atom_pos: The atom positions.
    :param mask: The mask to apply.
    :return: The indices of the three closest atoms to each atom.
    """
    atom_dims, device = atom_pos.shape[-3:-1], atom_pos.device
    num_atoms, has_batch = atom_pos.shape[-2], atom_pos.ndim == 3
    batch_size = 1 if not has_batch else atom_pos.shape[0]

    if num_atoms < 3:
        return atom_pos.new_full((*atom_dims, 3), -1).long()

    if not has_batch:
        atom_pos = rearrange(atom_pos, "... -> 1 ...")

        if exists(mask):
            mask = rearrange(mask, "... -> 1 ...")

    # figure out which set of atoms are less than 3 for masking out later

    if exists(mask):
        insufficient_atom_mask = mask.sum(dim=-1, keepdim=True) < 3

    # get distances between all atoms

    atom_dist = torch.cdist(atom_pos, atom_pos)

    # mask out the distance to self

    eye = torch.eye(num_atoms, device=device, dtype=torch.bool)

    mask_value = 1e4
    atom_dist.masked_fill_(eye, mask_value)

    # take care of padding

    if exists(mask):
        # pair_mask = einx.logical_and("... i, ... j -> ... i j", mask, mask)
        pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        atom_dist.masked_fill_(~pair_mask, mask_value)

    # will use topk on the negative of the distance

    _, two_closest_atom_indices = (-atom_dist).topk(2, dim=-1)

    # place each atom at the center of its frame

    three_atom_indices, _ = pack(
        (
            two_closest_atom_indices[..., 0],
            torch.arange(num_atoms, device=device).unsqueeze(0).expand(batch_size, -1),
            two_closest_atom_indices[..., 1],
        ),
        "b n *",
    )

    # mask out

    if exists(mask):
        three_atom_indices = torch.where(
            ~insufficient_atom_mask.unsqueeze(-1), three_atom_indices, -1
        )

    if not has_batch:
        three_atom_indices = rearrange(three_atom_indices, "1 ... -> ...")

    return three_atom_indices


@typecheck
def get_angle_between_edges(
    edge1: Float["... n 3"],  # type: ignore
    edge2: Float["... n 3"],  # type: ignore
) -> Float["... n"]:  # type: ignore
    """Get the angles between two edges for each node.

    :param edge1: The first edge.
    :param edge2: The second edge.
    :return: The angles between the two edges for each node.
    """
    cos = (l2norm(edge1) * l2norm(edge2)).sum(-1)
    return torch.acos(cos)


@typecheck
def get_frames_from_atom_pos(
    atom_pos: Float["... n d"],  # type: ignore
    mask: Bool["... n"] | None = None,  # type: ignore
    filter_colinear_pos: bool = False,
    is_colinear_angle_thres: float = 25.0,  # NOTE: DM uses 25 degrees as a way of filtering out invalid frames
) -> Int["... n 3"]:  # type: ignore
    """Get the nearest neighbor frames for all atom positions.

    :param atom_pos: The atom positions.
    :param filter_colinear_pos: Whether to filter colinear positions.
    :param is_colinear_angle_thres: The colinear angle threshold.
    :return: The frames for all atoms.
    """
    frames = get_indices_three_closest_atom_pos(atom_pos, mask=mask)

    if not filter_colinear_pos:
        return frames

    is_invalid = (frames == -1).any(dim=-1)

    # get the edges and derive angles

    three_atom_pos = torch.cat(
        [
            # einx.get_at("... [m] c, ... three -> ... three c", atom_pos, frame).unsqueeze(-3)
            atom_pos.gather(
                -2,
                (
                    (torch.zeros_like(frame) if frame_is_invalid else frame)
                    .unsqueeze(-1)
                    .repeat(1, 3)
                ),
            ).unsqueeze(-3)
            for frame, frame_is_invalid in zip(frames.unbind(dim=-2), is_invalid)
        ],
        dim=-3,
    )

    left_pos, center_pos, right_pos = three_atom_pos.unbind(dim=-2)

    edges1, edges2 = (left_pos - center_pos), (right_pos - center_pos)

    angle = get_angle_between_edges(edges1, edges2)

    degree = torch.rad2deg(angle)

    is_colinear = (degree.abs() < is_colinear_angle_thres) | (
        (180.0 - degree.abs()).abs() < is_colinear_angle_thres
    )

    # set any three atoms that are colinear to -1 indices

    # three_atom_indices = einx.where(
    #     "..., ... three, -> ... three", ~(is_colinear | is_invalid), frames, -1
    # )
    three_atom_indices = torch.where(~(is_colinear | is_invalid).unsqueeze(-1), frames, -1)
    return three_atom_indices


@typecheck
def create_uid_tensor(first_tensor: Int["b n"], second_tensor: Int["b n"]) -> Int["b n"]:  # type: ignore
    """Create a unique identifier (UID) tensor from two tensors.

    :param first_tensor: The first tensor.
    :param second_tensor: The second tensor.
    :return: The UID tensor.
    """
    batch_size = first_tensor.shape[0]
    device = first_tensor.device

    # Ensure both tensors have the same shape.
    assert first_tensor.shape == second_tensor.shape, "Tensors must have the same shape"

    # Combine the tensors into a single tensor of tuples.
    combined = torch.stack((first_tensor, second_tensor), dim=-1)

    # Convert each tuple to a unique integer using a hash function.
    uids = torch.stack(
        [
            torch.tensor(
                [hash(tuple(x.tolist())) for x in combined[i]],
                dtype=torch.long,
                device=device,
            )
            for i in range(batch_size)
        ],
        dim=0,
    )

    return uids


@typecheck
def autocasting_disable_decorator(disable_casting: bool, cache_enabled: bool = False) -> Callable:
    """Install a decorator to maybe disable autocasting for a function.

    :param disable_casting: If True, disables autocasting; otherwise, uses the default autocasting
        context.
    :param cache_enabled: If True, enables caching of the autocasting context.
    :return: A decorator that wraps the function with the specified autocasting context.
    """

    def func_wrapper(func):
        def new_func(*args, **kwargs):
            _amp_context = (
                torch.autocast(device_type="cuda", enabled=False, cache_enabled=cache_enabled)
                if disable_casting
                else nullcontext()
            )
            dtype = torch.float32 if disable_casting else None
            with _amp_context:
                return func(
                    *(
                        (
                            i.to(dtype=dtype)
                            if isinstance(i, torch.Tensor) and torch.is_floating_point(i)
                            else (
                                {
                                    k: (
                                        v.to(dtype=dtype)
                                        if isinstance(v, torch.Tensor)
                                        and torch.is_floating_point(v)
                                        else v
                                    )
                                    for k, v in i.items()
                                }
                                if isinstance(i, dict)
                                else i
                            )
                        )
                        for i in args
                    ),
                    **{
                        k: (
                            v.to(dtype=dtype)
                            if isinstance(v, torch.Tensor) and torch.is_floating_point(v)
                            else v
                        )
                        for k, v in kwargs.items()
                    },
                )

        return new_func

    return func_wrapper


# sampling functions for prior distributions


def collate_ligand_molecule_types(
    is_molecule_types: Bool["b n 5"],  # type: ignore
) -> Bool["b n 5"]:  # type: ignore
    """Collate ligand and ions within `is_molecule_types`.

    :param is_molecule_type: The molecule types.
    :return: The collated molecule types.
    """
    is_molecule_types = is_molecule_types.clone()

    # collate ligand and ions

    is_molecule_types[..., 3] = is_molecule_types[..., 3] | is_molecule_types[..., 4]
    is_molecule_types = is_molecule_types[..., :4]

    return is_molecule_types


def sample_harmonic_prior(
    batch_size: int,
    device: str | torch.device,
    dtype: torch.dtype,
    molecule_atom_lens: Int["b n"],  # type: ignore
    is_molecule_types: Bool[f"b n {IS_MOLECULE_TYPES}"],  # type: ignore
    additional_molecule_feats: Int[f"b n {ADDITIONAL_MOLECULE_FEATS}"],  # type: ignore
) -> Float["b m 3"]:  # type: ignore
    """Sample from a harmonic prior distribution.

    NOTE: This function assumes the first batch elements of `molecule_atom_lens`,
    `is_molecule_types`, and `additional_molecule_feats` represent all batch elements.
    That is, this function assumes that the (underlying) batch size is 1, whereby only
    a single unique complex is represented by these input features.

    :param batch_size: The batch size.
    :param device: The device.
    :param dtype: The data type.
    :param molecule_atom_lens: The molecule atom lengths.
    :param is_molecule_types: The molecule types.
    :param additional_molecule_feats: The additional molecule features.
    :return: The sampled harmonic prior noise.
    """
    mol_atom_lens = molecule_atom_lens[0]
    residue_index, _, chain_index, entity_index, _ = additional_molecule_feats[0].unbind(dim=-1)

    chain_residue_index = create_uid_tensor(
        chain_index.unsqueeze(0), residue_index.unsqueeze(0)
    ).squeeze(0)

    # distinguish all entities within each chain

    collated_is_molecule_types = collate_ligand_molecule_types(is_molecule_types)
    molecule_types = collated_is_molecule_types[0].long().argmax(dim=-1)

    entity_offset = 0
    entities = defaultdict(list)
    for idx in range(len(molecule_types)):
        entity_id = f"{chain_index[idx]}:{entity_index[idx] + entity_offset}:{molecule_types[idx]}"
        entities[entity_id].append(idx)

        # handle ligand residues located within a polymer-majority chain

        if (
            idx < len(molecule_types) - 1
            and molecule_types[idx] == 3
            and molecule_types[idx + 1] < 3
        ):
            entity_offset += 1

    # sample from a harmonic prior for each type of entity within each chain

    all_atom_pos = []

    for entity_id in entities:
        # collect entity metadata

        entity = tensor(entities[entity_id])
        entity_molecule_type = int(entity_id.split(":")[-1])
        entity_mol_atom_lens = mol_atom_lens[entity]

        entity_num_nodes = len(entity)

        if entity_molecule_type < 3:
            # collate atoms of each modified polymer residue for sampling

            entity_chain_residue_index = chain_residue_index[entity]
            remapped_entity_chain_residue_index = torch.unique_consecutive(
                entity_chain_residue_index, return_inverse=True
            )[-1]

            entity_num_nodes = len(entity_chain_residue_index.unique())

            collated_entity_mol_atom_lens = torch.zeros(
                entity_num_nodes, device=device, dtype=torch.long
            )
            collated_entity_mol_atom_lens.scatter_add_(
                0, remapped_entity_chain_residue_index, entity_mol_atom_lens
            )

            entity_mol_atom_gather_idx = torch.arange(
                entity_num_nodes,
                device=device,
            ).repeat_interleave(collated_entity_mol_atom_lens)

        # sample from a harmonic prior

        ptr = tensor([0, entity_num_nodes], device=device, dtype=torch.long)

        # NOTE: The following `a` values are chosen based on
        # (empirical) average covalent bond lengths between
        # polymer residues and ligands/ion atoms. For polymers,
        # values come from https://github.com/Profluent-Internships/MMDiff.
        # For ligands (and ions), values are inspired by https://en.wikipedia.org/wiki/Atomic_spacing.

        if entity_molecule_type == 0:
            D_inv, P = HarmonicSDE.diagonalize(
                entity_num_nodes,
                ptr,
                a=3 / (3.80**2),
            )
        elif entity_molecule_type in (1, 2):
            D_inv, P = HarmonicSDE.diagonalize(
                entity_num_nodes,
                ptr,
                a=3 / (6.12**2),
            )
        elif entity_molecule_type == 3:
            D_inv, P = HarmonicSDE.diagonalize(
                entity_num_nodes,
                ptr,
                a=3 / (1.50**2),
            )
        else:
            raise ValueError(f"Invalid molecule type: {entity_molecule_type}")

        noise = torch.randn((batch_size, entity_num_nodes, 3), device=device, dtype=dtype)

        # for polymers, separate noise for token center atoms (harmonic) and all other atoms (Gaussian)

        atom_pos = harmonic_token_center_atom_noise = P @ (torch.sqrt(D_inv)[:, None] * noise)

        if entity_molecule_type < 3:
            atom_pos = (
                torch.randn(
                    (batch_size, entity_mol_atom_lens.sum(), 3), device=device, dtype=dtype
                )
                + harmonic_token_center_atom_noise[:, entity_mol_atom_gather_idx]
            )

        all_atom_pos.append(atom_pos)

    # concatenate all sampled atom positions

    all_atom_pos = torch.cat(all_atom_pos, dim=1)

    return all_atom_pos


# modules for handling frames


class ExpressCoordinatesInFrame(Module):
    """Algorithm 29."""

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    @typecheck
    def forward(
        self,
        coords: Float["b m 3"],  # type: ignore
        frame: Float["b m 3 3"] | Float["b 3 3"] | Float["3 3"],  # type: ignore
    ) -> Float["b m 3"]:  # type: ignore
        """Express coordinates in the given frame.

        :param coords: Coordinates to be expressed in the given frame.
        :param frame: Frames defined by three points.
        :return: The transformed coordinates.
        """

        if frame.ndim == 2:
            frame = rearrange(frame, "fr fc -> 1 1 fr fc")
        elif frame.ndim == 3:
            frame = rearrange(frame, "b fr fc -> b 1 fr fc")

        # Extract frame atoms
        a, b, c = frame.unbind(dim=-1)
        w1 = l2norm(a - b, eps=self.eps)
        w2 = l2norm(c - b, eps=self.eps)

        # Build orthonormal basis
        e1 = l2norm(w1 + w2, eps=self.eps)
        e2 = l2norm(w2 - w1, eps=self.eps)
        e3 = torch.cross(e1, e2, dim=-1)

        # Project onto frame basis
        d = coords - b

        transformed_coords = torch.stack(
            (
                einsum(d, e1, "... i, ... i -> ..."),
                einsum(d, e2, "... i, ... i -> ..."),
                einsum(d, e3, "... i, ... i -> ..."),
            ),
            dim=-1,
        )

        return transformed_coords


class RigidFrom3Points(Module):
    """An implementation of Algorithm 21 in Section 1.8.1 in AlphaFold 2 paper:

    https://www.nature.com/articles/s41586-021-03819-2
    """

    @typecheck
    def forward(
        self,
        three_points: Tuple[Float["... 3"], Float["... 3"], Float["... 3"]] | Float["3 ... 3"],  # type: ignore
    ) -> Tuple[Float["... 3 3"], Float["... 3"]]:  # type: ignore
        """Compute a rigid transformation from three points."""
        if isinstance(three_points, tuple):
            three_points = torch.stack(three_points)

        # allow for any number of leading dimensions

        (x1, x2, x3), unpack_one = pack_one(three_points, "three * d")

        # main algorithm

        v1 = x3 - x2
        v2 = x1 - x2

        e1 = l2norm(v1)
        u2 = v2 - e1 @ (e1.t() @ v2)
        e2 = l2norm(u2)

        e3 = torch.cross(e1, e2, dim=-1)

        R = torch.stack((e1, e2, e3), dim=-1)
        t = x2

        # unpack

        R = unpack_one(R, "* r1 r2")
        t = unpack_one(t, "* c")

        return R, t


class RigidFromReference3Points(Module):
    """A modification of Algorithm 21 in Section 1.8.1 in AlphaFold 2 paper:

    https://www.nature.com/articles/s41586-021-03819-2

    Inpsired by the implementation in the OpenFold codebase:
    https://github.com/aqlaboratory/openfold/blob/6f63267114435f94ac0604b6d89e82ef45d94484/openfold/utils/feats.py#L143
    """

    @typecheck
    def forward(
        self,
        three_points: Tuple[Float["... 3"], Float["... 3"], Float["... 3"]] | Float["3 ... 3"],  # type: ignore
        eps: float = 1e-20,
    ) -> Tuple[Float["... 3 3"], Float["... 3"]]:  # type: ignore
        """Return a transformation object from reference coordinates.

        NOTE: This method does not take care of symmetries. If you
        provide the atom positions in the non-standard way,
        e.g., the N atom of amino acid residues will end up
        not at [-0.527250, 1.359329, 0.0] but instead at
        [-0.527250, -1.359329, 0.0]. You need to take care
        of such cases in your code.

        :param three_points: Three reference points to define the transformation.
        :param eps: A small value to avoid division by zero.
        :return: A transformation object. After applying the translation and
            rotation to the reference backbone, the coordinates will
            approximately equal to the input coordinates.
        """
        if isinstance(three_points, tuple):
            three_points = torch.stack(three_points)

        # allow for any number of leading dimensions

        (x1, x2, x3), unpack_one = pack_one(three_points, "three * d")

        # main algorithm

        t = -1 * x2
        x1 = x1 + t
        x3 = x3 + t

        x3_x, x3_y, x3_z = [x3[..., i] for i in range(3)]
        norm = torch.sqrt(eps + x3_x**2 + x3_y**2)
        sin_x3_1 = -x3_y / norm
        cos_x3_1 = x3_x / norm

        x3_1_R = sin_x3_1.new_zeros((*sin_x3_1.shape, 3, 3))
        x3_1_R[..., 0, 0] = cos_x3_1
        x3_1_R[..., 0, 1] = -1 * sin_x3_1
        x3_1_R[..., 1, 0] = sin_x3_1
        x3_1_R[..., 1, 1] = cos_x3_1
        x3_1_R[..., 2, 2] = 1

        norm = torch.sqrt(eps + x3_x**2 + x3_y**2 + x3_z**2)
        sin_x3_2 = x3_z / norm
        cos_x3_2 = torch.sqrt(x3_x**2 + x3_y**2) / norm

        x3_2_R = sin_x3_2.new_zeros((*sin_x3_2.shape, 3, 3))
        x3_2_R[..., 0, 0] = cos_x3_2
        x3_2_R[..., 0, 2] = sin_x3_2
        x3_2_R[..., 1, 1] = 1
        x3_2_R[..., 2, 0] = -1 * sin_x3_2
        x3_2_R[..., 2, 2] = cos_x3_2

        x3_R = einsum(x3_2_R, x3_1_R, "n i j, n j k -> n i k")
        x1 = einsum(x3_R, x1, "n i j, n j -> n i")

        _, x1_y, x1_z = [x1[..., i] for i in range(3)]
        norm = torch.sqrt(eps + x1_y**2 + x1_z**2)
        sin_x1 = -x1_z / norm
        cos_x1 = x1_y / norm

        x1_R = sin_x3_2.new_zeros((*sin_x3_2.shape, 3, 3))
        x1_R[..., 0, 0] = 1
        x1_R[..., 1, 1] = cos_x1
        x1_R[..., 1, 2] = -1 * sin_x1
        x1_R[..., 2, 1] = sin_x1
        x1_R[..., 2, 2] = cos_x1

        R = einsum(x1_R, x3_R, "n i j, n j k -> n i k")

        R = R.transpose(-1, -2)
        t = -1 * t

        # unpack

        R = unpack_one(R, "* r1 r2")
        t = unpack_one(t, "* c")

        return R, t


# pooling modules for OmniFold


def segment_sum(src, dst_idx, dst_size):
    """Computes the sum of each segment in a tensor."""
    out = torch.zeros(
        dst_size,
        *src.shape[1:],
        dtype=src.dtype,
        device=src.device,
    ).index_add_(0, dst_idx, src)
    return out


class SumPooling(Module):
    """Sum pooling layer."""

    def __init__(self, learnable: bool, hidden_dim: int = 1):
        """Initialize the SumPooling layer."""
        super().__init__()
        self.pooled_transform = Linear(hidden_dim, hidden_dim) if learnable else Identity()

    def forward(self, x, dst_idx, dst_size):
        """Forward pass through the SumPooling layer."""
        return self.pooled_transform(segment_sum(x, dst_idx, dst_size))


# loss modules for OmniFold


class DiffusionLossBreakdown(NamedTuple):
    """The DiffusionLossBreakdown class."""

    diffusion_mse: Float[""]  # type: ignore
    diffusion_bond: Float[""]  # type: ignore
    diffusion_smooth_lddt: Float[""]  # type: ignore


class LossBreakdown(NamedTuple):
    """The LossBreakdown class."""

    total_loss: Float[""]  # type: ignore
    total_diffusion: Float[""]  # type: ignore
    distogram: Float[""]  # type: ignore
    pae: Float[""]  # type: ignore
    pde: Float[""]  # type: ignore
    plddt: Float[""]  # type: ignore
    resolved: Float[""]  # type: ignore
    affinity: Float[""]  # type: ignore
    confidence: Float[""]  # type: ignore
    diffusion_mse: Float[""]  # type: ignore
    diffusion_bond: Float[""]  # type: ignore
    diffusion_smooth_lddt: Float[""]  # type: ignore


class SmoothLDDTLoss(Module):
    """Algorithm 27."""

    @typecheck
    def __init__(self, nucleic_acid_cutoff: float = 30.0, other_cutoff: float = 15.0):
        super().__init__()
        self.nucleic_acid_cutoff = nucleic_acid_cutoff
        self.other_cutoff = other_cutoff

        self.register_buffer("lddt_thresholds", torch.tensor([0.5, 1.0, 2.0, 4.0]))

    @typecheck
    def forward(
        self,
        pred_coords: Float["ba m 3"],  # type: ignore
        true_coords: Float["ba m 3"],  # type: ignore
        is_dna: Bool["b m"],  # type: ignore
        is_rna: Bool["b m"],  # type: ignore
        coords_mask: Bool["b m"] | None = None,  # type: ignore
        paired_coords_mask: Bool["b m m"] | None = None,  # type: ignore
        reduce: bool = False,
    ) -> Float[""]:  # type: ignore
        """Compute the SmoothLDDT loss.

        :param pred_coords: Predicted atom coordinates.
        :param true_coords: True atom coordinates.
        :param is_dna: A boolean Tensor denoting DNA atoms.
        :param is_rna: A boolean Tensor denoting RNA atoms.
        :param coords_mask: The atom coordinates mask.
        :param paired_coords_mask: The paired atom coordinates mask.
        :param reduce: Whether to reduce the output.
        :return: The output tensor.
        """
        dtype = pred_coords.dtype

        # Compute distances between all pairs of atoms
        pred_dists = torch.cdist(pred_coords.float(), pred_coords.float(), p=2).type(dtype)
        true_dists = torch.cdist(true_coords.float(), true_coords.float(), p=2).type(dtype)

        # Compute distance difference for all pairs of atoms
        dist_diff = torch.abs(true_dists - pred_dists)

        # Compute epsilon values
        # eps = einx.subtract("thresholds, ... -> ... thresholds", self.lddt_thresholds, dist_diff)
        eps = self.lddt_thresholds[None, None, None, :] - dist_diff[..., None]
        eps = eps.sigmoid().mean(dim=-1)

        # Restrict to bespoke inclusion radius
        is_nucleotide = is_dna | is_rna
        is_nucleotide_pair = to_pairwise_mask(is_nucleotide)

        inclusion_radius = torch.where(
            is_nucleotide_pair,
            true_dists < self.nucleic_acid_cutoff,
            true_dists < self.other_cutoff,
        )

        # Compute mean, avoiding self term
        mask = inclusion_radius & ~torch.eye(
            pred_coords.shape[1], dtype=torch.bool, device=pred_coords.device
        )

        # Take into account variable lengthed atoms in batch
        if exists(paired_coords_mask):
            mask = mask & paired_coords_mask
        elif exists(coords_mask):
            paired_coords_mask = to_pairwise_mask(coords_mask)
            mask = mask & paired_coords_mask

        # Calculate masked averaging
        lddt = masked_average(eps, mask=mask, dim=(-1, -2), eps=1)

        if not reduce:
            return lddt

        return 1.0 - lddt.mean()


class ComputeAlignmentError(Module):
    """Algorithm 30."""

    @typecheck
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.express_coordinates_in_frame = ExpressCoordinatesInFrame()

    @typecheck
    def forward(
        self,
        pred_coords: Float["b n 3"],  # type: ignore
        true_coords: Float["b n 3"],  # type: ignore
        pred_frames: Float["b n 3 3"],  # type: ignore
        true_frames: Float["b n 3 3"],  # type: ignore
        mask: Bool["b n"] | None = None,  # type: ignore
    ) -> Float["b n n"]:  # type: ignore
        """Compute the alignment errors.

        :param pred_coords: Predicted coordinates.
        :param true_coords: True coordinates.
        :param pred_frames: Predicted frames.
        :param true_frames: True frames.
        :param mask: The mask for variable lengths.
        :return: The alignment errors.
        """

        # to pairs

        seq = pred_coords.shape[1]

        pair2seq = partial(rearrange, pattern="b n m ... -> b (n m) ...")
        seq2pair = partial(rearrange, pattern="b (n m) ... -> b n m ...", n=seq, m=seq)

        pair_pred_coords = pair2seq(repeat(pred_coords, "b n d -> b n m d", m=seq))
        pair_true_coords = pair2seq(repeat(true_coords, "b n d -> b n m d", m=seq))
        pair_pred_frames = pair2seq(repeat(pred_frames, "b n d e -> b m n d e", m=seq))
        pair_true_frames = pair2seq(repeat(true_frames, "b n d e -> b m n d e", m=seq))

        # Express predicted coordinates in predicted frames
        pred_coords_transformed = self.express_coordinates_in_frame(
            pair_pred_coords, pair_pred_frames
        )

        # Express true coordinates in true frames
        true_coords_transformed = self.express_coordinates_in_frame(
            pair_true_coords, pair_true_frames
        )

        # Compute alignment errors
        alignment_errors = F.pairwise_distance(
            pred_coords_transformed, true_coords_transformed, eps=self.eps
        )

        alignment_errors = seq2pair(alignment_errors)

        # Masking
        if exists(mask):
            pair_mask = to_pairwise_mask(mask)
            # alignment_errors = einx.where(
            #     "b i j, b i j, -> b i j", pair_mask, alignment_errors, 0.0
            # )
            alignment_errors = alignment_errors * pair_mask

        return alignment_errors


class OmniFoldLoss(Module):
    """A composite loss function for OmniFold."""

    def __init__(
        self,
        distogram_weight: float,
        diffusion_weight: float,
        confidence_weight: float,
        distance_bins: torch.Tensor,
        pae_bins: torch.Tensor,
        pde_bins: torch.Tensor,
        num_plddt_bins: int = 50,
        diffusion_chunk_size: int = 4,
        diffusion_mse_weight: float = 1.0 / 3.0,
        lddt_mask_nucleic_acid_cutoff: float = 30.0,
        lddt_mask_other_cutoff: float = 15.0,
        min_conf_resolution: float = 0.1,
        max_conf_resolution: float = 4.0,
        diffusion_add_smooth_lddt_loss: bool = True,
        diffusion_add_bond_loss: bool = False,
        train_pae: bool = False,
        distogram_atom_resolution: bool = False,
        karras_formulation: bool = True,
        ignore_index: int = -1,
        smooth_lddt_loss_kwargs: dict = {},
    ):
        super().__init__()

        self.distogram_weight = distogram_weight
        self.diffusion_weight = diffusion_weight
        self.confidence_weight = confidence_weight
        self.diffusion_mse_weight = diffusion_mse_weight

        self.diffusion_chunk_size = diffusion_chunk_size

        self.lddt_mask_nucleic_acid_cutoff = lddt_mask_nucleic_acid_cutoff
        self.lddt_mask_other_cutoff = lddt_mask_other_cutoff
        self.min_conf_resolution = min_conf_resolution
        self.max_conf_resolution = max_conf_resolution

        self.diffusion_add_smooth_lddt_loss = diffusion_add_smooth_lddt_loss
        self.diffusion_add_bond_loss = diffusion_add_bond_loss
        self.train_pae = train_pae

        self.distogram_atom_resolution = distogram_atom_resolution
        self.karras_formulation = karras_formulation
        self.ignore_index = ignore_index

        self.num_plddt_bins = num_plddt_bins

        self.smooth_lddt_loss = SmoothLDDTLoss(**smooth_lddt_loss_kwargs)
        self.compute_alignment_error = ComputeAlignmentError()

        self.register_buffer("distance_bins", distance_bins)
        self.register_buffer("pae_bins", pae_bins)
        self.register_buffer("pde_bins", pde_bins)
        self.register_buffer("lddt_thresholds", torch.tensor([0.5, 1.0, 2.0, 4.0]))

        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

    @typecheck
    def calculate_distogram_labels(
        self,
        atom_pos: Float["b m 3"],  # type: ignore
        atom_mask: Bool["b m"],  # type: ignore
        distogram_atom_indices: Int["b n"],  # type: ignore
        valid_distogram_mask: Bool["b n"],  # type: ignore
    ) -> Float["b n n"] | Float["b m m"]:  # type: ignore
        """Calculate the distogram labels.

        :param atom_pos: The distogram logits.
        :param atom_mask: The atom mask.
        :param distogram_atom_indices: The distogram atom indices.
        :param valid_distogram_mask: The valid distogram mask.
        :return: The distogram labels.
        """
        dtype = atom_pos.dtype

        distogram_pos = atom_pos

        with torch.no_grad():
            if not self.distogram_atom_resolution:
                # molecule_pos = einx.get_at('b [m] c, b n -> b n c', atom_pos, distogram_atom_indices)

                distogram_atom_coords_indices = repeat(
                    distogram_atom_indices, "b n -> b n c", c=distogram_pos.shape[-1]
                )
                distogram_pos = atom_pos.gather(1, distogram_atom_coords_indices)
                distogram_mask = valid_distogram_mask
            else:
                distogram_mask = atom_mask

            distogram_dist = torch.cdist(distogram_pos.float(), distogram_pos.float(), p=2).type(
                dtype
            )
            distogram_labels = distance_to_dgram(
                distogram_dist, self.distance_bins, return_labels=True
            )

            # account for representative distogram atom missing from residue (`-1` set on the `distogram_atom_indices` field)

            distogram_labels.masked_fill_(~to_pairwise_mask(distogram_mask), self.ignore_index)

        return distogram_labels

    @typecheck
    def calculate_distogram_loss(
        self,
        distogram_logits: Float["b n n"],  # type: ignore
        distogram_labels: Float["b n n"],  # type: ignore
        atom_mask: Bool["b m"],  # type: ignore
        pairwise_mask: Bool["b n n"],  # type: ignore
    ) -> Float[" b"]:  # type: ignore
        """Calculate the distogram loss.

        :param distogram_logits: The distogram logits.
        :param distogram_labels: The distogram labels.
        :param atom_mask: The atom mask.
        :param pairwise_mask: The pairwise mask.
        :return: The distogram loss.
        """
        distogram_pairwise_mask = pairwise_mask
        if self.distogram_atom_resolution:
            distogram_pairwise_mask = to_pairwise_mask(atom_mask)

        distogram_labels = torch.where(
            distogram_pairwise_mask, distogram_labels, self.ignore_index
        )
        distogram_loss = F.cross_entropy(
            distogram_logits, distogram_labels, ignore_index=self.ignore_index
        )

        return distogram_loss

    @typecheck
    def calculate_bond_loss(self, denoised_atom_pos: Float["ba m 3"], atom_pos_aligned: Float["ba m 3"], bond_mask: Bool["b m m"] | None, eps: float = 1e-6) -> Float[" b"]:  # type: ignore
        """Calculate the bond loss.

        :param denoised_atom_pos: The denoised atom positions.
        :param atom_pos_aligned: The aligned atom positions.
        :param bond_mask: The bond mask.
        :param eps: The epsilon value.
        :return: The bond loss.
        """
        dtype = denoised_atom_pos.dtype

        denoised_dist = torch.cdist(
            denoised_atom_pos.float(), denoised_atom_pos.float(), p=2
        ).type(dtype)
        true_dist = torch.cdist(
            atom_pos_aligned.float(),
            atom_pos_aligned.float(),
            p=2,
        ).type(dtype)

        dist_squared_err = (denoised_dist - true_dist) ** 2
        bond_loss = torch.sum(dist_squared_err * bond_mask, dim=(-1, -2)) / torch.sum(
            bond_mask + eps, dim=(-1, -2)
        )

        return bond_loss

    @typecheck
    def calculate_diffusion_loss(
        self,
        denoised_atom_pos: Float["b m 3"],  # type: ignore
        atom_pos_aligned: Float["b m 3"],  # type: ignore
        align_weights: Float["b m"],  # type: ignore
        loss_weights: Float[" b"] | None,  # type: ignore
        molecule_atom_lens: Int["b n"],  # type: ignore
        atom_mask: Bool["b m"],  # type: ignore
        bond_mask: Bool["b m m"] | None = None,  # type: ignore
        missing_atom_mask: Bool["b m"] | None = None,  # type: ignore
        is_molecule_types: Bool["b n 5"] | None = None,  # type: ignore
        eps: float = 1e-6,
    ) -> Tuple[Float[""], DiffusionLossBreakdown]:  # type: ignore
        """Calculate the diffusion loss.

        :param denoised_atom_pos: The denoised atom positions.
        :param atom_pos_aligned: The aligned atom positions.
        :param align_weights: The alignment weights.
        :param loss_weights: The loss weights.
        :param molecule_atom_lens: The molecule atom lengths.
        :param is_molecule_types: The molecule types.
        :param atom_mask: The atom mask.
        :param bond_mask: The bond mask.
        :param missing_atom_mask: The missing atom mask.
        :param eps: The epsilon value.
        :return: The diffusion loss and the loss breakdown.
        """
        dtype = denoised_atom_pos.dtype

        # default the loss to zero

        diffusion_loss = self.zero.type(dtype)

        # if there are missing atoms, update the atom mask to not include them in the loss

        if exists(missing_atom_mask):
            atom_mask = atom_mask & ~missing_atom_mask

        # calculate main diffusion MSE loss

        per_atom_se = ((denoised_atom_pos - atom_pos_aligned) ** 2).sum(dim=-1)
        per_sample_weighted_mse = (align_weights * per_atom_se).sum(dim=-1) / (
            atom_mask.sum(dim=-1) + eps
        )

        if exists(loss_weights):
            per_sample_weighted_mse = per_sample_weighted_mse * loss_weights

        weighted_align_mse_loss = self.diffusion_mse_weight * (per_sample_weighted_mse).mean(
            dim=-1
        )

        mse_loss = weighted_align_mse_loss.mean()

        diffusion_loss = diffusion_loss + mse_loss

        # construct atom pair mask for either smooth lDDT loss or bond loss

        atompair_mask = to_pairwise_mask(atom_mask)

        # calculate bond loss during finetuning

        bond_loss = self.zero.type(dtype)

        if self.diffusion_add_bond_loss and exists(bond_mask):
            atompair_bond_mask = atompair_mask * bond_mask

            bond_losses = []
            num_augs = denoised_atom_pos.shape[-3]
            diffusion_num_chunks = num_augs // self.diffusion_chunk_size + (
                num_augs % self.diffusion_chunk_size != 0
            )
            for i in range(diffusion_num_chunks):
                bond_loss_i = checkpoint(
                    self.calculate_bond_loss,
                    denoised_atom_pos[
                        ...,
                        i * self.diffusion_chunk_size : (i + 1) * self.diffusion_chunk_size,
                        :,
                        :,
                    ],
                    atom_pos_aligned[
                        ...,
                        i * self.diffusion_chunk_size : (i + 1) * self.diffusion_chunk_size,
                        :,
                        :,
                    ],
                    atompair_bond_mask,
                )
                bond_losses.append(bond_loss_i)
            bond_losses = torch.cat(bond_losses, dim=-1)

            if exists(loss_weights):
                bond_losses = bond_losses * loss_weights

            bond_loss = bond_losses.mean(dim=-1).mean()

            diffusion_loss = diffusion_loss + bond_loss

        # calculate auxiliary smooth lDDT loss

        smooth_lddt_loss = self.zero.type(dtype)

        if self.diffusion_add_smooth_lddt_loss:
            assert exists(
                is_molecule_types
            ), "The argument `is_molecule_types` must be passed in if adding the smooth lDDT loss."

            is_nucleotide_or_ligand_fields = is_molecule_types.unbind(dim=-1)

            is_nucleotide_or_ligand_fields = tuple(
                batch_repeat_interleave(t, molecule_atom_lens)
                for t in is_nucleotide_or_ligand_fields
            )
            is_nucleotide_or_ligand_fields = tuple(
                pad_or_slice_to(t, length=align_weights.shape[-1], dim=-1)
                for t in is_nucleotide_or_ligand_fields
            )

            _, atom_is_dna, atom_is_rna, _, _ = is_nucleotide_or_ligand_fields

            lddt_losses = []
            num_augs = denoised_atom_pos.shape[-3]
            num_chunks = num_augs // self.diffusion_chunk_size + (
                num_augs % self.diffusion_chunk_size != 0
            )
            for i in range(num_chunks):
                lddt_i = checkpoint(
                    self.smooth_lddt_loss.__call__,
                    denoised_atom_pos[
                        ...,
                        i * self.diffusion_chunk_size : (i + 1) * self.diffusion_chunk_size,
                        :,
                        :,
                    ],
                    atom_pos_aligned[
                        ...,
                        i * self.diffusion_chunk_size : (i + 1) * self.diffusion_chunk_size,
                        :,
                        :,
                    ],
                    atom_is_dna,
                    atom_is_rna,
                    atom_mask,
                    atompair_mask,
                )
                lddt_losses.append(lddt_i)
            lddt_losses = torch.cat(lddt_losses, dim=-1)

            lddt_loss = lddt_losses.mean(dim=-1)
            smooth_lddt_loss = 1 - lddt_loss.mean()

            diffusion_loss = diffusion_loss + smooth_lddt_loss

        # calculate loss breakdown

        loss_breakdown = DiffusionLossBreakdown(mse_loss, bond_loss, smooth_lddt_loss)

        return diffusion_loss, loss_breakdown

    @typecheck
    def calculate_confidence_labels(
        self,
        denoised_atom_pos: Float["b m 3"],  # type: ignore
        atom_pos_aligned: Float["b m 3"],  # type: ignore
        resolved_labels: Int["b m"] | None,  # type: ignore
        atom_mask: Bool["b m"],  # type: ignore
        valid_atom_indices_for_frame: Bool["b n"],  # type: ignore
        atom_indices_for_frame: Int["b n three"],  # type: ignore
        is_molecule_types: Bool["b n 5"],  # type: ignore
        molecule_atom_lens: Int["b n"],  # type: ignore
        distogram_atom_indices: Int["b n"],  # type: ignore
        molecule_atom_indices: Int["b n"],  # type: ignore
        valid_molecule_atom_mask: Bool["b n"],  # type: ignore
    ) -> Tuple[
        Float["b n n"] | None, Float["b n n"], Float["b m"], Float["b m"] | None  # type: ignore
    ]:
        """Calculate the confidence labels.

        :param denoised_atom_pos: The denoised atom positions.
        :param atom_pos_aligned: The atom positions.
        :param resolved_labels: The resolved labels.
        :param atom_mask: The atom mask.
        :param valid_atom_indices_for_frame: The valid atom indices for the frame.
        :param atom_indices_for_frame: The atom indices for the frame.
        :param is_molecule_types: The molecule types.
        :param molecule_atom_lens: The molecule atom lengths.
        :param distogram_atom_indices: The distogram atom indices.
        :param molecule_atom_indices: The molecule atom indices.
        :param valid_molecule_atom_mask: The valid molecule atom mask.
        :return: The PAE labels, the PDE labels, the plDDT labels, and the resolved labels.
        """
        dtype = denoised_atom_pos.dtype
        device = denoised_atom_pos.device
        batch_size = denoised_atom_pos.shape[0]
        atom_seq_len = denoised_atom_pos.shape[1]

        # build molecule atom positions

        distogram_pos = atom_pos_aligned

        if not self.distogram_atom_resolution:
            # molecule_pos = einx.get_at('b [m] c, b n -> b n c', atom_pos, distogram_atom_indices)

            distogram_atom_coords_indices = repeat(
                distogram_atom_indices, "b n -> b n c", c=distogram_pos.shape[-1]
            )
            distogram_pos = distogram_pos.gather(1, distogram_atom_coords_indices)

        distogram_atom_coords_indices = repeat(
            distogram_atom_indices, "b n -> b n c", c=distogram_pos.shape[-1]
        )
        molecule_pos = atom_pos_aligned.gather(1, distogram_atom_coords_indices)

        # determine PAE labels if possible

        pae_labels = None

        if self.train_pae and exists(atom_indices_for_frame):
            denoised_molecule_pos = denoised_atom_pos.gather(1, distogram_atom_coords_indices)

            # get frame atom positions
            # three_atoms = einx.get_at('b [m] c, b n three -> three b n c', atom_pos_aligned, atom_indices_for_frame)
            # pred_three_atoms = einx.get_at('b [m] c, b n three -> three b n c', denoised_atom_pos, atom_indices_for_frame)

            atom_indices_for_frame = repeat(
                atom_indices_for_frame, "b n three -> three b n c", c=3
            )
            three_atom_pos = repeat(atom_pos_aligned, "b m c -> three b m c", three=3)
            three_denoised_atom_pos = repeat(denoised_atom_pos, "b m c -> three b m c", three=3)

            three_atoms = three_atom_pos.gather(2, atom_indices_for_frame)
            pred_three_atoms = three_denoised_atom_pos.gather(2, atom_indices_for_frame)

            # compute frames
            frame_atoms = rearrange(three_atoms, "three b n c -> b n c three")
            pred_frame_atoms = rearrange(pred_three_atoms, "three b n c -> b n c three")

            # determine mask
            # must be amino acid, nucleotide, or ligand with greater than 0 atoms
            align_error_mask = valid_atom_indices_for_frame

            # align error
            align_error = self.compute_alignment_error(
                denoised_molecule_pos.float(),
                molecule_pos.float(),
                pred_frame_atoms,  # NOTE: in paragraph 2 of AF3 Section 4.3.2, `\Phi_i` denotes the coordinates of the frame atoms rather than the frame rotation matrix
                frame_atoms,
                mask=align_error_mask,
            ).type(dtype)

            # calculate pae labels as alignment error binned to 64 (0 - 32A)
            pae_labels = distance_to_dgram(align_error, self.pae_bins, return_labels=True)

            # set ignore index for invalid molecules or frames
            pair_align_error_mask = to_pairwise_mask(align_error_mask)

            # pae_labels = einx.where(
            #     "b i j, b i j, -> b i j", pair_align_error_mask, pae_labels, self.ignore_index
            # )
            pae_labels = pae_labels.masked_fill(~pair_align_error_mask, self.ignore_index)

        # determine PDE labels

        # molecule_pos = einx.get_at('b [m] c, b n -> b n c', atom_pos_aligned, molecule_atom_indices)

        molecule_atom_coords_indices = repeat(
            molecule_atom_indices, "b n -> b n c", c=atom_pos_aligned.shape[-1]
        )

        molecule_pos = atom_pos_aligned.gather(1, molecule_atom_coords_indices)
        denoised_molecule_pos = denoised_atom_pos.gather(1, molecule_atom_coords_indices)

        molecule_mask = valid_molecule_atom_mask

        pde_gt_dist = torch.cdist(molecule_pos.float(), molecule_pos.float(), p=2).type(dtype)
        pde_pred_dist = torch.cdist(
            denoised_molecule_pos.float(),
            denoised_molecule_pos.float(),
            p=2,
        ).type(dtype)

        # calculate PDE labels as distance error binned to 64 (0 - 32A)
        pde_dist = torch.abs(pde_pred_dist - pde_gt_dist)
        pde_labels = distance_to_dgram(pde_dist, self.pde_bins, return_labels=True)

        # account for representative molecule atom missing from residue (`-1` set on the `molecule_atom_indices` field)
        pde_labels.masked_fill_(~to_pairwise_mask(molecule_mask), self.ignore_index)

        # determine plDDT labels if possible

        pred_coords, true_coords = denoised_atom_pos, atom_pos_aligned

        # compute distances between all pairs of atoms
        pred_dists = torch.cdist(pred_coords.float(), pred_coords.float(), p=2).type(dtype)
        true_dists = torch.cdist(true_coords.float(), true_coords.float(), p=2).type(dtype)

        # restrict to bespoke interaction types and inclusion radius on the atom level (Section 4.3.1)
        is_protein = batch_repeat_interleave(
            is_molecule_types[..., IS_PROTEIN_INDEX], molecule_atom_lens
        )
        is_rna = batch_repeat_interleave(is_molecule_types[..., IS_RNA_INDEX], molecule_atom_lens)
        is_dna = batch_repeat_interleave(is_molecule_types[..., IS_DNA_INDEX], molecule_atom_lens)

        is_nucleotide = is_rna | is_dna
        is_polymer = is_protein | is_rna | is_dna

        is_any_nucleotide_pair = repeat(
            is_nucleotide, "... j -> ... i j", i=is_nucleotide.shape[-1]
        )
        is_any_polymer_pair = repeat(is_polymer, "... j -> ... i j", i=is_polymer.shape[-1])

        inclusion_radius = torch.where(
            is_any_nucleotide_pair,
            true_dists < self.lddt_mask_nucleic_acid_cutoff,
            true_dists < self.lddt_mask_other_cutoff,
        )

        is_token_center_atom = torch.zeros_like(atom_pos_aligned[..., 0], dtype=torch.bool)
        is_token_center_atom[torch.arange(batch_size).unsqueeze(1), molecule_atom_indices] = True
        is_any_token_center_atom_pair = repeat(
            is_token_center_atom,
            "... j -> ... i j",
            i=is_token_center_atom.shape[-1],
        )

        # compute masks, avoiding self term
        plddt_mask = (
            inclusion_radius
            & is_any_polymer_pair
            & is_any_token_center_atom_pair
            & ~torch.eye(atom_seq_len, dtype=torch.bool, device=device)
        )

        plddt_mask = plddt_mask * to_pairwise_mask(atom_mask)

        # compute distance difference for all pairs of atoms
        dist_diff = torch.abs(true_dists - pred_dists)

        # lddt = einx.subtract(
        #     "thresholds, ... -> ... thresholds", self.lddt_thresholds, dist_diff
        # )
        lddt = self.lddt_thresholds[None, None, None, :] - dist_diff[..., None]
        lddt = (lddt >= 0).type(dtype).mean(dim=-1)

        # calculate masked averaging,
        # after which we assign each value to one of 50 equally sized bins
        lddt_mean = masked_average(lddt, plddt_mask, dim=-1)

        plddt_labels = torch.clamp(
            torch.floor(lddt_mean * self.num_plddt_bins).long(),
            max=self.num_plddt_bins - 1,
        )

        # account for missing atoms (`False` set on the `atom_mask` field)
        plddt_labels.masked_fill_(~atom_mask, self.ignore_index)

        # account for missing atoms in resolved labels (`False` set on the `atom_mask` field)

        if exists(resolved_labels):
            resolved_labels.masked_fill_(~atom_mask, self.ignore_index)

        return pae_labels, pde_labels, plddt_labels, resolved_labels

    @typecheck
    @staticmethod
    def calculate_cross_entropy_with_weight(
        logits: Float["b l ..."],  # type: ignore
        labels: Int["b ..."],  # type: ignore
        weight: Float[" b"],  # type: ignore
        mask: Bool["b ..."],  # type: ignore
        ignore_index: int,
    ) -> Float[""]:  # type: ignore
        """Compute cross entropy loss with weight and mask.

        :param logits: The logits.
        :param labels: The labels.
        :param weight: The weight.
        :param mask: The mask.
        :param ignore_index: The ignore index.
        :return: The loss.
        """
        labels = torch.where(mask, labels, ignore_index)

        # ensure (unused) logits are always in the computational graph

        if not (mask.any() and weight.any() and (mask * weight).any()):
            return (logits * 0.0).mean()

        ce = F.cross_entropy(
            logits,
            labels,
            ignore_index=ignore_index,
            reduction="none",
        )

        mean_ce = (ce * weight).sum() / (mask * weight).sum()

        return mean_ce

    @typecheck
    def calculate_confidence_loss(
        self,
        pae_logits: Float["b n n"],  # type: ignore
        pde_logits: Float["b n n"],  # type: ignore
        plddt_logits: Float["b m"],  # type: ignore
        resolved_logits: Float["b m"],  # type: ignore
        affinity_logits: List[Float[" *"]],  # type: ignore
        pae_labels: Int["b n n"] | None,  # type: ignore
        pde_labels: Int["b n n"] | None,  # type: ignore
        plddt_labels: Int["b m"] | None,  # type: ignore
        mask: Bool["b n"],  # type: ignore
        atom_mask: Bool["b m"],  # type: ignore
        resolved_labels: Int["b m"] | None,  # type: ignore
        affinity_labels: List[Float[" *"]] | None = None,  # type: ignore
        resolution: Float[" b"] | None = None,  # type: ignore
    ) -> Tuple[Float[""], Float[""], Float[""], Float[""], Float[""]]:  # type: ignore
        """Calculate the confidence loss.

        :param pae_logits: The PAE logits.
        :param pde_logits: The PDE logits.
        :param plddt_logits: The plDDT logits.
        :param resolved_logits: The resolved logits.
        :param affinity_logits: The affinity logits.
        :param pae_labels: The PAE labels.
        :param pde_labels: The PDE labels.
        :param plddt_labels: The plDDT labels.
        :param mask: The mask.
        :param atom_mask: The atom mask.
        :param resolved_labels: The resolved labels.
        :param affinity_labels: The affinity labels.
        :param resolution: The experimental resolution.
        :return: The confidence loss and the loss breakdown.
        """
        dtype = pde_logits.dtype
        device = pde_logits.device
        batch_size = pde_logits.shape[0]

        # determine which mask to use for confidence head labels

        label_mask = atom_mask
        label_pairwise_mask = to_pairwise_mask(mask)

        # prepare cross entropy losses

        confidence_mask = (
            (resolution >= self.min_conf_resolution) & (resolution <= self.max_conf_resolution)
            if exists(resolution)
            else torch.full((batch_size,), False, device=device)
        )

        confidence_weight = confidence_mask.type(dtype)

        # calculate PAE loss as requested

        if self.train_pae and exists(pae_labels):
            train_pae_weight = 1.0 if self.train_pae else 0.0
            pae_loss = (
                self.calculate_cross_entropy_with_weight(
                    pae_logits,
                    pae_labels,
                    confidence_weight,
                    label_pairwise_mask,
                    self.ignore_index,
                )
                * train_pae_weight
            )
        else:
            # ensure PAE logits always contribute to the loss
            pae_loss = (pae_logits * 0.0).mean()

        # calculate PDE loss as requested

        if exists(pde_labels):
            pde_loss = self.calculate_cross_entropy_with_weight(
                pde_logits,
                pde_labels,
                confidence_weight,
                label_pairwise_mask,
                self.ignore_index,
            )
        else:
            # ensure PDE logits always contribute to the loss
            pde_loss = (pde_logits * 0.0).mean()

        # calculate plDDT loss as requested

        if exists(plddt_labels):
            plddt_loss = self.calculate_cross_entropy_with_weight(
                plddt_logits, plddt_labels, confidence_weight, label_mask, self.ignore_index
            )
        else:
            # ensure plDDT logits always contribute to the loss
            plddt_loss = (plddt_logits * 0.0).mean()

        # calculate resolved loss as requested

        if exists(resolved_labels):
            resolved_loss = self.calculate_cross_entropy_with_weight(
                resolved_logits,
                resolved_labels,
                confidence_weight,
                label_mask,
                self.ignore_index,
            )
        else:
            # ensure resolved logits always contribute to the loss
            resolved_loss = (resolved_logits * 0.0).mean()

        # calculate affinity loss as requested

        if exists(affinity_labels):
            # find the (batched) mean squared error over all ligands in the same complex, then calculate the mean of each batch
            affinity_loss = sum(
                sum(
                    F.mse_loss(
                        affinity_logits[i][j],
                        (
                            affinity_labels[i][j]
                            if not affinity_labels[i][j].isnan().any()
                            else affinity_logits[i][j]
                        ),
                        reduction="none",
                    )
                    for j in range(len(affinity_logits[i]))
                )
                / len(affinity_logits[i])
                for i in range(len(affinity_logits))
            ) / len(affinity_logits)
        else:
            # ensure affinity logits always contribute to the loss
            affinity_loss = sum(
                sum((affinity_logits[i][j] * 0.0) for j in range(len(affinity_logits[i])))
                / len(affinity_logits[i])
                for i in range(len(affinity_logits))
            ) / len(affinity_logits)

        confidence_loss = pae_loss + pde_loss + plddt_loss + resolved_loss + affinity_loss

        return confidence_loss, pae_loss, pde_loss, plddt_loss, resolved_loss, affinity_loss

    @typecheck
    def forward(
        self,
        model_preds: Dict[str, Any],
        model_labels: Dict[str, Any],
        molecule_atom_lens: Int["b n"],  # type: ignore
        is_molecule_types: Bool["b n 5"] | None = None,  # type: ignore
        atom_mask: Bool["b m"] | None = None,  # type: ignore
        valid_atom_indices_for_frame: Bool["b n"] | None = None,  # type: ignore
        atom_indices_for_frame: Int["b n three"] | None = None,  # type: ignore
        bond_mask: Bool["b m m"] | None = None,  # type: ignore
        missing_atom_mask: Bool["b m"] | None = None,  # type: ignore
        distogram_atom_indices: Int["b n"] | None = None,  # type: ignore
        molecule_atom_indices: Int["b n"] | None = None,  # type: ignore
        valid_distogram_mask: Bool["b n"] | None = None,  # type: ignore
        valid_molecule_atom_mask: Bool["b n"] | None = None,  # type: ignore
    ) -> Tuple[Float[""], LossBreakdown]:  # type: ignore
        """Compute the loss for OmniFold.

        :param model_preds: The model predictions.
        :param model_labels: The model labels.
        :param molecule_atom_lens: The molecule atom lengths.
        :param atom_mask: The atom mask.
        :param valid_atom_indices_for_frame: The valid atom indices for the frame.
        :param atom_indices_for_frame: The atom indices for the frame.
        :param bond_mask: The bond mask.
        :param missing_atom_mask: The missing atom mask.
        :param distogram_atom_indices: The distogram atom indices.
        :param molecule_atom_indices: The molecule atom indices.
        :param valid_distogram_mask: The valid distogram mask.
        :param valid_molecule_atom_mask: The valid molecule atom mask.
        :return: The total loss and the loss breakdown.
        """
        dtype = molecule_atom_lens.dtype

        # default losses to 0

        distogram_loss = diffusion_loss = confidence_loss = pae_loss = pde_loss = plddt_loss = (
            resolved_loss
        ) = self.zero.type(dtype)

        # calculate masks

        mask = molecule_atom_lens > 0
        pairwise_mask = to_pairwise_mask(mask)

        # calculate the distogram loss as requested

        calculate_distogram_loss = (
            "distogram" in model_preds
            and "atom_pos" in model_labels
            and all(exists(t) for t in (atom_mask, distogram_atom_indices, valid_distogram_mask))
        )
        if calculate_distogram_loss:
            distance_labels = self.calculate_distogram_labels(
                atom_pos=model_labels["atom_pos"],
                atom_mask=atom_mask,
                distogram_atom_indices=distogram_atom_indices,
                valid_distogram_mask=valid_distogram_mask,
            )
            distogram_loss = self.calculate_distogram_loss(
                distogram_logits=model_preds["distogram"],
                distogram_labels=distance_labels,
                atom_mask=atom_mask,
                pairwise_mask=pairwise_mask,
            )

        # calculate the diffusion loss as requested

        calculate_diffusion_loss = (
            "denoised_atom_pos" in model_preds
            and all(
                l in model_labels
                for l in (
                    "diffusion_atom_pos_aligned",
                    "diffusion_align_weights",
                    "diffusion_loss_weights",
                )
            )
            and all(
                exists(t) for t in (atom_mask, bond_mask, missing_atom_mask, is_molecule_types)
            )
        )
        if calculate_diffusion_loss:
            diffusion_loss, diffusion_loss_breakdown = self.calculate_diffusion_loss(
                denoised_atom_pos=model_preds["denoised_atom_pos"],
                atom_pos_aligned=model_labels["diffusion_atom_pos_aligned"],
                align_weights=model_labels["diffusion_align_weights"],
                loss_weights=model_labels["diffusion_loss_weights"],
                molecule_atom_lens=molecule_atom_lens,
                atom_mask=atom_mask,
                bond_mask=bond_mask,
                missing_atom_mask=missing_atom_mask,
                is_molecule_types=is_molecule_types,
            )

        # calculate the confidence loss as requested

        skipped_confidence_loss = False

        calculate_confidence_loss = (
            "mini_denoised_atom_pos" in model_preds
            and "mini_aligned_atom_pos" in model_labels
            and exists(model_labels["mini_aligned_atom_pos"])
            and all(t in model_preds for t in ("pde", "pae", "plddt", "resolved"))
            and all(
                exists(t)
                for t in (
                    atom_mask,
                    valid_atom_indices_for_frame,
                    atom_indices_for_frame,
                    is_molecule_types,
                    molecule_atom_lens,
                    distogram_atom_indices,
                    molecule_atom_indices,
                    valid_molecule_atom_mask,
                )
            )
        )
        if calculate_confidence_loss:
            (
                pae_labels,
                pde_labels,
                plddt_labels,
                resolved_labels,
            ) = self.calculate_confidence_labels(
                denoised_atom_pos=model_preds["mini_denoised_atom_pos"],
                atom_pos_aligned=model_labels["mini_aligned_atom_pos"],
                resolved_labels=model_labels["resolved_labels"],
                atom_mask=atom_mask,
                valid_atom_indices_for_frame=valid_atom_indices_for_frame,
                atom_indices_for_frame=atom_indices_for_frame,
                is_molecule_types=is_molecule_types,
                molecule_atom_lens=molecule_atom_lens,
                distogram_atom_indices=distogram_atom_indices,
                molecule_atom_indices=molecule_atom_indices,
                valid_molecule_atom_mask=valid_molecule_atom_mask,
            )
        else:
            # ensure the confidence logits are always in the computational graph

            skipped_confidence_loss = True
            pae_labels = pde_labels = plddt_labels = resolved_labels = None

        (
            confidence_loss,
            pae_loss,
            pde_loss,
            plddt_loss,
            resolved_loss,
            affinity_loss,
        ) = self.calculate_confidence_loss(
            pae_logits=model_preds["pae"],
            pde_logits=model_preds["pde"],
            plddt_logits=model_preds["plddt"],
            resolved_logits=model_preds["resolved"],
            affinity_logits=model_preds["affinity"],
            pae_labels=pae_labels,
            pde_labels=pde_labels,
            plddt_labels=plddt_labels,
            mask=mask,
            atom_mask=atom_mask,
            resolved_labels=resolved_labels,
            affinity_labels=model_labels["affinities"],
            resolution=model_labels["resolution"],
        )

        # combine all the losses

        loss = (
            distogram_loss * self.distogram_weight
            + diffusion_loss * self.diffusion_weight
            + confidence_loss * self.confidence_weight
        )

        # nullify confidence loss if the confidence logits are not to be learned

        if skipped_confidence_loss:
            confidence_loss, pae_loss, pde_loss, plddt_loss, resolved_loss, affinity_loss = (
                torch.nan,
                torch.nan,
                torch.nan,
                torch.nan,
                torch.nan,
                torch.nan,
            )

        loss_breakdown = LossBreakdown(
            total_loss=loss,
            total_diffusion=diffusion_loss,
            pae=pae_loss,
            pde=pde_loss,
            plddt=plddt_loss,
            resolved=resolved_loss,
            affinity=affinity_loss,
            distogram=distogram_loss,
            confidence=confidence_loss,
            **diffusion_loss_breakdown._asdict(),
        )

        return loss, loss_breakdown

