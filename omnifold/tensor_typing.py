from __future__ import annotations

import importlib.metadata
from functools import partial

import numpy as np
import sh
import torch
from beartype import beartype
from beartype.door import is_bearable
from Bio.PDB.Atom import Atom, DisorderedAtom
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import DisorderedResidue, Residue
from environs import Env
from jaxtyping import Bool, Float, Int, Shaped, jaxtyped
from packaging import version
from torch import Tensor

from omnifold.utils.utils import always, identity

# environment

env = Env()
env.read_env()


# NOTE: `jaxtyping` is a misnomer, works for PyTorch as well


class TorchTyping:
    """Torch typing."""

    def __init__(self, abstract_dtype):
        self.abstract_dtype = abstract_dtype

    def __getitem__(self, shapes: str):
        """Get item."""
        return self.abstract_dtype[Tensor, shapes]


Shaped = TorchTyping(Shaped)
Float = TorchTyping(Float)
Int = TorchTyping(Int)
Bool = TorchTyping(Bool)

# helper type aliases

IntType = int | np.int32 | np.int64
AtomType = Atom | DisorderedAtom
ResidueType = Residue | DisorderedResidue
ChainType = Chain
TokenType = AtomType | ResidueType

# some more colocated environmental stuff


def package_available(package_name: str) -> bool:
    """Check if a package is available in your environment.

    :param package_name: The name of the package to be checked.
    :return: `True` if the package is available. `False` otherwise.
    """
    try:
        importlib.metadata.version(package_name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


# maybe deespeed checkpoint, and always use non reentrant checkpointing

DEEPSPEED_CHECKPOINTING = env.bool("DEEPSPEED_CHECKPOINTING", False)
from omnifold.evoformer import EvoformerAttention
if DEEPSPEED_CHECKPOINTING:
    assert package_available("deepspeed"), "DeepSpeed must be installed for checkpointing."

    import deepspeed

    checkpoint = deepspeed.checkpointing.non_reentrant_checkpoint
else:
    checkpoint = partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
    #def policy_fn(ctx, op, *args, **kwargs):
    #    print("Op: ", op)
    #    if op == EvoformerAttention:
    #        return torch.utils.checkpoint.CheckpointPolicy.MUST_SAVE
    #    else:
    #        return torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE
    #checkpoint = partial(torch.utils.checkpoint.checkpoint, use_reentrant=False, context_fn=partial(torch.utils.checkpoint.create_selective_checkpoint_contexts, policy_fn))

# whether to use Nim or not, depending on if available and version is adequate
try:
    sh.which("nim")
    HAS_NIM = True
    NIM_VERSION = sh.nim(eval="echo NimVersion", hints="off")
except sh.ErrorReturnCode_1:
    HAS_NIM = False
    NIM_VERSION = None

USE_NIM = env.bool("USE_NIM", HAS_NIM)

assert not (USE_NIM and not HAS_NIM), "you cannot use Nim if it is not available"
assert not HAS_NIM or version.parse(NIM_VERSION) >= version.parse(
    "2.0.8"
), "nim version must be 2.0.8 or above"

# check is GitHub CI

IS_GITHUB_CI = env.bool("IS_GITHUB_CI", False)

# NOTE: use env variable `TYPECHECK` to control whether to use `beartype` + `jaxtyping`
# NOTE: use env variable `DEBUG` to control whether to print debugging information

should_typecheck = env.bool("TYPECHECK", False)
IS_DEBUGGING = env.bool("DEBUG", False)

typecheck = jaxtyped(typechecker=beartype) if should_typecheck else identity

beartype_isinstance = is_bearable if should_typecheck else always(True)

if should_typecheck:
    print("Type checking is enabled.")
else:
    print("Type checking is disabled.")

if IS_DEBUGGING:
    print("Debugging is enabled.")
else:
    print("Debugging is disabled.")

__all__ = [
    beartype_isinstance,
    Bool,
    Float,
    Int,
    Shaped,
    checkpoint,
    should_typecheck,
    typecheck,
    IS_DEBUGGING,
    IS_GITHUB_CI,
    USE_NIM,
]
