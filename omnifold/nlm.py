from functools import wraps

import torch
from beartype.typing import Literal
from torch.nn import Module

from omnifold.common.biomolecule import get_residue_constants
from omnifold.inputs import IS_DNA, IS_RNA
from omnifold.tensor_typing import Float, Int, typecheck
from omnifold.utils.data_utils import join

# functions


def remove_nlms(fn):
    """Decorator to remove NLMs from the model before calling the inner function and then restore
    them afterwards."""

    @wraps(fn)
    def inner(self, *args, **kwargs):
        has_nlms = hasattr(self, "nlms")
        if has_nlms:
            nlms = self.nlms
            delattr(self, "nlms")

        out = fn(self, *args, **kwargs)

        if has_nlms:
            self.nlms = nlms

        return out

    return inner


# constants

rna_constants = get_residue_constants(res_chem_index=IS_RNA)
dna_constants = get_residue_constants(res_chem_index=IS_DNA)

rna_restypes = rna_constants.restypes + ["X"]
dna_restypes = dna_constants.restypes + ["X"]

rna_min_restype_num = rna_constants.min_restype_num
dna_min_restype_num = dna_constants.min_restype_num

RINALMO_MASK_TOKEN = "-"  # nosec

# class


class RiNALMoWrapper(Module):
    """A wrapper for the RiNALMo model to provide NLM embeddings."""

    def __init__(self):
        super().__init__()
        from multimolecule import RiNALMoModel, RnaTokenizer

        self.model = RiNALMoModel.from_pretrained("multimolecule/rinalmo", add_pooling_layer=False)
        self.tokenizer = RnaTokenizer.from_pretrained(
            "multimolecule/rinalmo", replace_T_with_U=False
        )

        self.model = self.model.cpu()

        self.embed_dim = 1280

    @torch.no_grad()
    @typecheck
    def forward(
        self, na_ids: Int["b n"]  # type: ignore
    ) -> Float["b n dne"]:  # type: ignore
        """Get NLM embeddings for a batch of (pseudo-)nucleotide sequences.

        :param na_ids: A batch of nucleotide residue indices.
        :return: The NLM embeddings for the input sequences.
        """
        device, seq_len = na_ids.device, na_ids.shape[-1]

        sequence_data = [
            join(
                [
                    (
                        RINALMO_MASK_TOKEN
                        if i == -1
                        else (
                            dna_restypes[i - dna_min_restype_num]
                            if i >= dna_min_restype_num
                            else rna_restypes[i - rna_min_restype_num]
                        )
                    )
                    for i in ids
                ]
            )
            for ids in na_ids
        ]

        # encode to ids

        inputs = self.tokenizer(sequence_data, return_tensors="pt")
        input_ids, attention_mask = inputs.input_ids.to(device), inputs.attention_mask.to(device)

        # forward through nlm

        embeddings = self.model(input_ids, attention_mask=attention_mask)

        # remove prefix

        nlm_embeddings = embeddings.last_hidden_state[:, 1 : (seq_len + 1)]

        return nlm_embeddings


# NLM embedding type and registry

NLMRegistry = dict(rinalmo=RiNALMoWrapper)

NLMEmbedding = Literal["rinalmo"]
