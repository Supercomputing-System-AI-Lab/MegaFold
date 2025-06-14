"""MSA loading functions used in OmniFold."""

# From: https://github.com/google-deepmind/alphafold/blob/f251de6613cb478207c732bf9627b1e853c99c2f/alphafold/data/parsers.py#L157

import dataclasses
import hashlib
import random
import re
import string

from beartype.typing import Literal, Sequence, Tuple
from cachetools import LRUCache, cached

from omnifold.tensor_typing import typecheck
from omnifold.utils.utils import exists, not_exists

DeletionMatrix = Sequence[Sequence[int]]

# Constants

MSA_TYPE = Literal["protein", "dna", "rna", "ligand"]

# Utilities for extracting identifiers from MSA sequence descriptions.


# Sequences coming from UniProtKB database come in the
# `db|UniqueIdentifier|EntryName` format, e.g. `tr|A0A146SKV9|A0A146SKV9_FUNHE`
# or `sp|P0C2L1|A3X1_LOXLA` (for TREMBL/Swiss-Prot respectively).
_UNIPROT_PATTERN = re.compile(
    r"""
    ^
    # UniProtKB/TrEMBL or UniProtKB/Swiss-Prot
    (?:tr|sp)
    \|
    # A primary accession number of the UniProtKB entry.
    (?P<AccessionIdentifier>[A-Za-z0-9]{6,10})
    # Occasionally there is a _0 or _1 isoform suffix, which we ignore.
    (?:_\d)?
    \|
    # TREMBL repeats the accession ID here. Swiss-Prot has a mnemonic
    # protein ID code.
    (?:[A-Za-z0-9]+)
    _
    # A mnemonic species identification code.
    (?P<SpeciesIdentifier>([A-Za-z0-9]){1,5})
    # Small BFD uses a final value after an underscore, which we ignore.
    (?:_\d+)?
    $
    """,
    re.VERBOSE,
)


@dataclasses.dataclass(frozen=True)
class Identifiers:
    species_id: str = ""


@typecheck
def get_msa_type(msa_chem_type: int) -> MSA_TYPE:
    """Get the molecule type of a residue.

    :param msa_chem_type: The chemical type of the MSA.
    :return: The MSA type.
    """
    if msa_chem_type == 0:
        return "protein"
    elif msa_chem_type == 1:
        return "rna"
    elif msa_chem_type == 2:
        return "dna"
    elif msa_chem_type in {3, 4}:
        return "ligand"
    else:
        raise ValueError(f"Invalid MSA chemical type: {msa_chem_type}")


@typecheck
def _parse_inference_species_identifier(description: str) -> Identifiers:
    """Gets species from an inference-specific MSA sequence identifier.

    :param description: a sequence identifier.
    :return: An `Identifiers` instance with species_id. These
        can be empty in the case where no identifier was found.
    """
    split_description = description.split("_")
    if len(split_description) > 1:
        return Identifiers(species_id=split_description[-1].strip())
    return Identifiers()


@typecheck
def _parse_species_identifier(description: str) -> Identifiers:
    """Gets species from an MSA sequence identifier.

    The sequence identifier in this instance has a tab-separated format,
    except for the query identifier which is not linked to a species.

    :param description: a sequence identifier.
    :return: An `Identifiers` instance with species_id. These
        can be empty in the case where no identifier was found.
    """
    split_description = description.split("\t")
    if len(split_description) > 1:
        return Identifiers(species_id=split_description[-1].strip())
    return Identifiers()


@typecheck
def _parse_sequence_identifier(msa_sequence_identifier: str) -> Identifiers:
    """Gets species from an MSA sequence identifier.

    The sequence identifier has the format specified by
    _UNIPROT_TREMBL_ENTRY_NAME_PATTERN or _UNIPROT_SWISSPROT_ENTRY_NAME_PATTERN.
    An example of a sequence identifier: `tr|A0A146SKV9|A0A146SKV9_FUNHE`

    :param msa_sequence_identifier: a sequence identifier.
    :return: An `Identifiers` instance with species_id. These
        can be empty in the case where no identifier was found.
    """
    matches = re.search(_UNIPROT_PATTERN, msa_sequence_identifier.strip())
    if matches:
        return Identifiers(species_id=matches.group("SpeciesIdentifier"))
    return Identifiers()


@typecheck
def _extract_sequence_identifier(description: str) -> str | None:
    """Extracts sequence identifier from description.

    :param description: a sequence description.
    :return: The sequence identifier.
    """
    split_description = description.split()
    if split_description:
        return split_description[0].partition("/")[0]
    else:
        return None


@typecheck
def _get_identifiers_make_key(
    description: str, tab_separated_alignment_headers: bool = False, inference: bool = False
) -> str:
    """Computes a key for the sequence description.

    :param description: The description of the sequence.
    :param tab_separated_alignment_headers: Whether the alignment headers are tab-separated.
    :param inference: Whether to extract the species identifier using inference-specific logic.
    :return: A key for the sequence description.
    """
    md5_digest = hashlib.md5(description.encode()).hexdigest()  # nosec
    return f"{md5_digest}:{tab_separated_alignment_headers}:{inference}"


@typecheck
@cached(cache=LRUCache(maxsize=512), key=_get_identifiers_make_key)
def get_identifiers(
    description: str, tab_separated_alignment_headers: bool = False, inference: bool = False
) -> Identifiers:
    """Computes extra MSA features from the description.

    :param description: The description of the sequence.
    :param tab_separated_alignment_headers: Whether the alignment headers are tab-separated.
    :param inference: Whether to extract the species identifier using inference-specific logic.
    :return: An `Identifiers` instance with species_id. These can be empty in the case
        where no identifier was found.
    """
    if inference:
        return _parse_inference_species_identifier(description)
    elif tab_separated_alignment_headers:
        return _parse_species_identifier(description)
    else:
        sequence_identifier = _extract_sequence_identifier(description)
        if not_exists(sequence_identifier):
            return Identifiers()
        else:
            return _parse_sequence_identifier(sequence_identifier)


@dataclasses.dataclass(frozen=True)
class Msa:
    """Class representing a parsed MSA file."""

    sequences: Sequence[str]
    deletion_matrix: DeletionMatrix
    descriptions: Sequence[str]
    msa_type: MSA_TYPE
    is_unfiltered: Sequence[bool]

    def __post_init__(self):
        """Checks that all fields have the same length."""
        if not (
            len(self.sequences)
            == len(self.deletion_matrix)
            == len(self.descriptions)
            == len(self.is_unfiltered)
        ):
            raise ValueError(
                "All fields for an MSA must have the same length. "
                f"Got {len(self.sequences)} sequences, "
                f"{len(self.deletion_matrix)} rows in the deletion matrix, "
                f"{len(self.descriptions)} descriptions, and "
                f"{len(self.is_unfiltered)} unfiltered sequence flags."
            )

    def __len__(self):
        """Returns the number of sequences in the MSA."""
        return len(self.sequences)

    def __add__(self, other):
        """Concatenates two MSAs."""
        return Msa(
            sequences=self.sequences + other.sequences,
            deletion_matrix=self.deletion_matrix + other.deletion_matrix,
            descriptions=self.descriptions + other.descriptions,
            msa_type=self.msa_type,
            is_unfiltered=self.is_unfiltered + other.is_unfiltered,
        )

    def truncate(self, max_seqs: int):
        """Truncates the MSA to the first `max_seqs` sequences."""
        max_seqs = min(len(self.sequences), max_seqs)
        return Msa(
            sequences=self.sequences[:max_seqs],
            deletion_matrix=self.deletion_matrix[:max_seqs],
            descriptions=self.descriptions[:max_seqs],
            msa_type=self.msa_type,
            is_unfiltered=self.is_unfiltered[:max_seqs],
        )

    def random_truncate(self, max_seqs: int):
        """Truncates the MSA to a random range of `max_seqs` sequences."""
        max_seqs = min(len(self.sequences), max_seqs)
        start = random.randint(0, len(self.sequences) - max_seqs)  # nosec
        return Msa(
            sequences=self.sequences[start : start + max_seqs],
            deletion_matrix=self.deletion_matrix[start : start + max_seqs],
            descriptions=self.descriptions[start : start + max_seqs],
            msa_type=self.msa_type,
            is_unfiltered=self.is_unfiltered[start : start + max_seqs],
        )

    def move_unfiltered_to_end(self):
        """If possible, moves the unfiltered sequences to the end of the MSA."""
        return Msa(
            sequences=[
                s
                for s, is_unfiltered in zip(self.sequences, self.is_unfiltered)
                if not is_unfiltered
            ]
            + [s for s, is_unfiltered in zip(self.sequences, self.is_unfiltered) if is_unfiltered],
            deletion_matrix=[
                d
                for d, is_unfiltered in zip(self.deletion_matrix, self.is_unfiltered)
                if not is_unfiltered
            ]
            + [
                d
                for d, is_unfiltered in zip(self.deletion_matrix, self.is_unfiltered)
                if is_unfiltered
            ],
            descriptions=[
                d
                for d, is_unfiltered in zip(self.descriptions, self.is_unfiltered)
                if not is_unfiltered
            ]
            + [
                d
                for d, is_unfiltered in zip(self.descriptions, self.is_unfiltered)
                if is_unfiltered
            ],
            msa_type=self.msa_type,
            is_unfiltered=[False] * sum(not is_unfiltered for is_unfiltered in self.is_unfiltered)
            + [True] * sum(is_unfiltered for is_unfiltered in self.is_unfiltered),
        )


@typecheck
def parse_fasta(
    fasta_string: str, max_sequences: int | None = None
) -> Tuple[Sequence[str], Sequence[str]]:
    """Parses FASTA string and returns list of strings with amino-acid sequences.

    :param fasta_string: The string contents of a FASTA file.
    :param max_sequences: The maximum number of sequences to parse from the MSA.
    :return: A tuple of two lists:
        * A list of sequences.
        * A list of sequence descriptions taken from the comment lines. In the
            same order as the sequences.
    """
    sequences = []
    descriptions = []

    index = -1
    num_sequences_parsed = 0

    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith(">"):
            index += 1
            descriptions.append(line[1:])  # Remove the '>' at the beginning.
            sequences.append("")
            continue
        elif not line:
            continue  # Skip blank lines.
        sequences[index] += line
        num_sequences_parsed += 1

        if exists(max_sequences) and num_sequences_parsed >= max_sequences:
            break

    return sequences, descriptions


@typecheck
def parse_a3m(
    a3m_string: str, msa_type: MSA_TYPE, max_sequences: int | None = None, unfiltered: bool = False
) -> Msa:
    """Parses sequences and deletion matrix from a3m format alignment.

    :param a3m_string: The string contents of a a3m file. The first sequence in the
        file should be the query sequence.
    :param msa_type: The type of the sequences in the MSA. This can be 'protein',
        'dna', 'rna', or 'ligand'.
    :param max_sequences: The maximum number of sequences to parse from the MSA.
    :param unfiltered: Whether the MSA is unfiltered and is to be used for MSA pairing.
    :return: A tuple of:
        * A list of sequences that have been aligned to the query. These
            might contain duplicates.
        * The deletion matrix for the alignment as a list of lists. The element
            at `deletion_matrix[i][j]` is the number of residues deleted from
            the aligned sequence i at residue position j.
        * A list of descriptions, one per sequence, from the a3m file.
        * The type of the sequences in the MSA.
        * A list of flags indicating whether each sequence is unfiltered.
    """

    sequences, descriptions = parse_fasta(a3m_string, max_sequences=max_sequences)
    deletion_matrix = []
    for msa_sequence in sequences:
        deletion_vec = []
        deletion_count = 0
        for j in msa_sequence:
            if j.islower():
                deletion_count += 1
            else:
                deletion_vec.append(deletion_count)
                deletion_count = 0
        deletion_matrix.append(deletion_vec)

    # Make the MSA matrix out of aligned (deletion-free) sequences.
    deletion_table = str.maketrans("", "", string.ascii_lowercase)
    aligned_sequences = [
        (
            s.translate(deletion_table).replace("U", "T")
            if msa_type == "dna"
            else s.translate(deletion_table)
        )
        for s in sequences
    ]
    is_unfiltered = [unfiltered for _ in sequences]
    return Msa(
        sequences=aligned_sequences,
        deletion_matrix=deletion_matrix,
        descriptions=descriptions,
        msa_type=msa_type,
        is_unfiltered=is_unfiltered,
    )
