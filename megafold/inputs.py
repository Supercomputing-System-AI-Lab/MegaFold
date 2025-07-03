from __future__ import annotations

import copy
import glob
import gzip
import json
import os
import random  # nosec
import re
import statistics
import traceback
from collections import defaultdict
from collections.abc import Iterable
from contextlib import redirect_stderr
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from functools import partial
from io import StringIO
from itertools import groupby
from pathlib import Path

# import einx
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import wrapt_timeout_decorator
from beartype.typing import Any, Callable, Dict, List, Literal, Set, Tuple, Type
from einops import pack, rearrange
from joblib import Parallel, delayed
from loguru import logger
from pdbeccdutils.core import ccd_reader
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import AllChem, rdDetermineBonds
from rdkit.Chem.rdchem import Atom, Mol
from rdkit.Geometry import Point3D
from torch import repeat_interleave, tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from megafold.model.attention import (
    full_attn_bias_to_windowed,
    full_pairwise_repr_to_windowed,
)
from megafold.common import (
    amino_acid_constants,
    dna_constants,
    ligand_constants,
    rna_constants,
)
from megafold.common.biomolecule import (
    Biomolecule,
    _from_mmcif_object,
    get_residue_constants,
    get_resname_and_modified_status,
    to_mmcif,
)
from megafold.data import mmcif_parsing, msa_pairing, msa_parsing, template_parsing
from megafold.data.data_pipeline import (
    FeatureDict,
    make_msa_features,
    make_msa_mask,
    make_template_features,
    merge_chain_features,
)
from megafold.data.weighted_pdb_sampler import WeightedPDBSampler
from megafold.life import (
    ATOM_BONDS,
    ATOMS,
    DNA_NUCLEOTIDES,
    HUMAN_AMINO_ACIDS,
    RNA_NUCLEOTIDES,
    generate_conformation,
    mol_from_smile,
    reverse_complement,
    reverse_complement_tensor,
)
from megafold.tensor_typing import Bool, Float, Int, typecheck
from megafold.utils.data_utils import (
    PDB_INPUT_RESIDUE_MOLECULE_TYPE,
    extract_mmcif_metadata_field,
    get_atom_perms,
    get_pdb_input_residue_molecule_type,
    get_residue_molecule_type,
    get_residue_molecule_type_index,
    index_to_pdb_chain_id,
    is_atomized_residue,
    is_gzip_file,
    is_polymer,
    make_one_hot,
    parse_pdbbind_binding_affinity_data_file,
)
from megafold.utils.model_utils import (
    create_uid_tensor,
    distance_to_dgram,
    exclusive_cumsum,
    get_frames_from_atom_pos,
    maybe,
    offset_only_positive,
    pack_one,
    pad_at_dim,
    remove_consecutive_duplicate,
    to_pairwise_mask,
)
from megafold.utils.utils import default, exists, first, maybe_cache, not_exists
from scripts.cluster_pdb_train_mmcifs import (
    NUCLEIC_LETTERS_3TO1_EXTENDED,
    PROTEIN_LETTERS_3TO1_EXTENDED,
)

# silence RDKit's warnings

RDLogger.DisableLog("rdApp.*")

# NOTE: see https://github.com/rdkit/rdkit/issues/7922 for more details
RDKIT_BOND_DETERMINATION_MAX_ATOMS = 31

# constants

PDB_INPUT_CHAIN_SEQS = List[List[Tuple[str, str, PDB_INPUT_RESIDUE_MOLECULE_TYPE]]]

PDB_INPUT_TO_MOLECULE_INPUT_MAX_SECONDS_PER_INPUT = 30
MOLECULE_TO_ATOM_INPUT_MAX_SECONDS_PER_INPUT = 10

IS_MOLECULE_TYPES = 5
IS_PROTEIN_INDEX = 0
IS_RNA_INDEX = 1
IS_DNA_INDEX = 2
IS_LIGAND_INDEX = -2
IS_METAL_ION_INDEX = -1

IS_BIOMOLECULE_INDICES = slice(0, 3)
IS_NON_PROTEIN_INDICES = slice(1, 5)
IS_NON_NA_INDICES = [0, 3, 4]

IS_PROTEIN, IS_RNA, IS_DNA, IS_LIGAND, IS_METAL_ION = tuple(
    (IS_MOLECULE_TYPES + i if i < 0 else i)
    for i in [
        IS_PROTEIN_INDEX,
        IS_RNA_INDEX,
        IS_DNA_INDEX,
        IS_LIGAND_INDEX,
        IS_METAL_ION_INDEX,
    ]
)

MOLECULE_GAP_ID = len(HUMAN_AMINO_ACIDS) + len(RNA_NUCLEOTIDES) + len(DNA_NUCLEOTIDES)
MOLECULE_METAL_ION_ID = MOLECULE_GAP_ID + 1
NUM_MOLECULE_IDS = len(HUMAN_AMINO_ACIDS) + len(RNA_NUCLEOTIDES) + len(DNA_NUCLEOTIDES) + 2

NUM_HUMAN_AMINO_ACIDS = len(HUMAN_AMINO_ACIDS) - 1  # exclude unknown amino acid type
NUM_MSA_ONE_HOT = len(HUMAN_AMINO_ACIDS) + len(RNA_NUCLEOTIDES) + len(DNA_NUCLEOTIDES) + 1

MIN_RNA_NUCLEOTIDE_ID = len(HUMAN_AMINO_ACIDS)
MAX_DNA_NUCLEOTIDE_ID = len(HUMAN_AMINO_ACIDS) + len(RNA_NUCLEOTIDES) + len(DNA_NUCLEOTIDES) - 1

MISSING_RNA_NUCLEOTIDE_ID = len(HUMAN_AMINO_ACIDS) + len(RNA_NUCLEOTIDES) - 1

DEFAULT_NUM_MOLECULE_MODS = 4  # `mod_protein`, `mod_rna`, `mod_dna`, and `mod_unk`
ADDITIONAL_MOLECULE_FEATS = 5

CONSTRAINTS = Literal["pocket", "contact", "docking"]
CONSTRAINT_DIMS = {
    # A mapping of constraint types to their respective input embedding dimensionalities.
    "pocket": 1,
    "contact": 1,
    "docking": 4,
}
CONSTRAINTS_MASK_VALUE = -1.0

INPUT_CONSTRAINTS = List[CONSTRAINTS] | Dict[CONSTRAINTS, List[int] | List[Tuple[int, ...]]]

CCD_COMPONENTS_FILEPATH = os.path.join("data", "ccd_data", "components.cif")
CCD_COMPONENTS_SMILES_FILEPATH = os.path.join("data", "ccd_data", "components_smiles.json")

# load all SMILES strings in the PDB Chemical Component Dictionary (CCD)

CCD_COMPONENTS_SMILES = None

if os.path.exists(CCD_COMPONENTS_SMILES_FILEPATH):
    print(f"Loading CCD component SMILES strings from {CCD_COMPONENTS_SMILES_FILEPATH}.")
    with open(CCD_COMPONENTS_SMILES_FILEPATH) as f:
        CCD_COMPONENTS_SMILES = json.load(f)
elif os.path.exists(CCD_COMPONENTS_FILEPATH):
    print(
        f"Loading CCD components from {CCD_COMPONENTS_FILEPATH} to extract all available SMILES strings (~3 minutes, one-time only)."
    )
    CCD_COMPONENTS = ccd_reader.read_pdb_components_file(
        CCD_COMPONENTS_FILEPATH,
        sanitize=False,  # Reduce loading time
    )
    print(
        f"Saving CCD component SMILES strings to {CCD_COMPONENTS_SMILES_FILEPATH} (one-time only)."
    )
    with open(CCD_COMPONENTS_SMILES_FILEPATH, "w") as f:
        CCD_COMPONENTS_SMILES = {
            ccd_code: Chem.MolToSmiles(CCD_COMPONENTS[ccd_code].component.mol_no_h)
            for ccd_code in CCD_COMPONENTS
        }
        json.dump(CCD_COMPONENTS_SMILES, f)


# simple caching

ATOMPAIR_IDS_CACHE = dict()

# get atompair bonds functions


@typecheck
def get_atompair_ids(mol: Mol, atom_bonds: List[str], directed_bonds: bool) -> Int["m m"] | None:  # type: ignore
    """Get atompair IDs from an RDKit molecule."""
    coordinates = []
    updates = []

    num_atoms = mol.GetNumAtoms()
    mol_atompair_ids = torch.zeros(num_atoms, num_atoms).long()

    bonds = mol.GetBonds()
    num_bonds = len(bonds)

    atom_bond_index = {symbol: (idx + 1) for idx, symbol in enumerate(atom_bonds)}

    num_atom_bond_types = len(atom_bond_index) + 1
    other_index = num_atom_bond_types

    for bond in bonds:
        atom_start_index = bond.GetBeginAtomIdx()
        atom_end_index = bond.GetEndAtomIdx()

        coordinates.extend(
            [
                [atom_start_index, atom_end_index],
                [atom_end_index, atom_start_index],
            ]
        )

        bond_type = str(bond.GetBondType())
        bond_id = atom_bond_index.get(bond_type, other_index)

        # default to symmetric bond type (undirected atom bonds)

        bond_to = bond_from = bond_id

        # if allowing for directed bonds, assume num_atompair_embeds = (2 * num_atom_bond_types) + 1
        # offset other edge by num_atom_bond_types

        if directed_bonds:
            bond_from += num_atom_bond_types

        updates.extend([bond_to, bond_from])

    if num_bonds == 0:
        return None

    coordinates = tensor(coordinates).long()
    updates = tensor(updates).long()

    # mol_atompair_ids = einx.set_at("[h w], c [2], c -> [h w]", mol_atompair_ids, coordinates, updates)

    molpair_strides = tensor(mol_atompair_ids.stride())
    flattened_coordinates = (coordinates * molpair_strides).sum(dim=-1)

    packed_atompair_ids, unpack_one = pack_one(mol_atompair_ids, "*")
    packed_atompair_ids[flattened_coordinates] = updates

    mol_atompair_ids = unpack_one(packed_atompair_ids)

    return mol_atompair_ids


@typecheck
def get_mol_has_bond(mol: Mol) -> Bool["m m"] | None:  # type: ignore
    """Get a mask of whether a bond exists between two atoms in an RDKit molecule."""
    coordinates = []

    bonds = mol.GetBonds()
    num_bonds = len(bonds)

    for bond in bonds:
        atom_start_index = bond.GetBeginAtomIdx()
        atom_end_index = bond.GetEndAtomIdx()

        coordinates.extend(
            [
                [atom_start_index, atom_end_index],
                [atom_end_index, atom_start_index],
            ]
        )

    if num_bonds == 0:
        return None

    num_atoms = mol.GetNumAtoms()
    has_bond = torch.zeros(num_atoms, num_atoms).bool()

    coordinates = tensor(coordinates).long()

    # has_bond = einx.set_at("[h w], c [2], c -> [h w]", has_bond, coordinates, updates)

    has_bond_stride = tensor(has_bond.stride())
    flattened_coordinates = (coordinates * has_bond_stride).sum(dim=-1)
    packed_has_bond, unpack_has_bond = pack_one(has_bond, "*")

    packed_has_bond[flattened_coordinates] = True
    has_bond = unpack_has_bond(packed_has_bond, "*")

    return has_bond


# functions


def flatten(arr):
    """Flatten a list of lists."""
    return [el for sub_arr in arr for el in sub_arr]


def without_keys(d: dict, exclude: set):
    """Remove keys from a dictionary."""
    return {k: v for k, v in d.items() if k not in exclude}


def pad_to_len(t, length, value=0, dim=-1):
    """Pad a tensor to a certain length."""
    assert dim < 0
    zeros = (0, 0) * (-dim - 1)
    return F.pad(t, (*zeros, 0, max(0, length - t.shape[dim])), value=value)


def compose(*fns: Callable):
    """Chain e.g., from MegaFoldInput -> MoleculeInput -> AtomInput."""

    def inner(x, *args, **kwargs):
        """Compose the functions."""
        for fn in fns:
            x = fn(x, *args, **kwargs)
        return x

    return inner


def compose_calls(*fns: Callable):
    """Chain one function call after the other."""

    def inner(*args, **kwargs):
        """Compose the functions."""
        for fn in fns:
            fn(*args, **kwargs)

    return inner


# validation functions


def hard_validate_atom_indices_ascending(
    indices: Int["b n"] | Int["b n 3"], error_msg_field: str = "indices", mask: Bool["b n"] | None = None  # type: ignore
):
    """Perform a hard validation on atom indices to ensure they are ascending. The function asserts
    if any of the indices that are not -1 (missing) are identical or descending. This will cover
    'distogram_atom_indices', 'molecule_atom_indices', and 'atom_indices_for_frame'.

    :param indices: The indices to validate.
    :param error_msg_field: The error message field.
    :param mask: The mask to apply to the indices. Note that, when a mask is specified, only masked
        values are expected to be ascending.
    """

    if indices.ndim == 2:
        indices = rearrange(indices, "... -> ... 1")

    for batch_index, sample_indices in enumerate(indices):
        if exists(mask):
            sample_indices = sample_indices[mask[batch_index]]

        all_present = (sample_indices >= 0).all(dim=-1)
        present_indices = sample_indices[all_present]

        # NOTE: this is a relaxed assumption, i.e., that if empty, all -1, or only one molecule, then it passes the test

        if present_indices.numel() == 0 or present_indices.shape[0] <= 1:
            continue

        # difference = einx.subtract(
        #     "n i, n j -> n (i j)", present_indices[1:], present_indices[:-1]
        # )
        difference = present_indices[1:] - present_indices[:-1]

        assert (
            difference >= 0
        ).all(), f"Detected invalid {error_msg_field} for a batch: {present_indices} with differences: {difference}"


# atom level, what MegaFold accepts

UNCOLLATABLE_ATOM_INPUT_FIELDS = {
    "filepath",
    "example_source",
    "molecule_atom_perms",
    "affinities",
    "num_ligands",
}

ATOM_INPUT_EXCLUDE_MODEL_FIELDS = {}

ATOM_DEFAULT_PAD_VALUES = dict(molecule_atom_lens=0, missing_atom_mask=True)


@typecheck
@dataclass
class AtomInput:
    """Dataclass for atom-level inputs."""

    atom_inputs: Float["m dai"]  # type: ignore
    molecule_ids: Int[" n"]  # type: ignore
    molecule_atom_lens: Int[" n"]  # type: ignore
    atompair_inputs: Float["m m dapi"] | Float["nw w (w*2) dapi"]  # type: ignore
    additional_molecule_feats: Int[f"n {ADDITIONAL_MOLECULE_FEATS}"]  # type: ignore
    is_molecule_types: Bool[f"n {IS_MOLECULE_TYPES}"]  # type: ignore
    molecule_atom_perms: List[List[int]] | None = None  # type: ignore
    is_molecule_mod: Bool["n num_mods"] | None = None  # type: ignore
    additional_msa_feats: Float["s n dmf"] | None = None  # type: ignore
    additional_token_feats: Float["n dtf"] | None = None  # type: ignore
    templates: Float["t n n dt"] | None = None  # type: ignore
    msa: Float["s n dmi"] | None = None  # type: ignore
    token_bonds: Bool["n n"] | None = None  # type: ignore
    atom_ids: Int[" m"] | None = None  # type: ignore
    atom_parent_ids: Int[" m"] | None = None  # type: ignore
    atompair_ids: Int["m m"] | Int["nw w (w*2)"] | None = None  # type: ignore
    template_mask: Bool[" t"] | None = None  # type: ignore
    msa_mask: Bool[" s"] | None = None  # type: ignore
    atom_pos: Float["m 3"] | None = None  # type: ignore
    missing_atom_mask: Bool[" m"] | None = None  # type: ignore
    molecule_atom_indices: Int[" n"] | None = None  # type: ignore
    distogram_atom_indices: Int[" n"] | None = None  # type: ignore
    atom_indices_for_frame: Int["n 3"] | None = None  # type: ignore
    resolved_labels: Int[" m"] | None = None  # type: ignore
    resolution: Float[""] | None = None  # type: ignore
    token_constraints: Float["n n dac"] | None = None  # type: ignore
    affinities: Float[" *"] | None = None  # type: ignore
    chains: Int[" 2"] | None = None  # type: ignore
    num_ligands: int | None = None
    filepath: str | None = None
    example_source: str | None = None

    def dict(self):
        """Return the dataclass as a dictionary."""
        return asdict(self)

    def model_forward_dict(self):
        """Return the dataclass as a dictionary without certain model fields."""
        return without_keys(self.dict(), ATOM_INPUT_EXCLUDE_MODEL_FIELDS)


@typecheck
@dataclass
class BatchedAtomInput:
    """Dataclass for batched atom-level inputs."""

    atom_inputs: Float["b m dai"]  # type: ignore
    molecule_ids: Int["b n"]  # type: ignore
    molecule_atom_lens: Int["b n"]  # type: ignore
    atompair_inputs: Float["b m m dapi"] | Float["b nw w (w*2) dapi"]  # type: ignore
    additional_molecule_feats: Int[f"b n {ADDITIONAL_MOLECULE_FEATS}"]  # type: ignore
    is_molecule_types: Bool[f"b n {IS_MOLECULE_TYPES}"]  # type: ignore
    molecule_atom_perms: List[List[List[int]]] | None = None  # type: ignore
    is_molecule_mod: Bool["b n num_mods"] | None = None  # type: ignore
    additional_msa_feats: Float["b s n dmf"] | None = None  # type: ignore
    additional_token_feats: Float["b n dtf"] | None = None  # type: ignore
    templates: Float["b t n n dt"] | None = None  # type: ignore
    msa: Float["b s n dmi"] | None = None  # type: ignore
    token_bonds: Bool["b n n"] | None = None  # type: ignore
    atom_ids: Int["b m"] | None = None  # type: ignore
    atom_parent_ids: Int["b m"] | None = None  # type: ignore
    atompair_ids: Int["b m m"] | Int["b nw w (w*2)"] | None = None  # type: ignore
    template_mask: Bool["b t"] | None = None  # type: ignore
    msa_mask: Bool["b s"] | None = None  # type: ignore
    atom_pos: Float["b m 3"] | None = None  # type: ignore
    missing_atom_mask: Bool["b m"] | None = None  # type: ignore
    molecule_atom_indices: Int["b n"] | None = None  # type: ignore
    distogram_atom_indices: Int["b n"] | None = None  # type: ignore
    atom_indices_for_frame: Int["b n 3"] | None = None  # type: ignore
    resolved_labels: Int["b m"] | None = None  # type: ignore
    resolution: Float[" b"] | None = None  # type: ignore
    token_constraints: Float["b n n dac"] | None = None  # type: ignore
    affinities: List[Float[" *"]] | None = None  # type: ignore
    chains: Int["b 2"] | None = None  # type: ignore
    num_ligands: List[int] | None = None
    filepath: List[str] | None = None
    example_source: List[str] | None = None

    def dict(self):
        """Return the dataclass as a dictionary."""
        return asdict(self)

    def cpu_dict(self):
        """Return the dataclass as a dictionary with all enclosed tensors residing in CPU
        memory."""
        return {k: (v.cpu() if torch.is_tensor(v) else v) for (k, v) in asdict(self).items()}

    def model_forward_dict(self):
        """Return the dataclass as a dictionary without certain model fields."""
        return without_keys(self.dict(), ATOM_INPUT_EXCLUDE_MODEL_FIELDS)


# functions for saving an AtomInput to disk or loading from disk to AtomInput


@typecheck
def atom_input_to_file(atom_input: AtomInput, path: str | Path, overwrite: bool = False) -> Path:
    """Save an AtomInput to disk."""

    if isinstance(path, str):
        path = Path(path)

    path = Path(path)

    if not overwrite:
        assert not path.exists()

    path.parents[0].mkdir(exist_ok=True, parents=True)

    torch.save(atom_input.dict(), str(path))  # nosec
    return path


@typecheck
def file_to_atom_input(path: str | Path) -> AtomInput:
    """Load an AtomInput from disk."""
    if isinstance(path, str):
        path = Path(path)

    assert path.is_file()

    atom_input_dict = torch.load(str(path), weights_only=True)  # nosec
    return AtomInput(**atom_input_dict)


@typecheck
def default_none_fields_atom_input(i: AtomInput | None) -> AtomInput | None:
    """Set default None fields in AtomInput to their default values."""
    if not_exists(i):
        # NOTE: this case occurs when pre-caching MSAs for each example
        return

    # if templates given but template mask isn't given, default to all True

    if exists(i.templates) and not_exists(i.template_mask):
        i.template_mask = torch.ones(i.templates.shape[0], dtype=torch.bool)

    # if msa given but msa mask isn't given default to all True

    if exists(i.msa) and not_exists(i.msa_mask):
        i.msa_mask = torch.ones(i.msa.shape[0], dtype=torch.bool)

    # default missing atom mask should be all False

    if not_exists(i.missing_atom_mask):
        i.missing_atom_mask = torch.zeros(i.atom_inputs.shape[0], dtype=torch.bool)

    return i


@typecheck
def pdb_dataset_to_atom_inputs(
    pdb_dataset: PDBDataset,
    *,
    output_atom_folder: str | Path | None = None,
    indices: Iterable | None = None,
    return_atom_dataset: bool = False,
    n_jobs: int = 8,
    parallel_kwargs: dict = dict(),
    overwrite_existing: bool = False,
) -> Path | AtomDataset:
    """Convert a PDBDataset to AtomInputs stored on disk."""
    if not_exists(output_atom_folder):
        pdb_folder = Path(pdb_dataset.folder).resolve()
        parent_folder = pdb_folder.parents[0]
        output_atom_folder = parent_folder / f"{pdb_folder.stem}.atom-inputs"

    if isinstance(output_atom_folder, str):
        output_atom_folder = Path(output_atom_folder)

    if not_exists(indices):
        indices = torch.randperm(len(pdb_dataset)).tolist()

    to_atom_input_fn = compose(pdb_input_to_molecule_input, molecule_to_atom_input)

    def should_process_pdb_input(index: int) -> bool:
        """Check if a PDB input should be processed."""
        atom_input_path = output_atom_folder / f"{index}.pt"
        return not atom_input_path.exists() or overwrite_existing

    @delayed
    def pdb_input_to_atom_file(index: int, path: str):
        """Convert a PDB input to an atom file."""
        pdb_input = pdb_dataset[index]

        atom_input = to_atom_input_fn(pdb_input)

        atom_input_path = path / f"{index}.pt"
        atom_input_to_file(atom_input, atom_input_path)

    Parallel(n_jobs=n_jobs, **parallel_kwargs)(
        pdb_input_to_atom_file(index, output_atom_folder)
        for index in filter(should_process_pdb_input, indices)
    )

    if not return_atom_dataset:
        return output_atom_folder

    return AtomDataset(output_atom_folder)


# Atom dataset that returns a AtomInput based on folders of atom inputs stored on disk


class AtomDataset(Dataset):
    """Dataset for AtomInput stored on disk."""

    def __init__(self, folder: str | Path, **kwargs):
        if isinstance(folder, str):
            folder = Path(folder)

        assert folder.exists() and folder.is_dir(), f"Atom dataset not found at {str(folder)}"

        self.folder = folder
        self.files = [*folder.glob("**/*.pt")]

        assert len(self) > 0, f"No valid atom `.pt` files found at {str(folder)}"

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.files)

    def __getitem__(self, idx: int) -> AtomInput:
        """Return an item from the dataset."""
        file = self.files[idx]
        return file_to_atom_input(file)


# functions for extracting atom and atompair features (atom_inputs, atompair_inputs)

# atom reference position to atompair inputs
# will be used in the `default_extract_atompair_feats_fn` below in MoleculeInput


@typecheck
def atom_ref_pos_to_atompair_inputs(
    atom_ref_pos: Float["m 3"],  # type: ignore
    atom_ref_space_uid: Int[" m"] | None = None,  # type: ignore
) -> Float["m m 5"]:  # type: ignore
    """Compute atompair inputs from atom reference positions.

    :param atom_ref_pos: The reference positions of the atoms.
    :param atom_ref_space_uid: The reference space UID of the atoms.
    :return: The atompair inputs.
    """
    # Algorithm 5 - lines 2-6

    # line 2

    # pairwise_rel_pos = einx.subtract("i c, j c -> i j c", atom_ref_pos, atom_ref_pos)

    pairwise_rel_pos = atom_ref_pos.unsqueeze(1) - atom_ref_pos.unsqueeze(0)

    # line 5 - pairwise inverse squared distance

    atom_inv_square_dist = (1 + pairwise_rel_pos.norm(dim=-1, p=2) ** 2) ** -1

    # line 3

    if exists(atom_ref_space_uid):
        # same_ref_space_mask = einx.equal("i, j -> i j", atom_ref_space_uid, atom_ref_space_uid)
        same_ref_space_mask = atom_ref_space_uid.unsqueeze(1) == atom_ref_space_uid.unsqueeze(0)
    else:
        same_ref_space_mask = torch.ones_like(atom_inv_square_dist).bool()

    # concat all into atompair_inputs for projection into atompair_feats within MegaFold

    atompair_inputs, _ = pack(
        (
            pairwise_rel_pos,
            atom_inv_square_dist,
            same_ref_space_mask.float(),
        ),
        "i j *",
    )

    # mask out

    # atompair_inputs = einx.where(
    #     "i j, i j dapi, -> i j dapi", same_ref_space_mask, atompair_inputs, 0.0
    # )

    atompair_inputs = atompair_inputs * same_ref_space_mask.unsqueeze(-1).float()

    # return

    return atompair_inputs


def default_extract_atom_feats_fn(atom: Atom):
    """Extract atom features from an RDKit atom."""
    return tensor([atom.GetFormalCharge(), atom.GetImplicitValence(), atom.GetExplicitValence()])


def default_extract_atompair_feats_fn(mol: Mol):
    """Extract atompair features from an RDKit molecule."""
    all_atom_pos = []

    for idx in range(mol.GetNumAtoms()):
        pos = mol.GetConformer().GetAtomPosition(idx)
        all_atom_pos.append([pos.x, pos.y, pos.z])

    all_atom_pos_tensor = tensor(all_atom_pos)

    return atom_ref_pos_to_atompair_inputs(
        all_atom_pos_tensor
    )  # what they did in the paper, but can be overwritten


# molecule input - accepting list of molecules as rdchem.Mol + the atomic lengths for how to pool into tokens
# `n` here is the token length, which accounts for molecules that are one token per atom


@typecheck
@dataclass
class MoleculeInput:
    """Dataclass for molecule-level inputs."""

    molecules: List[Mol]
    molecule_token_pool_lens: List[int]
    molecule_ids: Int[" n"]  # type: ignore
    additional_molecule_feats: Int[f"n {ADDITIONAL_MOLECULE_FEATS}"]  # type: ignore
    is_molecule_types: Bool[f"n {IS_MOLECULE_TYPES}"]  # type: ignore
    src_tgt_atom_indices: Int["n 2"]  # type: ignore
    token_bonds: Bool["n n"]  # type: ignore
    molecule_atom_perms: List[List[int]] | None = None  # type: ignore
    is_molecule_mod: Bool["n num_mods"] | Bool[" n"] | None = None  # type: ignore
    molecule_atom_indices: List[int | None] | None = None  # type: ignore
    distogram_atom_indices: List[int | None] | None = None  # type: ignore
    atom_indices_for_frame: Int["n 3"] | None = None  # type: ignore
    missing_atom_indices: List[Int[" _"] | None] | None = None  # type: ignore
    missing_token_indices: List[Int[" _"] | None] | None = None  # type: ignore
    atom_parent_ids: Int[" m"] | None = None  # type: ignore
    additional_msa_feats: Float["s n dmf"] | None = None  # type: ignore
    additional_token_feats: Float["n dtf"] | None = None  # type: ignore
    templates: Float["t n n dt"] | None = None  # type: ignore
    msa: Float["s n dmi"] | None = None  # type: ignore
    atom_pos: List[Float["_ 3"]] | Float["m 3"] | None = None  # type: ignore
    template_mask: Bool[" t"] | None = None  # type: ignore
    msa_mask: Bool[" s"] | None = None  # type: ignore
    resolved_labels: Int[" m"] | None = None  # type: ignore
    resolution: Float[""] | None = None  # type: ignore
    token_constraints: Float["n n dac"] | None = None  # type: ignore
    first_mol_token_indices: Int[" n"] | None = None  # type: ignore
    chains: Tuple[int | None, int | None] | None = (None, None)
    num_ligands: int | None = None
    filepath: str | None = None
    example_source: str | None = None
    add_atom_ids: bool = True
    add_atompair_ids: bool = True
    directed_bonds: bool = True
    extract_atom_feats_fn: Callable[[Atom], Float["m dai"]] = default_extract_atom_feats_fn  # type: ignore
    extract_atompair_feats_fn: Callable[[Mol], Float["m m dapi"]] = default_extract_atompair_feats_fn  # type: ignore
    custom_atoms: List[str] | None = None
    custom_bonds: List[str] | None = None


@typecheck
@wrapt_timeout_decorator.timeout(
    MOLECULE_TO_ATOM_INPUT_MAX_SECONDS_PER_INPUT,
    use_signals=True,
    # NOTE: we rely on reporting a `BaseException` to avoid catching
    # `TimeoutError` exceptions within other clauses of this function.
    timeout_exception=BaseException,
)
def molecule_to_atom_input(
    mol_input: MoleculeInput | None, verbose: bool = False
) -> AtomInput | None:
    """Convert a MoleculeInput to an AtomInput.

    NOTE: This function assumes that `distogram_atom_indices`,
    `molecule_atom_indices`, and `atom_indices_for_frame` are already
    offset as structure-global (and not molecule-local) atom indices.
    In contrast, `missing_atom_indices` and `missing_token_indices`
    are expected to be molecule-local atom indices.
    """
    i = mol_input

    if not_exists(i):
        # NOTE: this case occurs when pre-caching MSAs for each example
        return

    molecules = i.molecules
    molecule_ids = i.molecule_ids
    atom_lens = i.molecule_token_pool_lens
    extract_atom_feats_fn = i.extract_atom_feats_fn
    extract_atompair_feats_fn = i.extract_atompair_feats_fn

    # validate total number of atoms

    mol_total_atoms = sum([mol.GetNumAtoms() for mol in molecules])
    assert mol_total_atoms == sum(
        atom_lens
    ), f"Total atoms summed up from molecules passed in on `molecules` ({mol_total_atoms}) does not equal the number of atoms summed up in the field `molecule_token_pool_lens` {sum(atom_lens)}"

    atom_lens = tensor(atom_lens)
    total_atoms = atom_lens.sum().item()

    # molecule_atom_lens

    atoms: List[int] = []

    for mol in molecules:
        atoms.extend([*mol.GetAtoms()])

    # handle maybe atom embeds

    atom_ids = None

    if i.add_atom_ids:
        atom_list = default(i.custom_atoms, ATOMS)

        atom_index = {symbol: i for i, symbol in enumerate(atom_list)}

        atom_ids = []

        for atom in atoms:
            atom_symbol = atom.GetSymbol()
            if atom_symbol not in atom_index:
                if verbose:
                    logger.warning(
                        f"{atom_symbol} not found in the given list of possible atoms. Substituting with the last element type ({atom_list[-1]})."
                    )
                atom_symbol = atom_list[-1]

            atom_ids.append(atom_index[atom_symbol])

        atom_ids = tensor(atom_ids, dtype=torch.long)

    # get List[int] of number of atoms per molecule
    # for the offsets when building the atompair feature map / bonds

    all_num_atoms = tensor([mol.GetNumAtoms() for mol in molecules])
    offsets = exclusive_cumsum(all_num_atoms)

    # handle maybe missing atom indices

    missing_atom_mask = None
    missing_atom_indices = None
    missing_token_indices = None

    if exists(i.missing_atom_indices) and len(i.missing_atom_indices) > 0:
        assert len(molecules) == len(
            i.missing_atom_indices
        ), f"{len(i.missing_atom_indices)} missing atom indices does not match the number of molecules given ({len(molecules)})"

        missing_atom_indices: List[Int[" _"]] = [  # type: ignore
            default(indices, torch.empty((0,), dtype=torch.long))
            for indices in i.missing_atom_indices
        ]
        missing_token_indices: List[Int[" _"]] = [  # type: ignore
            default(indices, torch.empty((0,), dtype=torch.long))
            for indices in i.missing_token_indices
        ]

        missing_atom_mask: List[Bool[" _"]] = []  # type: ignore

        for num_atoms, mol_missing_atom_indices, mol_missing_token_indices, offset in zip(
            all_num_atoms, missing_atom_indices, missing_token_indices, offsets
        ):
            mol_miss_atom_mask = torch.zeros(num_atoms, dtype=torch.bool)

            if mol_missing_atom_indices.numel() > 0:
                mol_miss_atom_mask.scatter_(-1, mol_missing_atom_indices, True)
            if mol_missing_token_indices.numel() > 0:
                mol_missing_token_indices += offset

            missing_atom_mask.append(mol_miss_atom_mask)

        missing_atom_mask = torch.cat(missing_atom_mask)
        missing_token_indices = pad_sequence(
            # NOTE: padding value must be any negative integer besides -1,
            # to not erroneously detect "missing" token center/distogram atoms
            # within ligands
            missing_token_indices,
            batch_first=True,
            padding_value=-2,
        )

    # handle is_molecule_mod being one dimensional

    is_molecule_mod = i.is_molecule_mod

    if exists(is_molecule_mod) and is_molecule_mod.ndim == 1:
        is_molecule_mod = rearrange(is_molecule_mod, "n -> n 1")

    # handle maybe atompair embeds

    atompair_ids = None

    if i.add_atompair_ids:
        atompair_ids = torch.zeros(total_atoms, total_atoms).long()

        # need the asym_id (to keep track of each molecule for each chain ascending) as well as `is_protein | is_dna | is_rna | is_ligand | is_metal_ion` for is_molecule_types (chainable biomolecules)
        # will do a single bond from a peptide or nucleotide to the one before. derive a `is_first_mol_in_chain` from `asym_ids`

        asym_ids = i.additional_molecule_feats[..., 2]
        asym_ids = F.pad(asym_ids, (1, 0), value=-1)
        is_first_mol_in_chains = (asym_ids[1:] - asym_ids[:-1]) != 0

        is_chainable_biomolecules = i.is_molecule_types[..., IS_BIOMOLECULE_INDICES].any(dim=-1)

        # for every molecule, build the bonds id matrix and add to `atompair_ids`

        prev_mol = None
        prev_src_tgt_atom_indices = None

        atom_bonds = default(i.custom_bonds, ATOM_BONDS)

        src_tgt_atom_indices = tensor(i.src_tgt_atom_indices)

        # subset features to the first token of each molecule

        first_mol_token_indices = i.first_mol_token_indices

        num_tokens = len(i.token_bonds)

        assert exists(
            first_mol_token_indices
        ), "The `first_mol_token_indices` field is required for building atompair IDs."
        assert len(first_mol_token_indices) == len(
            molecules
        ), "The `first_mol_token_indices` field must have the same length as the number of molecules."
        assert len(molecule_ids[first_mol_token_indices]) == len(
            molecules
        ), "The `first_mol_token_indices` field's molecule ID mappings must have the same length as the number of molecules."

        for (
            mol,
            mol_id,
            is_first_mol_in_chain,
            is_chainable_biomolecule,
            is_mol_mod,
            src_tgt_atom_indices,
            offset,
        ) in zip(
            molecules,
            molecule_ids[first_mol_token_indices],
            is_first_mol_in_chains[first_mol_token_indices],
            is_chainable_biomolecules[first_mol_token_indices],
            default(
                is_molecule_mod,
                torch.zeros((num_tokens, DEFAULT_NUM_MOLECULE_MODS), dtype=torch.bool),
            )[first_mol_token_indices],
            src_tgt_atom_indices[first_mol_token_indices],
            offsets,
        ):
            maybe_cached_get_atompair_ids = maybe_cache(
                get_atompair_ids,
                cache=ATOMPAIR_IDS_CACHE,
                key=f"{mol_id.item()}:{i.directed_bonds}",
                should_cache=is_chainable_biomolecule.item() and not is_mol_mod.any().item(),
            )

            mol_atompair_ids = maybe_cached_get_atompair_ids(
                mol, atom_bonds, directed_bonds=i.directed_bonds
            )

            # /einx.set_at

            if exists(mol_atompair_ids) and mol_atompair_ids.numel() > 0:
                num_atoms = mol.GetNumAtoms()
                row_col_slice = slice(offset, offset + num_atoms)
                atompair_ids[row_col_slice, row_col_slice] = mol_atompair_ids

            # if is chainable biomolecule
            # and not the first biomolecule in the chain, add a single covalent bond between first atom of incoming biomolecule and the last atom of the last biomolecule

            # NOTE: modified polymer residues have bonds featurized here to encourage the model to place them sequentially within a chain

            if is_chainable_biomolecule and not is_first_mol_in_chain:
                _, last_atom_index = prev_src_tgt_atom_indices
                first_atom_index, _ = src_tgt_atom_indices

                last_atom_index_from_end = prev_mol.GetNumAtoms() - last_atom_index

                src_atom_offset = (offset - last_atom_index_from_end).item()
                tgt_atom_offset = (offset + first_atom_index).item()

                atompair_ids[src_atom_offset, tgt_atom_offset] = 1
                atompair_ids[tgt_atom_offset, src_atom_offset] = 1

            prev_mol = mol
            prev_src_tgt_atom_indices = src_tgt_atom_indices

    # atom_inputs

    atom_inputs: List[Float["m dai"]] = []  # type: ignore

    for mol in molecules:
        atom_feats = []

        for atom in mol.GetAtoms():
            atom_feats.append(extract_atom_feats_fn(atom))

        atom_inputs.append(torch.stack(atom_feats, dim=0))

    atom_inputs_tensor = torch.cat(atom_inputs).float()

    # atompair_inputs

    atompair_feats: List[Float["m m dapi"]] = []  # type: ignore

    for mol in molecules:
        atompair_feats.append(extract_atompair_feats_fn(mol))

    assert len(atompair_feats) > 0

    dim_atompair_inputs = first(atompair_feats).shape[-1]

    atompair_inputs = torch.zeros((total_atoms, total_atoms, dim_atompair_inputs))

    for atompair_feat, num_atoms, offset in zip(atompair_feats, all_num_atoms, offsets):
        row_col_slice = slice(offset, offset + num_atoms)
        atompair_inputs[row_col_slice, row_col_slice] = atompair_feat

    # mask out molecule atom indices and distogram atom indices where it is in the missing atom indices list

    molecule_atom_indices = i.molecule_atom_indices
    distogram_atom_indices = i.distogram_atom_indices
    atom_indices_for_frame = i.atom_indices_for_frame

    if exists(missing_token_indices) and missing_token_indices.shape[-1]:
        # is_missing_molecule_atom = einx.equal(
        #     "n missing, n -> n missing", missing_token_indices, molecule_atom_indices
        # ).any(dim=-1)
        # is_missing_distogram_atom = einx.equal(
        #     "n missing, n -> n missing", missing_token_indices, distogram_atom_indices
        # ).any(dim=-1)
        # is_missing_atom_indices_for_frame = einx.equal(
        #     "n missing, n c -> n c missing", missing_token_indices, atom_indices_for_frame
        # ).any(dim=(-1, -2))
        is_missing_molecule_atom = torch.any(
            missing_token_indices == molecule_atom_indices[..., None], dim=-1
        )
        is_missing_distogram_atom = torch.any(
            missing_token_indices == distogram_atom_indices[..., None], dim=-1
        )
        is_missing_atom_indices_for_frame = torch.any(
            missing_token_indices[..., None, :] == atom_indices_for_frame[..., None], dim=(-1, -2)
        )

        molecule_atom_indices = molecule_atom_indices.masked_fill(is_missing_molecule_atom, -1)
        distogram_atom_indices = distogram_atom_indices.masked_fill(is_missing_distogram_atom, -1)
        atom_indices_for_frame = atom_indices_for_frame.masked_fill(
            is_missing_atom_indices_for_frame[..., None], -1
        )

    # sanity-check the atom indices
    if not (-1 <= molecule_atom_indices.min() <= molecule_atom_indices.max() < total_atoms):
        raise ValueError(
            f"Invalid molecule atom indices found in `molecule_to_atom_input()` for {i.filepath}: {molecule_atom_indices}"
        )
    if not (-1 <= distogram_atom_indices.min() <= distogram_atom_indices.max() < total_atoms):
        raise ValueError(
            f"Invalid distogram atom indices found in `molecule_to_atom_input()` for {i.filepath}: {distogram_atom_indices}"
        )
    if not (-1 <= atom_indices_for_frame.min() <= atom_indices_for_frame.max() < total_atoms):
        raise ValueError(
            f"Invalid atom indices for frame found in `molecule_to_atom_input()` for {i.filepath}: {atom_indices_for_frame}"
        )

    # handle atom positions

    atom_pos = i.atom_pos

    if exists(atom_pos) and isinstance(atom_pos, list):
        atom_pos = torch.cat(atom_pos, dim=-2)

    # coerce chain indices into a tensor

    chains = tensor([default(chain, -1) for chain in i.chains]).long()

    # atom input

    atom_input = AtomInput(
        atom_inputs=atom_inputs_tensor,
        atompair_inputs=atompair_inputs,
        molecule_atom_lens=atom_lens.long(),
        molecule_ids=i.molecule_ids,
        molecule_atom_indices=molecule_atom_indices,
        distogram_atom_indices=distogram_atom_indices,
        atom_indices_for_frame=atom_indices_for_frame,
        molecule_atom_perms=i.molecule_atom_perms,
        is_molecule_mod=is_molecule_mod,
        msa=i.msa,
        templates=i.templates,
        msa_mask=i.msa_mask,
        template_mask=i.template_mask,
        missing_atom_mask=missing_atom_mask,
        additional_msa_feats=i.additional_msa_feats,
        additional_token_feats=i.additional_token_feats,
        additional_molecule_feats=i.additional_molecule_feats,
        is_molecule_types=i.is_molecule_types,
        atom_pos=atom_pos,
        token_bonds=i.token_bonds,
        atom_parent_ids=i.atom_parent_ids,
        atom_ids=atom_ids,
        atompair_ids=atompair_ids,
        resolved_labels=i.resolved_labels,
        resolution=i.resolution,
        token_constraints=i.token_constraints,
        chains=chains,
        num_ligands=i.num_ligands,
        filepath=i.filepath,
        example_source=i.example_source,
    )

    return atom_input


# molecule lengthed molecule input
# molecule input - accepting list of molecules as rdchem.Mol

# `n` here refers to the actual number of molecules, NOT the `n` used within MegaFold
# the proper token length needs to be correctly computed in the corresponding function for MoleculeLengthMoleculeInput -> AtomInput


@typecheck
@dataclass
class MoleculeLengthMoleculeInput:
    """Dataclass for molecule-level inputs with molecule lengths."""

    molecules: List[Mol]
    molecule_ids: Int[" n"]  # type: ignore
    additional_molecule_feats: Int[f"n {ADDITIONAL_MOLECULE_FEATS-1}"]  # type: ignore
    is_molecule_types: Bool[f"n {IS_MOLECULE_TYPES}"]  # type: ignore
    src_tgt_atom_indices: Int["n 2"]  # type: ignore
    token_bonds: Bool["n n"] | None = None  # type: ignore
    one_token_per_atom: List[bool] | None = None
    is_molecule_mod: Bool["n num_mods"] | Bool[" n"] | None = None  # type: ignore
    molecule_atom_indices: List[int | None] | None = None
    distogram_atom_indices: List[int | None] | None = None
    atom_indices_for_frame: List[Tuple[int, int, int] | None] | None = None
    missing_atom_indices: List[Int[" _"] | None] | None = None  # type: ignore
    missing_token_indices: List[Int[" _"] | None] | None = None  # type: ignore
    atom_parent_ids: Int[" m"] | None = None  # type: ignore
    additional_msa_feats: Float["s n dmf"] | None = None  # type: ignore
    additional_token_feats: Float["n dtf"] | None = None  # type: ignore
    templates: Float["t n n dt"] | None = None  # type: ignore
    msa: Float["s n dmi"] | None = None  # type: ignore
    atom_pos: List[Float["_ 3"]] | Float["m 3"] | None = None  # type: ignore
    template_mask: Bool[" t"] | None = None  # type: ignore
    msa_mask: Bool[" s"] | None = None  # type: ignore
    resolved_labels: Int[" m"] | None = None  # type: ignore
    token_constraints: Float["n n dac"] | None = None  # type: ignore
    chains: Tuple[int | None, int | None] | None = (None, None)
    filepath: str | None = None
    example_source: str | None = None
    add_atom_ids: bool = True
    add_atompair_ids: bool = True
    directed_bonds: bool = True
    extract_atom_feats_fn: Callable[[Atom], Float["m dai"]] = default_extract_atom_feats_fn  # type: ignore
    extract_atompair_feats_fn: Callable[[Mol], Float["m m dapi"]] = default_extract_atompair_feats_fn  # type: ignore
    custom_atoms: List[str] | None = None
    custom_bonds: List[str] | None = None


@typecheck
def molecule_lengthed_molecule_input_to_atom_input(
    mol_input: MoleculeLengthMoleculeInput, verbose: bool = False
) -> AtomInput:
    """Convert a MoleculeLengthMoleculeInput to an AtomInput."""
    i = mol_input

    molecules = i.molecules
    extract_atom_feats_fn = i.extract_atom_feats_fn
    extract_atompair_feats_fn = i.extract_atompair_feats_fn

    # derive `atom_lens` based on `one_token_per_atom`, for ligands and modified biomolecules

    atoms_per_molecule = tensor([mol.GetNumAtoms() for mol in molecules])
    ones = torch.ones_like(atoms_per_molecule)

    # `is_molecule_mod` can either be
    # 1. Bool['n'], in which case it will only be used for determining `one_token_per_atom`, or
    # 2. Bool['n num_mods'], where it will be passed to MegaFold for molecule modification embeds

    is_molecule_mod = i.is_molecule_mod
    is_molecule_any_mod = False

    if exists(is_molecule_mod):
        if i.is_molecule_mod.ndim == 2:
            is_molecule_any_mod = is_molecule_mod.any(dim=-1)
        else:
            is_molecule_any_mod = is_molecule_mod
            is_molecule_mod = None

    # get `one_token_per_atom`, which can be fully customizable

    if exists(i.one_token_per_atom):
        one_token_per_atom = tensor(i.one_token_per_atom)
    else:
        # if which molecule is `one_token_per_atom` is not passed in
        # default to what the paper did, which is ligands and any modified biomolecule
        is_ligand = i.is_molecule_types[..., IS_LIGAND_INDEX]
        one_token_per_atom = is_ligand | is_molecule_any_mod

    assert len(molecules) == len(one_token_per_atom)

    # derive the number of repeats needed to expand molecule lengths to token lengths

    token_repeats = torch.where(one_token_per_atom, atoms_per_molecule, ones)

    # derive atoms per token

    atom_repeat_input = torch.where(one_token_per_atom, ones, atoms_per_molecule)
    atoms_per_token = repeat_interleave(atom_repeat_input, token_repeats)

    total_atoms = atoms_per_molecule.sum().item()

    # derive `is_first_mol_in_chains` and `is_chainable_biomolecules` - needed for constructing `token_bonds

    # need the asym_id (to keep track of each molecule for each chain ascending) as well as `is_protein | is_dna | is_rna | is_ligand | is_metal_ion` for is_molecule_types (chainable biomolecules)
    # will do a single bond from a peptide or nucleotide to the one before. derive a `is_first_mol_in_chain` from `asym_ids`

    asym_ids = i.additional_molecule_feats[..., 2]
    asym_ids = F.pad(asym_ids, (1, 0), value=-1)
    is_first_mol_in_chains = (asym_ids[1:] - asym_ids[:-1]) != 0
    is_chainable_biomolecules = i.is_molecule_types[..., IS_BIOMOLECULE_INDICES].any(dim=-1)

    # repeat all the molecule lengths to the token lengths, using `one_token_per_atom`

    src_tgt_atom_indices = repeat_interleave(i.src_tgt_atom_indices, token_repeats, dim=0)
    is_molecule_types = repeat_interleave(i.is_molecule_types, token_repeats, dim=0)

    additional_molecule_feats = repeat_interleave(
        i.additional_molecule_feats, token_repeats, dim=0
    )

    # insert the 2nd entry into additional molecule feats, which is just an arange over the number of tokens

    additional_molecule_feats, _ = pack(
        (
            additional_molecule_feats[..., :1],
            torch.arange(additional_molecule_feats.shape[0]),
            additional_molecule_feats[..., 1:],
        ),
        "n *",
    )

    additional_msa_feats = repeat_interleave(i.additional_msa_feats, token_repeats, dim=1)

    additional_token_feats = repeat_interleave(i.additional_token_feats, token_repeats, dim=0)
    molecule_ids = repeat_interleave(i.molecule_ids, token_repeats)

    atom_indices_offsets = repeat_interleave(
        exclusive_cumsum(atoms_per_molecule), token_repeats, dim=0
    )

    msa = maybe(repeat_interleave)(i.msa, token_repeats, dim=-2)
    is_molecule_mod = maybe(repeat_interleave)(i.is_molecule_mod, token_repeats, dim=0)

    templates = maybe(repeat_interleave)(i.templates, token_repeats, dim=-3)
    templates = maybe(repeat_interleave)(templates, token_repeats, dim=-2)

    # impute distogram and molecule atom indices for ligands and modified polymer residues

    distogram_atom_indices = torch.cat(
        [
            (torch.arange(tr.item()) if dai.item() == -1 else repeat_interleave(dai, tr))
            for dai, tr in zip(i.distogram_atom_indices, token_repeats)
        ]
    )
    molecule_atom_indices = torch.cat(
        [
            (torch.arange(tr.item()) if mai.item() == -1 else repeat_interleave(mai, tr))
            for mai, tr in zip(i.molecule_atom_indices, token_repeats)
        ]
    )

    # get all atoms

    atoms: List[Atom] = []

    for mol in molecules:
        atoms.extend([*mol.GetAtoms()])

    # construct token bonds, which will be linearly connected for proteins
    # and nucleic acids, but for ligands and modified polymer residues
    # will have their atomic bond matrix (as ligands and modified polymer
    # residues are atom resolution)

    num_tokens = token_repeats.sum().item()

    token_bonds = torch.zeros(num_tokens, num_tokens).bool()

    offset = 0

    for (
        mol,
        mol_is_chainable_biomolecule,
        mol_is_mod,
        mol_is_first_mol_in_chain,
        mol_is_one_token_per_atom,
    ) in zip(
        molecules,
        is_chainable_biomolecules,
        default(
            i.is_molecule_mod,
            torch.zeros((len(molecules), DEFAULT_NUM_MOLECULE_MODS), dtype=torch.bool),
        ),
        is_first_mol_in_chains,
        one_token_per_atom,
    ):
        num_atoms = mol.GetNumAtoms()

        if mol_is_chainable_biomolecule and not mol_is_mod.any() and not mol_is_first_mol_in_chain:
            token_bonds[offset, offset - 1] = True
            token_bonds[offset - 1, offset] = True

        elif mol_is_one_token_per_atom:
            has_bond = get_mol_has_bond(mol)

            if exists(has_bond) and has_bond.numel() > 0:
                num_atoms = mol.GetNumAtoms()
                row_col_slice = slice(offset, offset + num_atoms)
                token_bonds[row_col_slice, row_col_slice] = has_bond

        offset += num_atoms if mol_is_one_token_per_atom else 1

    # handle maybe atom embeds

    atom_ids = None

    if i.add_atom_ids:
        atom_list = default(i.custom_atoms, ATOMS)

        atom_index = {symbol: i for i, symbol in enumerate(atom_list)}

        atom_ids = []

        for atom in atoms:
            atom_symbol = atom.GetSymbol()
            if atom_symbol not in atom_index:
                if verbose:
                    logger.warning(
                        f"{atom_symbol} not found in the given list of possible atoms. Substituting with the last element type ({atom_list[-1]})."
                    )
                atom_symbol = atom_list[-1]

            atom_ids.append(atom_index[atom_symbol])

        atom_ids = tensor(atom_ids, dtype=torch.long)

    # get List[int] of number of atoms per molecule
    # for the offsets when building the atompair feature map / bonds

    all_num_atoms = tensor([mol.GetNumAtoms() for mol in molecules])
    offsets = exclusive_cumsum(all_num_atoms)

    # handle maybe missing atom indices

    missing_atom_mask = None
    missing_atom_indices = None
    missing_token_indices = None

    if exists(i.missing_atom_indices) and len(i.missing_atom_indices) > 0:
        assert len(molecules) == len(
            i.missing_atom_indices
        ), f"{len(i.missing_atom_indices)} missing atom indices does not match the number of molecules given ({len(molecules)})"

        missing_atom_indices: List[Int[" _"]] = [  # type: ignore
            default(indices, torch.empty((0,), dtype=torch.long))
            for indices in i.missing_atom_indices
        ]
        missing_token_indices: List[Int[" _"]] = [  # type: ignore
            default(indices, torch.empty((0,), dtype=torch.long))
            for indices in i.missing_token_indices
        ]

        missing_atom_mask: List[Bool[" _"]] = []  # type: ignore

        for num_atoms, mol_missing_atom_indices in zip(all_num_atoms, missing_atom_indices):
            mol_miss_atom_mask = torch.zeros(num_atoms, dtype=torch.bool)

            if mol_missing_atom_indices.numel() > 0:
                mol_miss_atom_mask.scatter_(-1, mol_missing_atom_indices, True)

            missing_atom_mask.append(mol_miss_atom_mask)

        missing_atom_mask = torch.cat(missing_atom_mask)
        missing_token_indices = pad_sequence(
            # NOTE: padding value must be any negative integer besides -1,
            # to not erroneously detect "missing" token center/distogram atoms
            # within ligands
            missing_token_indices,
            batch_first=True,
            padding_value=-2,
        )

    # handle `atom_indices_for_frame` for the PAE

    atom_indices_for_frame = i.atom_indices_for_frame

    if exists(atom_indices_for_frame):
        atom_indices_for_frame = [
            default(indices, (-1, -1, -1)) for indices in i.atom_indices_for_frame
        ]
        atom_indices_for_frame = tensor(atom_indices_for_frame)

    atom_indices_for_frame = repeat_interleave(atom_indices_for_frame, token_repeats, dim=0)

    # handle maybe atompair embeds

    atompair_ids = None

    if i.add_atompair_ids:
        atompair_ids = torch.zeros(total_atoms, total_atoms).long()

        # for every molecule, build the bonds id matrix and add to `atompair_ids`

        prev_mol = None
        prev_src_tgt_atom_indices = None

        atom_bonds = default(i.custom_bonds, ATOM_BONDS)

        for (
            mol,
            mol_id,
            is_first_mol_in_chain,
            is_chainable_biomolecule,
            is_mol_mod,
            src_tgt_atom_indices,
            offset,
        ) in zip(
            molecules,
            i.molecule_ids,
            is_first_mol_in_chains,
            is_chainable_biomolecules,
            default(
                i.is_molecule_mod,
                torch.zeros((len(molecules), DEFAULT_NUM_MOLECULE_MODS), dtype=torch.bool),
            ),
            i.src_tgt_atom_indices,
            offsets,
        ):
            maybe_cached_get_atompair_ids = maybe_cache(
                get_atompair_ids,
                cache=ATOMPAIR_IDS_CACHE,
                key=f"{mol_id.item()}:{i.directed_bonds}",
                should_cache=is_chainable_biomolecule.item() and not is_mol_mod.any().item(),
            )

            mol_atompair_ids = maybe_cached_get_atompair_ids(
                mol, atom_bonds, directed_bonds=i.directed_bonds
            )

            # mol_atompair_ids = einx.set_at("[h w], c [2], c -> [h w]", mol_atompair_ids, coordinates, updates)

            if exists(mol_atompair_ids) and mol_atompair_ids.numel() > 0:
                num_atoms = mol.GetNumAtoms()
                row_col_slice = slice(offset, offset + num_atoms)
                atompair_ids[row_col_slice, row_col_slice] = mol_atompair_ids

            # if is chainable biomolecule
            # and not the first biomolecule in the chain, add a single covalent bond between first atom of incoming biomolecule and the last atom of the last biomolecule

            # NOTE: modified polymer residues have bonds featurized here to encourage the model to place them sequentially within a chain

            if is_chainable_biomolecule and not is_first_mol_in_chain:
                _, last_atom_index = prev_src_tgt_atom_indices
                first_atom_index, _ = src_tgt_atom_indices

                last_atom_index_from_end = prev_mol.GetNumAtoms() - last_atom_index

                src_atom_offset = offset - last_atom_index_from_end
                tgt_atom_offset = offset + first_atom_index

                atompair_ids[src_atom_offset, tgt_atom_offset] = 1
                atompair_ids[tgt_atom_offset, src_atom_offset] = 1

            prev_mol = mol
            prev_src_tgt_atom_indices = src_tgt_atom_indices

    # atom_inputs

    atom_inputs: List[Float["m dai"]] = []  # type: ignore

    for mol in molecules:
        atom_feats = []

        for atom in mol.GetAtoms():
            atom_feats.append(extract_atom_feats_fn(atom))

        atom_inputs.append(torch.stack(atom_feats, dim=0))

    atom_inputs_tensor = torch.cat(atom_inputs).float()

    # atompair_inputs

    atompair_feats: List[Float["m m dapi"]] = []  # type: ignore

    for mol, offset in zip(molecules, offsets):
        atompair_feats.append(extract_atompair_feats_fn(mol))

    assert len(atompair_feats) > 0

    dim_atompair_inputs = first(atompair_feats).shape[-1]

    atompair_inputs = torch.zeros((total_atoms, total_atoms, dim_atompair_inputs))

    for atompair_feat, num_atoms, offset in zip(atompair_feats, all_num_atoms, offsets):
        row_col_slice = slice(offset, offset + num_atoms)
        atompair_inputs[row_col_slice, row_col_slice] = atompair_feat

    # mask out molecule atom indices and distogram atom indices where it is in the missing atom indices list

    if exists(missing_token_indices) and missing_token_indices.shape[-1]:
        missing_token_indices = repeat_interleave(missing_token_indices, token_repeats, dim=0)

        # is_missing_molecule_atom = einx.equal(
        #     "n missing, n -> n missing", missing_token_indices, molecule_atom_indices
        # ).any(dim=-1)

        # is_missing_distogram_atom = einx.equal(
        #     "n missing, n -> n missing", missing_token_indices, distogram_atom_indices
        # ).any(dim=-1)

        # is_missing_atom_indices_for_frame = einx.equal(
        #     "n missing, n c -> n c missing", missing_token_indices, atom_indices_for_frame
        # ).any(dim=(-1, -2))
        is_missing_molecule_atom = torch.any(
            missing_token_indices == molecule_atom_indices[..., None], dim=-1
        )
        is_missing_distogram_atom = torch.any(
            missing_token_indices == distogram_atom_indices[..., None], dim=-1
        )
        is_missing_atom_indices_for_frame = torch.any(
            missing_token_indices[..., None, :] == atom_indices_for_frame[..., None], dim=(-1, -2)
        )

        molecule_atom_indices = molecule_atom_indices.masked_fill(is_missing_molecule_atom, -1)
        distogram_atom_indices = distogram_atom_indices.masked_fill(is_missing_distogram_atom, -1)

        atom_indices_for_frame = atom_indices_for_frame.masked_fill(
            is_missing_atom_indices_for_frame[..., None], -1
        )

    # offsets for all indices

    distogram_atom_indices = offset_only_positive(distogram_atom_indices, atom_indices_offsets)
    molecule_atom_indices = offset_only_positive(molecule_atom_indices, atom_indices_offsets)
    atom_indices_for_frame = offset_only_positive(
        atom_indices_for_frame, atom_indices_offsets[..., None]
    )

    # just use a hack to remove any duplicated indices (ligands and modified biomolecules) in a row

    atom_indices_for_frame = remove_consecutive_duplicate(atom_indices_for_frame)

    # handle atom positions

    atom_pos = i.atom_pos

    if exists(atom_pos) and isinstance(atom_pos, list):
        atom_pos = torch.cat(atom_pos, dim=-2)

    # coerce chain indices into a tensor

    chains = tensor([default(chain, -1) for chain in i.chains]).long()

    # determine the number of ligands present in the structure

    res_idx, _, chain_idx, _, _ = additional_molecule_feats.unbind(dim=-1)
    chain_residue_ids = create_uid_tensor(chain_idx.unsqueeze(0), res_idx.unsqueeze(0)).squeeze(0)

    ligand_chain_residue_ids = chain_residue_ids[is_molecule_types[..., IS_LIGAND]]
    num_ligands = len(ligand_chain_residue_ids.unique())

    # atom input

    atom_input = AtomInput(
        atom_inputs=atom_inputs_tensor,
        atompair_inputs=atompair_inputs,
        molecule_atom_lens=atoms_per_token,
        molecule_ids=molecule_ids,
        molecule_atom_indices=molecule_atom_indices,
        distogram_atom_indices=distogram_atom_indices,
        atom_indices_for_frame=atom_indices_for_frame,
        missing_atom_mask=missing_atom_mask,
        additional_msa_feats=additional_msa_feats,
        additional_token_feats=additional_token_feats,
        additional_molecule_feats=additional_molecule_feats,
        is_molecule_mod=is_molecule_mod,
        is_molecule_types=is_molecule_types,
        msa=msa,
        msa_mask=i.msa_mask,
        templates=templates,
        template_mask=i.template_mask,
        atom_pos=atom_pos,
        token_bonds=token_bonds,
        atom_parent_ids=i.atom_parent_ids,
        atom_ids=atom_ids,
        atompair_ids=atompair_ids,
        resolved_labels=i.resolved_labels,
        token_constraints=i.token_constraints,
        chains=chains,
        num_ligands=num_ligands,
        filepath=i.filepath,
        example_source=i.example_source,
    )

    return atom_input


# megafold input - support polypeptides, nucleic acids, metal ions + any number of ligands + misc biomolecules

imm_list = partial(field, default_factory=list)


@typecheck
@dataclass
class MegaFoldInput:
    """Dataclass for MegaFold inputs."""

    proteins: List[Int[" _"] | str] = imm_list()  # type: ignore
    ss_dna: List[Int[" _"] | str] = imm_list()  # type: ignore
    ss_rna: List[Int[" _"] | str] = imm_list()  # type: ignore
    metal_ions: Int[" _"] | List[str] = imm_list()  # type: ignore
    misc_molecule_ids: Int[" _"] | List[str] = imm_list()  # type: ignore
    ligands: List[Mol | str] = imm_list()  # can be given as smiles
    ds_dna: List[Int[" _"] | str] = imm_list()  # type: ignore
    ds_rna: List[Int[" _"] | str] = imm_list()  # type: ignore
    atom_parent_ids: Int[" m"] | None = None  # type: ignore
    missing_atom_indices: List[List[int] | None] = imm_list()  # type: ignore
    additional_msa_feats: Float["s n dmf"] | None = None  # type: ignore
    additional_token_feats: Float["n dtf"] | None = None  # type: ignore
    templates: Float["t n n dt"] | None = None  # type: ignore
    msa: Float["s n dmi"] | None = None  # type: ignore
    atom_pos: List[Float["_ 3"]] | Float["m 3"] | None = None  # type: ignore
    template_mask: Bool[" t"] | None = None  # type: ignore
    msa_mask: Bool[" s"] | None = None  # type: ignore
    resolved_labels: Int[" m"] | None = None  # type: ignore
    token_constraints: Float["n n dac"] | None = None  # type: ignore
    chains: Tuple[int | None, int | None] | None = (None, None)
    example_source: str | None = None
    add_atom_ids: bool = True
    add_atompair_ids: bool = True
    directed_bonds: bool = True
    extract_atom_feats_fn: Callable[[Atom], Float["m dai"]] = default_extract_atom_feats_fn  # type: ignore
    extract_atompair_feats_fn: Callable[[Mol], Float["m m dapi"]] = default_extract_atompair_feats_fn  # type: ignore
    custom_atoms: List[str] | None = None
    custom_bonds: List[str] | None = None
    input_id: str | None = None
    sequence_ordering: List[int] | None = None


@typecheck
def tokenize_polymer_string(input_string: str) -> List[str]:
    """Employ a regular expression to match either single characters or special regions enclosed in
    <>.

    Returns a list of polymer tokens.
    """
    pattern = re.compile(r"<[^>]+>|.")
    tokens = pattern.findall(input_string)
    return tokens


@typecheck
def map_int_or_string_indices_to_mol(
    entries: dict,
    polymer_type_index: Literal[IS_PROTEIN_INDEX, IS_RNA_INDEX, IS_DNA_INDEX],  # type: ignore
    indices: List[str] | str,
    mol_keyname="rdchem_mol",
    return_entries=False,
) -> Tuple[List[Mol], List[List[bool]]] | Tuple[List[Mol], List[List[bool]], List[dict]]:
    """Map indices to molecules."""
    if isinstance(indices, str):
        indices = tokenize_polymer_string(indices)

    # get all the peptide or nucleotide entries
    # while handling for modified residues

    mols = []
    is_mol_mod = []
    mapped_entries = []

    for idx, restype in enumerate(indices):
        is_mod = [False for _ in range(DEFAULT_NUM_MOLECULE_MODS)]

        mod_polymer_result = re.search(r"<(.*?)>", restype)

        if mod_polymer_result:
            smiles = CCD_COMPONENTS_SMILES.get(mod_polymer_result.group(1), None)
            if not_exists(smiles):
                raise ValueError(
                    f"Unknown modified residue at index {idx}: {mod_polymer_result.group(1)}. Please provide a valid CCD code for this residue."
                )

            mol = mol_from_smile(smiles)

            entry = {
                mol_keyname: mol,
                "token_center_atom_idx": -1,
                "distogram_atom_idx": -1,
                "first_atom_idx": 0,
                "last_atom_idx": mol.GetNumAtoms() - 1,
                "three_atom_indices_for_frame": None,
            }
            is_mod[polymer_type_index] = True

        else:
            entry = copy.deepcopy(entries[restype])

        mols.append(entry[mol_keyname])
        is_mol_mod.append(is_mod)
        mapped_entries.append(entry)

    if not return_entries:
        return mols, is_mol_mod

    return mols, is_mol_mod, mapped_entries


@typecheck
def maybe_string_to_int(
    entries: dict, extended_types: Dict[str, str], indices: List[str] | str
) -> Tuple[Int[" _"], str]:  # type: ignore
    """Convert string to int along with a mapped version of the string."""
    if isinstance(indices, str):
        indices = tokenize_polymer_string(indices)

    # get all the peptide or nucleotide sequences
    # while handling for modified residues

    unknown_index = len(entries) - 1

    index = {symbol: i for i, symbol in enumerate(entries.keys())}

    mol_ids = []
    mapped_restypes = []

    for restype in indices:
        mapped_restype = restype
        mod_polymer_result = re.search(r"<(.*?)>", restype)

        if mod_polymer_result:
            mapped_restype = (
                extended_types[mod_polymer_result.group(1)]
                if mod_polymer_result.group(1) in extended_types
                else "X"
            )

        mol_ids.append(index.get(mapped_restype, unknown_index))
        mapped_restypes.append(mapped_restype)

    mol_ids = tensor(mol_ids)
    return mol_ids, "".join(mapped_restypes)


@typecheck
def megafold_input_to_molecule_lengthed_molecule_input(
    megafold_input: MegaFoldInput,
) -> MoleculeLengthMoleculeInput:
    """Convert an MegaFoldInput to a MoleculeLengthMoleculeInput."""
    i = megafold_input

    proteins = list(i.proteins)
    ss_rnas = list(i.ss_rna)
    ss_dnas = list(i.ss_dna)
    ds_rnas = list(i.ds_rna)
    ds_dnas = list(i.ds_dna)
    ligands = list(i.ligands)
    metal_ions = list(i.metal_ions)

    # handle atom positions - need atom positions for deriving frame of ligand for PAE

    atom_pos = i.atom_pos

    if isinstance(atom_pos, list):
        atom_pos = torch.cat(atom_pos)

    # any double stranded nucleic acids is added to single stranded lists with its reverse complement
    # rc stands for reverse complement

    for ss_rna_index in range(len(ds_rnas)):
        # NOTE: the current implementation will not atomize modified RNA residues
        # in a double-stranded input sequence or its reverse complement
        _, seq = maybe_string_to_int(
            RNA_NUCLEOTIDES, NUCLEIC_LETTERS_3TO1_EXTENDED, ds_rnas[ss_rna_index]
        )

        rc_fn = (
            partial(reverse_complement, nucleic_acid_type="rna")
            if isinstance(seq, str)
            else reverse_complement_tensor
        )
        rc_seq = rc_fn(seq)
        ss_rnas.extend([seq, rc_seq])

        ds_rnas[ss_rna_index] = seq

    for ss_rna_index in range(len(ds_dnas)):
        # NOTE: the current implementation will not atomize modified DNA residues
        # in a double-stranded input sequence or its reverse complement
        _, seq = maybe_string_to_int(
            DNA_NUCLEOTIDES, NUCLEIC_LETTERS_3TO1_EXTENDED, ds_dnas[ss_rna_index]
        )

        rc_fn = (
            partial(reverse_complement, nucleic_acid_type="dna")
            if isinstance(seq, str)
            else reverse_complement_tensor
        )
        rc_seq = rc_fn(seq)
        ss_dnas.extend([seq, rc_seq])

        ds_dnas[ss_rna_index] = seq

    # handle arbitrary sequence ordering

    sequence_ordering = list(
        range(sum(map(len, [proteins, ss_rnas, ss_dnas, ligands, metal_ions])))
    )

    if exists(i.sequence_ordering):
        sequence_ordering = i.sequence_ordering

        if exists(i.atom_pos):
            raise NotImplementedError(
                "Arbitrary sequence ordering is not supported when atom positions are provided. "
                "Please remove the `atom_pos` field from the input."
            )
        if ds_rnas or ds_dnas:
            raise NotImplementedError(
                "Arbitrary sequence ordering is not supported when double-stranded nucleic acids are provided. "
                "Please remove the `ds_rna` and `ds_dna` fields from the input."
            )

    # keep track of input sequence ordering

    molecule_seq_ids = []

    # keep track of molecule_ids - for now it is
    # proteins (21) | rna (5) | dna (5) | gap? (1) - unknown for each biomolecule is the very last, ligand is 20

    rna_offset = len(HUMAN_AMINO_ACIDS)
    dna_offset = len(RNA_NUCLEOTIDES) + rna_offset
    ligand_id = len(HUMAN_AMINO_ACIDS) - 1

    molecule_ids = []

    # convert all proteins to a List[Mol] of each peptide

    mol_proteins = []
    protein_entries = []

    is_molecule_mod = []

    distogram_atom_indices = []
    molecule_atom_indices = []
    src_tgt_atom_indices = []
    atom_indices_for_frame = []

    for protein_index in range(len(proteins)):
        mol_peptides, is_mol_mod, protein_entries = map_int_or_string_indices_to_mol(
            entries=HUMAN_AMINO_ACIDS,
            polymer_type_index=IS_PROTEIN_INDEX,
            indices=proteins[protein_index],
            return_entries=True,
        )
        for mol_peptide in mol_peptides:
            mol_peptide.SetProp("mol_type", f"protein_{protein_index}")

        mol_proteins.append(mol_peptides)
        is_molecule_mod.extend(is_mol_mod)

        distogram_atom_indices.extend([entry["distogram_atom_idx"] for entry in protein_entries])
        molecule_atom_indices.extend([entry["token_center_atom_idx"] for entry in protein_entries])

        src_tgt_atom_indices.extend(
            [[entry["first_atom_idx"], entry["last_atom_idx"]] for entry in protein_entries]
        )

        atom_indices_for_frame.extend(
            [entry["three_atom_indices_for_frame"] for entry in protein_entries]
        )

        protein_ids, mapped_protein = maybe_string_to_int(
            HUMAN_AMINO_ACIDS, PROTEIN_LETTERS_3TO1_EXTENDED, proteins[protein_index]
        )
        molecule_ids.append(protein_ids)
        molecule_seq_ids.extend(
            [sequence_ordering[protein_index] for _ in range(len(protein_ids))]
        )

        proteins[protein_index] = mapped_protein

    # convert all single stranded nucleic acids to mol

    mol_ss_dnas = []
    mol_ss_rnas = []

    for ss_rna_index in range(len(ss_rnas)):
        mol_seq, is_mol_mod, ss_rna_entries = map_int_or_string_indices_to_mol(
            entries=RNA_NUCLEOTIDES,
            polymer_type_index=IS_RNA_INDEX,
            indices=ss_rnas[ss_rna_index],
            return_entries=True,
        )
        for mol_s in mol_seq:
            mol_s.SetProp("mol_type", f"rna_{ss_rna_index + len(proteins)}")

        mol_ss_rnas.append(mol_seq)
        is_molecule_mod.extend(is_mol_mod)

        distogram_atom_indices.extend([entry["distogram_atom_idx"] for entry in ss_rna_entries])
        molecule_atom_indices.extend([entry["token_center_atom_idx"] for entry in ss_rna_entries])

        src_tgt_atom_indices.extend(
            [[entry["first_atom_idx"], entry["last_atom_idx"]] for entry in ss_rna_entries]
        )

        atom_indices_for_frame.extend(
            [entry["three_atom_indices_for_frame"] for entry in ss_rna_entries]
        )

        rna_ids, mapped_rna = maybe_string_to_int(
            RNA_NUCLEOTIDES, NUCLEIC_LETTERS_3TO1_EXTENDED, ss_rnas[ss_rna_index]
        )
        molecule_ids.append(rna_ids + rna_offset)
        molecule_seq_ids.extend(
            [sequence_ordering[ss_rna_index + len(proteins)] for _ in range(len(rna_ids))]
        )

        ss_rnas[ss_rna_index] = mapped_rna

    for ss_dna_index in range(len(ss_dnas)):
        mol_seq, is_mol_mod, ss_dna_entries = map_int_or_string_indices_to_mol(
            entries=DNA_NUCLEOTIDES,
            polymer_type_index=IS_DNA_INDEX,
            indices=ss_dnas[ss_dna_index],
            return_entries=True,
        )
        for mol_s in mol_seq:
            mol_s.SetProp("mol_type", f"dna_{ss_dna_index + len(proteins) + len(ss_rnas)}")

        mol_ss_dnas.append(mol_seq)
        is_molecule_mod.extend(is_mol_mod)

        distogram_atom_indices.extend([entry["distogram_atom_idx"] for entry in ss_dna_entries])
        molecule_atom_indices.extend([entry["token_center_atom_idx"] for entry in ss_dna_entries])

        src_tgt_atom_indices.extend(
            [[entry["first_atom_idx"], entry["last_atom_idx"]] for entry in ss_dna_entries]
        )

        atom_indices_for_frame.extend(
            [entry["three_atom_indices_for_frame"] for entry in ss_dna_entries]
        )

        dna_ids, mapped_dna = maybe_string_to_int(
            DNA_NUCLEOTIDES, NUCLEIC_LETTERS_3TO1_EXTENDED, ss_dnas[ss_dna_index]
        )
        molecule_ids.append(dna_ids + dna_offset)
        molecule_seq_ids.extend(
            [
                sequence_ordering[ss_dna_index + len(proteins) + len(ss_rnas)]
                for _ in range(len(dna_ids))
            ]
        )

        ss_dnas[ss_dna_index] = mapped_dna

    # convert ligands to rdchem.Mol

    mol_ligands = [
        (mol_from_smile(ligand) if isinstance(ligand, str) else ligand) for ligand in ligands
    ]
    for ligand_idx, mol_ligand in enumerate(mol_ligands):
        mol_ligand.SetProp(
            "mol_type", f"ligand_{ligand_idx + len(proteins) + len(ss_rnas) + len(ss_dnas)}"
        )

    molecule_ids.append(tensor([ligand_id] * len(mol_ligands)))
    molecule_seq_ids.extend(
        [
            sequence_ordering[ligand_idx + len(proteins) + len(ss_rnas) + len(ss_dnas)]
            for ligand_idx in range(len(mol_ligands))
        ]
    )
    is_molecule_mod.extend([[False for _ in range(DEFAULT_NUM_MOLECULE_MODS)]] * len(mol_ligands))

    distogram_atom_indices.extend([-1] * len(mol_ligands))
    molecule_atom_indices.extend([-1] * len(mol_ligands))
    src_tgt_atom_indices.extend([[0, mol_ligand.GetNumAtoms()] for mol_ligand in mol_ligands])
    atom_indices_for_frame.extend([None] * len(mol_ligands))

    # handle frames for the ligands, which depends on knowing the atom positions (section 4.3.2)

    if exists(atom_pos):
        ligand_atom_pos_offset = 0

        for mol in flatten([*mol_proteins, *mol_ss_rnas, *mol_ss_dnas]):
            ligand_atom_pos_offset += mol.GetNumAtoms()

        for mol_ligand in mol_ligands:
            num_ligand_atoms = mol_ligand.GetNumAtoms()
            ligand_atom_pos = atom_pos[
                ligand_atom_pos_offset : (ligand_atom_pos_offset + num_ligand_atoms)
            ]

            frames = get_frames_from_atom_pos(ligand_atom_pos, filter_colinear_pos=True)

            # NOTE: since `MegaFoldInput` is only used for inference, we can safely assume that
            # the middle atom frame of each ligand molecule is a suitable representative frame for the ligand
            atom_indices_for_frame.append(frames[len(frames) // 2].tolist())

            ligand_atom_pos_offset += num_ligand_atoms

    # convert metal ions to rdchem.Mol

    mol_metal_ions = [
        (
            mol_from_smile(metal_ion)
            if isinstance(metal_ion, str)
            else CCD_COMPONENTS_SMILES.get(metal_ion, CCD_COMPONENTS_SMILES.get(metal_ion.upper()))
        )
        for metal_ion in metal_ions
    ]
    for metal_ion_idx, mol_metal_ion in enumerate(mol_metal_ions):
        mol_metal_ion.SetProp(
            "mol_type",
            f"metal_ion_{metal_ion_idx + len(proteins) + len(ss_rnas) + len(ss_dnas) + len(ligands)}",
        )

    molecule_ids.append(tensor([MOLECULE_METAL_ION_ID] * len(mol_metal_ions)))
    molecule_seq_ids.extend(
        [
            sequence_ordering[
                metal_ion_idx + len(proteins) + len(ss_rnas) + len(ss_dnas) + len(ligands)
            ]
            for metal_ion_idx in range(len(mol_metal_ions))
        ]
    )
    is_molecule_mod.extend(
        [[False for _ in range(DEFAULT_NUM_MOLECULE_MODS)]] * len(mol_metal_ions)
    )

    distogram_atom_indices.extend([-1] * len(mol_metal_ions))
    molecule_atom_indices.extend([-1] * len(mol_metal_ions))
    src_tgt_atom_indices.extend(
        [[0, mol_metal_ion.GetNumAtoms()] for mol_metal_ion in mol_metal_ions]
    )
    atom_indices_for_frame.extend([None] * len(mol_metal_ions))

    # create the molecule input

    all_protein_mols = flatten(mol_proteins)
    all_rna_mols = flatten(mol_ss_rnas)
    all_dna_mols = flatten(mol_ss_dnas)

    molecules_without_ligands = [
        *all_protein_mols,
        *all_rna_mols,
        *all_dna_mols,
    ]

    # correctly generate the is_molecule_types, which is a boolean tensor of shape [*, 5]
    # is_protein | is_rna | is_dna | is_ligand | is_metal_ions
    # this is needed for their special diffusion loss

    molecule_type_token_lens = [
        len(all_protein_mols),
        len(all_rna_mols),
        len(all_dna_mols),
        len(mol_ligands),
        len(mol_metal_ions),
    ]

    num_tokens = sum(molecule_type_token_lens)

    assert num_tokens > 0, "You have requested an empty MegaFoldInput."

    arange = torch.arange(num_tokens)[:, None]

    molecule_types_lens_cumsum = tensor([0, *molecule_type_token_lens]).cumsum(dim=-1)
    left, right = molecule_types_lens_cumsum[:-1], molecule_types_lens_cumsum[1:]

    is_molecule_types = (arange >= left) & (arange < right)

    # handle MSA and template features, if provided

    if exists(i.sequence_ordering) and any(
        map(
            exists,
            [
                i.msa,
                i.templates,
                i.msa_mask,
                i.template_mask,
                i.additional_msa_feats,
                i.additional_token_feats,
            ],
        )
    ):
        raise NotImplementedError(
            "Arbitrary sequence ordering is not supported when MSA or template features are provided. "
            "Please remove the `msa`, `templates`, `msa_mask`, `template_mask`, `additional_msa_feats`, and `additional_token_feats` fields from the input."
        )

    num_msas = len(i.msa) if exists(i.msa) else 1

    additional_msa_feats = default(i.additional_msa_feats, torch.zeros(num_msas, num_tokens, 2))
    additional_token_feats = default(i.additional_token_feats, torch.zeros(num_tokens, 33))

    # handle token_constraints

    if exists(i.sequence_ordering) and exists(i.token_constraints):
        raise NotImplementedError(
            "Arbitrary sequence ordering is not supported when token constraints are provided. "
            "Please remove the `token_constraints` field from the input."
        )

    # convert feature lists to tensors

    molecule_ids = torch.cat(molecule_ids).long()
    molecule_seq_ids = tensor(molecule_seq_ids)

    is_molecule_mod = tensor(is_molecule_mod)
    src_tgt_atom_indices = tensor(src_tgt_atom_indices)

    distogram_atom_indices = tensor(distogram_atom_indices)
    molecule_atom_indices = tensor(molecule_atom_indices)

    # all molecules, layout is
    # proteins | ss rna | ss dna | ligands | metal ions

    molecules = [*molecules_without_ligands, *mol_ligands, *mol_metal_ions]

    for mol in molecules:
        Chem.SanitizeMol(mol)

    # reorder the molecules according to the input sequence ordering,
    # to support arbitrary ordering of input sequence types

    molecule_sorted_indices = torch.argsort(molecule_seq_ids, stable=True)
    assert len(molecule_sorted_indices.unique()) == len(molecules), (
        "The input sequence ordering must be unique for each molecule. "
        "Please ensure that the `sequence_ordering` field is unique for each molecule."
    )

    molecule_sequences = [
        *proteins,
        *ss_rnas,
        *ss_dnas,
        *ligands,
        *metal_ions,
    ]
    molecule_sequence_types = [
        *(["protein"] * len(proteins)),
        *(["rna"] * len(ss_rnas)),
        *(["dna"] * len(ss_dnas)),
        *(["ligand"] * len(ligands)),
        *(["metal_ion"] * len(metal_ions)),
    ]

    sequences = [
        (molecule_sequences[i], molecule_sequence_types[i]) for i in np.argsort(sequence_ordering)
    ]
    molecules = [molecules[i] for i in molecule_sorted_indices]

    molecule_ids = molecule_ids[molecule_sorted_indices]
    is_molecule_mod = is_molecule_mod[molecule_sorted_indices]
    is_molecule_types = is_molecule_types[molecule_sorted_indices]

    distogram_atom_indices = distogram_atom_indices[molecule_sorted_indices]
    molecule_atom_indices = molecule_atom_indices[molecule_sorted_indices]
    src_tgt_atom_indices = src_tgt_atom_indices[molecule_sorted_indices]

    atom_indices_for_frame = [atom_indices_for_frame[i] for i in molecule_sorted_indices]

    # sanity-check atom_indices_for_frame and related features

    assert len(distogram_atom_indices) == len(molecules)
    assert len(molecule_atom_indices) == len(molecules)
    assert len(src_tgt_atom_indices) == len(molecules)
    assert len(atom_indices_for_frame) == len(molecules)

    # constructing the additional_molecule_feats
    # which is in turn used to derive relative positions

    # residue_index - an arange that restarts at 1 for each chain
    # token_index   - just an arange
    # asym_id       - unique id for each chain of a biomolecule
    # entity_id     - unique id for each biomolecule sequence
    # sym_id        - unique id for each chain of the same biomolecule sequence

    @typecheck
    def get_token_repeats(sequences: List[Tuple[str, str]]) -> Int[" num_chains"]:  # type: ignore
        """Get the number of tokens for each sequence."""
        token_repeats = []
        for sequence, sequence_type in sequences:
            token_repeats.append(1 if sequence_type in ("ligand", "metal_ion") else len(sequence))
        return tensor(token_repeats)

    token_repeats = get_token_repeats(sequences)

    # residue ids

    residue_index = torch.cat([torch.arange(i) for i in token_repeats])

    # asym ids

    asym_ids = repeat_interleave(torch.arange(len(token_repeats)), token_repeats)

    # entity ids

    unrepeated_entity_sequences = defaultdict(int)
    for entity_sequence, _ in sequences:
        if entity_sequence in unrepeated_entity_sequences:
            continue
        unrepeated_entity_sequences[entity_sequence] = len(unrepeated_entity_sequences)

    unrepeated_entity_ids = [
        unrepeated_entity_sequences[entity_sequence] for entity_sequence, _ in sequences
    ]

    entity_id_counts = token_repeats  # NOTE: we borrow the token_repeats for entity_id_counts

    entity_ids = repeat_interleave(tensor(unrepeated_entity_ids), entity_id_counts)

    # sym ids

    unrepeated_sym_ids = []
    unrepeated_sym_sequences = defaultdict(int)
    for entity_sequence, _ in sequences:
        unrepeated_sym_ids.append(unrepeated_sym_sequences[entity_sequence])
        if entity_sequence in unrepeated_sym_sequences:
            unrepeated_sym_sequences[entity_sequence] += 1

    sym_ids = repeat_interleave(tensor(unrepeated_sym_ids), entity_id_counts)

    # concat for all of additional_molecule_feats

    additional_molecule_feats = torch.stack((residue_index, asym_ids, entity_ids, sym_ids), dim=-1)

    # handle missing atom indices

    missing_atom_indices = None
    missing_token_indices = None

    if exists(i.missing_atom_indices) and len(i.missing_atom_indices) > 0:
        assert not exists(i.sequence_ordering), (
            "Arbitrary sequence ordering is not supported when missing atom indices are provided. "
            "Please remove the `missing_atom_indices` field from the input."
        )

        missing_atom_indices = []
        missing_token_indices = []

        for mol_miss_atom_indices, mol in zip(i.missing_atom_indices, molecules):
            mol_miss_atom_indices = default(mol_miss_atom_indices, [])
            mol_miss_atom_indices = tensor(mol_miss_atom_indices, dtype=torch.long)

            missing_atom_indices.append(mol_miss_atom_indices)
            missing_token_indices.append(mol_miss_atom_indices)

        assert len(molecules) == len(missing_atom_indices)
        assert len(missing_token_indices) == num_tokens

    # handle generated conformer coordinates

    if not_exists(atom_pos):
        atom_pos = torch.cat(
            [torch.from_numpy(mol.GetConformer().GetPositions()).float() for mol in molecules]
        )

    # create atom_parent_ids, which governs in the atom encoder / decoder which atom
    # attends to which, where a design choice is made such that chain indices are directly
    # adopted to group atoms belonging to the same (user-denoted) sequence chain

    @typecheck
    def get_atom_counts(molecules: List[Mol]) -> List[int]:
        """Get the number of atoms per chain."""
        atoms_per_chain = []

        chain_grouped_molecules = defaultdict(list)
        for mol in molecules:
            chain_grouped_molecules[mol.GetProp("mol_type")].append(mol)

        for chain in chain_grouped_molecules.values():
            num_atoms = 0
            for mol in chain:
                num_atoms += mol.GetNumAtoms()
            atoms_per_chain.append(num_atoms)

        return atoms_per_chain

    atom_counts = get_atom_counts(molecules)

    atom_parent_ids = repeat_interleave(torch.arange(len(atom_counts)), tensor(atom_counts))

    # create molecule input

    molecule_input = MoleculeLengthMoleculeInput(
        molecules=molecules,
        molecule_atom_indices=molecule_atom_indices,
        distogram_atom_indices=distogram_atom_indices,
        molecule_ids=molecule_ids,
        additional_molecule_feats=additional_molecule_feats,
        additional_msa_feats=additional_msa_feats,
        additional_token_feats=additional_token_feats,
        is_molecule_types=is_molecule_types,
        is_molecule_mod=is_molecule_mod,
        missing_atom_indices=missing_atom_indices,
        missing_token_indices=missing_token_indices,
        src_tgt_atom_indices=src_tgt_atom_indices,
        atom_indices_for_frame=atom_indices_for_frame,
        atom_pos=atom_pos,
        templates=i.templates,
        msa=i.msa,
        template_mask=i.template_mask,
        msa_mask=i.msa_mask,
        atom_parent_ids=atom_parent_ids,
        token_constraints=i.token_constraints,
        chains=i.chains,
        example_source=i.example_source,
        add_atom_ids=i.add_atom_ids,
        add_atompair_ids=i.add_atompair_ids,
        directed_bonds=i.directed_bonds,
        extract_atom_feats_fn=i.extract_atom_feats_fn,
        extract_atompair_feats_fn=i.extract_atompair_feats_fn,
        custom_atoms=i.custom_atoms,
        custom_bonds=i.custom_bonds,
    )

    return molecule_input


@typecheck
def megafold_inputs_to_batched_atom_input(
    inp: MegaFoldInput | PDBInput | List[MegaFoldInput | PDBInput], **collate_kwargs
) -> BatchedAtomInput:
    """Convert a list of MegaFoldInput or PDBInput objects into a BatchedAtomInput object."""
    if isinstance(inp, MegaFoldInput):
        inp = [inp]

    atom_inputs = maybe_transform_to_atom_inputs(inp)
    return collate_inputs_to_batched_atom_input(atom_inputs, **collate_kwargs)


@typecheck
def megafold_input_to_pdb_input(
    megafold_input: MegaFoldInput,
    atom_positions: np.ndarray | None = None,
    mmcif_dir: str | None = None,
    msa_dir: str | None = None,
    templates_dir: str | None = None,
    inference: bool = True,
    constraints: INPUT_CONSTRAINTS | None = None,
    kalign_binary_path: str | None = None,
) -> PDBInput:
    """Convert an MegaFoldInput object into a PDBInput object.

    :param megafold_input: The MegaFoldInput object for multi-domain biomolecules.
    :param atom_positions: The sampled or reference atom coordinates of shape [num_res,
        repr_dimension (47), 3]
    :param mmcif_dir: The directory containing input (template) mmCIF files.
    :param msa_dir: The directory containing the MSA files.
    :param templates_dir: The directory containing the template files.
    :param inference: Whether the input is for inference or training.
    :param constraints: The constraints to be provided at runtime.
    :param kalign_binary_path: The path to the Kalign binary for template alignment.
    :return: A PDBInput object for data handling within the rest of the codebase.
    """
    # Get a `Biomolecule` representation of the `MegaFoldInput` object
    biomol = megafold_input_to_biomolecule(megafold_input, atom_positions=atom_positions)

    # Craft a `PDBInput` object using the new `Biomolecule` representation
    stub_mmcif_filepath = f"{megafold_input.input_id}.cif"

    pdb_input = PDBInput(
        mmcif_filepath=stub_mmcif_filepath,
        biomol=biomol,
        msa_dir=msa_dir,
        templates_dir=templates_dir,
        template_mmcif_dir=mmcif_dir,
        inference=inference,
        constraints=constraints,
        kalign_binary_path=kalign_binary_path,
        example_source=megafold_input.example_source,
    )

    # Return the PDBInput object
    return pdb_input


@typecheck
def megafold_input_to_biomolecule(
    megafold_input: MegaFoldInput, atom_positions: np.ndarray | None = None
) -> Biomolecule:
    """Convert an MegaFoldInput object into a Biomolecule object.

    :param megafold_input: The MegaFoldInput Object for multi-domain biomolecules.
    :param atom_positions: The sampled or reference atom coordinates of shape [num_res,
        repr_dimension (47), 3]
    :return: A Biomolecule object for data handling with the rest of the codebase.
    """
    input_transform = INPUT_TO_ATOM_TRANSFORM[MegaFoldInput]
    megafold_atom_input = input_transform(megafold_input)

    # Ensure that the atom positions are of the correct shape
    if exists(atom_positions):
        assert atom_positions.shape[0] == len(
            megafold_atom_input.molecule_ids
        ), "Please ensure that the atoms are of the shape [num_res, repr, 3]"
        assert (
            atom_positions.shape[-1] == 3
        ), "Please ensure that the atoms are of the shape [num_res, repr, 3]"

    # Step 1. Get the various intermediate inputs
    # Hacky solution: Need to double up on ligand because metal constants dont exist yet
    ALL_restypes = np.concatenate(
        [
            amino_acid_constants.restype_atom47_to_compact_atom,
            rna_constants.restype_atom47_to_compact_atom,
            dna_constants.restype_atom47_to_compact_atom,
            ligand_constants.restype_atom47_to_compact_atom,
            ligand_constants.restype_atom47_to_compact_atom,
        ],
        axis=0,
    )

    molecule_ids = megafold_atom_input.molecule_ids.cpu().numpy()
    molecule_types = np.nonzero(megafold_atom_input.is_molecule_types)[:, 1]

    is_molecule_mod = megafold_atom_input.is_molecule_mod.cpu().numpy()
    molecule_atom_indices = megafold_atom_input.molecule_atom_indices.cpu().numpy()

    atom_ids = megafold_atom_input.atom_ids.cpu().numpy()

    restype_to_atom = np.array(
        [
            ALL_restypes[-1 if mol_type > 2 or is_mol_mod.any() else mol_idx]
            for mol_idx, mol_type, is_mol_mod in zip(molecule_ids, molecule_types, is_molecule_mod)
        ]
    )
    res_rep_atom_indices = np.array(
        [
            (
                atom_id
                if molecule_type > 2 or is_mol_mod.any()
                else get_residue_constants(res_chem_index=molecule_type.item()).res_rep_atom_index
            )
            for molecule_type, is_mol_mod, atom_id in zip(
                molecule_types, is_molecule_mod, atom_ids[molecule_atom_indices]
            )
        ]
    )

    # Step 2. Atom Names
    # atom_names: array of strings [num_res], each residue is denoted by representative atom name
    atom_names = []

    for res_idx in range(len(molecule_ids)):
        molecule_type = molecule_types[res_idx].item()
        is_mol_mod = is_molecule_mod[res_idx]

        residue_constants = (
            get_residue_constants("ligand")
            if is_mol_mod.any()
            else get_residue_constants(res_chem_index=molecule_type)
        )

        residue = molecule_ids[res_idx].item()
        residue_offset = residue_constants.min_restype_num
        residue_idx = residue - residue_offset
        atom_idx = res_rep_atom_indices[res_idx]

        # If the molecule type is unmodified protein, RNA, or DNA
        if molecule_type < 3 and not is_mol_mod.any():
            # Dictionary of Residue to Atoms
            res_to_atom = residue_constants.restype_name_to_compact_atom_names
            residue_name = residue_constants.resnames[residue_idx]
            atom_names.append(res_to_atom[residue_name][atom_idx])
        else:
            atom_name = residue_constants.restype_name_to_compact_atom_names["UNL"][atom_idx]
            atom_names.append(atom_name)

    # Step 3. Restypes
    # restypes: np.array [num_res] w/ values from 0 to 32
    res_types = molecule_ids.copy()

    # Step 4. Atom Masks
    # atom_masks: np.array [num_res, num_atom_types (47)]
    # Due to the first Atom that's present being a zero due to zero indexed counts we force it to be a one.
    atom_masks = np.stack(
        [
            np.array(np.concatenate([np.array([1]), r2a[1:]]) != 0).astype(int)
            for r2a in restype_to_atom
        ]
    )

    one_token_per_atom_mask = ((molecule_types > 2) | is_molecule_mod.any(-1)).bool()
    one_token_per_atom_mask_np = one_token_per_atom_mask.cpu().numpy()

    atom_masks[one_token_per_atom_mask_np] = False
    atom_masks[one_token_per_atom_mask_np, res_rep_atom_indices[one_token_per_atom_mask_np]] = True

    # Step 5. Residue Index
    # residue_index: np.array [num_res], 1-indexed
    residue_index = megafold_atom_input.additional_molecule_feats.cpu().numpy()[:, 0] + 1

    # Step 6. Chain Index
    # chain_index: np.array [num_res], borrow the entity IDs (vs sym_ids, idx3) as chain IDs
    chain_index = megafold_atom_input.additional_molecule_feats.cpu().numpy()[:, 2]

    # Step 7. Chain IDs
    # chain_ids: list of strings [num_res], each residue is denoted by chain ID
    chain_ids = np.array([index_to_pdb_chain_id(x) for x in chain_index])

    # Step 8. B-Factors
    # b_factors: np.ndarray [num_res, num_atom_type]
    b_factors = np.ones_like(atom_masks, dtype=float)

    # Step 9. ChemIDs and ChemComps
    # chemids: list of strings [num_res], each residue is denoted by chemical ID
    # residue_chem_comp_details: set of ChemComp objects

    @typecheck
    def get_sequences(i: MegaFoldInput, chain_index: Int[" n"], residue_index: Int[" n"], one_token_per_atom: Bool[" n"], is_molecule_mod: Bool[" n"]) -> List[str]:  # type: ignore
        """Get the sequences for each molecule."""
        proteins = list(i.proteins)
        ss_rnas = list(i.ss_rna)
        ss_dnas = list(i.ss_dna)
        ligands = list(i.ligands)
        metal_ions = list(i.metal_ions)

        # handle arbitrary sequence ordering

        sequence_ordering = list(
            range(sum(map(len, [proteins, ss_rnas, ss_dnas, ligands, metal_ions])))
        )

        if exists(i.sequence_ordering):
            sequence_ordering = i.sequence_ordering

        molecule_sequences = [
            *proteins,
            *ss_rnas,
            *ss_dnas,
            *ligands,
            *metal_ions,
        ]
        sequences = [molecule_sequences[i] for i in np.argsort(sequence_ordering)]

        sequence_index = 0
        mapped_sequences = []

        for token_idx, (chain_idx, residue_idx, is_mol_mod) in enumerate(
            zip(chain_index, residue_index, is_molecule_mod)
        ):
            seq = (
                sequences[sequence_index]
                if one_token_per_atom[token_idx].item() and not is_mol_mod.any()
                else tokenize_polymer_string(sequences[sequence_index])
            )

            if one_token_per_atom[token_idx].item():
                mapped_sequences.append(seq[residue_idx - 1] if is_mol_mod.any() else seq)
            else:
                mapped_sequences.append(seq[residue_idx - 1])

            if token_idx < len(chain_index) - 1 and chain_index[token_idx + 1] != chain_idx:
                sequence_index += 1

        return mapped_sequences

    chemids = []
    residue_chem_comp_details = set()

    sequences = get_sequences(
        megafold_input, chain_index, residue_index, one_token_per_atom_mask, is_molecule_mod
    )

    for idx in range(len(molecule_ids)):
        mt = molecule_types[idx].item()
        is_mol_mod = is_molecule_mod[idx]

        residue_constants = get_residue_constants(res_chem_index=mt)

        restypes = residue_constants.restypes
        min_res_offset = residue_constants.min_restype_num
        restype_dict = {min_res_offset + i: restype for i, restype in enumerate(restypes)}

        if mt < 3 and not is_mol_mod.any():
            restype_dict[residue_constants.restype_num] = residue_constants.restype_3to1[
                residue_constants.unk_restype
            ]
            one_letter = restype_dict[molecule_ids[idx].item()]
            chemid = residue_constants.restype_1to3[one_letter]
        elif mt < 3:
            chemid = re.search(r"<(.*?)>", sequences[idx]).group(1)
        else:
            chemid = sequences[idx]

        chemids.append(chemid)
        residue_chem_comp_details.add(
            mmcif_parsing.ChemComp(
                id=chemid,
                formula="?",
                formula_weight="0.0",
                mon_nstd_flag="no",
                name=chemid,
                type=residue_constants.unk_chemtype,
            )
        )

    chemids = np.array(chemids)

    # Step 10. ChemTypes
    # chemtypes: np.array [num_res], each residue is denoted by chemical type 0-4
    chemtypes = np.nonzero(megafold_atom_input.is_molecule_types.cpu().numpy())[1]

    # Step 11. Entity to Chain
    # entity_to_chain: dict, entity ID to chain ID
    # quick and dirty assignment
    entity_to_chain = {int(x): [int(x)] for x in dict.fromkeys(chain_index)}

    # Step 12. Atom positions
    atom_positions_ = (
        atom_positions
        if exists(atom_positions)
        else atom_masks[..., None].repeat(3, axis=-1).astype(float)
    )
    if not exists(atom_positions):
        atom_positions_[atom_masks.astype(bool)] = megafold_atom_input.atom_pos.cpu().numpy()

    # Step 13. Biomolecule Object
    biomol = Biomolecule(
        atom_positions=atom_positions_,
        atom_name=atom_names,
        restype=res_types,
        atom_mask=atom_masks,
        residue_index=residue_index,
        chain_index=chain_index,
        chain_id=chain_ids,
        b_factors=b_factors,
        chemid=chemids,
        chemtype=chemtypes,
        bonds=None,
        unique_res_atom_names=None,
        author_cri_to_new_cri=None,
        chem_comp_table=residue_chem_comp_details,
        entity_to_chain=entity_to_chain,
        mmcif_to_author_chain=None,
        mmcif_metadata={
            "_pdbx_audit_revision_history.revision_date": [
                f"{datetime.today().strftime('%Y-%m-%d')}"
            ]
        },
    )

    return biomol


# pdb input


@typecheck
@dataclass
class PDBInput:
    """Dataclass for PDB inputs."""

    mmcif_filepath: str | None = None
    biomol: Biomolecule | None = None
    chains: Tuple[str | None, str | None] | None = (None, None)
    num_ligands: int | None = None
    cropping_config: Dict[str, float | int] | None = None
    msa_dir: str | None = None
    msa_cache_dir: str | None = None
    input_cache_dir: str | None = None
    templates_dir: str | None = None
    template_mmcif_dir: str | None = None
    cache_msa_only: bool = False
    cache_input_only: bool = False
    add_atom_ids: bool = True
    add_atompair_ids: bool = True
    directed_bonds: bool = True
    training: bool = False
    inference: bool = False
    distillation: bool = False
    filter_for_alphabetic_chain_orderings: bool = (
        False  # NOTE: this is currently not referenced for `PDBInputs`
    )
    distillation_multimer_sampling_ratio: float = 2.0 / 3.0
    distillation_pdb_ids: List[str] | None = None
    distillation_template_mmcif_dir: str | None = None
    resolution: float | None = None
    constraints: INPUT_CONSTRAINTS | None = None
    constraints_ratio: float = 0.1
    max_msas_per_chain: int | None = 16_384
    max_num_msa_tokens: int | None = None
    max_templates_per_chain: int | None = 20
    num_templates_per_chain: int | None = 4
    max_num_template_tokens: int | None = None
    max_num_atoms: int | None = None
    min_length: int | None = None
    max_length: int | None = None
    cutoff_date: str | None = None  # NOTE: must be supplied in "%Y-%m-%d" format
    kalign_binary_path: str | None = None
    example_source: str | None = None
    extract_atom_feats_fn: Callable[[Atom], Float["m dai"]] = default_extract_atom_feats_fn  # type: ignore
    extract_atompair_feats_fn: Callable[[Mol], Float["m m dapi"]] = default_extract_atompair_feats_fn  # type: ignore
    custom_atoms: List[str] | None = None
    custom_bonds: List[str] | None = None

    def __post_init__(self):
        """Run post-init checks."""

        if exists(self.mmcif_filepath):
            if not os.path.exists(self.mmcif_filepath) and not_exists(self.biomol):
                raise FileNotFoundError(f"mmCIF file not found: {self.mmcif_filepath}.")
            if not (
                self.mmcif_filepath.endswith(".cif") or self.mmcif_filepath.endswith(".cif.gz")
            ):
                raise ValueError(
                    f"mmCIF file `{self.mmcif_filepath}` must have a `.cif` or `.cif.gz` file extension."
                )
        elif not_exists(self.biomol):
            raise ValueError("Either an mmCIF file or a `Biomolecule` object must be provided.")

        if exists(self.cropping_config):
            assert self.cropping_config.keys() == {
                "contiguous_weight",
                "spatial_weight",
                "spatial_interface_weight",
                "n_res",
            }, (
                f"Invalid cropping config keys: {self.cropping_config.keys()}. "
                "Please ensure that the cropping config has the correct keys."
            )
            assert (
                sum(
                    [
                        self.cropping_config["contiguous_weight"],
                        self.cropping_config["spatial_weight"],
                        self.cropping_config["spatial_interface_weight"],
                    ]
                )
                == 1.0
            ), (
                f"Invalid cropping config weights: ({self.cropping_config['contiguous_weight']}, {self.cropping_config['spatial_weight']}, {self.cropping_config['spatial_interface_weight']}). "
                "Please ensure that the cropping config weights sum to 1.0."
            )
            assert self.cropping_config["n_res"] > 0, (
                f"Invalid number of residues for cropping: {self.cropping_config['n_res']}. "
                "Please ensure that the number of residues for cropping is greater than 0."
            )

        if exists(self.msa_dir) and not os.path.exists(self.msa_dir):
            raise FileNotFoundError(f"Provided MSA directory not found: {self.msa_dir}.")

        if exists(self.templates_dir) and not os.path.exists(self.templates_dir):
            raise FileNotFoundError(
                f"Provided templates directory not found: {self.templates_dir}."
            )

        if exists(self.distillation_template_mmcif_dir) and not os.path.exists(
            self.distillation_template_mmcif_dir
        ):
            raise FileNotFoundError(
                f"Provided distillation templates directory not found: {self.distillation_template_mmcif_dir}."
            )

        if exists(self.kalign_binary_path) and not os.path.exists(self.kalign_binary_path):
            raise FileNotFoundError(
                f"Provided Kalign binary path not found: {self.kalign_binary_path}."
            )


@typecheck
def extract_chain_residues_from_biomolecule_chemical_components(
    biomol: Biomolecule,
    chem_comps: List[mmcif_parsing.ChemComp],
) -> PDB_INPUT_CHAIN_SEQS:
    """From `Biomolecule` chemical components, extract a list of lists of tuples for each chain,
    where each inner list contains tuples of the form (residue_id, residue_type,
    residue_chem_type).

    NOTE: This function splits up chain residues into separate "pseudochains" if they are of different
    (contiguous) chemical types.
    """
    chain_index = biomol.chain_index
    residue_index = biomol.residue_index

    assert len(chem_comps) == len(chain_index) == len(residue_index), (
        f"The number of chemical components ({len(chem_comps)}), chain indices ({len(chain_index)}), and residue indices ({len(residue_index)}) do not match. "
        "Please ensure that chain and residue indices are correctly assigned to each chemical component."
    )

    chain_residues = []
    current_chain_seq = []

    chain_res_idx_seen = set()
    for idx, (comp_details, chain_idx, res_idx) in enumerate(
        zip(chem_comps, chain_index, residue_index)
    ):
        # determine the residue type and whether it is a modified polymer residue

        residue_constants = get_residue_constants(comp_details.type)
        res_letter, is_modified_polymer_residue = get_resname_and_modified_status(
            comp_details, residue_constants
        )

        # map chemical types to protein, DNA, RNA, or ligand,
        # treating modified polymer residues as pseudoligands

        res_chem_type = get_pdb_input_residue_molecule_type(
            comp_details.type,
            is_modified_polymer_residue=is_modified_polymer_residue,
        )

        chain_ending = idx + 1 < len(chain_index) and chain_index[idx] != chain_index[idx + 1]
        chem_type_ending = idx + 1 < len(chem_comps) and (
            (
                res_chem_type.removeprefix("mod_")
                != (
                    get_pdb_input_residue_molecule_type(
                        chem_comps[idx + 1].type,
                        is_modified_polymer_residue=False,
                    )
                )
            )
            or not is_polymer(chem_comps[idx + 1].type)
        )

        # only consider the first atom of each (e.g., ligand) residue

        chain_res_idx = f"{chain_idx}:{res_idx}"
        if chain_res_idx in chain_res_idx_seen:
            if chain_ending or chem_type_ending:
                current_chain_seq = []
            continue

        # keep track of the chain-residue ID pairs seen so far

        chain_res_idx_seen.add(chain_res_idx)

        # aggregate the residue sequences of each chain

        if not current_chain_seq:
            chain_residues.append(current_chain_seq)

        current_chain_seq.append((comp_details.id, res_letter, res_chem_type))

        # reset the current chain sequence if the next residue is either
        # not part of the current chain or is of a different chemical type

        if chain_ending or chem_type_ending:
            current_chain_seq = []

    return chain_residues


@typecheck
def add_atom_positions_to_mol(
    mol: Mol,
    atom_positions: np.ndarray,
    missing_atom_indices: Set[int],
) -> Mol:
    """Add atom positions to an RDKit molecule's first conformer while accounting for missing
    atoms."""
    assert len(missing_atom_indices) <= mol.GetNumAtoms(), (
        f"The number of missing atom positions ({len(missing_atom_indices)}) and atoms in the RDKit molecule ({mol.GetNumAtoms()}) are not reconcilable. "
        "Please ensure that these input features are all correctly paired."
    )

    # set missing atom positions to (0, 0, 0) while preserving the order of the remaining atoms

    missing_atom_counter = 0
    conf = mol.GetConformer()
    for atom_idx in range(mol.GetNumAtoms()):
        if atom_idx in missing_atom_indices:
            conf.SetAtomPosition(atom_idx, (0.0, 0.0, 0.0))
            missing_atom_counter += 1
        else:
            conf.SetAtomPosition(atom_idx, atom_positions[atom_idx - missing_atom_counter])

    Chem.SanitizeMol(mol)

    # set a property to indicate the atom positions that are missing

    mol.SetProp("missing_atom_indices", ",".join(map(str, sorted(missing_atom_indices))))

    return mol


@typecheck
def determine_bonds_using_rdkit(mol: Mol) -> bool:
    """Determine the bonds of an RDKit molecule using RDKit's bond determination algorithm.

    :param mol: The RDKit molecule for which to determine the bonds.
    :return: Whether the molecule's bonds were successfully determined.
    """
    bonds_inferred = False

    with StringIO() as buf:
        with redirect_stderr(buf):
            # redirect RDKit's stderr to a buffer to suppress warnings
            rdDetermineBonds.DetermineBonds(mol, allowChargedFragments=False)

            bonds_inferred = True

    return bonds_inferred


@typecheck
def create_mol_from_atom_positions_and_types(
    name: str,
    atom_positions: np.ndarray,
    element_types: List[str],
    missing_atom_indices: Set[int],
    neutral_stable_mol_hypothesis: bool = True,
    verbose: bool = False,
) -> Tuple[Mol, bool]:
    """Create an RDKit molecule from a NumPy array of atom positions and a list of their element
    types.

    :param name: The name of the molecule.
    :param atom_positions: A NumPy array of shape (num_atoms, 3) containing the 3D coordinates of
        each atom.
    :param element_types: A list of element symbols for each atom in the molecule.
    :param missing_atom_indices: A set of atom indices that are missing from the atom_positions
        array.
    :param neutral_stable_mol_hypothesis: Whether to convert radical electrons into explicit
        hydrogens based on the `PDB neutral stable molecule` hypothesis.
    :param verbose: Whether to log warnings when bond determination fails.
    :return: An RDKit molecule with the specified atom positions and element types and a boolean
        indicating whether the molecule's bonds were successfully determined.
    """
    if len(atom_positions) != len(element_types):
        raise ValueError("The length of atom_elements and xyz_coordinates must be the same.")

    # populate an empty editable molecule

    mol = Chem.RWMol()
    mol.SetProp("_Name", name)

    for element_type in element_types:
        # NOTE: `rdDetermineBonds.DetermineBonds()` only supports basic element symbols
        atom = Chem.Atom(element_type)
        mol.AddAtom(atom)

    # set 3D coordinates

    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (x, y, z) in enumerate(atom_positions):
        conf.SetAtomPosition(i, Point3D(x, y, z))

    # add the conformer to the molecule

    mol.AddConformer(conf)

    # block the RDKit logger

    blocker = rdBase.BlockLogs()

    # finalize molecule by inferring bonds

    bonds_inferred = False

    if mol.GetNumAtoms() <= RDKIT_BOND_DETERMINATION_MAX_ATOMS:
        try:
            bonds_inferred = determine_bonds_using_rdkit(mol)
        except Exception as e:
            if "Explicit valence" in str(e):
                # NOTE: for now, we allow for valence errors in the inferred bonds
                bonds_inferred = True
            elif verbose:
                logger.warning(
                    f"Failed to determine bonds for the input molecule {name} due to: {e}. Skipping bond determination."
                )

    # clean up the molecule

    mol = Chem.RemoveHs(mol, sanitize=False)
    Chem.SanitizeMol(mol, catchErrors=True)

    # based on the `PDB neutral stable molecule` hypothesis
    # (see https://github.com/rdkit/rdkit/issues/2683#issuecomment-2273998084),
    # convert radical electrons into explicit hydrogens

    if neutral_stable_mol_hypothesis:
        for a in mol.GetAtoms():
            if a.GetNumRadicalElectrons():
                a.SetNumExplicitHs(a.GetNumRadicalElectrons())
                a.SetNumRadicalElectrons(0)
            Chem.SanitizeMol(mol, catchErrors=True)

    # unblock the RDKit logger

    del blocker

    # set a property to indicate the atom positions that are missing

    mol.SetProp("missing_atom_indices", ",".join(map(str, sorted(missing_atom_indices))))

    return mol, bonds_inferred


@typecheck
def extract_canonical_molecules_from_biomolecule_chains(
    biomol: Biomolecule,
    chain_seqs: PDB_INPUT_CHAIN_SEQS,
    mol_keyname: str = "rdchem_mol",
    filepath: str | None = None,
    inference: bool = False,
    verbose: bool = False,
) -> Tuple[List[Mol], List[PDB_INPUT_RESIDUE_MOLECULE_TYPE]]:
    """Extract RDKit canonical molecules and their types for the residues of each `Biomolecule`
    chain.

    NOTE: Missing atom indices are marked as a comma-separated property string for each RDKit molecule
    and can be retrieved via `mol.GetProp('missing_atom_indices')`.
    """
    chain_index = biomol.chain_index
    residue_index = biomol.residue_index
    residue_types = biomol.restype
    atom_positions = biomol.atom_positions
    atom_mask = biomol.atom_mask.astype(bool)

    assert len(chain_index) == len(residue_index) == len(residue_types) == len(atom_positions), (
        f"The number of chain indices ({len(chain_index)}), residue indices ({len(residue_index)}), residue types ({len(residue_types)}), and atom positions ({len(atom_positions)}) do not match. "
        "Please ensure that these input features are correctly paired."
    )
    assert atom_positions.shape[:-1] == atom_mask.shape, (
        f"The number of atom positions ({atom_positions.shape[:-1]}) and atom masks ({atom_mask.shape}) do not match. "
        "Please ensure that these input features are correctly paired."
    )
    assert exists(CCD_COMPONENTS_SMILES), (
        f"The PDB Chemical Component Dictionary (CCD) components SMILES file {CCD_COMPONENTS_SMILES_FILEPATH} does not exist. "
        f"Please re-run this script after ensuring the preliminary CCD file {CCD_COMPONENTS_FILEPATH} has been downloaded according to this project's `README.md` file."
        f"After doing so, the SMILES file {CCD_COMPONENTS_SMILES_FILEPATH} will be cached locally and used for subsequent runs."
    )

    lig_res_constants = get_residue_constants("ligand")

    token_index = 0
    molecules = []
    molecule_types = []

    for seq in chain_seqs:
        chain_mols = []
        for res in seq:
            # map chemical types to protein, DNA, RNA, or ligand sequences

            res_chem_id, res_letter, res_chem_type = res

            if res_chem_type == "protein":
                seq_mapping = HUMAN_AMINO_ACIDS
            elif res_chem_type == "rna":
                seq_mapping = RNA_NUCLEOTIDES
            elif res_chem_type == "dna":
                seq_mapping = DNA_NUCLEOTIDES
            elif is_atomized_residue(res_chem_type):
                seq_mapping = CCD_COMPONENTS_SMILES
            else:
                raise ValueError(f"Unrecognized residue chemical type: {res_chem_type}")

            # Ligand and modified polymer (i.e., atomized) residues
            if is_atomized_residue(res_chem_type):
                # construct canonical molecule for post-mapping bond orders

                canonical_mol = None
                smile = (
                    res_chem_id
                    if inference and "mod" not in res_chem_type
                    else seq_mapping.get(res_chem_id, None)
                )

                if exists(smile):
                    try:
                        canonical_mol = mol_from_smile(smile, sanitize=inference)
                    except Exception as e:
                        if verbose and not inference:
                            logger.warning(
                                f"Failed to construct canonical RDKit molecule from the SMILES string for residue {res_chem_id} of {filepath} due to: {e}. "
                                "Skipping canonical molecule construction."
                            )
                elif verbose and not inference:
                    logger.warning(
                        f"Could not locate the PDB CCD's SMILES string for atomized residue {res_chem_id} of {filepath}. "
                        "Skipping canonical molecule construction."
                    )

                # inference-time atomized residue processing

                if inference:
                    assert exists(canonical_mol), (
                        f"Failed to construct canonical RDKit molecule for residue {res_chem_id} of {filepath} during inference. "
                        "Please ensure that the SMILES string for this atomized residue is available. "
                        "Skipping this inference example."
                    )

                    canonical_mol.SetProp("missing_atom_indices", "")

                    # duplicate the ETKDGv3 conformer at the next available conformer ID

                    canonical_mol.AddConformer(canonical_mol.GetConformer(0), assignId=True)

                    mol = canonical_mol

                # training-time atomized residue preparation

                else:
                    # find all atom positions and masks for the current atomized residue

                    res_residue_index = residue_index[token_index]
                    res_chain_index = chain_index[token_index]
                    res_ligand_atom_mask = (residue_index == res_residue_index) & (
                        chain_index == res_chain_index
                    )
                    res_atom_positions = atom_positions[res_ligand_atom_mask]
                    res_atom_mask = atom_mask[res_ligand_atom_mask]

                    # manually construct an RDKit molecule from the atomized residue's atom positions and types

                    res_atom_type_indices = np.where(res_atom_positions.any(axis=-1))[1]
                    res_atom_elements = [
                        lig_res_constants.element_types[idx].replace("ATM", "*")
                        for idx in res_atom_type_indices
                    ]

                    mol, bonds_inferred = create_mol_from_atom_positions_and_types(
                        # NOTE: for now, we construct molecules without referencing canonical
                        # SMILES strings, which means there are no missing molecule atoms by design
                        name=res_chem_id,
                        atom_positions=res_atom_positions[res_atom_mask],
                        element_types=res_atom_elements,
                        missing_atom_indices=set(),
                        verbose=verbose,
                    )

                    if bonds_inferred and exists(canonical_mol):
                        try:
                            mol = AllChem.AssignBondOrdersFromTemplate(canonical_mol, mol)
                        except Exception as e:
                            if verbose:
                                logger.warning(
                                    f"Failed to assign bond orders from the canonical atomized molecule for residue {res_chem_id} of {filepath} due to: {e}. "
                                    "Skipping bond order assignment."
                                )

                    # if bonds could be inferred, try to generate an ETKDGv3 conformation
                    # for the atomized residue and install its coordinates at the next
                    # available conformer ID

                    generated_mol = copy.deepcopy(mol)
                    conf_generated = False

                    if bonds_inferred:
                        try:
                            generated_mol = generate_conformation(generated_mol, sanitize=False)
                            generated_conf = generated_mol.GetConformer(0)
                            conf_generated = True

                        except Exception as e:
                            if verbose:
                                logger.warning(
                                    f"Failed to generate an ETKDGv3 conformation for the atomized residue {res_chem_id} of {filepath} due to: {e}. "
                                    "Setting the atom coordinates (for pairwise atom featurization) to zeros."
                                )

                            generated_conf = generated_mol.GetConformer(0)
                            for atom_idx in range(generated_mol.GetNumAtoms()):
                                generated_conf.SetAtomPosition(atom_idx, (0.0, 0.0, 0.0))

                    else:
                        generated_conf = generated_mol.GetConformer(0)
                        for atom_idx in range(generated_mol.GetNumAtoms()):
                            generated_conf.SetAtomPosition(atom_idx, (0.0, 0.0, 0.0))

                    if verbose and conf_generated:
                        logger.info(
                            f"Generated ETKDGv3 conformation for the atomized residue {res_chem_id} of {filepath}."
                        )

                    mol.AddConformer(generated_conf, assignId=True)

                token_index += mol.GetNumAtoms()

            # (Unmodified) polymer residues
            else:
                mol = copy.deepcopy(seq_mapping[res_letter][mol_keyname])

                res_type = residue_types[token_index]
                res_atom_mask = atom_mask[token_index]

                # duplicate the ETKDGv3 conformer at the next available conformer ID

                mol.AddConformer(mol.GetConformer(0), assignId=True)

                # map residue types to atom positions

                res_constants = get_residue_constants(res_chem_type.replace("protein", "peptide"))
                atom_mapping = res_constants.restype_atom47_to_compact_atom

                # start by finding all possible atoms that may be present in the residue

                res_unique_atom_mapping_indices = np.unique(
                    atom_mapping[res_type - res_constants.min_restype_num],
                    return_index=True,
                )[1]
                res_unique_atom_mapping = np.array(
                    [
                        # NOTE: `atom_mapping[res_type - res_constants.min_restype_num]` represents
                        # a residue's atoms using `atom47` indices and `atom14` ordering
                        atom_mapping[res_type - res_constants.min_restype_num][idx]
                        for idx in sorted(res_unique_atom_mapping_indices)
                    ]
                )

                # then find the subset of atoms that are *actually* present in the residue,
                # and gather the corresponding indices needed to remap these atoms
                # from `atom47` atom type indexing to compact atom type indexing
                # (e.g., mapping from `atom47` coordinates to `atom14` coordinates
                # uniquely for each type of amino acid residue)

                res_atom_mapping = atom_mapping[res_type - res_constants.min_restype_num][
                    res_atom_mask
                ]
                res_atom_mapping_set = set(res_atom_mapping)

                # ensure any missing atoms are accounted for in the atom positions during index remapping

                missing_atom_indices = {
                    idx
                    for idx in range(len(res_unique_atom_mapping))
                    if res_unique_atom_mapping[idx] not in res_atom_mapping_set
                }

                contiguous_res_atom_mapping = {
                    # NOTE: `np.unique` already sorts the unique values
                    value: index
                    for index, value in enumerate(np.unique(res_atom_mapping))
                }
                contiguous_res_atom_mapping = np.vectorize(contiguous_res_atom_mapping.get)(
                    res_atom_mapping
                )
                res_atom_positions = atom_positions[token_index][res_atom_mask][
                    contiguous_res_atom_mapping
                ]

                num_atom_positions = len(res_atom_positions) + len(missing_atom_indices)
                if num_atom_positions != mol.GetNumAtoms():
                    raise ValueError(
                        f"The number of (missing and present) atom positions ({num_atom_positions}) for residue {res_chem_id} does not match the number of atoms in the RDKit molecule ({mol.GetNumAtoms()}) for {filepath}. "
                        "Please ensure that these input features are correctly paired. Skipping this example."
                    )

                mol = add_atom_positions_to_mol(
                    mol,
                    res_atom_positions.reshape(-1, 3),
                    missing_atom_indices,
                )
                mol.SetProp("_Name", res_chem_id)
                token_index += 1

            assert (
                mol.GetNumConformers() == 2
            ), "One ground-truth and one generated conformer should be available."
            assert mol.GetConformer(0).Is3D(), "The ground-truth conformer should be 3D."
            assert mol.GetConformer(1).Is3D(), "The generated conformer should be 3D."

            # # visualize the generated conformer using the SDF file format
            # sdf_file = "molecule.sdf"
            # with Chem.SDWriter(sdf_file) as writer:
            #     conf = mol.GetConformer(1)
            #     writer.write(mol, confId=conf.GetId())

            chain_mols.append(mol)
            molecule_types.append(res_chem_type)

        molecules.extend(chain_mols)

    assert token_index == len(atom_positions), (
        f"The number of tokens matched to atom positions ({token_index}) does not match the number of atom positions ({len(atom_positions)}) available. "
        "Please ensure that these input features were correctly paired."
    )
    return molecules, molecule_types


@typecheck
def get_token_index_from_composite_atom_id(
    biomol: Biomolecule,
    chain_id: str,
    res_id: int,
    atom_name: str,
    atom_index: int,
    is_polymer_residue: bool,
) -> np.int64:
    """Get the token index (indices) of an atom (residue) in a biomolecule from its chain ID,
    residue ID, and atom name."""
    chain_mask = biomol.chain_id == chain_id
    res_mask = biomol.residue_index == res_id
    atom_mask = biomol.atom_name == atom_name

    if is_polymer_residue:
        return np.where(chain_mask & res_mask)[0][atom_index]
    else:
        return np.where(chain_mask & res_mask & atom_mask)[0][atom_index]


@typecheck
def find_mismatched_symmetry(
    asym_ids: np.ndarray,
    entity_ids: np.ndarray,
    sym_ids: np.ndarray,
    chemid: np.ndarray,
) -> bool:
    """Find mismatched symmetry in a biomolecule's asymmetry, entity, symmetry, and token chemical
    IDs.

    This function compares the chemical IDs of (related) regions with the same entity ID but
    different symmetry IDs. If the chemical IDs of these regions' matching asymmetric chain ID
    regions are not equal, then their symmetry is "mismatched".

    :param asym_ids: An array of asymmetric unit (i.e., chain) IDs for each token in the
        biomolecule.
    :param entity_ids: An array of entity IDs for each token in the biomolecule.
    :param sym_ids: An array of symmetry IDs for each token in the biomolecule.
    :param chemid: An array of chemical IDs for each token in the biomolecule.
    :return: A boolean indicating whether the symmetry IDs are mismatched.
    """
    assert len(asym_ids) == len(entity_ids) == len(sym_ids) == len(chemid), (
        f"The number of asymmetric unit IDs ({len(asym_ids)}), entity IDs ({len(entity_ids)}), symmetry IDs ({len(sym_ids)}), and chemical IDs ({len(chemid)}) do not match. "
        "Please ensure that these input features are correctly paired."
    )

    # Create a combined array of tuples (asym_id, entity_id, sym_id, index)
    combined = np.array(list(zip(asym_ids, entity_ids, sym_ids, range(len(entity_ids)))))

    # Group by entity_id
    grouped_by_entity = defaultdict(list)
    for entity, group in groupby(combined, key=lambda x: x[1]):
        grouped_by_entity[entity].extend(list(group))

    # Compare regions with the same entity_id but different sym_id
    for entity, group in grouped_by_entity.items():
        # Group by sym_id within each entity_id group
        grouped_by_sym = defaultdict(list)
        for _, _, sym, idx in group:
            grouped_by_sym[sym].append(idx)

        # Compare chemid sequences for the asym_id regions of different sym_id groups within the same entity_id group
        sym_ids_keys = list(grouped_by_sym.keys())
        for i in range(len(sym_ids_keys)):
            for j in range(i + 1, len(sym_ids_keys)):
                indices1 = grouped_by_sym[sym_ids_keys[i]]
                indices2 = grouped_by_sym[sym_ids_keys[j]]
                indices1_asym_ids = np.unique(asym_ids[indices1])
                indices2_asym_ids = np.unique(asym_ids[indices2])
                chemid_seq1 = chemid[np.isin(asym_ids, indices1_asym_ids)]
                chemid_seq2 = chemid[np.isin(asym_ids, indices2_asym_ids)]
                if len(chemid_seq1) != len(chemid_seq2) or not np.array_equal(
                    chemid_seq1, chemid_seq2
                ):
                    return True

    return False


@typecheck
def extract_polymer_sequence_from_chain_residues(
    chain_chemtype: List[int],
    chain_restype: List[int],
    unique_chain_residue_indices: List[int],
    ligand_chemtype_index: int = 3,
) -> str:
    """Extract a polymer sequence string from a chain's chemical types and residue types.

    :param chain_chemtype: A list of chemical types for each residue in the chain.
    :param chain_restype: A list of residue types for each residue in the chain.
    :param unique_chain_residue_indices: A list of unique residue indices in the chain.
    :param ligand_chemtype_index: The index of the ligand chemical type.
    :return: A polymer sequence string representing the chain's residues.
    """
    polymer_sequence = []

    for unique_chain_res_index in unique_chain_residue_indices:
        chemtype = chain_chemtype[unique_chain_res_index]
        restype = chain_restype[unique_chain_res_index]

        if chemtype < ligand_chemtype_index:
            rc = get_residue_constants(res_chem_index=chemtype)
            rc_restypes = rc.restypes + ["X"]
            polymer_sequence.append(rc_restypes[restype - rc.min_restype_num])

    return "".join(polymer_sequence)


@typecheck
def load_msa_from_msa_dir(
    msa_dir: str | None,
    file_id: str,
    chain_id_to_residue: Dict[str, Dict[str, List[int]]],
    max_msas_per_chain: int | None = None,
    distillation: bool = False,
    distillation_pdb_ids: List[str] | None = None,
    inference: bool = False,
    verbose: bool = False,
) -> FeatureDict:
    """Load MSA from a directory containing MSA files."""
    if verbose and (not_exists(msa_dir) or not os.path.exists(msa_dir)):
        logger.warning(
            f"{msa_dir} MSA directory does not exist. Dummy MSA features for each chain of file {file_id} will instead be loaded."
        )

    msas = {}
    msa_types = set()
    for chain_id in chain_id_to_residue:
        # Construct a length-1 MSA containing only the query sequence as a fallback.
        chain_chemtype = chain_id_to_residue[chain_id]["chemtype"]
        chain_restype = chain_id_to_residue[chain_id]["restype"]
        unique_chain_residue_indices = chain_id_to_residue[chain_id][
            "unique_chain_residue_indices"
        ]

        chain_sequences = [
            extract_polymer_sequence_from_chain_residues(
                chain_chemtype, chain_restype, unique_chain_residue_indices
            )
        ]
        chain_sequence = chain_sequences[0]
        chain_deletion_matrix = [[0] * len(sequence) for sequence in chain_sequences]
        chain_descriptions = ["101" for _ in chain_sequences]

        majority_msa_chem_type = statistics.mode(chain_chemtype)
        chain_msa_type = msa_parsing.get_msa_type(majority_msa_chem_type)

        is_unfiltered = [False for _ in chain_sequences]

        dummy_msa = msa_parsing.Msa(
            sequences=chain_sequences,
            deletion_matrix=chain_deletion_matrix,
            descriptions=chain_descriptions,
            msa_type=chain_msa_type,
            is_unfiltered=is_unfiltered,
        )

        pdb_ids = distillation_pdb_ids if distillation else [file_id]
        for pdb_id in pdb_ids:
            # NOTE: For distillation PDB examples, we may search for all possible MSAs with a chain's expected sequence length
            # (since we don't have a precise mapping from AFDB UniProt accession IDs to PDB chain IDs), whereas for the original
            # PDB examples, we can directly identify the corresponding MSAs.
            msa_fpaths = []
            msa_fpath_pattern = ""
            if exists(msa_dir):
                pdb_code = pdb_id.split("-assembly1")[0]
                msa_fpath_pattern = (
                    os.path.join(msa_dir, f"{pdb_code}_*", "a3m", "*.a3m*")
                    if distillation
                    else os.path.join(
                        msa_dir, pdb_code[1:3], f"{pdb_id}{chain_id}_{chain_msa_type}*.a*m*"
                    )
                )
                msa_fpaths = glob.glob(msa_fpath_pattern)

                if distillation and not msa_fpaths:
                    # NOTE: This handles phage distillation examples.
                    msa_fpath_pattern = os.path.join(msa_dir, pdb_code, "a3m", "*.a3m*")
                    msa_fpaths = glob.glob(msa_fpath_pattern)

            if not msa_fpaths:
                # NOTE: Too verbose in this context.
                # if verbose:
                #     logger.warning(
                #         f"Could not find MSAs matching the pattern {msa_fpath_pattern} for chain {chain_id} of file {file_id}. If no other MSAs are found, a dummy MSA will be installed for this chain."
                #     )
                continue

            # NOTE: For distillation PDB examples, if we have already found
            # enough MSAs for a chain, we should move on to the next chain.
            if (
                exists(max_msas_per_chain)
                and chain_id in msas
                and len(msas[chain_id]) >= max_msas_per_chain
            ):
                continue

            try:
                # NOTE: Each chain-specific MSA file contains alignments for all polymer residues in the chain,
                # but the chain's ligands are not included in the MSA file and therefore must be manually inserted
                # into the MSAs as unknown amino acid residues.
                chain_msas = []
                for msa_fpath in msa_fpaths:
                    open_ = (
                        gzip.open
                        if msa_fpath.endswith(".gz") and is_gzip_file(msa_fpath)
                        else open
                    )
                    unfiltered_msa = msa_fpath.endswith(".pair.a3m")
                    with open_(msa_fpath, "r") as f:
                        msa = f.read()
                        msa = msa_parsing.parse_a3m(
                            msa,
                            chain_msa_type,
                            unfiltered=unfiltered_msa,
                        )
                        if len(chain_sequence) == len(msa.sequences[0]):
                            chain_msas.append(msa)

                if not chain_msas:
                    raise ValueError(
                        f"Could not find any MSAs with the expected sequence length {len(chain_sequence)} for chain {chain_id} of file {file_id}"
                    )

                # Sort the MSAs by the presence of unfiltered sequences to facilitate MSA pairing.
                # If no unfiltered sequences are found, proportionally truncate the MSAs to the top
                # `max_msas_per_chain` MSAs per chain as specified.
                found_unfiltered_msas = any(all(msa.is_unfiltered) for msa in chain_msas)

                max_msas_per_chain_ = (
                    max_msas_per_chain - len(msas[chain_id])
                    if exists(max_msas_per_chain) and chain_id in msas
                    else max_msas_per_chain
                )
                max_msas_per_chain_proportional = (
                    max_msas_per_chain_ // len(chain_msas)
                    if exists(max_msas_per_chain_) and not found_unfiltered_msas
                    else None
                )
                if exists(max_msas_per_chain_proportional) and max_msas_per_chain_proportional < 1:
                    # NOTE: This implies we have reached the maximum number of MSAs for this PDB distillation chain.
                    continue

                if found_unfiltered_msas:
                    chain_msas = sorted(
                        chain_msas, key=lambda msa: all(msa.is_unfiltered), reverse=True
                    )

                for chain_msa in chain_msas:
                    chain_msa = (
                        chain_msa.truncate(
                            max_msas_per_chain_proportional
                        )  # Keep the top `max_msas_per_chain` MSAs per chain proportionally across all available chain MSAs.
                        if exists(max_msas_per_chain_proportional)
                        else chain_msa
                    )
                    msas[chain_id] = msas[chain_id] + chain_msa if chain_id in msas else chain_msa

            except Exception as e:
                if verbose:
                    logger.warning(
                        f"Failed to load MSAs for chain {chain_id} of file {file_id} due to: {e}. If no other MSAs are found, a dummy MSA will be installed for this chain."
                    )

        # Install a dummy MSA as necessary.
        if chain_id not in msas:
            if verbose:
                logger.warning(
                    f"Failed to load any MSAs for chain {chain_id} of file {file_id}. A dummy MSA will be installed for this chain."
                )
            msas[chain_id] = dummy_msa

        msa_types.add(msas[chain_id].msa_type)

    # Determine whether MSA pairing is possible prior to MSA featurization.
    entity_ids = {}
    for chain_id in msas:
        query_sequence = msas[chain_id].sequences[0]
        if query_sequence not in entity_ids:
            entity_ids[query_sequence] = len(entity_ids)

    is_monomer_or_homomer = len(entity_ids) == 1
    pair_msa_sequences = not is_monomer_or_homomer

    # NOTE: If MSA pairing is not possible, we should move all unfiltered sequences to the
    # end of the MSA to ensure that any filtered sequences are at the beginning of the MSA.
    found_any_unfiltered_msas = any(any(msas[chain_id].is_unfiltered) for chain_id in msas)
    if found_any_unfiltered_msas and not pair_msa_sequences:
        for chain_id in msas:
            msas[chain_id] = msas[chain_id].move_unfiltered_to_end()

    chains = make_msa_features(
        msas,
        chain_id_to_residue,
        num_msa_one_hot=NUM_MSA_ONE_HOT,
        tab_separated_alignment_headers=not distillation,
        inference=inference,
    )

    if pair_msa_sequences:
        try:
            chains = msa_pairing.copy_unpaired_features(chains)
            chains = msa_pairing.create_paired_features(chains)
        except Exception as e:
            if verbose:
                logger.warning(
                    f"Failed to pair MSAs for file {file_id} due to: {e}. Skipping MSA pairing."
                )
            pair_msa_sequences = False

    features = merge_chain_features(
        chains,
        pair_msa_sequences,
        max_msas_per_chain=max_msas_per_chain,
        verbose=verbose,
    )

    # Convert MSA type strings to integers.
    features["msa_types"] = [get_residue_molecule_type_index(msa_type) for msa_type in msa_types]

    # Remove (now-)unused MSA features.
    del features["entity_id"], features["is_unfiltered"]

    # Minimize data type sizes.
    features = {
        k: (
            (
                v.type(torch.uint8)
                if not torch.is_floating_point(v) or k == "has_deletion"
                else v.bfloat16()
            )
            if torch.is_tensor(v)
            else v
        )
        for k, v in features.items()
    }

    return features


@typecheck
def load_templates_from_templates_dir(
    templates_dir: str | None,
    mmcif_dir: str | None,
    file_id: str,
    chain_id_to_residue: Dict[str, Dict[str, List[int]]],
    max_templates_per_chain: int | None = None,
    num_templates_per_chain: int | None = None,
    kalign_binary_path: str | None = None,
    template_cutoff_date: datetime | None = None,
    randomly_sample_num_templates: bool = False,
    distillation: bool = False,
    distillation_pdb_ids: List[str] | None = None,
    distillation_max_pdb_ids: int = 3,
    verbose: bool = False,
) -> FeatureDict:
    """Load templates from a directory containing template PDB mmCIF files."""
    if verbose and (not_exists(templates_dir) or not os.path.exists(templates_dir)):
        logger.warning(
            f"{templates_dir} templates directory does not exist. Dummy template features for each chain of file {file_id} will instead be loaded."
        )

    if verbose and (not_exists(mmcif_dir) or not os.path.exists(mmcif_dir)):
        logger.warning(
            f"{mmcif_dir} mmCIF templates directory does not exist. Dummy template features for each chain of file {file_id} will instead be loaded."
        )

    templates = defaultdict(list)
    template_types = set()
    for chain_id in chain_id_to_residue:
        pdb_ids = distillation_pdb_ids[:distillation_max_pdb_ids] if distillation else [file_id]
        for pdb_id in pdb_ids:
            # NOTE: For distillation PDB examples, we may search for all possible chain templates
            # (since we don't have a precise mapping from AFDB UniProt accession IDs to PDB chain IDs),
            # whereas for the original PDB examples, we can directly identify the corresponding templates.
            template_fpaths = []
            template_fpath_pattern = ""
            if exists(templates_dir):
                pdb_code = pdb_id.split("-assembly1")[0]
                template_fpath_pattern = (
                    os.path.join(templates_dir, f"{pdb_code}_*", "hhr", "*.hhr")
                    if distillation
                    else os.path.join(templates_dir, pdb_code[1:3], f"{pdb_id}{chain_id}_*.m8")
                )
                template_fpaths = glob.glob(template_fpath_pattern)

                if distillation and not template_fpaths:
                    # NOTE: This handles phage distillation examples.
                    template_fpath_pattern = os.path.join(
                        os.path.join(templates_dir, pdb_code, "m8", "*.m8")
                    )
                    template_fpaths = glob.glob(template_fpath_pattern)

            if not template_fpaths:
                # NOTE: Too verbose in this context.
                # if verbose:
                #     logger.warning(
                #         f"Could not find templates matching the pattern {template_fpath_pattern} for chain {chain_id} of file {file_id}. If no other templates are found, a dummy template will be installed for this chain."
                #     )
                continue

            # NOTE: For distillation PDB examples, if we have already found
            # enough templates for a chain, we should move on to the next chain.
            if (
                exists(num_templates_per_chain)
                and chain_id in templates
                and len(templates[chain_id]) >= num_templates_per_chain
            ):
                continue

            try:
                # NOTE: Each chain-specific template file contains a template for all polymer residues in the chain,
                # but the chain's ligands are not included in the template file and therefore must be manually inserted
                # into the templates as unknown amino acid residues.
                for template_fpath in template_fpaths:
                    query_id = pdb_id.split("-assembly1")[0]

                    template_parsing_fn = (
                        template_parsing.parse_hhr
                        if template_fpath.endswith(".hhr")
                        else template_parsing.parse_m8
                    )

                    template_biomols = template_parsing_fn(
                        template_fpath,
                        query_id,
                        mmcif_dir,
                        max_templates=max_templates_per_chain,
                        num_templates=num_templates_per_chain,
                        template_cutoff_date=template_cutoff_date,
                        randomly_sample_num_templates=randomly_sample_num_templates,
                        verbose=verbose,
                    )

                    for template_biomol, template_type in template_biomols:
                        # NOTE: For distillation PDB examples, if we have already found
                        # enough templates for a chain, we should move on to the next chain.
                        if (
                            exists(num_templates_per_chain)
                            and chain_id in templates
                            and len(templates[chain_id]) >= num_templates_per_chain
                        ):
                            break
                        templates[chain_id].append((template_biomol, template_type))
                        template_types.add(template_type)

                    # NOTE: For distillation PDB examples, if we have already found
                    # enough templates for a chain, we should move on to the next chain.
                    if (
                        exists(num_templates_per_chain)
                        and len(templates[chain_id]) >= num_templates_per_chain
                    ):
                        break

            except Exception as e:
                if verbose:
                    logger.warning(
                        f"Failed to load templates for chain {chain_id} of file {file_id} due to: {e}. If no other templates are found, a dummy template will be installed for this chain."
                    )

        if chain_id not in templates:
            if verbose:
                logger.warning(
                    f"Could not find any templates for chain {chain_id} of file {file_id}. A dummy template will be installed for this chain."
                )
            templates[chain_id] = []
            continue

    features = make_template_features(
        templates,
        chain_id_to_residue,
        num_templates=num_templates_per_chain,
        kalign_binary_path=kalign_binary_path,
        verbose=verbose,
    )

    features["template_types"] = list(template_types) if template_types else ["protein"]

    return features


@typecheck
@wrapt_timeout_decorator.timeout(
    PDB_INPUT_TO_MOLECULE_INPUT_MAX_SECONDS_PER_INPUT,
    use_signals=True,
    # NOTE: we rely on reporting a `BaseException` to avoid catching `TimeoutError`
    # exceptions with the `except Exception` clauses within this function.
    timeout_exception=BaseException,
)
def build_pdb_input_features(
    pdb_input: PDBInput,
    biomol: Biomolecule | None = None,
    verbose: Literal["", "standard", "extra"] = "",
) -> FeatureDict:
    """Convert a PDBInput to a portable feature dictionary."""
    i = pdb_input

    # collect input metadata

    filepath = i.mmcif_filepath
    file_id = os.path.splitext(os.path.basename(filepath))[0] if exists(filepath) else None
    resolution = i.resolution

    file_has_parents = len(Path(filepath).parents) > 2
    is_md_frame = file_has_parents and Path(filepath).parents[2].stem == "md_data"
    is_phage_example = file_has_parents and Path(filepath).parents[2].stem == "phage_data"

    # acquire a `Biomolecule` object for the given `PDBInput`

    mmcif_release_date = None

    if not_exists(biomol) and exists(i.biomol):
        biomol = i.biomol
    else:
        # construct a `Biomolecule` object from the input PDB mmCIF file

        assert os.path.exists(filepath), f"PDB input file `{filepath}` does not exist."

        if verbose == "extra":
            logger.info(f"Loading biomolecule from {filepath}...")

        mmcif_object = mmcif_parsing.parse_mmcif_object(
            filepath=filepath,
            file_id=file_id,
            verbose=not (
                is_md_frame or is_phage_example
            ),  # NOTE: MD frames and phage examples do not have release dates attached, so we ignore their warnings
        )
        mmcif_resolution = extract_mmcif_metadata_field(mmcif_object, "resolution")
        mmcif_release_date = extract_mmcif_metadata_field(mmcif_object, "release_date")
        biomol = _from_mmcif_object(mmcif_object)

        if not_exists(resolution) and exists(mmcif_resolution):
            resolution = mmcif_resolution

        # NOTE: molecular dynamics distillation frames have been prefiltered according to their PDB target release dates,
        # and (predicted) phage distillation data does not require release date filtering
        if (
            i.distillation
            and not (is_md_frame or is_phage_example)
            and not_exists(mmcif_release_date)
        ):
            raise ValueError(
                f"The release date of the PDB distillation example {filepath} is missing. Please ensure that the release date is available for distillation set training."
            )

    # perform release date filtering as requested

    mmcif_release_date = (
        datetime.strptime(mmcif_release_date, "%Y-%m-%d") if exists(mmcif_release_date) else None
    )

    if (
        exists(mmcif_release_date)
        and exists(i.cutoff_date)
        and not (is_md_frame or is_phage_example)
    ):
        cutoff_date = datetime.strptime(i.cutoff_date, "%Y-%m-%d")
        assert (
            mmcif_release_date <= cutoff_date
        ), f"The release date ({mmcif_release_date}) of the PDB example {filepath} exceeds the accepted cutoff date ({cutoff_date}). Skipping this example."

    # perform token count filtering as requested

    biomol_num_tokens = len(biomol.atom_mask)

    if exists(i.max_length):
        assert (
            biomol_num_tokens <= i.max_length
        ), f"The number of tokens ({biomol_num_tokens}) in {filepath} exceeds the maximum initial length allowed ({i.max_length})."

    # parse biomolecule metadata

    resolution = tensor(resolution) if exists(resolution) else None

    biomol_chain_ids = list(
        dict.fromkeys(biomol.chain_id.tolist())
    )  # NOTE: we must maintain the order of unique chain IDs

    biomol_residue_index = (
        torch.from_numpy(biomol.residue_index) - 1
    )  # NOTE: `Biomolecule.residue_index` is 1-based originally
    biomol_chain_index = torch.from_numpy(biomol.chain_index)

    # create unique chain-residue index pairs to identify the first atom of each residue
    biomol_chain_residue_index = np.array(list(zip(biomol.chain_index, biomol.residue_index)))

    biomol_chain_id_to_residue = {
        chain_id: {
            "chemtype": biomol.chemtype[biomol.chain_id == chain_id].tolist(),
            "restype": biomol.restype[biomol.chain_id == chain_id].tolist(),
            "residue_index": biomol_residue_index[biomol.chain_id == chain_id].tolist(),
            "unique_chain_residue_indices": np.sort(
                np.unique(
                    biomol_chain_residue_index[biomol.chain_id == chain_id],
                    axis=0,
                    return_index=True,
                )[-1]
            ).tolist(),
        }
        for chain_id in biomol_chain_ids
    }

    # construct template features - NOTE: we may cache template features once, which would mean only the first sampled templates will be cached

    mmcif_dir = (
        i.distillation_template_mmcif_dir
        if i.distillation
        else (
            i.template_mmcif_dir
            if exists(i.template_mmcif_dir)
            else str(Path(i.mmcif_filepath).parent.parent)
        )
    )

    # use the template cutoff dates listed in the AF3 supplement's Section 2.4
    if i.training:
        template_cutoff_date = (
            datetime.strptime("2018-04-30", "%Y-%m-%d")
            if i.distillation
            else (mmcif_release_date - timedelta(days=60))
        )
    else:
        # NOTE: this is the template cutoff date for all inference tasks
        template_cutoff_date = datetime.strptime("2021-01-12", "%Y-%m-%d")

    if (
        exists(i.max_num_template_tokens)
        and biomol_num_tokens * i.num_templates_per_chain > i.max_num_template_tokens
    ):
        if verbose:
            logger.warning(
                f"The number of tokens ({biomol_num_tokens}) multiplied by the number of templates per structure ({i.num_templates_per_chain}) exceeds the maximum total number of template tokens {(i.max_num_template_tokens)}. "
                "Skipping curation of template features for this example."
            )
        template_features = {}
    else:
        if verbose == "extra":
            logger.info(f"Loading template features for {filepath}...")

        try:
            template_features = load_templates_from_templates_dir(
                # NOTE: if templates are not locally available, no template features will be used
                i.templates_dir,
                mmcif_dir,
                file_id,
                biomol_chain_id_to_residue,
                max_templates_per_chain=i.max_templates_per_chain,
                num_templates_per_chain=i.num_templates_per_chain,
                kalign_binary_path=i.kalign_binary_path,
                template_cutoff_date=template_cutoff_date,
                randomly_sample_num_templates=i.training,
                distillation=i.distillation,
                distillation_pdb_ids=i.distillation_pdb_ids,
                verbose=verbose in ("standard", "extra"),
            )
        except Exception as e:
            if verbose:
                logger.warning(
                    f"Failed to load template features for {filepath} due to: {e}. Skipping curation of template features for this example."
                )
            template_features = {}

    templates = template_features.get("templates")
    template_mask = template_features.get("template_mask")

    num_templates = template_mask.sum().item() if exists(template_mask) else 0
    template_types = template_features.get("template_types")

    if verbose == "extra" and exists(templates):
        logger.info(
            f"Finished loading {num_templates} {template_types} template features for {filepath}..."
        )

    # retrieve features directly available within the `Biomolecule` object

    # create unique chain-residue index pairs to identify the first atom of each residue
    # NOTE: since the structure may start with any chain ID (not necessarily `A`), we must maintain the consecutive order of unique chain IDs
    biomol_chain_index_ordering = {
        c_id: idx for idx, c_id in enumerate(dict.fromkeys(biomol.chain_index))
    }
    biomol_chain_index_ordered = np.vectorize(biomol_chain_index_ordering.get)(biomol.chain_index)
    _, biomol_unique_chain_residue_indices = np.unique(
        np.array(list(zip(biomol_chain_index_ordered, biomol.residue_index))),
        axis=0,
        return_index=True,
    )

    assert all(
        biomol_unique_chain_residue_indices == sorted(biomol_unique_chain_residue_indices)
    ), "The unique chain-residue indices are not correctly ordered."

    # retrieve molecule_ids from the `Biomolecule` object, where here it is the mapping of 33 possible residue types
    # `proteins (20) | unknown protein (1) | rna (4) | unknown RNA (1) | dna (4) | unknown DNA (1) | gap (1) | metal ion (1)`,
    # where ligands are mapped to the unknown protein category (i.e., residue index 20)
    # NOTE: below, we will install values for our new (dedicated) type for metal ions
    molecule_ids = torch.from_numpy(biomol.restype)

    # retrieve is_molecule_types from the `Biomolecule` object, which is a boolean tensor of shape [*, 5]
    # is_protein | is_rna | is_dna | is_ligand | is_metal_ion
    # this is needed for their special diffusion loss
    # NOTE: below, we will install values for our new (dedicated) one-hot class for metal ions
    n_one_hot = IS_MOLECULE_TYPES
    is_molecule_types = F.one_hot(torch.from_numpy(biomol.chemtype), num_classes=n_one_hot).bool()

    # manually derive remaining features using the `Biomolecule` object

    if verbose == "extra":
        logger.info(f"Manually extracting remaining features of biomolecule for {filepath}...")

    # extract chain residues from the `Biomolecule` object
    chem_comp_table = {comp.id: comp for comp in biomol.chem_comp_table}
    chem_comp_details = [chem_comp_table[chemid] for chemid in biomol.chemid]
    chain_residues = extract_chain_residues_from_biomolecule_chemical_components(
        biomol,
        chem_comp_details,
    )
    chain_sequences = [
        "".join(
            [
                (res_chem_id if res_chem_type == "ligand" else res_letter)
                for res_chem_id, res_letter, res_chem_type in chain_residues
            ]
        )
        for chain_residues in chain_residues
    ]

    # retrieve RDKit canonical molecules for the residues of each chain,
    # and insert the input (and generated) atom coordinates into the canonical molecules
    molecules, molecule_types = extract_canonical_molecules_from_biomolecule_chains(
        biomol,
        chain_residues,
        filepath=filepath,
        inference=i.inference,
        verbose=verbose in ("standard", "extra"),
    )

    if verbose == "extra":
        logger.info(f"Collecting molecule-level features of biomolecule for {filepath}...")

    # collect pooling lengths and atom-wise molecule types for each molecule,
    # along with a token-wise one-hot tensor indicating whether each molecule is modified
    # and, if so, which type of modification it has (e.g., peptide vs. RNA modification);
    # also gather exhaustive atom permutations for each molecule
    molecule_idx = 0
    token_pool_lens = []
    molecule_atom_types = []
    is_molecule_mod = []
    molecule_atom_perms = []
    for mol, mol_type in zip(molecules, molecule_types):
        num_atoms = mol.GetNumAtoms()
        is_mol_mod_type = [False for _ in range(DEFAULT_NUM_MOLECULE_MODS)]
        molecule_atom_perms.append(get_atom_perms(mol))
        if is_atomized_residue(mol_type):
            # NOTE: in the paper, they treat each atom of the ligand and modified polymer residues as a token
            token_pool_lens.extend([1] * num_atoms)
            molecule_atom_types.extend([mol_type] * num_atoms)

            molecule_type_row_idx = slice(molecule_idx, molecule_idx + num_atoms)

            # NOTE: we reset all type annotations e.g., since ions are initially considered ligands
            is_molecule_types[molecule_type_row_idx] = False

            if mol_type == "ligand" and num_atoms == 1:
                # NOTE: we manually set the molecule ID of ions to a dedicated category
                molecule_ids[molecule_idx] = MOLECULE_METAL_ION_ID
                is_mol_type_index = IS_METAL_ION_INDEX
            elif mol_type == "ligand":
                is_mol_type_index = IS_LIGAND_INDEX
            elif mol_type == "mod_protein":
                is_mol_type_index = IS_PROTEIN_INDEX
                is_mol_mod_type_index = 0
            elif mol_type == "mod_rna":
                is_mol_type_index = IS_RNA_INDEX
                is_mol_mod_type_index = 1
            elif mol_type == "mod_dna":
                is_mol_type_index = IS_DNA_INDEX
                is_mol_mod_type_index = 2
            else:
                raise ValueError(f"Unrecognized molecule type: {mol_type}")

            is_molecule_types[molecule_type_row_idx, is_mol_type_index] = True

            if "mod" in mol_type:
                is_mol_mod_type[is_mol_mod_type_index] = True
            is_molecule_mod.extend([is_mol_mod_type] * num_atoms)

            molecule_idx += num_atoms
        else:
            token_pool_lens.append(num_atoms)
            molecule_atom_types.append(mol_type)
            is_molecule_mod.append(is_mol_mod_type)
            molecule_idx += 1

    token_pool_lens = tensor(token_pool_lens)
    is_molecule_mod = tensor(is_molecule_mod)

    if verbose == "extra":
        logger.info(f"Collecting token-level features of biomolecule for {filepath}...")

    # collect frame, token center, distogram, and source-target atom indices for each token
    atom_indices_for_frame = []
    is_ligand_frame = []
    molecule_atom_indices = []
    token_center_atom_indices = []
    distogram_atom_indices = []
    src_tgt_atom_indices = []

    current_atom_index = 0
    current_res_index = -1
    current_chain_index = -1

    for mol_type, atom_mask, chemid, res_index, res_chain_index in zip(
        molecule_atom_types,
        biomol.atom_mask,
        biomol.chemid,
        biomol.residue_index,
        biomol.chain_index,
    ):
        residue_constants = get_residue_constants(
            mol_type.replace("protein", "peptide").replace("mod_", "")
        )

        if mol_type == "protein":
            entry = HUMAN_AMINO_ACIDS[residue_constants.restype_3to1.get(chemid, "X")]
        elif mol_type == "rna":
            entry = RNA_NUCLEOTIDES[residue_constants.restype_3to1.get(chemid, "X")]
        elif mol_type == "dna":
            entry = DNA_NUCLEOTIDES[residue_constants.restype_3to1.get(chemid, "X")]

        if is_atomized_residue(mol_type):
            # collect indices for each ligand and modified polymer residue token (i.e., atom)
            if current_res_index == res_index and current_chain_index == res_chain_index:
                current_atom_index += 1
            else:
                current_atom_index = 0
                current_res_index = res_index
                current_chain_index = res_chain_index

            # NOTE: we have to dynamically determine the token center atom index for atomized residues
            token_center_atom_index = np.where(atom_mask)[0][0]

            num_mol_atoms = len(
                # NOTE: For modified polymer residues, in `molecule_to_atom_input()`, we later on
                # subset `src_tgt_atom_indices` to the first atom in each molecule, so we can safely
                # collect duplicate src-tgt pairs here
                np.where(
                    (biomol.residue_index == res_index) & (biomol.chain_index == res_chain_index)
                )[0]
            )

            atom_indices_for_frame.append(None)
            is_ligand_frame.append(True)
            molecule_atom_indices.append(current_atom_index)
            token_center_atom_indices.append(token_center_atom_index)
            distogram_atom_indices.append(current_atom_index)
            src_tgt_atom_indices.append([0, num_mol_atoms - 1])
        else:
            # collect indices for each polymer residue token
            atom_indices_for_frame.append(entry["three_atom_indices_for_frame"])
            is_ligand_frame.append(False)
            molecule_atom_indices.append(entry["token_center_atom_idx"])
            token_center_atom_indices.append(entry["token_center_atom_idx"])
            distogram_atom_indices.append(entry["distogram_atom_idx"])
            src_tgt_atom_indices.append([entry["first_atom_idx"], entry["last_atom_idx"]])

            # keep track of the current residue index and atom index for subsequent atomized tokens
            current_atom_index = 0
            current_res_index = res_index

    is_ligand_frame = tensor(is_ligand_frame)
    molecule_atom_indices = tensor(molecule_atom_indices)
    token_center_atom_indices = tensor(token_center_atom_indices)
    distogram_atom_indices = tensor(distogram_atom_indices)
    src_tgt_atom_indices = tensor(src_tgt_atom_indices)

    if verbose == "extra":
        logger.info(f"Constructing additional molecule features of biomolecule for {filepath}...")

    # constructing the additional_molecule_feats
    # which is in turn used to derive relative positions

    # residue_index - an arange that restarts at 1 for each chain - reuse biomol.residue_index here
    # token_index   - just an arange over len(biomol.atom_mask)
    # asym_id       - unique id for each chain of a biomolecule - reuse biomol.chain_index here
    # entity_id     - unique id for each biomolecule sequence
    # sym_id        - unique id for each chain of the same biomolecule sequence

    # entity ids
    unrepeated_entity_sequences = defaultdict(int)
    for entity_sequence in chain_sequences:
        if entity_sequence in unrepeated_entity_sequences:
            continue
        unrepeated_entity_sequences[entity_sequence] = len(unrepeated_entity_sequences)

    entity_idx = 0
    entity_id_counts = []
    unrepeated_entity_ids = []
    for entity_sequence, entity_residues in zip(chain_sequences, chain_residues):
        chain_entity_len = 0
        for residue in entity_residues:
            _, _, res_chem_type = residue

            entity_mol = molecules[entity_idx]
            chain_entity_len += (
                entity_mol.GetNumAtoms() if is_atomized_residue(res_chem_type) else 1
            )
            entity_idx += 1

        entity_id_counts.append(chain_entity_len)
        unrepeated_entity_ids.append(unrepeated_entity_sequences[entity_sequence])

    biomol_entity_ids = repeat_interleave(tensor(unrepeated_entity_ids), tensor(entity_id_counts))

    # sym ids
    unrepeated_sym_ids = []
    unrepeated_sym_sequences = defaultdict(int)
    for entity_sequence in chain_sequences:
        unrepeated_sym_ids.append(unrepeated_sym_sequences[entity_sequence])
        if entity_sequence in unrepeated_sym_sequences:
            unrepeated_sym_sequences[entity_sequence] += 1

    biomol_sym_ids = repeat_interleave(tensor(unrepeated_sym_ids), tensor(entity_id_counts))

    # concat for all of additional_molecule_feats
    additional_molecule_feats = torch.stack(
        (
            biomol_residue_index,
            torch.arange(biomol_num_tokens),
            biomol_chain_index,
            biomol_entity_ids,
            biomol_sym_ids,
        ),
        dim=-1,
    )

    if verbose == "extra":
        logger.info(f"Handling missing atoms of biomolecule for {filepath}...")

    # handle missing atom indices
    molecules_missing_atom_indices = [
        [int(idx) for idx in mol.GetProp("missing_atom_indices").split(",") if idx]
        for mol in molecules
    ]

    missing_atom_indices = []
    missing_token_indices = []

    for mol_miss_atom_indices, mol, mol_type in zip(
        molecules_missing_atom_indices, molecules, molecule_types
    ):
        mol_miss_atom_indices = default(mol_miss_atom_indices, [])
        mol_miss_atom_indices = tensor(mol_miss_atom_indices, dtype=torch.int16)

        missing_atom_indices.append(mol_miss_atom_indices.clone())
        if is_atomized_residue(mol_type):
            missing_token_indices.extend([mol_miss_atom_indices for _ in range(mol.GetNumAtoms())])
        else:
            missing_token_indices.append(mol_miss_atom_indices)

    assert len(molecules) == len(
        missing_atom_indices
    ), f"Number of molecules and missing atom indices do not match: {len(molecules)} != {len(missing_atom_indices)}"
    assert (
        len(missing_token_indices) == biomol_num_tokens
    ), f"Number of tokens and missing token indices do not match: {biomol_num_tokens} != {len(missing_token_indices)}"

    mol_total_atoms = sum([mol.GetNumAtoms() for mol in molecules])
    num_missing_atom_indices = sum(
        len(mol_miss_atom_indices) for mol_miss_atom_indices in missing_atom_indices
    )
    num_present_atoms = mol_total_atoms - num_missing_atom_indices
    assert num_present_atoms == int(
        biomol.atom_mask.sum()
    ), f"Number of present atoms does not match: {num_present_atoms} != {biomol.atom_mask.sum()}"

    # build offsets for all atom indices

    # derive `atom_lens` based on `one_token_per_atom`, for ligands and modified biomolecules
    atoms_per_molecule = tensor([mol.GetNumAtoms() for mol in molecules])
    ones = torch.ones_like(atoms_per_molecule)

    # `is_molecule_mod` can either be
    # 1. Bool['n'], in which case it will only be used for determining `one_token_per_atom`, or
    # 2. Bool['n num_mods'], where it will be passed to MegaFold for molecule modification embeds
    if is_molecule_mod.ndim == 2:
        is_molecule_any_mod = is_molecule_mod[biomol_unique_chain_residue_indices].any(dim=-1)
    else:
        is_molecule_any_mod = is_molecule_mod[biomol_unique_chain_residue_indices]

    # get `one_token_per_atom`
    # default to what the paper did, which is ligands and any modified biomolecule
    is_ligand = is_molecule_types[biomol_unique_chain_residue_indices][..., IS_LIGAND_INDEX]
    one_token_per_atom = (
        is_ligand | is_molecule_any_mod
    )  # NOTE: ions are implicitly handled below with `ones`

    assert len(molecules) == len(
        one_token_per_atom
    ), f"Number of molecules and one token per atom do not match: {len(molecules)} != {len(one_token_per_atom)}"

    # derive the number of repeats needed to expand molecule lengths to token lengths
    token_repeats = torch.where(one_token_per_atom, atoms_per_molecule, ones)

    if verbose == "extra":
        logger.info(f"Installing token bonds of biomolecule for {filepath}...")

    # derive `is_first_mol_in_chains` and `is_chainable_biomolecules` - needed for constructing `token_bonds

    # need the token_asym_id (to keep track of each molecule for each chain ascending) as well as `is_protein | is_dna | is_rna | is_ligand | is_metal_ion` for is_molecule_types (chainable biomolecules)
    # will do a single bond from a peptide or nucleotide to the one before. derive a `is_first_mol_in_chain` from `token_asym_ids`

    token_asym_ids = F.pad(biomol_chain_index, (1, 0), value=-1)
    is_first_mol_in_chains = (token_asym_ids[1:] - token_asym_ids[:-1]) != 0
    is_chainable_biomolecules = is_molecule_types[..., IS_BIOMOLECULE_INDICES].any(dim=-1)

    # construct token bonds, which will be linearly connected for proteins
    # and nucleic acids, but for ligands and modified polymer residues
    # will have their atomic bond matrix (as ligands and modified polymer
    # residues are atom resolution)

    num_tokens = token_repeats.sum().item()
    token_bonds = torch.zeros(num_tokens, num_tokens).bool()

    assert (
        biomol_num_tokens == num_tokens
    ), f"Number of tokens and token bonds do not match: {biomol_num_tokens} != {num_tokens}"

    offset = 0

    for (
        mol,
        mol_is_chainable_biomolecule,
        mol_is_mod,
        mol_is_first_mol_in_chain,
        mol_is_one_token_per_atom,
    ) in zip(
        molecules,
        is_chainable_biomolecules[biomol_unique_chain_residue_indices],
        is_molecule_mod[biomol_unique_chain_residue_indices],
        is_first_mol_in_chains[biomol_unique_chain_residue_indices],
        one_token_per_atom,
    ):
        num_atoms = mol.GetNumAtoms()

        if mol_is_chainable_biomolecule and not mol_is_mod.any() and not mol_is_first_mol_in_chain:
            token_bonds[offset, offset - 1] = True
            token_bonds[offset - 1, offset] = True

        elif mol_is_one_token_per_atom:
            has_bond = get_mol_has_bond(mol)

            if exists(has_bond) and has_bond.numel() > 0:
                num_atoms = mol.GetNumAtoms()
                row_col_slice = slice(offset, offset + num_atoms)
                token_bonds[row_col_slice, row_col_slice] = has_bond

        offset += num_atoms if mol_is_one_token_per_atom else 1

    if verbose == "extra":
        logger.info(f"Checking for mismatched symmetry of biomolecule for {filepath}...")

    # ensure mmCIF polymer-ligand (i.e., protein/RNA/DNA-ligand) and ligand-ligand bonds
    # (and bonds less than 2.4 Å) are installed in `MoleculeInput` during training only
    # per the AF3 supplement (Table 5, `token_bonds`)
    bond_atom_indices = defaultdict(int)
    biomol_bonds = biomol.bonds if exists(biomol.bonds) else []

    for bond in biomol_bonds:
        # ascertain whether homomeric (e.g., bonded ligand) symmetry is preserved,
        # which determines whether or not we use the mmCIF bond inputs (AF3 Section 5.1)
        if not i.training or find_mismatched_symmetry(
            biomol.chain_index,
            biomol_entity_ids.numpy(),
            biomol_sym_ids.numpy(),
            biomol.chemid,
        ):
            continue

        # determine bond type

        # NOTE: in this context, modified polymer residues will be treated as ligands
        ptnr1_is_polymer = any(
            bond.ptnr1_auth_comp_id in rc.restype_3to1
            for rc in {amino_acid_constants, rna_constants, dna_constants}
        )
        ptnr2_is_polymer = any(
            bond.ptnr2_auth_comp_id in rc.restype_3to1
            for rc in {amino_acid_constants, rna_constants, dna_constants}
        )
        ptnr1_is_ligand = not ptnr1_is_polymer
        ptnr2_is_ligand = not ptnr2_is_polymer
        is_polymer_ligand_bond = (ptnr1_is_polymer and ptnr2_is_ligand) or (
            ptnr1_is_ligand and ptnr2_is_polymer
        )
        is_ligand_ligand_bond = ptnr1_is_ligand and ptnr2_is_ligand

        # conditionally install bond

        if (
            is_polymer_ligand_bond
            or is_ligand_ligand_bond
            or (mmcif_parsing._is_set(bond.pdbx_dist_value) and float(bond.pdbx_dist_value) < 2.4)
        ):
            ptnr1_atom_id = (
                f"{bond.ptnr1_auth_asym_id}:{bond.ptnr1_auth_seq_id}:{bond.ptnr1_label_atom_id}"
            )
            ptnr2_atom_id = (
                f"{bond.ptnr2_auth_asym_id}:{bond.ptnr2_auth_seq_id}:{bond.ptnr2_label_atom_id}"
            )
            try:
                row_idx = get_token_index_from_composite_atom_id(
                    biomol,
                    bond.ptnr1_auth_asym_id,
                    int(bond.ptnr1_auth_seq_id),
                    bond.ptnr1_label_atom_id,
                    bond_atom_indices[ptnr1_atom_id],
                    ptnr1_is_polymer,
                )
            except Exception as e:
                if verbose:
                    logger.warning(
                        f"Could not find a matching token index for token1 {ptnr1_atom_id} due to: {e}. "
                        "Skipping installing the current bond associated with this token."
                    )
                continue
            try:
                col_idx = get_token_index_from_composite_atom_id(
                    biomol,
                    bond.ptnr2_auth_asym_id,
                    int(bond.ptnr2_auth_seq_id),
                    bond.ptnr2_label_atom_id,
                    bond_atom_indices[ptnr2_atom_id],
                    ptnr2_is_polymer,
                )
            except Exception as e:
                if verbose:
                    logger.warning(
                        f"Could not find a matching token index for token2 {ptnr1_atom_id} due to: {e}. "
                        "Skipping installing the current bond associated with this token."
                    )
                continue
            token_bonds[row_idx, col_idx] = True
            token_bonds[col_idx, row_idx] = True
            bond_atom_indices[ptnr1_atom_id] += 1
            bond_atom_indices[ptnr2_atom_id] += 1

    # create unique chain-residue index hashes for quick token-to-molecule lookups

    chain_residue_ids = create_uid_tensor(
        biomol_chain_index.unsqueeze(0), biomol_residue_index.unsqueeze(0)
    ).squeeze(0)

    # record input features

    input_features = {
        "file_id": file_id,
        "mmcif_release_date": str(mmcif_release_date),
        "resolution": resolution,
        "biomol": biomol,
        "biomol_num_tokens": biomol_num_tokens,
        "biomol_chain_ids": biomol_chain_ids,
        "templates": templates,
        "template_mask": template_mask,
        "num_templates": num_templates,
        "template_types": template_types,
        "molecule_ids": molecule_ids.type(torch.uint8),
        "is_molecule_types": is_molecule_types,
        "chain_sequences": chain_sequences,
        "molecules": molecules,
        "molecule_types": molecule_types,
        "token_pool_lens": token_pool_lens.type(torch.int16),
        "is_molecule_mod": is_molecule_mod,
        "molecule_atom_perms": molecule_atom_perms,
        "atom_indices_for_frame": atom_indices_for_frame,
        "is_ligand_frame": is_ligand_frame,
        "molecule_atom_indices": molecule_atom_indices.type(torch.uint8),
        "token_center_atom_indices": token_center_atom_indices.type(torch.uint8),
        "distogram_atom_indices": distogram_atom_indices.type(torch.uint8),
        "src_tgt_atom_indices": src_tgt_atom_indices.type(torch.int16),
        "additional_molecule_feats": additional_molecule_feats.type(torch.int32),
        "missing_atom_indices": missing_atom_indices,
        "missing_token_indices": missing_token_indices,
        "token_repeats": token_repeats.type(torch.int16),
        "token_bonds": token_bonds,
        "chain_residue_ids": chain_residue_ids,
    }

    return input_features


@typecheck
def pdb_input_to_molecule_input(
    pdb_input: PDBInput,
    biomol: Biomolecule | None = None,
    verbose: Literal["", "standard", "extra"] = "",
) -> MoleculeInput:
    """Convert a PDBInput to a MoleculeInput."""
    i = pdb_input

    filepath = i.mmcif_filepath
    file_id = os.path.splitext(os.path.basename(filepath))[0] if exists(filepath) else None
    resolution = i.resolution

    msa_cache_path = None
    input_cache_path = None

    file_has_parents = len(Path(filepath).parents) > 2
    is_md_frame = file_has_parents and Path(filepath).parents[2].stem == "md_data"

    if exists(i.msa_cache_dir):
        # NOTE: for molecular dynamics (MD) frames, we can reuse the same cached MSA features for each frame of the same trajectory
        file_code = file_id.split("-")[1] if i.distillation else file_id[1:3]
        msa_cache_path = os.path.join(
            i.msa_cache_dir, file_code, f"{file_code if is_md_frame else file_id}.npy.gz"
        )

    if exists(i.input_cache_dir):
        file_code = file_id.split("-")[1] if i.distillation else file_id[1:3]
        input_cache_path = os.path.join(i.input_cache_dir, file_code, f"{file_id}.npy.gz")

    if i.cache_msa_only and exists(msa_cache_path) and os.path.exists(msa_cache_path):
        # skip pre-caching MSA features if the cache file already exists
        return

    if i.cache_input_only and exists(input_cache_path) and os.path.exists(input_cache_path):
        # skip pre-caching input features if the cache file already exists
        return

    # cache the (pre-cropping) input features for each example if possible
    print("building input_features")
    if (
        exists(input_cache_path)
        and os.path.exists(input_cache_path)
        and os.access(input_cache_path, os.R_OK)
    ):
        with gzip.GzipFile(input_cache_path, "r") as f:
            input_features = np.load(f, allow_pickle=True).item()
            print("taking input_features from cache")
    else:
        print("building input_features from scratch")
        input_features = build_pdb_input_features(i, biomol=biomol, verbose=verbose)

        if exists(input_cache_path) and os.access(i.input_cache_dir, os.W_OK):
            os.makedirs(os.path.dirname(input_cache_path), exist_ok=True)
            with gzip.GzipFile(input_cache_path, "w") as f:
                np.save(f, input_features)

        if i.cache_input_only:
            return

    # upcast integer input features to long tensors

    input_features = {
        key: (
            value.type(torch.long)
            if key
            in {
                "molecule_ids",
                "token_pool_lens",
                "molecule_atom_indices",
                "token_center_atom_indices",
                "distogram_atom_indices",
                "src_tgt_atom_indices",
                "additional_molecule_feats",
                "token_repeats",
            }
            else value
        )
        for key, value in input_features.items()
    }
    input_features["missing_atom_indices"] = [
        missing_atom_indices.type(torch.long)
        for missing_atom_indices in input_features["missing_atom_indices"]
    ]
    input_features["missing_token_indices"] = [
        missing_token_indices.type(torch.long)
        for missing_token_indices in input_features["missing_token_indices"]
    ]

    # extract select input metadata prior to any cropping

    biomol = input_features["biomol"]
    biomol_chain_ids = input_features["biomol_chain_ids"]

    biomol_num_tokens = len(biomol.atom_mask)
    biomol_chain_index = torch.from_numpy(biomol.chain_index)

    templates = input_features["templates"]
    template_mask = input_features["template_mask"]
    resolution = input_features["resolution"]

    # sample a chain (or chain pair) for distillation examples

    if i.distillation and (
        not_exists(i.chains)
        or (exists(i.chains) and not (exists(i.chains[0]) or exists(i.chains[1])))
    ):
        chain_id_1, chain_id_2 = i.chains if exists(i.chains) else (None, None)

        if len(biomol_chain_ids) == 1:
            chain_id_1 = biomol_chain_ids[0]
        elif (
            len(biomol_chain_ids) > 1
            and random.random() < i.distillation_multimer_sampling_ratio  # nosec
        ):
            chain_id_1, chain_id_2 = random.sample(biomol_chain_ids, 2)  # nosec
        elif len(biomol_chain_ids) > 1:
            chain_id_1 = random.choice(biomol_chain_ids)  # nosec
        else:
            raise ValueError(
                f"Could not find any chain IDs for the distillation example {file_id}."
            )

        i.chains = (chain_id_1, chain_id_2)

    # map (sampled) chain IDs to indices prior to cropping

    chains = (None, None)

    if exists(i.chains):
        chain_id_1, chain_id_2 = i.chains
        biomol_chain_id_to_idx = {
            chain_id: chain_idx
            for (chain_id, chain_idx) in zip(biomol.chain_id, biomol.chain_index)
        }
        # NOTE: we have to manually nullify a chain ID value
        # e.g., if an empty string is passed in as a "null" chain ID
        if chain_id_1:
            chain_id_1 = biomol_chain_id_to_idx[chain_id_1]
        else:
            chain_id_1 = None
        if chain_id_2:
            chain_id_2 = biomol_chain_id_to_idx[chain_id_2]
        else:
            chain_id_2 = None
        chains = (chain_id_1, chain_id_2)

        if verbose == "extra":
            logger.info(f"Selected initial chain IDs {i.chains} ({chains}) for example {file_id}.")

    # construct multiple sequence alignment (MSA) and template features prior to cropping

    msa_dir = i.msa_dir
    max_msas_per_chain = i.max_msas_per_chain

    if (
        exists(i.max_num_msa_tokens)
        and biomol_num_tokens * i.max_msas_per_chain > i.max_num_msa_tokens
    ):
        msa_dir = None
        max_msas_per_chain = 1

        if verbose:
            logger.warning(
                f"The number of tokens ({biomol_num_tokens}) multiplied by the maximum number of MSAs per structure ({i.max_msas_per_chain}) exceeds the maximum total number of MSA tokens {(i.max_num_msa_tokens)}. "
                "Skipping curation of MSA features for this example by installing a dummy MSA for each chain."
            )

    if verbose == "extra":
        logger.info(f"Loading MSA features for {filepath}...")

    print("building msas")
    # cache the (deterministic, pre-cropping) MSA features for each example if possible
    if (
        exists(msa_cache_path)
        and os.path.exists(msa_cache_path)
        and os.access(msa_cache_path, os.R_OK)
    ):
        with gzip.GzipFile(msa_cache_path, "r") as f:
            msa_features = np.load(f, allow_pickle=True).item()
            print("taking msas from cache")
            #print(msa_features)
    else:
        print("building msas from scratch")
        # build required biomolecule features for MSA construction
        biomol_chain_residue_index = np.array(list(zip(biomol.chain_index, biomol.residue_index)))
        biomol_chain_id_to_residue = {
            chain_id: {
                "chemtype": biomol.chemtype[biomol.chain_id == chain_id].tolist(),
                "restype": biomol.restype[biomol.chain_id == chain_id].tolist(),
                "residue_index": (biomol.residue_index - 1)[biomol.chain_id == chain_id].tolist(),
                "unique_chain_residue_indices": np.sort(
                    np.unique(
                        biomol_chain_residue_index[biomol.chain_id == chain_id],
                        axis=0,
                        return_index=True,
                    )[-1]
                ).tolist(),
            }
            for chain_id in biomol_chain_ids
        }

        try:
            msa_features = load_msa_from_msa_dir(
                # NOTE: if MSAs are not locally available, no MSA features will be used
                msa_dir,
                file_id,
                biomol_chain_id_to_residue,
                max_msas_per_chain=max_msas_per_chain,
                distillation=i.distillation,
                distillation_pdb_ids=i.distillation_pdb_ids,
                inference=i.inference,
                verbose=verbose in ("standard", "extra"),
            )
        except Exception as e:
            if verbose:
                logger.warning(
                    f"Failed to load MSA features for {filepath} due to: {e}. A dummy MSA will be installed for each chain."
                )

            msa_features = load_msa_from_msa_dir(
                None,
                file_id,
                biomol_chain_id_to_residue,
                max_msas_per_chain=1,
                distillation=i.distillation,
                distillation_pdb_ids=i.distillation_pdb_ids,
                verbose=verbose in ("standard", "extra"),
            )

        if exists(msa_cache_path) and os.access(i.msa_cache_dir, os.W_OK):
            os.makedirs(os.path.dirname(msa_cache_path), exist_ok=True)
            with gzip.GzipFile(msa_cache_path, "w") as f:
                #print("Saving to cache msa_features")
                np.save(f, msa_features)

        if i.cache_msa_only:
            return

    msa_features = make_msa_mask(msa_features)

    msa = msa_features.get("msa")
    msa_row_mask = msa_features.get("msa_row_mask")

    has_deletion = msa_features.get("has_deletion")
    deletion_value = msa_features.get("deletion_value")

    profile = msa_features.get("profile")
    deletion_mean = msa_features.get("deletion_mean")

    msa_types = (
        [
            get_residue_molecule_type(res_chem_index=msa_type)
            for msa_type in msa_features["msa_types"]
        ]
        if "msa_types" in msa_features
        else None
    )

    # collect additional MSA and token features
    # 0: has_deletion (msa)
    # 1: deletion_value (msa)
    # 2: profile (token)
    # 3: deletion_mean (token)

    additional_msa_feats = None
    additional_token_feats = None

    all_msa_features_exist = all(
        exists(feat)
        for feat in [msa, msa_row_mask, has_deletion, deletion_value, profile, deletion_mean]
    )

    assert all_msa_features_exist, "All MSA features must be derived for each example."
    assert (
        msa.shape[-1] == biomol_num_tokens
    ), f"The number of tokens in the MSA ({msa.shape[-1]}) does not match the number of tokens in the biomolecule ({biomol_num_tokens}). "

    additional_msa_feats = torch.stack(
        [
            has_deletion.float(),
            deletion_value.float(),
        ],
        dim=-1,
    )

    additional_token_feats = torch.cat(
        [
            profile.float(),
            deletion_mean[:, None].float(),
        ],
        dim=-1,
    )

    # convert the MSA into a one-hot representation
    msa = make_one_hot(msa.long(), NUM_MSA_ONE_HOT)
    msa_row_mask = msa_row_mask.bool()

    num_msas = msa_row_mask.sum().item() if exists(msa_row_mask) else 0

    if verbose == "extra":
        logger.info(f"Finished loading {num_msas} {msa_types} MSA features for {filepath}...")

    # crop the `Biomolecule` object during training, validation, and testing (but not inference)

    if not i.inference:
        assert exists(
            i.cropping_config
        ), "A cropping configuration must be provided during training."
        try:
            assert exists(i.chains), "Chain IDs must be provided for cropping during training."
            chain_id_1, chain_id_2 = i.chains

            if verbose == "extra":
                logger.info(f"Cropping biomolecule for {filepath}...")

            cropped_biomol, crop_masks = biomol.crop(
                contiguous_weight=i.cropping_config["contiguous_weight"],
                spatial_weight=i.cropping_config["spatial_weight"],
                spatial_interface_weight=i.cropping_config["spatial_interface_weight"],
                n_res=i.cropping_config["n_res"],
                chain_1=chain_id_1 if chain_id_1 else None,
                chain_2=chain_id_2 if chain_id_2 else None,
            )

            # # maybe visualize the (uncropped) biomolecule
            # uncropped_mmcif_string = to_mmcif(biomol, file_id)
            # uncropped_output_filepath = os.path.join("outputs", "dl_samples", f"{file_id}_uncropped.cif")
            # os.makedirs(os.path.dirname(uncropped_output_filepath), exist_ok=True)
            # with open(uncropped_output_filepath, "w") as f:
            #     f.write(uncropped_mmcif_string)

            # # maybe visualize the (cropped) biomolecule
            # cropped_mmcif_string = to_mmcif(cropped_biomol, file_id)
            # cropped_output_filepath = os.path.join("outputs", "dl_samples", f"{file_id}_cropped.cif")
            # os.makedirs(os.path.dirname(cropped_output_filepath), exist_ok=True)
            # with open(cropped_output_filepath, "w") as f:
            #     f.write(cropped_mmcif_string)

            # update token metadata after cropping
            biomol_num_tokens = len(cropped_biomol.atom_mask)
            biomol_chain_ids = list(
                dict.fromkeys(cropped_biomol.chain_id.tolist())
            )  # NOTE: we must maintain the order of unique chain IDs

            biomol_residue_index = (
                torch.from_numpy(cropped_biomol.residue_index) - 1
            )  # NOTE: `Biomolecule.residue_index` is 1-based originally
            biomol_chain_index = torch.from_numpy(cropped_biomol.chain_index)

            # build singular crop mask
            crop_mask = np.concatenate(crop_masks)
            crop_mask_tensor = torch.from_numpy(crop_mask)

            # crop molecule-level features - NOTE: we use chain-residue index uniqueness to filter molecules
            chain_residue_ids = list(
                dict.fromkeys(input_features["chain_residue_ids"].tolist())
            )  # preserve molecule ordering
            input_features["chain_residue_ids"] = create_uid_tensor(
                biomol_chain_index.unsqueeze(0), biomol_residue_index.unsqueeze(0)
            ).squeeze(0)
            cropped_chain_residue_ids = set(input_features["chain_residue_ids"].tolist())

            input_features["molecules"] = [
                mol
                for idx, mol in enumerate(input_features["molecules"])
                if chain_residue_ids[idx] in cropped_chain_residue_ids
            ]
            input_features["molecule_atom_perms"] = [
                perm
                for idx, perm in enumerate(input_features["molecule_atom_perms"])
                if chain_residue_ids[idx] in cropped_chain_residue_ids
            ]
            input_features["missing_atom_indices"] = [
                i
                for idx, i in enumerate(input_features["missing_atom_indices"])
                if chain_residue_ids[idx] in cropped_chain_residue_ids
            ]
            input_features["token_repeats"] = tensor(
                [
                    r
                    for idx, r in enumerate(input_features["token_repeats"])
                    if chain_residue_ids[idx] in cropped_chain_residue_ids
                ]
            )

            # crop token-level features
            input_features["molecule_ids"] = input_features["molecule_ids"][crop_mask_tensor]
            input_features["is_molecule_types"] = input_features["is_molecule_types"][
                crop_mask_tensor
            ]
            input_features["token_pool_lens"] = input_features["token_pool_lens"][crop_mask_tensor]
            input_features["is_molecule_mod"] = input_features["is_molecule_mod"][crop_mask_tensor]
            input_features["is_ligand_frame"] = input_features["is_ligand_frame"][crop_mask_tensor]
            input_features["molecule_atom_indices"] = input_features["molecule_atom_indices"][
                crop_mask_tensor
            ]
            input_features["token_center_atom_indices"] = input_features[
                "token_center_atom_indices"
            ][crop_mask_tensor]
            input_features["distogram_atom_indices"] = input_features["distogram_atom_indices"][
                crop_mask_tensor
            ]
            input_features["src_tgt_atom_indices"] = input_features["src_tgt_atom_indices"][
                crop_mask_tensor
            ]
            input_features["additional_molecule_feats"] = input_features[
                "additional_molecule_feats"
            ][crop_mask_tensor]
            input_features["token_bonds"] = input_features["token_bonds"][crop_mask_tensor][
                :, crop_mask_tensor
            ]

            input_features["atom_indices_for_frame"] = [
                idx
                for i, idx in enumerate(input_features["atom_indices_for_frame"])
                if crop_mask[i]
            ]
            input_features["missing_token_indices"] = [
                idx
                for i, idx in enumerate(input_features["missing_token_indices"])
                if crop_mask[i]
            ]

            # crop template features
            if exists(templates):
                templates = templates[:, crop_mask_tensor][:, :, crop_mask_tensor]

                assert templates.shape[-3] == biomol_num_tokens, (
                    f"The number of tokens in the templates ({templates.shape[-3]}) does not match the number of tokens in the cropped biomolecule ({biomol_num_tokens}). "
                    "Please ensure that the template features are correctly shaped."
                )

            # crop MSA features
            if exists(msa):
                msa = msa[:, crop_mask_tensor]

                additional_token_feats = additional_token_feats[crop_mask_tensor]
                additional_msa_feats = additional_msa_feats[:, crop_mask_tensor]

                assert msa.shape[-2] == biomol_num_tokens, (
                    f"The number of tokens in the MSA ({msa.shape[-2]}) does not match the number of tokens in the cropped biomolecule ({biomol_num_tokens}). "
                    "Please ensure that the MSA features are correctly shaped."
                )

            # update sampled chain indices
            uncropped_chain_id_to_cropped_chain_idx = {
                uncropped_chain_id: cropped_chain_idx.item()
                for (uncropped_chain_id, cropped_chain_idx) in zip(
                    biomol.chain_id[crop_mask], cropped_biomol.chain_index
                )
            }

            if chain_id_1:
                chain_idx_1 = uncropped_chain_id_to_cropped_chain_idx.get(chain_id_1)
            else:
                chain_idx_1 = None
            if chain_id_2:
                chain_idx_2 = uncropped_chain_id_to_cropped_chain_idx.get(chain_id_2)
            else:
                chain_idx_2 = None

            # NOTE: e.g., when contiguously cropping structures, the sampled chains
            # may be missing from the cropped structure, in which case we must
            # re-sample new chains specifically for validation model selection scoring
            if not_exists(chain_idx_1) and not_exists(chain_idx_2):
                input_chain_id_1, input_chain_id_2 = i.chains

                if (
                    exists(input_chain_id_1)
                    and exists(input_chain_id_2)
                    and len(uncropped_chain_id_to_cropped_chain_idx) > 1
                ):
                    chain_idx_1, chain_idx_2 = sorted(
                        random.sample(list(uncropped_chain_id_to_cropped_chain_idx.values()), 2)
                    )  # nosec
                else:
                    chain_idx_1 = random.choice(
                        list(uncropped_chain_id_to_cropped_chain_idx.values())
                    )  # nosec

            chains = (chain_idx_1, chain_idx_2)

            # update biomolecule after cropping
            biomol = cropped_biomol

        except Exception as e:
            raise ValueError(f"Failed to crop the biomolecule for input {file_id} due to: {e}")

    # filter based on token count

    if exists(i.min_length):
        assert (
            biomol_num_tokens >= i.min_length
        ), f"The number of tokens ({biomol_num_tokens}) in {filepath} is less than the minimum (cropped) length allowed ({i.min_length})."

    # filter based on atom count

    num_atoms = int(biomol.atom_mask.sum())

    if exists(i.max_num_atoms) and num_atoms > i.max_num_atoms:
        if verbose:
            logger.warning(
                f"The number of atoms ({num_atoms}) exceeds the maximum number of atoms allowed ({i.max_num_atoms}). "
                "Skipping the current input."
            )
        raise ValueError(
            f"The number of atoms ({num_atoms}) exceeds the maximum number of atoms allowed ({i.max_num_atoms})."
        )

    # unpack (potentially cropped) input features

    molecules = input_features["molecules"]
    atom_indices_for_frame = input_features["atom_indices_for_frame"]
    is_ligand_frame = input_features["is_ligand_frame"]
    molecule_atom_indices = input_features["molecule_atom_indices"]
    token_center_atom_indices = input_features["token_center_atom_indices"]
    distogram_atom_indices = input_features["distogram_atom_indices"]
    token_repeats = input_features["token_repeats"]
    chain_residue_ids = input_features["chain_residue_ids"]

    biomol_atom_mask = torch.from_numpy(biomol.atom_mask)
    biomol_atom_positions = torch.from_numpy(biomol.atom_positions)
    atoms_per_molecule = tensor([mol.GetNumAtoms() for mol in molecules])

    molecule_atom_perms = [
        perm for mol_perm_list in input_features["molecule_atom_perms"] for perm in mol_perm_list
    ]  # flatten to atoms

    # (re)create unique chain-residue index pairs to identify the first atom of each residue
    # NOTE: since the structure may start with any chain ID (not necessarily `A`), we must maintain the consecutive order of unique chain IDs
    biomol_chain_index_ordering = {
        c_id: idx for idx, c_id in enumerate(dict.fromkeys(biomol.chain_index))
    }
    biomol_chain_index_ordered = np.vectorize(biomol_chain_index_ordering.get)(biomol.chain_index)
    _, biomol_unique_chain_residue_indices = np.unique(
        np.array(list(zip(biomol_chain_index_ordered, biomol.residue_index))),
        axis=0,
        return_index=True,
    )

    assert all(
        biomol_unique_chain_residue_indices == sorted(biomol_unique_chain_residue_indices)
    ), "The unique chain-residue indices are not correctly ordered."

    if verbose == "extra":
        logger.info(f"Building ligand frames of biomolecule for {filepath}...")

    # handle frames for ligands (AF3 Supplement, Section 4.3.2)

    chain_id_to_token_center_atom_positions = {
        # NOTE: Here, we improvise by using only the token center atom
        # positions of tokens in the same chain to derive ligand frames
        chain_id: torch.gather(
            biomol_atom_positions[biomol.chain_id == chain_id],
            1,
            token_center_atom_indices[biomol.chain_id == chain_id][..., None, None].expand(
                -1, -1, 3
            ),
        ).squeeze(1)
        for chain_id in biomol_chain_ids
    }
    chain_id_to_token_center_atom_mask = {
        chain_id: torch.gather(
            biomol_atom_mask[biomol.chain_id == chain_id],
            1,
            token_center_atom_indices[biomol.chain_id == chain_id].unsqueeze(-1),
        ).squeeze(1)
        for chain_id in biomol_chain_ids
    }

    chain_id_to_first_token_indices = {
        chain_id: np.where(biomol.chain_id == chain_id)[0].min() for chain_id in biomol_chain_ids
    }

    chain_id_to_frames = {
        chain_id: get_frames_from_atom_pos(
            atom_pos=chain_id_to_token_center_atom_positions[chain_id],
            mask=chain_id_to_token_center_atom_mask[chain_id].bool(),
            filter_colinear_pos=True,
        )
        + chain_id_to_first_token_indices[chain_id]
        for chain_id in biomol_chain_ids
    }
    token_index_to_frames = {
        token_index: frame
        for token_index, frame in enumerate(
            frame for chain_frames in chain_id_to_frames.values() for frame in chain_frames
        )
    }

    for token_index in range(len(atom_indices_for_frame)):
        if not_exists(atom_indices_for_frame[token_index]):
            atom_indices_for_frame[token_index] = tuple(
                token_index_to_frames[token_index].tolist()
            )

    atom_indices_for_frame = tensor(atom_indices_for_frame)

    if verbose == "extra":
        logger.info(f"Offsetting atom indices of biomolecule for {filepath}...")

    # craft offsets for all atom indices

    atom_indices_offsets = repeat_interleave(
        exclusive_cumsum(atoms_per_molecule), token_repeats, dim=0
    )

    # craft ligand frame offsets
    atom_indices_for_ligand_frame = torch.zeros_like(atom_indices_for_frame)
    for ligand_frame_index in torch.where(is_ligand_frame)[0]:
        if (atom_indices_for_frame[ligand_frame_index] == -1).any():
            atom_indices_for_ligand_frame[ligand_frame_index] = atom_indices_for_frame[
                ligand_frame_index
            ]
            continue

        global_atom_indices = torch.gather(
            atom_indices_offsets, 0, atom_indices_for_frame[ligand_frame_index]
        )

        is_ligand_frame_atom = torch.gather(
            is_ligand_frame, 0, atom_indices_for_frame[ligand_frame_index]
        )
        local_token_center_atom_offsets = torch.where(
            # NOTE: ligand frames are atomized, so for them we have to
            # offset the atom indices using (ligand) residue atom-sequential
            # offsets rather than fixed token center atom indices
            is_ligand_frame_atom,
            torch.gather(molecule_atom_indices, 0, atom_indices_for_frame[ligand_frame_index]),
            torch.gather(token_center_atom_indices, 0, atom_indices_for_frame[ligand_frame_index]),
        )

        atom_indices_for_ligand_frame[ligand_frame_index] = (
            global_atom_indices + local_token_center_atom_offsets
        )

    # offset only positive atom indices
    distogram_atom_indices = offset_only_positive(distogram_atom_indices, atom_indices_offsets)
    molecule_atom_indices = offset_only_positive(molecule_atom_indices, atom_indices_offsets)
    atom_indices_for_frame = torch.where(
        is_ligand_frame.unsqueeze(-1),
        atom_indices_for_ligand_frame,
        offset_only_positive(atom_indices_for_frame, atom_indices_offsets[..., None]),
    )

    if verbose == "extra":
        logger.info(f"Constructing atom-level features of biomolecule for {filepath}...")

    # construct atom-level features

    # construct ground-truth atom positions from canonical molecules after instantiating their 3D conformers
    atom_pos = torch.from_numpy(
        np.concatenate([mol.GetConformer(0).GetPositions() for mol in molecules]).astype(
            np.float32
        )
    )
    num_atoms = atom_pos.shape[0]

    # replace ground-truth coordinates with ETKDGv3-generated coordinates to minimize
    # input feature distribution shift between training and inference contexts
    for molecule in molecules:
        generated_positions = molecule.GetConformer(1).GetPositions()
        molecule.GetConformer(0).SetPositions(generated_positions)
        molecule.RemoveConformer(1)

    # sanity-check the atom indices
    if not (-1 <= distogram_atom_indices.min() <= distogram_atom_indices.max() < num_atoms):
        raise ValueError(
            f"Invalid distogram atom indices found in `pdb_input_to_molecule_input()` for {filepath}: {distogram_atom_indices}"
        )
    if not (-1 <= molecule_atom_indices.min() <= molecule_atom_indices.max() < num_atoms):
        raise ValueError(
            f"Invalid molecule atom indices found in `pdb_input_to_molecule_input()` for {filepath}: {molecule_atom_indices}"
        )
    if not (-1 <= atom_indices_for_frame.min() <= atom_indices_for_frame.max() < num_atoms):
        raise ValueError(
            f"Invalid atom indices for frame found in `pdb_input_to_molecule_input()` for {filepath}: {atom_indices_for_frame}"
        )

    # create atom_parent_ids using the `Biomolecule` object, which governs in the atom
    # encoder / decoder which atom attends to which, where a design choice is made such
    # that mmCIF author chain indices are directly adopted to group atoms belonging to
    # the same (author-denoted) chain
    atom_parent_ids = tensor(
        [
            biomol.chain_index[biomol_unique_chain_residue_indices][res_index]
            for res_index in range(len(molecules))
            for _ in range(molecules[res_index].GetNumAtoms())
        ]
    )

    # craft experimentally resolved labels per the AF2 supplement's Section 1.9.10
    resolved_labels = None

    if exists(resolution):
        if verbose == "extra":
            logger.info(f"Building resolved labels of biomolecule for {filepath}...")

        is_resolved_label = ((resolution >= 0.1) & (resolution <= 3.0)).item()
        resolved_labels = torch.full((num_atoms,), is_resolved_label, dtype=torch.long)
    elif i.distillation:
        # NOTE: distillation examples are assigned a minimal resolution label to enable confidence head scoring
        resolution = torch.tensor(0.1)

    # craft optional pairwise token constraints

    token_constraints = None

    if exists(i.constraints) and len(i.constraints):
        if verbose == "extra":
            logger.info(f"Constructing token constraints of biomolecule for {filepath}...")

        token_pos = torch.gather(
            biomol_atom_positions,
            1,
            token_center_atom_indices[..., None, None].expand(-1, -1, 3),
        ).squeeze(1)

        token_constraints = get_token_constraints(
            constraints=i.constraints,
            constraints_ratio=i.constraints_ratio,
            training=i.training,
            inference=i.inference,
            token_pos=token_pos,
            token_parent_ids=biomol_chain_index,
            verbose=verbose in ("standard", "extra"),
        )

    # determine the number of ligands present in the structure

    ligand_chain_residue_ids = chain_residue_ids[
        input_features["is_molecule_types"][..., IS_LIGAND]
    ]
    num_ligands = default(i.num_ligands, len(ligand_chain_residue_ids.unique()))

    # create molecule input

    molecule_input = MoleculeInput(
        molecules=molecules,
        molecule_token_pool_lens=input_features["token_pool_lens"].tolist(),
        molecule_ids=input_features["molecule_ids"],
        additional_molecule_feats=input_features["additional_molecule_feats"],
        is_molecule_types=input_features["is_molecule_types"],
        src_tgt_atom_indices=input_features["src_tgt_atom_indices"].tolist(),
        token_bonds=input_features["token_bonds"],
        molecule_atom_perms=molecule_atom_perms,
        is_molecule_mod=input_features["is_molecule_mod"],
        molecule_atom_indices=molecule_atom_indices,
        distogram_atom_indices=distogram_atom_indices,
        atom_indices_for_frame=atom_indices_for_frame,
        missing_atom_indices=input_features["missing_atom_indices"],
        missing_token_indices=input_features["missing_token_indices"],
        atom_parent_ids=atom_parent_ids,
        additional_msa_feats=default(
            additional_msa_feats, torch.zeros(num_msas, biomol_num_tokens, 2)
        ),
        additional_token_feats=default(additional_token_feats, torch.zeros(biomol_num_tokens, 33)),
        templates=templates,
        msa=msa,
        atom_pos=atom_pos,
        token_constraints=token_constraints,
        first_mol_token_indices=biomol_unique_chain_residue_indices,
        template_mask=template_mask,
        msa_mask=msa_row_mask,
        resolved_labels=resolved_labels,
        resolution=resolution,
        chains=chains,
        num_ligands=num_ligands,
        filepath=filepath,
        example_source=i.example_source,
        add_atom_ids=i.add_atom_ids,
        add_atompair_ids=i.add_atompair_ids,
        directed_bonds=i.directed_bonds,
        extract_atom_feats_fn=i.extract_atom_feats_fn,
        extract_atompair_feats_fn=i.extract_atompair_feats_fn,
        custom_atoms=i.custom_atoms,
        custom_bonds=i.custom_bonds,
    )

    if verbose == "extra":
        logger.info(f"Finished building MoleculeInput of biomolecule for {filepath}...")

    return molecule_input


@typecheck
def compute_pocket_constraint(
    token_dists: Float["n n"],  # type: ignore
    token_parent_ids: Int[" n"],  # type: ignore
    unique_token_parent_ids: Int[" p"],  # type: ignore
    theta_p_range: Tuple[float, float],
    geom_distr: torch.distributions.Geometric,
) -> Float["n n"]:  # type: ignore
    """Compute the pairwise token pocket constraint.

    :param token_dists: The pairwise token distances.
    :param token_parent_ids: The token parent (i.e., chain) IDs.
    :param unique_token_parent_ids: The unique token parent IDs.
    :param theta_p_range: The range of `theta_p` values to use for the pocket constraint.
    :param geom_distr: The geometric distribution to use for sampling.
    :return: The pairwise token pocket constraint.
    """

    # sample chain ID and distance threshold for pocket constraint

    sampled_target_parent_id = unique_token_parent_ids[
        torch.randint(0, len(unique_token_parent_ids), (1,))
    ]

    sampled_theta_p = random.uniform(*theta_p_range)  # nosec
    token_dists_mask = (token_dists > 0.0) & (token_dists < sampled_theta_p)

    # restrict to inter-chain distances between any non-sampled chain and the sampled chain

    # token_parent_mask = einx.not_equal("i, j -> i j", token_parent_ids, token_parent_ids)
    token_parent_mask = token_parent_ids.unsqueeze(0) != token_parent_ids.unsqueeze(1)
    token_parent_mask[:, token_parent_ids != sampled_target_parent_id] = False

    # sample pocket constraints

    pairwise_token_mask = token_dists_mask & token_parent_mask
    pairwise_token_sampled_mask = (geom_distr.sample(pairwise_token_mask.shape) == 1).squeeze(-1)
    pairwise_token_mask[~pairwise_token_sampled_mask] = False

    # for simplicity, define the pocket constraint as a diagonalized pairwise matrix

    pairwise_token_constraint = torch.diag(pairwise_token_mask.any(-1)).float()

    return pairwise_token_constraint


@typecheck
def compute_contact_constraint(
    token_dists: Float["n n"],  # type: ignore
    theta_d_range: Tuple[float, float],
    geom_distr: torch.distributions.Geometric,
) -> Float["n n"]:  # type: ignore
    """Compute the pairwise token contact constraint.

    :param token_dists: The pairwise token distances.
    :param theta_d_range: The range of `theta_d` values to use for the contact constraint.
    :param geom_distr: The geometric distribution to use for sampling.
    :return: The pairwise token contact constraint.
    """

    # sample distance threshold for contact constraint

    sampled_theta_d = random.uniform(*theta_d_range)  # nosec
    token_dists_mask = (token_dists > 0.0) & (token_dists < sampled_theta_d)

    # restrict to inter-token distances while sampling contact constraints

    pairwise_token_mask = token_dists_mask
    pairwise_token_sampled_mask = (geom_distr.sample(pairwise_token_mask.shape) == 1).squeeze(-1)
    pairwise_token_mask[~pairwise_token_sampled_mask] = False

    # define the contact constraint as a pairwise matrix

    pairwise_token_constraint = pairwise_token_mask.float()

    return pairwise_token_constraint


@typecheck
def compute_docking_constraint(
    token_dists: Float["n n"],  # type: ignore
    token_parent_ids: Int[" n"],  # type: ignore
    unique_token_parent_ids: Int[" p"],  # type: ignore
    dist_bins: Float["bins"],  # type: ignore
    geom_distr: torch.distributions.Geometric,
) -> Float["n n bins"]:  # type: ignore
    """Compute the pairwise token docking constraint.

    :param token_dists: The pairwise token distances.
    :param token_parent_ids: The token parent (i.e., chain) IDs.
    :param unique_token_parent_ids: The unique token parent IDs.
    :param dist_bins: The distance bins to use for the docking constraint.
    :param geom_distr: The geometric distribution to use for sampling.
    :return: The pairwise token docking constraint as a one-hot encoding.
    """

    # partition chains into two groups

    group1_mask = torch.isin(
        token_parent_ids, unique_token_parent_ids[: len(unique_token_parent_ids) // 2]
    )
    group2_mask = torch.isin(
        token_parent_ids, unique_token_parent_ids[len(unique_token_parent_ids) // 2 :]
    )

    # create masks for inter-group distances (group1 vs group2)

    inter_group_mask = (group1_mask.unsqueeze(1) & group2_mask.unsqueeze(0)) | (
        group2_mask.unsqueeze(1) & group1_mask.unsqueeze(0)
    )

    # apply binning to the pairwise distances while sampling docking constraints

    token_distogram = distance_to_dgram(token_dists, dist_bins).float()
    num_bins = token_distogram.shape[-1]

    pairwise_token_sampled_mask = (geom_distr.sample(token_dists.shape) == 1).expand(
        -1, -1, num_bins
    )
    token_distogram[~pairwise_token_sampled_mask] = 0.0

    # assign one-hot encoding for distances that are in inter-group positions

    pairwise_token_constraint = torch.zeros((*token_dists.shape, num_bins), dtype=torch.float32)
    pairwise_token_constraint[inter_group_mask] = token_distogram[inter_group_mask]

    return pairwise_token_constraint


@typecheck
def get_token_constraints(
    constraints: INPUT_CONSTRAINTS,
    constraints_ratio: float,
    training: bool,
    inference: bool,
    token_pos: Float["n 3"],  # type: ignore
    token_parent_ids: Int[" n"],  # type: ignore
    theta_p_range: Tuple[float, float] = (6.0, 20.0),
    theta_d_range: Tuple[float, float] = (6.0, 30.0),
    dist_bins: Float["bins"] = torch.tensor([0.0, 4.0, 8.0, 16.0]),  # type: ignore
    p: float = 1.0 / 3.0,
    verbose: bool = False,
) -> Float["n n dac"]:  # type: ignore
    """Construct pairwise token constraints for the given constraint strings and ratio.

    NOTE: The `pocket`, `contact`, and `docking` constraints are inspired by the Chai-1 model.

    :param constraints: The constraints to use.
    :param constraints_ratio: The constraints ratio to use during training.
    :param training: Whether the model is training.
    :param inference: Whether the model is in inference.
    :param token_pos: The token center atom positions.
    :param token_parent_ids: The token parent (i.e., chain) IDs.
    :param theta_p_range: The range of `theta_p` values to use for the pocket constraint.
    :param theta_d_range: The range of `theta_d` values to use for the contact constraint.
    :param dist_bins: The distance bins to use for the docking constraint.
    :param p: The probability of success for the geometric distribution.
    :param verbose: Whether to log additional information.
    :return: The pairwise token constraints.
    """
    assert (
        inference and isinstance(constraints, dict)
    ) or not inference, (
        f"Expected to receive a dictionary of pairwise constraints during inference."
    )
    assert 0 < constraints_ratio <= 1, "The constraints ratio must be in the range (0, 1]."
    assert (
        0 < theta_p_range[0] < theta_p_range[1]
    ), "The `theta_p_range` must be monotonically increasing."

    unique_token_parent_ids = tensor(list(dict.fromkeys(token_parent_ids.tolist())))
    num_chains = unique_token_parent_ids.shape[0]

    num_tokens = token_pos.shape[0]
    token_ids = torch.arange(num_tokens)

    geom_distr = torch.distributions.Geometric(torch.tensor([p]))

    token_constraints = []

    for constraint in constraints:
        constraint_dim = CONSTRAINT_DIMS[constraint]

        constraints_mask_value = 0.0 if inference else CONSTRAINTS_MASK_VALUE
        pairwise_token_constraint = torch.full(
            (num_tokens, num_tokens, constraint_dim), constraints_mask_value, dtype=torch.float32
        )

        token_dists = torch.cdist(token_pos, token_pos)
        keep_constraints = inference or (training and random.random() < constraints_ratio)  # nosec

        if keep_constraints and constraint == "pocket" and num_chains > 1:
            if inference:
                pairwise_constraints = constraints.get(constraint, None)
                if exists(pairwise_constraints):
                    assert isinstance(pairwise_constraints, list) and all(
                        isinstance(i, int) for i in pairwise_constraints
                    ), f"Expected to receive a list of pocket token indices for {constraint}."
                    for idx in pairwise_constraints:
                        assert (
                            0 <= idx < num_tokens
                        ), f"Expected the pocket token index to be in the range [0, {num_tokens})."
                        pairwise_token_constraint[idx, idx] = 1.0
            else:
                pairwise_token_constraint = compute_pocket_constraint(
                    token_dists=token_dists,
                    token_parent_ids=token_parent_ids,
                    unique_token_parent_ids=unique_token_parent_ids,
                    theta_p_range=theta_p_range,
                    geom_distr=geom_distr,
                ).unsqueeze(-1)

        elif keep_constraints and constraint == "contact":
            if inference:
                pairwise_constraints = constraints.get(constraint, None)
                if exists(pairwise_constraints):
                    assert isinstance(pairwise_constraints, list) and all(
                        isinstance(c, tuple) and len(c) == 2 for c in pairwise_constraints
                    ), f"Expected to receive a list of tuples of pairwise contact token indices for {constraint}."
                    for idx1, idx2 in pairwise_constraints:
                        assert (
                            0 <= idx1 < num_tokens and 0 <= idx2 < num_tokens
                        ), f"Expected the contact token indices to be in the range [0, {num_tokens})."
                        pairwise_token_constraint[idx1, idx2] = 1.0
                        pairwise_token_constraint[idx2, idx1] = 1.0
            else:
                pairwise_token_constraint = compute_contact_constraint(
                    token_dists=token_dists,
                    theta_d_range=theta_d_range,
                    geom_distr=geom_distr,
                ).unsqueeze(-1)

        elif keep_constraints and constraint == "docking" and num_chains > 1:
            if inference:
                pairwise_constraints = constraints.get(constraint, None)
                if exists(pairwise_constraints):
                    assert isinstance(pairwise_constraints, list) and all(
                        isinstance(c, tuple) and len(c) == 3 for c in pairwise_constraints
                    ), f"Expected to receive a list of tuples of pairwise docking token indices with their bin assignments for {constraint}."
                    for idx1, idx2, bin_idx in pairwise_constraints:
                        assert (
                            0 <= idx1 < num_tokens
                            and 0 <= idx2 < num_tokens
                            and 0 <= bin_idx < len(dist_bins)
                        ), f"Expected the docking token indices to be in the range [0, {num_tokens}) and the bin index to be in the range [0, {len(dist_bins)})."
                        pairwise_token_constraint[idx1, idx2, bin_idx] = 1.0
                        pairwise_token_constraint[idx2, idx1, bin_idx] = 1.0
            else:
                pairwise_token_constraint = compute_docking_constraint(
                    token_dists=token_dists,
                    token_parent_ids=token_parent_ids,
                    unique_token_parent_ids=unique_token_parent_ids,
                    dist_bins=dist_bins,
                    geom_distr=geom_distr,
                )

        # during training, dropout chains

        chain_dropout_constraints = random.random() < constraints_ratio  # nosec
        if keep_constraints and training and chain_dropout_constraints:
            if verbose:
                logger.info(f"Applying chain constraint dropout for {constraint}...")

            sampled_chains = unique_token_parent_ids[
                torch.randint(0, num_chains, (random.randint(1, num_chains),))  # nosec
            ]
            sampled_chain_tokens_pairwise_mask = to_pairwise_mask(
                torch.isin(token_parent_ids, sampled_chains)
            )
            pairwise_token_constraint[~sampled_chain_tokens_pairwise_mask] = 0.0

        # during training, dropout tokens

        token_dropout_constraints = random.random() < constraints_ratio  # nosec
        if keep_constraints and training and token_dropout_constraints:
            if verbose:
                logger.info(f"Applying token constraint dropout for {constraint}...")

            sampled_tokens = token_ids[
                torch.randint(0, num_tokens, (random.randint(1, num_tokens),))  # nosec
            ]
            sampled_tokens_pairwise_mask = to_pairwise_mask(torch.isin(token_ids, sampled_tokens))
            pairwise_token_constraint[~sampled_tokens_pairwise_mask] = 0.0

        # aggregate token constraints

        if keep_constraints and not pairwise_token_constraint.any():
            if verbose:
                logger.info(f"No constraints were applied for {constraint}.")

            # NOTE: if all constraints were dropped, we will not include the constraint
            pairwise_token_constraint.fill_(CONSTRAINTS_MASK_VALUE)
        elif keep_constraints and verbose:
            logger.info(f"Applied constraints for {constraint}.")

        token_constraints.append(pairwise_token_constraint)

    return torch.cat(token_constraints, dim=-1)


@typecheck
def pdb_inputs_to_batched_atom_input(
    inp: PDBInput | List[PDBInput], **collate_kwargs
) -> BatchedAtomInput:
    """Collate a list of `PDBInput` objects into a `BatchedAtomInput` object."""
    if isinstance(inp, PDBInput):
        inp = [inp]

    atom_inputs = maybe_transform_to_atom_inputs(inp)
    return collate_inputs_to_batched_atom_input(atom_inputs, **collate_kwargs)


# datasets

# PDB dataset that returns either a PDBInput or AtomInput based on folder


class PDBDataset(Dataset):
    """A PyTorch Dataset for PDB mmCIF files."""

    @typecheck
    def __init__(
        self,
        folder: str | Path,
        sampler: WeightedPDBSampler | None = None,
        sample_type: Literal["default", "clustered"] = "default",
        contiguous_weight: float = 0.2,
        spatial_weight: float = 0.4,
        spatial_interface_weight: float = 0.4,
        crop_size: int = 384,
        training: bool | None = None,
        inference: bool | None = None,
        filter_out_pdb_ids: Set[str] | None = None,
        sample_only_pdb_ids: Set[str] | None = None,
        mmcif_metadata_filepath: str | None = None,
        pdbbind_binding_affinity_values_path: str | None = None,
        return_atom_inputs: bool = False,
        **pdb_input_kwargs,
    ):
        assert (
            sum([contiguous_weight, spatial_weight, spatial_interface_weight]) == 1.0
        ), "The sum of contiguous_weight, spatial_weight, and spatial_interface_weight must be equal to 1.0."

        if isinstance(folder, str):
            folder = Path(folder)

        assert folder.exists() and folder.is_dir(), f"{str(folder)} does not exist for PDBDataset"

        if "cache_msa_only" in pdb_input_kwargs or "cache_input_only" in pdb_input_kwargs:
            # disable sampling to fully pass through the dataset once
            sampler = None

        if "cache_input_only" in pdb_input_kwargs:
            # set the maximum number of seconds per input for input caching
            global PDB_INPUT_TO_MOLECULE_INPUT_MAX_SECONDS_PER_INPUT
            PDB_INPUT_TO_MOLECULE_INPUT_MAX_SECONDS_PER_INPUT = 180

        self.sampler = sampler
        self.sample_type = sample_type
        self.training = training
        self.inference = inference
        self.filter_out_pdb_ids = filter_out_pdb_ids
        self.sample_only_pdb_ids = sample_only_pdb_ids
        self.return_atom_inputs = return_atom_inputs
        self.pdb_input_kwargs = pdb_input_kwargs

        self.cropping_config = {
            "contiguous_weight": contiguous_weight,
            "spatial_weight": spatial_weight,
            "spatial_interface_weight": spatial_interface_weight,
            "n_res": crop_size,
        }

        self.cached_msas = self.pdb_input_kwargs.get(
            "msa_cache_dir", None
        ) and not self.pdb_input_kwargs.get("cache_msa_only", False)

        self.cached_inputs = self.pdb_input_kwargs.get(
            "input_cache_dir", None
        ) and not self.pdb_input_kwargs.get("cache_input_only", False)

        # subsample mmCIF files to those that have a valid (post-filtering) association with a chain/interface cluster

        if exists(self.sampler):
            sampler_pdb_ids = set(self.sampler.mappings.get_column("pdb_id").to_list())
            self.files = {
                os.path.splitext(os.path.basename(filepath.name))[0]: filepath
                for filepath in folder.glob(os.path.join("**", "*.cif*"))
                if os.path.splitext(os.path.basename(filepath.name))[0] in sampler_pdb_ids
            }
        else:
            self.files = {
                os.path.splitext(os.path.basename(file.name))[0]: file
                for file in folder.glob(os.path.join("**", "*.cif*"))
            }

            cache_msa_only = pdb_input_kwargs.get("cache_msa_only", False)
            if cache_msa_only:
                # filter to only files for which MSA features have not already been cached
                self.files = {
                    pdb_id: file
                    for pdb_id, file in self.files.items()
                    if not os.path.exists(
                        os.path.join(
                            pdb_input_kwargs["msa_cache_dir"],
                            pdb_id[1:3],
                            f"{pdb_id}.npy.gz",
                        )
                    )
                }

            cache_input_only = pdb_input_kwargs.get("cache_input_only", False)
            #print("cache_input_only", cache_input_only)
            if cache_input_only:
                # filter to only files for which input features have not already been cached
                self.files = {
                    pdb_id: file
                    for pdb_id, file in self.files.items()
                    if not os.path.exists(
                        os.path.join(
                            pdb_input_kwargs["input_cache_dir"],
                            pdb_id[1:3],
                            f"{pdb_id}.npy.gz",
                        )
                    )
                }

            assert len(self) > 0 or (
                cache_msa_only or cache_input_only
            ), f"No valid mmCIFs / PDBs found at {str(folder)} after initial filtering."
        
        if exists(filter_out_pdb_ids):
            if exists(self.sampler) and exists(mmcif_metadata_filepath):
                assert not any(
                    pdb_id in filter_out_pdb_ids for pdb_id in sampler_pdb_ids
                ), "Some PDB IDs in the dataset's sampler mappings are present in `filter_out_pdb_ids`."
            elif exists(self.sampler):
                assert not any(
                    pdb_id in sampler_pdb_ids for pdb_id in filter_out_pdb_ids
                ), "Some PDB IDs in `filter_out_pdb_ids` are present in the dataset's sampler mappings."
            else:
                self.files = {
                    pdb_id: file
                    for pdb_id, file in self.files.items()
                    if pdb_id not in filter_out_pdb_ids
                }

        if exists(sample_only_pdb_ids):
            if exists(self.sampler) and exists(mmcif_metadata_filepath):
                assert all(
                    pdb_id in sample_only_pdb_ids for pdb_id in sampler_pdb_ids
                ), "Some PDB IDs in the dataset's sampler mappings are not present in `sample_only_pdb_ids`."
            elif exists(self.sampler):
                assert all(
                    pdb_id in sampler_pdb_ids for pdb_id in sample_only_pdb_ids
                ), "Some PDB IDs in `sample_only_pdb_ids` are not present in the dataset's sampler mappings."
            else:
                self.files = {
                    pdb_id: file
                    for pdb_id, file in self.files.items()
                    if pdb_id in sample_only_pdb_ids
                }

        # maybe parse binding affinity values

        self.binding_affinity_values_dict = None

        if exists(pdbbind_binding_affinity_values_path):
            self.binding_affinity_values_dict = parse_pdbbind_binding_affinity_data_file(
                pdbbind_binding_affinity_values_path
            )

        assert len(self) > 0 or not (
            cache_msa_only or cache_input_only
        ), f"No valid mmCIFs / PDBs found at {str(folder)}"

        logger.info(f"Number of PDB mmCIF files loaded by PDBDataset: {len(self)}")

    def __len__(self):
        """Return the number of PDB mmCIF files in the dataset."""
        return len(self.files)

    def get_item(
        self, idx: int | str, random_idx: bool = False, default_affinity_ccd_id: str = "XXX"
    ) -> PDBInput | AtomInput | None:
        """Return either a PDBInput or an AtomInput object for the specified index."""
        sampled_id = None

        if random_idx:
            if isinstance(idx, str):
                idx = [*self.files.keys()][np.random.randint(0, len(self))]
            else:
                idx = np.random.randint(0, len(self))

        if exists(self.sampler):
            sample_fn = (
                self.sampler.cluster_based_sample
                if self.sample_type == "clustered"
                else self.sampler.sample
            )
            (sampled_id,) = sample_fn(1)

            # ensure that the sampled PDB ID is in the specified set of PDB IDs from which to sample

            if exists(self.sample_only_pdb_ids):
                while sampled_id[0] not in self.sample_only_pdb_ids or (
                    self.cached_inputs
                    and not os.path.exists(
                        os.path.join(
                            self.pdb_input_kwargs["input_cache_dir"],
                            sampled_id[0][1:3],
                            f"{sampled_id[0]}.npy.gz",
                        )
                    )
                ):
                    (sampled_id,) = sample_fn(1)

            # ensure that the sampled PDB ID is in the set of PDB IDs that have had their features cached

            elif self.cached_inputs:
                while not os.path.exists(
                    os.path.join(
                        self.pdb_input_kwargs["input_cache_dir"],
                        sampled_id[0][1:3],
                        f"{sampled_id[0]}.npy.gz",
                    )
                ):
                    (sampled_id,) = sample_fn(1)

        pdb_id, chain_id_1, chain_id_2 = None, None, None

        if exists(sampled_id):
            pdb_id, chain_id_1, chain_id_2 = sampled_id
            mmcif_filepath = self.files.get(pdb_id, None)

        elif isinstance(idx, int):
            pdb_id, mmcif_filepath = [*self.files.items()][idx]

        elif isinstance(idx, str):
            pdb_id = idx
            mmcif_filepath = self.files.get(pdb_id, None)

        # if MSA caching has already been performed and the sampled
        # structure's MSA features are not cached, ignore these features

        pdb_input_kwargs = self.pdb_input_kwargs.copy()

        if self.cached_msas:
            msa_cache_path = os.path.join(
                pdb_input_kwargs["msa_cache_dir"],
                pdb_id[1:3],
                f"{pdb_id}.npy.gz",
            )
            if not os.path.exists(msa_cache_path):
                pdb_input_kwargs["msa_dir"] = None

        # get the mmCIF file corresponding to the sampled structure

        if not_exists(mmcif_filepath):
            logger.warning(f"mmCIF file for PDB ID {pdb_id} not found.")
            return None
        elif not os.path.exists(mmcif_filepath):
            logger.warning(f"mmCIF file {mmcif_filepath} not found.")
            return None

        cropping_config = self.cropping_config

        if self.inference:
            cropping_config = None

        i = PDBInput(
            mmcif_filepath=str(mmcif_filepath),
            chains=(chain_id_1, chain_id_2),
            cropping_config=cropping_config,
            training=self.training,
            inference=self.inference,
            example_source="pdb",
            **pdb_input_kwargs,
        )

        if self.return_atom_inputs:
            i = maybe_transform_to_atom_input(i)

        # during training on PDB examples, annotate a
        # randomly-selected ligand in the input complex
        # with a crystal binding affinity value to enable
        # multi-ligand binding affinity prediction

        affinity_pdb_id = pdb_id.lower().split("-assembly1")[0]

        if (
            self.training
            and exists(i)
            and exists(i.num_ligands)
            and exists(self.binding_affinity_values_dict)
            and i.num_ligands > 0
            and affinity_pdb_id in self.binding_affinity_values_dict
        ):
            affinity = self.binding_affinity_values_dict[affinity_pdb_id][default_affinity_ccd_id]

            affinities = [torch.nan] * i.num_ligands
            random_ligand_idx = np.random.randint(0, i.num_ligands)

            affinities[random_ligand_idx] = affinity
            i.affinities = torch.tensor(
                affinities,
                dtype=torch.float32,
            )

        return i

    def __getitem__(self, idx: int | str, max_attempts: int = 50) -> PDBInput | AtomInput:
        """Return either a PDBInput or an AtomInput object for the specified index."""
        assert max_attempts > 0, "The maximum number of attempts must be greater than 0."

        item = self.get_item(idx)

        if (
            not_exists(item)
            and not self.pdb_input_kwargs.get("cache_msa_only", False)
            and not self.pdb_input_kwargs.get("cache_input_only", False)
        ):
            item = self._attempt_get_item(idx, max_attempts)

        return item

    def _attempt_get_item(self, idx: int | str, max_attempts: int) -> PDBInput | AtomInput:
        """Attempt to retrieve a valid item with retries."""
        random_idx = not_exists(self.sampler)
        attempts = 0

        while attempts < max_attempts:
            item = self.get_item(idx, random_idx=random_idx)
            if exists(item):
                return item
            attempts += 1

        raise ValueError(
            f"Could not retrieve a valid PDBInput or AtomInput after {max_attempts} attempts."
        )


class PDBDistillationDataset(Dataset):
    """A PyTorch Dataset for PDB distillation mmCIF files."""

    @typecheck
    def __init__(
        self,
        folder: str | Path,
        md_folder: str | Path | None,
        phage_folder: str | Path | None,
        contiguous_weight: float = 0.2,
        spatial_weight: float = 0.4,
        spatial_interface_weight: float = 0.4,
        crop_size: int = 384,
        training: bool | None = None,
        inference: bool | None = None,
        filter_out_pdb_ids: Set[str] | None = None,
        sample_only_pdb_ids: Set[str] | None = None,
        return_atom_inputs: bool = False,
        multimer_sampling_ratio: float = (2.0 / 3.0),
        uniprot_to_pdb_id_mapping_filepath: str | Path | None = None,
        mmcif_metadata_filepath: str | None = None,
        **pdb_input_kwargs,
    ):
        if isinstance(folder, str):
            folder = Path(folder)

        md_data_provided = exists(md_folder)
        if md_data_provided and isinstance(md_folder, str):
            md_folder = Path(md_folder)
            assert md_folder.exists() and md_folder.is_dir(), f"{str(md_folder)} does not exist."

        phage_data_provided = exists(phage_folder)
        if phage_data_provided and isinstance(phage_folder, str):
            phage_folder = Path(phage_folder)
            assert (
                phage_folder.exists() and phage_folder.is_dir()
            ), f"{str(phage_folder)} does not exist."

        assert (
            folder.exists() and folder.is_dir()
        ), f"{str(folder)} does not exist for PDBDistillationDataset"

        if "cache_input_only" in pdb_input_kwargs:
            # set the maximum number of seconds per input for input caching
            global PDB_INPUT_TO_MOLECULE_INPUT_MAX_SECONDS_PER_INPUT
            PDB_INPUT_TO_MOLECULE_INPUT_MAX_SECONDS_PER_INPUT = 180

        self.training = training
        self.inference = inference
        self.filter_out_pdb_ids = filter_out_pdb_ids
        self.sample_only_pdb_ids = sample_only_pdb_ids
        self.return_atom_inputs = return_atom_inputs
        self.multimer_sampling_ratio = multimer_sampling_ratio
        self.pdb_input_kwargs = pdb_input_kwargs

        filter_subset_given = exists(filter_out_pdb_ids)
        sample_subset_given = exists(sample_only_pdb_ids)

        self.cropping_config = {
            "contiguous_weight": contiguous_weight,
            "spatial_weight": spatial_weight,
            "spatial_interface_weight": spatial_interface_weight,
            "n_res": crop_size,
        }

        self.cached_msas = pdb_input_kwargs.get(
            "msa_cache_dir", None
        ) and not pdb_input_kwargs.get("cache_msa_only", False)

        self.cached_inputs = pdb_input_kwargs.get(
            "input_cache_dir", None
        ) and not pdb_input_kwargs.get("cache_input_only", False)

        # subsample mmCIF files

        uniprot_to_pdb_id_mapping_df = pl.read_csv(
            uniprot_to_pdb_id_mapping_filepath,
            has_header=False,
            separator="\t",
            new_columns=["uniprot_accession", "database", "pdb_id"],
        )
        uniprot_to_pdb_id_mapping_df.drop_in_place("database")

        self.uniprot_to_pdb_id_mapping = defaultdict(set)
        for row in uniprot_to_pdb_id_mapping_df.iter_rows():
            self.uniprot_to_pdb_id_mapping[row[0]].add(f"{row[1].lower()}-assembly1")

        self.uniprot_to_pdb_id_mapping = dict(self.uniprot_to_pdb_id_mapping)

        self.files = {
            os.path.splitext(os.path.basename(file.name))[0]: file
            for file in folder.glob(os.path.join("**", "*.cif*"))
            if os.path.splitext(os.path.basename(file.name))[0].split("-")[1]
            in self.uniprot_to_pdb_id_mapping
        }

        if md_data_provided:
            self.files.update(
                {
                    os.path.splitext(os.path.basename(file.name))[0]: file
                    for file in md_folder.glob(os.path.join("**", "*.cif*"))
                }
            )

        if phage_data_provided:
            self.files.update(
                {
                    f"{os.path.splitext(os.path.basename(file.name))[0]}-phage": file
                    for file in phage_folder.glob(os.path.join("**", "*.cif*"))
                }
            )

        cache_msa_only = pdb_input_kwargs.get("cache_msa_only", False)
        if cache_msa_only:
            # filter to only files for which MSA features have not already been cached
            self.files = {
                accession_id: file
                for accession_id, file in self.files.items()
                if not os.path.exists(
                    os.path.join(
                        pdb_input_kwargs["msa_cache_dir"],
                        accession_id.split("-")[1],
                        f"{accession_id}.npy.gz",
                    )
                )
            }

        cache_input_only = pdb_input_kwargs.get("cache_input_only", False)
        if cache_input_only:
            # filter to only files for which input features have not already been cached
            self.files = {
                accession_id: file
                for accession_id, file in self.files.items()
                if not os.path.exists(
                    os.path.join(
                        pdb_input_kwargs["input_cache_dir"],
                        accession_id.split("-")[1],
                        f"{accession_id}.npy.gz",
                    )
                )
            }

        assert len(self) > 0 or not (
            cache_msa_only or cache_input_only
        ), f"No valid mmCIFs / PDBs found at {str(folder)} after initial filtering."

        # prefilter mmCIF files based on metadata if provided

        if exists(mmcif_metadata_filepath):
            mmcif_metadata_df = pl.read_csv(mmcif_metadata_filepath)

            # replace missing (e.g., MD) distillation release dates with a default date

            mmcif_metadata_df = mmcif_metadata_df.with_columns(
                pl.col("release_date").str.replace_all("None", "1970-01-01")
            )

            if "min_length" in pdb_input_kwargs and exists(pdb_input_kwargs["min_length"]):
                mmcif_metadata_df = mmcif_metadata_df.filter(
                    mmcif_metadata_df["num_tokens"] >= pdb_input_kwargs["min_length"]
                )

            if "max_length" in pdb_input_kwargs and exists(pdb_input_kwargs["max_length"]):
                mmcif_metadata_df = mmcif_metadata_df.filter(
                    mmcif_metadata_df["num_tokens"] <= pdb_input_kwargs["max_length"]
                )

            if "max_num_atoms" in pdb_input_kwargs and exists(pdb_input_kwargs["max_num_atoms"]):
                # NOTE: this serves simply as a cropping-centric heuristic to filter out mmCIF files that are too large
                mmcif_metadata_df = mmcif_metadata_df.filter(
                    (
                        (mmcif_metadata_df["num_atoms"] / mmcif_metadata_df["num_tokens"])
                        * crop_size
                    )
                    <= pdb_input_kwargs["max_num_atoms"]
                )

            if "cutoff_date" in pdb_input_kwargs and exists(pdb_input_kwargs["cutoff_date"]):
                mmcif_metadata_df = mmcif_metadata_df.filter(
                    mmcif_metadata_df["release_date"] <= pdb_input_kwargs["cutoff_date"]
                )

            sample_only_pdb_ids_ = set(
                [
                    pdb_id
                    for file_id in mmcif_metadata_df["file_id"].to_list()
                    for pdb_id in self.uniprot_to_pdb_id_mapping.get(
                        file_id, {file_id + "-assembly1"}
                    )
                ]
            )
            self.sample_only_pdb_ids = sample_only_pdb_ids = (
                sample_only_pdb_ids.intersection(sample_only_pdb_ids_)
                if exists(sample_only_pdb_ids)
                else sample_only_pdb_ids_
            )
            assert len(sample_only_pdb_ids) > 0 or not (
                cache_msa_only or cache_input_only
            ), "No PDB IDs found after filtering with mmCIF metadata."

        if exists(filter_out_pdb_ids):
            self.files = {
                filename: file
                for filename, file in self.files.items()
                if not any(
                    pdb_id in filter_out_pdb_ids
                    # NOTE: `get()` here handles AFDB (UniProt), molecular dynamics (PDB), and phage (UniProt) data filenames
                    for pdb_id in self.uniprot_to_pdb_id_mapping.get(
                        filename.split("-")[1], {filename.split("-")[1] + "-assembly1"}
                    )
                )
                or (filename.endswith("-phage") and not filter_subset_given)
            }

        if exists(sample_only_pdb_ids):
            self.files = {
                filename: file
                for filename, file in self.files.items()
                if any(
                    pdb_id in sample_only_pdb_ids
                    # NOTE: `get()` here handles AFDB (UniProt), molecular dynamics (PDB), and phage (UniProt) data filenames
                    for pdb_id in self.uniprot_to_pdb_id_mapping.get(
                        filename.split("-")[1], {filename.split("-")[1] + "-assembly1"}
                    )
                )
                or (filename.endswith("-phage") and not sample_subset_given)
            }

        # ensure that the sampled PDB ID is in the set of PDB IDs that have had their features cached

        if self.cached_inputs:
            self.files = {
                accession_id: file
                for accession_id, file in self.files.items()
                if os.path.exists(
                    os.path.join(
                        pdb_input_kwargs["input_cache_dir"],
                        accession_id.split("-")[1],
                        f"{accession_id}.npy.gz",
                    )
                )
            }

        self.file_index_to_id = {i: accession_id for i, accession_id in enumerate(self.files)}

        if len(self) == 0 and not (cache_msa_only or cache_input_only):
            logger.warning(f"No valid mmCIFs / PDBs found at {str(folder)}")

        logger.info(f"Number of PDB mmCIF files loaded by PDBDistillationDataset: {len(self)}")

    def __len__(self):
        """Return the number of PDB mmCIF files in the dataset."""
        return len(self.files)

    def get_item(self, idx: int | str, random_idx: bool = False) -> PDBInput | AtomInput | None:
        """Return either a PDBInput or an AtomInput object for the specified index."""
        if random_idx:
            if isinstance(idx, str):
                idx = [*self.files.keys()][np.random.randint(0, len(self))]
            else:
                idx = np.random.randint(0, len(self))

        accession_id, chain_id_1, chain_id_2 = None, None, None

        if isinstance(idx, int):
            accession_id = self.file_index_to_id.get(idx, None)

        elif isinstance(idx, str):
            accession_id = idx

        if not_exists(accession_id):
            logger.warning(f"Accession ID for index {idx} not found.")
            return None

        # if MSA caching has already been performed and the sampled
        # structure's MSA features are not cached, ignore these features

        pdb_input_kwargs = self.pdb_input_kwargs.copy()

        if self.cached_msas:
            msa_cache_path = os.path.join(
                pdb_input_kwargs["msa_cache_dir"],
                accession_id.split("-")[1],
                f"{accession_id}.npy.gz",
            )
            if not os.path.exists(msa_cache_path):
                pdb_input_kwargs["msa_dir"] = None

        # get the mmCIF file corresponding to the sampled structure

        mmcif_filepath = self.files.get(accession_id, None)

        if not_exists(mmcif_filepath):
            logger.warning(f"mmCIF file for accession ID {accession_id} not found.")
            return None
        elif not os.path.exists(mmcif_filepath):
            logger.warning(f"mmCIF file {mmcif_filepath} not found.")
            return None

        cropping_config = self.cropping_config

        if self.inference:
            cropping_config = None

        i = PDBInput(
            mmcif_filepath=str(mmcif_filepath),
            chains=(chain_id_1, chain_id_2),
            cropping_config=cropping_config,
            training=self.training,
            inference=self.inference,
            distillation_multimer_sampling_ratio=self.multimer_sampling_ratio,
            # NOTE: `get()` here handles AFDB (UniProt), molecular dynamics (PDB), and phage (UniProt) data filenames
            distillation_pdb_ids=list(
                self.uniprot_to_pdb_id_mapping.get(
                    accession_id.split("-")[1], {accession_id.split("-")[1] + "-assembly1"}
                )
            ),
            example_source="pdb_distillation",
            **pdb_input_kwargs,
        )

        if self.return_atom_inputs:
            i = maybe_transform_to_atom_input(i)

        return i

    def __getitem__(self, idx: int | str, max_attempts: int = 50) -> PDBInput | AtomInput:
        """Return either a PDBInput or an AtomInput object for the specified index."""
        assert max_attempts > 0, "The maximum number of attempts must be greater than 0."

        item = self.get_item(idx)

        if (
            not_exists(item)
            and not self.pdb_input_kwargs.get("cache_msa_only", False)
            and not self.pdb_input_kwargs.get("cache_input_only", False)
        ):
            item = self._attempt_get_item(idx, max_attempts)

        return item

    def _attempt_get_item(self, idx: int | str, max_attempts: int) -> PDBInput | AtomInput:
        """Attempt to retrieve a valid item with retries."""
        attempts = 0

        while attempts < max_attempts:
            item = self.get_item(idx, random_idx=True)
            if exists(item):
                return item
            attempts += 1

        raise ValueError(
            f"Could not retrieve a valid PDBInput or AtomInput after {max_attempts} attempts."
        )


# collation function


@typecheck
def collate_inputs_to_batched_atom_input(
    inputs: List,
    int_pad_value=-1,
    atoms_per_window: int | None = None,
    map_input_fn: Callable | None = None,
    transform_to_atom_inputs: bool = True,
) -> BatchedAtomInput:
    """Collate a list of inputs into a `BatchedAtomInput` object."""
    if all(not_exists(i) for i in inputs):
        # NOTE: this case occurs when pre-caching MSAs for each example
        return

    if exists(map_input_fn):
        inputs = [map_input_fn(i) for i in inputs]

    # go through all the inputs
    # and for any that is not AtomInput, try to transform it with the registered input type to corresponding registered function

    if transform_to_atom_inputs:
        atom_inputs = maybe_transform_to_atom_inputs(inputs)

        if len(atom_inputs) < len(inputs):
            # if some of the `inputs` could not be converted into `atom_inputs`,
            # randomly select a subset of the `atom_inputs` to duplicate to match
            # the expected number of `atom_inputs`
            assert (
                len(atom_inputs) > 0
            ), "No `AtomInput` objects could be created for the current batch."
            atom_inputs = random.choices(atom_inputs, k=len(inputs))  # nosec
    else:
        assert all(isinstance(i, AtomInput) for i in inputs), (
            "When `transform_to_atom_inputs=False`, all provided "
            "inputs must be of type `AtomInput`."
        )
        atom_inputs = inputs

    assert all(isinstance(i, AtomInput) for i in atom_inputs), (
        "All inputs must be of type `AtomInput`. "
        "If you want to transform the inputs to `AtomInput`, "
        "set `transform_to_atom_inputs=True`."
    )

    # take care of windowing the atompair_inputs and atompair_ids if they are not windowed already

    if exists(atoms_per_window):
        for atom_input in atom_inputs:
            atompair_inputs = atom_input.atompair_inputs
            atompair_ids = atom_input.atompair_ids

            atompair_inputs_is_windowed = atompair_inputs.ndim == 4

            if not atompair_inputs_is_windowed:
                atom_input.atompair_inputs = full_pairwise_repr_to_windowed(
                    atompair_inputs, window_size=atoms_per_window
                )

            # NOTE: we defer windowing atompair_ids to the model so we can retain all atomic bonds
            # if exists(atompair_ids):
            #     atompair_ids_is_windowed = atompair_ids.ndim == 3

            #     if not atompair_ids_is_windowed:
            #         atom_input.atompair_ids = full_attn_bias_to_windowed(
            #             atompair_ids, window_size=atoms_per_window
            #         )

    # separate input dictionary into keys and values

    keys = list(atom_inputs[0].dict().keys())
    atom_inputs = [i.dict().values() for i in atom_inputs]

    outputs = []

    for key, grouped in zip(keys, zip(*atom_inputs)):
        # if all None, just return None

        not_none_grouped = [*filter(exists, grouped)]

        if len(not_none_grouped) == 0:
            outputs.append(None)
            continue

        # collate lists for uncollatable fields

        if key in UNCOLLATABLE_ATOM_INPUT_FIELDS:
            outputs.append(grouped)
            continue

        # default to empty tensor for any Nones

        one_tensor = not_none_grouped[0]

        dtype = one_tensor.dtype
        ndim = one_tensor.ndim

        # use -1 for padding int values, for assuming int are labels - if not, handle within megafold

        if key in ATOM_DEFAULT_PAD_VALUES:
            pad_value = ATOM_DEFAULT_PAD_VALUES[key]
        elif dtype in (torch.int, torch.long):
            pad_value = int_pad_value
        elif dtype == torch.bool:
            pad_value = False
        else:
            pad_value = 0.0

        # get the max lengths across all dimensions

        shapes_as_tensor = torch.stack(
            [tensor(tuple(g.shape) if exists(g) else ((0,) * ndim)).int() for g in grouped], dim=-1
        )

        max_lengths = shapes_as_tensor.amax(dim=-1)

        default_tensor = torch.full(max_lengths.tolist(), pad_value, dtype=dtype)

        # pad across all dimensions

        padded_inputs = []

        for inp in grouped:
            if not_exists(inp):
                padded_inputs.append(default_tensor)
                continue

            for dim, max_length in enumerate(max_lengths.tolist()):
                inp = pad_at_dim(inp, (0, max_length - inp.shape[dim]), value=pad_value, dim=dim)

            padded_inputs.append(inp)

        # stack

        stacked = torch.stack(padded_inputs)

        outputs.append(stacked)

    # batched atom input dictionary

    batched_atom_input_dict = dict(tuple(zip(keys, outputs)))

    # reconstitute dictionary

    batched_atom_inputs = BatchedAtomInput(**batched_atom_input_dict)
    return batched_atom_inputs


# the config used for keeping track of all the disparate inputs and their transforms down to AtomInput
# this can be preprocessed or will be taken care of automatically within the Trainer during data collation

INPUT_TO_ATOM_TRANSFORM = {
    AtomInput: compose(default_none_fields_atom_input),
    MoleculeInput: compose(molecule_to_atom_input, default_none_fields_atom_input),
    MegaFoldInput: compose(
        megafold_input_to_molecule_lengthed_molecule_input,
        molecule_lengthed_molecule_input_to_atom_input,
        default_none_fields_atom_input,
    ),
    PDBInput: compose(
        pdb_input_to_molecule_input, molecule_to_atom_input, default_none_fields_atom_input
    ),
}

# function for extending the config


@typecheck
def register_input_transform(input_type: Type, fn: Callable[[Any], AtomInput]):
    """Register an input transform."""
    if input_type in INPUT_TO_ATOM_TRANSFORM:
        logger.warning(f"{input_type} is already registered, but overwriting")

    INPUT_TO_ATOM_TRANSFORM[input_type] = fn


# functions for transforming to atom inputs


@typecheck
def maybe_transform_to_atom_input(i: Any, raise_exception: bool = False) -> AtomInput | None:
    """Convert an input to an AtomInput."""
    maybe_to_atom_fn = INPUT_TO_ATOM_TRANSFORM.get(type(i), None)

    if not_exists(maybe_to_atom_fn):
        raise TypeError(
            f"Invalid input type {type(i)} being passed into Trainer that is not converted to AtomInput correctly"
        )

    try:
        return maybe_to_atom_fn(i)

    # catch a timeout and log it

    except BaseException as e:
        logger.warning(
            f"Failed to convert input {i} to AtomInput due to a timeout: {e}, {traceback.format_exc()}"
        )
        if raise_exception:
            raise e

        return None

    # catch any other exception and log it

    except Exception as e:
        if "exceeds the accepted cutoff date" not in str(e):
            # NOTE: by default, we don't raise exceptions for cutoff date violations
            logger.warning(
                f"Failed to convert input {i} to AtomInput due to exception: {e}, {traceback.format_exc()}"
            )
            if raise_exception:
                raise e

        return None


@typecheck
def maybe_transform_to_atom_inputs(inputs: List[Any]) -> List[AtomInput]:
    """Convert a list of inputs to AtomInputs."""
    maybe_atom_inputs = [maybe_transform_to_atom_input(i) for i in inputs]
    return [i for i in maybe_atom_inputs if exists(i)]
