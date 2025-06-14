import binascii
import csv
import itertools
import re
import subprocess  # nosec
from collections import defaultdict

import numpy as np
import torch
import wrapt_timeout_decorator
from beartype.typing import Any, Dict, Iterable, List, Literal, Set, Tuple
from Bio.PDB import PDBIO, MMCIFParser
from rdkit import Chem
from torch import Tensor

from omnifold.tensor_typing import ChainType, IntType, ResidueType, typecheck
from omnifold.utils.utils import exists

# constants

CALCULATE_TMSCORE_METRICS_MAX_SECONDS_PER_INPUT = 180
CALCULATE_DOCKQ_METRICS_MAX_SECONDS_PER_INPUT = 180

RESIDUE_MOLECULE_TYPE = Literal["protein", "rna", "dna", "ligand"]
RESIDUE_MOLECULE_TYPE_INDEX = Literal[0, 1, 2, 3]
PDB_INPUT_RESIDUE_MOLECULE_TYPE = Literal[
    "protein", "rna", "dna", "mod_protein", "mod_rna", "mod_dna", "ligand"
]
MMCIF_METADATA_FIELD = Literal[
    "structure_method", "release_date", "resolution", "structure_connectivity"
]


@typecheck
def is_polymer(
    res_chem_type: str, polymer_chem_types: Set[str] = {"peptide", "dna", "rna"}
) -> bool:
    """Check if a residue is polymeric using its chemical type string.

    :param res_chem_type: The chemical type of the residue as a descriptive string.
    :param polymer_chem_types: The set of polymer chemical types.
    :return: Whether the residue is polymeric.
    """
    return any(chem_type in res_chem_type.lower() for chem_type in polymer_chem_types)


@typecheck
def is_water(res_name: str, water_res_names: Set[str] = {"HOH", "WAT"}) -> bool:
    """Check if a residue is a water residue using its residue name string.

    :param res_name: The name of the residue as a descriptive string.
    :param water_res_names: The set of water residue names.
    :return: Whether the residue is a water residue.
    """
    return any(water_res_name in res_name.upper() for water_res_name in water_res_names)


@typecheck
def is_atomized_residue(
    res_name: str, atomized_res_mol_types: Set[str] = {"ligand", "mod"}
) -> bool:
    """Check if a residue is an atomized residue using its residue molecule type string.

    :param res_name: The name of the residue as a descriptive string.
    :param atomized_res_mol_types: The set of atomized residue molecule types as strings.
    :return: Whether the residue is an atomized residue.
    """
    return any(mol_type in res_name.lower() for mol_type in atomized_res_mol_types)


@typecheck
def get_residue_molecule_type(
    res_chem_type: str | None = None, res_chem_index: IntType | None = None
) -> RESIDUE_MOLECULE_TYPE:
    """Get the molecule type of a residue."""
    assert exists(res_chem_type) or exists(
        res_chem_index
    ), "Either `res_chem_type` or `res_chem_index` must be provided."
    if (exists(res_chem_type) and "peptide" in res_chem_type.lower()) or (
        exists(res_chem_index) and res_chem_index == 0
    ):
        return "protein"
    elif (exists(res_chem_type) and "rna" in res_chem_type.lower()) or (
        exists(res_chem_index) and res_chem_index == 1
    ):
        return "rna"
    elif (exists(res_chem_type) and "dna" in res_chem_type.lower()) or (
        exists(res_chem_index) and res_chem_index == 2
    ):
        return "dna"
    else:
        return "ligand"


@typecheck
def get_residue_molecule_type_index(res_chem_type: str) -> RESIDUE_MOLECULE_TYPE_INDEX:
    """Get the molecule type index of a residue."""
    if exists(res_chem_type) and (
        "peptide" in res_chem_type.lower() or "protein" in res_chem_type.lower()
    ):
        return 0
    elif exists(res_chem_type) and "rna" in res_chem_type.lower():
        return 1
    elif exists(res_chem_type) and "dna" in res_chem_type.lower():
        return 2
    else:
        return 3


@typecheck
def get_pdb_input_residue_molecule_type(
    res_chem_type: str, is_modified_polymer_residue: bool = False
) -> PDB_INPUT_RESIDUE_MOLECULE_TYPE:
    """Get the molecule type of a residue."""
    if "peptide" in res_chem_type.lower():
        return "mod_protein" if is_modified_polymer_residue else "protein"
    elif "rna" in res_chem_type.lower():
        return "mod_rna" if is_modified_polymer_residue else "rna"
    elif "dna" in res_chem_type.lower():
        return "mod_dna" if is_modified_polymer_residue else "dna"
    else:
        return "ligand"


@typecheck
def get_biopython_chain_residue_by_composite_id(
    chain: ChainType, res_name: str, res_id: int
) -> ResidueType:
    """Get a Biopython `Residue` or `DisorderedResidue` object by its residue name-residue index
    composite ID.

    :param chain: Biopython `Chain` object
    :param res_name: Residue name
    :param res_id: Residue index
    :return: Biopython `Residue` or `DisorderedResidue` object
    """
    if ("", res_id, " ") in chain:
        res = chain[("", res_id, " ")]
    elif (" ", res_id, " ") in chain:
        res = chain[(" ", res_id, " ")]
    elif (
        f"H_{res_name}",
        res_id,
        " ",
    ) in chain:
        res = chain[
            (
                f"H_{res_name}",
                res_id,
                " ",
            )
        ]
    else:
        assert (
            f"H_{res_name}",
            res_id,
            "A",
        ) in chain, f"Version A of residue {res_name} of ID {res_id} in chain {chain.id} was missing from the chain's structure."
        res = chain[
            (
                f"H_{res_name}",
                res_id,
                "A",
            )
        ]
    return res


@typecheck
def matrix_rotate(v: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Perform a rotation using a rotation matrix.

    :param v: The coordinates to rotate.
    :param matrix: The rotation matrix.
    :return: The rotated coordinates.
    """
    # For proper rotation reshape into a maximum of 2 dimensions
    orig_ndim = v.ndim
    if orig_ndim > 2:
        orig_shape = v.shape
        v = v.reshape(-1, 3)
    # Apply rotation
    v = np.dot(matrix, v.T).T
    # Reshape back into original shape
    if orig_ndim > 2:
        v = v.reshape(*orig_shape)
    return v


@typecheck
def deep_merge_dicts(
    dict1: Dict[Any, Any], dict2: Dict[Any, Any], value_op: Literal["union", "concat"]
) -> Dict[Any, Any]:
    """Deeply merge two dictionaries, merging values where possible.

    :param dict1: The first dictionary to merge.
    :param dict2: The second dictionary to merge.
    :param value_op: The merge operation to perform on the values of matching keys.
    :return: The merged dictionary.
    """
    # Iterate over items in dict2
    for key, value in dict2.items():
        # If key is in dict1, merge the values
        if key in dict1:
            merged_value = dict1[key] + value
            if value_op == "union":
                dict1[key] = list(dict.fromkeys(merged_value))  # Preserve order
            else:
                dict1[key] = merged_value
        else:
            # Otherwise, set/overwrite the key in dict1 with dict2's value
            dict1[key] = value
    return dict1


@typecheck
def coerce_to_float(obj: Any) -> float | None:
    """Coerce an object to a float, returning `None` if the object is not coercible.

    :param obj: The object to coerce to a float.
    :return: The object coerced to a float if possible, otherwise `None`.
    """
    try:
        if isinstance(obj, (int, float, str)):
            return float(obj)
        elif isinstance(obj, list):
            return float(obj[0])
        else:
            return None
    except (ValueError, TypeError):
        return None


@typecheck
def extract_mmcif_metadata_field(
    mmcif_object: Any,
    metadata_field: MMCIF_METADATA_FIELD,
    min_resolution: float = 0.0,
    max_resolution: float = 1000.0,
) -> str | float | None:
    """Extract a metadata field from an mmCIF object. If the field is not found, return `None`.

    :param mmcif_object: The mmCIF object to extract the metadata field from.
    :param metadata_field: The metadata field to extract.
    :return: The extracted metadata field.
    """
    # Extract structure method
    if metadata_field == "structure_method" and "_exptl.method" in mmcif_object.raw_string:
        return mmcif_object.raw_string["_exptl.method"]

    # Extract release date
    if (
        metadata_field == "release_date"
        and "_pdbx_audit_revision_history.revision_date" in mmcif_object.raw_string
    ):
        # Return the earliest release date
        return min(mmcif_object.raw_string["_pdbx_audit_revision_history.revision_date"])

    # Extract resolution
    if metadata_field == "resolution" and "_refine.ls_d_res_high" in mmcif_object.raw_string:
        resolution = coerce_to_float(mmcif_object.raw_string["_refine.ls_d_res_high"])
        if exists(resolution) and min_resolution <= resolution <= max_resolution:
            return resolution
    elif (
        metadata_field == "resolution"
        and "_em_3d_reconstruction.resolution" in mmcif_object.raw_string
    ):
        resolution = coerce_to_float(mmcif_object.raw_string["_em_3d_reconstruction.resolution"])
        if exists(resolution) and min_resolution <= resolution <= max_resolution:
            return resolution
    elif metadata_field == "resolution" and "_reflns.d_resolution_high" in mmcif_object.raw_string:
        resolution = coerce_to_float(mmcif_object.raw_string["_reflns.d_resolution_high"])
        if exists(resolution) and min_resolution <= resolution <= max_resolution:
            return resolution


@typecheck
def make_one_hot(x: Tensor, num_classes: int) -> Tensor:
    """Convert a tensor of indices to a one-hot tensor.

    :param x: A tensor of indices.
    :param num_classes: The number of classes.
    :return: A one-hot tensor.
    """
    x_one_hot = torch.zeros(*x.shape, num_classes, device=x.device)
    x_one_hot.scatter_(-1, x.unsqueeze(-1), 1)
    return x_one_hot


@typecheck
def make_one_hot_np(x: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert an array of indices to a one-hot encoded array.

    :param x: A NumPy array of indices.
    :param num_classes: The number of classes.
    :return: A one-hot encoded NumPy array.
    """
    x_one_hot = np.zeros((*x.shape, num_classes), dtype=np.int64)
    np.put_along_axis(x_one_hot, np.expand_dims(x, axis=-1), 1, axis=-1)
    return x_one_hot


@typecheck
def get_sorted_tuple_indices(
    tuples_list: List[Tuple[str, Any]], order_list: List[str]
) -> List[int]:
    """Get the indices of the tuples in the order specified by the order_list.

    :param tuples_list: A list of tuples containing a string and a value.
    :param order_list: A list of strings specifying the order of the tuples.
    :return: A list of indices of the tuples in the order specified by the order list.
    """
    # Create a mapping from the string values to their indices
    index_map = {value: index for index, (value, _) in enumerate(tuples_list)}

    # Generate the indices in the order specified by the order_list
    sorted_indices = [index_map[value] for value in order_list]

    return sorted_indices


@typecheck
def load_tsv_to_dict(filepath):
    """Load a two-column TSV file into a dictionary.

    :param filepath: The path to the TSV file.
    :return: A dictionary containing the TSV data.
    """
    result = {}
    with open(filepath, mode="r", newline="") as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            result[row[0]] = row[1]
    return result


@typecheck
def join(arr: Iterable[Any], delimiter: str = "") -> str:
    """Join the elements of an iterable into a string using a delimiter.

    :param arr: The iterable to join.
    :param delimiter: The delimiter to use.
    :return: The joined string.
    """
    # Re-do an ugly part of python
    return delimiter.join(arr)


@typecheck
def is_gzip_file(f: str) -> bool:
    """Check whether an input file (e.g., a `.a3m` MSA file) is gzipped.

    Method copied from Phispy, see
    https://github.com/linsalrob/PhiSpy/blob/master/PhiSpyModules/helper_functions.py.

    This is an elegant solution to test whether a file is gzipped by reading the first two
    characters.

    :param f: The file to test.
    :return: True if the file is gzip compressed, otherwise False.
    """
    with open(f, "rb") as i:
        return binascii.hexlify(i.read(2)) == b"1f8b"


@typecheck
def index_to_pdb_chain_id(index: int | np.int64) -> str:
    """Convert an index to a PDB chain ID.

    :param index: The index to convert.
    :return: The PDB chain ID.
    """
    if index < 0:
        raise ValueError("Index must be non-negative")

    letter = chr((index % 26) + ord("A"))
    number = (index + 26) // 26
    return f"{letter}{number}" if number > 0 else letter


@typecheck
def decrement_all_by_n(
    data: List[int] | List[Tuple[int, ...]], n: int
) -> List[int] | List[Tuple[int, ...]]:
    """Decrement all integers in a list or tuple of integers by n.

    :param data: List of integers or list of tuples of integers.
    :param n: Integer to decrement by.
    :return: List of integers or list of tuples of integers with all integers decremented by n.
    """
    if not data:
        # Case where no data is provided
        return data
    elif all(isinstance(item, int) for item in data):
        # Case where data is a list of integers
        return [item - n for item in data]
    elif all(isinstance(item, tuple) for item in data):
        # Case where data is a list of tuples of integers
        return [tuple(sub_item - n for sub_item in item) for item in data]
    else:
        raise ValueError("Input data must be a list of integers or a list of tuples of integers.")


@typecheck
@wrapt_timeout_decorator.timeout(
    CALCULATE_TMSCORE_METRICS_MAX_SECONDS_PER_INPUT,
    use_signals=True,
)
def calculate_tmscore_metrics(
    pred_filepath: str,
    reference_filepath: str,
    tmscore_exec_path: str,
    flags: List[str] | None = None,
) -> Dict[str, Any]:
    """Calculate TM-score structural metrics between predicted and reference biomolecular
    structures.

    :param pred_filepath: Filepath to predicted biomolecular structure in either PDB or mmCIF
        format.
    :param reference_filepath: Filepath to reference biomolecular structure in either PDB or mmCIF
        format.
    :param tmscore_exec_path: Filepath to the TM-score executable.
    :param flags: Command-line flags to pass to TM-score, optional.
    :return: Dictionary containing biomolecular TM-score structural metrics and metadata.
    """
    # Run TM-score with subprocess and capture output
    cmd = [tmscore_exec_path, pred_filepath, reference_filepath]
    if exists(flags):
        cmd += flags
    output = subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE)  # nosec

    # Parse TM-score output to extract structural metrics
    result = {}

    # Extract lengths of Structure1 and Structure2
    structure1_length = re.search(r"Structure1:.*Length=\s*(\d+)", output)
    structure2_length = re.search(r"Structure2:.*Length=\s*(\d+)", output)
    if structure1_length:
        result["Structure1_Length"] = int(structure1_length.group(1))
    if structure2_length:
        result["Structure2_Length"] = int(structure2_length.group(1))

    # Extract number of residues in common
    residues_common = re.search(r"Number of residues in common=\s*(\d+)", output)
    if residues_common:
        result["Number_of_residues_in_common"] = int(residues_common.group(1))

    # Extract RMSD of the common residues
    rmsd_common = re.search(r"RMSD of\s+the common residues=\s*([\d.]+)", output)
    if rmsd_common:
        result["RMSD_of_common_residues"] = float(rmsd_common.group(1))

    # Extract TM-score
    tm_score = re.search(r"TM-score\s*=\s*([\d.]+)", output)
    if tm_score:
        result["TM_score"] = float(tm_score.group(1))

    # Extract MaxSub-score
    maxsub_score = re.search(r"MaxSub-score=\s*([\d.]+)", output)
    if maxsub_score:
        result["MaxSub_score"] = float(maxsub_score.group(1))

    # Extract GDT-TS-score
    gdt_ts_score = re.search(r"GDT-TS-score=\s*([\d.]+)", output)
    if gdt_ts_score:
        result["GDT_TS_score"] = float(gdt_ts_score.group(1))

    # Extract GDT-HA-score
    gdt_ha_score = re.search(r"GDT-HA-score=\s*([\d.]+)", output)
    if gdt_ha_score:
        result["GDT_HA_score"] = float(gdt_ha_score.group(1))

    return result


@typecheck
@wrapt_timeout_decorator.timeout(
    CALCULATE_DOCKQ_METRICS_MAX_SECONDS_PER_INPUT,
    use_signals=True,
)
def calculate_dockq_metrics(
    prediction: str,
    reference: str,
    json_results_path: str,
    flags: List[str] | None = None,
) -> subprocess.CompletedProcess:
    """Calculate DockQ metrics between predicted and reference biomolecular structures.

    :param prediction: Filepath to predicted biomolecular structure in PDB format.
    :param reference: Filepath to reference biomolecular structure in PDB format.
    :param json_results_path: Filepath to save DockQ results in JSON format.
    :param flags: Command-line flags to pass to DockQ, optional.
    :return: Completed process object containing DockQ result logs.
    """
    # Run DockQ with subprocess and capture output
    cmd = ["DockQ", prediction, reference, "--json", json_results_path]
    if exists(flags):
        cmd += flags
    result = subprocess.run(  # nosec
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    return result


@typecheck
def convert_mmcif_to_pdb(
    input_mmcif_filepath: str,
    output_pdb_filepath: str,
) -> str:
    """Convert an mmCIF file to a PDB file using BioPython.

    :param input_mmcif_filepath: The input mmCIF file to convert.
    :param output_pdb_filepath: The output PDB file to save.
    :return: The output PDB file path.
    """
    p = MMCIFParser()
    struct = p.get_structure("", input_mmcif_filepath)

    io = PDBIO()
    io.set_structure(struct)
    io.save(output_pdb_filepath)

    return output_pdb_filepath


@typecheck
def neutralize_atoms(mol: Chem.Mol) -> Chem.Mol:
    """Neutralize the formal charges of a molecule by redistributing them to the hydrogens.

    :param mol: RDKit molecule object.
    :return: The neutralized RDKit molecule object.
    """
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!#4!#5!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]

    for at_idx in at_matches_list:
        atom = mol.GetAtomWithIdx(at_idx)
        chg = atom.GetFormalCharge()
        hcount = atom.GetTotalNumHs()
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(hcount - chg)
        atom.UpdatePropertyCache()

    return mol


@typecheck
def recursive_permutation(
    atom_inds: List[int], permutation_list: List[List[Tuple[int, int]]], res: List[int]
):
    """Recursively permute the atom indices according to the given permutation list.

    :param atom_inds: List of atom (positional) indices.
    :param permutation_list: List of tuples of atom indices to be permuted.
    :param res: List in which to store the permuted atom indices.
    """

    @typecheck
    def _permute_atom_ind(atom_inds: List[int], permutation: Tuple[int, int]) -> List[int]:
        """Permute the atom indices according to the given permutation."""
        permute_inds = [i for i, a in enumerate(atom_inds) if a in permutation]
        for i, perm_ind in enumerate(permute_inds):
            atom_inds[perm_ind] = permutation[i]
        return atom_inds

    if len(permutation_list) == 0:
        res.append(atom_inds)
    else:
        current_permutation_list = permutation_list.copy()
        for permutation in current_permutation_list.pop(0):
            atom_inds_permed = _permute_atom_ind(atom_inds.copy(), permutation)
            recursive_permutation(atom_inds_permed, current_permutation_list, res)


@typecheck
def augment_atom_maps_with_conjugate_terminal_groups(
    original_maps: Tuple[Tuple[int, ...], ...],
    atomic_number_mapping: Dict[int, int],
    terminal_group_tuples: Tuple[Tuple[int, int]],
    max_matches: int = 1e6,
) -> Tuple[Tuple[int, ...], ...]:
    """Augment atom maps from RDKit's `GetSubstructMatches()` with extra symmetry from conjugated
    terminal groups.

    :param original_maps: All possible atom index mappings, notably where we require that the mappings should range from `0` to `n_heavy_atom - 1` (i.e., there is no gap in indexing).
    :param atomic_number_mapping: Mapping from atoms' (positional) indices to their atomic numbers, for splitting/removing different types of atoms in each terminal group.
    :param terminal_group_tuples: A group of pairs of atoms whose bonds match the current SMARTS string, e.g., `((0, 1), (2, 1), (10, 9), (11, 9), (12, 9), (14, 13), (15, 13))`.
    :param max_matches: Cutoff for total number of matches (n_original_perm * n_conjugate perm)
    :return: The `original_maps` augmented by multiplying the permutations induced by `terminal_group_tuples`.
    """

    @typecheck
    def _terminal_atom_cluster_from_pairs(edges: Tuple[Tuple[int, int]]) -> Dict[int, Set[int]]:
        """Add terminal groups to the graph."""
        graph = defaultdict(set)
        for u, v in edges:
            graph[u].add(v)
            graph[v].add(u)
        return graph

    @typecheck
    def _split_sets_by_mapped_values(
        list_of_sets: List[Set[int]], mapping: Dict[int, int]
    ) -> List[Set[int]]:
        """Split sets by mapped values."""
        result = []
        for s in list_of_sets:
            mapped_sets = {}
            for elem in s:
                mapped_value = mapping.get(elem)
                if mapped_value not in mapped_sets:
                    mapped_sets[mapped_value] = set()
                mapped_sets[mapped_value].add(elem)
            result.extend(mapped_sets.values())
        return result

    # Group terminal group tuples with common atom_indices, e.g., [{0, 2}, {10, 11, 12}, {14, 15}]
    terminal_atom_clusters = _terminal_atom_cluster_from_pairs(terminal_group_tuples)
    max_terminal_groups = max(1, int(np.ceil(np.emath.logn(3, max_matches / len(original_maps)))))

    # NOTE: If `max_terminal_groups` is less than the total number terminal groups, sample the first `max_terminal_groups` groups to remove randomness
    perm_groups = sorted(
        [atom_inds for atom_inds in terminal_atom_clusters.values() if len(atom_inds) > 1]
    )[: min(max_terminal_groups, len(terminal_atom_clusters))]

    # Within each terminal group, if there are different atom types, split by atom type (NOTE: if only one left, discard)
    perm_groups = _split_sets_by_mapped_values(perm_groups, atomic_number_mapping)
    perm_groups = [p for p in perm_groups if len(p) > 1]

    # Derive all permutations according to symmetric conjugate terminal atoms, e.g., [[(0, 2), (2, 0)], [(10, 11, 12), (10, 12, 11), (11, 10, 12), (11, 12, 10), (12, 10, 11), (12, 11, 10)], [(14, 15), (15, 14)]]
    perm_groups = [sorted(list(itertools.permutations(g))) for g in perm_groups]

    # Recursively permute the original mappings
    augmented_maps = []
    for initial_mapping in original_maps:
        recursive_permutation(list(initial_mapping), perm_groups, augmented_maps)

    # Convert to the same data type as `original_maps`
    augmented_maps = tuple(tuple(a) for a in augmented_maps)

    # Remove duplicates, since `original_maps` might have already permutated some of the `conjugate_terminal` group indices
    return tuple(set(augmented_maps))


@typecheck
def _get_substructure_perms(
    mol: Chem.Mol,
    neutralize: bool = False,
    check_stereochem: bool = True,
    symmetrize_conjugated_terminal: bool = True,
    max_matches: int = 1000,
) -> np.ndarray:
    """Get the exhaustive list of substructure permutations for a given RDKit molecule.

    :param neutralize: If True, neutralize the molecule before computing the permutations.
    :param check_stereochem: Whether to assure stereochemistry does not change after permutation.
    :param symmetrize_conjugated_terminal: If True, consider symmetrization of conjugated terminal
        groups.
    :param max_matches: Cutoff for total number of matches.
    :return: The exhaustive list of substructure permutations as a NumPy array.
    """
    orig_idx_with_h = []
    for atom in mol.GetAtoms():
        atom.SetProp("orig_idx_with_h", str(atom.GetIdx()))
        orig_idx_with_h.append(atom.GetIdx())

    # NOTE: Must remove hydrogens (h), or there will be too many matches
    mol = Chem.RemoveHs(mol)
    if neutralize:
        mol = neutralize_atoms(mol)

    # Get substructure matches
    base_perms = np.array(mol.GetSubstructMatches(mol, uniquify=False, maxMatches=max_matches))
    assert len(base_perms) > 0, "No atom permutation matches found."

    # Check stereochemistry
    if check_stereochem:
        chem_order = np.array(list(Chem.rdmolfiles.CanonicalRankAtoms(mol, breakTies=False)))
        perms_mask = (chem_order[base_perms] == chem_order[None]).sum(-1) == mol.GetNumAtoms()
        base_perms = base_perms[perms_mask]

    # Add terminal conjugate groups
    sma = "[O,N;D1;$([O,N;D1]-[*]=[O,N;D1]),$([O,N;D1]=[*]-[O,N;D1])]~[*]"
    patt = Chem.MolFromSmarts(sma)
    terminal_group_tuples = mol.GetSubstructMatches(patt)

    if (
        len(terminal_group_tuples) > 0 and symmetrize_conjugated_terminal
    ):  # Only augment if there exist conjugate pairs and user wants to symmetrize
        atomic_number_mapping = {i: atom.GetAtomicNum() for i, atom in enumerate(mol.GetAtoms())}
        base_perms = augment_atom_maps_with_conjugate_terminal_groups(
            tuple(tuple(a) for a in base_perms),
            atomic_number_mapping,
            terminal_group_tuples,
            max_matches=max_matches,
        )
        base_perms = np.array(base_perms)

    if len(base_perms) > max_matches:
        base_perms = base_perms[:max_matches]

    new_to_orig_idx_map = {}
    orig_to_new_idx_map = {}
    for atom in mol.GetAtoms():
        orig_idx = int(atom.GetProp("orig_idx_with_h"))
        new_idx = atom.GetIdx()
        new_to_orig_idx_map[new_idx] = orig_idx
        orig_to_new_idx_map[orig_idx] = new_idx

    base_perms = np.vectorize(new_to_orig_idx_map.get)(base_perms)
    perms = np.zeros(shape=(base_perms.shape[0], len(orig_idx_with_h)), dtype=int)
    for i in range(len(orig_idx_with_h)):
        if i in orig_to_new_idx_map:
            perms[:, i] = base_perms[:, orig_to_new_idx_map[i]]
        else:
            # NOTE: The position of the H atom will not be exchanged
            perms[:, i] = i

    return perms


@typecheck
def get_substructure_perms(
    mol: Chem.Mol,
    check_stereochem: bool = True,
    symmetrize_conjugated_terminal: bool = True,
    max_matches: int = 1000,
    keep_protonation: bool = False,
) -> np.ndarray:
    """Get the exhaustive list of substructure permutations for a given RDKit molecule.

    :param mol: RDKit molecule object.
    :param check_stereochem: Whether to assure stereochemistry does not change after permutation.
    :param symmetrize_conjugated_terminal: If True, consider symmetrization of conjugated terminal
        groups.
    :param max_matches: Cutoff for total number of matches.
    :param keep_protonation: If True, keep the protonation state of the molecule.
    :return: The exhaustive list of substructure permutations as a NumPy array.
    """
    kwargs = {
        "check_stereochem": check_stereochem,
        "symmetrize_conjugated_terminal": symmetrize_conjugated_terminal,
        "max_matches": max_matches,
    }

    if keep_protonation:
        perms = _get_substructure_perms(mol, neutralize=False, **kwargs)
    else:
        # NOTE: Have to deduplicate permutations across the two protonation states
        perms = np.unique(
            np.row_stack(
                (
                    _get_substructure_perms(mol, neutralize=False, **kwargs),
                    _get_substructure_perms(mol, neutralize=True, **kwargs),
                )
            ),
            axis=0,
        )

    nperm = len(perms)
    if nperm > max_matches:
        perms = perms[np.random.choice(range(nperm), max_matches, replace=False)]

    return perms


@typecheck
def get_atom_perms(
    mol: Chem.Mol, max_matches: int = 1000, verbose: bool = False
) -> List[List[int]]:
    """Get the exhaustive list of atom permutations for a given RDKit molecule.

    :param mol: RDKit molecule object.
    :param max_matches: Cutoff for total number of matches.
    :param verbose: Whether to print verbose output.
    :return: The exhaustive integer list of atom permutations.
    """
    try:
        Chem.SanitizeMol(mol)
        perm = get_substructure_perms(mol, max_matches=max_matches)

    except Exception as e:
        # Sanitization failed, so permutations are unavailable
        if verbose:
            print(f"Warning: Sanitization in `get_atom_perms()` failed due to: {e}")

        perm = np.array([[i for i, atom in enumerate(mol.GetAtoms()) if atom.GetAtomicNum() != 1]])

    perm_array = perm.T  # NOTE: Has shape [num_atoms_without_h, num_perms]

    return perm_array.tolist()


@typecheck
def parse_pdbbind_binding_affinity_data_file(
    data_filepath: str, default_ligand_ccd_id: str = "XXX"
) -> Dict[str, Dict[str, float]]:
    """Extract binding affinities from the PDBBind database's metadata.

    :param data_filepath: Path to the PDBBind database's metadata file.
    :param default_ligand_ccd_id: The default CCD ID to use for PDBBind ligands, since PDBBind
        complexes only have a single ligand.
    :return: A dictionary mapping PDB codes to ligand CCD IDs and their corresponding binding
        affinities.
    """
    binding_affinity_scores_dict = {}
    with open(data_filepath) as file:
        for line in file:
            columns = line.strip().split()
            if len(columns) in {8, 9}:
                pdb_code = columns[0]
                pK_value = float(columns[3])
                # NOTE: we have to handle for multi-ligands here
                if pdb_code in binding_affinity_scores_dict:
                    assert (
                        pK_value == binding_affinity_scores_dict[pdb_code][default_ligand_ccd_id]
                    ), "PDBBind complexes should only have a single ligand."
                else:
                    binding_affinity_scores_dict[pdb_code] = {default_ligand_ccd_id: pK_value}
    return binding_affinity_scores_dict
