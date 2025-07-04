import os
import random
import statistics
from datetime import datetime

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from beartype.typing import Any, Dict, List, Literal, Mapping, Tuple
import einops
from loguru import logger

from megafold.common.biomolecule import (
    Biomolecule,
    _from_mmcif_object,
    get_residue_constants,
    get_resname_and_modified_status,
)
from megafold.data import mmcif_parsing
from megafold.life import DNA_NUCLEOTIDES, HUMAN_AMINO_ACIDS, LIGANDS, RNA_NUCLEOTIDES
from megafold.tensor_typing import typecheck
from megafold.utils.data_utils import (
    extract_mmcif_metadata_field,
    get_residue_molecule_type,
)
from megafold.utils.model_utils import (
    RigidFromReference3Points,
    distance_to_dgram,
    get_frames_from_atom_pos,
)
from megafold.utils.utils import exists, is_float, not_exists

# Constants

TEMPLATE_TYPE = Literal["protein", "dna", "rna", "ligand"]


@typecheck
def parse_m8(
    m8_filepath: str,
    query_id: str,
    mmcif_dir: str,
    max_templates: int | None = None,
    num_templates: int | None = None,
    template_cutoff_date: datetime | None = None,
    randomly_sample_num_templates: bool = False,
    verbose: bool = False,
) -> List[Tuple[Biomolecule, TEMPLATE_TYPE]]:
    """Parse an M8 file and return a list of template Biomolecule objects.

    :param m8_filepath: The path to the M8 file.
    :param query_id: The ID of the query sequence.
    :param mmcif_dir: The directory containing mmCIF files.
    :param max_templates: The (optional) maximum number of templates to return.
    :param num_templates: The (optional) number of templates to return.
    :param template_cutoff_date: The (optional) cutoff date for templates.
    :param randomly_sample_num_templates: Whether to randomly sample the number of templates to
        return.
    :param verbose: Whether to log verbose output.
    :return: A list of template Biomolecule objects and their template types.
    """
    # Define the column names.
    columns = [
        "ID",
        "Template ID",
        "Identity",
        "Alignment Length",
        "Mismatches",
        "Gap Openings",
        "Query Start",
        "Query End",
        "Template Start",
        "Template End",
        "E-Value",
        "Bit Score",
        "Match String",
    ]

    # Read the M8 file as a DataFrame.
    try:
        df = pl.read_csv(m8_filepath, separator="\t", has_header=False, new_columns=columns)
    except Exception as e:
        if verbose:
            logger.warning(f"Skipping loading M8 file {m8_filepath} due to: {e}")
        return []

    # Filter the DataFrame to only include rows where
    # (1) the template ID does not contain any part of the query ID;
    # (2) the template's identity is between 0.5 and 0.95, exclusively;
    # (3) the alignment length is greater than 0;
    # (4) the template's length is at least 10; and
    # (5) the number of templates is less than the (optional) maximum number of templates.
    df = df.filter(~pl.col("Template ID").str.contains(query_id))
    df = df.filter((pl.col("Identity") > 0.5) & (pl.col("Identity") < 0.95))
    df = df.filter(pl.col("Alignment Length") > 0)
    df = df.filter((pl.col("Template End") - pl.col("Template Start")) >= 9)
    if exists(max_templates):
        df = df.head(max_templates)

    # Select the number of templates to return.
    if len(df) and exists(num_templates) and randomly_sample_num_templates:
        n = len(df)
        k = min(random.randint(0, n), num_templates)  # nosec
        df = df.sample(k)
    elif exists(num_templates):
        df = df.head(num_templates)

    # Load each template chain as a Biomolecule object.
    template_biomols = []
    for i in range(len(df)):
        row = df[i]
        row_template_id = row["Template ID"].item()
        template_id, template_chain = row_template_id.split("_")
        template_fpath = os.path.join(mmcif_dir, template_id[1:3], f"{template_id}-assembly1.cif")
        if not os.path.exists(template_fpath):
            continue
        try:
            template_mmcif_object = mmcif_parsing.parse_mmcif_object(
                template_fpath, row_template_id
            )
            template_release_date = extract_mmcif_metadata_field(
                template_mmcif_object, "release_date"
            )
            if (
                exists(template_cutoff_date)
                and datetime.strptime(template_release_date, "%Y-%m-%d") > template_cutoff_date
            ):
                continue
            template_biomol = _from_mmcif_object(
                template_mmcif_object, chain_ids=set(template_chain)
            )
            if len(template_biomol.atom_positions):
                template_type = get_residue_molecule_type(
                    res_chem_index=statistics.mode(list(template_biomol.chemtype))
                )
                template_biomols.append((template_biomol, template_type))
        except Exception as e:
            if verbose:
                logger.warning(f"Skipping loading template {template_id} due to: {e}")

    return template_biomols


@typecheck
def parse_hhr(
    hhr_filepath: str,
    query_id: str,
    mmcif_dir: str,
    max_templates: int | None = None,
    num_templates: int | None = None,
    template_cutoff_date: datetime | None = None,
    randomly_sample_num_templates: bool = False,
    verbose: bool = False,
) -> List[Tuple[Biomolecule, TEMPLATE_TYPE]]:
    """Parse an HHR file and return a list of template Biomolecule objects.

    :param hhr_filepath: The path to the HHR file.
    :param query_id: The ID of the query sequence.
    :param mmcif_dir: The directory containing mmCIF files.
    :param max_templates: The (optional) maximum number of templates to return.
    :param num_templates: The (optional) number of templates to return.
    :param template_cutoff_date: The (optional) cutoff date for templates.
    :param randomly_sample_num_templates: Whether to randomly sample the number of templates to
        return.
    :param verbose: Whether to log verbose output.
    :return: A list of template Biomolecule objects and their template types.
    """
    # Define the column names and types.
    schema = {
        "No": pl.Int32,
        "Hit": pl.Utf8,
        "Prob": pl.Float64,
        "E-value": pl.Float64,
        "P-value": pl.Float64,
        "Score": pl.Float64,
        "SS": pl.Float64,
        "Cols": pl.Int32,
        "Query HMM": pl.Utf8,
        "Template HMM": pl.Utf8,
    }

    # Identify how many rows to parse.
    rows = []
    rows_found = False
    with open(hhr_filepath, "r") as f:
        for line in f:
            if line.startswith(" No Hit"):
                rows_found = True
            elif line.startswith("No 1"):
                rows_found = False

            if rows_found and not line.startswith(" No Hit") and line.strip():
                line_parts = line.strip().split()

                # NOTE: The `Hit` and `Template HMM` columns may contain spaces.
                if len(line_parts) > 10:
                    num_hit_parts = len(line_parts) - 10
                    line_parts = (
                        [line_parts[0], " ".join(line_parts[1 : 1 + num_hit_parts])]
                        + line_parts[1 + num_hit_parts : -2]
                        + [" ".join(line_parts[-2:])]
                    )

                if is_float(line_parts[2]):
                    # NOTE: Occasionally, the `Hit` column is not parsed successfully,
                    # so we use this condition to skip rows that were misparsed.
                    rows.append(line_parts)

    assert len(rows) > 0, f"No parseable rows found in HHR file {hhr_filepath}."

    # Read the HHR file as a DataFrame.
    try:
        df = pl.DataFrame(rows, schema=schema, orient="row")
    except Exception as e:
        if verbose:
            logger.warning(f"Skipping loading HHR file {hhr_filepath} due to: {e}")
        return []

    # Add shortcut columns to the DataFrame.
    df = df.with_columns(
        [
            # `Identity` is `Cols` divided by the second integer in `Query HMM`
            (
                pl.col("Cols")
                / pl.col("Query HMM").map_elements(
                    lambda x: int(x.split("-")[1]), return_dtype=pl.Float64
                )
            ).alias("Identity"),
            # `Template Start` extracted from `Template HMM`
            pl.col("Template HMM")
            .map_elements(lambda x: extract_template_hmm_range(x)[0], return_dtype=pl.Int32)
            .alias("Template Start"),
            # `Template End` extracted from `Template HMM`
            pl.col("Template HMM")
            .map_elements(lambda x: extract_template_hmm_range(x)[1], return_dtype=pl.Int32)
            .alias("Template End"),
        ]
    )

    # Filter the DataFrame to only include rows where
    # (1) the template hit ID does not contain any part of the query ID;
    # (2) the template's identity is between 0.5 and 0.95, exclusively;
    # (3) the alignment length is greater than 0;
    # (4) the template's length is at least 10; and
    # (5) the number of templates is less than the (optional) maximum number of templates.
    df = df.filter(~pl.col("Hit").str.contains(query_id.upper()))
    df = df.filter((pl.col("Identity") > 0.5) & (pl.col("Identity") < 0.95))
    df = df.filter(pl.col("Cols") > 0)
    df = df.filter((pl.col("Template End") - pl.col("Template Start")) >= 9)
    if exists(max_templates):
        df = df.head(max_templates)

    # Select the number of templates to return.
    if len(df) and exists(num_templates) and randomly_sample_num_templates:
        n = len(df)
        k = min(random.randint(0, n), num_templates)  # nosec
        df = df.sample(k)
    elif exists(num_templates):
        df = df.head(num_templates)

    # Load each template chain as a Biomolecule object.
    template_biomols = []
    for i in range(len(df)):
        row = df[i]
        row_template_id = row["Hit"].item().lower()
        template_id, template_chain = row_template_id.split(" ")[0].split("_")
        template_fpath = os.path.join(mmcif_dir, template_id[1:3], f"{template_id}-assembly1.cif")
        if not os.path.exists(template_fpath):
            continue
        try:
            template_mmcif_object = mmcif_parsing.parse_mmcif_object(
                template_fpath, row_template_id
            )
            template_release_date = extract_mmcif_metadata_field(
                template_mmcif_object, "release_date"
            )
            if (
                exists(template_cutoff_date)
                and datetime.strptime(template_release_date, "%Y-%m-%d") > template_cutoff_date
            ):
                continue
            template_biomol = _from_mmcif_object(
                template_mmcif_object, chain_ids=set(template_chain.upper())
            )
            if len(template_biomol.atom_positions):
                template_type = get_residue_molecule_type(
                    res_chem_index=statistics.mode(list(template_biomol.chemtype))
                )
                template_biomols.append((template_biomol, template_type))
        except Exception as e:
            if verbose:
                logger.warning(f"Skipping loading template {template_id} due to: {e}")

    return template_biomols


def extract_template_hmm_range(template_hmm: str):
    """Extract the start and end indices of the template HMM range.

    :param template_hmm: The template HMM range string.
    :return: A tuple containing the start and end indices of the template HMM range.
    """
    start, end = template_hmm.split("-")
    return int(start), int(end.split()[0])  # Split further to remove the parenthesis


def _extract_template_features(
    template_biomol: Biomolecule,
    mapping: Mapping[int, int],
    template_sequence: str,
    template_res_chem_comp_details: List[mmcif_parsing.ChemComp],
    query_sequence: str,
    query_chemtype: List[str],
    num_restype_classes: int = 32,
    num_distogram_bins: int = 39,
    distance_bins: List[float] = torch.linspace(3.25, 50.75, 39).float(),
    verbose: bool = False,
    eps: float = 1e-20,
) -> Dict[str, Any]:
    """Parse atom positions in the target structure and align with the query.

    Atoms for each residue in the template structure are indexed to coincide
    with their corresponding residue in the query sequence, according to the
    alignment mapping provided.

    Adapted from:
    https://github.com/aqlaboratory/openfold/blob/6f63267114435f94ac0604b6d89e82ef45d94484/openfold/data/templates.py#L16

    :param template_biomol: `Biomolecule` representing the template.
    :param mapping: Dictionary mapping indices in the query sequence to indices in
        the template sequence.
    :param template_sequence: String describing the residue sequence for the
        template.
    :param template_res_chem_comp_details: List of `ChemComp` objects describing
        the chemical components of each residue in the template sequence.
    :param query_sequence: String describing the residue sequence for the query.
    :param query_chemtype: List of strings describing the chemical type of each
        residue in the query sequence.
    :param num_restype_classes: The total number of residue types.
    :param num_dist_bins: The total number of distance bins.
    :param distance_bins: List of floats representing the bins for the distance
        histogram (i.e., distogram).
    :param verbose: Whether to log verbose output.
    :param eps: A small value to prevent division by zero.

    :return: A dictionary containing the extra features derived from the template
        structure.
    """
    assert len(mapping) == len(query_sequence) == len(query_chemtype), (
        f"Mapping length {len(mapping)} must match query sequence length {len(query_sequence)} "
        f"and query chemtype length {len(query_chemtype)}."
    )
    assert len(template_sequence) == len(template_res_chem_comp_details), (
        f"Template sequence length {len(template_sequence)} must match template residue "
        f"chemical component details length {len(template_res_chem_comp_details)}."
    )
    assert num_distogram_bins == len(distance_bins), (
        f"Number of distance bins {num_distogram_bins} must match the length of distance bins "
        f"{len(distance_bins)}."
    )

    all_atom_positions = template_biomol.atom_positions
    all_atom_mask = template_biomol.atom_mask

    all_atom_positions = np.split(all_atom_positions, all_atom_positions.shape[0])
    all_atom_masks = np.split(all_atom_mask, all_atom_mask.shape[0])

    template_restype = []
    template_all_atom_mask = []
    template_all_atom_positions = []

    template_distogram_atom_indices = []
    template_token_center_atom_indices = []
    template_three_atom_indices_for_frame = []

    for _, chemtype in zip(query_sequence, query_chemtype):
        # Handle residues in `query_sequence` that are not in `template_sequence`.
        query_chem_residue_constants = get_residue_constants(res_chem_index=chemtype)

        template_restype.append(query_chem_residue_constants.MSA_CHAR_TO_ID["-"])
        template_all_atom_mask.append(
            np.zeros(query_chem_residue_constants.atom_type_num, dtype=bool)
        )
        template_all_atom_positions.append(
            np.zeros((query_chem_residue_constants.atom_type_num, 3), dtype=np.float32)
        )

        template_distogram_atom_indices.append(0)
        template_token_center_atom_indices.append(0)
        template_three_atom_indices_for_frame.append(None)

    for query_index, template_index in mapping.items():
        # NOTE: Here, we assume that the query sequence's chemical types are the same as the
        # template sequence's chemical types. This is a reasonable assumption since the template
        # sequences are chemical type-specific search results for the query sequences.
        chemtype = query_chemtype[query_index]
        query_chem_residue_constants = get_residue_constants(res_chem_index=chemtype)

        if chemtype == 0:
            seq_mapping = HUMAN_AMINO_ACIDS
        elif chemtype == 1:
            seq_mapping = RNA_NUCLEOTIDES
        elif chemtype == 2:
            seq_mapping = DNA_NUCLEOTIDES
        elif chemtype == 3:
            seq_mapping = LIGANDS
        else:
            raise ValueError(f"Unrecognized chain chemical type: {chemtype}")

        if not (0 <= template_index < len(template_sequence)):
            if verbose:
                logger.warning(
                    f"Query index {query_index} is not mappable to the template sequence. "
                    f"Substituting with zero templates features for this position."
                )
            continue

        template_residue = template_sequence[template_index]
        template_res = seq_mapping.get(template_residue)

        # Handle modified polymer residues.
        is_ligand_res = chemtype >= 3
        _, is_modified_polymer_res = get_resname_and_modified_status(
            template_res_chem_comp_details[template_index], query_chem_residue_constants
        )
        if is_modified_polymer_res:
            template_res = LIGANDS.get("X")

        # Extract residue metadata.
        distogram_atom_idx = template_res["distogram_atom_idx"]
        token_center_atom_idx = template_res["token_center_atom_idx"]
        three_atom_indices_for_frame = template_res["three_atom_indices_for_frame"]

        if is_ligand_res or is_modified_polymer_res:
            # NOTE: Ligand and modified polymer residue representative atoms are located at
            # arbitrary indices, so we must use the atom mask to dynamically retrieve them.
            distogram_atom_idx = token_center_atom_idx = np.where(
                all_atom_masks[template_index][0]
            )[0][0]

        template_restype[query_index] = query_chem_residue_constants.MSA_CHAR_TO_ID.get(
            template_residue, query_chem_residue_constants.restype_num
        )
        template_all_atom_mask[query_index] = all_atom_masks[template_index][0]
        template_all_atom_positions[query_index] = all_atom_positions[template_index][0]

        template_distogram_atom_indices[query_index] = distogram_atom_idx
        template_token_center_atom_indices[query_index] = token_center_atom_idx
        template_three_atom_indices_for_frame[query_index] = three_atom_indices_for_frame

    # Assemble the template features tensors.
    template_restype = F.one_hot(torch.tensor(template_restype), num_classes=num_restype_classes)
    template_all_atom_mask = torch.from_numpy(np.stack(template_all_atom_mask))
    template_all_atom_positions = torch.from_numpy(np.stack(template_all_atom_positions))

    template_token_center_atom_indices = torch.tensor(template_token_center_atom_indices)
    template_token_center_atom_positions = torch.gather(
        template_all_atom_positions,
        1,
        template_token_center_atom_indices[..., None, None].expand(-1, -1, 3),
    ).squeeze(1)

    # Handle ligand and modified polymer residue frames.
    ligand_frames_present = not all(template_three_atom_indices_for_frame)
    template_backbone_frame_atom_mask = torch.ones(len(template_restype), dtype=torch.bool)
    if ligand_frames_present:
        template_token_center_atom_mask = torch.gather(
            template_all_atom_mask, 1, template_token_center_atom_indices.unsqueeze(-1)
        ).squeeze(1)
        new_frame_token_indices = get_frames_from_atom_pos(
            atom_pos=template_token_center_atom_positions,
            mask=template_token_center_atom_mask.bool(),
            filter_colinear_pos=True,
        )
        for token_index, frame_token_indices in enumerate(template_three_atom_indices_for_frame):
            if not_exists(frame_token_indices):
                # Track invalid ligand frames.
                if (new_frame_token_indices[token_index] == -1).any():
                    template_backbone_frame_atom_mask[token_index] = False
                    template_three_atom_indices_for_frame[token_index] = (0, 0, 0)
                    continue

                # Collect the (token center) atom positions of the ligand frame atoms.
                new_frame_atom_positions = []
                new_frame_atom_mask = []
                for new_frame_token_index in new_frame_token_indices[token_index]:
                    new_frame_token_center_atom_index = template_token_center_atom_indices[
                        new_frame_token_index
                    ]
                    new_frame_atom_positions.append(
                        template_all_atom_positions[
                            new_frame_token_index, new_frame_token_center_atom_index
                        ].clone()
                    )
                    new_frame_atom_mask.append(
                        template_all_atom_mask[
                            new_frame_token_index, new_frame_token_center_atom_index
                        ].clone()
                    )

                # Move the ligand frame atoms to the first three positions of the
                # ligand residue's atom positions tensor.
                for local_new_frame_atom_index in range(len(new_frame_token_indices[token_index])):
                    template_all_atom_positions[
                        token_index, local_new_frame_atom_index
                    ] = new_frame_atom_positions[local_new_frame_atom_index]
                    template_all_atom_mask[
                        token_index, local_new_frame_atom_index
                    ] = new_frame_atom_mask[local_new_frame_atom_index]

                # Update ligand metadata after moving the frame atoms.
                template_distogram_atom_indices[token_index] = 1
                template_token_center_atom_indices[token_index] = 1
                template_three_atom_indices_for_frame[token_index] = (0, 1, 2)

    # Assemble the distogram and frame atom index tensors.
    template_distogram_atom_indices = torch.tensor(template_distogram_atom_indices)
    template_three_atom_indices_for_frame = torch.tensor(template_three_atom_indices_for_frame)

    # Construct pseudo beta mask.
    template_pseudo_beta_mask = torch.gather(
        template_all_atom_mask, 1, template_distogram_atom_indices.unsqueeze(-1)
    ).squeeze(-1)

    # Construct backbone frame mask.
    template_backbone_frame_mask = template_backbone_frame_atom_mask & torch.gather(
        template_all_atom_mask,
        1,
        template_three_atom_indices_for_frame,
    ).all(-1)

    # Construct distogram.
    template_distogram_atom_positions = torch.gather(
        template_all_atom_positions,
        1,
        template_distogram_atom_indices[..., None, None].expand(-1, -1, 3),
    ).squeeze(1)
    template_distogram_dist = torch.cdist(
        template_distogram_atom_positions, template_distogram_atom_positions, p=2
    )
    template_distogram = distance_to_dgram(template_distogram_dist, distance_bins)

    # Construct unit vectors.
    template_unit_vector = torch.zeros(
        (len(template_restype), len(template_restype), 3), dtype=torch.float32
    )

    template_backbone_frame_atom_positions = torch.gather(
        template_all_atom_positions,
        1,
        template_three_atom_indices_for_frame.unsqueeze(-1).expand(-1, -1, 3),
    )

    rigid_from_reference_3_points = RigidFromReference3Points()
    template_backbone_frames, template_backbone_points = rigid_from_reference_3_points(
        template_backbone_frame_atom_positions.unbind(-2)
    )

    inv_template_backbone_frames = template_backbone_frames.transpose(-1, -2)
    template_backbone_vec = einops.einsum(
        inv_template_backbone_frames,
        template_backbone_points.unsqueeze(-2) - template_backbone_points.unsqueeze(-3),
        "n i j, m n j -> m n i",
    )
    template_inv_distance_scalar = torch.rsqrt(eps + torch.sum(template_backbone_vec**2, dim=-1))
    template_inv_distance_scalar = (
        template_inv_distance_scalar * template_backbone_frame_mask.unsqueeze(-1)
    )

    # NOTE: The unit vectors are initially of shape (j, i, 3), so they need to be transposed
    template_unit_vector = template_backbone_vec * template_inv_distance_scalar.unsqueeze(-1)
    template_unit_vector = template_unit_vector.transpose(-3, -2)

    return {
        "template_restype": template_restype.float(),
        "template_pseudo_beta_mask": template_pseudo_beta_mask.bool(),
        "template_backbone_frame_mask": template_backbone_frame_mask,
        "template_distogram": template_distogram,
        "template_unit_vector": template_unit_vector,
    }


class QueryToTemplateAlignError(Exception):
    """An error indicating that the query can't be aligned to the template."""
