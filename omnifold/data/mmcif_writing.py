"""An mmCIF file format writer."""

import numpy as np
from loguru import logger

from omnifold.common.biomolecule import _from_mmcif_object, to_mmcif
from omnifold.data.data_pipeline import get_assembly
from omnifold.data.mmcif_parsing import MmcifObject, parse_mmcif_object
from omnifold.utils.utils import exists


def write_mmcif_from_filepath_and_id(
    input_filepath: str, output_filepath: str, file_id: str, **kwargs
):
    """Write an input mmCIF file to an output mmCIF filepath using the provided keyword arguments
    (e.g., sampled coordinates)."""
    try:
        mmcif_object = parse_mmcif_object(filepath=input_filepath, file_id=file_id)
        return write_mmcif(mmcif_object, output_filepath=output_filepath, **kwargs)
    except Exception as e:
        logger.warning(
            f"Failed to write mmCIF file {output_filepath} due to: {e}. Perhaps cropping was performed on this example?"
        )


def write_mmcif(
    mmcif_object: MmcifObject,
    output_filepath: str,
    gapless_poly_seq: bool = True,
    insert_orig_atom_names: bool = True,
    insert_omnifold_mmcif_metadata: bool = True,
    sampled_atom_positions: np.ndarray | None = None,
    b_factors: np.ndarray | None = None,
    assume_assembly_structure: bool = True,
    allow_atom_mask_mismatch: bool = False,
):
    """Write a BioPython `Structure` object to an mmCIF file using an intermediate `Biomolecule`
    object."""
    biomol = (
        _from_mmcif_object(mmcif_object)
        if "assembly" in mmcif_object.file_id or assume_assembly_structure
        else get_assembly(_from_mmcif_object(mmcif_object))
    )
    if exists(sampled_atom_positions):
        atom_mask = biomol.atom_mask.astype(bool)

        if allow_atom_mask_mismatch:
            # NOTE: this may be useful for debugging, but should not be used in general
            num_atom_positions = min(
                biomol.atom_positions.reshape(-1, 3).shape[0], sampled_atom_positions.shape[0]
            )
            biomol.atom_positions.reshape(-1, 3)[:num_atom_positions] = sampled_atom_positions[
                :num_atom_positions
            ]
            if exists(b_factors):
                num_b_factors = min(biomol.b_factors.reshape(-1).shape[0], b_factors.shape[0])
                biomol.b_factors.reshape(-1)[:num_b_factors] = b_factors[:num_b_factors]
        else:
            assert biomol.atom_positions[atom_mask].shape == sampled_atom_positions.shape, (
                f"Expected sampled atom positions to have masked shape {biomol.atom_positions[atom_mask].shape}, "
                f"but got {sampled_atom_positions.shape}."
            )
            biomol.atom_positions[atom_mask] = sampled_atom_positions
            if exists(b_factors):
                assert biomol.b_factors[atom_mask].shape == b_factors.shape, (
                    f"Expected B-factors to have shape {biomol.b_factors[atom_mask].shape}, "
                    f"but got {b_factors.shape}."
                )
                biomol.b_factors[atom_mask] = b_factors

    unique_res_atom_names = biomol.unique_res_atom_names if insert_orig_atom_names else None
    mmcif_string = to_mmcif(
        biomol,
        mmcif_object.file_id,
        gapless_poly_seq=gapless_poly_seq,
        insert_omnifold_mmcif_metadata=insert_omnifold_mmcif_metadata,
        unique_res_atom_names=unique_res_atom_names,
    )
    with open(output_filepath, "w") as f:
        f.write(mmcif_string)
