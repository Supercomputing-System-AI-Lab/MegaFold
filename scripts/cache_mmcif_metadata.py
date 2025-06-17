import argparse
import glob
import os
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool

import rootutils
import wrapt_timeout_decorator
from beartype.typing import Any, Dict, Tuple
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from megafold.common.biomolecule import _from_mmcif_object
from megafold.data import mmcif_parsing
from megafold.utils.data_utils import extract_mmcif_metadata_field
from megafold.utils.utils import exists, not_exists

# Constants
PROCESS_MMCIF_MAX_SECONDS_PER_INPUT = 15


@wrapt_timeout_decorator.timeout(PROCESS_MMCIF_MAX_SECONDS_PER_INPUT, use_signals=True)
def process_mmcif_with_timeout(filepath: str, is_pdb_input: bool) -> Dict[str, Any]:
    """Process a single mmCIF file under a timeout constraint and return its metadata.

    :param filepath: Path to the mmCIF file.
    :param is_pdb_input: Flag to indicate if input is PDB data.
    :return: Metadata of the mmCIF file.
    """
    file_id = os.path.splitext(os.path.basename(filepath))[0]

    # Handle distillation examples
    if not is_pdb_input:
        file_id = file_id.split("-")[1]

    mmcif_object = mmcif_parsing.parse_mmcif_object(
        filepath=filepath,
        file_id=file_id,
    )
    mmcif_resolution = extract_mmcif_metadata_field(mmcif_object, "resolution")
    mmcif_release_date = extract_mmcif_metadata_field(mmcif_object, "release_date")
    biomol = _from_mmcif_object(mmcif_object)

    # Filter out entries with missing release date metadata
    if is_pdb_input and (
        not_exists(mmcif_release_date)
        or not_exists(datetime.strptime(mmcif_release_date, "%Y-%m-%d"))
    ):
        return {}

    mmcif_num_tokens = len(biomol.atom_mask)
    mmcif_chain_ids = list(dict.fromkeys(biomol.chain_id))
    mmcif_chemtypes = [str(chemtype) for chemtype in dict.fromkeys(biomol.chemtype)]
    mmcif_chemids = [str(chemid) for chemid in dict.fromkeys(biomol.chemid)]

    mmcif_metadata = {
        "file_id": file_id,
        "num_tokens": mmcif_num_tokens,
        "num_atoms": int(biomol.atom_mask.sum()),
        "num_chains": len(mmcif_chain_ids),
        "chain_ids": "-".join(mmcif_chain_ids),
        "chemtypes": "-".join(mmcif_chemtypes),
        "chemids": "-".join(mmcif_chemids),
        "resolution": mmcif_resolution if exists(mmcif_resolution) else -1,
        "release_date": mmcif_release_date,
    }

    return mmcif_metadata


def process_mmcif(inputs: Tuple[str, bool]) -> Dict[str, Any]:
    """Process a single mmCIF file and return its metadata.

    :param inputs: Tuple of the mmCIF file path and a flag to indicate if input is PDB data.
    :return: Metadata of the mmCIF file.
    """
    filepath, is_pdb_input = inputs
    try:
        return process_mmcif_with_timeout(filepath, is_pdb_input)
    except Exception as e:
        print(f"Processing of mmCIF {filepath} was terminated due to: {e}. Skipping this file...")
        return {}


def cache_mmcif_metadata(
    input_mmcif_dir: str,
    metadata_filepath: str,
    is_pdb_input: bool,
    num_processes: int,
):
    """Cache metadata of each mmCIF file at a given metadata filepath."""
    os.makedirs(os.path.dirname(metadata_filepath), exist_ok=True)

    mmcif_file_pattern = os.path.join(input_mmcif_dir, "*.cif")
    mmcif_files = [
        fp
        for fp in list(glob.glob(mmcif_file_pattern))
        if "unfiltered" not in fp
        and ((is_pdb_input and "pdb_data" in fp) or (not is_pdb_input and "pdb_data" not in fp))
    ]

    mmcifs_to_keep = defaultdict(set)
    for mmcif_file in tqdm(
        mmcif_files,
        desc="Identifying mmCIF files by filtered status",
    ):
        input_id = os.path.splitext(os.path.basename(mmcif_file))[0]
        if not is_pdb_input:
            input_id = input_id.split("-")[1]
        mmcifs_to_keep[input_id].add(mmcif_file)

    # Prepare the multiprocessing pool
    pool = Pool(processes=num_processes)

    # Prepare arguments for each worker
    mmcif_info = [
        (mmcif_file, is_pdb_input)
        for input_id in mmcifs_to_keep
        for mmcif_file in mmcifs_to_keep[input_id]
    ]

    # Process mmCIFs in parallel
    metadata_dicts = []
    for metadata_dict in tqdm(
        pool.imap_unordered(process_mmcif, mmcif_info),
        total=len(mmcif_info),
        desc="Processing mmCIFs",
    ):
        if metadata_dict:
            metadata_dicts.append(metadata_dict)

    pool.close()
    pool.join()

    assert len(metadata_dicts) > 0

    # Write metadata to file
    with open(metadata_filepath, "w") as f:
        keys = metadata_dicts[0].keys()
        f.write(",".join(keys) + "\n")
        for metadata_dict in metadata_dicts:
            f.write(",".join([str(metadata_dict[key]) for key in keys]) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cache metadata from mmCIF files into a CSV file."
    )
    parser.add_argument(
        "--is_pdb_input",
        action="store_true",
        help="Flag to indicate if input is PDB data (default: False for AFDB data)",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Base directory containing mmCIF files. Defaults based on is_pdb_input.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path for output CSV file. Defaults based on is_pdb_input.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=12,
        help="Number of parallel processes (default: 12)",
    )
    args = parser.parse_args()

    # Use provided paths or construct defaults
    if args.input_dir is None:
        mmcif_dir = "pdb_data" if args.is_pdb_input else "*_data"
        args.input_dir = os.path.join("data", mmcif_dir, "*_mmcifs", "*")

    if args.output_path is None:
        metadata_dir = "pdb_data" if args.is_pdb_input else "afdb_data"
        args.output_path = os.path.join("caches", metadata_dir, "metadata", "mmcif.csv")

    cache_mmcif_metadata(
        input_mmcif_dir=args.input_dir,
        metadata_filepath=args.output_path,
        is_pdb_input=args.is_pdb_input,
        num_processes=args.num_processes,
    )
