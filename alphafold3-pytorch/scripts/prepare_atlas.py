import argparse
import os
import tempfile
from multiprocessing import Pool

import mdtraj
import pandas as pd
import tqdm
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.PDBParser import PDBParser


def main(args: argparse.Namespace):
    """Prepare the ATLAS dataset for training or testing."""
    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.split, index_col="PDB")

    jobs = []
    for name in df.index:
        # if os.path.exists(f"{args.out_dir}/{name.split('_')[0]}"):
        #     continue
        jobs.append(name)

    if args.num_workers > 1:
        p = Pool(args.num_workers)
        p.__enter__()
        __map__ = p.imap
    else:
        __map__ = map
    for _ in tqdm.tqdm(__map__(prepare_target, jobs), total=len(jobs)):
        pass
    if args.num_workers > 1:
        p.__exit__(None, None, None)


def prepare_target(name: str):
    """Prepare a single target for training or testing."""
    traj = (
        mdtraj.load(
            f"{args.atlas_dir}/{name}/{name}_prod_R1_fit.xtc",
            top=f"{args.atlas_dir}/{name}/{name}.pdb",
        )
        # + mdtraj.load(
        #     f"{args.atlas_dir}/{name}/{name}_prod_R2_fit.xtc",
        #     top=f"{args.atlas_dir}/{name}/{name}.pdb",
        # )
        # + mdtraj.load(
        #     f"{args.atlas_dir}/{name}/{name}_prod_R3_fit.xtc",
        #     top=f"{args.atlas_dir}/{name}/{name}.pdb",
        # )
    )
    ref = mdtraj.load(f"{args.atlas_dir}/{name}/{name}.pdb")
    traj = ref + traj

    f, temp_path = tempfile.mkstemp()
    os.close(f)

    parser = PDBParser(QUIET=True)

    os.makedirs(f"{args.out_dir}/{name.split('_')[0]}")

    for i in tqdm.trange(0, len(traj), 20):
        traj[i].save_pdb(temp_path)
        pdb_structure = parser.get_structure(name, temp_path)

        io = MMCIFIO()
        io.set_structure(pdb_structure)
        io.save(f"{args.out_dir}/{name.split('_')[0]}/{i}-{name.split('_')[0]}.cif")

    os.remove(temp_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="data/md_data/data_caches/atlas_train.csv")
    parser.add_argument("--atlas_dir", type=str, default="data/md_data/raw_data/")
    parser.add_argument("--out_dir", type=str, default="data/md_data/train_mmcifs/")
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    main(args)
