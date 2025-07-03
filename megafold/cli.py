from __future__ import annotations

import ast
import copy
import os
import shutil
from pathlib import Path

import click
import numpy as np
import torch
from beartype.typing import Any, Dict, Literal

from megafold.common.biomolecule import to_inference_mmcif
from megafold.inputs import (
    CCD_COMPONENTS_SMILES,
    MegaFoldInput,
    megafold_input_to_pdb_input,
)

from megafold.model.megafold import (
    ComputeConfidenceScore,
    ComputeRankingScore,
    ConfidenceHeadLogits,
    MegaFold,
)
from megafold.tensor_typing import typecheck
from megafold.utils.data_utils import decrement_all_by_n
from megafold.utils.model_utils import batch_repeat_interleave, lens_to_mask, not_exists
from megafold.utils.utils import default, exists
from scripts.generate_id import generate_id

KALIGN_BINARY_PATH = shutil.which("kalign")
INFERENCE_DTYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float32
)

# cache model weights in memory

CKPT_PATH = (
    Path("outputs")
    / "dev-overfitting-e1-initial-and-fine-tuning"
    / "(u4ybq4zo)_megafold.ckpt.3.pt"
)

megafold = None

# helper functions


@typecheck
def np_cast(x: torch.Tensor) -> np.ndarray:
    """Cast a PyTorch tensor to a 32-bit floating point NumPy array."""
    return x.float().cpu().numpy()


@typecheck
def rank_structures(
    sampled_atom_pos: torch.Tensor,
    sample_logits: ConfidenceHeadLogits,
    sample_atom_dict: Dict[str, Any],
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, ConfidenceHeadLogits]:
    """Rank structures by sample ranking score."""
    molecule_atom_lens = sample_atom_dict["molecule_atom_lens"]
    atom_indices_for_frame = sample_atom_dict["atom_indices_for_frame"]

    has_frame = (
        (atom_indices_for_frame >= 0).all(dim=-1)
        & (molecule_atom_lens >= 0)
        & (
            # NOTE: invalid ligand or metal ion frames contain the same atom index for all three frame atoms
            ~torch.all(atom_indices_for_frame == atom_indices_for_frame[..., :1], dim=-1)
        )
    )

    atom_is_modified_residue = batch_repeat_interleave(
        sample_atom_dict["is_molecule_mod"].any(dim=-1),
        molecule_atom_lens,
    )
    atom_seq_len = atom_is_modified_residue.shape[-1]

    total_atoms = molecule_atom_lens.sum(dim=-1)
    atom_mask = lens_to_mask(total_atoms, max_len=atom_seq_len)

    asym_id = sample_atom_dict["additional_molecule_feats"].unbind(dim=-1)[2]

    sample_logits = sample_logits.to(device)
    sampled_atom_pos = sampled_atom_pos.to(device)

    compute_ranking_score = ComputeRankingScore().to(device)
    ranking_scores = compute_ranking_score.compute_score(
        confidence_head_logits=sample_logits,
        atom_pos=sampled_atom_pos,
        has_frame=has_frame.to(device),
        atom_is_modified_residue=atom_is_modified_residue.to(device),
        atom_mask=atom_mask.to(device),
        asym_id=asym_id.to(device),
        molecule_atom_lens=molecule_atom_lens.to(device),
        molecule_ids=sample_atom_dict["molecule_ids"].to(device),
        is_molecule_types=sample_atom_dict["is_molecule_types"].to(device),
    )
    sorted_indices = torch.argsort(ranking_scores, descending=True)

    return sampled_atom_pos[sorted_indices], sample_logits.rank_order_confidence_head_logits(
        sorted_indices
    )


# simple cli using click


@click.command()
@click.option(
    "-ckpt", "--checkpoint", type=str, default=CKPT_PATH, help="Path to MegaFold checkpoint"
)
@click.option(
    "-prot",
    "--protein",
    type=str,
    multiple=True,
    help="Protein sequences, with support for modified residues using e.g., <A5N> syntax",
)
@click.option(
    "-rna",
    "--rna",
    type=str,
    multiple=True,
    help="Single-stranded RNA sequences, with support for modified residues using e.g., <U2P> syntax",
)
@click.option(
    "-dna",
    "--dna",
    type=str,
    multiple=True,
    help="Single-stranded DNA sequences, with support for modified residues using e.g., <QCK> syntax",
)
@click.option(
    "-lig", "--ligand", type=str, multiple=True, help="Ligand SMILES strings or CCD codes"
)
@click.option("-met", "--metal-ion", type=str, multiple=True, help="Metal ion names or CCD codes")
@click.option(
    "-seq-order",
    "--sequence-ordering",
    type=str,
    help="Sequence ordering as a string (e.g., 1-3-2-4), if not provided, will be inferred from the presence of the different input sequence types",
)
@click.option("-id", "--input-id", type=str, help="Input ID", default=generate_id())
@click.option("-steps", "--num-sample-steps", type=int, help="Number of sampling steps to take")
@click.option(
    "-recycling",
    "--num-recycling-steps",
    type=int,
    help="Number of recycling steps to take",
    default=10,
)
@click.option(
    "-structures",
    "--num-sample-structures",
    type=int,
    help="Number of structures to sample",
    default=5,
)
@click.option("-cuda", "--use-cuda", type=bool, help="Use CUDA if available")
@click.option("-trajectory", "--sample-trajectory", type=bool, help="Sample a single trajectory")
@click.option(
    "-evo",
    "--use-optimized-evo",
    type=str,
    default=None,
    help="Use optimized Evoformer kernel (e.g., `deepspeed`, `triton`, or `None`) for faster and more memory efficient inference",
)
@click.option("-ema", "--load-ema-weights", type=bool, help="Use EMA weights if available")
@click.option(
    "-mmcif",
    "--mmcif-dir",
    type=str,
    help="MSA path",
    default=os.path.join("data", "pdb_data", "train_mmcifs"),
)
@click.option(
    "-msa",
    "--msa-dir",
    type=str,
    help="MSA path",
    default=os.path.join("data", "inference_data", "data_caches", "msa", "inference_msas"),
)
@click.option(
    "-template",
    "--templates-dir",
    type=str,
    help="Templates path",
    default=os.path.join(
        "data", "inference_data", "data_caches", "template", "inference_templates"
    ),
)
@click.option(
    "-con",
    "--constraints",
    type=str,
    multiple=True,
    help="Constraints, using 1-based indices - must be any of e.g., 'pocket:[1,3]', 'contact:[(1,3),(2,4)]', 'docking:[(2,4,3)]'",
)
@click.option("-o", "--output", type=str, help="Output path", default="output.cif")
def cli(
    checkpoint: str,
    protein: list[str],
    rna: list[str],
    dna: list[str],
    ligand: list[str],
    metal_ion: list[str],
    sequence_ordering: str | None,
    input_id: str,
    num_sample_steps: int,
    num_recycling_steps: int,
    num_sample_structures: int,
    use_cuda: bool | None,
    sample_trajectory: bool | None,
    use_optimized_evo: Literal["deepspeed", "triton"] | None,
    load_ema_weights: bool | None,
    mmcif_dir: str,
    msa_dir: str,
    templates_dir: str,
    constraints: list[str],
    output: str,
):
    """Run MegaFold on the given protein, RNA, DNA, ligand, metal ion, or modified polymer residue
    sequences."""
    assert exists(KALIGN_BINARY_PATH), "kalign binary must be installed and in the PATH."

    use_cuda = default(use_cuda, False)
    sample_trajectory = default(sample_trajectory, False)

    checkpoint_path = Path(checkpoint)
    assert checkpoint_path.exists(), f"MegaFold checkpoint must exist at {str(checkpoint_path)}"

    assert len(input_id) >= 3, "Input ID must be at least 3 characters long."
    assert (
        sample_trajectory and num_sample_structures == 1
    ) or not sample_trajectory, (
        "Can only sample a single trajectory if `sample_trajectory` is True."
    )

    assert output.endswith(".cif"), "Output must be a `.cif` file."

    mapped_ligand = []
    for lig in ligand:
        if lig in CCD_COMPONENTS_SMILES:
            lig = CCD_COMPONENTS_SMILES[lig]
        elif lig.upper() in CCD_COMPONENTS_SMILES:
            lig = CCD_COMPONENTS_SMILES[lig.upper()]

        mapped_ligand.append(lig)

    ligand = mapped_ligand

    mapped_metal_ion = []
    for met in metal_ion:
        if met in CCD_COMPONENTS_SMILES:
            met = CCD_COMPONENTS_SMILES[met]
        elif met.upper() in CCD_COMPONENTS_SMILES:
            met = CCD_COMPONENTS_SMILES[met.upper()]

        mapped_metal_ion.append(met)

    metal_ion = mapped_metal_ion

    sequence_ordering = (
        [int(idx) for idx in sequence_ordering.split("-")] if exists(sequence_ordering) else None
    )

    megafold_input = MegaFoldInput(
        proteins=protein,
        ss_rna=rna,
        ss_dna=dna,
        ligands=ligand,
        metal_ions=metal_ion,
        input_id=input_id,
        sequence_ordering=sequence_ordering,
    )

    assert all(
        len(c.split(":")) == 2 for c in constraints
    ), "Constraints must be formatted as colon-separated key-value pairs - e.g., `contact:[(1,3),(2,4)]`."
    megafold_constraints = {
        c.split(":")[0]: decrement_all_by_n(ast.literal_eval(c.split(":")[1]), n=1)
        for c in constraints
    }

    pdb_input = megafold_input_to_pdb_input(
        megafold_input,
        mmcif_dir=mmcif_dir,
        msa_dir=msa_dir,
        templates_dir=templates_dir,
        inference=True,
        constraints=megafold_constraints,
        kalign_binary_path=KALIGN_BINARY_PATH,
    )

    global megafold
    if not_exists(megafold):
        megafold = MegaFold.init_and_load(checkpoint_path, load_ema_weights=load_ema_weights)

        if use_cuda and torch.cuda.is_available():
            megafold = megafold.cuda()

        megafold.eval()

    with torch.no_grad():
        (
            sampled_atom_pos,
            sample_logits,
        ), sample_atom_dict = megafold.forward_with_megafold_inputs(
            [pdb_input] * num_sample_structures,
            dtype=INFERENCE_DTYPE,
            return_atom_dict=True,
            return_loss=False,
            return_confidence_head_logits=True,
            num_sample_steps=num_sample_steps,
            num_recycling_steps=num_recycling_steps,
            return_all_diffused_atom_pos=sample_trajectory,
            use_optimized_evo=use_optimized_evo,
        )

    if sample_trajectory:
        num_sample_structures = len(sampled_atom_pos)
        sampled_atom_pos = sampled_atom_pos.squeeze(-3)
        sample_logits = sample_logits.to("cpu").repeat(num_sample_structures)
    else:
        sampled_atom_pos, sample_logits = rank_structures(
            sampled_atom_pos, sample_logits, sample_atom_dict, device="cpu"
        )

    os.makedirs(os.path.dirname(output), exist_ok=True)

    for rank in range(num_sample_structures):
        atom_pos = sampled_atom_pos[rank]
        plddt = ComputeConfidenceScore.compute_plddt(sample_logits.plddt[rank : rank + 1]).squeeze(
            0
        )

        biomol = copy.deepcopy(pdb_input.biomol)

        biomol.atom_positions[~biomol.atom_mask.astype(bool)] = 0.0
        biomol.atom_positions[biomol.atom_mask.astype(bool)] = np_cast(atom_pos)

        biomol.b_factors[~biomol.atom_mask.astype(bool)] = 0.0
        biomol.b_factors[biomol.atom_mask.astype(bool)] = np_cast(plddt)

        mmcif_string = to_inference_mmcif(
            biomol,
            f"{input_id}_rank{rank + 1}",
            return_only_atom_site_records=sample_trajectory and rank > 0,
            model_number=rank + 1 if sample_trajectory else 1,
            # NOTE: for visualizing per-atom plDDT scores, we reference the last model number
            confidence_model_number=num_sample_structures if sample_trajectory else None,
        )

        if sample_trajectory and rank == 0:
            mmcif_string = "\n".join(mmcif_string.split("\n")[:-2]) + "\n"
        elif sample_trajectory and rank > 0:
            traj_loop_end = "#\n" if rank == num_sample_structures - 1 else ""
            mmcif_string = (
                "\n".join(
                    [
                        line
                        for line in mmcif_string.split("\n")
                        if line.startswith("ATOM") or line.startswith("HETATM")
                    ]
                )
                + "\n"
                + traj_loop_end
            )

        mmcif_output_suffix = "_traj" if sample_trajectory else f"_rank{rank + 1}"
        mmcif_output_path = output.replace(".cif", f"{mmcif_output_suffix}.cif")

        logits_output_path = mmcif_output_path.replace(".cif", "_logits")

        if sample_trajectory and rank == 0 and os.path.exists(mmcif_output_path):
            os.remove(mmcif_output_path)

        with open(mmcif_output_path, "a" if sample_trajectory else "w") as f:
            f.write(mmcif_string)

        if (sample_trajectory and rank == num_sample_structures - 1) or not sample_trajectory:
            np.savez(
                logits_output_path,
                pae=np_cast(sample_logits.pae[rank]),
                pde=np_cast(sample_logits.pde[rank]),
                plddt=np_cast(sample_logits.plddt[rank]),
                resolved=np_cast(sample_logits.resolved[rank]),
                affinity=np_cast(sample_logits.affinity[rank]),
            )

            print(
                f"mmCIF file and confidence head logits for rank {rank + 1} saved to {mmcif_output_path} and {logits_output_path + '.npz'}, respectively."
            )


if __name__ == "__main__":
    cli()
