import glob
import json
import os
from collections import defaultdict

import polars as pl
import rootutils
from loguru import logger

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from omnifold.data.msa_parsing import parse_fasta


def analyze_pdb_test_dataset(test_clusterings_dir: str):
    """Analyze the PDB test dataset's clusterings."""
    for csv_path in glob.glob(os.path.join(test_clusterings_dir, "*_cluster_mapping.csv")):
        cluster_df = pl.read_csv(csv_path)

        cluster_id_key = "interface_cluster_id" if "interface" in csv_path else "cluster_id"
        cluster_reps = cluster_df.group_by(cluster_id_key).first().sort(cluster_id_key)

        if "interface" in csv_path:
            logger.info(
                f"{len(cluster_reps)} interface cluster representatives for {csv_path}. First 5:"
                f"\n{cluster_reps.head(5)}"
            )
        else:
            logger.info(
                f"{len(cluster_reps)} cluster representatives for {csv_path}. First 5:"
                f"\n{cluster_reps.head(5)}"
            )
            if "nucleic_acid" in csv_path:
                logger.info(
                    f"\nMolecule type breakdown:\n{cluster_reps['molecule_id'].value_counts()}"
                )


def subset_pdb_test_dataset(test_clusterings_dir: str, max_chain_sequence_length: int = 2048):
    """Subset the PDB test dataset's examples to cluster representatives and examples containing no
    modified polymer residues."""
    for csv_path in glob.glob(os.path.join(test_clusterings_dir, "*_cluster_mapping.csv")):
        # Load the cluster mapping

        logger.info(f"Subsetting with {csv_path}")
        cluster_df = pl.read_csv(csv_path)

        cluster_id_key = (
            "interface_cluster_id" if "interface" in os.path.basename(csv_path) else "cluster_id"
        )
        cluster_reps = (
            cluster_df.group_by(cluster_id_key).first().unique("pdb_id").sort(cluster_id_key)
        )

        if "interface" in os.path.basename(csv_path):
            # NOTE: We exclude ligand and peptide interfaces for now
            cluster_reps = cluster_reps.filter(
                (~pl.col("interface_molecule_id_1").str.contains("ligand"))
                & (~pl.col("interface_molecule_id_2").str.contains("ligand"))
                & (~pl.col("interface_molecule_id_1").str.contains("peptide"))
                & (~pl.col("interface_molecule_id_2").str.contains("peptide"))
            )

        # Find the corresponding monomer FASTA file

        if "protein_chain" in os.path.basename(csv_path):
            fasta_path = os.path.join(os.path.dirname(csv_path), "sequences_monomer_protein.fasta")
        elif "nucleic_acid_chain" in os.path.basename(csv_path):
            fasta_path = os.path.join(
                os.path.dirname(csv_path), "sequences_monomer_nucleic_acid.fasta"
            )
        elif "peptide_chain" in os.path.basename(csv_path):
            fasta_path = os.path.join(os.path.dirname(csv_path), "sequences_monomer_peptide.fasta")
        elif "ligand_chain" in os.path.basename(csv_path):
            # NOTE: We don't subset the ligand chains for now
            continue
        elif "interface" not in os.path.basename(csv_path):
            raise ValueError(f"Unknown cluster mapping CSV: {csv_path}")

        if "interface" in os.path.basename(csv_path):
            # Find the corresponding multimer FASTA files

            multimer_protein_fasta_path = os.path.join(
                os.path.dirname(csv_path), "sequences_multimer_protein.fasta"
            )
            multimer_nucleic_acid_fasta_path = os.path.join(
                os.path.dirname(csv_path), "sequences_multimer_nucleic_acid.fasta"
            )

            # Load the multimer FASTA files

            with open(multimer_protein_fasta_path, "r") as fasta_file:
                multimer_protein_fasta_string = fasta_file.read()
                multimer_protein_sequences, multimer_protein_descriptions = parse_fasta(
                    multimer_protein_fasta_string
                )

                multimer_proteins = defaultdict(list)
                for sequence, description in zip(
                    multimer_protein_sequences, multimer_protein_descriptions
                ):
                    pdb_id = description.split("-assembly1")[0] + "-assembly1"
                    multimer_proteins[pdb_id].append((description, sequence))

            with open(multimer_nucleic_acid_fasta_path, "r") as fasta_file:
                multimer_nucleic_acid_fasta_string = fasta_file.read()
                multimer_nucleic_acid_sequences, multimer_nucleic_acid_descriptions = parse_fasta(
                    multimer_nucleic_acid_fasta_string
                )

                multimer_nucleic_acids = defaultdict(list)
                for sequence, description in zip(
                    multimer_nucleic_acid_sequences, multimer_nucleic_acid_descriptions
                ):
                    pdb_id = description.split("-assembly1")[0] + "-assembly1"
                    multimer_nucleic_acids[pdb_id].append((description, sequence))

            # Collect all sequences associated with a subset PDB ID

            subset_sequences, subset_descriptions = [], []
            with open(
                os.path.join(os.path.dirname(csv_path), "all_chain_sequences.json"), "r"
            ) as all_chain_sequences_file:
                all_chain_sequences = {
                    list(d.keys())[0]: list(d.values())[0]
                    for d in json.load(all_chain_sequences_file)
                }

            # Subset multimer sequences within a new FASTA file

            filtered_subset_sequences, filtered_subset_descriptions = [], []
            for cluster_rep in cluster_reps.iter_rows(named=True):
                cluster_chain_sequences = all_chain_sequences[cluster_rep["pdb_id"]]
                cluster_protein_sequences = multimer_proteins[cluster_rep["pdb_id"]]
                cluster_nucleic_acid_sequences = multimer_nucleic_acids[cluster_rep["pdb_id"]]

                # Filter by presence of ligands or modified residues as well as sequence length

                if (
                    any("ligand" in description for description in cluster_chain_sequences.keys())
                    or any("X" in sequence for sequence in cluster_chain_sequences.values())
                    or (
                        sum(len(sequence) for sequence in cluster_chain_sequences.values())
                        > max_chain_sequence_length
                    )
                ):
                    continue

                # All chain types

                for description, sequence in cluster_chain_sequences.items():
                    subset_sequences.append(sequence)
                    subset_descriptions.append(cluster_rep["pdb_id"] + description)

                # Non-redundant protein chains

                for description, sequence in cluster_protein_sequences:
                    filtered_subset_sequences.append(sequence)
                    filtered_subset_descriptions.append(description)

                # Non-redundant nucleic acid chains

                for description, sequence in cluster_nucleic_acid_sequences:
                    filtered_subset_sequences.append(sequence)
                    filtered_subset_descriptions.append(description)

            # Write the subset FASTA files

            subset_fasta_path = os.path.join(
                os.path.dirname(csv_path), "subset_sequences_multimer.fasta"
            )
            with open(subset_fasta_path, "w") as subset_fasta_file:
                for description, sequence in zip(subset_descriptions, subset_sequences):
                    subset_fasta_file.write(f">{description}\n{sequence}\n")

            filtered_subset_fasta_path = os.path.join(
                os.path.dirname(csv_path), "filtered_subset_sequences_multimer.fasta"
            )
            with open(filtered_subset_fasta_path, "w") as subset_fasta_file:
                for description, sequence in zip(
                    filtered_subset_descriptions, filtered_subset_sequences
                ):
                    subset_fasta_file.write(f">{description}\n{sequence}\n")

            # Log some multimer statistics

            unique_multimers = defaultdict(list)
            for description in subset_descriptions:
                pdb_id = description.split("-assembly1")[0] + "-assembly1"
                unique_multimers[pdb_id].append(description)

            logger.info(f"Number of multimer PDB IDs: {len(unique_multimers)}")

            logger.info(
                f"Number of PDB multimers associated with only protein chains: {len([pdb_id for pdb_id, descriptions in unique_multimers.items() if all('protein' in description or 'peptide' in description for description in descriptions)])}"
            )
            logger.info(
                f"Number of PDB multimers associated with only RNA chains: {len([pdb_id for pdb_id, descriptions in unique_multimers.items() if all('rna' in description for description in descriptions)])}"
            )
            logger.info(
                f"Number of PDB multimers associated with only DNA chains: {len([pdb_id for pdb_id, descriptions in unique_multimers.items() if all('dna' in description for description in descriptions)])}"
            )

            logger.info(
                f"Number of PDB multimers associated with both protein and RNA chains: {len([pdb_id for pdb_id, descriptions in unique_multimers.items() if any('protein' in description or 'peptide' in description for description in descriptions) and any('rna' in description for description in descriptions) and not any('dna' in description for description in descriptions)])}"
            )
            logger.info(
                f"Number of PDB multimers associated with both protein and DNA chains: {len([pdb_id for pdb_id, descriptions in unique_multimers.items() if any('protein' in description or 'peptide' in description for description in descriptions) and any('dna' in description for description in descriptions) and not any('rna' in description for description in descriptions)])}"
            )
            logger.info(
                f"Number of PDB multimers associated with both RNA and DNA chains: {len([pdb_id for pdb_id, descriptions in unique_multimers.items() if any('rna' in description for description in descriptions) and any('dna' in description for description in descriptions) and not any('protein' in description or 'peptide' in description for description in descriptions)])}"
            )

            logger.info(
                f"Number of PDB multimers associated with protein, RNA, and DNA chains: {len([pdb_id for pdb_id, descriptions in unique_multimers.items() if any('protein' in description or 'peptide' in description for description in descriptions) and any('rna' in description for description in descriptions) and any('dna' in description for description in descriptions)])}"
            )

        else:
            # Load the monomer FASTA file

            cluster_reps_set = set(cluster_reps["pdb_id"] + cluster_reps["chain_id"])

            with open(fasta_path, "r") as fasta_file:
                fasta_string = fasta_file.read()
                sequences, descriptions = parse_fasta(fasta_string)

            # Subset monomer sequences within a new FASTA file

            filtered_subset_sequences, filtered_subset_descriptions = [], []
            for sequence, description in zip(sequences, descriptions):
                pdb_description = description.split(":")[0]

                if (
                    pdb_description in cluster_reps_set
                    and "X" not in sequence
                    and len(sequence) <= max_chain_sequence_length
                ):
                    filtered_subset_sequences.append(sequence)
                    filtered_subset_descriptions.append(description)

            filtered_subset_fasta_path = os.path.join(
                os.path.dirname(csv_path), f"subset_{os.path.basename(fasta_path)}"
            )
            with open(filtered_subset_fasta_path, "w") as subset_fasta_file:
                for description, sequence in zip(
                    filtered_subset_descriptions, filtered_subset_sequences
                ):
                    subset_fasta_file.write(f">{description}\n{sequence}\n")


if __name__ == "__main__":
    analyze_pdb_test_dataset(
        test_clusterings_dir=os.path.join("data", "pdb_data", "data_caches", "test_clusterings")
    )
    subset_pdb_test_dataset(
        test_clusterings_dir=os.path.join("data", "pdb_data", "data_caches", "test_clusterings")
    )
