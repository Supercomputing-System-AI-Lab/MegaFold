from pathlib import Path

import pandas as pd


def main():
    """Subset the ATLAS dataset for training and testing."""
    # Load the PDB training split CSV file
    pdb_train_file_path = Path(
        "data/pdb_data/data_caches/train_clusterings/protein_chain_cluster_mapping.csv"
    )
    pdb_train_csv_data = pd.read_csv(pdb_train_file_path)
    pdb_train_csv_data["pdb_name"] = pdb_train_csv_data.apply(
        lambda row: f"{row['pdb_id'].split('-assembly1')[0]}_{row['chain_id']}", axis=1
    )

    # Load the PDB test split CSV file
    pdb_test_file_path = Path(
        "data/pdb_data/data_caches/test_clusterings/protein_chain_cluster_mapping.csv"
    )
    pdb_test_csv_data = pd.read_csv(pdb_test_file_path)
    pdb_test_csv_data["pdb_name"] = pdb_test_csv_data.apply(
        lambda row: f"{row['pdb_id'].split('-assembly1')[0]}_{row['chain_id']}", axis=1
    )

    # Load the PDB's metadata CSV file to perform release date filtering of the PDB training split
    pdb_metadata_file_path = Path("caches/pdb_data/metadata/mmcif.csv")
    pdb_metadata_csv_data = pd.read_csv(pdb_metadata_file_path)
    pdb_id_to_release_date_mapping = {
        k.split("-assembly1")[0]: v
        for (k, v) in dict(
            zip(pdb_metadata_csv_data["file_id"], pdb_metadata_csv_data["release_date"])
        ).items()
    }

    # Subset training and test rows to cluster representatives (i.e., the first row of each cluster)
    pdb_train_csv_data = pdb_train_csv_data.drop_duplicates(subset=["cluster_id"], keep="first")
    pdb_test_csv_data = pdb_test_csv_data.drop_duplicates(subset=["cluster_id"], keep="first")

    # Subset the training split to only include PDBs that were released before 2021-01-13
    pdb_train_csv_data = pdb_train_csv_data[
        pdb_train_csv_data.apply(
            lambda row: pdb_id_to_release_date_mapping[row["pdb_id"].split("-assembly1")[0]]
            <= "2021-01-12",
            axis=1,
        )
    ]

    # Load the chameleon split TSV file
    cham_tsv_file_path = Path("data/md_data/data_caches/2022_06_13_chameleon_info.tsv")
    cham_tsv_data = pd.read_csv(cham_tsv_file_path, sep="\t")

    # Load the DPF split TSV file
    dpf_tsv_file_path = Path("data/md_data/data_caches/2022_06_13_DPF_info.tsv")
    dpf_tsv_data = pd.read_csv(dpf_tsv_file_path, sep="\t")

    # Load the full ATLAS TSV file
    atlas_tsv_file_path = Path("data/md_data/data_caches/2023_03_09_ATLAS_info.tsv")
    atlas_tsv_data = pd.read_csv(atlas_tsv_file_path, sep="\t")

    # Subset the chameleon and DPF data to only include the PDB and chain IDs that are in the PDB training split
    train_cham_tsv_data = cham_tsv_data[cham_tsv_data["PDB"].isin(pdb_train_csv_data["pdb_name"])]
    train_dpf_tsv_data = dpf_tsv_data[dpf_tsv_data["PDB"].isin(pdb_train_csv_data["pdb_name"])]

    # Subset the ATLAS data to only include the PDB and chain IDs that are in the PDB test split
    test_atlas_tsv_data = atlas_tsv_data[atlas_tsv_data["PDB"].isin(pdb_test_csv_data["pdb_name"])]

    # Subset the chameleon and DPF data to only include a randomly-sampled (fixed-size) subset of the PDB training split
    subset_size = 10

    train_subset_cham_tsv_data = train_cham_tsv_data.sample(n=subset_size, random_state=42)
    train_subset_dpf_tsv_data = train_dpf_tsv_data.sample(n=subset_size, random_state=42)

    # Subset the ATLAS data to only include a randomly-sampled (fixed-size) subset of the PDB test split
    subset_size = min(
        subset_size, len(test_atlas_tsv_data)
    )  # NOTE: Currently, only a single chain is available as an ATLAS-based test split

    test_subset_atlas_tsv_data = test_atlas_tsv_data.sample(n=subset_size, random_state=42)

    # Write out the subsetted chameleon and DPF data to a combined CSV file with a column denoting the dataset of origin
    train_subset_cham_tsv_data["dataset"] = "chameleon"
    train_subset_dpf_tsv_data["dataset"] = "DPF"
    test_subset_atlas_tsv_data["dataset"] = "ATLAS"

    combined_train_subset_tsv_data = pd.concat(
        [train_subset_cham_tsv_data, train_subset_dpf_tsv_data]
    )
    combined_train_subset_tsv_data.to_csv("data/md_data/data_caches/atlas_train.csv", index=False)

    # Write out the subsetted ATLAS data to a separate CSV file
    test_subset_atlas_tsv_data.to_csv("data/md_data/data_caches/atlas_test.csv", index=False)

    # Write out a combined CSV file with all subsetted data
    combined_subset_tsv_data = pd.concat(
        [combined_train_subset_tsv_data, test_subset_atlas_tsv_data]
    )
    combined_subset_tsv_data.to_csv("data/md_data/data_caches/atlas.csv", index=False)


if __name__ == "__main__":
    main()
