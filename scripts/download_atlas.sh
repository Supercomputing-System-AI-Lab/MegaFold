#!/bin/bash

mkdir -p "$(dirname "$0")/../data/md_data/raw_data/"
while IFS=, read -r name _; do
    if [[ "$name" == "PDB" ]]; then
        continue
    fi
    url="https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/${name}/${name}_protein.zip"
    wget "$url" -P "$(dirname "$0")/../data/md_data/raw_data/" || {
        url="https://www.dsimb.inserm.fr/ATLAS/database/chameleon/${name}/${name}_protein.zip"
        wget "$url" -P "$(dirname "$0")/../data/md_data/raw_data/" || {
            url="https://www.dsimb.inserm.fr/ATLAS/database/DPF/${name}/${name}_protein.zip"
            wget "$url" -P "$(dirname "$0")/../data/md_data/raw_data/" || {
                echo "Failed to download $name"
                continue
            }
        }
    }
    mkdir -p "$(dirname "$0")/../data/md_data/raw_data/${name}"
    unzip "$(dirname "$0")/../data/md_data/raw_data/${name}_protein.zip" -d "$(dirname "$0")/../data/md_data/raw_data/${name}"
    rm "$(dirname "$0")/../data/md_data/raw_data/${name}_protein.zip"
done < <(grep -v PDB "$(dirname "$0")/../data/md_data/data_caches/atlas.csv")
