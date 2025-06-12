#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpuH200x8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --job-name=megafold_seqlen384
#SBATCH --output=logs/%x_%j.out

module load anaconda3_gpu
source activate venv

AF3_OPTIMIZATIONS_MODE="megafold" python3 train.py --config configs/megafold.yaml --trainer_name initial_training
