#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gpuH200x8
#SBATCH --gpus=2
#SBATCH --mem=64G
#SBATCH --job-name=megafold_seqlen384_2gpu
#SBATCH --output=logs/%x_%j.out

module load anaconda3_gpu
source activate venv

AF3_OPTIMIZATIONS_MODE="megafold" srun python3 train.py --config configs/megafold_2gpu.yaml --trainer_name initial_training

