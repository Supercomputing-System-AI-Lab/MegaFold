#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --partition=gpuH200x8
#SBATCH --mem=64G
#SBATCH --job-name=megafold
#SBATCH --output=logs/%x_%j.out

module load anaconda3_gpu
source activate venv

NUM_NODES=$SLURM_JOB_NUM_NODES
NUM_GPUS=$SLURM_NTASKS_PER_NODE
GLOBAL_BATCH_SIZE=$((NUM_NODES * NUM_GPUS))
if [ "$GLOBAL_BATCH_SIZE" -gt 1 ]; then
  NUM_WORKERS=1
  MULTIPROCESSING_CONTEXT="spawn"
  PREFETCH_FACTOR=1
  PERSISTENT_WORKERS=true
else
  NUM_WORKERS=0
  MULTIPROCESSING_CONTEXT=
  PREFETCH_FACTOR=
  PERSISTENT_WORKERS=false
fi

export NUM_NODES NUM_GPUS GLOBAL_BATCH_SIZE NUM_WORKERS MULTIPROCESSING_CONTEXT PREFETCH_FACTOR PERSISTENT_WORKERS

envsubst < configs/megafold.yaml > configs/megafold_rendered.yaml
if [ "$GLOBAL_BATCH_SIZE" -gt 1 ]; then
  srun python3 train.py --config configs/megafold_rendered.yaml --trainer_name initial_training
else
  python3 train.py --config configs/megafold_rendered.yaml --trainer_name initial_training
fi

