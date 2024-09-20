#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=065
#SBATCH --output=data/slurm_logs/065.out
#SBATCH --error=data/slurm_logs/065.err
#SBATCH --time=24:00:00
#SBATCH --mem=36000
#SBATCH --gres=gpu:1
#SBATCH --qos=medium
#SBATCH --nodelist=noether

srun -u /slurm-storage/leobuj/.conda/envs/leobuj-env/bin/python -m exp.spgat_mnist \
    --model-description lin-no-pe --batch-size 32 \
    --pe 0