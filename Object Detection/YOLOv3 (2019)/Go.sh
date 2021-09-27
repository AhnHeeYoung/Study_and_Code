#!/bin/bash

#SBATCH --clusters=brain
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=0

#SBATCH --partition=gpu_T.q
#SBATCH  --nodelist=learningT1
#SBATCH --job-name=tr
#SBATCH --output=train.out
#SBATCH --error=train.err

echo "--------"
echo "HOSTNAME = ${HOSTNAME}"
echo "SLURM_JOB_NAME = ${SLURM_JOB_NAME}"
echo "SLURM_CPUS_PER_TASK = ${SLURM_CPUS_PER_TASK}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "--------"

nvcc --version
python train.py