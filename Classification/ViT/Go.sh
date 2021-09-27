#!/bin/bash

#SBATCH --clusters=brain
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=0

#SBATCH --partition=gpu_T.q
#SBATCH  --nodelist=learningT1
#SBATCH --job-name=tr_myprebat41
#SBATCH --output=train_my_pre_epochstep_batchsize41_withouttorchgrad.out
#SBATCH --error=train_my_pre_epochstep_batchsize41_withouttorchgrad.err

echo "--------"
echo "HOSTNAME = ${HOSTNAME}"
echo "SLURM_JOB_NAME = ${SLURM_JOB_NAME}"
echo "SLURM_CPUS_PER_TASK = ${SLURM_CPUS_PER_TASK}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "--------"

nvcc --version
python train_my.py