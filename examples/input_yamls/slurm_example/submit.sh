#!/bin/bash

#SBATCH -p gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00
#SBATCH --output=job.out           # File to which standard out will be written
#SBATCH --error=job.err
#SBATCH --job-name=etnet

export SLURM_CPU_BIND=none

echo $HOSTNAME
echo All CUDA DEVICEs:
echo $CUDA_VISIBLE_DEVICES

source ~/.bashrc

conda activate cg310

module load gcc/11.3.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python /path/to/mlcg/scripts/mlcg-train_h5.py fit --config train.yaml --config model.yaml

