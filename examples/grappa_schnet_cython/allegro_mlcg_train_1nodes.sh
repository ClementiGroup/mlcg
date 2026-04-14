#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --mem=30G
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=./logs/train-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rockykamenrubio@gmail.com
#SBATCH --constraint="RTX-A5000"


##SBATCH --requeue
##SBATCH --constraint=GTX-1080ti

export SLURM_CPU_BIND=none

echo "# what GPUs have we been assigned?"
nvidia-smi --query-gpu=index,name,serial,gpu_bus_id,memory.total --format=csv

echo "# what cores have we been assigned?"
taskset -pc $$

echo "# what node are we running on?"
hostname -s

# CONDAPATH=/data/scratch/${USER}/mambaforge
# source ${CONDAPATH}/etc/profile.d/conda.sh
# conda activate ${ENVNAME}

# source /data/scratch/${USER}/nov_2025_runs/mlcg_env/bin/activate
source /data/scratch/kamenrur95/my_envs/cython_mlcg_grappa_env/bin/activate

# Get the directory where this script is located
# SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# # Change to that directory
# cd "$SCRIPT_DIR"
cd ${SLURM_SUBMIT_DIR}

echo "# Working directory:"
pwd

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH=/home/kamenrur95/NO_BACKUP/grappa_priors_test:${PYTHONPATH}
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL


srun mlcg-train_h5_ng fit "$@" --trainer.devices 4 --trainer.num_nodes 1
