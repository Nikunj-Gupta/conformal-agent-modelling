#!/bin/bash

#SBATCH --time=00:30:00
#SBATCH --account=rrg-ebrahimi
#SBATCH --mem=32G
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.err
#SBATCH --cpus-per-task=4
#SBATCH --job-name=mpe_giam
#SBATCH --array=0-20

source venv/bin/activate
export OMP_NUM_THREADS=1 #init weights fails otherwise (see https://github.com/pytorch/pytorch/issues/21956)
runs=${SLURM_ARRAY_TASK_ID}

python giam/giam.py --n_agents=2 --log_dir=mpe_logs --seed=$runs 