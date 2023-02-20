#!/bin/bash

#SBATCH --time=10:00:00
#SBATCH --account=rrg-ebrahimi
#SBATCH --mem=64G
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.err
#SBATCH --cpus-per-task=6
#SBATCH --array=0-20


source venv/bin/activate

export OMP_NUM_THREADS=1 #init weights fails otherwise (see https://github.com/pytorch/pytorch/issues/21956)
env=${1}
baseline=${2}

runs=${SLURM_ARRAY_TASK_ID}

time python baselines/baselines.py --envname=$env --baseline=$baseline --log_dir=env-search-2 --seed=$runs 