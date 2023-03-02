#!/bin/bash

#SBATCH --time=20:00:00
#SBATCH --account=def-ebrahimi
#SBATCH --mem=64G
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.err
#SBATCH --cpus-per-task=6
#SBATCH --array=0-20


source venv/bin/activate

export OMP_NUM_THREADS=1 #init weights fails otherwise (see https://github.com/pytorch/pytorch/issues/21956)
cp_update_timestep=${1}

runs=${SLURM_ARRAY_TASK_ID}

# time python conformal-action-prediction/conformal-rl-reps-2.py --cp_update_timestep=$cp_update_timestep --log_dir=cp-reps-penultimate --seed=$runs 
time python conformal-action-prediction/conformal-rl.py --cp_update_timestep=$cp_update_timestep --log_dir="./debug_logs/cam-actions" --seed=$runs 