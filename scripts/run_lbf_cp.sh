#!/bin/bash

#SBATCH --time=40:00:00
#SBATCH --account=rrg-ebrahimi
#SBATCH --mem=64G
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.err
#SBATCH --cpus-per-task=6
#SBATCH --array=0-20


source venv/bin/activate

export OMP_NUM_THREADS=1 #init weights fails otherwise (see https://github.com/pytorch/pytorch/issues/21956)
cp_update_timestep=${1}

runs=${SLURM_ARRAY_TASK_ID}

tensorboard --logdir="./debug_logs/lbf-exact-actions" --host 0.0.0.0 --load_fast false &
time python cam-lbf/exact-actions.py --cp_update_timestep=$cp_update_timestep --log_dir="./debug_logs/lbf-exact-actions" --seed=$runs 