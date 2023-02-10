#!/bin/bash

#SBATCH --time=20:00:00
#SBATCH --account=rrg-ebrahimi
#SBATCH --mem=64G
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.err
#SBATCH --cpus-per-task=6
#SBATCH --array=0-20

source venv/bin/activate
export OMP_NUM_THREADS=1 #init weights fails otherwise (see https://github.com/pytorch/pytorch/issues/21956)
runs=${SLURM_ARRAY_TASK_ID}

time python conformal-action-prediction/conformal-rl-2.py --n_agents=2 --log_dir=new_cp_mpe_logs --seed=$runs --cp_update_timestep 50 
