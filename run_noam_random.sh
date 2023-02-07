#!/bin/bash

#SBATCH --time=00:30:00
#SBATCH --account=rrg-ebrahimi
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-20

source venv/bin/activate
module load python/3.10
export OMP_NUM_THREADS=1 #init weights fails otherwise (see https://github.com/pytorch/pytorch/issues/21956)
runs=${SLURM_ARRAY_TASK_ID}

python ppo_mpe/train.py --n_agents=2 --random_other_agent=0 --log_dir=mpe_logs --seed=$runs 