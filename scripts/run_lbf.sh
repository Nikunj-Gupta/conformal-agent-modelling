#!/bin/bash

#SBATCH --time=10:00:00
#SBATCH --account=def-ebrahimi
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.err
#SBATCH --cpus-per-task=6
#SBATCH --array=0-20


source venv/bin/activate

export OMP_NUM_THREADS=1 #init weights fails otherwise (see https://github.com/pytorch/pytorch/issues/21956)
baseline=${1}

runs=${SLURM_ARRAY_TASK_ID}

tensorboard --logdir="./debug_logs/lbf" --host 0.0.0.0 --load_fast false & 
time python baselines/baselines_lbf.py \
                    --env="Foraging-12x12-2p-4f-v1" \
                    --baseline=$baseline \
                    --log_dir="./debug_logs/lbf/" \
                    --seed=$runs
