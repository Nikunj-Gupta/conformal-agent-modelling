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
env=${1}
baseline=${2}
obstacles=${3}

runs=${SLURM_ARRAY_TASK_ID}

# time python ../baselines/baselines.py --envname=$env --baseline=$baseline --log_dir="debug_logs/simple-tag-1_adv-2_obs/" --seed=$runs 
# time python baselines/baselines.py --envname=$env --baseline=$baseline --num_good=2 --log_dir="debug_logs/simple_spread_all_baselines/" --seed=$runs --max_episodes=30000
time python baselines/baselines.py \
                    --envname=$env \
                    --baseline=$baseline \
                    --num_good=1 \
                    --num_adversaries=2 \
                    --num_obstacles=$obstacles \
                    --log_dir="./debug_logs/simple_tag_adversary_cooperation_all_baselines/" \
                    --seed=$runs \
                    --max_episodes=200000