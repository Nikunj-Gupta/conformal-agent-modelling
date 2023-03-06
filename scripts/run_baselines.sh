#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --account=def-ebrahimi
#SBATCH --mem=64G
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.err
#SBATCH --cpus-per-task=6
#SBATCH --array=0-20


source venv/bin/activate

export OMP_NUM_THREADS=1 #init weights fails otherwise (see https://github.com/pytorch/pytorch/issues/21956)
env=${1}
baseline=${2}
# obstacles=${3}
# adversaries=${4}

runs=${SLURM_ARRAY_TASK_ID}

# time python ../baselines/baselines.py --envname=$env --baseline=$baseline --log_dir="debug_logs/simple-tag-1_adv-2_obs/" --seed=$runs 
# time python baselines/baselines.py --envname=$env --baseline=$baseline --num_good=2 --log_dir="debug_logs/simple_spread_all_baselines/" --seed=$runs --max_episodes=30000
# tensorboard --logdir="./debug_logs/" --host 0.0.0.0 --load_fast false & 
# time python baselines/baselines.py \
#                     --envname=$env \
#                     --baseline=$baseline \
#                     --num_good=2 \
#                     --num_adversaries=$adversaries \
#                     --num_obstacles=$obstacles \
#                     --log_dir="./debug_logs/simple_tag_all_baselines/" \
#                     --seed=$runs \
#                     --max_episodes=750000 
tensorboard --logdir="./debug_logs/" --host 0.0.0.0 --load_fast false & 
time python baselines/baselines.py \
                    --envname=$env \
                    --baseline=$baseline \
                    --num_good=2 \
                    --num_adversaries=4 \
                    --num_obstacles=1 \
                    --num_food=2 \
                    --num_forests=2 \
                    --log_dir="./debug_logs/simple_world_all_baselines/" \
                    --seed=$runs \
                    --max_episodes=500000 
