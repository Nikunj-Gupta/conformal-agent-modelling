#!/bin/bash
#SBATCH --account=rrg-ebrahimi
#SBATCH --mem=32G       
#SBATCH --cpus-per-task=6
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=atari-vae-run.out

# module load python/python/3.10
source venv/bin/activate

tensorboard --logdir=encoder_decoder/logs --host 0.0.0.0 --load_fast false &
time python encoder_decoder/test_atari.py > log1.out 