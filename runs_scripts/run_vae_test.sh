#!/bin/bash
#SBATCH --account=rrg-ebrahimi
#SBATCH --mem=64G       
#SBATCH --cpus-per-task=12
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=vae-run.out

# module load python/python/3.10
source venv/bin/activate

time python encoder_decoder/vae.py > log1.out 