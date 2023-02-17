#!/bin/bash
#SBATCH --account=rrg-ebrahimi
#SBATCH --mem=32G       
#SBATCH --cpus-per-task=6
#SBATCH --time=00:15:00
#SBATCH --output=atari-vae-run.out

source venv/bin/activate

time pip install -r req_cc.txt 