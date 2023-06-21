#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4  
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:4
#SBATCH --job-name=ROB_SEG
#SBATCH --output=../JobLogs/ROB_SEG_%j.out
#SBATCH --error=../JobLogs/ROB_SEG_%j.err
# print info about current job

scontrol show job $SLURM_JOB_ID 
#conda activate main_py

python3 ./tools/train.py --cfg $1  --world_size $2