#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4  
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:8
#SBATCH --job-name=ADE_CVXT_T
#SBATCH --output=../JobLogs/ADE_CVXT_T_%j.out
#SBATCH --error=../JobLogs/ADE_CVXT_T_%j.err
# print info about current job

scontrol show job $SLURM_JOB_ID 
#conda activate main_py

python3 ./tools/train.py --cfg $1  --world_size $2