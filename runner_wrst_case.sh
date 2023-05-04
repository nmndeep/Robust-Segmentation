#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1  
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=ADE_CVXT_T_EVAL
#SBATCH --output=../JobLogs/ADE_CVXT_T_%j.out
#SBATCH --error=../JobLogs/ADE_CVXT_T_%j.err
# print info about current job

scontrol show job $SLURM_JOB_ID 
#conda activate main_py

python3 ./tools/worst_case_stats.py