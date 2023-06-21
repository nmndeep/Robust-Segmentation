#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2  
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=ADE_CVXT_T
#SBATCH --output=../JobLogs/ADE_CVXT_T_%j.out
#SBATCH --error=../JobLogs/ADE_CVXT_T_%j.err
# print info about current job

scontrol show job $SLURM_JOB_ID 

		

python3 ./tools/infer.py --adversarial --cfg $1 --eps $2 

