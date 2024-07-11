#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8  
#SBATCH --time=19:00:00
#SBATCH --gres=gpu:8
#SBATCH --job-name=SEG_ROB
#SBATCH --output=../JobLogs/SEG_ROB_%j.out
#SBATCH --error=../JobLogs/SEG_ROB_%j.err
# print info about current job

scontrol show job $SLURM_JOB_ID 
export WORLD_SIZE=8


python3 ./tools/train_rob_seg.py --cfg ./configs/ade20k_convnext_cvst.yaml --world_size 8


