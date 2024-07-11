#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

sleep 2s

source init conda environement
conda activate env


python3 tools/train_rob_seg.py --cfg ./configs/voc_pspnet.yaml --world_size 8
