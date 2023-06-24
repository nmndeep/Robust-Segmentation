#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# rn50_configs/rn50_16_epochs.yaml
# #--config-file $1 \
sleep 2s

python3 ./tools/train.py --cfg ./configs/ade20k_convnext_cvst.yaml --world_size 8
