#!/bin/sh

#init conda environment
# source init conda 
# conda activate environment

CUDA_VISIBLE_DEVICES=0 python3 -m tools.infer --cfg ./configs/pascalvoc_convnext.yaml --adversarial --eps $1
