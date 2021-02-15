#!/bin/bash

# export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64/:$LD_LIBRARY_PATH
# export PATH=/usr/local/cuda-10.1/bin/:$PATH
# source var-caps/bin/activate

CUDA_VISIBLE_DEVICES=3 python -u main.py |& tee -a logs/logs_7
