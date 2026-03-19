#!/bin/bash

cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}:${PWD}/..:${PYTHONPATH}"

torchrun --nproc_per_node=8 run_vq_training.py \
  --distributed \
  --data_root ./data_aemg \
  --output_dir ./output_vq \
  --batch_size 64 \
  --epochs 100 \
  --lr 5e-5 \
  --num_workers 8 \
  --save_freq 20
