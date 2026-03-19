#!/bin/bash

cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}:${PWD}/..:${PYTHONPATH}"

VQ_CKPT=${1:-./output_vq/checkpoint-100.pth}

torchrun --nproc_per_node=8 run_pretraining.py \
  --distributed \
  --data_root ./data_aemg \
  --vq_checkpoint "${VQ_CKPT}" \
  --output_dir ./output_pretrain \
  --batch_size 64 \
  --epochs 200 \
  --lr 5e-4 \
  --mask_ratio 0.5 \
  --num_workers 8 \
  --save_freq 20
