#!/bin/bash

cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}:${PWD}/..:${PYTHONPATH}"

PRETRAIN_CKPT=${1:-./output_pretrain/checkpoint-200.pth}
DATA_DIR=${2:-./run_tests/data}
DATASET=${3:-ToroOssaba}

python run_finetuning.py \
  --pretrain_ckpt "${PRETRAIN_CKPT}" \
  --data_dir "${DATA_DIR}" \
  --dataset "${DATASET}" \
  --output_dir "./output_finetune_${DATASET}" \
  --linear_probe \
  --batch_size 32 \
  --epochs 100 \
  --lr 5e-3 \
  --model_size base \
  --seed 0
