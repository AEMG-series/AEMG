# AEMG: Any Electromyography

## Reproduction guide

### 1. Pre-training (Stage 1 - VQ)

```bash
cd AEMG
bash scripts/pretrain_vq_8gpu.sh
# or
python run_vq_training.py --data_root ./data_aemg --output_dir ./output_vq --epochs 100
```

Details:

```shellscript
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
```

### 2. Pre-training (Stage 2 - MEM)

```bash
bash scripts/pretrain_mem_8gpu.sh ./output_vq/checkpoint-100.pth
# or
python run_pretraining.py --data_root ./data_aemg --vq_checkpoint ./output_vq/checkpoint-100.pth --output_dir ./output_pretrain --epochs 200
```

Details:

```shellscript
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
```

### 3. Downstream fine-tuning (LOSO)

```bash
bash scripts/finetune_loso.sh ./output_pretrain/checkpoint-200.pth ./run_tests/data ToroOssaba
# or
python run_finetuning.py --pretrain_ckpt ./output_pretrain/checkpoint-200.pth --dataset ToroOssaba --data_dir ./run_tests/data --linear_probe
```

Details:

```shellscript
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

```

Supported datasets: `ninaproDB4`, `ToroOssaba`, `EMG_EPN_612`

## Data layout

- **Pre-training**: `AEMG/data_aemg/`
- **Downstream**: under `AEMG/run_tests/data/`: `ninaproDB4_folder`, `ToroOssaba_folder`, `EMG-EPN612_folder`

## Full-GPU training

### 8x RTX 4090

```bash
# Stage 1: VQ, 8 GPUs, total batch 512
bash scripts/pretrain_vq_8gpu.sh
# or
torchrun --nproc_per_node=8 run_vq_training.py --distributed --batch_size 64

# Stage 2: MEM pre-training
bash scripts/pretrain_mem_8gpu.sh ./output_vq/checkpoint-100.pth
# or
torchrun --nproc_per_node=8 run_pretraining.py --distributed --vq_checkpoint ./output_vq/checkpoint-100.pth --batch_size 64
```

See `requirements.txt`.
