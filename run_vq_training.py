import argparse
import os
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from NCT import NCT, WINDOW_LEN, TARGET_CHANNELS
from datasets.data_loader import build_aemg_pretrain_dataset
from modeling_vqaemg import vqaemg_base, vqaemg_large
from utils.utils import (
    set_seed, cosine_scheduler, save_model, MetricLogger, NativeScalerWithGradNormCount,
    init_distributed_mode, get_rank, get_world_size, is_main_process
)


def get_args():
    p = argparse.ArgumentParser("AEMG VQ Training")
    p.add_argument("--data_root", type=str, default="./data_aemg", help="Path to data_aemg")
    p.add_argument("--output_dir", type=str, default="./output_vq", help="Output directory")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--save_freq", type=int, default=20)
    p.add_argument("--model_size", type=str, default="base", choices=["base", "large"])
    p.add_argument("--vocab_size", type=int, default=8192)
    p.add_argument("--code_dim", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--multi_gpu", action="store_true", help="Use DataParallel for 2-4 GPUs")
    p.add_argument("--distributed", action="store_true", help="Use DDP for 8-GPU training (launch with torchrun)")
    p.add_argument("--gpu_ids", type=str, default=None, help="GPU ids, e.g. 0,1. Default: all visible")
    return p.parse_args()


def train_one_epoch(model, loader, optimizer, device, scaler, lr_schedule, start_step):
    model.train()
    logger = MetricLogger()
    for step, batch in enumerate(loader):
        it = start_step + step
        if it < len(lr_schedule):
            for pg in optimizer.param_groups:
                pg["lr"] = lr_schedule[it]
        x = batch[0].float().to(device)
        ic = batch[1].to(device)
        it_b = batch[2].to(device)
        loss, log = model(x, in_chan_matrix=ic, in_time_matrix=it_b)
        optimizer.zero_grad()
        scaler(loss, optimizer, parameters=model.parameters())
        logger.update(**{k.replace("train/", ""): v.item() if torch.is_tensor(v) else v for k, v in log.items()})
    return {k: m.global_avg for k, m in logger.meters.items()}


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in loader:
        x = batch[0].float().to(device)
        ic = batch[1].to(device)
        it_b = batch[2].to(device)
        loss, _ = model(x, in_chan_matrix=ic, in_time_matrix=it_b)
        total_loss += loss.item() * x.shape[0]
        n += x.shape[0]
    model.train()
    return total_loss / n if n > 0 else 0.0


def main():
    args = get_args()
    if args.distributed:
        init_distributed_mode(args)
    else:
        args.gpu = 0
    set_seed(args.seed)
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
    if args.gpu_ids and not args.distributed:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = get_world_size() if args.distributed else torch.cuda.device_count()
    if is_main_process():
        print(f"Using {n_gpu} GPU(s), batch_size={args.batch_size} per GPU, total={args.batch_size * n_gpu}")

    nct = NCT(fs=2000, max_len=256, token_len=WINDOW_LEN, target_channels=TARGET_CHANNELS)
    train_ds, val_ds = build_aemg_pretrain_dataset(args.data_root, nct, datasets=["MCS_EMG", "UCI"])
    train_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0) if len(val_ds) > 0 else None
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0) if len(val_ds) > 0 else None

    if args.model_size == "large":
        model = vqaemg_large(n_embed=args.vocab_size, code_dim=args.code_dim)
    else:
        model = vqaemg_base(n_embed=args.vocab_size, code_dim=args.code_dim)
    model = model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    elif args.multi_gpu and n_gpu > 1:
        model = torch.nn.DataParallel(model)

    lr = args.lr * n_gpu
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = NativeScalerWithGradNormCount()
    n_steps = len(train_loader) * args.epochs
    lr_schedule = cosine_scheduler(lr, args.min_lr, args.epochs, len(train_loader), warmup_epochs=args.warmup_epochs)

    if is_main_process():
        print(f"Training VQ-AEMG for {args.epochs} epochs, {len(train_ds)} samples, lr={lr:.2e}")
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch - 1)
        start_step = (epoch - 1) * len(train_loader)
        stats = train_one_epoch(model, train_loader, optimizer, device, scaler, lr_schedule, start_step)
        val_loss = validate(model, val_loader, device) if (val_loader and is_main_process()) else 0.0
        if is_main_process():
            print(f"Epoch {epoch}: train_loss={stats.get('total_loss', 0):.4f}, val_loss={val_loss:.4f}")
        model_to_save = model.module if hasattr(model, "module") else model
        save_model(argparse.Namespace(output_dir=args.output_dir), model, model_to_save, optimizer, scaler, epoch, args.save_freq)
    if is_main_process():
        print(f"Done in {time.time()-start:.1f}s. Checkpoints in {args.output_dir}")


if __name__ == "__main__":
    main()
