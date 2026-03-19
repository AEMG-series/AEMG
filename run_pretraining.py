import argparse
import os
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from NCT import NCT, WINDOW_LEN, TARGET_CHANNELS
from datasets.data_loader import build_aemg_pretrain_dataset
from modeling_pretrain import aemg_pretrain_base, aemg_pretrain_large, random_masking
from modeling_vqaemg import vqaemg_base, vqaemg_large
from utils.utils import (
    set_seed, cosine_scheduler, save_model, MetricLogger, NativeScalerWithGradNormCount,
    init_distributed_mode, get_world_size, is_main_process
)


def get_args():
    p = argparse.ArgumentParser("AEMG MEM Pre-training")
    p.add_argument("--data_root", type=str, default="./data_aemg")
    p.add_argument("--vq_checkpoint", type=str, required=True, help="Path to VQ checkpoint")
    p.add_argument("--output_dir", type=str, default="./output_pretrain")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--mask_ratio", type=float, default=0.5)
    p.add_argument("--save_freq", type=int, default=20)
    p.add_argument("--model_size", type=str, default="base", choices=["base", "large"])
    p.add_argument("--vocab_size", type=int, default=8192)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--multi_gpu", action="store_true", help="Use DataParallel for 2-4 GPUs")
    p.add_argument("--distributed", action="store_true", help="Use DDP for 8-GPU training (launch with torchrun)")
    p.add_argument("--gpu_ids", type=str, default=None, help="GPU ids, e.g. 0,1. Default: all visible")
    return p.parse_args()


def train_one_epoch(model, vq_model, loader, optimizer, device, scaler, lr_schedule, start_step, mask_ratio):
    model.train()
    vq_model.eval()
    loss_fn = nn.CrossEntropyLoss()
    logger = MetricLogger()
    for step, batch in enumerate(loader):
        it = start_step + step
        if it < len(lr_schedule):
            for pg in optimizer.param_groups:
                pg["lr"] = lr_schedule[it]
        x = batch[0].float().to(device)
        ic = batch[1].to(device)
        it_b = batch[2].to(device)
        with torch.no_grad():
            labels = vq_model.get_codebook_indices(x, in_chan_matrix=ic, in_time_matrix=it_b)
        mask = random_masking(x, mask_ratio).to(device)
        if not mask.any():
            continue
        logits = model(x, mask_bool_matrix=mask, in_chan_matrix=ic, in_time_matrix=it_b, return_all_tokens=False)
        target = labels[mask]
        loss = loss_fn(logits, target)
        optimizer.zero_grad()
        scaler(loss, optimizer, parameters=model.parameters())
        acc = (logits.argmax(-1) == target).float().mean().item()
        logger.update(loss=loss.item(), acc=acc)
    return {k: m.global_avg for k, m in logger.meters.items()}


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
    train_ds, _ = build_aemg_pretrain_dataset(args.data_root, nct, datasets=["MCS_EMG", "UCI"])
    train_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    if args.model_size == "large":
        vq_model = vqaemg_large(n_embed=args.vocab_size)
    else:
        vq_model = vqaemg_base(n_embed=args.vocab_size)
    ckpt = torch.load(args.vq_checkpoint, map_location="cpu")
    vq_model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    vq_model = vq_model.to(device).eval()
    for p in vq_model.parameters():
        p.requires_grad = False

    if args.model_size == "large":
        model = aemg_pretrain_large(vocab_size=args.vocab_size)
    else:
        model = aemg_pretrain_base(vocab_size=args.vocab_size)
    model = model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        vq_model = torch.nn.parallel.DistributedDataParallel(vq_model, device_ids=[args.gpu])
    elif args.multi_gpu and n_gpu > 1:
        model = torch.nn.DataParallel(model)
        vq_model = torch.nn.DataParallel(vq_model)

    lr = args.lr * n_gpu
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scaler = NativeScalerWithGradNormCount()
    lr_schedule = cosine_scheduler(lr, args.min_lr, args.epochs, len(train_loader), warmup_epochs=args.warmup_epochs)

    if is_main_process():
        print(f"Pre-training AEMG MEM for {args.epochs} epochs, mask_ratio={args.mask_ratio}, lr={lr:.2e}")
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch - 1)
        start_step = (epoch - 1) * len(train_loader)
        stats = train_one_epoch(model, vq_model, train_loader, optimizer, device, scaler,
                               lr_schedule, start_step, args.mask_ratio)
        if is_main_process():
            print(f"Epoch {epoch}: loss={stats.get('loss', 0):.4f}, acc={stats.get('acc', 0):.4f}")
        model_to_save = model.module if hasattr(model, "module") else model
        save_model(argparse.Namespace(output_dir=args.output_dir), model, model_to_save, optimizer, scaler, epoch, args.save_freq)
    if is_main_process():
        print(f"Done in {time.time()-start:.1f}s. Checkpoints in {args.output_dir}")


if __name__ == "__main__":
    main()
