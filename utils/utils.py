import os
import math
import random
import numpy as np
import torch
import torch.distributed as dist


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_scheduler(base_value, final_value, epochs, niter_per_epoch, warmup_epochs=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_epoch if warmup_epochs > 0 else 0
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    if warmup_iters > 0:
        warmup_schedule = np.linspace(0, base_value, warmup_iters)
    iters = np.arange(epochs * niter_per_epoch - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    schedule = np.concatenate((warmup_schedule, schedule))
    return schedule


class NativeScalerWithGradNormCount:
    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        if self._scaler is not None:
            self._scaler.scale(loss).backward(create_graph=create_graph)
            if update_grad:
                if clip_grad is not None:
                    self._scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
                self._scaler.step(optimizer)
                self._scaler.update()
        else:
            loss.backward(create_graph=create_graph)
            if update_grad:
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
                optimizer.step()
        return 1.0

    def state_dict(self):
        return self._scaler.state_dict() if self._scaler else {}


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ.get("LOCAL_RANK", 0))
    else:
        args.distributed = False
        return
    args.distributed = True
    torch.cuda.set_device(args.gpu)
    dist.init_process_group(backend="nccl", init_method="env://")
    dist.barrier()


def save_model(args, model, model_without_ddp, optimizer, loss_scaler, epoch, save_ckpt_freq=20):
    if epoch % save_ckpt_freq != 0:
        return
    if is_dist_avail_and_initialized() and not is_main_process():
        return
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt = {
        'model': model_without_ddp.state_dict(),
        'epoch': epoch,
        'optimizer': optimizer.state_dict() if optimizer else None,
    }
    if hasattr(loss_scaler, 'state_dict') and loss_scaler.state_dict():
        ckpt['scaler'] = loss_scaler.state_dict()
    torch.save(ckpt, os.path.join(args.output_dir, f'checkpoint-{epoch}.pth'))


def load_model(path, model, optimizer=None, loss_scaler=None):
    ckpt = torch.load(path, map_location='cpu')
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    if optimizer and 'optimizer' in ckpt and ckpt['optimizer']:
        optimizer.load_state_dict(ckpt['optimizer'])
    if loss_scaler and 'scaler' in ckpt:
        loss_scaler.load_state_dict(ckpt['scaler'])
    return ckpt.get('epoch', 0)


class MetricLogger:
    def __init__(self, delimiter="  "):
        self.meters = {}
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if k not in self.meters:
                self.meters[k] = SmoothedValue()
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        raise AttributeError

    def synchronize_between_processes(self):
        pass

    def __str__(self):
        return self.delimiter.join([f"{k}: {v.global_avg:.4f}" for k, v in self.meters.items()])


class SmoothedValue:
    def __init__(self, window_size=20, fmt="{value:.4f}"):
        self.deque = []
        self.total = 0.0
        self.count = 0
        self.fmt = fmt
        self.window_size = window_size

    def update(self, value, n=1):
        self.deque.append(value)
        if len(self.deque) > self.window_size:
            self.deque.pop(0)
        self.count += n
        self.total += value * n

    @property
    def global_avg(self):
        return self.total / self.count if self.count > 0 else 0
