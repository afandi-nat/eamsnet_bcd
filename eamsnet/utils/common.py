"""Utilitas umum: reproducibility, device, EMA, scheduler, optimizer, benchmark."""
import copy
import time

import numpy as np
import torch
import torch.optim as optim


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Device: {device} | GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    else:
        print(f"Device: {device}")
    return device


class WarmupCosineScheduler:
    """Linear warmup lalu cosine decay; mempertahankan rasio lr antar param-group."""

    def __init__(self, optimizer, warmup, total, min_lr=1e-6):
        self.opt = optimizer
        self.warmup = warmup
        self.total = total
        self.min_lr = min_lr
        self.base_lr = [g["lr"] for g in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup:
            f = (epoch + 1) / self.warmup
        else:
            prog = (epoch - self.warmup) / (self.total - self.warmup)
            f = self.min_lr / self.base_lr[0] + (1 - self.min_lr / self.base_lr[0]) * 0.5 * (
                1 + np.cos(np.pi * prog))
        for g, lr0 in zip(self.opt.param_groups, self.base_lr):
            g["lr"] = lr0 * f


class ModelEMA:
    """Exponential Moving Average dari bobot model."""

    def __init__(self, model, decay_max=0.999, warmup_steps=2000):
        self.ema = copy.deepcopy(model).eval()
        self.decay_max = decay_max
        self.warmup_steps = warmup_steps
        self.step_n = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.step_n += 1
        d = min(self.decay_max, (1 + self.step_n) / (10 + self.step_n))
        for e, m in zip(self.ema.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach(), alpha=1 - d)
            else:
                e.copy_(m)


def build_optimizer(model, lr_enc=1e-4, lr_other=3e-4, wd=1e-2):
    """AdamW dengan param-group terpisah backbone vs modul lain, dan no-decay utk bias/norm."""
    enc_decay, enc_no, decay, no_decay = [], [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_enc = n.startswith("encoder.")
        if p.ndim <= 1 or "norm" in n.lower() or n.endswith(".bias"):
            (enc_no if is_enc else no_decay).append(p)
        else:
            (enc_decay if is_enc else decay).append(p)
    return optim.AdamW([
        {"params": enc_decay, "lr": lr_enc, "weight_decay": wd},
        {"params": enc_no, "lr": lr_enc, "weight_decay": 0.0},
        {"params": decay, "lr": lr_other, "weight_decay": wd},
        {"params": no_decay, "lr": lr_other, "weight_decay": 0.0},
    ])


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


@torch.no_grad()
def benchmark(model, device, img_size=256, runs=50):
    """Ukur latency (ms) dan throughput (FPS) pada batch tunggal."""
    model.eval()
    a = torch.randn(1, 3, img_size, img_size).to(device)
    b = torch.randn(1, 3, img_size, img_size).to(device)
    for _ in range(10):
        model(a, b)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ts = []
    for _ in range(runs):
        t0 = time.time()
        model(a, b)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append(time.time() - t0)
    ms = float(np.mean(ts) * 1000)
    return ms, 1000 / ms
