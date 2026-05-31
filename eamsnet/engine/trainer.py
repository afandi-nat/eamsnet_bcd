"""Loop pelatihan & evaluasi per-epoch, pencarian threshold, post-processing."""
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from ..utils.metrics import CDMetrics


def train_epoch_ema(model, ema, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    metrics = CDMetrics()
    for imgA, imgB, label in loader:
        imgA, imgB, label = imgA.to(device), imgB.to(device), label.to(device)
        optimizer.zero_grad()
        with autocast():
            out, aux = model(imgA, imgB)
            loss = criterion(out, aux, label)
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)
        total_loss += loss.item()
        metrics.update(out.detach().float(), label)
    return total_loss / len(loader), metrics.compute()


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    metrics = CDMetrics()
    for imgA, imgB, label in loader:
        imgA, imgB, label = imgA.to(device), imgB.to(device), label.to(device)
        with autocast():
            out, _ = model(imgA, imgB)
            loss = criterion._loss(out, label)
        total_loss += loss.item()
        metrics.update(out.float(), label)
    return total_loss / len(loader), metrics.compute()


@torch.no_grad()
def search_threshold(model, loader, device, grid=None):
    """Cari threshold (di set validasi) yang memaksimalkan F1."""
    if grid is None:
        grid = np.arange(0.30, 0.71, 0.02)
    model.eval()
    best_thr, best_f1 = 0.5, -1
    stats = {round(t, 3): [0, 0, 0] for t in grid}
    for imgA, imgB, label in loader:
        imgA, imgB = imgA.to(device), imgB.to(device)
        with autocast():
            out, _ = model(imgA, imgB)
        p = torch.sigmoid(out.float()).cpu().numpy().flatten()
        t = label.long().numpy().flatten()
        for thr in grid:
            pb = p > thr
            s = stats[round(thr, 3)]
            s[0] += int(((pb == 1) & (t == 1)).sum())
            s[1] += int(((pb == 1) & (t == 0)).sum())
            s[2] += int(((pb == 0) & (t == 1)).sum())
    for thr in grid:
        tp, fp, fn = stats[round(thr, 3)]
        e = 1e-7
        P = tp / (tp + fp + e)
        R = tp / (tp + fn + e)
        F1 = 2 * P * R / (P + R + e)
        if F1 > best_f1:
            best_f1, best_thr = F1, float(thr)
    return best_thr, best_f1 * 100


def postprocess(prob_map, thr, min_area=4, fill_holes=True):
    """Binarisasi + isi lubang + buang komponen kecil (butuh scipy)."""
    try:
        from scipy import ndimage
    except ImportError:
        return (prob_map > thr).astype(np.uint8)
    binm = (prob_map > thr).astype(np.uint8)
    if fill_holes:
        binm = ndimage.binary_fill_holes(binm).astype(np.uint8)
    if min_area > 0:
        lab, n = ndimage.label(binm)
        if n > 0:
            sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
            for k, sz in enumerate(sizes, 1):
                if sz < min_area:
                    binm[lab == k] = 0
    return binm


@torch.no_grad()
def evaluate_full(model, loader, device, thr=0.5, tta=False, use_postproc=False):
    """Evaluasi test dengan opsi flip-TTA dan post-processing."""
    model.eval()
    metrics = CDMetrics()
    for imgA, imgB, label in loader:
        imgA, imgB = imgA.to(device), imgB.to(device)
        with autocast():
            p = torch.sigmoid(model(imgA, imgB)[0].float())
            if tta:
                p1 = torch.sigmoid(model(imgA.flip(3), imgB.flip(3))[0].float()).flip(3)
                p2 = torch.sigmoid(model(imgA.flip(2), imgB.flip(2))[0].float()).flip(2)
                p3 = torch.sigmoid(
                    model(imgA.flip(2).flip(3), imgB.flip(2).flip(3))[0].float()
                ).flip(2).flip(3)
                p = (p + p1 + p2 + p3) / 4
        p_np = p.cpu().numpy()
        for b in range(p_np.shape[0]):
            pm = p_np[b, 0]
            if use_postproc:
                pbin = postprocess(pm, thr)
            else:
                pbin = (pm > thr).astype(np.uint8)
            metrics.update_from_binary(torch.from_numpy(pbin), label[b])
    return metrics.compute()
