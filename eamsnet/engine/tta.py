"""Inferensi multi-scale + flip TTA (tanpa retraining)."""
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from ..utils.metrics import CDMetrics


@torch.no_grad()
def predict_msf(model, imgA, imgB, scales=(0.75, 1.0, 1.25), flips=True):
    """Rata-rata probabilitas di beberapa skala dan flip."""
    H, W = imgA.shape[2:]
    acc = torch.zeros(imgA.shape[0], 1, H, W, device=imgA.device)
    n = 0

    def run(a, b):
        nonlocal acc, n
        variants = [(a, b)]
        if flips:
            variants += [
                (a.flip(3), b.flip(3)),
                (a.flip(2), b.flip(2)),
                (a.flip(2).flip(3), b.flip(2).flip(3)),
            ]
        for i, (aa, bb) in enumerate(variants):
            with autocast():
                p = torch.sigmoid(model(aa, bb)[0].float())
            if i == 1:
                p = p.flip(3)
            elif i == 2:
                p = p.flip(2)
            elif i == 3:
                p = p.flip(2).flip(3)
            if p.shape[2:] != (H, W):
                p = F.interpolate(p, (H, W), mode="bilinear", align_corners=False)
            acc += p
            n += 1

    for s in scales:
        if s == 1.0:
            run(imgA, imgB)
        else:
            nh, nw = int(round(H * s)), int(round(W * s))
            a = F.interpolate(imgA, (nh, nw), mode="bilinear", align_corners=False)
            b = F.interpolate(imgB, (nh, nw), mode="bilinear", align_corners=False)
            run(a, b)
    return acc / n


@torch.no_grad()
def search_threshold_msf(model, loader, device, scales, grid=None):
    if grid is None:
        grid = np.arange(0.30, 0.71, 0.02)
    model.eval()
    stats = {round(t, 3): [0, 0, 0] for t in grid}
    for imgA, imgB, label in loader:
        imgA, imgB = imgA.to(device), imgB.to(device)
        p = predict_msf(model, imgA, imgB, scales).cpu().numpy().flatten()
        t = label.long().numpy().flatten()
        for thr in grid:
            pb = p > thr
            s = stats[round(thr, 3)]
            s[0] += int(((pb == 1) & (t == 1)).sum())
            s[1] += int(((pb == 1) & (t == 0)).sum())
            s[2] += int(((pb == 0) & (t == 1)).sum())
    best_thr, best_f1 = 0.5, -1
    for thr in grid:
        tp, fp, fn = stats[round(thr, 3)]
        e = 1e-7
        P = tp / (tp + fp + e)
        R = tp / (tp + fn + e)
        F1 = 2 * P * R / (P + R + e)
        if F1 > best_f1:
            best_f1, best_thr = F1, float(thr)
    return best_thr, best_f1 * 100


@torch.no_grad()
def evaluate_msf(model, loader, device, scales, thr=0.5):
    model.eval()
    met = CDMetrics()
    for imgA, imgB, label in loader:
        imgA, imgB = imgA.to(device), imgB.to(device)
        p = predict_msf(model, imgA, imgB, scales).cpu().numpy()
        for b in range(p.shape[0]):
            met.update_from_binary(torch.from_numpy((p[b, 0] > thr).astype(np.uint8)), label[b])
    return met.compute()
