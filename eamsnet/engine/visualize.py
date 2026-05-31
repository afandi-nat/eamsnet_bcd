"""Visualisasi kualitatif & ablation (disimpan ke PNG)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from ..data.levircd import IMAGENET_MEAN, IMAGENET_STD

_MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)

SHORT = {
    "Baseline": "Baseline",
    "Baseline + ATDAM": "+ ATDAM",
    "Baseline + MSDA": "+ MSDA",
    "Baseline + EABRM": "+ EABRM",
    "Baseline + ATDAM + MSDA": "+ ATDAM+MSDA",
    "Baseline + ATDAM + MSDA + EABRM": "Full (A+M+E)",
}


def _denorm(t):
    return (t * _STD + _MEAN).permute(1, 2, 0).numpy().clip(0, 1)


def _error_map(pm, lv):
    err = np.zeros((*lv.shape, 3))
    err[(pm == 1) & (lv == 1)] = [0, 1, 0]  # TP hijau
    err[(pm == 1) & (lv == 0)] = [1, 0, 0]  # FP merah
    err[(pm == 0) & (lv == 1)] = [0, 0, 1]  # FN biru
    return err


@torch.no_grad()
def visualize_qualitative(model, dataset, device, n=6, thr=0.5, path="qualitative.png"):
    """Grid n sampel: T1 | T2 | GT | Prediction | Error Map untuk satu model."""
    model.eval()
    idxs = np.random.choice(len(dataset), n, replace=False)
    fig, axes = plt.subplots(n, 5, figsize=(20, 4 * n))
    titles = ["T1 Image", "T2 Image", "Ground Truth", "Prediction", "Error Map"]
    for row, idx in enumerate(idxs):
        iA, iB, lb = dataset[idx]
        pred, _ = model(iA.unsqueeze(0).to(device), iB.unsqueeze(0).to(device))
        pm = (torch.sigmoid(pred) > thr).float().cpu().squeeze().numpy()
        lv = lb.squeeze().numpy()
        panels = [_denorm(iA), _denorm(iB), lv, pm, _error_map(pm, lv)]
        for col, img in enumerate(panels):
            kw = {"cmap": "gray"} if col in (2, 3) else {}
            axes[row, col].imshow(img, **kw)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(titles[col], fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


@torch.no_grad()
def _feature_heatmap(model, iA, iB, device, out_hw):
    model.eval()
    _, _, feat = model(iA.unsqueeze(0).to(device), iB.unsqueeze(0).to(device), return_feat=True)
    mag = feat.pow(2).sum(1, keepdim=True).sqrt()
    mag = F.interpolate(mag, out_hw, mode="bilinear", align_corners=True)
    mag = mag.squeeze().float().cpu().numpy()
    mn, mx = mag.min(), mag.max()
    return (mag - mn) / (mx - mn + 1e-8)


@torch.no_grad()
def visualize_ablation_heatmap(dataset, device, models_dict, sample_idxs,
                               path="ablation_heatmap.png"):
    """Grid heatmap: baris T1/T2/label lalu satu baris per konfigurasi.

    models_dict: dict {label: model_terlatih(eval)}.
    """
    tags = list(models_dict.keys())
    row_labels = ["T1", "T2", "label"] + [SHORT.get(t, t) for t in tags]
    n = len(sample_idxs)
    n_rows = len(row_labels)
    fig, axes = plt.subplots(n_rows, n, figsize=(2.1 * n, 2.1 * n_rows))
    if n == 1:
        axes = axes[:, None]
    for c, idx in enumerate(sample_idxs):
        iA, iB, lb = dataset[idx]
        lv = lb.squeeze().numpy()
        axes[0, c].imshow(_denorm(iA))
        axes[1, c].imshow(_denorm(iB))
        axes[2, c].imshow(lv, cmap="gray", vmin=0, vmax=1)
        for r, tag in enumerate(tags, start=3):
            hm = _feature_heatmap(models_dict[tag], iA, iB, device, out_hw=lv.shape)
            axes[r, c].imshow(hm, cmap="jet", vmin=0, vmax=1)
    for r in range(n_rows):
        for c in range(n):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(row_labels[r], rotation=90, fontsize=10,
                              fontweight="bold", labelpad=8, va="center")
    plt.subplots_adjust(wspace=0.02, hspace=0.02, left=0.10, right=0.99, top=0.99, bottom=0.01)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


@torch.no_grad()
def visualize_ablation_sample(dataset, device, models_dict, idx, thrs=None,
                              path="ablation_sample.png"):
    """Satu sampel diuji pada semua konfigurasi.

    Kolom: T1 | T2 | GT | Prediction | Error Map; satu baris per konfigurasi.
    """
    tags = list(models_dict.keys())
    if thrs is None:
        thrs = {t: 0.5 for t in tags}
    iA, iB, lb = dataset[idx]
    lv = lb.squeeze().numpy()
    titles = ["T1 Image", "T2 Image", "Ground Truth", "Prediction", "Error Map"]
    n_rows = len(tags)
    fig, axes = plt.subplots(n_rows, 5, figsize=(13, 2.6 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]
    for r, tag in enumerate(tags):
        m = models_dict[tag]
        out = m(iA.unsqueeze(0).to(device), iB.unsqueeze(0).to(device))[0]
        pm = (torch.sigmoid(out) > thrs.get(tag, 0.5)).float().cpu().squeeze().numpy()
        panels = [_denorm(iA), _denorm(iB), lv, pm, _error_map(pm, lv)]
        for c, img in enumerate(panels):
            kw = {"cmap": "gray", "vmin": 0, "vmax": 1} if c in (2, 3) else {}
            axes[r, c].imshow(img, **kw)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=12, fontweight="bold")
        axes[r, 0].set_ylabel(SHORT.get(tag, tag), rotation=90, fontsize=11,
                              fontweight="bold", labelpad=8, va="center")
    plt.subplots_adjust(wspace=0.03, hspace=0.05, left=0.07, right=0.99, top=0.95, bottom=0.01)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path
