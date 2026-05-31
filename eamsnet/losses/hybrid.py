"""Fungsi loss untuk deteksi perubahan.

HybridLoss menggabungkan Focal + Tversky + Edge-Aware BCE + Lovasz hinge,
serta deep supervision pada keluaran auxiliary decoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, smooth=0.02):
        super().__init__()
        self.alpha, self.gamma, self.smooth = alpha, gamma, smooth

    def forward(self, pred, target):
        pred, target = pred.float(), target.float()
        target = target * (1 - self.smooth) + 0.5 * self.smooth
        p = torch.sigmoid(pred).clamp(1e-6, 1 - 1e-6)
        bce = -(target * torch.log(p) + (1 - target) * torch.log(1 - p))
        pt = target * p + (1 - target) * (1 - p)
        w = (target * self.alpha + (1 - target) * (1 - self.alpha)) * (1 - pt) ** self.gamma
        return (w * bce).mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.a, self.b, self.smooth = alpha, beta, smooth

    def forward(self, pred, target):
        p = torch.sigmoid(pred.float())
        t = target.float()
        tp = (p * t).sum(dim=(2, 3))
        fp = (p * (1 - t)).sum(dim=(2, 3))
        fn = ((1 - p) * t).sum(dim=(2, 3))
        tv = (tp + self.smooth) / (tp + self.a * fp + self.b * fn + self.smooth)
        return 1 - tv.mean()


class EdgeAwareLoss(nn.Module):
    def __init__(self, edge_weight=3.0):
        super().__init__()
        self.ew = edge_weight

    def forward(self, pred, target):
        p = torch.sigmoid(pred.float()).clamp(1e-6, 1 - 1e-6)
        target = target.float()
        edge = F.max_pool2d(target, 3, 1, 1) - (1 - F.max_pool2d(1 - target, 3, 1, 1))
        w = 1.0 + (self.ew - 1.0) * edge
        bce = -(target * torch.log(p) + (1 - target) * torch.log(1 - p))
        return (w * bce).mean()


def lovasz_hinge(logits, labels):
    logits = logits.reshape(-1)
    labels = labels.reshape(-1).float()
    if labels.sum() == 0:
        return logits.mean() * 0.0
    signs = 2.0 * labels - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, descending=True)
    gt_sorted = labels[perm]
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    inter = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1.0 - inter / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:p - 1]
    return torch.dot(F.relu(errors_sorted), jaccard)


class HybridLoss(nn.Module):
    def __init__(self, ds_weights=(0.3, 0.2, 0.1)):
        super().__init__()
        self.focal = FocalLoss()
        self.tversky = TverskyLoss()
        self.edge = EdgeAwareLoss()
        self.ds_w = list(ds_weights)

    def _loss(self, pred, target):
        pred, target = pred.float(), target.float()
        return (self.focal(pred, target) + self.tversky(pred, target)
                + 0.5 * self.edge(pred, target) + 0.5 * lovasz_hinge(pred, target))

    def forward(self, main, aux_list, target):
        loss = self._loss(main, target)
        for i, aux in enumerate(aux_list):
            if i < len(self.ds_w):
                loss += self.ds_w[i] * self._loss(aux, target)
        return loss
