"""Metrik deteksi perubahan: Precision, Recall, F1, IoU, OA, Kappa."""
import torch


class CDMetrics:
    """Akumulasi confusion-matrix biner lalu hitung metrik (dalam persen)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = self.fp = self.tn = self.fn = 0

    def update_from_binary(self, pbin, target):
        p = pbin.long().cpu().numpy().flatten()
        t = target.long().cpu().numpy().flatten()
        self.tp += int(((p == 1) & (t == 1)).sum())
        self.fp += int(((p == 1) & (t == 0)).sum())
        self.tn += int(((p == 0) & (t == 0)).sum())
        self.fn += int(((p == 0) & (t == 1)).sum())

    def update(self, pred, target, thr=0.5):
        self.update_from_binary((torch.sigmoid(pred) > thr), target)

    def compute(self):
        e = 1e-7
        P = self.tp / (self.tp + self.fp + e)
        R = self.tp / (self.tp + self.fn + e)
        F1 = 2 * P * R / (P + R + e)
        IoU = self.tp / (self.tp + self.fp + self.fn + e)
        OA = (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn + e)
        N = self.tp + self.fp + self.tn + self.fn + e
        pe = ((self.tp + self.fp) * (self.tp + self.fn)
              + (self.tn + self.fp) * (self.tn + self.fn)) / N ** 2
        K = (OA - pe) / (1 - pe + e)
        return {k: v * 100 for k, v in
                {"Precision": P, "Recall": R, "F1": F1, "IoU": IoU, "OA": OA, "Kappa": K}.items()}
