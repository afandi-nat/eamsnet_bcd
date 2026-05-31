"""Visualisasi ablation dari checkpoint yang sudah dilatih.

Dua mode:
    --mode heatmap : grid magnitudo fitur perubahan (gaya peta panas)
    --mode sample  : 1 sampel diuji pada semua konfigurasi (T1|T2|GT|Pred|Error)

Contoh:
    python scripts/visualize_ablation.py --mode sample --idx 25
    python scripts/visualize_ablation.py --mode heatmap --n 5
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from eamsnet.data import build_dataloaders
from eamsnet.models import EAMSNet
from eamsnet.engine import (search_threshold, visualize_ablation_heatmap,
                            visualize_ablation_sample)
from eamsnet.utils import set_seed, get_device
from _config import load_config, add_common_overrides, merge_overrides, config_tag
from train_ablation import ABLATION_CONFIGS


def main():
    parser = add_common_overrides(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--mode", choices=["heatmap", "sample"], default="sample")
    parser.add_argument("--idx", type=int, default=0, help="indeks sampel (mode sample)")
    parser.add_argument("--n", type=int, default=5, help="jumlah kolom (mode heatmap)")
    args = parser.parse_args()

    cfg = merge_overrides(load_config(args.config), args)
    set_seed(cfg["seed"])
    device = get_device()

    (_, _, test_ds), (_, val_loader, _) = build_dataloaders(
        cfg["data_root"], cfg["img_size"], cfg["batch_size"], cfg["num_workers"])

    models_dict, thrs = {}, {}
    for tag, flags in ABLATION_CONFIGS:
        safe = config_tag(cfg, **flags)
        path = os.path.join(cfg["out_dir"], f"ablation_{safe}.pth")
        if not os.path.exists(path):
            print(f"[lewati] checkpoint tidak ditemukan: {path}")
            continue
        m = EAMSNet(cfg["backbone"], pretrained=False, width=cfg["width"], **flags).to(device)
        ck = torch.load(path, map_location=device, weights_only=False)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        models_dict[tag] = m
        thr, _ = search_threshold(m, val_loader, device)
        thrs[tag] = thr

    if not models_dict:
        print("Tidak ada checkpoint ablation. Jalankan scripts/train_ablation.py dulu.")
        return

    if args.mode == "heatmap":
        idxs = np.random.choice(len(test_ds), args.n, replace=False)
        out = os.path.join(cfg["out_dir"], "ablation_heatmap.png")
        visualize_ablation_heatmap(test_ds, device, models_dict, idxs, path=out)
    else:
        out = os.path.join(cfg["out_dir"], f"ablation_sample_{args.idx}.png")
        visualize_ablation_sample(test_ds, device, models_dict, args.idx, thrs=thrs, path=out)
    print(f"Tersimpan -> {out}")


if __name__ == "__main__":
    main()
