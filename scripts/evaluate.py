"""Evaluasi checkpoint EAMSNet pada test set LEVIR-CD.

Contoh:
    python scripts/evaluate.py --ckpt outputs/best_eamsnet.pth --tta --msf --visualize
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from eamsnet.data import build_dataloaders
from eamsnet.models import EAMSNet
from eamsnet.engine import (search_threshold, evaluate_full,
                            search_threshold_msf, evaluate_msf, visualize_qualitative)
from eamsnet.utils import set_seed, get_device, count_params, benchmark
from _config import load_config, add_common_overrides, merge_overrides


def _show(name, m):
    print("\n" + "=" * 60 + f"\n  {name}\n" + "=" * 60)
    for k, v in m.items():
        print(f"  {k:>12s}: {v:.2f}%")


def main():
    parser = add_common_overrides(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--ckpt", required=True, help="path checkpoint (best_eamsnet.pth)")
    parser.add_argument("--tta", action="store_true", help="aktifkan flip-TTA")
    parser.add_argument("--postproc", action="store_true", help="aktifkan post-processing")
    parser.add_argument("--msf", action="store_true", help="evaluasi multi-scale + flip TTA")
    parser.add_argument("--visualize", action="store_true", help="simpan grid kualitatif")
    args = parser.parse_args()

    cfg = merge_overrides(load_config(args.config), args)
    set_seed(cfg["seed"])
    device = get_device()
    os.makedirs(cfg["out_dir"], exist_ok=True)

    (_, _, test_ds), (_, val_loader, test_loader) = build_dataloaders(
        cfg["data_root"], cfg["img_size"], cfg["batch_size"], cfg["num_workers"])

    model = EAMSNet(cfg["backbone"], pretrained=False, width=cfg["width"],
                    use_atdam=cfg["use_atdam"], use_msda=cfg["use_msda"],
                    use_eabrm=cfg["use_eabrm"]).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(f"Loaded {args.ckpt} | {count_params(model):.2f}M params")

    thr, val_f1 = search_threshold(model, val_loader, device)
    print(f"Threshold optimal (VAL): {thr:.2f}  (Val-F1 @thr {val_f1:.2f}%)")

    _show("TEST — standard (thr=0.5)", evaluate_full(model, test_loader, device, thr=0.5))
    _show("TEST — threshold-optimized", evaluate_full(model, test_loader, device, thr=thr))
    if args.tta:
        _show("TEST — threshold + TTA",
              evaluate_full(model, test_loader, device, thr=thr, tta=True))
    if args.postproc:
        _show("TEST — threshold + TTA + post-proc",
              evaluate_full(model, test_loader, device, thr=thr, tta=True, use_postproc=True))
    if args.msf:
        scales = (0.75, 1.0, 1.25)
        thr_msf, _ = search_threshold_msf(model, val_loader, device, scales)
        _show(f"TEST — multi-scale+flip TTA (thr={thr_msf:.2f})",
              evaluate_msf(model, test_loader, device, scales, thr=thr_msf))

    ms, fps = benchmark(model, device, cfg["img_size"])
    print(f"\nComplexity: {count_params(model):.2f}M params | {ms:.1f} ms | {fps:.1f} FPS")

    if args.visualize:
        out_png = os.path.join(cfg["out_dir"], "qualitative.png")
        visualize_qualitative(model, test_ds, device, n=6, thr=thr, path=out_png)
        print(f"Qualitative grid -> {out_png}")


if __name__ == "__main__":
    main()
