"""Latih satu model EAMSNet pada LEVIR-CD (mendukung resume).

Contoh:
    python scripts/train.py --data-root /path/LEVIR-CD-256 --epochs 250
"""
import argparse
import os
import sys
import time

import torch
from torch.cuda.amp import GradScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from eamsnet.data import build_dataloaders
from eamsnet.losses import HybridLoss
from eamsnet.models import EAMSNet
from eamsnet.engine import train_epoch_ema, evaluate
from eamsnet.utils import (set_seed, get_device, WarmupCosineScheduler, ModelEMA,
                           build_optimizer, count_params)
from _config import load_config, add_common_overrides, merge_overrides


def main():
    parser = add_common_overrides(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--resume", action="store_true", help="lanjut dari checkpoint terakhir")
    args = parser.parse_args()

    cfg = merge_overrides(load_config(args.config), args)
    set_seed(cfg["seed"])
    device = get_device()
    os.makedirs(cfg["out_dir"], exist_ok=True)

    _, (train_loader, val_loader, _) = build_dataloaders(
        cfg["data_root"], cfg["img_size"], cfg["batch_size"], cfg["num_workers"])

    model = EAMSNet(cfg["backbone"], pretrained=cfg["pretrained"], width=cfg["width"],
                    use_atdam=cfg["use_atdam"], use_msda=cfg["use_msda"],
                    use_eabrm=cfg["use_eabrm"]).to(device)
    print(f"Backbone={cfg['backbone']} width={cfg['width']} | "
          f"{count_params(model):.2f}M params")

    crit = HybridLoss().to(device)
    opt = build_optimizer(model, cfg["lr_encoder"], cfg["lr_other"], cfg["weight_decay"])
    sched = WarmupCosineScheduler(opt, cfg["warmup"], cfg["epochs"])
    scaler = GradScaler()
    ema = ModelEMA(model) if cfg["ema"] else None

    best_path = os.path.join(cfg["out_dir"], "best_eamsnet.pth")
    last_path = os.path.join(cfg["out_dir"], "last_eamsnet.pth")
    start_epoch, best_f1, pc = 1, 0.0, 0

    if args.resume and os.path.exists(last_path):
        st = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        if ema is not None and st.get("ema") is not None:
            ema.ema.load_state_dict(st["ema"])
            ema.step_n = st["ema_step"]
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        start_epoch, best_f1, pc = st["epoch"] + 1, st["best_f1"], st["patience"]
        print(f"[resume] lanjut dari epoch {start_epoch} (best Val-F1 {best_f1:.2f}%)")

    print("=" * 70 + "\n  EAMSNet training on LEVIR-CD\n" + "=" * 70)
    for ep in range(start_epoch, cfg["epochs"] + 1):
        t0 = time.time()
        sched.step(ep - 1)
        _, tr_m = train_epoch_ema(model, ema, train_loader, crit, opt, scaler, device)
        _, vl_raw = evaluate(model, val_loader, crit, device)
        if ema is not None:
            _, vl_ema = evaluate(ema.ema, val_loader, crit, device)
            if vl_ema["F1"] >= vl_raw["F1"]:
                vl_m, which, state = vl_ema, "EMA", ema.ema.state_dict()
            else:
                vl_m, which, state = vl_raw, "raw", model.state_dict()
        else:
            vl_m, which, state = vl_raw, "raw", model.state_dict()

        is_best = vl_m["F1"] > best_f1 or not os.path.exists(best_path)
        if is_best:
            best_f1, pc = max(vl_m["F1"], best_f1), 0
            torch.save({"epoch": ep, "state_dict": state, "best_f1": best_f1,
                        "metrics": vl_m, "src": which}, best_path)
        else:
            pc += 1

        torch.save({"epoch": ep, "model": model.state_dict(),
                    "ema": ema.ema.state_dict() if ema is not None else None,
                    "ema_step": ema.step_n if ema is not None else 0,
                    "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "best_f1": best_f1, "patience": pc}, last_path)

        print(f"Ep {ep:3d}/{cfg['epochs']} ({time.time()-t0:.0f}s) "
              f"TrF1={tr_m['F1']:.1f} VlF1={vl_m['F1']:.1f}({which}) IoU={vl_m['IoU']:.1f}"
              + (f"  *** BEST {best_f1:.2f}% ***" if is_best else ""))
        if pc >= cfg["patience"]:
            print(f"Early stopping @ epoch {ep}")
            break

    print(f"\nDone. Best Val-F1 = {best_f1:.2f}%  ->  {best_path}")


if __name__ == "__main__":
    main()
