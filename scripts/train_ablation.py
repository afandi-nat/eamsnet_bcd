"""Ablation study: latih 6 konfigurasi modul (resumable) lalu evaluasi test.

Skema:
    Baseline | +ATDAM | +MSDA | +EABRM | +ATDAM+MSDA | +ATDAM+MSDA+EABRM (full)

Bisa dihentikan kapan saja; jalankan ulang untuk melanjutkan. Konfigurasi yang
sudah selesai dilewati, yang terputus dilanjutkan dari epoch terakhir.

Contoh:
    python scripts/train_ablation.py --data-root /path/LEVIR-CD-256 --epochs 120
"""
import argparse
import csv
import os
import sys
import time

import torch
from torch.cuda.amp import GradScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from eamsnet.data import build_dataloaders
from eamsnet.losses import HybridLoss
from eamsnet.models import EAMSNet
from eamsnet.engine import (train_epoch_ema, evaluate, search_threshold, evaluate_full)
from eamsnet.utils import (set_seed, get_device, WarmupCosineScheduler, ModelEMA,
                           build_optimizer, count_params, benchmark)
from _config import load_config, add_common_overrides, merge_overrides, config_tag

ABLATION_CONFIGS = [
    ("Baseline", dict(use_atdam=False, use_msda=False, use_eabrm=False)),
    ("Baseline + ATDAM", dict(use_atdam=True, use_msda=False, use_eabrm=False)),
    ("Baseline + MSDA", dict(use_atdam=False, use_msda=True, use_eabrm=False)),
    ("Baseline + EABRM", dict(use_atdam=False, use_msda=False, use_eabrm=True)),
    ("Baseline + ATDAM + MSDA", dict(use_atdam=True, use_msda=True, use_eabrm=False)),
    ("Baseline + ATDAM + MSDA + EABRM", dict(use_atdam=True, use_msda=True, use_eabrm=True)),
]


def train_one(tag, flags, cfg, device, train_loader, val_loader):
    safe = config_tag(cfg, **flags)
    best_path = os.path.join(cfg["out_dir"], f"ablation_{safe}.pth")
    last_path = os.path.join(cfg["out_dir"], f"ablation_{safe}_last.pth")

    model = EAMSNet(cfg["backbone"], pretrained=cfg["pretrained"], width=cfg["width"],
                    **flags).to(device)
    crit = HybridLoss().to(device)
    opt = build_optimizer(model, cfg["lr_encoder"], cfg["lr_other"], cfg["weight_decay"])
    sched = WarmupCosineScheduler(opt, cfg["warmup"], cfg["epochs"])
    scaler = GradScaler()
    ema = ModelEMA(model) if cfg["ema"] else None

    start_epoch, best_f1, best_metrics, pc = 1, 0.0, None, 0
    if os.path.exists(last_path):
        st = torch.load(last_path, map_location=device, weights_only=False)
        if st.get("done"):
            print(f"  [skip] {tag} sudah selesai (Val-F1 {st['best_f1']:.2f}%)")
            return best_path, st["best_metrics"]
        model.load_state_dict(st["model"])
        if ema is not None and st.get("ema") is not None:
            ema.ema.load_state_dict(st["ema"])
            ema.step_n = st["ema_step"]
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        start_epoch, best_f1, best_metrics, pc = (st["epoch"] + 1, st["best_f1"],
                                                  st["best_metrics"], st["patience"])
        print(f"  [resume] {tag} lanjut dari epoch {start_epoch} (best {best_f1:.2f}%)")
    else:
        print("\n" + "=" * 72 + f"\n  >> {tag}  flags={flags}\n" + "=" * 72)

    for ep in range(start_epoch, cfg["epochs"] + 1):
        t0 = time.time()
        sched.step(ep - 1)
        _, tr_m = train_epoch_ema(model, ema, train_loader, crit, opt, scaler, device)
        _, vl_raw = evaluate(model, val_loader, crit, device)
        if ema is not None:
            _, vl_ema = evaluate(ema.ema, val_loader, crit, device)
            vl_m, which, state = ((vl_ema, "EMA", ema.ema.state_dict())
                                  if vl_ema["F1"] >= vl_raw["F1"]
                                  else (vl_raw, "raw", model.state_dict()))
        else:
            vl_m, which, state = vl_raw, "raw", model.state_dict()

        # tulis "best" jika F1 membaik ATAU bila belum ada file best sama sekali
        is_best = vl_m["F1"] > best_f1 or not os.path.exists(best_path)
        if is_best:
            best_f1, best_metrics, pc = max(vl_m["F1"], best_f1), vl_m, 0
            torch.save({"tag": tag, "flags": flags, "state_dict": state,
                        "best_f1": best_f1, "metrics": vl_m, "src": which}, best_path)
        else:
            pc += 1

        torch.save({"tag": tag, "flags": flags, "epoch": ep, "done": False,
                    "model": model.state_dict(),
                    "ema": ema.ema.state_dict() if ema is not None else None,
                    "ema_step": ema.step_n if ema is not None else 0,
                    "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "best_f1": best_f1, "best_metrics": best_metrics, "patience": pc}, last_path)

        print(f"  Ep {ep:3d}/{cfg['epochs']} ({time.time()-t0:.0f}s) "
              f"TrF1={tr_m['F1']:.1f} VlF1={vl_m['F1']:.1f}({which}) IoU={vl_m['IoU']:.1f}"
              + (f"  *** BEST {best_f1:.2f}% ***" if is_best else ""))
        if pc >= cfg["patience"]:
            print(f"  Early stop @ ep {ep}")
            break

    st = torch.load(last_path, map_location="cpu", weights_only=False)
    st["done"] = True
    torch.save(st, last_path)
    print(f"  Selesai {tag}: Val-F1 {best_f1:.2f}%")
    return best_path, best_metrics


def main():
    parser = add_common_overrides(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--no-train", action="store_true",
                        help="lewati pelatihan, hanya evaluasi checkpoint yang ada")
    args = parser.parse_args()

    cfg = merge_overrides(load_config(args.config), args)
    set_seed(cfg["seed"])
    device = get_device()
    os.makedirs(cfg["out_dir"], exist_ok=True)

    (_, _, _), (train_loader, val_loader, test_loader) = build_dataloaders(
        cfg["data_root"], cfg["img_size"], cfg["batch_size"], cfg["num_workers"])

    rows = []
    for tag, flags in ABLATION_CONFIGS:
        safe = config_tag(cfg, **flags)
        best_path = os.path.join(cfg["out_dir"], f"ablation_{safe}.pth")
        if not args.no_train:
            best_path, _ = train_one(tag, flags, cfg, device, train_loader, val_loader)

        model = EAMSNet(cfg["backbone"], pretrained=False, width=cfg["width"], **flags).to(device)
        ck = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        thr, _ = search_threshold(model, val_loader, device)
        m = evaluate_full(model, test_loader, device, thr=thr)
        ms, fps = benchmark(model, device, cfg["img_size"])
        rows.append({"Config": tag, **{k: round(v, 2) for k, v in m.items()},
                     "Params(M)": round(count_params(model), 2),
                     "FPS": round(fps, 1), "thr": thr})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    csv_path = os.path.join(cfg["out_dir"], "ablation_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 90 + "\n  ABLATION STUDY — Test set\n" + "=" * 90)
    hdr = list(rows[0].keys())
    print("  ".join(f"{h:>10s}" for h in hdr))
    for r in rows:
        print("  ".join(f"{str(r[h]):>10s}" for h in hdr))
    print(f"\nDisimpan -> {csv_path}")


if __name__ == "__main__":
    main()
