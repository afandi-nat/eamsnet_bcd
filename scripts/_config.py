"""Helper bersama untuk skrip CLI: muat YAML, override via argparse, util path."""
import argparse
import os

import yaml

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), os.pardir, "configs", "default.yaml")


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def add_common_overrides(parser):
    """Tambahkan flag yang boleh menimpa nilai dari file config."""
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path file YAML config")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--width", default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def merge_overrides(cfg, args):
    """Timpa nilai config dengan argumen CLI yang tidak None."""
    mapping = {
        "data_root": "data_root", "backbone": "backbone", "width": "width",
        "img_size": "img_size", "batch_size": "batch_size", "num_workers": "num_workers",
        "epochs": "epochs", "out_dir": "out_dir", "seed": "seed",
    }
    for attr, key in mapping.items():
        val = getattr(args, attr, None)
        if val is not None:
            cfg[key] = val
    return cfg


def config_tag(cfg, use_atdam, use_msda, use_eabrm):
    """Tanda konfigurasi modul untuk nama file checkpoint."""
    parts = []
    if use_atdam:
        parts.append("ATDAM")
    if use_msda:
        parts.append("MSDA")
    if use_eabrm:
        parts.append("EABRM")
    return "Baseline" + ("_" + "_".join(parts) if parts else "")
