"""Dataset LEVIR-CD dan pembuat DataLoader.

Struktur folder yang diharapkan::

    DATA_ROOT/
        train/  A/  B/  label/
        val/    A/  B/  label/
        test/   A/  B/  label/

A = citra T1, B = citra T2, label = mask perubahan biner (putih = berubah).
Augmentasi hanya aktif pada split 'train'.
"""
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class LEVIRCDDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=256, augment=True):
        self.root_dir = os.path.join(root_dir, split)
        self.augment = augment and (split == "train")
        self.img_size = img_size
        self.img_A_dir = os.path.join(self.root_dir, "A")
        self.img_B_dir = os.path.join(self.root_dir, "B")
        self.label_dir = os.path.join(self.root_dir, "label")
        self.filenames = sorted(
            f for f in os.listdir(self.img_A_dir) if f.endswith((".png", ".jpg", ".tif"))
        )
        self.norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        print(f"[{split.upper()}] {len(self.filenames)} pairs loaded")

    def __len__(self):
        return len(self.filenames)

    def _augment(self, imgA, imgB, label):
        TF = transforms.functional
        if np.random.random() > 0.5:
            scale = np.random.uniform(1.0, 1.3)
            ns = int(self.img_size * scale)
            imgA = TF.resize(imgA, (ns, ns))
            imgB = TF.resize(imgB, (ns, ns))
            label = TF.resize(label, (ns, ns), interpolation=transforms.InterpolationMode.NEAREST)
            top = np.random.randint(0, ns - self.img_size + 1)
            left = np.random.randint(0, ns - self.img_size + 1)
            imgA = TF.crop(imgA, top, left, self.img_size, self.img_size)
            imgB = TF.crop(imgB, top, left, self.img_size, self.img_size)
            label = TF.crop(label, top, left, self.img_size, self.img_size)
        if np.random.random() > 0.5:
            imgA, imgB, label = (TF.hflip(x) for x in (imgA, imgB, label))
        if np.random.random() > 0.5:
            imgA, imgB, label = (TF.vflip(x) for x in (imgA, imgB, label))
        angle = int(np.random.choice([0, 90, 180, 270]))
        if angle > 0:
            imgA = TF.rotate(imgA, angle)
            imgB = TF.rotate(imgB, angle)
            label = TF.rotate(label, angle)
        if np.random.random() > 0.5:
            jitter = transforms.ColorJitter(0.3, 0.3, 0.2, 0.05)
            imgA, imgB = jitter(imgA), jitter(imgB)
        if np.random.random() > 0.7:
            blur = transforms.GaussianBlur(3, sigma=(0.1, 1.2))
            imgA, imgB = blur(imgA), blur(imgB)
        if np.random.random() > 0.5:  # tukar temporal: tahan terhadap urutan waktu
            imgA, imgB = imgB, imgA
        return imgA, imgB, label

    def __getitem__(self, idx):
        TF = transforms.functional
        f = self.filenames[idx]
        imgA = Image.open(os.path.join(self.img_A_dir, f)).convert("RGB")
        imgB = Image.open(os.path.join(self.img_B_dir, f)).convert("RGB")
        label = Image.open(os.path.join(self.label_dir, f)).convert("L")
        imgA = TF.resize(imgA, (self.img_size, self.img_size))
        imgB = TF.resize(imgB, (self.img_size, self.img_size))
        label = TF.resize(label, (self.img_size, self.img_size),
                          interpolation=transforms.InterpolationMode.NEAREST)
        if self.augment:
            imgA, imgB, label = self._augment(imgA, imgB, label)
        tA = self.norm(TF.to_tensor(imgA))
        tB = self.norm(TF.to_tensor(imgB))
        tL = (TF.to_tensor(label) > 0.5).float()
        return tA, tB, tL


def build_dataloaders(data_root, img_size=256, batch_size=16, num_workers=0):
    """Bangun train/val/test dataset + dataloader sekaligus."""
    train_ds = LEVIRCDDataset(data_root, "train", img_size, augment=True)
    val_ds = LEVIRCDDataset(data_root, "val", img_size, augment=False)
    test_ds = LEVIRCDDataset(data_root, "test", img_size, augment=False)

    pw = num_workers > 0
    pm = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=pm, drop_last=True, persistent_workers=pw)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=pm, persistent_workers=pw)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=pm, persistent_workers=pw)
    return (train_ds, val_ds, test_ds), (train_loader, val_loader, test_loader)
