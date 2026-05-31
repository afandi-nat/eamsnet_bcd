"""Arsitektur EAMSNet.

Modul ATDAM / MSDA / EABRM dapat dimatikan lewat flag ``use_*`` untuk keperluan
ablation study. Saat sebuah modul dimatikan, jalurnya diganti operasi setara
dimensi: ATDAM -> selisih absolut |fA - fB|, MSDA -> identitas, EABRM ->
fitur decoder diteruskan apa adanya. Default (semua True) = EAMSNet penuh.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import SiameseEncoder
from .modules import ATDAM, MSDA, EABRM, DecoderBlock


def _resolve_width(width):
    if isinstance(width, (list, tuple)):
        return list(width)
    if width == "lite":
        return [32, 64, 128, 256]
    return [64, 128, 256, 512]


class EAMSNet(nn.Module):
    """EAMSNet untuk deteksi perubahan citra (binary change detection).

    Args:
        backbone: nama backbone encoder.
        pretrained: muat bobot pretrained ImageNet pada encoder.
        width: 'lite' | 'full' | daftar 4 channel kustom.
        use_atdam / use_msda / use_eabrm: toggle modul untuk ablation.
    """

    def __init__(self, backbone="mobilenet_v2", pretrained=True, width="lite",
                 use_atdam=True, use_msda=True, use_eabrm=True):
        super().__init__()
        ch = _resolve_width(width)
        self.ch = ch
        self.use_atdam = use_atdam
        self.use_msda = use_msda
        self.use_eabrm = use_eabrm

        self.encoder = SiameseEncoder(backbone, pretrained, out_ch=ch)

        if use_atdam:
            self.atdam = nn.ModuleList([ATDAM(c) for c in ch])
        if use_msda:
            self.msda = MSDA(ch[3])

        self.dec3 = DecoderBlock(ch[3], ch[2], ch[2])
        self.dec2 = DecoderBlock(ch[2], ch[1], ch[1])
        self.dec1 = DecoderBlock(ch[1], ch[0], ch[0])

        if use_eabrm:
            self.eabrm3 = EABRM(ch[2])
            self.eabrm2 = EABRM(ch[1])
            self.eabrm1 = EABRM(ch[0])

        self.head = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            nn.Conv2d(ch[0], 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Dropout2d(0.1), nn.Conv2d(32, 1, 1),
        )

        self.ds3 = nn.Conv2d(ch[2], 1, 1)
        self.ds2 = nn.Conv2d(ch[1], 1, 1)
        self.ds1 = nn.Conv2d(ch[0], 1, 1)

    def _diff(self, i, a, b):
        if self.use_atdam:
            return self.atdam[i](a, b)
        return torch.abs(a - b)

    def forward(self, imgA, imgB, return_feat=False):
        H, W = imgA.shape[2:]
        fA = self.encoder(imgA)
        fB = self.encoder(imgB)

        diffs = [self._diff(i, fA[i], fB[i]) for i in range(4)]
        if self.use_msda:
            diffs[3] = self.msda(diffs[3])

        def _match(skipA, skipB, ref):
            if skipA.shape[2:] != ref.shape[2:]:
                skipA = F.interpolate(skipA, ref.shape[2:], mode="bilinear", align_corners=True)
                skipB = F.interpolate(skipB, ref.shape[2:], mode="bilinear", align_corners=True)
            return skipA, skipB

        s3 = self.dec3(diffs[3], diffs[2])
        d3 = self.eabrm3(s3, *_match(fA[2], fB[2], s3)) if self.use_eabrm else s3
        s2 = self.dec2(d3, diffs[1])
        d2 = self.eabrm2(s2, *_match(fA[1], fB[1], s2)) if self.use_eabrm else s2
        s1 = self.dec1(d2, diffs[0])
        d1 = self.eabrm1(s1, *_match(fA[0], fB[0], s1)) if self.use_eabrm else s1

        out = self.head(d1)
        if out.shape[2:] != (H, W):
            out = F.interpolate(out, (H, W), mode="bilinear", align_corners=True)

        aux = []
        if self.training:
            up = lambda t: F.interpolate(t, (H, W), mode="bilinear", align_corners=True)
            aux = [up(self.ds3(d3)), up(self.ds2(d2)), up(self.ds1(d1))]

        if return_feat:
            return out, aux, d1
        return out, aux


# Alias kompatibilitas dengan nama lama di notebook.
EAMSNetPP = EAMSNet
