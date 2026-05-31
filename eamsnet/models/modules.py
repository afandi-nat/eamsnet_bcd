"""Modul usulan EAMSNet.

- ATDAM (Attentive Temporal Difference Attention Module): membentuk fitur
  perbedaan T1/T2 dengan atensi channel + spasial + gating.
- MSDA (Multi-Scale Dilated Aggregation): agregasi konteks multi-skala pada
  fitur perbedaan terdalam memakai konvolusi dilatasi.
- EABRM (Edge-Aware Boundary Refinement Module): mempertajam batas objek
  perubahan memakai operator Sobel pada fitur tiap citra.
- DecoderBlock: blok upsampling + skip-connection standar.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ATDAM(nn.Module):
    """Attentive Temporal Difference Attention Module."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        r = max(channels // reduction, 8)
        self.ch_pool = nn.AdaptiveAvgPool2d(1)
        self.ch_fc = nn.Sequential(
            nn.Linear(channels * 3, r), nn.ReLU(inplace=True),
            nn.Linear(r, channels), nn.Sigmoid(),
        )
        self.sp_conv = nn.Sequential(
            nn.Conv2d(6, 16, 7, padding=3, bias=False), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.Conv2d(16, 1, 7, padding=3, bias=False), nn.Sigmoid(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, f1, f2):
        B, C = f1.shape[:2]
        diff = f1 - f2
        abs_diff = torch.abs(diff)
        ch_in = torch.cat([
            self.ch_pool(f1).view(B, C),
            self.ch_pool(f2).view(B, C),
            self.ch_pool(abs_diff).view(B, C),
        ], dim=1)
        ch_att = self.ch_fc(ch_in).view(B, C, 1, 1)

        def _sp(t):
            return torch.cat([t.mean(1, True), t.max(1, True)[0]], 1)

        sp_att = self.sp_conv(torch.cat([_sp(f1), _sp(f2), _sp(abs_diff)], 1))
        attended = diff * ch_att * sp_att * self.gate(abs_diff)
        return self.relu(self.proj(attended) + diff)


class EABRM(nn.Module):
    """Edge-Aware Boundary Refinement Module (operator Sobel)."""

    def __init__(self, channels):
        super().__init__()
        self.edge_x = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.edge_y = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self._init_sobel()
        mid = max(channels // 4, 16)
        self.enhance = nn.Sequential(
            nn.Conv2d(channels * 2, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(True),
            nn.Conv2d(mid, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.Sigmoid()
        )
        self.out = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(True)
        )

    def _init_sobel(self):
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        with torch.no_grad():
            self.edge_x.weight.copy_(sx.view(1, 1, 3, 3).repeat(self.edge_x.weight.shape[0], 1, 1, 1))
            self.edge_y.weight.copy_(sy.view(1, 1, 3, 3).repeat(self.edge_y.weight.shape[0], 1, 1, 1))

    def _edge(self, feat):
        return self.enhance(torch.cat([self.edge_x(feat), self.edge_y(feat)], 1))

    def forward(self, semantic, f1, f2):
        edge_diff = torch.abs(self._edge(f1) - self._edge(f2))
        g = self.gate(torch.cat([semantic, edge_diff], 1))
        return self.out(semantic + g * edge_diff)


class MSDA(nn.Module):
    """Multi-Scale Dilated Aggregation (ASPP-like) dengan koneksi residual."""

    def __init__(self, channels):
        super().__init__()
        m = channels // 4
        self.b1 = nn.Sequential(nn.Conv2d(channels, m, 1, bias=False), nn.BatchNorm2d(m), nn.ReLU(True))
        self.b2 = nn.Sequential(nn.Conv2d(channels, m, 3, padding=2, dilation=2, bias=False), nn.BatchNorm2d(m), nn.ReLU(True))
        self.b3 = nn.Sequential(nn.Conv2d(channels, m, 3, padding=4, dilation=4, bias=False), nn.BatchNorm2d(m), nn.ReLU(True))
        self.b4 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, m, 1, bias=False), nn.BatchNorm2d(m), nn.ReLU(True))
        self.fuse = nn.Sequential(
            nn.Conv2d(m * 4, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(True), nn.Dropout2d(0.1)
        )

    def forward(self, x):
        H, W = x.shape[2:]
        g = F.interpolate(self.b4(x), (H, W), mode="bilinear", align_corners=True)
        return self.fuse(torch.cat([self.b1(x), self.b2(x), self.b3(x), g], 1)) + x


class DecoderBlock(nn.Module):
    """Upsample 2x, reduksi channel, gabung skip, dua konvolusi 3x3."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.reduce = nn.Conv2d(in_ch, in_ch // 2, 1, bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True),
        )

    def forward(self, x, skip):
        x = self.reduce(self.up(x))
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, skip.shape[2:], mode="bilinear", align_corners=True)
        return self.conv(torch.cat([x, skip], 1))
