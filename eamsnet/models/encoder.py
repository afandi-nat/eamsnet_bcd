"""Siamese encoder dengan beberapa pilihan backbone (MobileNetV2 / ResNet-50 / ConvNeXt-Tiny).

Kedua citra temporal (T1, T2) diproses oleh encoder yang sama (berbagi bobot),
lalu fitur tiap stage diproyeksikan ke jumlah channel yang seragam (``out_ch``)
agar cocok dengan modul ATDAM/MSDA/EABRM dan decoder.
"""
import torch.nn as nn
from torchvision import models


class SiameseEncoder(nn.Module):
    """Encoder berbagi-bobot yang mengeluarkan fitur 4 skala.

    Args:
        backbone: 'mobilenet_v2' | 'resnet50' | 'convnext_tiny'.
        pretrained: muat bobot pretrained ImageNet.
        out_ch: daftar 4 channel keluaran tiap stage (mis. [32, 64, 128, 256]).
    """

    OUT_CH = [64, 128, 256, 512]

    def __init__(self, backbone="mobilenet_v2", pretrained=True, out_ch=None):
        super().__init__()
        self.backbone_name = backbone
        self.OUT_CH = list(out_ch) if out_ch is not None else [64, 128, 256, 512]

        if backbone == "convnext_tiny":
            import timm  # opsional; hanya diperlukan untuk backbone ini

            self.body = timm.create_model(
                "convnext_tiny", pretrained=pretrained, features_only=True,
                out_indices=(0, 1, 2, 3), drop_path_rate=0.2,
            )
            raw_ch = self.body.feature_info.channels()
        elif backbone in ("mobilenet_v2", "mobilen_v2"):
            mnet = models.mobilenet_v2(
                weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            )
            feats = mnet.features
            self.m_stage1 = feats[0:4]
            self.m_stage2 = feats[4:7]
            self.m_stage3 = feats[7:14]
            self.m_stage4 = feats[14:]
            raw_ch = [24, 32, 96, 1280]
        elif backbone == "resnet50":
            resnet = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None
            )
            resnet.maxpool = nn.Identity()
            self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4
            raw_ch = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Backbone tidak dikenal: {backbone}")

        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(raw_ch[i], self.OUT_CH[i], 1, bias=False),
                nn.BatchNorm2d(self.OUT_CH[i]),
                nn.ReLU(True),
            )
            for i in range(4)
        ])

    def forward(self, x):
        if self.backbone_name == "convnext_tiny":
            feats = self.body(x)
        elif self.backbone_name in ("mobilenet_v2", "mobilen_v2"):
            f1 = self.m_stage1(x)
            f2 = self.m_stage2(f1)
            f3 = self.m_stage3(f2)
            f4 = self.m_stage4(f3)
            feats = [f1, f2, f3, f4]
        else:
            x = self.stem(x)
            f1 = self.layer1(x)
            f2 = self.layer2(f1)
            f3 = self.layer3(f2)
            f4 = self.layer4(f3)
            feats = [f1, f2, f3, f4]
        return [self.proj[i](feats[i]) for i in range(4)]
