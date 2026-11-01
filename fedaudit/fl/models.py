"""Models. CompactCNN is the default (fast on 2×T4); ResNet18 is optional.

Every model returns logits and exposes .features(x) (penultimate embedding) so MOON can add a
projection head and Stage C can compute attributions.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.MaxPool2d(2))


class CompactCNN(nn.Module):
    def __init__(self, in_ch: int = 1, n_classes: int = 4, width: int = 32, proj_dim: int = 128):
        super().__init__()
        w = width
        self.body = nn.Sequential(_block(in_ch, w), _block(w, 2 * w), _block(2 * w, 4 * w), _block(4 * w, 8 * w))
        self.feat_dim = 8 * w
        self.register_buffer("chan_mask", torch.ones(self.feat_dim))  # local feature mask (feature-masking baseline)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(self.feat_dim, n_classes)
        self.proj = nn.Sequential(nn.Linear(self.feat_dim, self.feat_dim), nn.ReLU(), nn.Linear(self.feat_dim, proj_dim))

    def features(self, x):
        h = self.body(x)
        return F.adaptive_avg_pool2d(h, 1).flatten(1) * self.chan_mask

    def forward(self, x):
        return self.fc(self.drop(self.features(x)))


class ResNet18(nn.Module):
    def __init__(self, in_ch: int = 1, n_classes: int = 4, pretrained: bool = True, proj_dim: int = 128):
        super().__init__()
        import torchvision
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        if in_ch != 3:
            w = m.conv1.weight.data.mean(1, keepdim=True).repeat(1, in_ch, 1, 1)
            m.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
            m.conv1.weight.data = w
        self.feat_dim = m.fc.in_features
        m.fc = nn.Identity()
        self.body = m
        self.register_buffer("chan_mask", torch.ones(self.feat_dim))
        self.fc = nn.Linear(self.feat_dim, n_classes)
        self.proj = nn.Sequential(nn.Linear(self.feat_dim, self.feat_dim), nn.ReLU(), nn.Linear(self.feat_dim, proj_dim))

    def features(self, x):
        return self.body(x) * self.chan_mask

    def forward(self, x):
        return self.fc(self.features(x))


def build_model(name: str, in_ch: int, n_classes: int) -> nn.Module:
    if name == "cnn":
        return CompactCNN(in_ch, n_classes)
    if name == "resnet18":
        return ResNet18(in_ch, n_classes)
    raise ValueError(name)
