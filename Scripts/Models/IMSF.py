import numpy as np
import torch
import torch.nn as nn
import torchinfo
from dotenv import load_dotenv
import os
import math

class ECA(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECA, self).__init__()
        self.t = int(abs((math.log2(channel) + b) / gamma))
        self.k = self.t if self.t % 2 else self.t + 1

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=self.k, padding=self.k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
class IMSF_Block(nn.Module):
    def __init__(self, low_level_channel, high_level_channel, fusion_channel):
        super(IMSF_Block, self).__init__()
        self.conv7x7 = nn.Sequential(
            nn.LazyConv1d(high_level_channel, kernel_size=7, padding=3),
            nn.BatchNorm1d(high_level_channel),
            nn.ReLU(),
        )
        self.conv9x9 = nn.Sequential(
            nn.LazyConv1d(high_level_channel, kernel_size=9, padding=4),
            nn.BatchNorm1d(high_level_channel),
            nn.ReLU(),
        )
        self.conv11x11 = nn.Sequential(
            nn.LazyConv1d(high_level_channel, kernel_size=11, padding=5),
            nn.BatchNorm1d(high_level_channel),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.LazyConv1d(fusion_channel, kernel_size=9, padding=4),
            nn.BatchNorm1d(fusion_channel),
            nn.ReLU(),
        )
        self.conv3x3 = nn.Sequential(
            nn.LazyConv1d(low_level_channel, kernel_size=3, padding=1),
            nn.BatchNorm1d(low_level_channel),
            nn.ReLU(),
        )

    def forward(self, x):
        x3 = self.conv3x3(x)
        x7 = self.conv7x7(x)
        x9 = self.conv9x9(x)
        x11 = self.conv11x11(x)
        x = torch.cat([x7, x9, x11], dim=1)
        x = self.fusion(x)
        x = torch.cat([x, x3], dim=1)
        return x
