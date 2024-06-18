import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False)  # Still use pointwise here
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = DepthwiseSeparableConv1d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * self.expansion, kernel_size=1, bias=False)  # And here
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.se = SEBlock(planes * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_channels=2):
        super(ResNet1D, self).__init__()
        self.in_planes = 32
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        self.linear = nn.Linear(256 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool1d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

'''

Inside each ResNet1d Block : There are 4 layers
    ┌───────────────────────────────────────┐
    │               Input                   │
    │                │                      │
    │                ▼                      │
    │       ┌──────────────┐                │
    │       │   Conv1D (1x1)│               │
    │       └──────────────┘                │
    │                │                      │
    │                ▼                      │
    │       ┌──────────────┐                │
    │       │ BatchNorm +  │                │
    │       │     ReLU     │                │
    │       └──────────────┘                │
    │                │                      │
    │                ▼                      │
    │   ┌────────────────────────────┐      │
    │   │ DepthwiseSeparableConv1D   │      │
    │   │      (3x3, stride, padding)│      │
    │   └──────────────┬─────────────┘      │
    │                  │                    │
    │                  ▼                    │
    │       ┌──────────────┐                │
    │       │ BatchNorm +  │                │
    │       │     ReLU     │                │
    │       └──────────────┘                │
    │                │                      │
    │                ▼                      │
    │       ┌──────────────┐                │
    │       │   Conv1D (1x1)│               │
    │       └──────────────┘                │
    │                │                      │
    │                ▼                      │
    │       ┌──────────────┐                │
    │       │   BatchNorm  │                │
    │       └──────────────┘                │
    │                │                      │
    │                ▼                      │
    │            SEBlock                    │
    │                │                      │
    │                ▼                      │
    │               ReLU                    │
    │                │                      │
    └────────────────┴──────────────────────┘
                      │
                      ▼
    ┌───────────────────────────────────────┐
    │            Shortcut Connection        │
    │                │                      │
    │                ▼                      │
    │    ┌─────────────────────────────┐    │
    │    │ Conv1D (1x1) (if needed)    │    │
    │    │ BatchNorm (if needed)       │    │
    │    └─────────────────────────────┘    │
    │                │                      │
    │                ▼                      │
    └────────────────┴──────────────────────┘
                      │
                      ▼
    ┌───────────────────────────────────────┐
    │              Addition                 │
    │          (Main + Shortcut)            │
    │                │                      │
    │                ▼                      │
    │               ReLU                    │
    └───────────────────────────────────────┘
                      │
                      ▼
                   Output


    '''