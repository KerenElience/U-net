import torch
import torch.nn as nn
from .attention import AttUnet
from torchvision.models.resnet import resnet34, ResNet34_Weights

class ResUnetEnocder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet34(weights = ResNet34_Weights.IMAGENET1K_V1)

        self.conv1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )

        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        e0 = self.conv1(x)
        e0_pool = self.maxpool(e0)
        e1 = self.layer1(e0_pool)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        return e0, e1, e2, e3, e4

class AttUnetPretrained(AttUnet):
    def __init__(self, input_channel, out_channel, enc_channels=[64, 64, 128 ,256, 512]):
        super().__init__(input_channel, out_channel, enc_channels)
        self.encoder = ResUnetEnocder()
        
    def initial_weight_bias(self):
        for name, m in self.named_modules():
            if "encoder" in name:
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

