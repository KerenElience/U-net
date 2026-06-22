import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import ConvBlock

class BasicBlock(nn.Module):
    def __init__(self, input_channel, out_channel, stride = 1, downsample = None):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, out_channel, 3, stride, 1),
            nn.BatchNorm2d(out_channel)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias = False),
            nn.BatchNorm2d(out_channel)
        )
        self.relu = nn.ReLU(inplace = True)
    
    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out

class ResUnetEncoder(nn.Module):
    def __init__(self, inplanes, block = BasicBlock, layers = [3, 4, 6, 3],
                 enc_channels = [64, 64, 128, 256, 512]):
        super().__init__()

        self.inplanes = enc_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, enc_channels[0], 7, 2, 3, bias=False),
            nn.BatchNorm2d(enc_channels[0]),
            nn.ReLU(inplace=True)
        ) # dim 1/2
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        self.layer1 = self._make_layer(block, enc_channels[1], layers[0], 1)
        self.layer2 = self._make_layer(block, enc_channels[2], layers[1], 2) #dim 1/8
        self.layer3 = self._make_layer(block, enc_channels[3], layers[2], 2) #dim 1/16
        self.layer4 = self._make_layer(block, enc_channels[4], layers[3], 2) #dim 1/32

    def _make_layer(self, block, out_channel, blocks, stride = 1):
        downsample = None
        if stride != 1 or self.inplanes != out_channel:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, out_channel, 1, stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )
        
        layers = []
        layers.append(block(self.inplanes, out_channel, stride, downsample))
        self.inplanes = out_channel
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, out_channel))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        e0 = self.conv1(x) #1/2
        e0_pool = self.maxpool(e0) #1/4
        e1 = self.layer1(e0_pool) #1/4
        e2 = self.layer2(e1) #1/8
        e3 = self.layer3(e2) #1/16
        e4 = self.layer4(e3) #1/32
        return e0, e1, e2, e3, e4

class UpSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
    
    def forward(self, x, feature_map):
        x = self.up(x)
        if x.shape[2:] != feature_map.shape[2:]:
            x = F.interpolate(x, size = feature_map.shape[2:], mode="bilinear", align_corners=True)
        return torch.cat([x, feature_map], dim = 1)
        
class ResUnetDecoder(nn.Module):
    def __init__(self, input_channel, enc_channel, out_channel):
        super().__init__()

        self.upsample = UpSample()
        self.conv = ConvBlock(input_channel + enc_channel, out_channel)
        
    def forward(self, x, feature_map):
        x = self.upsample(x, feature_map)
        return self.conv(x)

class ResUnet(nn.Module):
    def __init__(self, input_channel, out_channel, enc_channels = [64, 64, 128 ,256, 512]):
        super().__init__()
        self.name = "resunet"

        self.encoder = ResUnetEncoder(input_channel, enc_channels=enc_channels)
        self.bottle_block = ConvBlock(enc_channels[-1], enc_channels[-1]*2) #1024
        
        self.d1 = ResUnetDecoder(enc_channels[-4], enc_channels[-5], enc_channels[-5])
        self.d2 = ResUnetDecoder(enc_channels[-3], enc_channels[-4], enc_channels[-4])
        self.d3 = ResUnetDecoder(enc_channels[-2], enc_channels[-3], enc_channels[-3])
        self.d4 = ResUnetDecoder(enc_channels[-1]*2, enc_channels[-2], enc_channels[-2])
        
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode = "bilinear", align_corners=True),
            nn.Conv2d(enc_channels[-5], 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.outconv = nn.Conv2d(64, out_channel, 1, 1)

        self.initial_weight_bias()

    def initial_weight_bias(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        e0, e1, e2, e3, e4 = self.encoder(x)
        out = self.bottle_block(e4)
        out = self.d4(out, e3)
        out = self.d3(out, e2)
        out = self.d2(out, e1)
        out = self.d1(out, e0) # n, c, H/2, W/2
        out = self.final_upsample(out)
        out = self.outconv(out)
        return out