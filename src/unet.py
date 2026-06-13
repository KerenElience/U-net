import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, input_channel, out_channel):
        super().__init__()
        self.layer = nn.Sequential(
            ## 对称填充, 能提取更多有效特征
            nn.Conv2d(input_channel, out_channel, 3, 1, 1, padding_mode="reflect", bias = False),
            ## BatchNorm也是进行了偏置计算，所以conv可以关闭
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode="reflect", bias = False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Dropout2d(0.3)
        )
    
    def forward(self, x):
        return self.layer(x)

## 最大池化丢失特征太多了，使用卷积操作来进行下采样
class DownSample(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 2, 1, padding_mode="reflect", bias = False),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)

## 上采样完成之后和上一步卷积块的结果进行了一个拼接
## 上采样使用邻近插值法，保证数据密度，放弃了原文的转置卷积
class UpSample(nn.Module):
    def __init__(self, channel, nearest = True):
        super().__init__()
        assert channel %2==0, "Upsample channel must be integert divid by 2."
        if nearest:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = nn.Conv2d(channel, channel//2, 1, 1)
        else:
            self.up = nn.Identity()
            self.conv = nn.ConvTranspose2d(channel, channel//2, 2, 2)

    def forward(self, x, feature_map):
        x = self.up(x)
        diffY = feature_map.size()[2] - x.size()[2] 
        diffX = feature_map.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = self.conv(x)
        return torch.cat((feature_map, x), dim = 1)

class UnetEncoder(nn.Module):
    def __init__(self, input_channel, out_channel):
        super().__init__()
        self.conv = ConvBlock(input_channel, out_channel)
        self.downsample = DownSample(out_channel)
    
    def forward(self, x):
        out = self.conv(x)
        down_ = self.downsample(out)
        return out, down_

class UnetDecoder(nn.Module):
    def __init__(self, input_channel, out_channel):
        super().__init__()
        self.upsample = UpSample(input_channel)
        self.conv = ConvBlock(input_channel, out_channel)
        
    def forward(self, x, feature_map):
        out = self.upsample(x, feature_map)
        out = self.conv(out)
        return out

class UNet(nn.Module):
    def __init__(self, input_channel, out_channel):
        super().__init__()
        self.e1 = UnetEncoder(input_channel, 64)
        self.e2 = UnetEncoder(64, 128)
        self.e3 = UnetEncoder(128, 256)
        self.e4 = UnetEncoder(256, 512)

        self.e5 = ConvBlock(512, 1024)

        self.d1 = UnetDecoder(1024, 512)
        self.d2 = UnetDecoder(512, 256)
        self.d3 = UnetDecoder(256, 128)
        self.d4 = UnetDecoder(128, 64)

        self.outconv = nn.Conv2d(64, out_channel, 1, 1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        f1, out1 = self.e1(x)
        f2, out2 = self.e2(out1)
        f3, out3 = self.e3(out2)
        f4, out4 = self.e4(out3)
        f5 = self.e5(out4)

        out = self.d1(f5, f4)
        out = self.d2(out, f3)
        out = self.d3(out, f2)
        out = self.d4(out, f1)
        return self.outconv(out)
