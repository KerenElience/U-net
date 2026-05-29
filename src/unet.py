import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, input_channel, out_channel):
        super().__init__()
        self.layer = nn.Sequential(
            ## 对称填充, 能提取更多有效特征
            nn.Conv2d(input_channel, out_channel, 3, 1, 1, padding_mode="reflect", bias = False),
            ## BatchNorm也是进行了偏置计算，所以上方可以关闭
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3,inplace=True),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode="reflect", bias = False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3,inplace=True),
            nn.LeakyReLU(),
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
            nn.LeakyReLU()
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
            self.layer = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(channel, channel//2, 1, 1),
                )
        else:
            self.layer = nn.ConvTranspose2d(channel, channel//2, 2, 2)

    def forward(self, x, feature_map):
        out = self.layer(x)
        return torch.cat((feature_map, out), dim = 1)

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
        up_ = self.upsample(x, feature_map)
        out = self.conv(up_)
        return out

class UNet(nn.Module):
    def __init__(self, input_channel):
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

        self.outconv = nn.Conv2d(64, input_channel, 1, 1)
        self.softmax = nn.Softmax(dim = 1)
    
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
        return self.softmax(self.outconv(out))
