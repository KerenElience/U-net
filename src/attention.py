import torch
import torch.nn as nn
import torch.nn.functional as F
from .resunet import ResUnet, ConvBlock

## 加性注意力
class Attention(nn.Module):
    """
    Use in decoder layer before skip connection.
    
    input_ch is the decoder layer input channel.
    """
    def __init__(self, F_g, F_l, input_ch):
        super().__init__()

        self.Wg = nn.Sequential(
            nn.Conv2d(F_g, input_ch, 1, 1, bias = False),
            nn.BatchNorm2d(input_ch)
        )

        self.Wx = nn.Sequential(
            nn.Conv2d(F_l, input_ch, 1, 1, bias = False),
            nn.BatchNorm2d(input_ch)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(input_ch, 1, 1, 1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.Wg(g)
        x1 = self.Wx(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return torch.mul(x, psi)
    
class ResAttUnetDecoder(nn.Module):
    def __init__(self, input_channel, enc_channel, out_channel):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=True),
            nn.Conv2d(input_channel, enc_channel, 3, 1, 1),
        )
        self.att = Attention(enc_channel, out_channel, out_channel)
        self.conv = ConvBlock(enc_channel + out_channel, out_channel)

    def forward(self, x, encoder_x):
        x = self.upsample(x)
        # if x.shape[2:] != encoder_x.shape[2:]:
        # x = F.interpolate(x, size = encoder_x.shape[2:], mode="bilinear", align_corners=True)
        encoder_x = self.att(x, encoder_x)
        x = torch.cat([x, encoder_x], dim = 1)
        return self.conv(x)

class AttUnet(ResUnet):
    def __init__(self, input_channel, out_channel, enc_channels=[64, 64, 128 ,256, 512]):
        super().__init__(input_channel, out_channel, enc_channels)
        self.name = "attunet"

        self.d4 = ResAttUnetDecoder(enc_channels[-1]*2, enc_channels[-1], enc_channels[-2])
        self.d3 = ResAttUnetDecoder(enc_channels[-2], enc_channels[-3], enc_channels[-3])
        self.d2 = ResAttUnetDecoder(enc_channels[-3], enc_channels[-4], enc_channels[-4])
        self.d1 = ResAttUnetDecoder(enc_channels[-4], enc_channels[-5], enc_channels[-5])

        self.initial_weight_bias()

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