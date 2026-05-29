import torch
import torch.nn as nn
import torch.nn.functional as f

class ConvBlock(nn.Module):
    def __init__(self, input_channel, out_channel):
        super().__init__()
    
    def forward(self, x):
        pass

class DownSample(nn.Module):
    def __init__(self, channel):
        super().__init__()

    def forward(self, x):
        pass