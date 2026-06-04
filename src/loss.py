import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    def __init__(self, smooth = 1e-5, bce_weight = 0.5):
        super().__init__()
        self.smooth = smooth
        self.bce_weight = bce_weight

    def dice_loss(self, pred, target):
        batch_size = pred.shape[0]
        i_flat = pred.view(batch_size, -1)
        t_flat = target.view(batch_size, -1)

        intersection = (i_flat * t_flat).sum(dim = 1)
        union = i_flat.sum(dim = 1) + t_flat.sum(dim = 1)

        return 1 - ((2. * intersection + self.smooth)/(union + self.smooth))

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target)
        pred = torch.sigmoid(pred)
        dice = self.dice_loss(pred, target)
        loss = bce*self.bce_weight + dice*(1-self.bce_weight)
        return loss.mean()