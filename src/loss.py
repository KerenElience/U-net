import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth = 1e-5, ignore_index = None):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        pred = F.softmax(pred, dim = 1)
        if self.ignore_index is not None:
            mask = (target != self.ignore_index).float()
            target = target.clone()
            target[target == self.ignore_index] = 0
        else:
            mask = torch.ones_like(target).float()
        target_one_hot = F.one_hot(target, num_classes=self.num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)
        
        mask = mask.unsqueeze(1)
        target_one_hot = target_one_hot * mask
        pred = pred * mask

        intersection = (pred * target_one_hot).sum(dim = (0, 2, 3))
        union = pred.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))
        dice_loss = (2. * intersection + self.smooth)/(union + self.smooth)
        return 1 - dice_loss.mean()

class CEDiceLoss(nn.Module):
    def __init__(self, num_classes, ce_weight = 0.5, ignore_index = None):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index= ignore_index)
        self.dice_loss = DiceLoss(num_classes, ignore_index=ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = 1 - ce_weight

    def forward(self, pred, target):
        loss = self.ce_weight * self.ce_loss(pred, target) + self.dice_weight*self.dice_loss(pred, target)
        return loss