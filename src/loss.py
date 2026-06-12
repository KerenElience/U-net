import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):      
    def __init__(self, num_classes, smooth = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, pred, target):
        pred = F.softmax(pred, dim = 1)
        mask = (target != 255)
        target_valid = target.clone()
        target_valid[~mask] = 0
        target_one_hot = F.one_hot(target_valid, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        intersection = (pred * target_one_hot* mask.unsqueeze(1)).sum(dim = (0, 2, 3))
        union = pred.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))
        union = torch.where(union ==0, intersection, union)
        dice_loss = (2. * intersection + self.smooth)/(union + self.smooth)
        return 1 - dice_loss.mean()

class CEDiceLoss(nn.Module):
    def __init__(self, num_classes, ce_weight = 0.5):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.dice_loss = DiceLoss(num_classes)
        self.ce_weight = ce_weight
        self.dice_weight = 1 - ce_weight

    def forward(self, pred, target):
        loss = self.ce_weight * self.ce_loss(pred, target) + self.dice_weight*self.dice_loss(pred, target)
        return loss