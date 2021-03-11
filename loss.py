import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "DiceLoss",
    "DiceBCELoss",
]


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    @staticmethod
    def forward(output, target, smooth=1e-5):
        output = F.sigmoid(output)

        batch = target.size(0)
        input_flat = output.view(batch, -1)
        target_flat = target.view(batch, -1)

        intersection = input_flat * target_flat
        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / batch
        return loss


class DiceBCELoss(nn.Module):
    def __init__(self, weight_ce=0.6):
        super(DiceBCELoss, self).__init__()
        self.weight_ce = weight_ce
        self.dc = DiceLoss()

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        ce_loss = F.binary_cross_entropy_with_logits(net_output, target)
        result = self.weight_ce * ce_loss + (1 - self.weight_ce) * dc_loss
        return result
