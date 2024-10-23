import torch
from torch import Tensor
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, multiclass: bool = False, epsilon: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.multiclass = multiclass
        self.epsilon = epsilon

    def dice_coeff(self, input: Tensor, target: Tensor, reduce_batch_first: bool = False) -> Tensor:
        # Ensure input and target have the same size and valid dimensions
        assert input.size() == target.size()
        assert input.dim() == 3 or not reduce_batch_first
        
        # Summing over the spatial dimensions (height, width)
        sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

        # Calculate intersection and union
        inter = 2 * (input * target).sum(dim=sum_dim)
        sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)

        # Handle cases where the union is zero to avoid division by zero
        sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

        # Dice coefficient calculation
        dice = (inter + self.epsilon) / (sets_sum + self.epsilon)
        return dice.mean()

    def multiclass_dice_coeff(self, input: Tensor, target: Tensor, reduce_batch_first: bool = False) -> Tensor:
        # Flatten the input and target across class dimension and calculate dice coefficient for multiclass cases
        return self.dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # Decide whether to compute multiclass dice or binary dice based on the multiclass flag
        if self.multiclass:
            fn = self.multiclass_dice_coeff
        else:
            fn = self.dice_coeff

        # Compute dice loss (1 - dice coefficient)
        return 1 - fn(input, target, reduce_batch_first=True)

class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, pred, target, smooth=1e-6):
        
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Compute the intersection
        intersection = (pred * target).sum()
        
        # Compute the union
        union = pred.sum() + target.sum() - intersection
        
        # Compute IoU
        iou = (intersection + smooth) / (union + smooth)
        
        # IoU loss (1 - IoU)
        iou_loss = 1 - iou
        
        return iou_loss
