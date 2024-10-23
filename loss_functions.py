import torch
from torch import Tensor
import torch.nn as nn

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)

def get_dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, pred, target, smooth=1e-6):
        """
        Compute IoU loss.

        Parameters:
        - pred: torch.Tensor, predicted mask (output of the model), shape (batch_size, height, width)
        - target: torch.Tensor, ground truth mask, shape (batch_size, height, width)
        - smooth: float, smoothing factor to avoid division by zero

        Returns:
        - iou_loss: float, the IoU loss (1 - IoU)
        """
        # Flatten the tensors
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