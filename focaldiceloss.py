import segmentation_models_pytorch as smp
import torch

class FocalDiceLoss(torch.nn.Module):
    def __init__(self, focal_weight=1, dice_weight=1):
        super(FocalDiceLoss, self).__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal_loss = smp.losses.FocalLoss(mode='multilabel')
        self.dice_loss = smp.losses.DiceLoss(mode='multilabel')

    def forward(self, inputs, targets):
        # Calculate Focal Loss
        focal_loss = self.focal_loss(inputs, targets)

        # Calculate Dice Loss
        dice_loss = self.dice_loss(inputs, targets)

        # Combine the losses
        loss = (self.focal_weight * focal_loss) + (self.dice_weight * dice_loss)

        return loss