import torch.nn as nn
import torch
import segmentation_models_pytorch as smp

class CEDiceLoss(nn.Module):
    def __init__(self, ce_weight=0.75, dice_weight=0.25):
        super(CEDiceLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = smp.losses.DiceLoss(mode='binary')  # Replace 'multilabel' with your desired mode

    def forward(self, inputs, targets):
        # Calculate Cross Entropy Loss
        ce_loss = self.ce_loss(inputs, torch.argmax(targets, dim=1))

        # Calculate Dice Loss
        dice_loss = self.dice_loss(inputs, targets)

        # Combine the losses
        loss = (self.ce_weight * ce_loss) + (self.dice_weight * dice_loss)

        return loss