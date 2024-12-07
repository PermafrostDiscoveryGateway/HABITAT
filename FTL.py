import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=0.75, smooth=1e-6, weight_tversky=0.5, weight_ce=0.5):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.weight_tversky = weight_tversky  # Weight for Tversky loss
        self.weight_ce = weight_ce  # Weight for Cross Entropy loss

    def class_tversky(self, y_true, y_pred):
        # Flatten predictions and targets
        y_true_flat = y_true.view(-1)
        y_pred_flat = y_pred.view(-1)

        # True Positives, False Negatives, and False Positives
        true_pos = torch.sum(y_true_flat * y_pred_flat)
        false_neg = torch.sum(y_true_flat * (1 - y_pred_flat))
        false_pos = torch.sum((1 - y_true_flat) * y_pred_flat)

        # Tversky Index Calculation
        tversky_index = (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + (1 - self.alpha) * false_pos + self.smooth)

        return tversky_index

    def forward(self, y_true, y_pred):
        # Apply softmax to predictions (assuming multi-class segmentation)
        y_pred_soft = torch.softmax(y_pred, dim=1)

        # Calculate Tversky index
        tversky = self.class_tversky(y_true, y_pred_soft)

        # Focal Tversky Loss
        focal_tversky = torch.pow((1 - tversky), self.gamma)

        # Cross Entropy Loss
        ce_loss = F.cross_entropy(y_pred, y_true)

        # Final loss as a weighted sum of Focal Tversky and Cross Entropy
        total_loss = (self.weight_tversky * focal_tversky.sum()) + (self.weight_ce * ce_loss)

        return total_loss
