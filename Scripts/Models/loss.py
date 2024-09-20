import torch
import torch.nn as nn
import os

IS_CUDA = torch.cuda.is_available() and torch.backends.cudnn.enabled and os.getenv('CUDA').lower() == 'true'
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, predicted, target):
        predicted = predicted[:, 0:1]
        target = target[:, 0:1]
        loss = nn.MSELoss()(predicted, target)
        return loss

class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, predicted, target):
        predicted = predicted[:, 0:1]
        target = target[:, 0:1]
        loss = nn.L1Loss()(predicted, target)
        return loss


class CenterLoss(nn.Module):
    """
    Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes, feat_dim):

        super(CenterLoss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.centers = nn.Parameter(torch.rand(self.num_classes, self.feat_dim))

    def centre_loss(self, features, labels):
        labels = labels[:, -1]
        distances = torch.cdist(features, self.centers, p=2) ** 2
        classes = torch.arange(self.num_classes).long().cuda if IS_CUDA \
            else torch.arange(self.num_classes).long()
        mask = labels.unsqueeze(1) == classes
        distances = distances[mask].clamp(min=1e-12, max=1e+12)
        loss = distances.mean()
        return loss

    def forward(self, features, labels):
        return self.centre_loss(features, labels)

class MixedLoss(nn.Module):
    def __init__(self, loss_dict):
        super(MixedLoss, self).__init__()
        self.loss_dict = loss_dict

    def forward(self, predicted, features, target):
        loss = self.loss_dict['Main'](predicted, target)
        if 'Center' in self.loss_dict:
            loss += self.loss_dict['Center'][0](features, target) * self.loss_dict['Center'][1]
        return loss
