import torch
import torch.nn as nn
import numpy as np
import os

IS_CUDA = torch.cuda.is_available() and torch.backends.cudnn.enabled and os.getenv('CUDA').lower() == 'true'
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, predicted, target):
        predicted = predicted[:, 0:2]
        target = target[:, 0:2]
        loss = torch.sum((predicted - target) ** 2)
        return loss

class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, predicted, target):
        predicted = predicted[:, 0:2]
        target = target[:, 0:2]
        loss = torch.sum(torch.abs(predicted - target))
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
        centers = torch.rand(self.num_classes, self.feat_dim)
        #centers = nn.functional.normalize(centers, p=2, dim=1)
        self.centers = nn.Parameter(centers)

    def centre_loss(self, features, labels):
        distances = torch.cdist(features, self.centers, p=2) ** 2
        classes = torch.arange(self.num_classes).long()
        if IS_CUDA: classes = classes.cuda()
        mask = labels.unsqueeze(1) == classes
        distances = distances[mask].clamp(min=1e-12, max=1e+12)
        loss = distances.mean()
        return loss

    def forward(self, features, labels):
        return self.centre_loss(features, labels)

class BinLoss(CenterLoss):
    def __init__(self, num_bins, ranges, feature_dim):
        super(BinLoss, self).__init__(np.prod(num_bins), feature_dim)
        self.num_bins = num_bins
        self.ranges = ranges
        self.edges = []
        for i, value_range in enumerate(ranges.values()):
            f_min, f_max = value_range
            edges = torch.linspace(f_min, f_max, num_bins[i] - 1).cuda() if IS_CUDA \
                else torch.linspace(f_min, f_max, num_bins[i] - 1)
            self.edges.append(edges)

    def get_class(self, values):
        with torch.no_grad():
            n_features = values.shape[1]
            combined_bin = torch.zeros(values.shape[0], dtype=torch.int32, device=values.device)
    
            for i in range(n_features):
                f_min, f_max = self.ranges[i]
                feature_bin = torch.searchsorted(self.edges[i], values[:, i].contiguous(), right=False)
                feature_bin = torch.where(values[:, i] < f_min, torch.tensor(0, device=values.device), feature_bin)
                feature_bin = torch.where(values[:, i] > f_max, torch.tensor(self.num_bins[i] - 1, device=values.device), feature_bin)
    
                combined_bin = combined_bin * self.num_bins[i] + feature_bin
            return combined_bin

    def forward(self, features, target):
        return self.centre_loss(features, self.get_class(target))

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()

    def forward(self, predicted, target):
        loss = nn.MSELoss()(predicted, target)
        return loss

class SubjectLoss(CenterLoss):
    def __init__(self, num_classes, feat_dim):
        super(SubjectLoss, self).__init__(num_classes, feat_dim)

    def forward(self, features, labels):
        return super().forward(features, labels[:, -1])

class BPClassLoss(BinLoss):
    def __init__(self, num_bins, ranges, feature_dim):
        super(BPClassLoss, self).__init__(num_bins, ranges, feature_dim)

    def forward(self, features, target):
        loss = super().forward(features, target[:, 0:2])
        return loss

class CompactnessLoss(nn.Module):
    def __init__(self):
        super(CompactnessLoss, self).__init__()

    def forward(self, centers, labels, classes):
        same_class_mask = (classes.unsqueeze(1) == classes.unsqueeze(0))
        diff_class_mask = ~same_class_mask
        score_matrix = (centers[labels.unsqueeze(1)] - centers[labels.unsqueeze(0)]) ** 2
        score_matrix = score_matrix.mean(dim=-1)
        loss = 0
        if same_class_mask.any():
            loss += score_matrix[same_class_mask].mean()
        if diff_class_mask.any():
            loss += (1 / (1 + score_matrix[diff_class_mask])).mean()
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, loss_dict):
        super(CombinedLoss, self).__init__()
        self.loss_dict = loss_dict

    def forward(self, predicted, features, target):
        loss = 0
        for loss_name, (loss_fn, weight) in self.loss_dict.items():
            if "features" in loss_name:
                loss += weight * loss_fn(features, target)
            elif "Main" in loss_name:
                loss += weight * loss_fn(predicted, target[:, 0:2])
            else:
                loss += weight * loss_fn(predicted, target)
        return loss
