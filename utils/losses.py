import torch
import torch.nn as nn
import torch
from utils.extras import aves_hard_classes_set
import torch.nn.functional as F
import numpy as np


def set_loss(args):
    if args.loss_name == 'CE':
        loss = nn.CrossEntropyLoss()
    elif args.loss_name == 'WeightedCE':
        loss = WeightedCELoss(fewshot_weight=args.fewshot_weight)        
    elif args.loss_name == 'Focal':
        loss = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
    elif args.loss_name == 'BalancedSoftmax':
        loss = BalancedSoftmaxLoss(cls_num_list=args.cls_num_list)
    else:
        raise NotImplementedError(f'Loss {args.loss_name} not implemented.')

    args.loss = loss

    return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        # Ensure numerical stability with epsilon
        ce_loss = torch.clamp(ce_loss, min=1e-8)

        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * ((1-pt)**self.gamma) * ce_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class WeightedLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # assign higher weights to hard classes in aves_hard_classes_set
        weights = torch.ones(inputs.shape[0])
        for i in range(inputs.shape[0]):
            if str(targets[i].item()) in aves_hard_classes_set: 
                weights[i] = 3
                # print('hard class:', targets[i])
                # stop
        weights = weights.to(inputs.device)
        W_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        W_loss = W_loss * weights

        if self.reduction == 'mean':
            return torch.mean(W_loss)
        elif self.reduction == 'sum':
            return torch.sum(W_loss)
        else:
            return W_loss


class WeightedCELoss(nn.Module):
    def __init__(self, fewshot_weight=1.0, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.fewshot_weight = fewshot_weight

    # source in the text file uses 1 as fewshot data, 0 as retrived data
    def forward(self, inputs, targets, source):
        # print('inputs.shape:', inputs.shape)
        # print('targets.shape:', targets.shape)
        
        # fewshot data has higher weight, retrived data has weight 1.0
        weights = source * self.fewshot_weight + (1.0 - source)
        W_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        weights = weights.to(inputs.device)
        W_loss = W_loss * weights

        if self.reduction == 'mean':
            return torch.mean(W_loss)
        elif self.reduction == 'sum':
            return torch.sum(W_loss)
        else:
            return W_loss


class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, cls_num_list):
        super().__init__()
        cls_num_list = torch.from_numpy(np.array(cls_num_list)).float().cuda()
        cls_prior = cls_num_list / sum(cls_num_list)
        self.log_prior = torch.log(cls_prior).unsqueeze(0)
        # self.min_prob = 1e-9
        # print(f'Use BalancedSoftmaxLoss, class_prior: {cls_prior}')

    def forward(self, logits, labels):
        adjusted_logits = logits + self.log_prior        
        label_loss = F.cross_entropy(adjusted_logits, labels)

        return label_loss