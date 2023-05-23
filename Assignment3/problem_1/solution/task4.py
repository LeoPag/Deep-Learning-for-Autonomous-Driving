import torch
import torch.nn as nn
import torch.nn.functional as F


class RegressionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss = nn.SmoothL1Loss()

    def forward(self, pred, target, iou):
        '''
        Task 4.a
        We do not want to define the regression loss over the entire input space.
        While negative samples are necessary for the classification network, we
        only want to train our regression head using positive samples. Use 3D
        IoU ≥ 0.55 to determine positive samples and alter the RegressionLoss
        module such that only positive samples contribute to the loss.
        input
            pred (N,7) predicted bounding boxes
            target (N,7) target bounding boxes
            iou (N,) initial IoU of all paired proposal-targets
        useful config hyperparameters
            self.config['positive_reg_lb'] lower bound for positive samples
        '''

        pred = pred[iou >= self.config['positive_reg_lb'], :]
        target = target[iou >= self.config['positive_reg_lb'], :]

        if torch.cuda.is_available():
            pred = pred.type(torch.cuda.FloatTensor)
            target = target.type(torch.cuda.FloatTensor)
        else:
            pred = pred.type(torch.FloatTensor)
            target = target.type(torch.FloatTensor)

        pred_location = pred[:, :3]
        pred_size = pred[:, 3:6]
        pred_rotation = pred[:, 6]
        
        target_location = target[:, :3]
        target_size = target[:, 3:6]
        target_rotation = target[:, 6]
        
        loss_location = self.loss(pred_location, target_location)
        loss_size = self.loss(pred_size, target_size)
        loss_rotation = self.loss(pred_rotation, target_rotation)

        loss = loss_location + 3 * loss_size + loss_rotation

        return loss


class ClassificationLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss = nn.BCELoss()

    def forward(self, pred, iou):
        '''
        Task 4.b
        Extract the target scores depending on the IoU. For the training
        of the classification head we want to be more strict as we want to
        avoid incorrect training signals to supervise our network.  A proposal
        is considered as positive (class 1) if its maximum IoU with ground
        truth boxes is ≥ 0.6, and negative (class 0) if its maximum IoU ≤ 0.45.
            pred (N,) predicted bounding boxes
            iou (N,) initial IoU of all paired proposal-targets
        useful config hyperparameters
            self.config['positive_cls_lb'] lower bound for positive samples
            self.config['negative_cls_ub'] upper bound for negative samples
        '''

        # Remove the preds whose IoU is between 0.45 and 0.6
        mask = (iou >= self.config['positive_cls_lb']) | (iou <= self.config['negative_cls_ub'])
        iou = iou[mask]
        pred = pred[mask].squeeze()

        target = torch.round(iou)  # 0 if IoU <= 0.45 (negative), 1 if IoU >= 0.6 (positive)

        if torch.cuda.is_available():
            pred = pred.type(torch.cuda.FloatTensor)
            target = target.type(torch.cuda.FloatTensor)
        else:
            pred = pred.type(torch.FloatTensor)
            target = target.type(torch.FloatTensor)

        loss = self.loss(pred, target)

        return loss
