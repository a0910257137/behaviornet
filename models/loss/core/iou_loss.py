import numpy as np
import tensorflow as tf
from .utils import weight_reduce_loss
from .iou2d_calculator import bbox_overlapping


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    # none: 0, elementwise_mean:1, sum: 2
    if reduction == 'origin':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()


def diou_loss(pred, target, eps=1e-7):
    r"""`Implementation of Distance-IoU Loss: Faster and Better
    Learning for Bounding Box Regression, https://arxiv.org/abs/1911.08287`_.

    Code is modified from https://github.com/Zzh-tju/DIoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """

    INF = 1e8
    # overlap
    lt = tf.math.maximum(pred[:, :2], target[:, :2])

    rb = tf.math.minimum(pred[:, 2:], target[:, 2:])
    wh = tf.clip_by_value((rb - lt), clip_value_min=0, clip_value_max=INF)

    overlap = wh[:, 0] * wh[:, 1]
    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union
    # enclose area
    enclose_x1y1 = tf.math.minimum(pred[:, :2], target[:, :2])
    enclose_x2y2 = tf.math.maximum(pred[:, 2:], target[:, 2:])
    enclose_wh = tf.clip_by_value((enclose_x2y2 - enclose_x1y1),
                                  clip_value_min=0,
                                  clip_value_max=INF)
    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]
    c2 = cw**2 + ch**2 + eps
    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]

    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]
    left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2))**2 / 4
    right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2))**2 / 4
    rho2 = left + right
    # DIoU
    dious = ious - rho2 / c2
    loss = 1 - dious
    return loss


class DIoULoss:

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    @tf.function
    def __call__(self,
                 pred,
                 target,
                 weight=None,
                 avg_factor=None,
                 reduction_override=None,
                 **kwargs):
        reduction = (reduction_override
                     if reduction_override else self.reduction)
        loss = diou_loss(pred, target)
        loss = self.loss_weight * weight_reduce_loss(loss, weight, reduction,
                                                     avg_factor)
        return loss


class GIoULoss:

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def __call__(self,
                 pred,
                 target,
                 weight=None,
                 avg_factor=None,
                 reduction_override=None):
        reduction = (reduction_override
                     if reduction_override else self.reduction)
        loss = self.giou_loss(pred, target, eps=self.eps)
        loss = self.loss_weight * weight_reduce_loss(loss, weight, reduction,
                                                     avg_factor)
        return loss

    def giou_loss(self, pred, target, eps=1e-7):
        r"""`Generalized Intersection over Union: A Metric and A Loss for Bounding
        Box Regression <https://arxiv.org/abs/1902.09630>`_.

        Args:
            pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
            eps (float): Eps to avoid log(0).

        Return:
            Tensor: Loss tensor.
        """
        # TODO: debugs
        gious = bbox_overlapping(pred,
                                 target,
                                 mode='giou',
                                 is_aligned=True,
                                 eps=eps)
        loss = 1 - gious
        return loss