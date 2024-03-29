import numpy as np
import tensorflow as tf
from .utils import weight_reduce_loss


def smooth_l1_loss(pred, target, beta=1.0):
    """Smooth L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert beta > 0

    # assert tf.shape(pred) == tf.shape(target) and len(tf.shape(target)) > 0
    diff = tf.math.abs(pred - target)
    loss = tf.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    return loss


class SmoothL1Loss:
    """Smooth L1 loss.

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def __call__(self,
                 pred,
                 target,
                 weight=None,
                 avg_factor=None,
                 reduction_override=None,
                 **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override
                     if reduction_override else self.reduction)
        loss = smooth_l1_loss(pred, target, beta=self.beta)
        loss_bbox = self.loss_weight * weight_reduce_loss(
            loss, weight, reduction, avg_factor)
        return loss_bbox
