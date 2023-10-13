import numpy as np
from .utils import weight_reduce_loss
import tensorflow as tf


class QualityFocalLoss:
    r"""Quality Focal Loss (QFL) is a variant of `Generalized Focal Loss:
    Learning Qualified and Distributed Bounding Boxes for Dense Object
    Detection <https://arxiv.org/abs/2006.04388>`_.

    Args:
        use_sigmoid (bool): Whether sigmoid operation is conducted in QFL.
            Defaults to True.
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self,
                 use_sigmoid=True,
                 beta=2.0,
                 reduction='mean',
                 loss_weight=1.0):
        assert use_sigmoid is True, 'Only sigmoid in QFL supported now.'
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.bce1 = tf.keras.losses.BinaryCrossentropy(
            from_logits=True,
            axis=-1,
            reduction=tf.keras.losses.Reduction.NONE,
            name='binary_crossentropy')
        self.bce2 = tf.keras.losses.BinaryCrossentropy(
            from_logits=True,
            axis=-1,
            reduction=tf.keras.losses.Reduction.NONE,
            name='binary_crossentropy')

    # @tf.function
    def __call__(self,
                 pred,
                 target,
                 weight=None,
                 avg_factor=None,
                 reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted joint representation of
                classification and quality (IoU) estimation with shape (N, C),
                C is the number of classes.
            target (tuple([torch.Tensor])): Target category label with shape
                (N,) and target quality label with shape (N,).
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
        if self.use_sigmoid:
            loss = self.quality_focal_loss(pred, target, beta=self.beta)
            loss_cls = self.loss_weight * weight_reduce_loss(
                loss, weight, reduction, avg_factor)
        else:
            raise NotImplementedError
        return loss_cls

    def quality_focal_loss(self, pred, target, beta=2.0):
        r"""Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
        Qualified and Distributed Bounding Boxes for Dense Object Detection
        <https://arxiv.org/abs/2006.04388>`_.
        Args:
            pred (torch.Tensor): Predicted joint representation of classification
                and quality (IoU) estimation with shape (N, C), C is the number of
                classes.
            target (tuple([torch.Tensor])): Target category label with shape (N,)
                and target quality label with shape (N,).
            beta (float): The beta parameter for calculating the modulating factor.
                Defaults to 2.0.

        Returns:
            torch.Tensor: Loss tensor with shape (N,).
        """

        assert len(
            target) == 2, """target for QFL must be a tuple of two elements,
            including category label and quality label, respectively"""
        # label denotes the category id, score denotes the quality score
        # save a score value in
        label, score = target
        # alignment_metrics = tf.expand_dims(alignment_metrics, axis=-1)
        pred_sigmoid = tf.math.sigmoid(pred)
        zerolabel = tf.zeros_like(pred)
        scale_factor = pred_sigmoid
        loss = self.bce1(tf.expand_dims(zerolabel, axis=-1),
                         tf.expand_dims(pred, axis=-1)) * tf.math.pow(
                             scale_factor, beta)
        bg_class_ind = tf.cast(tf.shape(pred)[1], tf.float32)
        pos = (label >= 0.) & (label < bg_class_ind)
        pos = tf.where(pos == True)
        pos_label = tf.cast(tf.gather_nd(label, pos), tf.int32)
        pred_pos = tf.concat(
            [tf.cast(pos, tf.int32),
             tf.expand_dims(pos_label, axis=-1)],
            axis=-1)
        scale_factor = tf.gather_nd(score, pos) - tf.gather_nd(
            pred_sigmoid, pred_pos)
        pos_score = tf.expand_dims(tf.gather_nd(score, pos), axis=-1)
        pos_pred = tf.expand_dims(tf.gather_nd(pred, pred_pos), axis=-1)
        pos_loss = self.bce2(pos_score, pos_pred) * tf.math.pow(
            tf.math.abs(scale_factor), beta)
        loss = tf.tensor_scatter_nd_update(loss, pred_pos, pos_loss)
        loss = tf.math.reduce_sum(loss, axis=1, keepdims=False)
        return loss