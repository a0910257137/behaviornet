from .utils import weight_reduce_loss
import tensorflow as tf


class FocalLossWithProb:

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_
        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        # assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.bce = tf.keras.losses.BinaryCrossentropy(
            from_logits=True,
            axis=-1,
            reduction=tf.keras.losses.Reduction.NONE,
            name='binary_crossentropy')

    def __call__(self,
                 prob,
                 target,
                 weight=None,
                 avg_factor=None,
                 reduction_override='mean'):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override
                     if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss = self.focal_loss_with_prob(prob,
                                             target,
                                             weight,
                                             gamma=self.gamma,
                                             alpha=self.alpha,
                                             reduction=reduction,
                                             avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss

    # python version no_sigmoid
    def focal_loss_with_prob(self,
                             prob,
                             target,
                             weight=None,
                             gamma=2.0,
                             alpha=0.25,
                             reduction='mean',
                             avg_factor=None):
        bg_class_ind = tf.cast(tf.shape(prob)[1], tf.float32)
        pos = (target >= 0.) & (target < bg_class_ind)
        pos = tf.where(pos == True)
        target_one_hot = tf.one_hot(tf.cast(target, tf.int32),
                                    tf.cast(bg_class_ind, tf.int32),
                                    dtype=tf.float32)
        N, C = tf.shape(prob)[0], tf.shape(prob)[1]
        flatten_alpha = tf.fill([N, C], (1 - alpha))
        idxs = tf.cast(tf.where(target_one_hot == 1), tf.int32)
        vals = tf.ones(shape=tf.shape(idxs)[0]) * alpha
        flatten_alpha = tf.tensor_scatter_nd_update(flatten_alpha, idxs, vals)
        # flatten_alpha = tf.gather_nd(flatten_alpha, idxs)
        pt = tf.where(target_one_hot == 1., prob, 1 - prob)
        target_one_hot = tf.expand_dims(target_one_hot, axis=-1)
        prob = tf.expand_dims(prob, axis=-1)
        ce_loss = self.bce(target_one_hot, prob)
        loss = flatten_alpha * tf.math.pow(1 - pt, gamma) * ce_loss
        if weight is not None:
            weight = weight[:, None]
            loss = self.loss_weight * weight_reduce_loss(
                loss, weight, reduction, avg_factor)
        else:
            loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss


class TaskAlignedFocalLoss:

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 reduction='mean',
                 loss_weight=1.0):
        # assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.bce = tf.keras.losses.BinaryCrossentropy(
            from_logits=True,
            axis=-1,
            reduction=tf.keras.losses.Reduction.NONE,
            name='binary_crossentropy')

    def __call__(self,
                 prob,
                 target,
                 alignment_metric,
                 weight=None,
                 avg_factor=None,
                 reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override
                     if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * self.task_aigned_focal_loss(
                prob,
                target,
                alignment_metric,
                weight,
                gamma=self.gamma,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls

    def task_aigned_focal_loss(self,
                               prob,
                               target,
                               alignment_metric,
                               weight=None,
                               gamma=2.0,
                               reduction='mean',
                               avg_factor=None):
        bg_class_ind = tf.cast(tf.shape(prob)[1], tf.float32)
        pos = (target >= 0.) & (target < bg_class_ind)
        pos = tf.where(pos == True)
        target_one_hot = tf.one_hot(tf.cast(target, tf.int32),
                                    tf.cast(bg_class_ind, tf.int32))
        N, C = tf.shape(prob)[0], tf.shape(prob)[1]
        soft_label = tf.expand_dims(alignment_metric, axis=-1) * target_one_hot
        soft_label = tf.expand_dims(soft_label, axis=-1)
        prob = tf.expand_dims(prob, axis=-1)
        ce_loss = self.bce(soft_label, prob)
        soft_label = tf.squeeze(soft_label, axis=-1)
        prob = tf.squeeze(prob, axis=-1)
        loss = tf.math.pow(tf.math.abs(soft_label - prob), gamma) * ce_loss
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss
