import tensorflow as tf
import numpy as np
import os
import json
from abc import ABCMeta, abstractmethod
import commentjson
from pprint import pprint
from functools import partial

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

train_cfg = {
    'assigner': {
        'type': 'ATSSAssigner',
        'topk': 9
    },
    'allowed_border': -1,
    'pos_weight': -1,
    'debug': False
}


def read_commentjson(path):
    with open(path, mode="r") as f:
        return commentjson.load(f)


def dump_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def ensure_rng(rng=None):
    """Simple version of the ``kwarray.ensure_rng``

    Args:
        rng (int | numpy.random.RandomState | None):
            if None, then defaults to the global rng. Otherwise this can be an
            integer or a RandomState class
    Returns:
        (numpy.random.RandomState) : rng -
            a numpy random number generator

    References:
        https://gitlab.kitware.com/computer-vision/kwarray/blob/master/kwarray/util_random.py#L270
    """

    if rng is None:
        rng = np.random.mtrand._rand
    elif isinstance(rng, int):
        rng = np.random.RandomState(rng)
    else:
        rng = rng
    return rng


def random_boxes(num=1, scale=1, rng=None):
    """Simple version of ``kwimage.Boxes.random``

    Returns:
        Tensor: shape (n, 4) in x1, y1, x2, y2 format.

    References:
        https://gitlab.kitware.com/computer-vision/kwimage/blob/master/kwimage/structs/boxes.py#L1390

    Example:
        >>> num = 3
        >>> scale = 512
        >>> rng = 0
        >>> boxes = random_boxes(num, scale, rng)
        >>> print(boxes)
        tensor([[280.9925, 278.9802, 308.6148, 366.1769],
                [216.9113, 330.6978, 224.0446, 456.5878],
                [405.3632, 196.3221, 493.3953, 270.7942]])
    """
    rng = ensure_rng(rng)

    tlbr = rng.rand(num, 4).astype(np.float32)

    tl_x = np.minimum(tlbr[:, 0], tlbr[:, 2])
    tl_y = np.minimum(tlbr[:, 1], tlbr[:, 3])
    br_x = np.maximum(tlbr[:, 0], tlbr[:, 2])
    br_y = np.maximum(tlbr[:, 1], tlbr[:, 3])

    tlbr[:, 0] = tl_x * scale
    tlbr[:, 1] = tl_y * scale
    tlbr[:, 2] = br_x * scale
    tlbr[:, 3] = br_y * scale
    return tlbr


"""This module defines the :class:`NiceRepr` mixin class, which defines a
``__repr__`` and ``__str__`` method that only depend on a custom ``__nice__``
method, which you must define. This means you only have to overload one
function instead of two.  Furthermore, if the object defines a ``__len__``
method, then the ``__nice__`` method defaults to something sensible, otherwise
it is treated as abstract and raises ``NotImplementedError``.

To use simply have your object inherit from :class:`NiceRepr`
(multi-inheritance should be ok).

This code was copied from the ubelt library: https://github.com/Erotemic/ubelt

Example:
    >>> # Objects that define __nice__ have a default __str__ and __repr__
    >>> class Student(NiceRepr):
    ...    def __init__(self, name):
    ...        self.name = name
    ...    def __nice__(self):
    ...        return self.name
    >>> s1 = Student('Alice')
    >>> s2 = Student('Bob')
    >>> print(f's1 = {s1}')
    >>> print(f's2 = {s2}')
    s1 = <Student(Alice)>
    s2 = <Student(Bob)>

Example:
    >>> # Objects that define __len__ have a default __nice__
    >>> class Group(NiceRepr):
    ...    def __init__(self, data):
    ...        self.data = data
    ...    def __len__(self):
    ...        return len(self.data)
    >>> g = Group([1, 2, 3])
    >>> print(f'g = {g}')
    g = <Group(3)>
"""
import warnings


class NiceRepr(object):
    """Inherit from this class and define ``__nice__`` to "nicely" print your
    objects.

    Defines ``__str__`` and ``__repr__`` in terms of ``__nice__`` function
    Classes that inherit from :class:`NiceRepr` should redefine ``__nice__``.
    If the inheriting class has a ``__len__``, method then the default
    ``__nice__`` method will return its length.

    Example:
        >>> class Foo(NiceRepr):
        ...    def __nice__(self):
        ...        return 'info'
        >>> foo = Foo()
        >>> assert str(foo) == '<Foo(info)>'
        >>> assert repr(foo).startswith('<Foo(info) at ')

    Example:
        >>> class Bar(NiceRepr):
        ...    pass
        >>> bar = Bar()
        >>> import pytest
        >>> with pytest.warns(None) as record:
        >>>     assert 'object at' in str(bar)
        >>>     assert 'object at' in repr(bar)

    Example:
        >>> class Baz(NiceRepr):
        ...    def __len__(self):
        ...        return 5
        >>> baz = Baz()
        >>> assert str(baz) == '<Baz(5)>'
    """

    def __nice__(self):
        """str: a "nice" summary string describing this module"""
        if hasattr(self, '__len__'):
            # It is a common pattern for objects to use __len__ in __nice__
            # As a convenience we define a default __nice__ for these objects
            return str(len(self))
        else:
            # In all other cases force the subclass to overload __nice__
            raise NotImplementedError(
                f'Define the __nice__ method for {self.__class__!r}')

    def __repr__(self):
        """str: the string of the module"""
        try:
            nice = self.__nice__()
            classname = self.__class__.__name__
            return f'<{classname}({nice}) at {hex(id(self))}>'
        except NotImplementedError as ex:
            warnings.warn(str(ex), category=RuntimeWarning)
            return object.__repr__(self)

    def __str__(self):
        """str: the string of the module"""
        try:
            classname = self.__class__.__name__
            nice = self.__nice__()
            return f'<{classname}({nice})>'
        except NotImplementedError as ex:
            warnings.warn(str(ex), category=RuntimeWarning)
            return object.__repr__(self)


class SamplingResult(NiceRepr):
    """Bbox sampling result.

    Example:
        >>> # xdoctest: +IGNORE_WANT
        >>> from mmdet.core.bbox.samplers.sampling_result import *  # NOQA
        >>> self = SamplingResult.random(rng=10)
        >>> print(f'self = {self}')
        self = <SamplingResult({
            'neg_bboxes': torch.Size([12, 4]),
            'neg_inds': tensor([ 0,  1,  2,  4,  5,  6,  7,  8,  9, 10, 11, 12]),
            'num_gts': 4,
            'pos_assigned_gt_inds': tensor([], dtype=torch.int64),
            'pos_bboxes': torch.Size([0, 4]),
            'pos_inds': tensor([], dtype=torch.int64),
            'pos_is_gt': tensor([], dtype=torch.uint8)
        })>
    """

    def __init__(self, pos_inds, neg_inds, bboxes, gt_bboxes, assign_result,
                 gt_flags):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_bboxes = bboxes[pos_inds]
        self.neg_bboxes = bboxes[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]

        self.num_gts = gt_bboxes.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        if np.size(gt_bboxes) == 0:
            # hack for index error case
            assert np.size(self.pos_assigned_gt_inds) == 0
            self.pos_gt_bboxes = np.empty_like(gt_bboxes).reshape([-1, 4])
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.reshape([-1, 4])

            self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds, :]

        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None

    @property
    def bboxes(self):
        """torch.Tensor: concatenated positive and negative boxes"""
        return np.concatenate([self.pos_bboxes, self.neg_bboxes])

    def to(self, device):
        """Change the device of the data inplace.

        Example:
            >>> self = SamplingResult.random()
            >>> print(f'self = {self.to(None)}')
            >>> # xdoctest: +REQUIRES(--gpu)
            >>> print(f'self = {self.to(0)}')
        """
        _dict = self.__dict__
        for key, value in _dict.items():
            if isinstance(value, np.ndarray):
                _dict[key] = value.to(device)
        return self

    def __nice__(self):
        data = self.info.copy()
        data['pos_bboxes'] = data.pop('pos_bboxes').shape
        data['neg_bboxes'] = data.pop('neg_bboxes').shape
        parts = [f"'{k}': {v!r}" for k, v in sorted(data.items())]
        body = '    ' + ',\n    '.join(parts)
        return '{\n' + body + '\n}'

    @property
    def info(self):
        """Returns a dictionary of info about the object."""
        return {
            'pos_inds': self.pos_inds,
            'neg_inds': self.neg_inds,
            'pos_bboxes': self.pos_bboxes,
            'neg_bboxes': self.neg_bboxes,
            'pos_is_gt': self.pos_is_gt,
            'num_gts': self.num_gts,
            'pos_assigned_gt_inds': self.pos_assigned_gt_inds,
        }

    @classmethod
    def random(cls, rng=None, **kwargs):
        """
        Args:
            rng (None | int | numpy.random.RandomState): seed or state.
            kwargs (keyword arguments):
                - num_preds: number of predicted boxes
                - num_gts: number of true boxes
                - p_ignore (float): probability of a predicted box assinged to \
                    an ignored truth.
                - p_assigned (float): probability of a predicted box not being \
                    assigned.
                - p_use_label (float | bool): with labels or not.

        Returns:
            :obj:`SamplingResult`: Randomly generated sampling result.

        Example:
            >>> from mmdet.core.bbox.samplers.sampling_result import *  # NOQA
            >>> self = SamplingResult.random()
            >>> print(self.__dict__)
        """
        # from mmdet.core.bbox.samplers.random_sampler import RandomSampler
        # from mmdet.core.bbox.assigners.assign_result import AssignResult
        # from mmdet.core.bbox import demodata
        rng = ensure_rng(rng)

        # make probabalistic?
        num = 32
        pos_fraction = 0.5
        neg_pos_ub = -1

        assign_result = AssignResult.random(rng=rng, **kwargs)

        # Note we could just compute an assignment
        bboxes = random_boxes(assign_result.num_preds, rng=rng)
        gt_bboxes = random_boxes(assign_result.num_gts, rng=rng)

        if rng.rand() > 0.2:
            # sometimes algorithms squeeze their data, be robust to that
            gt_bboxes = gt_bboxes.squeeze()
            bboxes = bboxes.squeeze()

        if assign_result.labels is None:
            gt_labels = None
        else:
            gt_labels = None  # todo

        if gt_labels is None:
            add_gt_as_proposals = False
        else:
            add_gt_as_proposals = True  # make probabalistic?

        sampler = RandomSampler(num,
                                pos_fraction,
                                neg_pos_ub=neg_pos_ub,
                                add_gt_as_proposals=add_gt_as_proposals,
                                rng=rng)
        self = sampler.sample(assign_result, bboxes, gt_bboxes, gt_labels)
        return self


class BaseSampler(metaclass=ABCMeta):
    """Base class of samplers."""

    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 **kwargs):
        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub
        self.add_gt_as_proposals = add_gt_as_proposals
        self.pos_sampler = self
        self.neg_sampler = self

    @abstractmethod
    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Sample positive samples."""
        pass

    @abstractmethod
    def _sample_neg(self, assign_result, num_expected, **kwargs):
        """Sample negative samples."""
        pass

    def sample(self,
               assign_result,
               bboxes,
               gt_bboxes,
               gt_labels=None,
               **kwargs):
        """Sample positive and negative bboxes.

        This is a simple implementation of bbox sampling given candidates,
        assigning results and ground truth bboxes.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            bboxes (Tensor): Boxes to be sampled from.
            gt_bboxes (Tensor): Ground truth bboxes.
            gt_labels (Tensor, optional): Class labels of ground truth bboxes.

        Returns:
            :obj:`SamplingResult`: Sampling result.

        Example:
            >>> from mmdet.core.bbox import RandomSampler
            >>> from mmdet.core.bbox import AssignResult
            >>> from mmdet.core.bbox.demodata import ensure_rng, random_boxes
            >>> rng = ensure_rng(None)
            >>> assign_result = AssignResult.random(rng=rng)
            >>> bboxes = random_boxes(assign_result.num_preds, rng=rng)
            >>> gt_bboxes = random_boxes(assign_result.num_gts, rng=rng)
            >>> gt_labels = None
            >>> self = RandomSampler(num=32, pos_fraction=0.5, neg_pos_ub=-1,
            >>>                      add_gt_as_proposals=False)
            >>> self = self.sample(assign_result, bboxes, gt_bboxes, gt_labels)
        """
        if len(bboxes.shape) < 2:
            bboxes = bboxes[None, :]

        bboxes = bboxes[:, :4]

        gt_flags = bboxes.new_zeros((bboxes.shape[0], ), dtype=np.uint8)
        if self.add_gt_as_proposals and len(gt_bboxes) > 0:
            if gt_labels is None:
                raise ValueError(
                    'gt_labels must be given when add_gt_as_proposals is True')
            bboxes = np.concatenate([gt_bboxes, bboxes], dim=0)
            assign_result.add_gt_(gt_labels)
            gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype=np.uint8)
            gt_flags = np.concatenate([gt_ones, gt_flags])

        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self.pos_sampler._sample_pos(assign_result,
                                                num_expected_pos,
                                                bboxes=bboxes,
                                                **kwargs)
        # We found that sampled indices have duplicated items occasionally.
        # (may be a bug of PyTorch)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self.neg_sampler._sample_neg(assign_result,
                                                num_expected_neg,
                                                bboxes=bboxes,
                                                **kwargs)
        neg_inds = neg_inds.unique()

        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)
        return sampling_result


class PseudoSampler(BaseSampler):
    """A pseudo sampler that does not do sampling actually."""

    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        """Sample positive samples."""
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        """Sample negative samples."""
        raise NotImplementedError

    def sample(self, assign_result, bboxes, gt_bboxes, **kwargs):
        """Directly returns the positive and negative indices  of samples.

        Args:
            assign_result (:obj:`AssignResult`): Assigned results
            bboxes (torch.Tensor): Bounding boxes
            gt_bboxes (torch.Tensor): Ground truth boxes

        Returns:
            :obj:`SamplingResult`: sampler results
        """

        pos_inds = np.unique(np.nonzero(assign_result.gt_inds > 0)[0])
        neg_inds = np.unique(np.nonzero(assign_result.gt_inds == 0)[0])
        gt_flags = np.zeros(shape=(bboxes.shape[0]), dtype=np.uint8)
        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)
        return sampling_result


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction == 'origin':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction_enum == 'sum':
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """

    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def quality_focal_loss(pred, target, beta=2.0):
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

    def binary_cross_entropy_with_logits(pred, zerolabel):
        sig_pred = 1.0 / (1.0 + np.exp(-pred))
        return -(np.log(sig_pred) * zerolabel + np.log(
            (1 - sig_pred)) * (1 - zerolabel))

    assert len(target) == 2, """target for QFL must be a tuple of two elements,
        including category label and quality label, respectively"""
    # label denotes the category id, score denotes the quality score
    label, score = target
    # negatives are supervised by 0 quality score
    # pass logit
    pred_sigmoid = 1.0 / (1.0 + np.exp(-pred))
    scale_factor = pred_sigmoid
    zerolabel = np.zeros(shape=(pred.shape))
    loss = binary_cross_entropy_with_logits(pred, zerolabel) * np.power(
        scale_factor, beta)
    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    bg_class_ind = pred.shape[1]
    pos = np.nonzero((label >= 0) & (label < bg_class_ind))[0]
    pos_label = label[pos].astype(np.int32)

    # positives are supervised by bbox quality (IoU) score
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = binary_cross_entropy_with_logits(
        pred[pos, pos_label], score[pos]) * np.power(np.abs(scale_factor), beta)
    loss = np.sum(loss, axis=1, keepdims=False)
    return loss


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
            loss = quality_focal_loss(pred, target, beta=self.beta)
            loss_cls = self.loss_weight * weight_reduce_loss(
                loss, weight, reduction, avg_factor)
        else:
            raise NotImplementedError
        return loss_cls


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
    lt = np.maximum(pred[:, :2], target[:, :2])
    rb = np.minimum(pred[:, 2:], target[:, 2:])
    wh = np.clip((rb - lt), a_min=0, a_max=INF)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = np.minimum(pred[:, :2], target[:, :2])
    enclose_x2y2 = np.maximum(pred[:, 2:], target[:, 2:])
    enclose_wh = np.clip((enclose_x2y2 - enclose_x1y1), a_min=0, a_max=INF)

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

    def __call__(self,
                 pred,
                 target,
                 weight=None,
                 avg_factor=None,
                 reduction_override=None,
                 **kwargs):

        if weight is not None and not np.any(weight > 0):
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override
                     if reduction_override else self.reduction)
        if weight is not None and len(weight.shape) > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = np.mean(weight, axis=-1)
        loss = diou_loss(pred, target)
        loss = self.loss_weight * weight_reduce_loss(loss, weight, reduction,
                                                     avg_factor)
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
        return


class AssignResult(NiceRepr):
    """Stores assignments between predicted and truth boxes.

    Attributes:
        num_gts (int): the number of truth boxes considered when computing this
            assignment

        gt_inds (LongTensor): for each predicted box indicates the 1-based
            index of the assigned truth box. 0 means unassigned and -1 means
            ignore.

        max_overlaps (FloatTensor): the iou between the predicted box and its
            assigned truth box.

        labels (None | LongTensor): If specified, for each predicted box
            indicates the category label of the assigned truth box.

    Example:
        >>> # An assign result between 4 predicted boxes and 9 true boxes
        >>> # where only two boxes were assigned.
        >>> num_gts = 9
        >>> max_overlaps = torch.LongTensor([0, .5, .9, 0])
        >>> gt_inds = torch.LongTensor([-1, 1, 2, 0])
        >>> labels = torch.LongTensor([0, 3, 4, 0])
        >>> self = AssignResult(num_gts, gt_inds, max_overlaps, labels)
        >>> print(str(self))  # xdoctest: +IGNORE_WANT
        <AssignResult(num_gts=9, gt_inds.shape=(4,), max_overlaps.shape=(4,),
                      labels.shape=(4,))>
        >>> # Force addition of gt labels (when adding gt as proposals)
        >>> new_labels = torch.LongTensor([3, 4, 5])
        >>> self.add_gt_(new_labels)
        >>> print(str(self))  # xdoctest: +IGNORE_WANT
        <AssignResult(num_gts=9, gt_inds.shape=(7,), max_overlaps.shape=(7,),
                      labels.shape=(7,))>
    """

    def __init__(self, num_gts, gt_inds, max_overlaps, labels=None):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels
        # Interface for possible user-defined properties
        self._extra_properties = {}

    @property
    def num_preds(self):
        """int: the number of predictions in this assignment"""
        return len(self.gt_inds)

    def set_extra_property(self, key, value):
        """Set user-defined new property."""
        assert key not in self.info
        self._extra_properties[key] = value

    def get_extra_property(self, key):
        """Get user-defined property."""
        return self._extra_properties.get(key, None)

    @property
    def info(self):
        """dict: a dictionary of info about the object"""
        basic_info = {
            'num_gts': self.num_gts,
            'num_preds': self.num_preds,
            'gt_inds': self.gt_inds,
            'max_overlaps': self.max_overlaps,
            'labels': self.labels,
        }
        basic_info.update(self._extra_properties)
        return basic_info

    def __nice__(self):
        """str: a "nice" summary string describing this assign result"""
        parts = []
        parts.append(f'num_gts={self.num_gts!r}')
        if self.gt_inds is None:
            parts.append(f'gt_inds={self.gt_inds!r}')
        else:
            parts.append(f'gt_inds.shape={tuple(self.gt_inds.shape)!r}')
        if self.max_overlaps is None:
            parts.append(f'max_overlaps={self.max_overlaps!r}')
        else:
            parts.append('max_overlaps.shape='
                         f'{tuple(self.max_overlaps.shape)!r}')
        if self.labels is None:
            parts.append(f'labels={self.labels!r}')
        else:
            parts.append(f'labels.shape={tuple(self.labels.shape)!r}')
        return ', '.join(parts)

    @classmethod
    def random(cls, **kwargs):
        """Create random AssignResult for tests or debugging.

        Args:
            num_preds: number of predicted boxes
            num_gts: number of true boxes
            p_ignore (float): probability of a predicted box assinged to an
                ignored truth
            p_assigned (float): probability of a predicted box not being
                assigned
            p_use_label (float | bool): with labels or not
            rng (None | int | numpy.random.RandomState): seed or state

        Returns:
            :obj:`AssignResult`: Randomly generated assign results.

        Example:
            >>> from mmdet.core.bbox.assigners.assign_result import *  # NOQA
            >>> self = AssignResult.random()
            >>> print(self.info)
        """
        rng = ensure_rng(kwargs.get('rng', None))

        num_gts = kwargs.get('num_gts', None)
        num_preds = kwargs.get('num_preds', None)
        p_ignore = kwargs.get('p_ignore', 0.3)
        p_assigned = kwargs.get('p_assigned', 0.7)
        p_use_label = kwargs.get('p_use_label', 0.5)
        num_classes = kwargs.get('p_use_label', 3)

        if num_gts is None:
            num_gts = rng.randint(0, 8)
        if num_preds is None:
            num_preds = rng.randint(0, 16)

        if num_gts == 0:
            max_overlaps = np.zeros(num_preds, dtype=np.float32)
            gt_inds = np.zeros(num_preds, dtype=np.int32)
            if p_use_label is True or p_use_label < rng.rand():
                labels = np.zeros(num_preds, dtype=np.int32)
            else:
                labels = None
        else:
            # Create an overlap for each predicted box
            max_overlaps = rng.rand(num_preds)

            # Construct gt_inds for each predicted box
            is_assigned = rng.rand(num_preds) < p_assigned
            # maximum number of assignments constraints
            n_assigned = min(num_preds, min(num_gts, is_assigned.sum()))

            assigned_idxs = np.where(is_assigned)[0]
            rng.shuffle(assigned_idxs)
            assigned_idxs = assigned_idxs[0:n_assigned]
            assigned_idxs.sort()

            is_assigned[:] = 0
            is_assigned[assigned_idxs] = True

            is_ignore = (rng.rand(num_preds)) < p_ignore & is_assigned

            gt_inds = np.zeros(num_preds, dtype=np.int32)

            true_idxs = np.arange(num_gts)
            rng.shuffle(true_idxs)
            true_idxs = true_idxs
            gt_inds[is_assigned] = true_idxs[:n_assigned]

            gt_inds = rng.randint(1, num_gts + 1, size=num_preds)
            gt_inds[is_ignore] = -1
            gt_inds[~is_assigned] = 0
            max_overlaps[~is_assigned] = 0

            if p_use_label is True or p_use_label < rng.rand():
                if num_classes == 0:
                    labels = np.zeros(num_preds, dtype=np.int32)
                else:
                    labels = rng.randint(0, num_classes, size=num_preds)
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class

                    labels[~is_assigned] = 0
            else:
                labels = None

        self = cls(num_gts, gt_inds, max_overlaps, labels)
        return self

    def add_gt_(self, gt_labels):
        """Add ground truth as assigned results.

        Args:
            gt_labels (torch.Tensor): Labels of gt boxes
        """
        self_inds = np.arange(1, len(gt_labels) + 1, dtype=np.int32)
        self.gt_inds = np.concatenate([self_inds, self.gt_inds])

        self.max_overlaps = np.concatenate(
            [self.max_overlaps.new_ones(len(gt_labels)), self.max_overlaps])

        if self.labels is not None:
            self.labels = np.concatenate([gt_labels, self.labels])


class BboxOverlaps2D(object):
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        """Calculate IoU between 2D bboxes.

        Args:
            bboxes1 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, or shape (m, 5) in <x1, y1, x2, y2, score> format.
            bboxes2 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, shape (m, 5) in <x1, y1, x2, y2, score> format, or be
                empty. If ``is_aligned `` is ``True``, then m and n must be
                equal.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground), or "giou" (generalized intersection over
                union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.shape[-1] in [0, 4, 5]
        assert bboxes2.shape[-1] in [0, 4, 5]
        if bboxes2.shape[-1] == 5:
            bboxes2 = bboxes2[..., :4]
        if bboxes1.shape[-1] == 5:
            bboxes1 = bboxes1[..., :4]
        return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned `` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned `` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )

    Example:
        >>> empty = torch.empty(0, 4)
        >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """
    INF = 1e6
    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes's last dimenstion is 4
    assert (bboxes1.shape[-1] == 4 or bboxes1.shape[0] == 0)
    assert (bboxes2.shape[-1] == 4 or bboxes2.shape[0] == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]
    rows = bboxes1.shape[-2]
    cols = bboxes2.shape[-2]

    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return np.zeros(shape=(batch_shape, rows))
        else:
            return np.zeros(shape=(batch_shape, rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] -
                                                   bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] -
                                                   bboxes2[..., 1])
    if is_aligned:
        lt = np.maximum(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = np.minimum(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = np.clip((rb - lt), a_min=0, a_max=INF)  # [B, rows, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = np.minimum(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = np.maximum(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = np.maximum(bboxes1[..., :, None, :2],
                        bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = np.minimum(bboxes1[..., :, None, 2:],
                        bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = np.clip((rb - lt), a_min=0, a_max=INF)  # [B, rows, cols, 2]

        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = np.minimum(bboxes1[..., :, None, :2],
                                     bboxes2[..., None, :, :2])
            enclosed_rb = np.maximum(bboxes1[..., :, None, 2:],
                                     bboxes2[..., None, :, 2:])
    eps = np.array([eps])
    # eps = union.new_tensor([eps])
    union = np.maximum(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = np.clip((enclosed_rb - enclosed_lt), a_min=0, a_max=INF)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]

    enclose_area = np.maximum(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


def unmap(data, count, inds, fill=0):
    """Unmap a subset of item (data) back to the original set of items (of size
    count)"""
    if len(data.shape) == 1:

        ret = np.full((count, ), fill_value=fill)
        ret[inds.astype(np.bool_)] = data
    else:
        new_size = (count, ) + data.shape[1:]
        ret = np.full(new_size, fill)
        #print(inds)
        #print('CCC', ret.shape, inds.shape, data.shape)
        ret[inds.astype(np.bool_), :] = data
    return ret


class ATSSAssigner:
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        topk (float): number of bbox selected in each level
    """

    def __init__(self,
                 topk,
                 mode=0,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 ignore_iof_thr=-1):
        self.topk = topk
        self.mode = mode
        self.iou_calculator = BboxOverlaps2D()
        self.ignore_iof_thr = ignore_iof_thr

    def assign(self,
               bboxes,
               num_level_bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None):
        INF = 100000000
        bboxes = bboxes[:, :4]
        num_gt, num_bboxes = gt_bboxes.shape[0], bboxes.shape[0]
        #print('AT1:', num_gt, num_bboxes)
        # compute iou between all bbox and gt
        overlaps = self.iou_calculator(bboxes, gt_bboxes)

        assigned_gt_inds = np.full(shape=(num_bboxes, ),
                                   fill_value=0,
                                   dtype=np.int32)
        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = np.zeros(shape=(num_bboxes, ))
            # max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = np.full(shape=(num_bboxes, ),
                                          fill_value=-1,
                                          dtype=np.int32)
            return AssignResult(num_gt,
                                assigned_gt_inds,
                                max_overlaps,
                                labels=assigned_labels)

        # assign 0 by default
        # compute center distance between all bbox and gt
        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0

        gt_points = np.stack((gt_cx, gt_cy), axis=-1)

        gt_width = gt_bboxes[:, 2] - gt_bboxes[:, 0]
        gt_height = gt_bboxes[:, 3] - gt_bboxes[:, 1]
        gt_area = np.sqrt(np.clip(gt_width * gt_height, a_min=1e-4, a_max=INF))

        bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        bboxes_points = np.stack((bboxes_cx, bboxes_cy), axis=-1)

        # distances = (bboxes_points[:, None, :] -
        #              gt_points[None, :, :]).pow(2).sum(-1).sqrt()
        distances = np.sqrt(
            np.sum(np.square(bboxes_points[:, None, :] - gt_points[None, :, :]),
                   axis=-1))
        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and np.any(gt_bboxes_ignore == True)):
            ignore_overlaps = self.iou_calculator(bboxes,
                                                  gt_bboxes_ignore,
                                                  mode='iof')
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
            distances[ignore_idxs, :] = INF
            assigned_gt_inds[ignore_idxs] = -1
        # Selecting candidates based on the center distance
        candidate_idxs = []
        start_idx = 0
        for level, bboxes_per_level in enumerate(num_level_bboxes):
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[start_idx:end_idx, :]  #(A,G)
            selectable_k = min(self.topk, bboxes_per_level)
            topk_idxs_per_level = np.argsort(distances_per_level, axis=0)
            topk_idxs_per_level = topk_idxs_per_level[:selectable_k]
            #print('AT-LEVEL:', start_idx, end_idx, bboxes_per_level, topk_idxs_per_level.shape)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = np.concatenate(
            candidate_idxs,
            axis=0)  # candidate anchors (topk*num_level_bboxes, G) = (AK, G)

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        candidate_overlaps = overlaps[candidate_idxs,
                                      np.arange(num_gt)]  #(AK,G)
        overlaps_mean_per_gt = np.mean(candidate_overlaps, axis=0)
        overlaps_std_per_gt = np.std(candidate_overlaps, axis=0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]
        #print('CAND:', candidate_idxs.shape, candidate_overlaps.shape, is_pos.shape)
        #print('BOXES:', bboxes_cx.shape)

        # limit the positive sample's center in gt
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        ep_bboxes_cx = np.tile(np.reshape(bboxes_cx, [1, -1]),
                               [num_gt, 1]).reshape([-1])
        ep_bboxes_cy = np.tile(np.reshape(bboxes_cy, [1, -1]),
                               [num_gt, 1]).reshape([-1])
        candidate_idxs = candidate_idxs.reshape([-1])
        # calculate the left, top, right, bottom distance between positive
        # bbox center and gt side
        l_ = ep_bboxes_cx[candidate_idxs].reshape([-1, num_gt]) - gt_bboxes[:,
                                                                            0]
        t_ = ep_bboxes_cy[candidate_idxs].reshape([-1, num_gt]) - gt_bboxes[:,
                                                                            1]
        r_ = gt_bboxes[:, 2] - ep_bboxes_cx[candidate_idxs].reshape(
            [-1, num_gt])
        b_ = gt_bboxes[:, 3] - ep_bboxes_cy[candidate_idxs].reshape(
            [-1, num_gt])
        #is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01

        dist_min = np.stack([l_, t_, r_, b_], axis=1).min(axis=1)  # (A,G)
        dist_min /= gt_area
        #print('ATTT:', l_.shape, t_.shape, dist_min.shape, self.mode)
        if self.mode == 0:
            is_in_gts = dist_min > 0.001
        elif self.mode == 1:
            is_in_gts = dist_min > -0.25
        elif self.mode == 2:
            is_in_gts = dist_min > -0.15
            #dist_expand = torch.clamp(gt_area / 16.0, min=1.0, max=3.0)
            #dist_min.mul_(dist_expand)
            #is_in_gts = dist_min > -0.25
        elif self.mode == 3:
            dist_expand = np.clip(gt_area / 16.0, a_min=1.0, a_max=6.0)
            dist_min *= dist_expand
            # dist_min.mul_(dist_expand)
            is_in_gts = dist_min > -0.2
        elif self.mode == 4:
            dist_expand = np.clip(gt_area / 16.0, a_min=0.5, a_max=6.0)
            dist_min *= dist_expand
            is_in_gts = dist_min > -0.2
        elif self.mode == 5:
            dist_div = np.clip(gt_area / 16.0, min=0.5, max=3.0)
            dist_min /= dist_div
            is_in_gts = dist_min > -0.2
        else:
            raise ValueError

        #print(gt_area.shape, is_in_gts.shape, is_pos.shape)
        is_pos = is_pos & is_in_gts

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf = np.full_like(overlaps, -INF).T.reshape([-1])

        index = candidate_idxs.reshape([-1])[is_pos.reshape([-1])]

        overlaps_inf[index] = overlaps.T.reshape([-1])[index]
        overlaps_inf = overlaps_inf.reshape([num_gt, -1]).T

        max_overlaps = np.max(overlaps_inf, axis=1)
        argmax_overlaps = np.argmax(overlaps_inf, axis=1)

        assigned_gt_inds[
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

        if gt_labels is not None:
            assigned_labels = np.full(shape=(num_bboxes, ),
                                      fill_value=-1,
                                      dtype=np.int32)
            pos_inds = np.nonzero(assigned_gt_inds > 0)[0]
            if np.size(pos_inds) > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds]
                                                      - 1]
        else:
            assigned_labels = None
        return AssignResult(num_gt,
                            assigned_gt_inds,
                            max_overlaps,
                            labels=assigned_labels)


def images_to_levels(target, num_levels):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = np.stack(target, axis=0)
    level_targets = []
    start = 0
    for n in num_levels:
        end = start + n
        # level_targets.append(target[:, start:end].squeeze(0))
        level_targets.append(target[:, start:end])
        start = end
    return level_targets


class Integral:
    """A fixed layer for calculating integral result from distribution.

    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}

    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=16):
        self.reg_max = reg_max

    def __call__(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.

        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.

        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        self.softmax(x.reshape([-1, self.reg_max + 1]), axis=1)

        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)
        return x

    def softmax(self, x, axis):

        max = np.max(
            x, axis=axis,
            keepdims=True)  #returns max of each row and keeps same dims
        e_x = np.exp(x - max)  #subtracts each row with its max value
        sum = np.sum(
            e_x, axis=axis,
            keepdims=True)  #returns sum of each row and keeps same dims
        f_x = e_x / sum
        return f_x


mobilenet = read_commentjson("./config/mobilenet.json")
loss_cfg = mobilenet["models"]["loss"]
loss_cls_cfg = loss_cfg["loss_cls"]
loss_kps_cfg = loss_cfg["loss_kps"]

assigner = ATSSAssigner(topk=loss_cfg["train_cfg"]["assigner"]["topk"])
sampler = PseudoSampler()

NK = 5
num_classes = 1
cls_out_channels = 1
use_dfl = False
use_kps = False
reg_max = 8
loss_kps_std = 1.0
integral = Integral(reg_max)

diou_func = DIoULoss(loss_weight=1.0)
smooth_func = SmoothL1Loss(beta=loss_kps_cfg["beta"],
                           loss_weight=loss_kps_cfg["loss_weight"])

cls_func = QualityFocalLoss(use_sigmoid=loss_cls_cfg["use_sigmoid"],
                            beta=loss_cls_cfg["beta"],
                            loss_weight=loss_cls_cfg["loss_weight"])


def get_num_level_anchors_inside(num_level_anchors, inside_flags):
    a1, a2, a3 = num_level_anchors
    split_inside_flags = [
        inside_flags[:a1], inside_flags[a1:a1 + a2],
        inside_flags[a1 + a2:a1 + a2 + a3]
    ]

    num_level_anchors_inside = [
        int(flags.sum()) for flags in split_inside_flags
    ]
    return num_level_anchors_inside


def anchor_inside_flags(flat_anchors, valid_flags, img_shape, allowed_border=0):
    """Check whether the anchors are inside the border.

    Args:
        flat_anchors (torch.Tensor): Flatten anchors, shape (n, 4).
        valid_flags (torch.Tensor): An existing valid flags of anchors.
        img_shape (tuple(int)): Shape of current image.
        allowed_border (int, optional): The border to allow the valid anchor.
            Defaults to 0.

    Returns:
        torch.Tensor: Flags indicating whether the anchors are inside a \
            valid range.
    """
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        inside_flags = valid_flags & \
            (flat_anchors[:, 0] >= -allowed_border) & \
            (flat_anchors[:, 1] >= -allowed_border) & \
            (flat_anchors[:, 2] < img_w + allowed_border) & \
            (flat_anchors[:, 3] < img_h + allowed_border)
    else:
        inside_flags = valid_flags
    return inside_flags


def anchor_center(anchors):
    """Get anchor centers from anchors.

    Args:
        anchors (Tensor): Anchor list with shape (N, 4), "xyxy" format.

    Returns:
        Tensor: Anchor centers with shape (N, 2), "xy" format.
    """
    anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
    anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
    return np.stack([anchors_cx, anchors_cy], axis=-1)


def kps2distance(points, kps, max_dis=None, eps=0.1):
    """Decode bounding box based on distances.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        kps (Tensor): Shape (n, K), "xyxy" format
        max_dis (float): Upper bound of the distance.
        eps (float): a small value to ensure target < max_dis, instead <=

    Returns:
        Tensor: Decoded distances.
    """

    preds = []
    for i in range(0, kps.shape[1], 2):
        px = kps[:, i] - points[:, i % 2]
        py = kps[:, i + 1] - points[:, i % 2 + 1]
        if max_dis is not None:
            px = np.clip(px, a_min=0, a_max=max_dis - eps)
            py = np.clip(py, a_min=0, a_max=max_dis - eps)
        preds.append(px)
        preds.append(py)
    return np.stack(preds, -1)


def bbox2distance(points, bbox, max_dis=None, eps=0.1):
    """Decode bounding box based on distances.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        bbox (Tensor): Shape (n, 4), "xyxy" format
        max_dis (float): Upper bound of the distance.
        eps (float): a small value to ensure target < max_dis, instead <=

    Returns:
        Tensor: Decoded distances.
    """
    left = points[:, 0] - bbox[:, 0]
    top = points[:, 1] - bbox[:, 1]
    right = bbox[:, 2] - points[:, 0]
    bottom = bbox[:, 3] - points[:, 1]
    if max_dis is not None:
        left = np.clip(left, a_min=0, a_max=max_dis - eps)
        top = np.clip(top, a_min=0, a_max=max_dis - eps)
        right = np.clip(right, a_min=0, a_max=max_dis - eps)
        bottom = np.clip(bottom, a_min=0, a_max=max_dis - eps)
    return np.stack([left, top, right, bottom], axis=-1)


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = np.clip(x1, a_min=0, a_max=max_shape[1])
        y1 = np.clip(y1, a_min=0, a_max=max_shape[0])
        x2 = np.flip(x2, a_min=0, a_max=max_shape[1])
        y2 = np.clip(y2, a_min=0, a_max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def _get_targets_single(flat_anchors,
                        valid_flags,
                        num_level_anchors,
                        gt_bboxes,
                        gt_bboxes_ignore,
                        gt_keypoints,
                        gt_labels,
                        label_channels=1,
                        unmap_outputs=True):
    img_shape = (640, 640)
    inside_flags = anchor_inside_flags(flat_anchors, valid_flags, img_shape,
                                       loss_cfg["train_cfg"]["allowed_border"])
    if not inside_flags.any():
        return (None, ) * 7
    # assign gt and sample anchors
    anchors = flat_anchors[inside_flags, :]
    num_level_anchors_inside = get_num_level_anchors_inside(
        num_level_anchors, inside_flags)

    assign_result = assigner.assign(anchors, num_level_anchors_inside,
                                    gt_bboxes, gt_bboxes_ignore, gt_labels)
    sampling_result = sampler.sample(assign_result, anchors, gt_bboxes)
    num_valid_anchors = anchors.shape[0]
    bbox_targets = np.zeros_like(anchors)
    bbox_weights = np.zeros_like(anchors)

    kps_targets = np.zeros(shape=(anchors.shape[0], NK * 2))
    kps_weights = np.zeros(shape=(anchors.shape[0], NK * 2))
    labels = np.full(shape=(num_valid_anchors, ),
                     fill_value=num_classes,
                     dtype=np.int32)

    label_weights = np.zeros(shape=(num_valid_anchors), dtype=np.float32)
    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds

    if len(pos_inds) > 0:
        pos_bbox_targets = sampling_result.pos_gt_bboxes
        bbox_targets[pos_inds, :] = pos_bbox_targets
        bbox_weights[pos_inds, :] = 1.0
        if use_kps:
            pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds
            #print('BBB', anchors.shape, gt_bboxes.shape, gt_keypointss.shape, pos_inds.shape, bbox_targets.shape, pos_bbox_targets.shape)
            kps_targets[pos_inds, :] = gt_keypoints[
                pos_assigned_gt_inds, :, :2].reshape((-1, NK * 2))
            kps_weights[pos_inds, :] = np.mean(
                gt_keypoints[pos_assigned_gt_inds, :, 2], axis=1, keepdims=True)
        #kps_weights[pos_inds, :] = 1.0
        if gt_labels is None:
            # Only rpn gives gt_labels as None
            # Foreground is the first class
            labels[pos_inds] = 0
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        if loss_cfg["train_cfg"]["pos_weight"] <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = loss_cfg["train_cfg"]["pos_weight"]
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0

    # map up to original set of anchors
    if unmap_outputs:
        num_total_anchors = flat_anchors.shape[0]
        anchors = unmap(anchors, num_total_anchors, inside_flags)
        labels = unmap(labels,
                       num_total_anchors,
                       inside_flags,
                       fill=num_classes)
        label_weights = unmap(label_weights, num_total_anchors, inside_flags)
        bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
        if use_kps:
            kps_targets = unmap(kps_targets, num_total_anchors, inside_flags)
            kps_weights = unmap(kps_weights, num_total_anchors, inside_flags)

    return (anchors, labels, label_weights, bbox_targets, bbox_weights,
            kps_targets, kps_weights, pos_inds, neg_inds)


def loss_single(anchors, cls_score, bbox_pred, kps_pred, labels, label_weights,
                bbox_targets, kps_targets, kps_weights, stride,
                num_total_samples):
    """Compute loss of a single scale level.

    Args:
        anchors (Tensor): Box reference for each scale level with shape
            (N, num_total_anchors, 4).
        cls_score (Tensor): Cls and quality joint scores for each scale
            level has shape (N, num_classes, H, W).
        bbox_pred (Tensor): Box distribution logits for each scale
            level with shape (N, 4*(n+1), H, W), n is max value of integral
            set.
        labels (Tensor): Labels of each anchors with shape
            (N, num_total_anchors).
        label_weights (Tensor): Label weights of each anchor with shape
            (N, num_total_anchors)
        bbox_targets (Tensor): BBox regression targets of each anchor wight
            shape (N, num_total_anchors, 4).
        stride (tuple): Stride in this scale level.
        num_total_samples (int): Number of positive samples that is
            reduced over all GPUs.

    Returns:
        dict[str, Tensor]: A dictionary of loss components.
    """
    assert stride[0] == stride[1], 'h stride is not equal to w stride!'
    use_qscore = True

    anchors = anchors.reshape([-1, 4])

    cls_score = np.transpose(cls_score,
                             [0, 2, 3, 1]).reshape([-1, cls_out_channels])
    if not use_dfl:
        bbox_pred = np.transpose(bbox_pred, (0, 2, 3, 1)).reshape([-1, 4])
    else:
        bbox_pred = np.transpose(bbox_pred,
                                 (0, 2, 3, 1)).reshape([-1, 4 * (reg_max + 1)])

    bbox_targets = bbox_targets.reshape([-1, 4])
    labels = labels.reshape([-1])
    label_weights = label_weights.reshape([-1])
    if use_kps:
        kps_pred = np.transpose(kps_pred, [0, 2, 3, 1]).reshape([-1, NK * 2])
        kps_targets = kps_targets.reshape((-1, NK * 2))
        kps_weights = kps_weights.reshape((-1, NK * 2))
        #print('AAA000', kps_targets.shape, kps_weights.shape)

    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    bg_class_ind = num_classes
    pos_inds = np.nonzero((labels >= 0) & (labels < bg_class_ind))[0]
    score = np.zeros(shape=(labels.shape))
    # score = label_weights.new_zeros(labels.shape)
    if len(pos_inds) > 0:
        pos_bbox_targets = bbox_targets[pos_inds]
        pos_bbox_pred = bbox_pred[pos_inds]
        pos_anchors = anchors[pos_inds]
        pos_anchor_centers = anchor_center(pos_anchors) / stride[0]
        weight_targets = 1 / (1.0 + np.exp(-cls_score))
        weight_targets = np.max(weight_targets, axis=1)[pos_inds]
        pos_decode_bbox_targets = pos_bbox_targets / stride[0]
        if use_dfl:
            pos_bbox_pred_corners = integral(pos_bbox_pred)
            pos_decode_bbox_pred = distance2bbox(pos_anchor_centers,
                                                 pos_bbox_pred_corners)
        else:
            pos_decode_bbox_pred = distance2bbox(pos_anchor_centers,
                                                 pos_bbox_pred)
        if use_kps:
            pos_kps_targets = kps_targets[pos_inds]
            pos_kps_pred = kps_pred[pos_inds]
            #print('CCC000', kps_weights.shape)
            pos_kps_weights = np.max(kps_weights,
                                     axis=1)[pos_inds] * weight_targets
            #pos_kps_weights = kps_weights.max(dim=1)[0][pos_inds]
            pos_kps_weights = pos_kps_weights.reshape((-1, 1))
            #pos_kps_weights = kps_weights.max(dim=1, keepdims=True)[0][pos_inds]
            #print('SSS', pos_kps_weights.sum())

            #pos_decode_kps_targets = pos_kps_targets / stride[0]
            #pos_decode_kps_pred = distance2kps(pos_anchor_centers, pos_kps_pred)

            pos_decode_kps_targets = kps2distance(pos_anchor_centers,
                                                  pos_kps_targets / stride[0])
            pos_decode_kps_pred = pos_kps_pred
            #print('ZZZ', pos_decode_kps_targets.shape, pos_decode_kps_pred.shape)
            #print(pos_kps_weights[0,:].detach().cpu().numpy())
            #print(pos_decode_kps_targets[0,:].detach().cpu().numpy())
            #print(pos_decode_kps_pred[0,:].detach().cpu().numpy())

            #print('CCC111', weight_targets.shape, pos_bbox_pred.shape, pos_decode_bbox_pred.shape, pos_kps_pred.shape, pos_decode_kps_pred.shape, pos_kps_weights.shape)

        if use_qscore:
            score[pos_inds] = bbox_overlaps(pos_decode_bbox_pred,
                                            pos_decode_bbox_targets,
                                            is_aligned=True)
        else:
            score[pos_inds] = 1.0

        # regression loss
        loss_bbox = diou_func(pos_decode_bbox_pred,
                              pos_decode_bbox_targets,
                              weight=weight_targets,
                              avg_factor=1.0)

        if use_kps:
            loss_kps = smooth_func(pos_decode_kps_pred * loss_kps_std,
                                   pos_decode_kps_targets * loss_kps_std,
                                   weight=pos_kps_weights,
                                   avg_factor=1.0)
        else:
            loss_kps = np.sum(kps_pred) * 0.

        # dfl loss
        if use_dfl:
            pred_corners = pos_bbox_pred.reshape(-1, reg_max + 1)
            target_corners = bbox2distance(pos_anchor_centers,
                                           pos_decode_bbox_targets,
                                           reg_max).reshape(-1)
            loss_dfl = dfl_func(pred_corners,
                                target_corners,
                                weight=weight_targets[:, None].expand(
                                    -1, 4).reshape(-1),
                                avg_factor=4.0)
        else:
            loss_dfl = bbox_pred.sum() * 0.
    else:
        loss_bbox = bbox_pred.sum() * 0.
        loss_dfl = bbox_pred.sum() * 0.
        loss_kps = kps_pred.sum() * 0.
        weight_targets = 0

    loss_cls = cls_func(cls_score, (labels, score),
                        weight=label_weights,
                        avg_factor=num_total_samples)
    return loss_cls, loss_bbox, loss_dfl, loss_kps, np.sum(weight_targets)


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def get_targets(batch_size,
                b_anchors,
                b_valid_flags,
                b_gt_bboxes,
                b_gt_bboxes_ignore=None,
                b_gt_keypoints=None,
                b_gt_labels=None,
                label_channels=1,
                unmap_outputs=True,
                return_sampling_results=False):
    num_imgs = batch_size
    b_anchors = b_anchors.numpy()
    b_valid_flags = b_valid_flags.numpy()
    b_gt_bboxes = b_gt_bboxes.numpy()
    b_gt_bboxes_ignore = b_gt_bboxes_ignore.numpy()
    b_gt_keypoints = b_gt_keypoints.numpy()
    b_gt_labels = b_gt_labels.numpy()
    assert np.shape(b_anchors)[0] == np.shape(b_valid_flags)[0] == num_imgs
    # anchor number of multi levels
    num_level_anchors = [12800, 3200, 800]
    b_num_level_anchors = np.tile(
        np.asarray(num_level_anchors)[None, :], [num_imgs, 1])

    # compute targets for each image
    if b_gt_bboxes_ignore is None:
        b_gt_bboxes_ignore = [None for _ in range(num_imgs)]
    if b_gt_labels is None:
        b_gt_labels = [None for _ in range(num_imgs)]

    #print('QQQ:', num_imgs, gt_bboxes_list[0].shape)
    (all_anchors, all_labels, all_label_weights, all_bbox_targets,
     all_bbox_weights, all_keypoints_targets, all_keypoints_weights,
     pos_inds_list, neg_inds_list) = multi_apply(_get_targets_single,
                                                 b_anchors,
                                                 b_valid_flags,
                                                 b_num_level_anchors,
                                                 b_gt_bboxes,
                                                 b_gt_bboxes_ignore,
                                                 b_gt_keypoints,
                                                 b_gt_labels,
                                                 label_channels=label_channels,
                                                 unmap_outputs=unmap_outputs)
    # no valid anchors
    if any([labels is None for labels in all_labels]):
        return None
    # sampled anchors of all images
    num_total_pos = sum([max(np.size(inds), 1) for inds in pos_inds_list])
    num_total_neg = sum([max(np.size(inds), 1) for inds in neg_inds_list])
    # split targets to a list w.r.t. multiple levels
    anchors_list = images_to_levels(all_anchors, num_level_anchors)
    labels_list = images_to_levels(all_labels, num_level_anchors)
    label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
    bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
    bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
    keypoints_targets_list = images_to_levels(all_keypoints_targets,
                                              num_level_anchors)
    keypoints_weights_list = images_to_levels(all_keypoints_weights,
                                              num_level_anchors)

    num_total_samples = np.mean(num_total_pos, dtype=np.float32)
    num_total_samples = max(num_total_samples, 1.0)
    path = "/aidata/anders/3D-head/SCRFD/gt/logists"
    cls_score_0 = np.load(os.path.join(path, "cls_score_0.npy"))
    cls_score_1 = np.load(os.path.join(path, "cls_score_1.npy"))
    cls_score_2 = np.load(os.path.join(path, "cls_score_2.npy"))

    cls_scores = [cls_score_0, cls_score_1, cls_score_2]
    bbox_pred_0 = np.load(os.path.join(path, "bbox_pred_0.npy"))
    bbox_pred_1 = np.load(os.path.join(path, "bbox_pred_1.npy"))
    bbox_pred_2 = np.load(os.path.join(path, "bbox_pred_2.npy"))

    bbox_preds = [bbox_pred_0, bbox_pred_1, bbox_pred_2]
    kps_pred_0 = np.load(os.path.join(path, "kps_pred_0.npy"))
    kps_pred_1 = np.load(os.path.join(path, "kps_pred_1.npy"))
    kps_pred_2 = np.load(os.path.join(path, "kps_pred_2.npy"))
    kps_preds = [kps_pred_0, kps_pred_1, kps_pred_2]
    anchor_generator_strides = [(8, 8), (16, 16), (32, 32)]
    losses_cls, losses_bbox, losses_dfl, losses_kps,\
    avg_factor = multi_apply(
        loss_single,
        anchors_list,
        cls_scores,
        bbox_preds,
        kps_preds,
        labels_list,
        label_weights_list,
        bbox_targets_list,
        keypoints_targets_list,
        keypoints_weights_list,
        anchor_generator_strides,
        num_total_samples=num_total_samples)

    avg_factor = sum(avg_factor)
    losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
    losses = dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
    if use_kps:
        losses_kps = list(map(lambda x: x / avg_factor, losses_kps))
        losses['loss_kps'] = losses_kps
    if use_dfl:
        losses_dfl = list(map(lambda x: x / avg_factor, losses_dfl))
        losses['loss_dfl'] = losses_dfl
    return losses


path = "/aidata/anders/3D-head/SCRFD/gt"
img_metas = load_json(os.path.join(path, "img_metas.json"))
# img_metas = [img_metas]
valid_flag_list_0 = np.load(os.path.join(path, "valid_flag_list_0.npy"))
valid_flag_list_1 = np.load(os.path.join(path, "valid_flag_list_1.npy"))
valid_flag_list_2 = np.load(os.path.join(path, "valid_flag_list_2.npy"))

valid_flag_list = tf.concat([
    tf.cast(valid_flag_list_0, tf.bool),
    tf.cast(valid_flag_list_1, tf.bool),
    tf.cast(valid_flag_list_2, tf.bool)
],
                            axis=-1)
valid_flag_list = valid_flag_list[None, :]
anchor_list_0 = np.load(os.path.join(path, "anchor_list_0.npy"))
anchor_list_1 = np.load(os.path.join(path, "anchor_list_1.npy"))
anchor_list_2 = np.load(os.path.join(path, "anchor_list_2.npy"))

anchor_list_0 = tf.cast(anchor_list_0, tf.float32)  # (12800, 4)
anchor_list_1 = tf.cast(anchor_list_1, tf.float32)  # (3200, 4)
anchor_list_2 = tf.cast(anchor_list_2, tf.float32)  # (800, 4)

anchor_list = tf.concat([anchor_list_0, anchor_list_1, anchor_list_2], axis=-2)

anchor_list = anchor_list[None, :]

gt_bboxes_list = np.load(os.path.join(path, "gt_bboxes_list.npy"))
gt_bboxes = tf.cast(gt_bboxes_list, tf.float32)
gt_bboxes = gt_bboxes[None, :]

gt_keypointss_list = np.load(os.path.join(path, "gt_keypointss_list.npy"))
gt_keypointss = tf.cast(gt_keypointss_list, tf.float32)
gt_keypointss = gt_keypointss[None, :]

gt_labels_list = np.load(os.path.join(path, "gt_labels_list.npy"))
gt_labels = tf.cast(gt_labels_list, tf.float32)
gt_labels = gt_labels[None, :]

img_metas = [img_metas]

b_gt_bboxes_ignore = tf.zeros(shape=gt_bboxes.shape[:2], dtype=tf.bool)

cls_reg_targets = tf.py_function(get_targets,
                                 inp=[
                                     1, anchor_list, valid_flag_list, gt_bboxes,
                                     b_gt_bboxes_ignore, gt_keypointss,
                                     gt_labels, 1
                                 ],
                                 Tout=tf.float32)

print(cls_reg_targets)