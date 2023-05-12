import numpy as np
import tensorflow as tf


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


def bbox_overlapping(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    INF = 1e6
    batch_shape = tf.shape(bboxes1)
    rows = tf.shape(bboxes1)[-2]
    cols = tf.shape(bboxes2)[-2]

    # if is_aligned:
    #     assert rows == cols
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] -
                                                   bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] -
                                                   bboxes2[..., 1])
    if is_aligned:
        lt = tf.math.maximum(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = tf.math.minimum(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = tf.clip_by_value((rb - lt), clip_value_min=0,
                              clip_value_max=INF)  # [B, rows, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = tf.math.minimum(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = tf.math.maximum(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = tf.math.maximum(bboxes1[..., :, None, :2],
                             bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = tf.math.minimum(bboxes1[..., :, None, 2:],
                             bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = tf.clip_by_value((rb - lt), clip_value_min=0,
                              clip_value_max=INF)  # [B, rows, cols, 2]

        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]

    eps = tf.constant([eps])
    # eps = union.new_tensor([eps])
    union = tf.math.maximum(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
