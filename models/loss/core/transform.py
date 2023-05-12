import numpy as np
import tensorflow as tf


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
        x1 = tf.clip_by_value(x1, clip_value_min=0, clip_value_max=max_shape[1])
        y1 = tf.clip_by_value(y1, clip_value_min=0, clip_value_max=max_shape[0])
        x2 = tf.clip_by_value(x2, clip_value_min=0, clip_value_max=max_shape[1])
        y2 = tf.clip_by_value(y2, clip_value_min=0, clip_value_max=max_shape[0])
    return tf.stack([x1, y1, x2, y2], axis=-1)


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
        left = tf.clip_by_value(left,
                                clip_value_min=0,
                                clip_value_max=max_dis - eps)
        top = tf.clip_by_value(top,
                               clip_value_min=0,
                               clip_value_max=max_dis - eps)
        right = tf.clip_by_value(right,
                                 clip_value_min=0,
                                 clip_value_max=max_dis - eps)
        bottom = tf.clip_by_value(bottom,
                                  clip_value_min=0,
                                  clip_value_max=max_dis - eps)
    return tf.stack([left, top, right, bottom], axis=-1)


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
    C = 10
    for i in range(0, C, 2):
        px = kps[:, i] - points[:, i % 2]
        py = kps[:, i + 1] - points[:, i % 2 + 1]
        if max_dis is not None:
            px = tf.clip_by_value(px,
                                  clip_value_min=0,
                                  clip_value_max=max_dis - eps)
            py = tf.clip_by_value(py,
                                  clip_value_min=0,
                                  clip_value_max=max_dis - eps)
        preds.append(px)
        preds.append(py)
    return tf.stack(preds, axis=-1)
