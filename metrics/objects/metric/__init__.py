from .attributes import AttributeEqual
from .base import Metric
from .iou import IoU2D

from .shift import HausdorffDistance, PointAxialShift, PointDistance, NLE

MATCHER_FACTORY = {
    'PointDistance': PointDistance(),
    'IoU2D': IoU2D(),
    'NLE': NLE()
}
