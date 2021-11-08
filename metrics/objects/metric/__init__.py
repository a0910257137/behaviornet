from .attributes import AttributeEqual
from .base import Metric
from .iou import IoU2D
from .shift import HausdorffDistance, PointAxialShift, PointDistance

MATCHER_FACTORY = {'PointDistance': PointDistance(), 'IoU2D': IoU2D()}
