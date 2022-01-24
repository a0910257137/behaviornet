from .head import Head
from .yolo_head import YDetHead

HEAD_FACTORY = dict(head=Head, yolo_head=YDetHead)
