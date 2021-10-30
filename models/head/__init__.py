from .head import Head
from .nanodet_head import NanoDetHead
from .yolo_head import YDetHead

HEAD_FACTORY = dict(head=Head, nanodet_head=NanoDetHead, yolo_head=YDetHead)
