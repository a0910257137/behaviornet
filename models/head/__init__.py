from .head import Head
from .nanodet_head import NanoDetHead

HEAD_FACTORY = dict(head=Head, nanodet_head=NanoDetHead)
