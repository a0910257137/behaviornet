from .fpn import FPN
from .hard_fpn import HFPN
from .yolo_fpn import YFPN

NECK_FACTORY = dict(fpn=FPN, hard_fpn=HFPN, yolo_fpn=YFPN)
