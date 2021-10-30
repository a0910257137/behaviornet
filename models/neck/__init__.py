from .pan import PAN
from .fpn import FPN
from .simple_fpn import SFPN
from .yolo_fpn import YFPN

NECK_FACTORY = dict(sfpn=SFPN, fpn=FPN, pan=PAN, yolo_fpn=YFPN)
