from .fpn import FPN
from .pafpn import PAFPN
from .slim_neck import SlimNeck

NECK_FACTORY = dict(fpn=FPN, pafpn=PAFPN, slim_neck=SlimNeck)
