from .fpn import FPN
from .pafpn import PAFPN

NECK_FACTORY = dict(fpn=FPN, pafpn=PAFPN)
