from .pan import PAN
from .fpn import FPN
from .simple_fpn import SFPN

NECK_FACTORY = dict(sfpn=SFPN, fpn=FPN, pan=PAN)
