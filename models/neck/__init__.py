from models.neck.fpn import FPN
from .pan import PAN
from .fpn import FPN

NECK_FACTORY = dict(fpn=FPN, pan=PAN)
