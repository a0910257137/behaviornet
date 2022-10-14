from .center_od import CenterODLoss
from .center_head import CenterHeadLoss

LOSS_FACTORY = dict(center_od=CenterODLoss, center_head=CenterHeadLoss)
