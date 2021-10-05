from .gfc_loss import GFCLoss
from .center_od import CenterODLoss

LOSS_FACTORY = dict(center_od=CenterODLoss, gfc=GFCLoss)
