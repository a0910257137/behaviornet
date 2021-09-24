from .od_loss import ObjectDetLoss
from .gfc_loss import GFCLoss

LOSS_FACTORY = dict(od=ObjectDetLoss, gfc=GFCLoss)
