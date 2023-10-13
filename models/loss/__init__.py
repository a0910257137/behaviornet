from .center_od import CenterODLoss
from .center_head import CenterHeadLoss
from .anchor_loss import AnchorLoss
from .anchor_tood_loss import AnchorTOODLoss

LOSS_FACTORY = dict(
    center_od=CenterODLoss,
    center_head=CenterHeadLoss,
    anchor_loss=AnchorLoss,
    anchor_tood_loss=AnchorTOODLoss,
)
