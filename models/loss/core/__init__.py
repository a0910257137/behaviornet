from .smooth_l1_loss import SmoothL1Loss
from .gfocal_loss import QualityFocalLoss
from .task_aligned_focal_loss import FocalLossWithProb, TaskAlignedFocalLoss
from .iou_loss import DIoULoss, GIoULoss

LOSS_FUNCS_FACTORY = dict(
    QualityFocalLoss=QualityFocalLoss,
    FocalLossWithProb=FocalLossWithProb,
    TaskAlignedFocalLoss=TaskAlignedFocalLoss,
    SmoothL1Loss=SmoothL1Loss,
    DIoULoss=DIoULoss,
    GIoULoss=GIoULoss,
)
