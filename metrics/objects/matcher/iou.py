import dataclasses
from typing import Dict, List

from ..metric.iou import IoU2D
from .base import PairwiseObjectMatcher
from .utils.misc import match_by_cost


@dataclasses.dataclass
class IoU2DMatcher(PairwiseObjectMatcher):
    def __post_init__(self):
        super().__post_init__()
        self.metric = IoU2D()

    def __call__(self, user_data_dict: Dict[str, List[dict]]):
        gt_lbs, eval_lbs = self.preprocess(user_data_dict)
        ret_metric = self.eval(gt_lbs, eval_lbs)
        self._cost_matrix = ret_metric
        self._match_index = match_by_cost(-ret_metric, -self.threshold)
        self.postprocess(user_data_dict)
        return self
