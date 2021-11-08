import dataclasses
from typing import Dict, List
import numpy as np
from ..metric.shift import PointDistance
from .base import PairwiseObjectMatcher
from .utils.misc import match_by_cost


@dataclasses.dataclass
class PointDistanceMatcher(PairwiseObjectMatcher):
    """An object matcher based on target shape, compare mutual point distance and sum over each key point, and match labels by threshold.
        Usage:
            object_matcher = PointDistanceMatcher(threshold=2, transform=Compose([Box2DToKeyPoints(), MinimalBox2D()]))
            for gt_matched_lb, eval_matched_lb in object_matcher(dict(gt=gt_lbs, eval=eval_lbs)):
                # do comparison on matched gt, eval labels
    """
    def __post_init__(self):
        super().__post_init__()
        self.metric = PointDistance()

    def __call__(self, user_data_dict: Dict[str, List[dict]]):

        gt_lbs, eval_lbs = self.preprocess(user_data_dict)
        ret_metric = self.eval(gt_lbs, eval_lbs)

        # post process on cost matrix
        cost_matrix = sum(ret_metric.values(), 0) if isinstance(
            ret_metric, dict) else ret_metric  # sum over all mutual metric
        assert isinstance(cost_matrix, np.ndarray)
        self._cost_matrix = cost_matrix

        self._match_index = match_by_cost(cost_matrix, self.threshold)
        self.postprocess(user_data_dict)
        return self
