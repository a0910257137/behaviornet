import dataclasses
from typing import Dict, List, Optional, Union
from .accuracy import AccCalculator
from pprint import pprint


@dataclasses.dataclass
class PRFCalculator(AccCalculator):
    """PRFCalculator supports precision, recall, f1 score calculation

    Usage:
        metric = PointAxialShift()
        transform = Compose([Box2DToKeyPoints(), MinimalBox2D(), GetShape()])
        calculator = PRFCalculator(metric, transform, parser)
        calculator(matched_objects={'gt':gt_lbs, 'eval':eval_lbs})
        print(calculator.report)
    """
    def __call__(self,
                 matched_objects: Dict[str, List],
                 raw_objects: Dict[str, List],
                 transform: object,
                 metric: object,
                 reporter: object,
                 threshold: Optional[Union[int, str, float]],
                 threshold_amplifier: Optional[Union[dict, int, float]] = None,
                 discarded_keys: Optional[Union[list, tuple]] = None,
                 **kwargs):
        super().__call__(matched_objects, transform, metric, reporter,
                         threshold, discarded_keys, threshold_amplifier)

        self.reporter.collect_object_counts(raw_objects=raw_objects)
