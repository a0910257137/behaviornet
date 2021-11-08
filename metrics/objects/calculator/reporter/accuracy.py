import copy
import dataclasses
from collections import defaultdict
from typing import Optional, Union

from ....utils.io import path_converting
from ....utils.misc import sum_over_list_of_dict
from .base import Reporter

ROUND = 3


@dataclasses.dataclass
class AccReporter(Reporter):
    """Accuracy reporter takes calculated results and makes statistics
    
    Usage:
        # obj_level is default
        reporter = AccReporter(threshold=2)
        reporter.update({'top_left':1, 'bottom_right':3})
        print(reporter.report)
    """
    threshold: dataclasses.InitVar[Union[dict, int, float]] = None
    img_level: bool = False
    obj_level: bool = True
    _matched_count: dict = dataclasses.field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    _matched_accum: dict = dataclasses.field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    _imgs: list = dataclasses.field(default_factory=lambda: list())

    def __post_init__(self, threshold):
        if isinstance(threshold, property):
            threshold = None

        assert isinstance(threshold, (int, float, dict)) or threshold is None
        self.raw_thresh = threshold
        self._threshold = defaultdict(lambda: threshold) if isinstance(
            threshold, (int, float)) else threshold
        assert isinstance(self.img_level,
                          bool), f'Expected bool, got {type(self.img_level)}'
        assert isinstance(self.obj_level,
                          bool), f'Expected bool, got {type(self.obj_level)}'
        assert self.img_level or self.obj_level, f'Expected one of img_level or obj_level be true'

    def reset(self):
        self._matched_count = defaultdict(lambda: defaultdict(int))
        self._matched_accum = defaultdict(lambda: defaultdict(int))
        self._imgs = []

    @property
    def threshold(self):
        return self._threshold

    @property
    def matched_accumulations(self):
        return self._matched_accum

    @property
    def matched_count(self):
        return self._matched_count

    @property
    def imgs(self):
        return sorted(list(set(self._imgs)))

    @property
    def obj_level_report(self):
        report = defaultdict(float)
        if not self.matched_count.values():
            return report
        matched_count = sum_over_list_of_dict(list(
            self.matched_count.values()))
        report['matched_count'] = matched_count

        # when using Counter + Counter, if dict contains {'a':0}, the output will
        # not contain key:'a'. Therefore, using defaultdict(int) and update
        matched_accum = defaultdict(int)
        matched_accum.update(
            sum_over_list_of_dict(list(self.matched_accumulations.values())))
        report['accuracy'] = {
            k: round(matched_accum[k] / matched_count[k], ROUND)
            for k in matched_count.keys() if k not in ['n_update']
        }
        return report

    @property
    def img_level_report(self):
        report = defaultdict(lambda: dict())
        for frame_id in self.imgs:
            matched_count_in_frame = dict(
                sorted(self.matched_count[frame_id].items()))
            matched_accum_in_frame = dict(
                sorted(self.matched_accumulations[frame_id].items()))

            report[path_converting(
                frame_id)]['matched_count'] = matched_count_in_frame
            report[path_converting(frame_id)]['accuracy'] = {
                k: round(matched_accum_in_frame[k] / matched_count_in_frame[k],
                         ROUND)
                for k in matched_accum_in_frame
            }
        return report

    @property
    def report(self):
        report = dict()
        if self.img_level:
            report['img_level'] = self.img_level_report
        if self.obj_level:
            report['obj_level'] = self.obj_level_report
        return report

    def get_threshold(self,
                      amplifier: Optional[Union[dict, int, float]] = None):
        """set the threshold based on amplifier

        Args:
            amplifier (Optional[Union[dict, int, float]], optional): threshold amplifier. Defaults to None.

        Returns:
            [type]: threshold
        """
        assert isinstance(amplifier, (dict, int, float)) or amplifier is None
        if self.threshold is None:
            # for non regression case
            threshold = None
        else:
            if isinstance(amplifier, dict):
                threshold = copy.deepcopy(self.threshold)
                if isinstance(self.threshold, dict) and self.threshold:
                    assert set(amplifier.keys()).issubset(
                        self.threshold.keys())
                threshold.update(
                    {k: self.threshold[k] * v
                     for k, v in amplifier.items()})
            elif isinstance(amplifier, (int, float)):
                if not self.threshold:
                    assert isinstance(self.raw_thresh, (int, float))
                    threshold = defaultdict(
                        lambda: self.raw_thresh * amplifier)
                else:
                    threshold = {
                        k: amplifier * v
                        for k, v in self.threshold.items()
                    }
            else:
                threshold = self.threshold
        return threshold

    @staticmethod
    def result_thresholding(metric_result: dict,
                            threshold: Optional[dict] = None,
                            reversed_op: bool = False) -> dict:
        """Thresholding function

        Args:
            metric_result (dict): metric comparison result
            threshold (Optional[dict], optional): threshold. Defaults to None.
            reversed_op (bool, optional): comparision operation only used in regression, reversed_op=False takes value <= threhsold, reversed_op=True takes value >= threshold. Defaults to False.

        Returns:
            dict: thresholding results in bool type
        """
        if threshold is not None:
            # If threshold is specified, applying regression comparison
            # str case will be key missing or key redundant
            assert all(
                isinstance(v, (int, float)) or v in ['missing', 'redundant']
                for v in metric_result.values())

            for k, v in metric_result.items():
                if isinstance(v, str):
                    metric_result[k] = False
                elif reversed_op:
                    metric_result[k] = v >= threshold
                else:
                    metric_result[k] = v <= threshold
        else:
            # for attributes
            for k, v in metric_result.items():
                if isinstance(v, str):
                    metric_result[k] = False
        assert all(isinstance(v, bool) for v in metric_result.values())
        return metric_result

    def collect(self, results: dict, frame_id: str):
        """collect info from thresholding results and update frame_id.
        For each update, if all keys return ture, reporter will add up 'all_correct'. 

        Args:
            results (dict): [description]
            frame_id (str): [description]
        
        Note:
            structure of self._matched_count, self._matched_accum
            self._matched_count : {
                'frame1': {
                    'k1': int,
                    'k2': int,
                    'all_correct': int,
                    'n_update':int
                }
            }
            self._matched_accum : {
                'frame1': {
                    'k1': int,
                    'k2': int,
                    'all_correct':int
                }
            }
        """
        all_correct = all(list(results.values()))
        results.update({'all_correct': all_correct})

        for k, v in results.items():
            self._matched_accum[frame_id][k] += v
            self._matched_count[frame_id][k] += 1
        self._matched_count[frame_id]['n_update'] += 1
        self._imgs.append(frame_id)

    def update(self,
               value_with_frame: dict,
               threshold: Optional[Union[str, int, float]],
               threshold_amplifier: Optional[Union[dict, int, float]] = None,
               reversed_op: bool = False):
        """Update reporter statistic based on value

        Args:
            value (dict): value with different keys
            threshold_amplifier (Optional[Union[dict, int, float]], optional): threshold amplifier. Defaults to None.
            reversed_op (bool, optional): comparision operation only used in regression, reversed_op=False takes value <= threhsold, reversed_op=True takes value >= threshold. Defaults to False.
        """
        assert isinstance(value_with_frame,
                          dict), f'Expected dict, got {type(value_with_frame)}'
        if 'frame_id' not in value_with_frame:
            result = {}
            result['frame_id'] = self.__default_frame_id__
            result['metric'] = value_with_frame
        else:
            result = value_with_frame
        frame_id = result.get('frame_id', self.__default_frame_id__)

        metric_result = result.get('metric')
        # threshold = self.get_threshold(threshold_amplifier)
        results = self.result_thresholding(metric_result=metric_result,
                                           threshold=threshold,
                                           reversed_op=reversed_op)
        self.collect(results, frame_id)
