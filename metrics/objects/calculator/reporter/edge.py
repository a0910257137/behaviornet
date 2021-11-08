import dataclasses
from collections import defaultdict
from typing import Union

import numpy as np

from ....utils.io import path_converting
from ....utils.misc import sum_over_list_of_dict
from .accuracy import AccReporter

ROUND = 3


@dataclasses.dataclass
class EdgePRFReporter(AccReporter):
    """EdgePRFReporter supports edge metric calculation (thresholding)
    main edge metric calculation:
    for calculated metric result for a paired box2d
        box_edge_distance = dict(top_left_x=1, top_left_y=2, bottom_right_x=3, bottom_right_y=1)
        after thresholding by edge threshold 2
        distance_thresholding = dict(top_left_x=True, top_left_y=True, bottom_right_x=False, bottom_right_y=True)
        metric = 3/4
    Finally, for object level report, sum over metric and create `precision`, `recall`, 'f1-score'
    """
    threshold: dataclasses.InitVar[Union[dict, int, float]] = 0
    img_level: bool = False
    obj_level: bool = True
    _matched_count: dict = dataclasses.field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    _matched_accum: dict = dataclasses.field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    _imgs: list = dataclasses.field(default_factory=lambda: list())
    _object_count: dict = dataclasses.field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int)))

    def __post_init__(self, threshold):
        super().__post_init__(threshold)

    def reset(self):
        super().reset()
        self._object_count = defaultdict(lambda: defaultdict(int))

    @property
    def threshold(self):
        return self._threshold

    @property
    def object_count(self):
        return self._object_count

    @property
    def obj_level_report(self):
        report = defaultdict(float)
        if not self.matched_count.values():
            return report

        matched_count = sum_over_list_of_dict(list(
            self.matched_count.values()))
        matched_accum = sum_over_list_of_dict(
            list(self.matched_accumulations.values()))
        total_object_count = sum_over_list_of_dict(
            list(self.object_count.values()))

        report['matched_count'] = matched_count
        report['accuracy'] = {
            k: np.round(matched_accum[k] / report['matched_count'][k], ROUND)
            for k in matched_accum.keys()
        }
        report['recall'] = {
            k:
            np.round(matched_accum[k] / (total_object_count['gt'] or np.nan),
                     ROUND)
            for k in matched_accum.keys() if k not in ['n_update']
        }
        report['precision'] = {
            k:
            np.round(matched_accum[k] / (total_object_count['eval'] or np.nan),
                     ROUND)
            for k in matched_accum.keys() if k not in ['n_update']
        }
        report['f1-score'] = {
            k: np.round(
                (2 * report['recall'][k] * report['precision'][k]) /
                ((report['recall'][k] + report['precision'][k]) or np.nan),
                ROUND)
            for k in report['recall'].keys()
        }
        return report

    @property
    def img_level_report(self):
        report = defaultdict(lambda: dict())
        for frame_id in self.imgs:
            totol_object_count_in_frame = self.object_count[frame_id]
            matched_count_in_frame = dict(
                sorted((self.matched_count[frame_id].items())))
            self.matched_count[frame_id] = matched_count_in_frame
            matched_accumulations_in_frame = dict(
                sorted(self.matched_accumulations[frame_id].items()))
            report[path_converting(
                frame_id)]['matched_count'] = self.matched_count[frame_id]

            report[path_converting(frame_id)]['accuracy'] = {
                k: np.round(
                    matched_accumulations_in_frame[k] /
                    matched_count_in_frame[k], 3)
                for k in matched_accumulations_in_frame.keys()
            }
            precision = {
                k: np.round(
                    matched_accumulations_in_frame[k] /
                    totol_object_count_in_frame['eval'], ROUND)
                for k in matched_accumulations_in_frame.keys()
            }
            recall = {
                k: np.round(
                    matched_accumulations_in_frame[k] /
                    totol_object_count_in_frame['gt'], ROUND)
                for k in matched_accumulations_in_frame.keys()
            }
            f1_score = {
                k: np.round((2 * precision[k] * recall[k]) /
                            ((precision[k] + recall[k]) or np.nan), ROUND)
                for k in matched_accumulations_in_frame.keys()
            }
            report[path_converting(frame_id)]['precision'] = precision
            report[path_converting(frame_id)]['recall'] = recall
            report[path_converting(frame_id)]['f1-score'] = f1_score
        return report

    def collect_object_counts(self, raw_objects: dict):
        """collect gt and eval object counts based on frame

        Args:
            raw_objects (dict): raw object lists
        """
        for user, lbs_in_frame in raw_objects.items():
            for lb in lbs_in_frame:
                self._object_count[lb['frame_id']][user] += 1

    def collect(self, results: dict, frame_id: str):
        """collect the thresholding results
           distance_thresholding = dict(top_left_x=True, top_left_y=True, bottom_right_x=False, bottom_right_y=True)
           edge_metric = 3/4
        """
        edge_metric = sum(results.values()) / len(results.values())
        all_correct = all(list(results.values()))

        results.update({'all_correct': all_correct, 'avg': edge_metric})

        for k, v in results.items():
            self._matched_accum[frame_id][k] += v
            self._matched_count[frame_id][k] += 1
        self._matched_count[frame_id]['n_update'] += 1
        self._imgs.append(frame_id)
