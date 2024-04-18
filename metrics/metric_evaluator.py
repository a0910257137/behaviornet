import numpy as np
from pprint import pprint
from metrics.objects.metric.iou.iou import IoU
from .base import Base
from .objects.metric import PointDistance
from typing import Any, Dict, List, Tuple
from .objects.matcher.utils.misc import *
from .objects.transform import TRANSORM_FACTORY
from .objects.metric import MATCHER_FACTORY
from .objects.calculator import CALCULATOR_FACTORY
from .objects.calculator.reporter import REPORTER_FACTORY


class BDDMetricEvaluator(Base):
    """Evaluator for bdd files which is compatible with bddhelper
    """

    def __init__(self, cfg):
        super(BDDMetricEvaluator, self).__init__(cfg)

        self._data = None
        self.data_fmt = namedtuple('Data',
                                   ['raw_data', 'matched', 'unmatched'])
        self.paired_lbs: list = []
        self.cfg = cfg
        self.condition_cfg = self.cfg.conditions

        self.metric_type = self.cfg.metric_type
        self.matcher_threshold = self.condition_cfg.matcher_threshold
        self.reporter_threshold = self.condition_cfg.reporter_threshold

        self._transorm_func = TRANSORM_FACTORY.get(
            self.condition_cfg.transformer_method)
        self.metric = MATCHER_FACTORY.get(
            self.condition_cfg.object_matcher_method)
        self.calculator = CALCULATOR_FACTORY.get(
            self.condition_cfg.calculator_method)
        self.reporter = REPORTER_FACTORY.get(
            self.condition_cfg.reporter_method)

    def __call__(self, bdd_gt_annos, bdd_eval_annos):
        bdd_gt_annos, bdd_eval_annos = bdd_gt_annos[
            'frame_list'], bdd_eval_annos['frame_list']
        # users = {'gt', 'eval'}
        users = sorted(set(('gt', 'eval')), reverse=True)
        # print(bdd_gt_annos)
        # print('-' * 100)
        # print(bdd_eval_annos)
        # xxx
        matched_gt_frames_objects, matched_eval_frames_objects = self._matched_frame(
            bdd_gt_annos, bdd_eval_annos)
        users_matched_dc = {user: [] for user in users}
        users_unmatched_dc = {user: [] for user in users}
        raw_data = []
        # build user-frame-dict
        # call object matcher and need unders conditions
        gt_frames, eval_frames = matched_gt_frames_objects, matched_eval_frames_objects
        # frame-wise iterable
        if self.metric_type == 'NLE':
            return self.cal_nle(gt_frames, eval_frames)

        for gt_frame, eval_frame in zip(gt_frames, eval_frames):
            users_object_iterable_uc = (gt_frame, eval_frame)

            users_object_iterable_uc = dict(
                zip(users, users_object_iterable_uc))
            gt_frame, eval_frame = self.preprocess(users_object_iterable_uc)
            gt_shapes, eval_shapes, raw_gt_data, raw_eval_data, ret_metric = self.cal_matched_metric(
                gt_frame, eval_frame)
            # if metric type is norm localization error, we can not run mach object
            # because one frame with one face

            raw_data.append(dict(zip(users, (raw_gt_data, raw_eval_data))))
            cost_matrix = sum(ret_metric.values(), 0) if isinstance(
                ret_metric, dict) else ret_metric  # sum over all mutual metric

            assert isinstance(cost_matrix, np.ndarray)
            self._cost_matrix = cost_matrix

            if isinstance(self.metric, IoU):
                self._match_index = match_by_cost(-cost_matrix,
                                                  -self.matcher_threshold)
            elif isinstance(self.metric, PointDistance):
                self._match_index = match_by_cost(cost_matrix,
                                                  self.matcher_threshold)

            # (gt_frame, eval_frame)
            self.postprocess(users_object_iterable_uc)
            self.paired_lbs += list(self.__iter__())
            # collect non-matched-data
            for user in users:
                users_unmatched_dc[user].append(self.unmatched[user])
                users_matched_dc[user].append(self.matched[user])

        for user in users:
            users_unmatched_dc[user] = [users_unmatched_dc[user]]
            users_matched_dc[user] = [users_matched_dc[user]]

        self._data = self.data_fmt(raw_data=raw_data,
                                   matched=users_matched_dc,
                                   unmatched=users_unmatched_dc)

        matched_objects_wc = self.reform_data(self.data.matched)
        unmatched_objects_wc = self.reform_data(self.data.unmatched)
        # partial unders same condition
        matched_objects_uc_uf = matched_objects_wc[0]
        unmatched_objects_uc_uf = unmatched_objects_wc[0]
        raw_objects_uc_uf = self.data.raw_data

        # under the same frame
        for matched_objects_uf, unmatched_objects_uf, raw_objects_uf in zip(
                matched_objects_uc_uf, unmatched_objects_uc_uf,
                raw_objects_uc_uf):
            self.calculator(
                matched_objects_uf,
                raw_objects_uf,
                self._transorm_func,
                self.metric,
                self.reporter,
                self.reporter_threshold,
            )
        return self.calculator.report

    @property
    def data(self):
        return self._data

    def cal_nle(self, gt_frames, eval_frames):
        tmp_nle, tmp_interocular = [], []
        for i, (gt_frame, eval_frame) in enumerate(zip(gt_frames,
                                                       eval_frames)):
            _, _, _, _, ret_metric = self.cal_matched_metric(
                gt_frame, eval_frame)
            if i > 0:
                if not isinstance(ret_metric, dict):
                    continue
                tmp_nle += ret_metric['nle']
                tmp_interocular += ret_metric['interocular']
        ret_metric = {}
        num_samples = len(tmp_nle)

        ret_metric['nle'] = tmp_nle
        ret_metric['interocular'] = tmp_interocular
        ret_metric['num_samples'] = num_samples

        return ret_metric

    def cal_matched_metric(self, gt_lbs, eval_lbs):
        gt_shapes, eval_shapes = [], []
        raw_gt_data, raw_eval_data = [], []
        if not len(gt_lbs) or not len(eval_lbs):
            ret_metric = np.zeros(shape=(len(gt_lbs), len(eval_lbs)))
            for lb in gt_lbs:
                raw_gt_data.append(*lb)
            for lb in eval_lbs:
                raw_eval_data.append(*lb)
        else:
            for lb in gt_lbs:
                gt_shapes.append(self._transorm_func(*lb))
                raw_gt_data.append(*lb)
            for lb in eval_lbs:
                eval_shapes.append(self._transorm_func(*lb))
                raw_eval_data.append(*lb)
            ret_metric = self.metric(gt_shapes, eval_shapes)
        return gt_shapes, eval_shapes, raw_gt_data, raw_eval_data, ret_metric

    def preprocess(self, user_data_dict):
        """check and parser user-data-dict

        Args:
            user_data_dict ([type]): [description]

        Returns:
            [type]: [description]
        """
        assert user_data_dict.keys() == set(
            ['gt', 'eval']
        ), f'Expect user-data-dict has only `gt`, `eval` as key, got {user_data_dict.keys()}'
        gt_lbs = user_data_dict['gt']
        eval_lbs = user_data_dict['eval']

        assert isinstance(gt_lbs, list) and isinstance(eval_lbs, list)

        return gt_lbs, eval_lbs

    def _matched_frame(self, gt_annos, eval_annos):

        def iter_lb(frame, lb):
            outputs = []

            if self.metric_type.lower() == 'box2d':
                base_frame = {
                    'frame_id':
                    (frame['dataset'], frame['sequence'], frame['name']),
                    'box2d':
                    lb['box2d'],
                    'category':
                    lb['category']
                }
            elif self.metric_type.lower() in ['keypoints', 'landmarks', 'nle']:
                base_frame = {
                    'frame_id':
                    (frame['dataset'], frame['sequence'], frame['name']),
                    'keypoints':
                    lb['keypoints'],
                    'category':
                    lb['category']
                }
            outputs.append(base_frame)
            return outputs

        matched_gt_frames_objects, matched_eval_frames_objects = [], []
        for gt_frame, eval_frame in zip(gt_annos, eval_annos):
            gt_lbs = list(
                map(lambda lb: iter_lb(gt_frame, lb), gt_frame['labels']))
            eval_lbs = list(
                map(lambda lb: iter_lb(eval_frame, lb), eval_frame['labels']))
            matched_gt_frames_objects.append(gt_lbs)
            matched_eval_frames_objects.append(eval_lbs)
        return matched_gt_frames_objects, matched_eval_frames_objects

    @staticmethod
    def reform_data(user_cond_frame_iterable: Dict[str, List]):
        """reform executor.data.matched, executor.data.unmatched, executor.data.raw
        from:
            For example:
                executor.data.matched: {gt: [gt_cond1, gt_cond2, ...], 'eval': [eval_cond1, eval_cond2, ...]}
                gt_cond1: [gt_frame1, gt_frame2, ...]
                gt_frame1: gt_matched_objects(list)
        To:
                matched_objects_wc: [cond1, cond2, ...]
                cond1: [frame1, frame2, ...]
                frame1: {'gt': gt_matched_objects(list), 'eval': eval_matched_objects(list)}
        """
        # wc is short for with condition
        # uc is short for under condition
        user_objects_wc = []
        users = user_cond_frame_iterable.keys()
        for user_frame_iterable_uc in zip(*user_cond_frame_iterable.values()):
            cond_frame_iterable = []
            # loop over frame
            for user_object_iterable_uc_uf in zip(*user_frame_iterable_uc):
                user_object_iterable_dict = dict(
                    zip(users, user_object_iterable_uc_uf))
                cond_frame_iterable.append(user_object_iterable_dict)
            user_objects_wc.append(cond_frame_iterable)
        return user_objects_wc
