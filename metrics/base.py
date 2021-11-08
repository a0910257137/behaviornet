from abc import abstractmethod
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union
import numpy as np
from pprint import pprint


class Base:
    def __init__(self, cfg):
        self.cfg = cfg

    def __iter__(self):
        # how to iterate all matched gt label and eval label
        # yield Dict[user, object]
        for a, b, *c in zip(self.matched['gt'], self.matched['eval'],
                            *self._extra_info.values()):
            a = a[0]
            b = b[0]
            info = dict(zip(self._extra_info.keys(), c))
            a.update(info)
            b.update(info)
            yield dict(gt=a, eval=b)

    def preprocess(self, user_data_dict) -> Tuple[List[dict], List[dict]]:
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

    def postprocess(self, user_data_dict) -> None:
        """[summary]
             post-process data with match index, and input data

        Args:
            user_data_dict ([type]): [description]
            match_index ([type]): [description]
            cost_matrix ([type]): [description]
        """
        gt_match_idx = self._match_index.gt_match_idx
        eval_match_idx = self._match_index.eval_match_idx
        gt_non_match_idx = self._match_index.gt_non_match_idx
        eval_non_match_idx = self._match_index.eval_non_match_idx

        # user object iterable for matched-data
        self._matched = dict()
        self._matched['gt'] = [
            user_data_dict['gt'][idx] for idx in gt_match_idx
        ]
        self._matched['eval'] = [
            user_data_dict['eval'][idx] for idx in eval_match_idx
        ]
        # user object iterable for unmatched-data
        self._unmatched = dict()
        self._unmatched['gt'] = [
            user_data_dict['gt'][idx] for idx in gt_non_match_idx
        ]
        self._unmatched['eval'] = [
            user_data_dict['eval'][idx] for idx in eval_non_match_idx
        ]
        self._extra_info = dict()
        self._extra_info.update(
            cost=self.cost_matrix[gt_match_idx, eval_match_idx])

    @property
    def matched(self):
        return self._matched

    @property
    def unmatched(self):
        return self._unmatched

    @property
    def cost_matrix(self):
        return self._cost_matrix
