import dataclasses
from abc import abstractmethod
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
from bddhelper.objects import BaseLabelObject

from ...utils.typing import BDDFrameDict, LabelObject
from ..transform import ENT_TRANSFORMS, Compose, Transformer, TransformerFactory


@dataclasses.dataclass
class ObjectMatcher:
    """Object Matcher is used for matching gt labels and eval labels
    `threshold` and `DEFAULT_THRESH` is used to prevent user forgets to pass threshold to object 
    matcher. If there is any object matcher which does not use `threshold`, just inherit from ObjectMatcher
    and set `DEFAULT_THRESH` = None to pass the check
    """
    threshold: Optional[Union[int, float]] = None
    transform: Optional[Union[dict, Compose, Transformer, List[
        Union[Transformer, dict]]]] = None
    DEFAULT_THRESH: ClassVar[int] = -1

    def __post_init__(self):
        assert isinstance(
            self.threshold,
            (int, float)) or self.threshold == self.DEFAULT_THRESH
        if not isinstance(self.transform, list) and self.transform is not None:
            self.transform = [self.transform]

        if self.transform is not None:
            _transform = []
            for trans in self.transform:
                if isinstance(trans, (Transformer, Compose)):
                    _transform.append(trans)
                else:
                    assert isinstance(trans, dict) and 'name' in trans
                    _transform.append(TransformerFactory.create(**trans))
            assert self.is_valid_transform(
                _transform
            ), f'Expected the last transform is a transform whose `ENT` is True. Valid: {ENT_TRANSFORMS}, Got {_transform[-1]}'
            self.transform = Compose(_transform)

    @staticmethod
    def is_valid_transform(transform: Union[Transformer, Compose]) -> bool:
        """check the transform is valid or not. In general case of ObjectMatcher, `GetShape` or `GetAttributes` are expected.
        """
        last_transform = transform[-1]
        if isinstance(last_transform, Compose):
            return ObjectMatcher.is_valid_transform(last_transform)
        else:
            return last_transform.ENT

    @staticmethod
    def parse(
            compound_lb: Union[LabelObject, Dict[str, Union[LabelObject, Any]]]
    ):
        # gt_compound_lb: LabelObject or {'frame_id': Any, 'object': Union[BDDObject, dict]}
        assert isinstance(
            compound_lb, (dict, BaseLabelObject)
        ), 'Expected compound_lb in "{"frame_id":Any, "object": Union[dict, BaseLabelObject]}" or Union[dict, BaseLabelObject] directly, got compound_lb: {compound_lb}'
        try:
            return compound_lb['object']
        except KeyError:
            return compound_lb

    @abstractmethod
    def __call__(self):
        # how to match gt labels and eval labels
        pass

    @abstractmethod
    def __iter__(self):
        # how to iterate all matched objects
        # yield Dict[user, objects]
        pass

    @abstractmethod
    def eval(self, **kwargs):
        pass


@dataclasses.dataclass
class PairwiseObjectMatcher(ObjectMatcher):
    """Object Matcher is used for matching gt labels and eval labels
    """
    @abstractmethod
    def __call__(self, user_data_dict: BDDFrameDict):
        # how to match gt labels and eval labels
        """
        1. process input user-data-dict
            a get user-defined target shape from each label
        2. eval cost-matrix [ and post-process of cost-matrix]
        3. deduce matching from cost-matrix in post-process
        2. transform each shape of label
        3. compare by metric and return results between mutual labels
        4. matched each labels by user-defined threshold

        Note:

        gt_lbs: List[lb]
        lb: {'frame_id': Any, 'object': Union[BDDObject, dict]}
        return metric
        e.g.

        gt_lbs, eval_lbs = self.preprocess(user_data_dict)
        ret_metric = self.eval(gt_lbs, eval_lbs)
        self._cost_matrix = ret_metric
        self._match_index = match_by_cost(-ret_metric, -self.threshold)
        self.postprocess(user_data_dict)
        return self


        """
        pass

    def __iter__(self):
        # how to iterate all matched gt label and eval label
        # yield Dict[user, object]
        for a, b, *c in zip(self.matched['gt'], self.matched['eval'],
                            *self._extra_info.values()):
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

    def eval(self, gt_lbs, eval_lbs):
        """
        1. get user-defined target shape from each label
        2. transform each shape of label
        3. get cost-matrix

        Note:
        gt_lbs: List[lb]
        lb: {'frame_id': Any, 'object': Union[BDDObject, dict]}
        return metric
        """

        if not gt_lbs or not eval_lbs:
            # return np.emtpy will cause some un-expected matching results
            return np.zeros(shape=(len(gt_lbs), len(eval_lbs)))

        # parse shape and apply transform
        gt_shapes = [
            self.transform(self.parse(lb))
            if self.transform is not None else self.parse(lb) for lb in gt_lbs
        ]
        eval_shapes = [
            self.transform(self.parse(lb))
            if self.transform is not None else self.parse(lb)
            for lb in eval_lbs
        ]
        ret_metric = self.metric(gt_shapes, eval_shapes)
        return ret_metric

    @property
    def cost_matrix(self):
        return self._cost_matrix
