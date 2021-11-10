import dataclasses
from typing import Union

import numpy as np
from monitor import logger
from .base import Metric


class AttributeEqual(Metric):
    """Compare two dict the same or not (key by key)
    
    Usage:
        
            gt = dict(a=1, b=2, d=2)
            eval = dict(a=1, b=1, c=1)
            metric = AttributeEqual()
            eqaul = metric(gt, eval)
            print(equal)
    Output:
            
            {'a': True, 'b': False, 'c': 'redundant', 'd': 'missing'}
    """
    order: Union[dict, bool] = True

    def __post_init__(self):
        assert isinstance(self.order, (dict, bool))

    def __call__(self, gt_lb: dict, eval_lb: dict) -> dict:
        assert self.check_attributes(gt_lb) and self.check_attributes(eval_lb)
        result = self.calculate(gt_lb, eval_lb, order=self.order)
        return dict(sorted(result.items(), key=lambda kv: kv[0]))

    def calculate(self, gt_lb, eval_lb, **kwargs):
        return self.check_equal(gt_lb, eval_lb, **kwargs)

    @staticmethod
    def check_attributes(lb: dict):
        assert isinstance(lb, dict), f'Expected a dict, got {type(lb)}'
        return True

    @staticmethod
    def check_equal(gt, eval, **kwargs) -> Union[bool, dict]:
        """Recurrively compare two object equal or not.
        if gt and eval are dicts, return will be dict with the values(bool)
        """
        if isinstance(gt, dict):
            assert isinstance(
                eval, dict
            ), f'Expected eval has the same type as gt: {type(gt)}, got {type(eval)}'
            results = {}
            all_keys = set(gt.keys()).union(eval.keys())
            for k in all_keys:
                # {key: value} pairs in kwargs should have same structure as the gt and eval
                # otherwise, pairs will be vinished.
                k_kwargs = {
                    key: val[k]
                    for key, val in kwargs.items()
                    if isinstance(val, dict) and k in val
                }
                results[k] = AttributeEqual.check_equal(
                    gt.get(k, 'empty'), eval.get(k, 'empty'), **k_kwargs)
            return results
        elif 'empty' in [gt, eval]:
            return 'redundant' if gt == 'empty' else 'missing'
        elif isinstance(gt, np.ndarray):
            if not isinstance(eval, np.ndarray):
                logger.info(
                    f'Expected eval has the same type as gt: {type(gt)}, got {type(eval)}'
                )
                return False
            return np.array_equal(gt, eval)
        elif isinstance(gt, (list, tuple)):
            if not isinstance(eval, (list, tuple)):
                logger.info(
                    f'Expected eval has the same type as gt: {type(gt)}, got {type(eval)}'
                )
                return False
            order = kwargs.get('order', True)  # True: order-sensitive
            return gt == eval if order else (set(gt) == set(eval)
                                             and len(gt) == len(eval)
                                             and type(gt) is type(eval))
        else:
            return gt == eval
