from collections import namedtuple
from typing import Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def match_by_cost(cost_matrix: np.ndarray,
                  thresh: float) -> Tuple[list, list, list, list]:
    """Given the matching and non-matching index for gt and eval objs
    Arguments:
        cost_matrix {np.ndarray} -- Cost matrix for evaluating each gt and each eval mutually
        thresh {float} -- threshold of mapping
    Returns:
        Tuple[list, list, List, List] -- In order is gt_match_idx, gt_non_match_idx, eval_match_idx, eval_non_match_idx
    """
    DataIndex = namedtuple('DataIndex', [
        'gt_match_idx', 'gt_non_match_idx', 'eval_match_idx',
        'eval_non_match_idx'
    ])
    assert isinstance(cost_matrix, np.ndarray) and cost_matrix.ndim == 2
    assert isinstance(thresh, (int, float))
    
    gt_idx, eval_idx = linear_sum_assignment(cost_matrix)
    match_idx = cost_matrix[gt_idx, eval_idx] <= thresh

    gt_match_idx = gt_idx[match_idx].tolist()
    eval_match_idx = eval_idx[match_idx].tolist()

    n_gt, n_eval = cost_matrix.shape

    gt_non_match_idx = list(set(range(n_gt)) - set(gt_match_idx))
    eval_non_match_idx = list(set(range(n_eval)) - set(eval_match_idx))

    return DataIndex(gt_match_idx=gt_match_idx,
                     gt_non_match_idx=gt_non_match_idx,
                     eval_match_idx=eval_match_idx,
                     eval_non_match_idx=eval_non_match_idx)
