import dataclasses
from typing import List, Union
import numpy as np
from ..base import Metric


@dataclasses.dataclass
class PointShift(Metric):
    def __call__(self, gt_pts: Union[List[dict], dict],
                 eval_pts: Union[List[dict], dict]) -> dict:
        """deal with single point to point metric and multiple points to points metric
        """
        single_item_flag = True if isinstance(gt_pts, dict) and isinstance(
            eval_pts, dict) else False
        if isinstance(gt_pts, dict):
            gt_pts = [gt_pts]
        if isinstance(eval_pts, dict):
            eval_pts = [eval_pts]

        assert self.check_shape(gt_pts) and self.check_shape(eval_pts)

        # shape will be (Batch, n_dim)
        gt_pts = {
            k: np.array([pt[k].flatten() for pt in gt_pts])
            for k in gt_pts[0].keys()
        }
        eval_pts = {
            k: np.array([pt[k].flatten() for pt in eval_pts])
            for k in eval_pts[0].keys()
        }

        result = self.calculate(gt_pts, eval_pts)

        result = {
            k: v[0][0].item()
            if single_item_flag and isinstance(v, np.ndarray) else v
            for k, v in result.items()
        }
        return result

    @staticmethod
    def calculate(gt_pts, eval_pts):
        """Expected output for "calculate" function will be NxM numpy array (row(N)-> gt, col(M)-> eval)
        """
        raise NotImplementedError

    @staticmethod
    def check_shape(pts: List[dict]):
        keys = None
        array_shape = None
        for pt in pts:
            assert isinstance(pt, dict), f'Expected dict, got {type(pt)}'
            if keys is None:
                keys = set(pt.keys())
            assert set(pt.keys()) == keys

            for coord in pt.values():
                assert isinstance(coord, np.ndarray)
                if array_shape is None:
                    array_shape = coord.shape
                # allow (1, n_dim), (n_dim,1), (n_dim,) for each point
                # all points from the same source(gt, eval) should be the same shape
                assert coord.ndim <= 2 and coord.shape == array_shape, f'Expected dim <= 2 and shape: {array_shape}, got dim: {coord.ndim}, shape: {coord.shape}'
                assert all(
                    isinstance(p, (np.integer, np.float))
                    for p in coord.flatten()
                ), f'coords must be int or float, got {[type(p) for p in coord.flatten()]}'
        return True


class PointAxialShift(PointShift):
    '''
    point-wise shift distance by axis
    e.g.
    gt_pts = {'head':np.array([1, 2]), 'eye':np.array([2, 3]), 'nose': np.array([4,5])}
    eval_pts = {'head':np.array([1.5, 2]), 'eye': np.array([2, 3.7]), 'hand': np.array([10,10])}
    metric = PointAxialShift()
    shift = metric(gt_pts, eval_pts)
    shift: {'head_x': 0.5, 'head_y':0, 'eye_x':0, 'eye_y':0.7, 'nose_x': 'missing', 'nose_y':'missing','hand_x':'redundant', 'hand_y':'redundant'}
    '''
    SUFFIXES = ['x', 'y', 'z']

    def calculate(self, gt_pts: dict, eval_pts: dict) -> dict:
        # pts in gt_pts and eval_pts must be (Batch, n_dim)
        assert isinstance(gt_pts, dict) and isinstance(eval_pts, dict)
        assert all(
            isinstance(coord, np.ndarray) and coord.ndim == 2
            for coord in gt_pts.values())
        assert all(
            isinstance(coord, np.ndarray) and coord.ndim == 2
            for coord in eval_pts.values())
        shifts = dict()
        all_keys = set(gt_pts.keys()).union(set(eval_pts.keys()))
        for key in all_keys:
            gt_value = gt_pts.get(key, 'redundant')
            eval_value = eval_pts.get(key, 'missing')
            if isinstance(gt_value, str) or isinstance(eval_value, str):
                n_dim, phase = (gt_value.shape[-1], eval_value) if isinstance(
                    gt_value, np.ndarray) else (eval_value.shape[-1], gt_value)
                for suffix in self.SUFFIXES[:n_dim]:
                    shifts['_'.join([key, suffix])] = phase
                continue
            assert gt_value.shape[1:] == eval_value.shape[1:]
            for gt_coord, eval_coord, suffix in zip(gt_value.T[:, None, :],
                                                    eval_value.T[..., None],
                                                    self.SUFFIXES):
                shifts['_'.join([key,
                                 suffix])] = np.abs(gt_coord.T - eval_coord)
        return shifts


@dataclasses.dataclass
class PointDistance(PointShift):
    '''
    point distance directly
    e.g.
    gt_pts = {'head':np.array([[1, 2]]), 'eye':np.array([[2, 3]]), 'nose':np.array([])}
    eval_pts = {'head':np.array([[4, 6]]), 'eye':np.array([[6, 6]])}
    metric = PointDistance()
    shift = metric(gt_pts, eval_pts)
    shift: {'head': 5.0, 'eye': 5.0}
    '''
    @staticmethod
    def calculate(gt_pts: dict, eval_pts: dict) -> dict:
        # pts in gt_pts and eval_pts must be (Batch, n_dim)
        assert isinstance(gt_pts, dict) and isinstance(eval_pts, dict)
        assert all(
            isinstance(coord, np.ndarray) and coord.ndim == 2
            for coord in gt_pts.values())
        assert all(
            isinstance(coord, np.ndarray) and coord.ndim == 2
            for coord in eval_pts.values())
        all_keys = set(gt_pts.keys()).union(set(eval_pts.keys()))
        distance = dict()
        for key in all_keys:
            gt_value = gt_pts.get(key, 'redundant')
            eval_value = eval_pts.get(key, 'missing')
            if isinstance(gt_value, str) or isinstance(eval_value, str):
                distance[key] = gt_value if isinstance(gt_value,
                                                       str) else eval_value
            else:
                assert gt_value.shape[1:] == eval_value.shape[
                    1:], f'got invalid shape: {gt_value.shape} and {eval_value.shape}. The shape after first dimension should be the same'
                distance[key] = np.linalg.norm(gt_value[:, None, :] -
                                               eval_value[None, ...],
                                               axis=-1)
        return distance


@dataclasses.dataclass
class NLE(PointShift):
    '''
    point distance directly
    e.g.
    gt_pts = {'head':np.array([[1, 2]]), 'eye':np.array([[2, 3]]), 'nose':np.array([])}
    eval_pts = {'head':np.array([[4, 6]]), 'eye':np.array([[6, 6]])}
    metric = PointDistance()
    shift = metric(gt_pts, eval_pts)
    shift: {'head': 5.0, 'eye': 5.0}
    '''
    @staticmethod
    def calculate(gt_pts: dict, eval_pts: dict) -> dict:
        # pts in gt_pts and eval_pts must be (Batch, n_dim)
        assert isinstance(gt_pts, dict) and isinstance(eval_pts, dict)
        assert all(
            isinstance(coord, np.ndarray) and coord.ndim == 2
            for coord in gt_pts.values())
        assert all(
            isinstance(coord, np.ndarray) and coord.ndim == 2
            for coord in eval_pts.values())
        interocular_keys = ('left_eye_lnmk_36', 'right_eye_lnmk_45')
        interocular = []
        all_keys = set(gt_pts.keys()).union(set(eval_pts.keys()))
        all_keys = sorted(all_keys)
        nle = dict()
        gt_lnmks, eval_lnmks = [], []
        for key in all_keys:
            gt_value = gt_pts.get(key, 'redundant')
            eval_value = eval_pts.get(key, 'missing')
            assert gt_value.shape[1:] == eval_value.shape[
                1:], f'got invalid shape: {gt_value.shape} and {eval_value.shape}. The shape after first dimension should be the same'
            if key in interocular_keys:
                interocular.append(gt_value)
            gt_lnmks.append(gt_value)
            eval_lnmks.append(eval_value)
        gt_lnmks = np.squeeze(np.stack(gt_lnmks), axis=-2)
        eval_lnmks = np.squeeze(np.stack(eval_lnmks), axis=-2)
        interocular = np.squeeze(np.stack(interocular), axis=-2)
        # frobenius norm
        interocular = np.linalg.norm(interocular[0] - interocular[1])
        localization_error = np.sum(
            np.linalg.norm(eval_lnmks - gt_lnmks, axis=0))

        nle.update({
            'nle': [float(localization_error / (interocular * len(all_keys)))],
            'interocular': [float(interocular)]
        })
        return nle