import dataclasses
from itertools import product
from typing import ClassVar, List, Union
import numpy as np
from ....utils.typing import Poly2D
from ..base import Metric


class HausdorffDistance(Metric):
    '''
    The original Hausdorff distance is defined by the following definition:
        HD(A,B) := max(p(A,B), p(B,A)),
    where p(A,B) := max{ min{d(a,b): b in B}: a in A}.
    This means if we expand the point set A by d pixels where d = HD(A,B), then it will cover B, and vice versa.
    Here in the implementation, we use a weak formation by defining a COVER_PERCENTAGE (1 ~ 100):
    If we expand the lane line A by d pixels, then it will cover COVER_PERCENTAGE% of B, and vice versa.
    '''
    COVER_PERCENTAGE: ClassVar[int] = 95

    def __call__(self, gt_pts: Union[List[Poly2D], Poly2D],
                 eval_pts: Union[List[Poly2D], Poly2D]) -> dict:
        """deal with single point to point metric and multiple points to points metric
        """
        # gt_pts and eval_pts are supposed to be list of Poly2D,
        # where each Poly2D is a list of components whose 'vertices' is [[x1, y1], ...] or 2d np.ndarray.
        if isinstance(gt_pts[0], dict): gt_pts = [gt_pts]
        if isinstance(eval_pts[0], dict): eval_pts = [eval_pts]
        assert self.check_lane_line_shape(
            gt_pts) and self.check_lane_line_shape(eval_pts)

        result = [
            self.calculate(single_gt_lane, single_eval_lane)
            for single_gt_lane, single_eval_lane in product(gt_pts, eval_pts)
        ]
        return np.asarray(result).reshape(len(gt_pts), len(eval_pts))

    @staticmethod
    def check_lane_line_shape(lanes: List[Poly2D]):
        assert isinstance(lanes, list)
        for single_lane in lanes:
            assert isinstance(single_lane, list)
            for comp in single_lane:
                assert isinstance(comp,
                                  dict), f'Expected dict, got {type(comp)}'
                # after transforming, vertices in components are np.ndarray's.
                assert 'vertices' in comp
                coord = comp['vertices']
                assert isinstance(coord, np.ndarray)
                assert coord.ndim == 2 and coord.shape[
                    1] == 2, f'Expected dim == 2 and shape: N * 2, got dim: {coord.ndim}, shape: {coord.shape}'
                assert all(
                    isinstance(p, (np.integer, np.float32))
                    for p in coord.ravel()
                ), f'coords must be int or float, got {[type(p) for p in coord.ravel()]}'
        return True

    @staticmethod
    def ptwise_diff(refer_curve: np.ndarray,
                    target_curve: np.ndarray) -> np.ndarray:
        """[summary]
            get pointwise differences in each coordinates between refer-curve and target-curve
        Args:
            refer_curve (np.ndarray):  an (n,2) numpy array
            target_curve (np.ndarray): an (m,2) numpy array
        Returns:
            np.ndarray: an (n,m,2) numpy array
        """
        return (target_curve[np.newaxis, :, :] - refer_curve[:, np.newaxis, :])

    @staticmethod
    def ptwise_squared_dist(refer_curve: np.ndarray,
                            target_curve: np.ndarray) -> np.ndarray:
        return np.sum((HausdorffDistance.ptwise_diff(refer_curve,
                                                     target_curve))**2,
                      axis=2)

    @staticmethod
    def get_idx_of_most_closed_pt(curve: np.ndarray,
                                  point_set: np.ndarray) -> np.ndarray:
        """ At each pt of [curve], obtain the index of the most closed pt in [point_set].
        Args:
            curve (np.ndarray):     an (n,2) numpy array
            point_set (np.ndarray): an (m,2) numpy array
        Returns:
            np.ndarray: an (n) numpy vector
        """
        ptwise_diff = HausdorffDistance.ptwise_diff(curve, point_set)
        inds = np.argmin(np.sum(ptwise_diff**2, axis=2), axis=1)
        return inds

    @staticmethod
    def get_coord_shift_to_match_target(refer_curve: np.ndarray,
                                        target_curve: np.ndarray) -> np.ndarray:
        """ At each pt of [curve], obtain the shift in each coordinate to touch the target_curve
        Args:
            refer_curve (np.ndarray):  an (n,2) numpy array
            target_curve (np.ndarray): an (m,2) numpy array

        Returns:
            np.ndarray: an (n,2) numpy array
        """
        ptwise_diff = HausdorffDistance.ptwise_diff(refer_curve, target_curve)
        inds = np.argmin(np.sum(ptwise_diff**2, axis=2), axis=1)
        return ptwise_diff[range(len(refer_curve)), inds, :]

    @staticmethod
    def dist_to_cover_target_curve_in_percentage(
            refer_curve: Union[np.ndarray, List],
            target_curve: Union[np.ndarray, List],
            cover_percentage=95) -> np.float32:
        """ calculate the distance for refer-curve to cover COVER-PERCENTAGE of points of target-curve.

        Args:
            refer_curve (Union[np.ndarray, List]):  an (n,2) numpy array or length-n list of [x,y] points.
            target_curve (Union[np.ndarray, List]): an (m,2) numpy array or length-n list of [x,y] points.
            cover_percentage (int, optional): set cover percentage. Defaults to 95.

        Returns:
            np.float: the distance to cover COVER-PERCENTAGE of pts of target_curves.
        """
        refer_curve = np.asarray(refer_curve)
        target_curve = np.asarray(target_curve)
        assert refer_curve.ndim == 2 and target_curve.ndim == 2

        # the n-by-m matrix [p_ij] means the square of distance from i to j
        ptwise_squared_dist = HausdorffDistance.ptwise_squared_dist(
            refer_curve, target_curve)

        # take minimum along axis 0, got a vector of length m:
        # target_curve: q1, ..., qm -> each component is d(q_i, refer_curve)
        dist_to_cover_target_curve = np.sqrt(np.min(ptwise_squared_dist,
                                                    axis=0))

        return np.percentile(dist_to_cover_target_curve, cover_percentage)

    @staticmethod
    def dist_to_cover_each_other_lane_in_percentage(
            refer_curve: Union[np.ndarray, List],
            target_curve: Union[np.ndarray, List],
            cover_percentage=95) -> np.float32:
        """ This is the same as the above function 'dist_to_cover_target_curve_in_percentage'.
            But just calculate the case [refer to cover target] and [target to cover refer] in one time.
        Args:
            refer_curve (Union[np.ndarray, List]):  an (n,2) numpy array or length-n list of [x,y] points.
            target_curve (Union[np.ndarray, List]): an (m,2) numpy array or length-n list of [x,y] points.
            cover_percentage (int, optional): set cover percentage. Defaults to 95.

        Returns:
            np.float: the distance to cover COVER-PERCENTAGE of pts of target_curves.
        """
        refer_curve = np.asarray(refer_curve)
        target_curve = np.asarray(target_curve)
        assert refer_curve.ndim == 2 and target_curve.ndim == 2

        # the n-by-m matrix [p_ij] means the square of distance from i to j
        ptwise_dist = np.sqrt(
            HausdorffDistance.ptwise_squared_dist(refer_curve, target_curve))

        # take minimum along axis 0, got a vector of length m:
        # target_curve: q1, ..., qm -> each component is d(q_i, refer_curve)
        # take minimum along axis 1, got a vector of length n:
        # refer_curve: p1, ..., pn -> each component is d(p_j, target_curve)

        # Then calculate the 95 percentile and return the maximal value.
        dist_to_cover_target_curve = [
            np.percentile(np.min(ptwise_dist, axis=0), cover_percentage),
            np.percentile(np.min(ptwise_dist, axis=1), cover_percentage)
        ]
        return np.max(dist_to_cover_target_curve)

    @staticmethod
    def calculate(single_gt_lane: Poly2D, single_eval_lane: Poly2D) -> dict:
        # merge all components of single_gt_lane, also do the samething for single_eval_lane.
        gt_lane_pts = np.concatenate(
            [gt_comp['vertices'] for gt_comp in single_gt_lane], axis=0)
        eval_lane_pts = np.concatenate(
            [eval_comp['vertices'] for eval_comp in single_eval_lane], axis=0)

        assert gt_lane_pts.ndim == 2, gt_lane_pts.shape[1] == 2
        assert eval_lane_pts.ndim == 2, eval_lane_pts.shape[1] == 2

        hd_dist = HausdorffDistance.dist_to_cover_each_other_lane_in_percentage(
            gt_lane_pts, eval_lane_pts)

        return hd_dist
