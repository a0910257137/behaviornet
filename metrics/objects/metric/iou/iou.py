from typing import List, Union

import numpy as np

from ..base import Metric


class IoU(Metric):
    """Base IOU Metric for box2d or polygon
    """
    def __call__(self, gt_boxes: Union[List[dict], dict],
                 eval_boxes: Union[List[dict], dict]
                 ) -> Union[float, np.ndarray]:
        """entry function for calculating iou

        Args:
            gt_boxes (Union[List[dict], dict]): one box or multiple boxes with "bottom_right" and "top_left" key
            eval_boxes (Union[List[dict], dict]): one box or multiple boxes with "bottom_right" and "top_left" key

        Returns:
            Union[float, np.ndarray]: iou
        """
        single_item_flag = True if isinstance(gt_boxes, dict) and isinstance(
            eval_boxes, dict) else False

        if isinstance(gt_boxes, dict):
            gt_boxes = [gt_boxes]
        if isinstance(eval_boxes, dict):
            eval_boxes = [eval_boxes]

        assert gt_boxes and isinstance(
            gt_boxes, list), 'boxes should be list and not empty'
        assert eval_boxes and isinstance(
            eval_boxes, list), 'boxes should be list and not empty'
        assert self.check_shape(gt_boxes) and self.check_shape(eval_boxes)

        gt_boxes = np.asarray([self.parse_shape(box) for box in gt_boxes])
        eval_boxes = np.asarray([self.parse_shape(box) for box in eval_boxes])

        iou = self.calculate(gt_boxes, eval_boxes)
        return iou[0][0].item() if single_item_flag else iou

    @staticmethod
    def parse_shape(box):
        raise NotImplementedError

    @staticmethod
    def calculate(gt_boxes, eval_boxes):
        raise NotImplementedError

    @staticmethod
    def check_shape(boxes):
        raise NotImplementedError


class IoU2D(IoU):
    """Box2D intersection of union

    box1 = {'top_left': np.array([3, 3]), 'bottom_right': np.array([10, 10])}
    box2 = {'top_left': np.array([5, 5]), 'bottom_right': np.array([10, 10])}
    metric = IoU2D()
    iou = metric(box1, box2)
    """
    @staticmethod
    def check_shape(boxes: List[dict]):
        """check if list of boxes matches format or not. 
        
        Each box needs to be:
        
        example:
        
        box = {'top_left': np.array([5,5]),'bottom_right': np.array([10,10])}
        
        1. box should contain 'top_left' and bottom_right point and 'bottom_right' coord larger than 'top_left'.
        2. each coord should be: np.ndarray; type: int or float; size: 2 
        """
        for box in boxes:
            assert set(['bottom_right', 'top_left']) == set(
                box.keys()
            ), f'For each box2d should contains "top_left" and "bottom_right" keys. Got {set(box.keys())}'
            for coord in box.values():
                assert isinstance(
                    coord, np.ndarray
                ), f'for each coord, expected np.ndarray. Got {type(coord)}'
                assert coord.size == 2, f'for each coord, expected coord with in size 2, got {coord.size}'
                assert coord.dtype in [
                    np.float, np.int
                ], f'coords must be int or float, got {coord.dtype}'
            assert np.all(
                box['bottom_right'].flatten() > box['top_left'].flatten()
            ), f'bottom_right:{box["bottom_right"].flatten()} coord should larger than top left coord {box["top_left"].flatten()}'
        return True

    @staticmethod
    def parse_shape(box):
        return [*box['top_left'].flatten(), *box['bottom_right'].flatten()]

    @staticmethod
    def calculate(gt_coords: np.ndarray,
                  eval_coords: np.ndarray) -> np.ndarray:
        """calculate box2d iou
        output axis0:gt axis1: eval
        ref1: https://medium.com/@venuktan/vectorized-intersection-over-union-iou-in-numpy-and-tensor-flow-4fa16231b63d
        ref2: https://chadrick-kwag.net/vectorized-calculatation-of-iou-and-removing-duplicate-boxes/

        Args:
            gt_boxes (np.ndarray): gt coords numpy array (N,4) in 'xyxy' format
            eval_boxes (np.ndarray): eval coords numpy array (M,4) in 'xyxy' format

        Returns:
            np.ndarray: iou
        """
        assert isinstance(gt_coords, np.ndarray) and gt_coords.ndim == 2
        assert isinstance(eval_coords, np.ndarray) and eval_coords.ndim == 2

        gt_coords[:, :2] -= 1
        eval_coords[:, :2] -= 1

        gt_x1s, gt_y1s, gt_x2s, gt_y2s = np.array_split(gt_coords, 4, axis=1)
        eval_x1s, eval_y1s, eval_x2s, eval_y2s = np.array_split(eval_coords,
                                                                4,
                                                                axis=1)
        # broadcasing
        max_x1s, max_y1s = np.maximum(gt_x1s, eval_x1s.T), np.maximum(
            gt_y1s, eval_y1s.T)
        min_x2s, min_y2s = np.minimum(gt_x2s, eval_x2s.T), np.minimum(
            gt_y2s, eval_y2s.T)
        intersections = np.maximum((min_x2s - max_x1s), 0) * np.maximum(
            (min_y2s - max_y1s), 0)
        gt_boxes_area = (gt_x2s - gt_x1s) * (gt_y2s - gt_y1s)
        eval_boxes_area = (eval_x2s - eval_x1s) * (eval_y2s - eval_y1s)
        unions = gt_boxes_area + eval_boxes_area.T - intersections

        return intersections / unions


box1 = {'top_left': np.array([3, 3]), 'bottom_right': np.array([10, 10])}
box2 = {'top_left': np.array([5, 5]), 'bottom_right': np.array([10, 10])}
