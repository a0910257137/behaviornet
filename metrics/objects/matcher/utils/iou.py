from typing import List

import numpy as np
from shapely.geometry import Polygon


def box2d_iou(gt_coords: List[dict], eval_coords: List[dict]) -> np.ndarray:
    """Calculate gt boxes and eval boxes iou with broadcasting. List x1,y1,x2,y2 as ordered

    Arguments:
        gt_coords {List[dict]} -- list of box2d coords. len: N
        eval_coords {List[dict]} -- list of box2d coords. len: M

    Returns:
        np.ndarray -- iou of each gt and eval box2d. Shape: [N,M]
    """
    # output axis0:gt axis1: eval
    # ref1: https://medium.com/@venuktan/vectorized-intersection-over-union-iou-in-numpy-and-tensor-flow-4fa16231b63d
    # ref2: https://chadrick-kwag.net/vectorized-calculatation-of-iou-and-removing-duplicate-boxes/
    if not gt_coords or not eval_coords:
        return np.empty(shape=(len(gt_coords), len(eval_coords)))

    order_2d = lambda box2d: [
        box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2']
    ]
    gt_coords, eval_coords = np.asarray([
        order_2d(coords) for coords in gt_coords
    ]), np.asarray([order_2d(coords) for coords in eval_coords])

    gt_x1s, gt_y1s, gt_x2s, gt_y2s = np.array_split(gt_coords, 4, axis=1)
    eval_x1s, eval_y1s, eval_x2s, eval_y2s = np.array_split(eval_coords,
                                                            4,
                                                            axis=1)
    # broadcasing
    max_x1s, max_y1s = np.maximum(gt_x1s,
                                  eval_x1s.T), np.maximum(gt_y1s, eval_y1s.T)
    min_x2s, min_y2s = np.minimum(gt_x2s,
                                  eval_x2s.T), np.minimum(gt_y2s, eval_y2s.T)

    intersections = np.maximum((min_x2s - max_x1s), 0) * np.maximum(
        (min_y2s - max_y1s), 0)
    gt_boxes_area = (gt_x2s - gt_x1s) * (gt_y2s - gt_y1s)
    eval_boxes_area = (eval_x2s - eval_x1s) * (eval_y2s - eval_y1s)
    unions = gt_boxes_area + eval_boxes_area.T - intersections
    return intersections / unions


def lshape_iou(gt_coords: List[dict], eval_coords: List[dict]) -> np.ndarray:
    """Calculating iou based on polygon

    Arguments:
        gt_coords {List[dict]} -- List of objs with specific keys
        eval_coords {List[dict]} -- List of objs with specific keys
        coords should contains {'x1': x1, 'x2': x2, 'y1':y1, 'y2':y2}, Optional: {'sideX': sideX, 'sideTop':sideTop, 'sideBotton': sideBotton}
    Returns:
        np.ndarray -- iou array, axis-0 represents gt, axis-1 represents eval
    """

    def _to_polygon(bbox):
        if "sideX" in bbox:
            if bbox["x2"] > bbox["sideX"]:
                return Polygon([(bbox["sideX"], bbox["sideTop"]),
                                (bbox["x1"], bbox["y1"]),
                                (bbox["x2"], bbox["y1"]),
                                (bbox["x2"], bbox["y2"]),
                                (bbox["x1"], bbox["y2"]),
                                (bbox["sideX"], bbox["sideBottom"])])
            else:
                return Polygon([(bbox["x1"], bbox["y1"]),
                                (bbox["x2"], bbox["y1"]),
                                (bbox["sideX"], bbox["sideTop"]),
                                (bbox["sideX"], bbox["sideBottom"]),
                                (bbox["x2"], bbox["y2"]),
                                (bbox["x1"], bbox["y2"])])
        else:
            return Polygon([(bbox["x1"], bbox["y1"]), (bbox["x2"], bbox["y1"]),
                            (bbox["x2"], bbox["y2"]), (bbox["x1"], bbox["y2"])])

    # {'x1': x1, 'x2': x2, 'y1':y1, 'y2':y2, 'sideX': sideX, 'sideTop':sideTop, 'sideBotton': sideBotton}
    assert isinstance(gt_coords, list) and isinstance(eval_coords, list)
    iou_array = np.empty(shape=(len(gt_coords), len(eval_coords)))
    for gt_idx, gt_coord in enumerate(gt_coords):
        for eval_idx, eval_coord in enumerate(eval_coords):
            gt_polygon = _to_polygon(gt_coord)
            eval_polygon = _to_polygon(eval_coord)
            intersection = gt_polygon.intersection(eval_polygon).area
            union = gt_polygon.area + eval_polygon.area - intersection
            iou_array[gt_idx, eval_idx] = intersection / union
    return iou_array
