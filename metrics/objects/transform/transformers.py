import numpy as np
from typing import Dict, List
from .base import Base


class Numpify:
    """Support transform Dict[list, tuple] -> Dict[np.ndarray] or list -> np.ndarray
    """
    def __call__(self, item):

        return np.asarray(item)


class MinimalBox2D:
    """Support transform Dict[Union[list, tuple]], List[list, tuple], np.ndarray -> box2d
    """
    def __call__(self, coords):
        # coords contain all coords and in shape: (N,2)
        top_left_coord, bottom_right_coord = np.min(coords,
                                                    axis=0), np.max(coords,
                                                                    axis=0)
        return dict(top_left=top_left_coord, bottom_right=bottom_right_coord)


class LandMarks(Base):
    """Support transform keypoints:{each of facial landmarks}[y1, x1] to
       serial landmark number {top_left, bottom_right }
    """
    def __call__(self, item):
        item = item['keypoints']
        return self.parse_lnmk(item)


class Box2DToKeyPoints:
    """Support transform {x1, x2, y1, y2} to {top_left, bottom_right }
    """
    def __call__(self, item):
        item = item['box2d']
        return dict(top_left=np.asarray((item['x1'], item['y1'])),
                    bottom_right=np.asarray((item['x2'], item['y2'])))


class Box2DToKeyPointsWithCenter:
    """Support transform {x1, x2, y1, y2} to {top_left, bottom_right }
    """
    def __call__(self, item: Dict[str, List[dict]]):
        """
        1. get user-defined target shape from each label
        2. transform each shape of label
        3. get cost-matrix

        Note:
        gt_lbs: List[lb]
        lb: {'frame_id': Any, 'object': Union[BDDObject, dict]}
        return metric
        """
        item = item['box2d']
        return dict(top_left=np.asarray((item['x1'], item['y1'])),
                    bottom_right=np.asarray((item['x2'], item['y2'])),
                    center_point=np.asarray(((item['x1'] + item['x2']) / 2,
                                             (item['y1'] + item['y2']) / 2)))


class LshapeToKeyPoints:
    """Support transform {x1, x2, y1, y2} to {top_left, bottom_right}
        'box3d': {
            'l_shape': {
                'facing': 'REAR', 'is_3d': False,
                'facing_box': 'x1': 1170, 'y1': 435, 'x2': 1217, 'y2': 482}
            }
        }
    """
    def __call__(self, item):
        return dict(top_left=(item['l_shape']['facing_box']['x1'],
                              item['l_shape']['facing_box']['y1']),
                    bottom_right=(item['l_shape']['facing_box']['x2'],
                                  item['l_shape']['facing_box']['y2']))
