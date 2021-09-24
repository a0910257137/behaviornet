import tensorflow as tf
import numpy as np
from pprint import pprint


class GFCBase:
    def __init__(self, config, **kwargs):
        self.config = config

    def get_grid_cells(self, featmap_size, grid_scale, stride):
        """
            Generate grid cells of a feature map for target assignment.
            :param featmap_size: Size of a single level feature map.
            :param scale: Grid cell scale.
            :param stride: Down sample stride of the feature map.
            :param dtype: Data type of the tensors.
            :param device: Device of the tensors.
            :return: Grid_cells xyxy position. Size should be [feat_w * feat_h, 4]
        """
        cell_size = stride * grid_scale
        y, x = self.get_single_level_center_point(featmap_size,
                                                  stride,
                                                  flatten=True)
        y = tf.expand_dims(tf.cast(y, tf.float32), axis=-1)
        x = tf.expand_dims(tf.cast(x, tf.float32), axis=-1)
        grid_cells = tf.concat(
            [
                x - 0.5 * cell_size,
                y - 0.5 * cell_size,
                x + 0.5 * cell_size,
                y + 0.5 * cell_size,
            ],
            axis=-1,
        )

        return grid_cells, grid_cells.get_shape().as_list()[0]

    def get_single_level_center_point(self,
                                      featmap_size,
                                      stride,
                                      flatten=True):
        """
        Generate pixel centers of a single stage feature map.
        :param featmap_size: height and width of the feature map
        :param stride: down sample stride of the feature map
        :param flatten: flatten the x and y tensors
        :return: y and x of the center points
        """
        h, w = featmap_size
        y_range = (np.arange(h) + 0.5) * stride
        x_range = (np.arange(w) + 0.5) * stride
        y, x = np.meshgrid(y_range, x_range)
        if flatten:
            y = np.reshape(y, [-1])
            x = np.reshape(x, [-1])
        return y, x
