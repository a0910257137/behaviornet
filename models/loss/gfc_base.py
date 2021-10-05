import tensorflow as tf
import numpy as np
from pprint import pprint


class GFCBase:
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
                y - 0.5 * cell_size,
                x - 0.5 * cell_size,
                y + 0.5 * cell_size,
                x + 0.5 * cell_size,
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
        # 40 , 32
        h, w = featmap_size
        y_range = (np.arange(h) + 0.5) * stride
        x_range = (np.arange(w) + 0.5) * stride
        y, x = tf.meshgrid(y_range, x_range)
        y = tf.transpose(y)
        x = tf.transpose(x)
        if flatten:
            y = np.reshape(y, [-1])
            x = np.reshape(x, [-1])
        return y, x

    def images_to_levels(self, target, num_level_anchors):
        """Convert targets by image to targets by feature level.

        [target_img0, target_img1] -> [target_level0, target_level1, ...]
        """
        target = tf.stack(target, axis=0)
        level_targets = []
        start = 0
        for n in num_level_anchors:
            end = start + n
            assigned_vals = target[:, start:end]
            level_targets.append(assigned_vals)
            start = end
        return level_targets

    def sample(self, assign_result, gt_bboxes):
        poses = tf.squeeze(tf.where(assign_result.gt_inds > 0), axis=-1)
        pos_inds, idx = tf.unique(poses)
        negs = tf.squeeze(tf.where(assign_result.gt_inds == 0), axis=-1)
        neg_inds, idx = tf.unique(negs)
        pos_assigned_gt_inds = tf.gather(assign_result.gt_inds, pos_inds) - 1
        # we need to clean all of the zero ground truth bboxes
        gt_bboxes = tf.reshape(gt_bboxes, (-1, 4))

        pos_assigned_gt_inds = tf.cast(pos_assigned_gt_inds, tf.int32)
        pos_gt_bboxes = tf.gather_nd(gt_bboxes, pos_assigned_gt_inds[:, None])
        return pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds

    def grid_cells_to_center(self, grid_cells):
        """
        Get center location of each gird cell
        :param grid_cells: grid cells of a feature map
        :return: center points
        """
        # why inverse grid_cells !?
        cells_cx = (grid_cells[:, 2] + grid_cells[:, 0]) / 2
        cells_cy = (grid_cells[:, 3] + grid_cells[:, 1]) / 2

        return tf.concat([cells_cy[:, None], cells_cx[:, None]], axis=-1)

    def integral_distribution(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.
        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.
        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        x = tf.nn.softmax(tf.reshape(x, [-1, self.reg_max + 1]), axis=-1)
        ln = tf.range(self.reg_max + 1, dtype=tf.float32)
        x = tf.linalg.matmul(x, ln[:, None])
        x = tf.reshape(x, [-1, 4])
        return x

    def distance2bbox(self, points, distance, max_shape=None):
        """Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (n, 2), [x, y].
            distance (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom).
            max_shape (tuple): Shape of the image.

        Returns:
            Tensor: Decoded bboxes.
        """
        #TODO: change the h and w clip vals
        y1 = points[:, 0] - distance[:, 0]
        x1 = points[:, 1] - distance[:, 1]
        y2 = points[:, 0] + distance[:, 2]
        x2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            # max_shape = h, w
            y1 = tf.clip_by_value(y1,
                                  clip_value_min=0.,
                                  clip_value_max=max_shape[0])
            x1 = tf.clip_by_value(x1,
                                  clip_value_min=0.,
                                  clip_value_max=max_shape[1])
            y2 = tf.clip_by_value(y2,
                                  clip_value_min=0.,
                                  clip_value_max=max_shape[0])

            x2 = tf.clip_by_value(x2,
                                  clip_value_min=0.,
                                  clip_value_max=max_shape[1])
        return tf.concat([y1[:, None], x1[:, None], y2[:, None], x2[:, None]],
                         axis=-1)

    def bbox2distance(self, points, bbox, max_dis=None, eps=0.1):
        """Decode bounding box based on distances.

        Args:
            points (Tensor): Shape (n, 2), [x, y].
            bbox (Tensor): Shape (n, 4), "xyxy" format
            max_dis (float): Upper bound of the distance.
            eps (float): a small value to ensure target < max_dis, instead <=

        Returns:
            Tensor: Decoded distances.
        """

        top = points[:, 0] - bbox[:, 0]
        left = points[:, 1] - bbox[:, 1]
        bottom = bbox[:, 2] - points[:, 0]
        right = bbox[:, 3] - points[:, 1]
        if max_dis is not None:
            top = tf.clip_by_value(top,
                                   clip_value_min=0.,
                                   clip_value_max=max_dis - eps)
            left = tf.clip_by_value(left,
                                    clip_value_min=0.,
                                    clip_value_max=max_dis - eps)

            bottom = tf.clip_by_value(bottom,
                                      clip_value_min=0.,
                                      clip_value_max=max_dis - eps)
            right = tf.clip_by_value(right,
                                     clip_value_min=0.,
                                     clip_value_max=max_dis - eps)

        return tf.concat(
            [top[:, None], left[:, None], bottom[:, None], right[:, None]],
            axis=-1)