# Modification 2020 RangiLyu
# Copyright 2018-2019 Open-MMLab.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from numpy import PINF
import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import gather, shape

from ...loss.iou_loss import bbox_overlaps
from .assign_result import AssignResult
from .base_assigner import BaseAssigner
import numpy as np


class ATSSAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        topk (float): number of bbox selected in each level
    """
    def __init__(self, topk):
        self.topk = topk

    # https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py

    def assign(self,
               bboxes,
               num_level_bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None,
               num_bbox=0):
        """Assign gt to bboxes.

        The assignment is done in following steps

        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as postive
        6. limit the positive sample's center in gt


        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            num_level_bboxes (List): num of bboxes in each level
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        INF = 1e8
        num_gt = num_bbox[0]
        num_bboxes = tf.shape(bboxes)[0]
        gt_bboxes = tf.reshape(gt_bboxes, [num_gt, 4])
        # compute iou between all bbox and gt
        # calculate iou
        overlaps = bbox_overlaps(bboxes, gt_bboxes)
        # assign 0 by default
        assigned_gt_inds = tf.zeros_like(overlaps[:, 0], dtype=tf.float32)
        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = tf.zeros_like(overlaps)
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is -1.:
                assigned_labels = -1.
            else:
                assigned_labels = tf.constant(-1., shape=(num_bboxes))
            return (num_gt, assigned_gt_inds, max_overlaps, assigned_labels)
            
        # compute center distance between all bbox and gt
        gt_cy = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cx = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        gt_cy = tf.reshape(gt_cy, [-1, 1])
        gt_cx = tf.reshape(gt_cx, [-1, 1])
        gt_points = tf.concat([gt_cy, gt_cx], axis=-1)
        bboxes_cy = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        bboxes_cx = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        bboxes_cy = tf.reshape(bboxes_cy, [-1, 1])
        bboxes_cx = tf.reshape(bboxes_cx, [-1, 1])
        bboxes_points = tf.concat((bboxes_cy, bboxes_cx), axis=-1)
        gt_points = gt_points[:, ::-1]
        bboxes_points = bboxes_points[:, ::-1]
        sqaure_distances = tf.math.square(
            (bboxes_points[:, None, :] - gt_points[None, :, :]))

        distances = tf.math.sqrt(tf.math.reduce_sum(sqaure_distances, axis=-1))

        # Selecting candidates based on the center distance
        candidate_idxs = []
        start_idx = 0
        for level, bboxes_per_level in enumerate(num_level_bboxes):

            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[start_idx:end_idx, :]

            selectable_k = min(self.topk, bboxes_per_level)
            distances_per_level += 1e4
            distances_per_level = -tf.transpose(distances_per_level)

            values, topk_idxs_per_level = tf.math.top_k(distances_per_level,
                                                        k=selectable_k,
                                                        sorted=True)

            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = tf.concat(candidate_idxs, axis=-1)
        # shape  is [9*3, N]
        candidate_idxs = tf.transpose(candidate_idxs)

        for_candidate_idxs = candidate_idxs

        n_idx = tf.range(num_gt, dtype=tf.int32)
        # pass to next step
        for_n_idx = n_idx
        n_idx = tf.tile(n_idx[tf.newaxis, :], [tf.shape(candidate_idxs)[0], 1])
        n_idx = tf.expand_dims(n_idx, axis=-1)
        candidate_idxs = tf.expand_dims(candidate_idxs, axis=-1)
        candidate_idxs = tf.concat([candidate_idxs, n_idx], axis=-1)
        # gen idx for 0~ n gt bboxes
        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        candidate_overlaps = tf.gather_nd(overlaps, candidate_idxs)

        overlaps_mean_per_gt = tf.math.reduce_mean(candidate_overlaps, axis=0)

        overlaps_std_per_gt = tf.math.reduce_std(candidate_overlaps, axis=0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        is_pos = candidate_overlaps >= overlaps_thr_per_gt[tf.newaxis, :]

        # limit the positive sample's center in gt
        # Latter check
        for_candidate_idxs += for_n_idx * num_bboxes
        num_bboxes = tf.cast(num_bboxes, tf.float32)
        #----------------reshape  and make eps bboxes for two coordinations----------------
        ep_bboxes_cy = tf.reshape(bboxes_cy, [1, -1])
        ep_bboxes_cy = tf.tile(ep_bboxes_cy, [num_gt, num_bboxes])
        ep_bboxes_cy = tf.reshape(ep_bboxes_cy, [-1])

        ep_bboxes_cx = tf.reshape(bboxes_cx, [1, -1])
        ep_bboxes_cx = tf.tile(ep_bboxes_cx, [num_gt, num_bboxes])
        ep_bboxes_cx = tf.reshape(ep_bboxes_cx, [-1])
        #----------------reshape  and make eps bboxes for two coordinations----------------
        candidate_idxs = tf.reshape(for_candidate_idxs, [-1])
        # calculate the left, top, right, bottom distance between positive
        # bbox center and gt side
        l_ = tf.reshape(tf.gather(ep_bboxes_cx, candidate_idxs),
                        [-1, num_gt]) - gt_bboxes[:, 1]  # x1
        t_ = tf.reshape(tf.gather(ep_bboxes_cy, candidate_idxs),
                        [-1, num_gt]) - gt_bboxes[:, 0]  # y1
        r_ = gt_bboxes[:, 3] - tf.reshape(
            tf.gather(ep_bboxes_cx, candidate_idxs), (-1, num_gt))  #x2
        b_ = gt_bboxes[:, 2] - tf.reshape(
            tf.gather(ep_bboxes_cy, candidate_idxs), (-1, num_gt))  #y2
        is_in_gts = tf.concat(
            [t_[:, None, :], l_[:, None, :], b_[:, None, :], r_[:, None, :]],
            axis=-2)
        is_in_gts = tf.math.reduce_min(is_in_gts, axis=-2) > .01
        # is_in_gts_gt = np.load('../nanodet/is_in_gts.npy')
        # vals = np.isclose(is_in_gts_gt, is_in_gts)
        # vals = np.where(vals == False)
        # vals = np.asarray(vals)
        # if vals.shape[-1] != 0:
        #     print(vals)
        #     print('-' * 100)
        is_pos = is_pos & is_in_gts
        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf = tf.transpose(tf.ones_like(overlaps) * (-INF))

        overlaps_inf = tf.reshape(overlaps_inf, [-1])

        index = candidate_idxs[tf.reshape(is_pos, [-1])]

        ov_vals = tf.gather(tf.reshape(tf.transpose(overlaps), [-1]), index)

        overlaps_inf = tf.tensor_scatter_nd_update(overlaps_inf,
                                                   index[:,
                                                         tf.newaxis], ov_vals)

        overlaps_inf = tf.transpose(tf.reshape(overlaps_inf, [num_gt, -1]))
        max_overlaps = tf.math.reduce_max(overlaps_inf, axis=-1)
        argmax_overlaps = tf.math.argmax(overlaps_inf, axis=-1)

        valid_mask = (max_overlaps != -INF)
        need_assigned = (argmax_overlaps[valid_mask] + 1)
        valid_inds = tf.where(valid_mask == True)
        need_assigned = tf.cast(need_assigned, tf.float32)
        assigned_gt_inds = tf.tensor_scatter_nd_update(assigned_gt_inds,
                                                       valid_inds,
                                                       need_assigned)
        if tf.math.reduce_all(gt_labels != -1.) or gt_labels != -1.:
            assigned_labels = -tf.ones_like(assigned_gt_inds)
            pos_inds = tf.where(assigned_gt_inds > 0)

            num_elements = tf.shape(tf.reshape(pos_inds, [-1]))[0]
            if num_elements > 0:
                gath_vals_as_idx = tf.gather(assigned_gt_inds, pos_inds) - 1.
                gath_vals_as_idx = tf.cast(gath_vals_as_idx, tf.int32)
                picked_gt_lb = tf.gather(gt_labels, gath_vals_as_idx)
                picked_gt_lb = tf.squeeze(tf.cast(picked_gt_lb, tf.float32),
                                          axis=-1)
                assigned_labels = tf.tensor_scatter_nd_update(
                    assigned_labels[:, None], pos_inds, picked_gt_lb)
                assigned_labels = tf.cast(assigned_labels, tf.int32)
                assigned_labels = tf.squeeze(assigned_labels, axis=-1)
        else:
            assigned_labels = None

        return AssignResult(num_gt,
                            assigned_gt_inds,
                            max_overlaps,
                            labels=assigned_labels)