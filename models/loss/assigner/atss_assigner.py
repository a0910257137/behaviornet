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

import tensorflow as tf

from ...loss.iou_loss import bbox_overlaps
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


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
        INF = 100000000
        # bboxes = bboxes[:, :4]
        num_gt = num_bbox[0]
        num_bboxes = tf.shape(bboxes)[0]

        gt_valids = tf.math.reduce_all(tf.math.is_finite(gt_bboxes), axis=-1)
        gt_bboxes = tf.reshape(gt_bboxes[gt_valids], [-1, 4])

        # gt_bboxes = tf.reshape(gt_bboxes, [num_gt, 4])

        # compute iou between all bbox and gt
        overlaps = bbox_overlaps(bboxes, gt_bboxes)

        # assign 0 by default
        assigned_gt_inds = tf.zeros_like(overlaps[:, 0], dtype=tf.float32)

        #TODO: latter might be trouble-shot
        # if num_gt == 0 or num_bboxes == 0:
        #     # No ground truth or boxes, return empty assignment
        #     if num_gt == 0:
        #         # No truth, assign everything to background
        #         assigned_gt_inds[:] = 0
        #     if gt_labels is None:
        #         assigned_labels = None
        #     else:
        #         assigned_labels = tf.constant(-1., shape=(num_bboxes))
        #     return AssignResult(num_gt,
        #                         assigned_gt_inds,
        #                         assigned_gt_inds,
        #                         labels=assigned_labels)

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

        sqaure_distances = tf.math.square(
            (bboxes_points[:, None, :] - gt_points[None, :, :]))

        distances = tf.math.sqrt(tf.math.reduce_sum(sqaure_distances, axis=-1))
        # distances = ((bboxes_points[:, None, :] -
        #               gt_points[None, :, :]).pow(2).sum(-1).sqrt())
        # Selecting candidates based on the center distance
        candidate_idxs = []
        start_idx = 0

        # lv_shapes = num_level_bboxes.get_shape().as_list()
        for level, bboxes_per_level in enumerate(num_level_bboxes):

            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[start_idx:end_idx, :]
            selectable_k = min(self.topk, bboxes_per_level)
            distances_per_level = -tf.transpose(distances_per_level)
            values, topk_idxs_per_level = tf.math.top_k(distances_per_level,
                                                        k=selectable_k,
                                                        sorted=True)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx

        candidate_idxs = tf.concat(candidate_idxs, axis=-1)
        candidate_idxs = tf.transpose(candidate_idxs)
        print(candidate_idxs)
        print(overlaps)

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        candidate_overlaps = overlaps[candidate_idxs,
                                      tf.range(num_gt, dtype=tf.int32)]

        overlaps_mean_per_gt = candidate_overlaps.mean(0)
        overlaps_std_per_gt = candidate_overlaps.std(0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]

        # limit the positive sample's center in gt
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        ep_bboxes_cx = (bboxes_cx.view(1, -1).expand(
            num_gt, num_bboxes).contiguous().view(-1))
        ep_bboxes_cy = (bboxes_cy.view(1, -1).expand(
            num_gt, num_bboxes).contiguous().view(-1))
        candidate_idxs = candidate_idxs.view(-1)

        # calculate the left, top, right, bottom distance between positive
        # bbox center and gt side
        l_ = ep_bboxes_cx[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 0]
        t_ = ep_bboxes_cy[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - ep_bboxes_cx[candidate_idxs].view(-1, num_gt)
        b_ = gt_bboxes[:, 3] - ep_bboxes_cy[candidate_idxs].view(-1, num_gt)
        is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01
        is_pos = is_pos & is_in_gts

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf = torch.full_like(overlaps,
                                       -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()

        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[max_overlaps != -INF] = (
            argmax_overlaps[max_overlaps != -INF] + 1)

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(assigned_gt_inds > 0,
                                     as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        return AssignResult(num_gt,
                            assigned_gt_inds,
                            max_overlaps,
                            labels=assigned_labels)
