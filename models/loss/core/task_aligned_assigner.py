import numpy as np
from .task_aligned_assign_result import TaskAlignedAssignResult
from .iou2d_calculator import BboxOverlaps2D
from .transform import distance2bbox


class TaskAlignedAssigner:

    def __init__(self,
                 topk,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 num_level_anchors=[3200, 800, 200],
                 anchor_generator_strides=[(8, 8), (16, 16), (32, 32)],
                 ignore_iof_thr=-1):
        self.topk = topk
        self.iou_calculator = BboxOverlaps2D()
        self.num_level_anchors = num_level_anchors
        self.anchor_generator_strides = anchor_generator_strides
        self.ignore_iof_thr = ignore_iof_thr

    def assign(self,
               scores,
               encoded_bboxes,
               anchor_centers,
               num_level_bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None,
               alpha=1,
               beta=6):
        """Assign gt to bboxes.

        The assignment is done in following steps

        1. compute alignment metric between all bbox (bbox of all pyramid levels) and gt
        2. select top-k bbox as candidates for each gt
        3. limit the positive sample's center in gt (because the anchor-free detector only can predict positive distance)


        Args:
            scores (Tensor): predicted class probability, shape(n, 80)
            decode_bboxes (Tensor): predicted bounding boxes, shape(n, 4)
            anchors (Tensor): pre-defined anchors, shape(n, 4).
            num_level_bboxes (List): num of bboxes in each level
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`TaskAlignedAssignResult`: The assign result.
        """
        INF = 1e8
        num_gt, num_bboxes = np.shape(gt_bboxes)[0], np.shape(anchor_centers)[0]
        start = 0
        tmp = []
        for n, stride in zip(self.num_level_anchors,
                             self.anchor_generator_strides):
            end = start + n
            decode_bboxes = distance2bbox(
                anchor_centers[start:end, :],
                encoded_bboxes[start:end, :] * stride[0])
            tmp.append(decode_bboxes)
        decode_bboxes = np.concatenate(tmp, axis=0)
        # compute alignment metric between all bbox and gt
        overlaps = self.iou_calculator(decode_bboxes, gt_bboxes)
        gt_labels = gt_labels.astype(np.int32)
        bbox_scores = scores[:, gt_labels]
        bbox_scores = 1 / (1 + np.exp(-bbox_scores))
        # mask = bbox_scores > 0.5
        # mask_bbox_scores = bbox_scores[mask]
        # mask_overlaps = overlaps[mask]
        alignment_metrics = bbox_scores**alpha * overlaps**beta
        # mask = alignment_metrics > 0.1
        # print(alignment_metrics[mask])
        # xxx
        # assign 0 by default
        assigned_gt_inds = np.full(shape=(num_bboxes, ),
                                   fill_value=0,
                                   dtype=np.int32)

        assign_metrics = np.zeros(shape=(num_bboxes, ))
        # assign_metrics = alignment_metrics.new_zeros((num_bboxes, ))

        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = np.zeros(shape=(num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_gt_inds = np.full(shape=(num_bboxes, ),
                                           fill_value=-1,
                                           dtype=np.int32)
            return TaskAlignedAssignResult(num_gt,
                                           assigned_gt_inds,
                                           max_overlaps,
                                           assign_metrics,
                                           labels=assigned_labels)

        # select top-k bbox as candidates for each gt

        topk_alignment_metrics = np.argsort(-alignment_metrics, axis=0)
        candidate_idxs = topk_alignment_metrics[:self.topk]
        candidate_metrics = alignment_metrics[candidate_idxs, np.arange(num_gt)]

        is_pos = candidate_metrics > 0

        # limit the positive sample's center in gt
        anchors_cx, anchors_cy = anchor_centers[:, 0], anchor_centers[:, 1]

        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes

        ep_anchors_cx = np.tile(np.reshape(anchors_cx, (1, -1)),
                                (num_gt, 1)).reshape([-1])
        ep_anchors_cy = np.tile(np.reshape(anchors_cy, (1, -1)),
                                (num_gt, 1)).reshape([-1])
        candidate_idxs = candidate_idxs.reshape([-1])

        # calculate the left, top, right, bottom distance between positive
        # bbox center and gt side

        l_ = ep_anchors_cx[candidate_idxs].reshape([-1, num_gt]) - gt_bboxes[:,
                                                                             0]
        t_ = ep_anchors_cy[candidate_idxs].reshape([-1, num_gt]) - gt_bboxes[:,
                                                                             1]
        r_ = gt_bboxes[:, 2] - ep_anchors_cx[candidate_idxs].reshape(
            [-1, num_gt])
        b_ = gt_bboxes[:, 3] - ep_anchors_cy[candidate_idxs].reshape(
            [-1, num_gt])
        is_in_gts = np.stack([l_, t_, r_, b_], axis=1).min(axis=1)[0] > 0.01
        is_pos = is_pos & is_in_gts
        # if an anchor box is assigned to multiple gts,
        # the one with the highest iou will be selected.
        # assigned_gt_inds = np.full(shape=(num_bboxes, ),
        #                            fill_value=0,
        #                            dtype=np.int32)
        overlaps_inf = np.full_like(overlaps, -INF).T.reshape([-1])
        index = candidate_idxs.reshape([-1])[is_pos.reshape([-1])]
        overlaps_inf[index] = overlaps.T.reshape([-1])[index]
        overlaps_inf = overlaps_inf.reshape([num_gt, -1]).T

        max_overlaps = np.max(overlaps_inf, axis=1)
        argmax_overlaps = np.argmax(overlaps_inf, axis=1)
        # max_overlaps, argmax_overlaps = overlaps_inf.max(axis=1)
        assigned_gt_inds[
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1
        assign_metrics[max_overlaps != -INF] = alignment_metrics[
            max_overlaps != -INF, argmax_overlaps[max_overlaps != -INF]]

        if gt_labels is not None:

            assigned_labels = np.full(shape=(num_bboxes, ),
                                      fill_value=-1,
                                      dtype=np.int32)

            pos_inds = np.nonzero(assigned_gt_inds > 0)[0]
            if np.size(pos_inds) > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds]
                                                      - 1]
        else:
            assigned_labels = None
        return TaskAlignedAssignResult(num_gt,
                                       assigned_gt_inds,
                                       max_overlaps,
                                       assign_metrics,
                                       labels=assigned_labels)
