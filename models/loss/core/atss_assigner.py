import numpy as np
from .assign_result import AssignResult
from .iou2d_calculator import BboxOverlaps2D


class ATSSAssigner:
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        topk (float): number of bbox selected in each level
    """

    def __init__(self,
                 topk,
                 mode=0,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 ignore_iof_thr=-1):
        self.topk = topk
        self.mode = mode
        self.iou_calculator = BboxOverlaps2D()
        self.ignore_iof_thr = ignore_iof_thr

    def assign(self,
               bboxes,
               num_level_bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None):
        INF = 100000000
        bboxes = bboxes[:, :4]
        num_gt, num_bboxes = gt_bboxes.shape[0], bboxes.shape[0]
        #print('AT1:', num_gt, num_bboxes)
        # compute iou between all bbox and gt
        overlaps = self.iou_calculator(bboxes, gt_bboxes)
        # mask = overlaps > 0.5

        assigned_gt_inds = np.full(shape=(num_bboxes, ),
                                   fill_value=0,
                                   dtype=np.int32)
        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = np.zeros(shape=(num_bboxes, ))
            # max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = np.full(shape=(num_bboxes, ),
                                          fill_value=-1,
                                          dtype=np.int32)
            return AssignResult(num_gt,
                                assigned_gt_inds,
                                max_overlaps,
                                labels=assigned_labels)

        # assign 0 by default
        # compute center distance between all bbox and gt
        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        gt_points = np.stack((gt_cx, gt_cy), axis=-1)
        gt_width = gt_bboxes[:, 2] - gt_bboxes[:, 0]
        gt_height = gt_bboxes[:, 3] - gt_bboxes[:, 1]
        gt_area = np.sqrt(np.clip(gt_width * gt_height, a_min=1e-4, a_max=INF))
        bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        bboxes_points = np.stack((bboxes_cx, bboxes_cy), axis=-1)
        distances = np.sqrt(
            np.sum(np.square(bboxes_points[:, None, :] - gt_points[None, :, :]),
                   axis=-1))

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and np.any(gt_bboxes_ignore == True)):
            ignore_overlaps = self.iou_calculator(bboxes,
                                                  gt_bboxes_ignore,
                                                  mode='iof')
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
            distances[ignore_idxs, :] = INF
            assigned_gt_inds[ignore_idxs] = -1
        # Selecting candidates based on the center distance
        candidate_idxs = []
        start_idx = 0

        for level, bboxes_per_level in enumerate(num_level_bboxes):
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[start_idx:end_idx, :]  #(A,G)

            selectable_k = min(self.topk, bboxes_per_level)
            topk_idxs_per_level = np.argsort(distances_per_level, axis=0)
            topk_idxs_per_level = topk_idxs_per_level[:selectable_k]
            #print('AT-LEVEL:', start_idx, end_idx, bboxes_per_level, topk_idxs_per_level.shape)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = np.concatenate(
            candidate_idxs,
            axis=0)  # candidate anchors (topk*num_level_bboxes, G) = (AK, G)

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        candidate_overlaps = overlaps[candidate_idxs,
                                      np.arange(num_gt)]  #(AK,G)

        overlaps_mean_per_gt = np.mean(candidate_overlaps, axis=0)
        overlaps_std_per_gt = np.std(candidate_overlaps, axis=0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]
        #print('CAND:', candidate_idxs.shape, candidate_overlaps.shape, is_pos.shape)
        #print('BOXES:', bboxes_cx.shape)
        # limit the positive sample's center in gt
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes

        # ep_bboxes_cx = bboxes_cx.view(1, -1).expand(
        #     num_gt, num_bboxes).contiguous().view(-1)
        ep_bboxes_cx = np.tile(np.reshape(bboxes_cx, [1, -1]),
                               [num_gt, 1]).reshape([-1])
        ep_bboxes_cy = np.tile(np.reshape(bboxes_cy, [1, -1]),
                               [num_gt, 1]).reshape([-1])
        candidate_idxs = candidate_idxs.reshape([-1])
        # calculate the left, top, right, bottom distance between positive
        # bbox center and gt side
        l_ = ep_bboxes_cx[candidate_idxs].reshape([-1, num_gt]) - gt_bboxes[:,
                                                                            0]
        t_ = ep_bboxes_cy[candidate_idxs].reshape([-1, num_gt]) - gt_bboxes[:,
                                                                            1]
        r_ = gt_bboxes[:, 2] - ep_bboxes_cx[candidate_idxs].reshape(
            [-1, num_gt])
        b_ = gt_bboxes[:, 3] - ep_bboxes_cy[candidate_idxs].reshape(
            [-1, num_gt])
        #is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01

        dist_min = np.stack([l_, t_, r_, b_], axis=1).min(axis=1)  # (A,G)
        dist_min /= gt_area
        #print('ATTT:', l_.shape, t_.shape, dist_min.shape, self.mode)
        if self.mode == 0:
            is_in_gts = dist_min > 0.001
        elif self.mode == 1:
            is_in_gts = dist_min > -0.25
        elif self.mode == 2:
            is_in_gts = dist_min > -0.15
            #dist_expand = torch.clamp(gt_area / 16.0, min=1.0, max=3.0)
            #dist_min.mul_(dist_expand)
            #is_in_gts = dist_min > -0.25
        elif self.mode == 3:
            dist_expand = np.clip(gt_area / 16.0, a_min=1.0, a_max=6.0)
            dist_min *= dist_expand
            # dist_min.mul_(dist_expand)
            is_in_gts = dist_min > -0.2
        elif self.mode == 4:
            dist_expand = np.clip(gt_area / 16.0, a_min=0.5, a_max=6.0)
            dist_min *= dist_expand
            is_in_gts = dist_min > -0.2
        elif self.mode == 5:
            dist_div = np.clip(gt_area / 16.0, min=0.5, max=3.0)
            dist_min /= dist_div
            is_in_gts = dist_min > -0.2
        else:
            raise ValueError

        #print(gt_area.shape, is_in_gts.shape, is_pos.shape)
        is_pos = is_pos & is_in_gts

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf = np.full_like(overlaps, -INF).T.reshape([-1])

        index = candidate_idxs.reshape([-1])[is_pos.reshape([-1])]

        overlaps_inf[index] = overlaps.T.reshape([-1])[index]
        overlaps_inf = overlaps_inf.reshape([num_gt, -1]).T

        max_overlaps = np.max(overlaps_inf, axis=1)
        argmax_overlaps = np.argmax(overlaps_inf, axis=1)

        assigned_gt_inds[
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

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
        return AssignResult(num_gt,
                            assigned_gt_inds,
                            max_overlaps,
                            labels=assigned_labels)