import tensorflow as tf
import numpy as np
from .loss_base import LossBase
from .core.atss_assigner import ATSSAssigner
from .core.pseudo_sampler import PseudoSampler
from .core.task_aligned_assigner import TaskAlignedAssigner
from .core.transform import distance2bbox, bbox2distance, kps2distance
from .core.utils import anchor_inside_flags, unmap, multi_apply
from pprint import pprint
from utils.io import load_BFM
from .core import LOSS_FUNCS_FACTORY
import cv2

EPS = 1e-12


class AnchorTOODLoss(LossBase):

    def __init__(self, config):
        self.max_obj_num = config.max_obj_num
        self.config = config['loss']
        self.train_cfg = self.config['train_cfg']
        self.test_cfg = self.config['test_cfg']

        self.init_loss_cls_cfg = self.config['initial_loss_cls']
        self.loss_cls_cfg = self.config['loss_cls']

        self.loss_bbox_cfg = self.config['loss_bbox']
        self.loss_kps_cfg = self.config['loss_kps']
        self.anchor_generator = self.config["anchor_generator"]
        self.num_anchors = len(self.anchor_generator.scales)
        self.use_sigmoid_cls = True
        self.num_classes = self.config.num_classes
        self.height, self.width = self.config.resize_size
        self.batch_size = self.config.batch_size
        self.initial_epoch = self.train_cfg.initial_epoch
        self.alpha = self.train_cfg.alpha
        self.beta = self.train_cfg.beta
        self.num_level_anchors = [3200, 800, 200]
        self.anchor_generator_strides = [(8, 8), (16, 16), (32, 32)]
        self.pad_shape = tf.constant([self.height, self.width],
                                     dtype=tf.dtypes.float32)
        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1
        self.initial_assigner = ATSSAssigner(
            topk=self.train_cfg["initial_assigner"]["topk"])

        self.task_assigner = TaskAlignedAssigner(
            topk=self.train_cfg["task_assigner"]["topk"],
            num_level_anchors=self.num_level_anchors,
            anchor_generator_strides=self.anchor_generator_strides)
        self.sampler = PseudoSampler()
        # configurize
        self.use_dfl = False
        self.use_params = True
        self.use_qscore = False
        self.reg_max = 8
        self.loss_kps_std = 1.0
        self.diou_func = LOSS_FUNCS_FACTORY["DIoULoss"](loss_weight=1.0)
        self.init_cls_func = LOSS_FUNCS_FACTORY[self.init_loss_cls_cfg["type"]](
            ["use_sigmoid"],
            gamma=self.init_loss_cls_cfg["gamma"],
            alpha=self.init_loss_cls_cfg["alpha"],
            loss_weight=self.init_loss_cls_cfg["loss_weight"])
        self.cls_func = LOSS_FUNCS_FACTORY[self.loss_cls_cfg["type"]](
            ["use_sigmoid"], gamma=self.loss_cls_cfg["gamma"])
        self.loss_bbox_func = LOSS_FUNCS_FACTORY[self.loss_bbox_cfg["type"]](
            self.loss_bbox_cfg["loss_weight"])
        self.feat_size = tf.constant([(40, 40), (20, 20), (10, 10)])
        multi_level_anchors, self.target_num_lv_anchors, multi_level_flags = self.get_anchors(
            self.batch_size, self.feat_size)
        self.b_anchors = tf.concat(multi_level_anchors, axis=1)
        self.b_flags = tf.concat(multi_level_flags, axis=1)
        self.b_anchor_centers = self.anchor_center(self.b_anchors)
        self.b_gt_bboxes_ignore = np.zeros(shape=(self.batch_size,
                                                  self.max_obj_num)).astype(
                                                      np.bool_)

    def build_loss(self, epochs, logits, targets, training):
        """
            Get targets for GFL head.
            This method is almost the same as `AnchorHead.get_targets()`. Besides
            returning the targets as the parent method does, it also returns the
            anchors as the first element of the returned tuple.
        """
        self.epochs = epochs
        b_gt_bboxes, b_gt_labels = targets['b_bboxes'], targets['b_labels']
        b_gt_kps, b_origin_sizes = targets['b_kps'], targets['b_origin_sizes']
        b_gt_params = targets['params']
        tmp_cls_preds, tmp_reg_dist_preds = [], []
        for lv_feats in logits['multi_lv_feats']:
            b_cls_preds, b_reg_dist, _ = lv_feats
            b_cls_preds = tf.reshape(
                b_cls_preds, (self.batch_size, -1, self.cls_out_channels))
            b_reg_dist = tf.reshape(b_reg_dist, (self.batch_size, -1, 4))

            tmp_cls_preds.append(b_cls_preds)
            tmp_reg_dist_preds.append(b_reg_dist)
        all_cls_scores = tf.concat(tmp_cls_preds, axis=1)
        all_reg_dist_preds = tf.concat(tmp_reg_dist_preds, axis=1)
        with tf.device('CPU'):
            anchors_0, anchors_1, anchors_2, labels_0, labels_1, labels_2, label_weights_0, label_weights_1, label_weights_2, bbox_targets_0, bbox_targets_1, bbox_targets_2, bbox_weights_0, bbox_weights_1, bbox_weights_2, norm_alignment_metric_0, norm_alignment_metric_1, norm_alignment_metric_2, num_total_samples = tf.py_function(
                self.get_targets,
                inp=[
                    all_cls_scores, all_reg_dist_preds,
                    self.target_num_lv_anchors, self.b_anchors, self.b_flags,
                    b_gt_bboxes, b_gt_kps, b_gt_params, b_gt_labels,
                    self.num_classes
                ],
                Tout=(tf.float32, tf.float32, tf.float32, tf.float32,
                      tf.float32, tf.float32, tf.float32, tf.float32,
                      tf.float32, tf.float32, tf.float32, tf.float32,
                      tf.float32, tf.float32, tf.float32, tf.float32,
                      tf.float32, tf.float32, tf.float32))
        # num_total_samples
        anchors_list = [anchors_0, anchors_1, anchors_2]
        labels_list = [labels_0, labels_1, labels_2]
        label_weights_list = [label_weights_0, label_weights_1, label_weights_2]
        bbox_targets_list = [bbox_targets_0, bbox_targets_1, bbox_targets_2]
        bbox_weights_list = [bbox_weights_0, bbox_weights_1, bbox_weights_2]
        # keypoints_targets_list = [
        #     keypoints_targets_0, keypoints_targets_1, keypoints_targets_2
        # ]
        # params_targets_list = [
        #     params_targets_0, params_targets_1, params_targets_2
        # ]
        alignment_metrics_list = [
            norm_alignment_metric_0, norm_alignment_metric_1,
            norm_alignment_metric_2
        ]
        loss_cls_list, loss_bbox_list, \
              cls_avg_factors, bbox_avg_factors = self.compute_loss(
            anchors_list, logits['multi_lv_feats'], labels_list,
            label_weights_list, bbox_targets_list, alignment_metrics_list,
            self.anchor_generator_strides, num_total_samples)
        losses = {'total': 0}
        cls_avg_factor = tf.math.reduce_sum(cls_avg_factors)
        if cls_avg_factor < EPS:
            cls_avg_factor = 1.
        loss_cls_list = list(map(lambda x: x / cls_avg_factor, loss_cls_list))
        bbox_avg_factors = tf.math.reduce_sum(bbox_avg_factors)
        loss_bbox_list = list(
            map(lambda x: x / bbox_avg_factors, loss_bbox_list))
        losses['loss_cls'] = tf.math.reduce_mean(loss_cls_list)
        losses['loss_bbox'] = tf.math.reduce_mean(loss_bbox_list)

        for key in losses:
            losses['total'] += losses[key]
        return losses

    # @tf.function
    def compute_loss(self, anchors_list, multi_lv_feats, labels_list,
                     label_weights_list, bbox_targets_list,
                     alignment_metrics_list, anchor_generator_strides,
                     num_total_samples):
        """Compute loss of a single scale level.
        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Cls and quality joint scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_pred (Tensor): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (N, num_total_anchors, 4).
            stride (tuple): Stride in this scale level.
            num_total_samples (int): Number of positive samples that is
                reduced over all GPUs.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        loss_cls_list, loss_bbox_list = [], []
        aligned_metric_list, pos_bbox_weight_list = [], []
        for i, (anchors, lv_feats, labels, label_weights, bbox_targets,
                alignment_metrics) in enumerate(
                    zip(anchors_list, multi_lv_feats, labels_list,
                        label_weights_list, bbox_targets_list,
                        alignment_metrics_list)):
            # if i == 0:
            #     continue
            b_cls_preds, b_reg_dist_preds, b_reg_offset = lv_feats
            b_reg_dist_preds = self.offset_sampling(labels, b_reg_dist_preds,
                                                    b_reg_offset)
            anchors = tf.reshape(anchors, [-1, 4])
            b_cls_preds = tf.reshape(b_cls_preds, [-1, self.cls_out_channels])
            stride = anchor_generator_strides[i]

            b_reg_dist_preds = tf.reshape(b_reg_dist_preds, [-1, 4])
            bbox_targets = tf.reshape(bbox_targets, [-1, 4])
            labels = tf.reshape(labels, [-1])
            label_weights = tf.reshape(label_weights, [-1])
            bg_class_ind = self.num_classes
            pos_inds = (labels >= 0.) & (labels < bg_class_ind)
            pos_inds = tf.where(pos_inds == True)
            score = tf.zeros_like(labels)
            pos_bbox_targets = tf.gather_nd(bbox_targets, pos_inds)
            # poas_bbox_weights = tf.gather_nd(bbox_weights, pos_inds)
            pos_reg_dist_pred = tf.gather_nd(b_reg_dist_preds, pos_inds)
            pos_anchors = tf.gather_nd(anchors, pos_inds)
            # pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]
            pos_anchor_centers = tf.gather_nd(self.b_anchor_centers[i],
                                              pos_inds)
            pos_anchor_centers /= stride[0]
            pos_decode_bbox_targets = pos_bbox_targets / stride[0]
            pos_decode_bbox_pred = distance2bbox(pos_anchor_centers,
                                                 pos_reg_dist_pred)
            pos_bbox_weights = self.centerness_target(pos_anchors,
                                                      pos_bbox_targets)
            if self.epochs > self.initial_epoch:
                loss_cls = self.init_cls_func(b_cls_preds,
                                              labels,
                                              label_weights,
                                              avg_factor=1.0)

                pos_bbox_weights = self.centerness_target(
                    pos_anchors, pos_bbox_targets)
            else:
                alignment_metrics = tf.reshape(alignment_metrics, (-1))
                pos_bbox_weights = tf.gather_nd(alignment_metrics, pos_inds)
                loss_cls = self.cls_func(b_cls_preds,
                                         labels,
                                         alignment_metrics,
                                         avg_factor=1.0)  # num_total_samples)
            loss_bbox = self.loss_bbox_func(pos_decode_bbox_pred,
                                            pos_decode_bbox_targets,
                                            pos_bbox_weights,
                                            avg_factor=num_total_samples)

            aligned_metric_list.append(tf.math.reduce_sum(alignment_metrics))
            pos_bbox_weight_list.append(tf.math.reduce_sum(pos_bbox_weights))
            loss_cls_list.append(loss_cls)
            loss_bbox_list.append(loss_bbox)
        return loss_cls_list, loss_bbox_list, aligned_metric_list, pos_bbox_weight_list

    def get_targets(self,
                    all_cls_scores,
                    all_dist_preds,
                    b_num_level_anchors,
                    b_anchors,
                    b_flags,
                    b_gt_bboxes,
                    b_gt_kps,
                    b_gt_params=None,
                    b_gt_labels=None,
                    label_channels=1,
                    unmap_outputs=True,
                    return_sampling_results=False):
        """Get targets for GFL head.
        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        all_cls_scores = all_cls_scores.numpy()
        all_dist_preds = all_dist_preds.numpy()
        b_num_level_anchors = b_num_level_anchors.numpy()
        b_anchors = b_anchors.numpy()
        b_flags = b_flags.numpy()
        b_gt_bboxes = b_gt_bboxes.numpy()
        b_gt_kps = b_gt_kps.numpy()
        b_gt_bboxes_ignore = self.b_gt_bboxes_ignore
        b_gt_params = b_gt_params.numpy()
        b_gt_labels = b_gt_labels.numpy()
        #print('QQQ:', num_imgs, gt_bboxes_list[0].shape)
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list,
         pos_assigned_gt_inds_list, assign_metrics_list, assign_ious_list,
         inside_flags_list) = multi_apply(self._get_targets_single,
                                          all_cls_scores,
                                          all_dist_preds,
                                          self.b_anchor_centers,
                                          b_anchors,
                                          b_flags,
                                          b_num_level_anchors,
                                          b_gt_bboxes,
                                          b_gt_bboxes_ignore,
                                          b_gt_kps,
                                          b_gt_params,
                                          b_gt_labels,
                                          label_channels=label_channels,
                                          unmap_outputs=unmap_outputs)

        # sampled anchors of all images
        num_total_pos = sum([max(np.size(inds), 1) for inds in pos_inds_list])
        # num_total_neg = sum([max(np.size(inds), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        anchors_0, anchors_1, anchors_2 = self.images_to_levels(
            all_anchors, self.num_level_anchors)
        labels_0, labels_1, labels_2 = self.images_to_levels(
            all_labels, self.num_level_anchors)
        label_weights_0, label_weights_1, label_weights_2 = self.images_to_levels(
            all_label_weights, self.num_level_anchors)

        bbox_targets_0, bbox_targets_1, bbox_targets_2 = self.images_to_levels(
            all_bbox_targets, self.num_level_anchors)
        bbox_weights_0, bbox_weights_1, bbox_weights_2 = self.images_to_levels(
            all_bbox_weights, self.num_level_anchors)
        # keypoints_targets_0, keypoints_targets_1, keypoints_targets_2 = self.images_to_levels(
        #     all_kps_targets, self.num_level_anchors)
        # params_targets_0, params_targets_1, params_targets_2 = self.images_to_levels(
        #     all_params_targets, self.num_level_anchors)
        num_total_samples = np.mean(num_total_pos, dtype=np.float32)
        num_total_samples = max(num_total_samples, 1.0)

        if self.epochs < self.initial_epoch:
            norm_alignment_metric_0 = bbox_weights_0[:, :, 0]
            norm_alignment_metric_1 = bbox_weights_1[:, :, 0]
            norm_alignment_metric_2 = bbox_weights_2[:, :, 0]
        else:
            # for alignment metric
            all_norm_alignment_metrics = []
            for i in range(self.batch_size):
                inside_flags = inside_flags_list[i]
                image_norm_alignment_metrics = np.zeros(
                    shape=(all_label_weights[i].shape[0], ))
                image_norm_alignment_metrics_inside = np.zeros(
                    shape=(np.sum(inside_flags).astype(np.int32), ))
                pos_assigned_gt_inds = pos_assigned_gt_inds_list[i]
                pos_inds = pos_inds_list[i]
                class_assigned_gt_inds = np.unique(pos_assigned_gt_inds)
                for gt_inds in class_assigned_gt_inds:
                    gt_class_inds = pos_inds[pos_assigned_gt_inds == gt_inds]
                    pos_alignment_metrics = assign_metrics_list[i][
                        gt_class_inds]
                    pos_ious = assign_ious_list[i][gt_class_inds]
                    pos_norm_alignment_metrics = pos_alignment_metrics / (
                        pos_alignment_metrics.max() + 10e-8) * pos_ious.max()

                    image_norm_alignment_metrics_inside[
                        gt_class_inds] = pos_norm_alignment_metrics

                image_norm_alignment_metrics[
                    inside_flags] = image_norm_alignment_metrics_inside
                all_norm_alignment_metrics.append(image_norm_alignment_metrics)

            norm_alignment_metrics_list = self.images_to_levels(
                all_norm_alignment_metrics, self.num_level_anchors)
            norm_alignment_metric_0 = norm_alignment_metrics_list[0]
            norm_alignment_metric_1 = norm_alignment_metrics_list[1]
            norm_alignment_metric_2 = norm_alignment_metrics_list[2]

        return (anchors_0, anchors_1, anchors_2, labels_0, labels_1, labels_2,
                label_weights_0, label_weights_1, label_weights_2,
                bbox_targets_0, bbox_targets_1, bbox_targets_2, bbox_weights_0,
                bbox_weights_1, bbox_weights_2, norm_alignment_metric_0,
                norm_alignment_metric_1, norm_alignment_metric_2,
                num_total_samples)

    def _get_targets_single(self,
                            cls_scores,
                            dist_preds,
                            anchor_centers,
                            flat_anchors,
                            valid_flags,
                            num_level_anchors,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_kps,
                            gt_params,
                            gt_labels,
                            label_channels=1,
                            unmap_outputs=True):
        img_shape = self.pad_shape
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags, img_shape,
                                           self.train_cfg["allowed_border"])
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]
        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        valid_mask = np.all(np.isfinite(gt_bboxes), axis=-1)
        gt_bboxes = gt_bboxes[valid_mask].reshape([-1, 4])
        gt_bboxes_ignore = gt_bboxes_ignore[valid_mask[:, 0]]
        gt_labels = gt_labels[valid_mask[:, 0]]
        gt_kps = gt_kps[valid_mask[:, 0]]
        gt_params = gt_params[valid_mask[:, 0]]
        # implement taskalignment
        if self.epochs < self.initial_epoch:
            assign_result = self.initial_assigner.assign(
                anchors, num_level_anchors_inside, gt_bboxes, gt_bboxes_ignore,
                gt_labels)
            assign_ious = assign_result.max_overlaps
            assign_metrics = None
        else:
            assign_result = self.task_assigner.assign(
                cls_scores[inside_flags, :], dist_preds[inside_flags, :],
                anchor_centers, num_level_anchors_inside, gt_bboxes,
                gt_bboxes_ignore, gt_labels, self.alpha, self.beta)
            assign_ious = assign_result.max_overlaps
            assign_metrics = assign_result.assign_metrics
        sampling_result = self.sampler.sample(assign_result, anchors, gt_bboxes)
        num_valid_anchors = anchors.shape[0]
        bbox_targets = np.zeros_like(anchors)
        bbox_weights = np.zeros_like(anchors)
        # kps_targets = np.zeros(shape=(anchors.shape[0], 2))
        # params_targets = np.zeros(shape=(anchors.shape[0],
        #                                  self.params_channels))
        labels = np.full(shape=(num_valid_anchors, ),
                         fill_value=self.num_classes,
                         dtype=np.int32)
        label_weights = np.zeros(shape=(num_valid_anchors), dtype=np.float32)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            # params_targets[pos_inds, :] = gt_params[
            #     sampling_result.pos_assigned_gt_inds, :]
            # kps_targets[pos_inds, :] = gt_kps[
            #     sampling_result.pos_assigned_gt_inds, :2].reshape((-1, 2))
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg["pos_weight"] <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg["pos_weight"]
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.shape[0]
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(labels,
                           num_total_anchors,
                           inside_flags,
                           fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            # kps_targets = unmap(kps_targets, num_total_anchors, inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
        return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds, sampling_result.pos_assigned_gt_inds,
                assign_metrics, assign_ious, inside_flags)

    @tf.function
    def offset_sampling(self, labels, b_bbox_preds, b_reg_offset):
        n, h, w, c = [tf.shape(b_bbox_preds)[i] for i in range(4)]
        b_map_lbs = tf.reshape(labels, (self.batch_size, h, w, -1))
        pos_map_idxs = (b_map_lbs >= 0.) & (b_map_lbs < self.num_classes)
        pos_map_idxs = tf.where(pos_map_idxs == True)[..., :-1]
        offset_vals = tf.gather_nd(b_reg_offset, pos_map_idxs)
        pos_map_idxs = tf.cast(pos_map_idxs, tf.int32)
        b_idxs = pos_map_idxs[:, :1]
        map_idxs = tf.cast(pos_map_idxs[:, None, 1:], tf.float32) + tf.reshape(
            offset_vals, (-1, 4, 2)) + 0.5

        map_idxs = tf.cast(map_idxs, tf.int32)
        map_idxs = tf.tile(map_idxs, [1, self.num_anchors, 1])
        # (10, 4, 3),
        N = tf.shape(map_idxs)[0]

        b_idxs = tf.tile(b_idxs[:, None, :], [1, 4 * self.num_anchors, 1])
        b_map_idxs = tf.concat([b_idxs, map_idxs], axis=-1)
        c_idxs = tf.range(4 * self.num_anchors, dtype=tf.int32)
        c_idxs = tf.tile(c_idxs[None, :, None], [N, 1, 1])
        b_map_idxs = tf.concat([b_map_idxs, c_idxs], axis=-1)
        b_offset_bboxes = tf.gather_nd(b_bbox_preds, b_map_idxs)
        b_bbox_preds = tf.tensor_scatter_nd_update(b_bbox_preds, pos_map_idxs,
                                                   b_offset_bboxes)
        return b_bbox_preds

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        a1, a2, a3 = num_level_anchors
        split_inside_flags = [
            inside_flags[:a1], inside_flags[a1:a1 + a2],
            inside_flags[a1 + a2:a1 + a2 + a3]
        ]

        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside

    def images_to_levels(self, target, num_levels):
        """Convert targets by image to targets by feature level.
        [target_img0, target_img1] -> [target_level0, target_level1, ...]
        """
        target = np.stack(target, axis=0)
        level_targets = []
        start = 0
        for n in num_levels:
            end = start + n
            feats = target[:, start:end]
            level_targets.append(feats)
            start = end
        return level_targets
        # return np.concatenate(level_targets, axis=1)

    @tf.function
    def get_anchors(self, batch, featmap_sizes):
        multi_level_anchors, num_level_anchors = self.anchor_generator.grid_anchors(
            batch, featmap_sizes)
        multi_level_flags = self.anchor_generator.valid_flags(
            batch, featmap_sizes, self.pad_shape)
        return multi_level_anchors, num_level_anchors, multi_level_flags

    @tf.function
    def split2level(self, inputs):
        sp0 = inputs[:, :self.num_level_anchors[0]]
        sp1 = inputs[:, self.num_level_anchors[0]:self.num_level_anchors[0] +
                     self.num_level_anchors[1]]
        sp2 = inputs[:, self.num_level_anchors[0] +
                     self.num_level_anchors[1]:self.num_level_anchors[0] +
                     self.num_level_anchors[1] + self.num_level_anchors[2]]
        return [sp0, sp1, sp2]

    @tf.function
    def anchor_center(self, anchors):
        """Get anchor centers from anchors.

        Args:
            anchors (Tensor): Anchor list with shape (B, N, 4), "xyxy" format.

        Returns:
            Tensor: Anchor centers with shape (B, N, 2), "xy" format.
        """
        anchors_cx = (anchors[:, :, 2] + anchors[:, :, 0]) / 2
        anchors_cy = (anchors[:, :, 3] + anchors[:, :, 1]) / 2
        return tf.stack([anchors_cx, anchors_cy], axis=-1)

    @tf.function
    def resample(self, shapeMU, shapePC, expPC, kpt_ind):
        index = tf.random.shuffle(tf.range(start=0, limit=53215,
                                           dtype=tf.int32))[:132]
        keypoints_resample = tf.stack([3 * index, 3 * index + 1, 3 * index + 2])
        keypoints_mix = tf.concat([kpt_ind, keypoints_resample], axis=-1)
        keypoints_mix = tf.reshape(tf.transpose(keypoints_mix), [-1])
        u_base = tf.gather(shapeMU, keypoints_mix)
        shp_base = tf.gather(shapePC, keypoints_mix)
        exp_base = tf.gather(expPC, keypoints_mix)
        return u_base, shp_base, exp_base

    def centerness_target(self, anchors, bbox_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        # for bbox-based
        # gts = self.bbox_coder.decode(anchors, bbox_targets)
        # for point-based
        gts = bbox_targets
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        l_ = anchors_cx - gts[:, 0]
        t_ = anchors_cy - gts[:, 1]
        r_ = gts[:, 2] - anchors_cx
        b_ = gts[:, 3] - anchors_cy

        left_right = tf.stack([l_, r_], axis=1)
        top_bottom = tf.stack([t_, b_], axis=1)
        centerness = tf.math.sqrt((tf.math.reduce_min(left_right, axis=-1) /
                                   tf.math.reduce_max(left_right, axis=-1)) *
                                  (tf.math.reduce_min(top_bottom, axis=-1) /
                                   tf.math.reduce_max(top_bottom, axis=-1)))
        return centerness