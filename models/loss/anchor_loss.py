import tensorflow as tf
from .loss_base import LossBase
from .core.atss_assigner import ATSSAssigner
from .core.pseudo_sampler import PseudoSampler
from .core.iou_loss import DIoULoss
from .core.smooth_l1_loss import SmoothL1Loss
from .core.gfocal_loss import QualityFocalLoss
from .core.transform import distance2bbox, bbox2distance, kps2distance
from .core.iou2d_calculator import bbox_overlaps, bbox_overlapping
from .core.utils import anchor_inside_flags, unmap, multi_apply
from pprint import pprint
import numpy as np


class AnchorLoss(LossBase):

    def __init__(self, config):

        self.max_obj_num = config.max_obj_num
        self.config = config['loss']
        self.train_cfg = self.config['train_cfg']
        self.test_cfg = self.config['test_cfg']
        self.loss_cls_cfg = self.config['loss_cls']
        self.loss_bbox_cfg = self.config['loss_bbox']
        self.loss_kps_cfg = self.config['loss_kps']
        self.anchor_generator = self.config["anchor_generator"]
        self.use_sigmoid_cls = True
        self.num_classes = self.config.num_classes
        self.height, self.width = self.config.resize_size
        self.batch_size = self.config.batch_size
        self.NK = 5
        # self.num_level_anchors = [12800, 3200, 800]
        self.num_level_anchors = [3200, 800, 200]
        self.pad_shape = tf.constant([self.height, self.width],
                                     dtype=tf.dtypes.float32)
        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1
        self.assigner = ATSSAssigner(topk=self.train_cfg["assigner"]["topk"])
        self.sampler = PseudoSampler()
        # configurize
        self.use_dfl = False
        self.use_kps = self.config.use_kps
        self.use_qscore = True
        self.reg_max = 8
        self.anchor_generator_strides = [(8, 8), (16, 16), (32, 32)]
        self.loss_kps_std = 1.0
        self.diou_func = DIoULoss(loss_weight=1.0)
        self.smooth_func = SmoothL1Loss(
            beta=self.loss_kps_cfg["beta"],
            loss_weight=self.loss_kps_cfg["loss_weight"])
        self.cls_func = QualityFocalLoss(
            use_sigmoid=self.loss_cls_cfg["use_sigmoid"],
            beta=self.loss_cls_cfg["beta"],
            loss_weight=self.loss_cls_cfg["loss_weight"])
        self.feat_size = tf.constant([(40, 40), (20, 20), (10, 10)])
        multi_level_anchors, self.target_num_lv_anchors, multi_level_flags = self.get_anchors(
            self.batch_size, self.feat_size)
        self.b_anchors = tf.concat(multi_level_anchors, axis=1)
        self.b_flags = tf.concat(multi_level_flags, axis=1)

    def build_loss(self, logits, targets, training):
        """
            Get targets for GFL head.
            This method is almost the same as `AnchorHead.get_targets()`. Besides
            returning the targets as the parent method does, it also returns the
            anchors as the first element of the returned tuple.
        """
        b_gt_bboxes, b_gt_keypointss, b_gt_labels = targets[
            'b_bboxes'], targets['b_keypoints'], targets['b_labels']
        with tf.device('CPU'):
            anchors_list, labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, keypoints_targets_list, keypoints_weights_list, num_total_samples = tf.py_function(
                self.get_targets,
                inp=[
                    self.target_num_lv_anchors, self.b_anchors, self.b_flags,
                    b_gt_bboxes, b_gt_keypointss, b_gt_labels, self.num_classes
                ],
                Tout=(tf.float32, tf.float32, tf.float32, tf.float32,
                      tf.float32, tf.float32, tf.float32, tf.float32))

        anchors_list = self.split2level(anchors_list)
        labels_list = self.split2level(labels_list)
        label_weights_list = self.split2level(label_weights_list)
        bbox_targets_list = self.split2level(bbox_targets_list)
        bbox_weights_list = self.split2level(bbox_weights_list)
        keypoints_targets_list = self.split2level(keypoints_targets_list)
        keypoints_weights_list = self.split2level(keypoints_weights_list)
        loss_cls_list, loss_bbox_list, weight_targets_list = self.compute_loss(
            anchors_list, logits['multi_lv_feats'], labels_list,
            label_weights_list, bbox_targets_list, keypoints_targets_list,
            keypoints_weights_list, self.anchor_generator_strides,
            num_total_samples)
        losses = {'total': 0}
        avg_factor = tf.math.reduce_sum(weight_targets_list)
        loss_bbox_list = list(map(lambda x: x / avg_factor, loss_bbox_list))
        if self.use_kps:
            loss_kps_list = list(map(lambda x: x / avg_factor, loss_kps_list))
            losses['loss_kps'] = tf.math.reduce_mean(loss_kps_list)

        losses['loss_cls'] = tf.math.reduce_mean(loss_cls_list)
        losses['loss_bbox'] = tf.math.reduce_mean(loss_bbox_list)
        for key in losses:
            losses['total'] += losses[key]
        return losses

    # @tf.function
    def compute_loss(self, anchors_list, multi_lv_feats, labels_list,
                     label_weights_list, bbox_targets_list,
                     keypoints_targets_list, keypoints_weights_list,
                     anchor_generator_strides, num_total_samples):
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
        loss_cls_list, loss_bbox_list, loss_kps_list = [], [], []
        weight_targets_list = []
        for i, (anchors, lv_feats, labels, label_weights, bbox_targets,
                kps_targets, kps_weights) in enumerate(
                    zip(anchors_list, multi_lv_feats, labels_list,
                        label_weights_list, bbox_targets_list,
                        keypoints_targets_list, keypoints_weights_list)):
            b_cls_preds, b_bbox_preds = lv_feats
            anchors = tf.reshape(anchors, [-1, 4])
            b_cls_preds = tf.reshape(b_cls_preds, [-1, self.cls_out_channels])
            stride = anchor_generator_strides[i]
            if not self.use_dfl:
                b_bbox_preds = tf.reshape(b_bbox_preds, [-1, 4])
            else:
                b_bbox_preds = tf.reshape(b_bbox_preds,
                                          [-1, 4 * (self.reg_max + 1)])
            bbox_targets = tf.reshape(bbox_targets, [-1, 4])
            labels = tf.reshape(labels, [-1])
            label_weights = tf.reshape(label_weights, [-1])
            if self.use_kps:
                b_kp_preds = tf.reshape(b_kp_preds, [-1, self.NK * 2])
                kps_targets = tf.reshape(kps_targets, [-1, self.NK * 2])
                kps_weights = tf.reshape(kps_weights, [-1, self.NK * 2])
            bg_class_ind = self.num_classes
            pos_inds = (labels >= 0.) & (labels < bg_class_ind)
            pos_inds = tf.where(pos_inds == True)
            score = tf.zeros_like(labels)
            pos_bbox_targets = tf.gather_nd(bbox_targets, pos_inds)
            pos_bbox_pred = tf.gather_nd(b_bbox_preds, pos_inds)
            pos_anchors = tf.gather_nd(anchors, pos_inds)
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]
            weight_targets = tf.math.sigmoid(b_cls_preds)
            weight_targets = tf.gather_nd(
                tf.math.reduce_max(weight_targets, axis=1), pos_inds)
            pos_decode_bbox_targets = pos_bbox_targets / stride[0]
            pos_decode_bbox_pred = distance2bbox(pos_anchor_centers,
                                                 pos_bbox_pred)
            if self.use_kps:
                pos_kps_targets = tf.gather_nd(kps_targets, pos_inds)
                pos_kps_pred = tf.gather_nd(b_kp_preds, pos_inds)
                pos_kps_weights = tf.math.reduce_max(kps_weights, axis=1)
                pos_kps_weights = tf.gather_nd(pos_kps_weights,
                                               pos_inds) * weight_targets
                pos_kps_weights = tf.reshape(pos_kps_weights, [-1, 1])
                pos_decode_kps_targets = kps2distance(
                    pos_anchor_centers, pos_kps_targets / stride[0])
                pos_decode_kps_pred = pos_kps_pred
            if self.use_qscore:
                s = bbox_overlapping(pos_decode_bbox_pred,
                                     pos_decode_bbox_targets,
                                     mode='iou',
                                     is_aligned=True)
                score = tf.tensor_scatter_nd_update(score, pos_inds, s)
            else:
                score = tf.tensor_scatter_nd_update(
                    score, pos_inds, tf.ones(shape=(tf.shape(pos_inds)[0])))
            # regression loss
            loss_bbox = self.diou_func(pos_decode_bbox_pred,
                                       pos_decode_bbox_targets,
                                       weight=weight_targets,
                                       avg_factor=1.0)
            if self.use_kps:
                loss_kps = self.smooth_func(
                    pos_decode_kps_pred * self.loss_kps_std,
                    pos_decode_kps_targets * self.loss_kps_std,
                    weight=pos_kps_weights,
                    avg_factor=1.0)
            # else:
            #     loss_kps = tf.math.reduce_sum(b_kp_preds) * 0.
            if self.use_dfl:
                pred_corners = tf.reshape(pos_bbox_pred, (-1, self.reg_max + 1))
                target_corners = bbox2distance(pos_anchor_centers,
                                               pos_decode_bbox_targets,
                                               self.reg_max).reshape(-1)
                loss_dfl = self.dfl_func(pred_corners,
                                         target_corners,
                                         weight=weight_targets[:, None].expand(
                                             -1, 4).reshape(-1),
                                         avg_factor=4.0)
            else:
                loss_dfl = tf.math.reduce_sum(b_bbox_preds) * 0.
            loss_cls = self.cls_func(b_cls_preds, (labels, score),
                                     weight=label_weights,
                                     avg_factor=num_total_samples)
            loss_cls_list.append(loss_cls)
            loss_bbox_list.append(loss_bbox)
            # loss_kps_list.append(loss_kps)
            weight_targets_list.append(tf.math.reduce_sum(weight_targets))

        return loss_cls_list, loss_bbox_list, weight_targets_list

    def get_targets(self,
                    b_num_level_anchors,
                    b_anchors,
                    b_flags,
                    b_gt_bboxes,
                    b_gt_keypoints=None,
                    b_gt_labels=None,
                    label_channels=1,
                    unmap_outputs=True,
                    return_sampling_results=False):
        """Get targets for GFL head.
        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        b_num_level_anchors = b_num_level_anchors.numpy()
        b_anchors = b_anchors.numpy()
        b_flags = b_flags.numpy()
        b_gt_bboxes = b_gt_bboxes.numpy()
        b_gt_bboxes_ignore = np.zeros(shape=(self.batch_size,
                                             self.max_obj_num)).astype(np.bool_)
        b_gt_keypoints = b_gt_keypoints.numpy()
        b_gt_labels = b_gt_labels.numpy()
        #print('QQQ:', num_imgs, gt_bboxes_list[0].shape)
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, all_keypoints_targets, all_keypoints_weights,
         pos_inds_list,
         neg_inds_list) = multi_apply(self._get_targets_single,
                                      b_anchors,
                                      b_flags,
                                      b_num_level_anchors,
                                      b_gt_bboxes,
                                      b_gt_bboxes_ignore,
                                      b_gt_keypoints,
                                      b_gt_labels,
                                      label_channels=label_channels,
                                      unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(np.size(inds), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(np.size(inds), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = self.images_to_levels(all_anchors,
                                             self.num_level_anchors)
        labels_list = self.images_to_levels(all_labels, self.num_level_anchors)
        label_weights_list = self.images_to_levels(all_label_weights,
                                                   self.num_level_anchors)
        bbox_targets_list = self.images_to_levels(all_bbox_targets,
                                                  self.num_level_anchors)
        bbox_weights_list = self.images_to_levels(all_bbox_weights,
                                                  self.num_level_anchors)
        keypoints_targets_list = self.images_to_levels(all_keypoints_targets,
                                                       self.num_level_anchors)
        keypoints_weights_list = self.images_to_levels(all_keypoints_weights,
                                                       self.num_level_anchors)
        num_total_samples = np.mean(num_total_pos, dtype=np.float32)
        num_total_samples = max(num_total_samples, 1.0)
        return anchors_list, labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, keypoints_targets_list, keypoints_weights_list, num_total_samples
        # return anchors_list_0, anchors_list_1, anchors_list_2, labels_list_0, labels_list_1, labels_list_2, label_weights_list_0, label_weights_list_1, label_weights_list_2, bbox_targets_list_0, bbox_targets_list_1, bbox_targets_list_2, bbox_weights_list_0, bbox_weights_list_1, bbox_weights_list_2, keypoints_targets_list_0, keypoints_targets_list_1, keypoints_targets_list_2, keypoints_weights_list_0, keypoints_weights_list_1, keypoints_weights_list_2, num_total_samples

    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            num_level_anchors,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_keypoints,
                            gt_labels,
                            label_channels=1,
                            unmap_outputs=True):
        img_shape = self.pad_shape
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags, img_shape,
                                           self.train_cfg["allowed_border"])
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]
        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        valid_mask = np.all(np.isfinite(gt_bboxes), axis=-1)
        gt_bboxes = gt_bboxes[valid_mask].reshape([-1, 4])

        gt_bboxes_ignore = gt_bboxes_ignore[valid_mask[:, 0]]
        gt_labels = gt_labels[valid_mask[:, 0]]
        gt_keypoints = gt_keypoints[valid_mask[:, 0]]
        assign_result = self.assigner.assign(anchors, num_level_anchors_inside,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors, gt_bboxes)
        num_valid_anchors = anchors.shape[0]
        bbox_targets = np.zeros_like(anchors)
        bbox_weights = np.zeros_like(anchors)

        kps_targets = np.zeros(shape=(anchors.shape[0], self.NK * 2))
        kps_weights = np.zeros(shape=(anchors.shape[0], self.NK * 2))
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
            if self.use_kps:
                pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds
                #print('BBB', anchors.shape, gt_bboxes.shape, gt_keypointss.shape, pos_inds.shape, bbox_targets.shape, pos_bbox_targets.shape)
                kps_targets[pos_inds, :] = gt_keypoints[
                    pos_assigned_gt_inds, :, :2].reshape((-1, self.NK * 2))
                kps_weights[pos_inds, :] = np.mean(
                    gt_keypoints[pos_assigned_gt_inds, :, 2],
                    axis=1,
                    keepdims=True)
            #kps_weights[pos_inds, :] = 1.0
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
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
            if self.use_kps:
                kps_targets = unmap(kps_targets, num_total_anchors,
                                    inside_flags)
                kps_weights = unmap(kps_weights, num_total_anchors,
                                    inside_flags)
        return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                kps_targets, kps_weights, pos_inds, neg_inds)

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
        # return level_targets
        return np.concatenate(level_targets, axis=1)

    @tf.function
    def get_anchors(self, batch, featmap_sizes):
        multi_level_anchors, num_level_anchors = self.anchor_generator.grid_anchors(
            batch, featmap_sizes)
        multi_level_flags = self.anchor_generator.valid_flags(
            batch, featmap_sizes, self.pad_shape)
        return multi_level_anchors, num_level_anchors, multi_level_flags

    # @tf.function
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
            anchors (Tensor): Anchor list with shape (N, 4), "xyxy" format.

        Returns:
            Tensor: Anchor centers with shape (N, 2), "xy" format.
        """
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        return tf.stack([anchors_cx, anchors_cy], axis=-1)
