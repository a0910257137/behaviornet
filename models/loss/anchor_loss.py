import tensorflow as tf
from .loss_base import LossBase
from pprint import pprint


class AnchorLoss(LossBase):

    def __init__(self, config):
        self.config = config
        self.anchor_generator = self.config["anchor_generator"]
        self.use_sigmoid_cls = True
        self.num_classes = self.config.num_classes

        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1

    def build_loss(self, logits, targets, batch, training):
        # cls_scores,
        # bbox_preds,
        # kps_preds,
        # gt_bboxes,
        # gt_labels,
        # gt_keypointss,
        # img_metas,
        # gt_bboxes_ignore=None

        multi_lv_feats = logits['multi_lv_feats']
        featmap_sizes = [featmap.shape[1:3] for featmap in multi_lv_feats[0]]
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        anchor_list, valid_flag_list = self.get_anchors(batch, featmap_sizes)
        """Get targets for GFL head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        num_imgs = batch
        assert len(anchor_list) == len(valid_flag_list) == num_imgs
        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            gt_keypointss,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        print(num_level_anchors_list)
        xxx
        if cls_reg_targets is None:
            return None
        # with tf.name_scope("losses_collections"):

        return

    def get_anchors(self, batch, featmap_sizes):
        valid_flag_list = []
        multi_level_anchors = self.anchor_generator.grid_anchors(featmap_sizes)
        anchor_list = [multi_level_anchors for _ in range(batch)]
        pad_shape = tf.constant([192, 320, 3], dtype=tf.dtypes.float32)
        for _ in range(batch):
            multi_level_flags = self.anchor_generator.valid_flags(
                featmap_sizes, pad_shape)
            valid_flag_list.append(multi_level_flags)
        return anchor_list, valid_flag_list

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    gt_keypointss_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True):
        """Get targets for GFL head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.shape[0] for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        return cls_reg_targets
