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
        multi_level_anchors = self.anchor_generator.grid_anchors(featmap_sizes)
        anchor_list = [multi_level_anchors for _ in range(batch)]
        valid_flag_list = []

        pad_shape = tf.constant([192, 320, 3], dtype=tf.dtypes.float32)

        for _ in range(batch):
            multi_level_flags = self.anchor_generator.valid_flags(
                featmap_sizes, pad_shape)
            valid_flag_list.append(multi_level_flags)
        # with tf.name_scope("losses_collections"):

        return

    def grid_anchors(self, featmap_sizes, anchor_generator):
        """
            Generate grid anchors in multiple feature levels.
            Args:
                featmap_sizes (list[tuple]): List of feature map sizes in
                    multiple feature levels.
                device (str): Device where the anchors will be put on.

            Return:
                list[torch.Tensor]: Anchors in multiple feature levels. \
                    The sizes of each tensor should be [N, 4], where \
                    N = width * height * num_base_anchors, width and height \
                    are the sizes of the corresponding feature level, \
                    num_base_anchors is the number of anchors for that level.
        """
        num_levels = self.anchor_generator.num_levels
        base_anchors = anchor_generator.base_anchors
        strides = anchor_generator.strides
        assert num_levels == len(featmap_sizes)
        multi_level_anchors = []
        print(num_levels)
        xxx
        for i in range(num_levels):

            print(base_anchors[i])
            print('-' * 100)
            print(featmap_sizes[i])

            print(strides[i])
            xxx
            anchors = self.single_level_grid_anchors(base_anchors[i],
                                                     featmap_sizes[i],
                                                     strides[i])
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def single_level_grid_anchors(self,
                                  base_anchors,
                                  featmap_size,
                                  stride=(16, 16),
                                  device='cuda'):
        """Generate grid anchors of a single level.

        Note:
            This function is usually called by method ``self.grid_anchors``.

        Args:
            base_anchors (torch.Tensor): The base anchors of a feature grid.
            featmap_size (tuple[int]): Size of the feature maps.
            stride (tuple[int], optional): Stride of the feature map in order
                (w, h). Defaults to (16, 16).

        Returns:
            torch.Tensor: Anchors in the overall feature maps.
        """
        feat_h, feat_w = featmap_size
        # convert Tensor to int, so that we can covert to ONNX correctlly
        feat_h = int(feat_h)
        feat_w = int(feat_w)
        shift_x = torch.arange(0, feat_w, device=device) * stride[0]
        shift_y = torch.arange(0, feat_h, device=device) * stride[1]

        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts = shifts.type_as(base_anchors)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors