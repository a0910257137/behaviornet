import tensorflow as tf
from ..utils import ConvBlock, DepthwiseSeparableConv
from ..loss.core.anchor_generator import AnchorGenerator
from pprint import pprint
from functools import partial
import tensorflow_addons as tfa

conv_mode = 'sp_conv2d'
norm_method = 'bn'


class SCRFDHead(tf.keras.Model):

    def __init__(self, config, *args, **kwargs):
        super(SCRFDHead, self).__init__(*args, **kwargs)

        self.config = config
        self.anchor_generator = self.config["anchor_generator"]
        self.feat_channels = self.config.feat_channels
        self.in_channels = self.config.in_channels
        self.num_classes = self.config.num_classes

        self.stacked_convs = self.config.stacked_convs
        self.feat_mults = None
        self.conv_cfg = None
        self.norm_cfg = self.config.norm_cfg
        self.reg_max = 8
        self.cls_reg_share = self.config.cls_reg_share
        self.strides_share = self.config.strides_share
        self.scale_mode = self.config.scale_mode
        self.use_dfl = True
        self.dw_conv = self.config.dw_conv
        self.NK = 5
        self.extra_flops = 0.0
        self.loss_dfl = self.config.loss_dfl

        if self.loss_dfl is None or not self.loss_dfl:
            self.use_dfl = False
        self.use_scale = False
        self.use_kps = self.config.use_kps
        if self.scale_mode > 0 and (self.strides_share or self.scale_mode == 2):
            self.use_scale = True
        self.use_sigmoid_cls = True
        # TODO better way to determine whether sample or not
        self.sampling = False

        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1

        if self.cls_out_channels <= 0:
            raise ValueError(f'num_classes={self.num_classes} is too small')
        #print('USE-SCALE:', self.use_scale)
        self.reg_decoded_bbox = False
        # hard code
        self.num_anchors = 2
        self.__init_layers()

    @tf.function
    def call(self, x):
        pred_branches = {
            "multi_lv_feats":
            self.multi_apply(self.single_run, x, self.scales,
                             self.anchor_generator.strides)
        }
        return pred_branches

    def single_run(self, x, scale, stride):
        cls_feat = x
        reg_feat = x
        #print('forward_single in stride:', stride)
        cls_convs = self.cls_stride_convs[
            '0'] if self.strides_share else self.cls_stride_convs[str(stride)]

        for cls_conv in cls_convs:
            cls_feat = cls_conv(cls_feat)

        if not self.cls_reg_share:
            reg_convs = self.reg_stride_convs[
                '0'] if self.strides_share else self.reg_stride_convs[str(
                    stride)]

            for reg_conv in reg_convs:
                reg_feat = reg_conv(reg_feat)
        else:
            reg_feat = cls_feat
        cls_pred_module = self.stride_cls[
            '0'] if self.strides_share else self.stride_cls[str(stride)]
        cls_score = cls_pred_module(cls_feat)
        reg_pred_module = self.stride_reg[
            '0'] if self.strides_share else self.stride_reg[str(stride)]
        _bbox_pred = reg_pred_module(reg_feat)

        if self.use_scale:
            bbox_pred = scale * _bbox_pred
        else:
            bbox_pred = _bbox_pred

        if self.use_kps:
            kps_pred_module = self.stride_kps[
                '0'] if self.strides_share else self.stride_kps[str(stride)]
            kps_pred = kps_pred_module(reg_feat)
        else:
            kps_pred = tf.zeros(shape=(bbox_pred.shape[0], bbox_pred.shape[1],
                                       bbox_pred.shape[2], self.NK * 2))
        return cls_score, bbox_pred, kps_pred

    def multi_apply(self, func, *args, **kwargs):
        """Apply function to a list of arguments.

        Note:
            This function applies the ``func`` to multiple inputs and
            map the multiple outputs of the ``func`` into different
            list. Each list contains the same type of outputs corresponding
            to different inputs.

        Args:
            func (Function): A function that will be applied to a list of
                arguments

        Returns:
            tuple(list): A tuple containing multiple list, each list contains \
                a kind of returned results by the function
        """
        pfunc = partial(func, **kwargs) if kwargs else func
        map_results = map(pfunc, *args)
        return tuple(map(list, zip(*map_results)))

    def _get_conv_module(self, in_channel, out_channel):

        if not self.dw_conv:
            conv = ConvBlock(filters=out_channel,
                             kernel_size=3,
                             strides=1,
                             groups=self.norm_cfg.num_groups,
                             norm_method=self.norm_cfg.type.lower())
        else:
            conv = DepthwiseSeparableConv(
                in_channel=in_channel,
                out_channel=out_channel,
                kernel_size=3,
                strides=1,
                groups=self.norm_cfg.num_groups,
                use_bias=False,
                norm_method=self.norm_cfg.type.lower())

        return conv

    def __init_layers(self):
        """Initialize layers of the head."""
        self.relu = tf.keras.layers.Activation(activation='relu')
        conv_strides = [
            0
        ] if self.strides_share else self.anchor_generator.strides
        self.cls_stride_convs, self.reg_stride_convs = {}, {}
        self.stride_cls, self.stride_reg = {}, {}

        if self.use_kps:
            self.stride_kps = {}

        for stride_idx, conv_stride in enumerate(conv_strides):
            #print('create convs for stride:', conv_stride)
            key = str(conv_stride)
            cls_convs, reg_convs = [], []
            stacked_convs = self.stacked_convs[stride_idx] if isinstance(
                self.stacked_convs, (list, tuple)) else self.stacked_convs
            feat_mult = self.feat_mults[
                stride_idx] if self.feat_mults is not None else 1

            feat_ch = int(self.feat_channels * feat_mult)
            for i in range(stacked_convs):
                chn = self.in_channels if i == 0 else last_feat_ch

                cls_convs.append(self._get_conv_module(chn, feat_ch))

                if not self.cls_reg_share:
                    reg_convs.append(self._get_conv_module(chn, feat_ch))
                last_feat_ch = feat_ch

            self.cls_stride_convs[key] = cls_convs
            self.reg_stride_convs[key] = reg_convs

            self.stride_cls[key] = ConvBlock(filters=self.cls_out_channels *
                                             self.num_anchors,
                                             kernel_size=3,
                                             strides=1)
            if not self.use_dfl:
                self.stride_reg[key] = ConvBlock(filters=4 * self.num_anchors,
                                                 kernel_size=3,
                                                 strides=1)
            else:
                self.stride_reg[key] = ConvBlock(
                    filters=4 * (self.reg_max + 1) * self.num_anchors,
                    kernel_size=3,
                    strides=1)
            if self.use_kps:
                self.stride_kps[key] = ConvBlock(filters=self.NK * 2 *
                                                 self.num_anchors,
                                                 kernel_size=3,
                                                 strides=1)

        #assert self.num_anchors == 1, 'anchor free version'
        #extra_gflops /= 1e9
        #print('extra_gflops: %.6fG'%extra_gflops)
        if self.use_scale:
            self.scales = [1.0 for _ in self.anchor_generator.strides]
        else:
            self.scales = [None for _ in self.anchor_generator.strides]