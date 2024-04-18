import tensorflow as tf
from ..utils import ConvBlock
from pprint import pprint
from functools import partial

act = "relu"
conv_mode = 'conv2d'
norm_method = 'bn'

norma_init_layer = tf.keras.initializers.RandomNormal


class SCRFDHead(tf.keras.Model):

    def __init__(self, config, *args, **kwargs):
        super(SCRFDHead, self).__init__(*args, **kwargs)

        self.config = config
        self.anchor_generator = self.config["anchor_generator"]
        self.feat_channels = self.config.feat_channels
        self.head_branch = self.config.head_branch
        self.in_channels = self.config.in_channels
        self.num_classes = self.config.num_classes
        self.stacked_convs = self.config.stacked_convs
        self.feat_mults = None
        self.norm_cfg = self.config.norm_cfg
        self.reg_max = 8
        self.cls_reg_share = self.config.cls_reg_share
        self.strides_share = self.config.strides_share
        self.params_share = self.config.params_share
        self.scale_mode = self.config.scale_mode
        self.use_dfl = True
        self.dw_conv = self.config.dw_conv
        self.loss_dfl = self.config.loss_dfl
        if self.loss_dfl is None or not self.loss_dfl:
            self.use_dfl = False
        self.use_scale = False
        self.use_params = True
        if self.scale_mode > 0 and (self.strides_share
                                    or self.scale_mode == 2):
            self.use_scale = True
        self.use_sigmoid_cls = True

        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1

        if self.cls_out_channels <= 0:
            raise ValueError(f'num_classes={self.num_classes} is too small')
        self.num_anchors = 2
        self.__init_layers()

    # @tf.function
    def call(self, x):
        pred_branches = {"multi_lv_feats": []}
        for i, (xx, scale, stride) in enumerate(
                zip(x, self.scales, self.anchor_generator.strides)):
            if self.head_branch != None and i != self.head_branch:
                continue
            pred_branches['multi_lv_feats'].append(
                tuple(self.single_run(xx, scale, stride)))
        return pred_branches

    def single_run(self, x, scale, stride):
        cls_feat, reg_feat = tf.identity(x), tf.identity(x)
        param_feat, kp_feat = tf.identity(x), tf.identity(x)
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
        # parameters
        if self.use_params:
            param_convs = self.params_stride_convs[
                '0'] if self.strides_share else self.params_stride_convs[str(
                    stride)]
            for param_conv in param_convs:
                param_feat = param_conv(param_feat)
            param_pred_module = self.stride_params[
                '0'] if self.params_share else self.stride_params[str(stride)]
            param_pred = param_pred_module(param_feat)
            kp_convs = self.kps_stride_convs[
                '0'] if self.strides_share else self.kps_stride_convs[str(
                    stride)]
            for kp_conv in kp_convs:
                kp_feat = kp_conv(kp_feat)
            kps_pred_module = self.stride_kps[
                '0'] if self.params_share else self.stride_kps[str(stride)]
            kps_pred = kps_pred_module(kp_feat)
        if self.use_scale:
            bbox_pred = scale * _bbox_pred
        else:
            bbox_pred = _bbox_pred
        if self.use_params:
            return cls_score, bbox_pred, param_pred, kps_pred
        return cls_score, reg_feat, x

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

    def _get_conv_module(self, out_channel, name):
        conv = ConvBlock(filters=out_channel,
                         kernel_size=3,
                         strides=1,
                         norm_method=self.norm_cfg.type.lower(),
                         activation=act,
                         kernel_initializer=norma_init_layer(mean=0.0,
                                                             stddev=0.01),
                         name=name)

        return conv

    def __init_layers(self):
        """Initialize layers of the head."""
        conv_strides = [
            0
        ] if self.strides_share else self.anchor_generator.strides
        self.cls_stride_convs, self.reg_stride_convs, self.params_stride_convs, self.kps_stride_convs = {}, {}, {}, {}
        self.stride_cls, self.stride_reg, self.stride_params, self.stride_kps = {}, {}, {}, {}

        for stride_idx, conv_stride in enumerate(conv_strides):
            #print('create convs for stride:', conv_stride)
            if self.head_branch != None and stride_idx != self.head_branch:
                continue
            key = str(conv_stride)
            cls_convs, reg_convs, params_convs, kps_convs = [], [], [], []
            stacked_convs = self.stacked_convs[stride_idx] if isinstance(
                self.stacked_convs, (list, tuple)) else self.stacked_convs
            feat_mult = self.feat_mults[
                stride_idx] if self.feat_mults is not None else 1
            feat_ch = int(self.feat_channels * feat_mult)
            for i in range(stacked_convs):
                chn = self.in_channels if i == 0 else last_feat_ch
                cls_convs.append(
                    self._get_conv_module(
                        feat_ch, 'conv3x3_cls_{}_{}'.format(stride_idx, i)))
                if not self.cls_reg_share:
                    reg_convs.append(
                        self._get_conv_module(
                            feat_ch,
                            'conv3x3_bbox_{}_{}'.format(stride_idx, i)))
                if self.use_params:
                    params_convs.append(
                        self._get_conv_module(
                            feat_ch,
                            'conv3x3_params_{}_{}'.format(stride_idx, i)))
                    kps_convs.append(
                        self._get_conv_module(
                            feat_ch, 'conv3x3_kps_{}_{}'.format(stride_idx,
                                                                i)))
                last_feat_ch = feat_ch
            self.cls_stride_convs[key] = cls_convs
            self.reg_stride_convs[key] = reg_convs
            self.params_stride_convs[key] = params_convs
            self.kps_stride_convs[key] = kps_convs
            self.stride_cls[key] = ConvBlock(
                filters=self.cls_out_channels * self.num_anchors,
                kernel_size=3,
                strides=1,
                norm_method=None,
                activation=None,
                kernel_initializer=norma_init_layer(mean=0.0, stddev=0.01),
                bias_initializer=tf.constant_initializer(value=-4.595),
                name='pred_cls_{}'.format(stride_idx))
            if not self.use_dfl:
                self.stride_reg[key] = ConvBlock(
                    filters=4 * self.num_anchors,
                    kernel_size=3,
                    strides=1,
                    norm_method=None,
                    activation=None,
                    kernel_initializer=norma_init_layer(mean=0.0, stddev=0.01),
                    name='pred_bbox_{}'.format(stride_idx))
            else:
                self.stride_reg[key] = ConvBlock(
                    filters=4 * (self.reg_max + 1) * self.num_anchors,
                    kernel_size=3,
                    strides=1,
                    norm_method=None,
                    activation=None,
                    conv_mode='conv2d',
                    kernel_initializer=norma_init_layer(mean=0.0, stddev=0.01),
                    name='pred_bbox_{}'.format(stride_idx))

            if self.use_params:
                self.stride_params[key] = ConvBlock(
                    filters=60 * self.num_anchors,
                    kernel_size=1,
                    strides=1,
                    norm_method=None,
                    activation=None,
                    conv_mode='conv2d',
                    kernel_initializer=norma_init_layer(mean=0.0001,
                                                        stddev=0.1),
                    name='pred_params_{}'.format(stride_idx))

                self.stride_kps[key] = ConvBlock(
                    filters=2 * self.num_anchors,
                    kernel_size=1,
                    strides=1,
                    norm_method=None,
                    activation=None,
                    conv_mode='conv2d',
                    kernel_initializer=norma_init_layer(mean=0.0, stddev=0.1),
                    name='pred_kp_{}'.format(stride_idx))
        if self.use_scale:
            s0 = tf.Variable(initial_value=1.,
                             trainable=True,
                             name="scale_bbox_0")
            s1 = tf.Variable(initial_value=1.,
                             trainable=True,
                             name="scale_bbox_1")
            s2 = tf.Variable(initial_value=1.,
                             trainable=True,
                             name="scale_bbox_2")
            self.scales = [s0, s1, s2]
        else:
            self.scales = [None for _ in self.anchor_generator.strides]
