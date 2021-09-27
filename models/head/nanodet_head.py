import tensorflow as tf
from ..utils.conv_module import ConvBlock
from pprint import pprint


class NanoDetHead(tf.keras.Model):
    def __init__(self, config, *args, **kwargs):
        super(NanoDetHead, self).__init__(*args, **kwargs)

        def _build_no_shared_head():
            cls_convs, reg_convs = [], []
            for i in range(self.stacked_convs):
                cls_convs.append(
                    ConvBlock(
                        filters=self.feat_channels,
                        kernel_size=3,
                        strides=1,
                        kernel_initializer=tf.keras.initializers.RandomNormal(
                            mean=0,
                            stddev=0.01,
                        ),
                        activation='hs',
                        conv_mode='sp_conv2d'))

                if not self.share_cls_reg:
                    reg_convs.append(
                        ConvBlock(filters=self.feat_channels,
                                  kernel_size=3,
                                  strides=1,
                                  kernel_initializer=tf.keras.initializers.
                                  RandomNormal(
                                      mean=0,
                                      stddev=0.01,
                                  ),
                                  activation='hs',
                                  conv_mode='sp_conv2d'))

            return cls_convs, reg_convs

        self.config = config
        self.head_cfg = self.config.head
        self.loss_cfg = self.config.loss

        self.stacked_convs = self.head_cfg.stacked_convs
        self.strides = self.head_cfg.strides
        self.reg_max = self.head_cfg.reg_max
        self.share_cls_reg = self.head_cfg.share_cls_reg
        self.feat_channels = 128

        self.use_sigmoid = self.config.loss.loss_qfl.use_sigmoid
        if self.use_sigmoid:
            self.cls_out_channels = self.head_cfg.pred_layer.num_class
        else:
            self.cls_out_channels = self.head_cfg.pred_layer.num_class + 1
        self.cls_convs = []
        self.reg_convs = []
        # implement different init weights
        for _ in self.strides:
            cls_convs, reg_convs = _build_no_shared_head()
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)
        self.gfl_cls = [
            ConvBlock(
                filters=self.cls_out_channels + 4 * (self.reg_max + 1)
                if self.share_cls_reg else self.cls_out_channels,
                kernel_size=1,
                strides=1,
                kernel_initializer=tf.keras.initializers.RandomNormal(
                    mean=0,
                    stddev=0.01,
                ),
                bias_initializer=tf.keras.initializers.Constant(value=-4.595),
                activation='hs') for _ in self.strides
        ]
        # TODO: if
        self.gfl_reg = [
            ConvBlock(filters=4 * (self.reg_max + 1),
                      kernel_size=1,
                      strides=1,
                      kernel_initializer=tf.keras.initializers.RandomNormal(
                          mean=0,
                          stddev=0.01,
                      ),
                      activation='hs') for _ in self.strides
        ]

    @tf.function
    def call(self, feats):
        layer_ouputs = {}
        for i, (feat, cls_convs, reg_convs, gfl_cls, gfl_reg) in enumerate(
                zip(feats, self.cls_convs, self.reg_convs, self.gfl_cls,
                    self.gfl_reg)):
            layer_ouputs['layer_output_{}'.format(i)] = self.run_single(
                feat, cls_convs, reg_convs, gfl_cls, gfl_reg)
        return layer_ouputs

    def run_single(self, x, cls_convs, reg_convs, gfl_cls, gfl_reg):
        cls_feat = x
        reg_feat = x
        for cls_conv in cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in reg_convs:
            reg_feat = reg_conv(reg_feat)
        if self.share_cls_reg:
            feat = gfl_cls(cls_feat)
            cls_score, bbox_pred = tf.split(
                feat, [self.cls_out_channels, 4 * (self.reg_max + 1)], axis=-1)
        else:
            cls_score = gfl_cls(cls_feat)
            bbox_pred = gfl_reg(reg_feat)
        print(bbox_pred)
        xxx
        return {'cls_scores': cls_score, 'bbox_pred': bbox_pred}
