import tensorflow as tf
from ..utils import ConvBlock, DepthwiseSeparableConv
from ..loss.core.anchor_generator import AnchorGenerator
from pprint import pprint
from functools import partial
import tensorflow_addons as tfa

conv_mode = 'sp_conv2d'
norm_method = 'bn'

norma_init_layer = tf.keras.initializers.RandomNormal


class TaskDecomposition(tf.keras.layers.Layer):

    def __init__(self,
                 feat_channels,
                 stacked_convs,
                 la_down_rate=8,
                 name="TDC"):
        super(TaskDecomposition, self).__init__(name=name)
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.in_channels = self.feat_channels * self.stacked_convs

        self.la_conv1 = ConvBlock(filters=self.in_channels // la_down_rate,
                                  kernel_size=1,
                                  strides=1,
                                  norm_method=None,
                                  activation="relu")
        self.la_conv2 = ConvBlock(self.stacked_convs,
                                  kernel_size=1,
                                  strides=1,
                                  norm_method=None,
                                  activation='sigmoid')

        self.reduction_weights = self.add_weight(
            name='reduction_weights',
            shape=(1, self.feat_channels, self.stacked_convs,
                   self.feat_channels),
            initializer=norma_init_layer(mean=0.0, stddev=0.01),
            trainable=True)
        self.adap_avg_pool = tf.keras.layers.GlobalAvgPool2D()
        self.bn_norm = tf.keras.layers.BatchNormalization(name='bn')

        self.relu = tf.keras.layers.Activation(activation='relu',
                                               name='act_relu')

    def __call__(self, x, avg_x=None):
        b, h, w, c = x.get_shape().as_list()
        if avg_x is None:
            avg_x = self.adap_avg_pool(x)
        weight = self.la_conv1(avg_x)
        weight = self.la_conv2(weight)
        # here we first compute the product between layer attention weight and conv weight,
        # and then compute the convolution between new conv weight and feature map,
        # in order to save memory and FLOPs.
        conv_weight = tf.reshape(
            weight, (-1, 1, self.stacked_convs, 1)) * self.reduction_weights
        conv_weight = tf.reshape(conv_weight,
                                 (-1, self.feat_channels, self.in_channels))
        x = tf.reshape(x, (-1, h * w, self.in_channels))
        # B, C, H, W
        x = tf.reshape(tf.matmul(conv_weight, x, transpose_b=(0, 2, 1)),
                       (-1, self.feat_channels, h, w))
        # B, H, W, C
        x = tf.transpose(x, (0, 2, 3, 1))
        # gn?
        x = self.bn_norm(x)
        x = self.relu(x)
        return x


class TOODHead(tf.keras.Model):

    def __init__(self, config, *args, **kwargs):
        super(TOODHead, self).__init__(*args, **kwargs)
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
        self.params_share = self.config.params_share
        self.scale_mode = self.config.scale_mode
        self.use_dfl = True
        self.dw_conv = self.config.dw_conv
        self.loss_dfl = self.config.loss_dfl
        if self.loss_dfl is None or not self.loss_dfl:
            self.use_dfl = False
        self.use_scale = False
        self.use_params = True
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
        self.num_anchors = 2
        self.sigmoid_activation = tf.keras.layers.Activation(
            activation='sigmoid', name='act_sigmoid')
        self.__init_layers()

    # @tf.function
    def call(self, x):
        pred_branches = {"multi_lv_feats": []}
        for xx, scale, stride in zip(x, self.scales,
                                     self.anchor_generator.strides):
            pred_branches['multi_lv_feats'].append(
                tuple(self.single_run(xx, scale, stride)))
            # pred_branches['multi_lv_feats'].append(
            #     tuple(self.single_run(xx, scale, stride)))
        return pred_branches

    def single_run(self, xx, scale, stride):
        b, h, w, c = xx.get_shape().as_list()
        # extract task interactive features
        inter_feats = []
        for i, inter_conv in enumerate(self.inter_convs):
            xx = inter_conv(xx)
            inter_feats.append(xx)

        xx = tf.concat(inter_feats, axis=-1)
        avg_xx = self.adap_avg_pool(xx)
        cls_xx = self.cls_decomp(xx, avg_xx)
        reg_xx = self.reg_decomp(xx, avg_xx)
        # cls prediction and alignment
        cls_logits = self.tood_cls(cls_xx)
        cls_prob = self.cls_prob_conv1(xx)
        cls_prob = self.cls_prob_conv2(cls_prob)
        cls_score = tf.math.sqrt(
            self.sigmoid_activation(cls_logits) *
            self.sigmoid_activation(cls_prob))

        reg_dist = scale * tf.math.exp(self.tood_reg(reg_xx))
        # reg_dist = tf.reshape(reg_dist, (-1, 4))
        # offset for precise bbox
        reg_offset = self.reg_offset_conv1(xx)
        reg_offset = self.reg_offset_conv2(reg_offset)
        return cls_score, reg_dist, reg_offset

    def __init_layers(self):
        self.adap_avg_pool = tf.keras.layers.GlobalAvgPool2D(keepdims=True)
        self.inter_convs = []
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.inter_convs.append(
                ConvBlock(
                    self.feat_channels,
                    kernel_size=3,
                    strides=1,
                    norm_method=None,
                    activation='relu',
                ))

        self.cls_decomp = TaskDecomposition(self.feat_channels,
                                            self.stacked_convs,
                                            self.stacked_convs * 8)
        self.reg_decomp = TaskDecomposition(self.feat_channels,
                                            self.stacked_convs,
                                            self.stacked_convs * 8)
        self.tood_cls = ConvBlock(
            filters=self.num_anchors * self.cls_out_channels,
            kernel_size=3,
            strides=1,
            norm_method=None,
            activation=None,
            kernel_initializer=norma_init_layer(mean=0.0, stddev=0.01))
        self.tood_reg = ConvBlock(self.num_anchors * 4,
                                  kernel_size=3,
                                  strides=1,
                                  norm_method=None,
                                  activation=None,
                                  kernel_initializer=norma_init_layer(
                                      mean=0.0, stddev=0.01))

        self.cls_prob_conv1 = ConvBlock(self.feat_channels // 4,
                                        kernel_size=1,
                                        strides=1,
                                        norm_method=None,
                                        activation='relu',
                                        kernel_initializer=norma_init_layer(
                                            mean=0.0, stddev=0.01))
        self.cls_prob_conv2 = ConvBlock(1,
                                        kernel_size=3,
                                        strides=1,
                                        norm_method=None,
                                        activation=None,
                                        kernel_initializer=norma_init_layer(
                                            mean=0.0, stddev=0.01))

        self.reg_offset_conv1 = ConvBlock(self.feat_channels // 4,
                                          kernel_size=1,
                                          strides=1,
                                          norm_method=None,
                                          activation='relu',
                                          kernel_initializer=norma_init_layer(
                                              mean=0.0, stddev=0.5))

        self.reg_offset_conv2 = ConvBlock(4 * 2,
                                          kernel_size=3,
                                          strides=1,
                                          norm_method=None,
                                          activation=None,
                                          kernel_initializer=norma_init_layer(
                                              mean=0.0, stddev=0.5))
        s0 = tf.Variable(initial_value=1., trainable=True, name="scale_bbox_0")
        s1 = tf.Variable(initial_value=1., trainable=True, name="scale_bbox_1")
        s2 = tf.Variable(initial_value=1., trainable=True, name="scale_bbox_2")
        self.scales = [s0, s1, s2]