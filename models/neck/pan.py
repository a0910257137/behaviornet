import tensorflow as tf
from ..utils.conv_module import *
# from ..utils import (ChannelAttention, SelfAttention, PositionEmbeddingSine)
from pprint import pprint


class PAN(tf.keras.Model):

    def __init__(self, config, **kwargs):
        super(PAN, self).__init__(**kwargs)
        self.config = config
        self.lateral_ch = self.config.neck.lateral_ch
        self.used_backbone_levels = len(self.lateral_ch)
        self.laterals = []
        self.upsampling_layers = []
        self.downsampling_layers = []
        for i, out_ch in enumerate(self.lateral_ch):
            self.laterals += [
                ConvBlock(filters=out_ch,
                          kernel_size=1,
                          strides=1,
                          name='lateral_conv{}'.format(i),
                          norm_method='bn',
                          activation='hs')
            ]

    @tf.function
    def call(self, inputs):
        x, skip_connections = inputs[0], inputs[1]
        sc_keys = list(skip_connections.keys())
        lateral_x = []
        # build lateral
        for i, k in enumerate(sc_keys):
            sc_x = skip_connections[k]
            lateral_x += [self.laterals[i](sc_x)]
        # build up-sampling part
        for i in range(self.used_backbone_levels - 1, 0, -1):
            size = lateral_x[i - 1].get_shape().as_list()[1:3]
            # lateral_x[i - 1] += self.upsampling_layers[i](lateral_x[i])
            lateral_x[i - 1] += tf.image.resize(lateral_x[i],
                                                size,
                                                method='bilinear',
                                                preserve_aspect_ratio=False,
                                                antialias=False,
                                                name='bilinear_downsampling')
        # build down-sampling part
        for i in range(0, self.used_backbone_levels - 1):
            size = lateral_x[i + 1].get_shape().as_list()[1:3]
            lateral_x[i + 1] += tf.image.resize(lateral_x[i],
                                                size,
                                                method='bilinear',
                                                preserve_aspect_ratio=False,
                                                antialias=False,
                                                name='bilinear_downsampling')
        # latter we use concate as upsampling
        return lateral_x