import tensorflow as tf
from ..utils.conv_module import *
from pprint import pprint

conv_mode = 'sp_conv2d'


class SFPN(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super(SFPN, self).__init__(**kwargs)
        self.config = config
        up_filters = [64] * 4
        self.transpose_up_layers = []
        for i in range(4):
            self.transpose_up_layers.append(
                TransposeUp(filters=up_filters[i],
                            scale=2,
                            norm_method="bn",
                            activation="relu"))
            # if up_idx == 1:
        #     spatial_atten = self_atten_layer(fmap, 128, 'spatial')
        #     channel_atten = channel_atten_layer(fmap, 'channel')
        #     fmap = conv_1(spatial_atten + channel_atten,
        #                   128,
        #                   'atten_conv',
        #                   activation=None,
        #                   norm_method=None)

    @tf.function
    def call(self, inputs):
        x, _ = inputs
        for i in range(4):
            # if i==1:
            # channel_atten = self.channel_attention(x)
            # x = self.conv_atten(self_atten + channel_atten)
            x = self.transpose_up_layers[i](inputs=x, skip=None, concat=False)
        return x