import tensorflow as tf
from ..utils.conv_module import *
from ..backbone.hardnet import *
from ..utils import ChannelAttention, SelfAttention, PositionEmbeddingSine
from pprint import pprint

conv_mode = 'sp_conv2d'


class FPN(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super(FPN, self).__init__(**kwargs)
        self.config = config
        self.structure = self.config.structure
        up_transi_lists = [
            224 + self.SC[0], 96 + self.SC[1], 64 + self.SC[2], 32 + self.SC[3]
        ]
        self.skip_lv = len(self.SC) - 1
        self.conv1x1_ups = []
        for i in range(4):
            self.conv1x1_ups.append(
                ConvBlock(up_transi_lists[i],
                          kernel_size=1,
                          use_bias=False,
                          conv_mode=conv_mode,
                          name='up_trans{}'.format(i + 1)))

        self.transitionUp = TransitionUp(name='up_last_trans{}'.format(i + 1))
        self.avg_pool_concat = AvgPoolConcat()
        up_filters = [388, 232, 54, 48]
        self.transpose_up_layers = []
        for i in range(4):
            self.transpose_up_layers.append(
                TransposeUp(filters=up_filters[i],
                            scale=2,
                            norm_method="bn",
                            activation="relu"))
        self.final_up = TransposeUp(filters=self.structure.inter_ch,
                                    scale=2,
                                    norm_method="bn",
                                    activation="relu")

    @tf.function
    def call(self, inputs):
        x, skip_connections = inputs[0], inputs[1]
        sc_keys = list(skip_connections.keys())[::-1]
        for i in range(4):
            x = self.conv1x1_ups[i](x)
            x = self.avg_pool_concat(x)
            skip = skip_connections[sc_keys[i]]
            x = self.transpose_up_layers[i](inputs=x,
                                            skip=skip,
                                            concat=i < self.skip_lv)
        x = self.final_up(x)
        return x
