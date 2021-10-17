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
        # self.final_transition_layer = ConvBlock(self.structure.inter_ch * 2,
        #                                         kernel_size=1,
        #                                         use_bias=False)
        # self.sp_pe = PositionEmbeddingSine(output_dim=120, temperature=120)
        # self.self_attention = SelfAttention(120, 'self_attention')
        # self.channel_attention = ChannelAttention('channel_attiontion')
        # self.conv_atten = ConvBlock(120, 1, activation=None, norm_method=None)

        up_filters = [96, 96, 64, 64]
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
            if i == 1:
                sp_mask = tf.ones_like(x, dtype=tf.bool)[..., 0]
                sp_pe = self.sp_pe(sp_mask)
                x = self.self_attention(x, sp_pe)
            x = self.avg_pool_concat(x)
            skip = skip_connections[sc_keys[i]]
            x = self.transpose_up_layers[i](inputs=x,
                                            skip=skip,
                                            concat=i < self.skip_lv)
        x = self.final_up(x)
        return x
