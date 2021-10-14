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
        self.structure = self.config.neck.structure
        self.h, self.w = 96, 160
        grmul = 1.7
        self.SC = self.structure.skip_conv_ch
        up_n_layers = self.structure.up_n_layers
        up_gr = self.structure.up_growth_layer
        up_transi_lists = [
            224 + self.SC[0], 96 + self.SC[1], 64 + self.SC[2], 32 + self.SC[3]
        ]
        self.skip_lv = len(self.SC) - 1
        self._base_up = []
        self.conv1x1_ups = []
        for i in range(4):
            if i == 0:
                ch = (up_transi_lists[i] - self.SC[i]) * 3
            # blk = HardBlock(ch,
            #                 up_gr[i],
            #                 grmul,
            #                 up_n_layers[i],
            #                 kernel_initializer=tf.keras.initializers.HeUniform,
            #                 name='up_hard_blk{}'.format(i + 1))
            # ch = blk.get_out_ch
            # self._base_up.append(blk)
            self.conv1x1_ups.append(
                ConvBlock(up_transi_lists[i],
                          kernel_size=1,
                          use_bias=False,
                          conv_mode=conv_mode,
                          name='up_trans{}'.format(i + 1)))

        self.transitionUp = TransitionUp(name='up_last_trans{}'.format(i + 1))
        self.avg_pool_concat = AvgPoolConcat()
        self.final_transition_layer = ConvBlock(self.structure.inter_ch * 2,
                                                kernel_size=1,
                                                use_bias=False)
        self.sp_pe = PositionEmbeddingSine(output_dim=248, temperature=248)
        self.self_attention = SelfAttention(248, 'self_attention')
        # self.channel_attention = ChannelAttention('channel_attiontion')
        self.conv_atten = ConvBlock(248, 1, activation=None, norm_method=None)

        up_filters = [388, 232, 54, 48]
        self.transpose_up_layers = []
        for i in range(4):
            self.transpose_up_layers.append(
                TransposeUp(filters=up_filters[i], scale=2))

    @tf.function
    def call(self, inputs):
        x, skip_connections = inputs[0], inputs[1]
        sc_keys = list(skip_connections.keys())[::-1]
        # conv_sc = []
        scs = []
        for i in range(4):
            # channel_atten = self.channel_attention(x)
            # x = self.conv_atten(self_atten + channel_atten)
            skip = skip_connections[sc_keys[i]]
            x = self.transpose_up_layers[i](inputs=x,
                                            skip=skip,
                                            concat=i < self.skip_lv)
            # x = self.transitionUp(inputs=x,
            #   up_method='bilinear',
            #   skip=skip,
            #   concat=i < self.skip_lv)
            x = self.conv1x1_ups[i](x)
            if i == 0:
                sp_mask = tf.ones_like(x, dtype=tf.bool)[..., 0]
                sp_pe = self.sp_pe(sp_mask)
                self_atten = self.self_attention(x, sp_pe)
                x = self.conv_atten(self_atten)
            end = x.get_shape().as_list()[-1]
            # conv_sc.append(x[..., end - self.SC[i]:])
            if self.SC[i] > 0:
                up_conv = tf.image.resize(x[..., end - self.SC[i]:],
                                          (self.h, self.w),
                                          method='bilinear',
                                          preserve_aspect_ratio=False,
                                          antialias=False,
                                          name='bilinear_upsampling')
                scs.append(up_conv)
            x = x[..., :end - self.SC[i]]
            x = self.avg_pool_concat(x)
            # x = self._base_up[i](x)
        scs += [x]
        x = tf.concat(scs, axis=-1)
        x = self.final_transition_layer(x)
        return x
