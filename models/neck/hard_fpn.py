import tensorflow as tf
from ..backbone.hardnet import *
from ..utils.conv_module import *
from pprint import pprint


class HFPN(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super(HFPN, self).__init__(**kwargs)
        self.config = config
        self.structure = self.config.neck.structure
        grmul = 1.6
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
            blk = HardBlock(ch,
                            up_gr[i],
                            grmul,
                            up_n_layers[i],
                            kernel_initializer=tf.keras.initializers.HeUniform,
                            name='up_hard_blk{}'.format(i + 1))
            ch = blk.get_out_ch
            self._base_up.append(blk)
            self.conv1x1_ups.append(
                ConvBlock(up_transi_lists[i],
                          kernel_size=1,
                          use_bias=False,
                          name='up_trans{}'.format(i + 1)))

        self.transitionUp = TransitionUp(name='up_last_trans{}'.format(i + 1))
        self.avg_pool_concat = AvgPoolConcat()
        self.final_up = TransposeUp(filters=self.structure.inter_ch * 2,
                                    scale=2,
                                    norm_method="bn",
                                    activation="relu")

    @tf.function
    def call(self, inputs):
        x, skip_connections = inputs[0], inputs[1]
        sc_keys = list(skip_connections.keys())[::-1]
        conv_sc = []
        for i in range(4):
            skip = skip_connections[sc_keys[i]]
            x = self.transitionUp(x, "bilinear", skip, concat=i < self.skip_lv)
            x = self.conv1x1_ups[i](x)
            end = x.get_shape().as_list()[-1]
            conv_sc.append(x[..., end - self.SC[i]:])
            x = x[..., :end - self.SC[i]]
            x = self.avg_pool_concat(x)
            x = self._base_up[i](x)
        scs = [x]
        up_h, up_w = x.get_shape().as_list()[1:3]
        for i in range(len(self.SC)):
            if self.SC[i] > 0:
                up_conv = tf.image.resize(conv_sc[i], (up_h, up_w),
                                          method='bilinear',
                                          preserve_aspect_ratio=False,
                                          antialias=False,
                                          name='bilinear_upsampling')
                scs.append(up_conv)
        x = tf.concat(scs, axis=-1)
        x = self.final_up(x)
        # x = self.final_transition_layer(x)
        return x
