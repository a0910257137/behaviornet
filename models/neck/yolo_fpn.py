import tensorflow as tf
from ..utils.conv_module import *
from pprint import pprint

conv_mode = 'sp_conv2d'


class C3(tf.keras.layers.Layer):
    def __init__(self, out_channels, n=1, shortcut=True, **kwargs):
        super(C3, self).__init__(**kwargs)
        self.out_channels = out_channels
        self.branch_channels = self.out_channels // 2
        self.shortcut = shortcut
        # self.initial_layer = tf.keras.initializers.RandomNormal(mean=0,
        #                                                         stddev=.1)

        self.initial_layer = tf.keras.initializers.HeNormal()
        self.conv_1 = ConvBlock(filters=self.branch_channels,
                                kernel_size=1,
                                strides=1,
                                norm_method='bn',
                                use_bias=False,
                                kernel_initializer=self.initial_layer,
                                activation='swish')
        self.conv_2 = ConvBlock(filters=self.branch_channels,
                                kernel_size=1,
                                strides=1,
                                norm_method='bn',
                                use_bias=False,
                                kernel_initializer=self.initial_layer,
                                activation='swish')
        self.mixed_conv_1 = ConvBlock(filters=self.out_channels,
                                      kernel_size=1,
                                      strides=1,
                                      norm_method='bn',
                                      use_bias=False,
                                      kernel_initializer=self.initial_layer,
                                      activation='swish')
        # TODO: bottle neck
        self.bottleneck = tf.keras.Sequential(name='bottleneck')
        self.bottleneck.add(
            ConvBlock(filters=self.branch_channels,
                      kernel_size=1,
                      strides=1,
                      norm_method='bn',
                      use_bias=False,
                      kernel_initializer=self.initial_layer,
                      activation='swish'))
        self.bottleneck.add(
            ConvBlock(filters=self.branch_channels,
                      kernel_size=3,
                      strides=1,
                      norm_method='bn',
                      use_bias=False,
                      conv_mode=conv_mode,
                      kernel_initializer=self.initial_layer,
                      activation='swish'))

    def call(self, x):

        x = tf.concat([self.conv_1(x), self.conv_2(x)], axis=-1)
        x = self.mixed_conv_1(x)
        # shrink dimensions
        identity = x
        c2 = identity.get_shape().as_list()[-1]
        x = self.bottleneck(x)
        c1 = x.get_shape().as_list()[-1]
        if self.shortcut and c1 == c2:
            x = identity + x
        return x


class YFPN(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super(YFPN, self).__init__(**kwargs)
        self.config = config
        up_filters = [128, 128, 256]
        self._base_up = []
        self._merge_layers = []
        self._pro_layers = []
        for i in range(3):
            tmp_layers = [
                ConvBlock(filters=up_filters[i],
                          kernel_size=1,
                          strides=1,
                          norm_method='bn',
                          use_bias=False,
                          kernel_initializer=tf.keras.initializers.HeNormal(),
                          activation='swish',
                          name="stage_{}_CBS_{}".format(4 - i, i)),
                TransitionUp(name="stage_{}_skip_concat_{}".format(4 - i, i)),
                C3(up_filters[i], name="stage_{}_C3_{}".format(4 - i, i))
            ]
            self._base_up.append(tmp_layers)
            if i > 0:
                self._merge_ops = tf.keras.Sequential(name='merge_layer')
                self._merge_ops.add(
                    C3(up_filters[i], name="merge_{}_C3".format(i + 1)))
                self._merge_ops.add(
                    ConvBlock(
                        filters=up_filters[i],
                        kernel_size=3,
                        strides=2,
                        norm_method='bn',
                        use_bias=False,
                        conv_mode=conv_mode,
                        kernel_initializer=tf.keras.initializers.HeNormal(),
                        activation='swish',
                        name="merge_down{}".format((i + 1)),
                    ))

                self._merge_layers.append(self._merge_ops)
                self._pro_layers.append(
                    C3(256, name="merge_{}_C3".format(i + 1)))

    @tf.function
    def call(self, inputs):
        x, skip_connections = inputs
        tmp = []
        outputs = []
        skip_keys = list(skip_connections.keys())[::-1]
        for i, op_layers in enumerate(self._base_up):
            for j, op_layer in enumerate(op_layers):
                if j == 0:
                    x = op_layer(x)
                    # keep items
                    cbs = x
                    tmp.append(cbs)
                    if i > 0:
                        copied_x = tmp[1]
                        merge_x = self._merge_layers[i - 1](tmp[1])
                        merge_x = tf.concat([tmp[0], merge_x], axis=-1)
                        outputs.append(self._pro_layers[i - 1](merge_x))
                        tmp = [copied_x]
                elif j == 1:
                    x = op_layer(inputs=x,
                                 up_method="nearest",
                                 skip=skip_connections[skip_keys[i + 1]],
                                 concat=True)
                else:
                    x = op_layer(x)
        outputs.append(x)
        return outputs