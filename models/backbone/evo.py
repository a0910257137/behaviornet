import imp
import tensorflow as tf
from .kernel_initializers import KernelInitializers
from ..utils.conv_module import ConvBlock, DW, SE
from pprint import pprint

conv_mode = 'sp_conv2d'


class EvoModel(tf.keras.Model):
    def __init__(self, arch, *args, **kwargs):
        super(EvoModel, self).__init__(*args, **kwargs)
        self.stem_dim = 32
        if arch == "base":
            self.settings = [{
                "out_dim": 16,
                "repeat": 1,
                "kernel_size": 3,
                "stride": 1,
                "expand_ratio": 1,
                "id_skip": True,
                "se_ratio": 0.25
            }, {
                "out_dim": 24,
                "repeat": 3,
                "kernel_size": 3,
                "stride": 2,
                "expand_ratio": 6,
                "id_skip": True,
                "se_ratio": 0.25
            }, {
                "out_dim": 40,
                "repeat": 2,
                "kernel_size": 5,
                "stride": 2,
                "expand_ratio": 6,
                "id_skip": True,
                "se_ratio": 0.25
            }, {
                "out_dim": 80,
                "repeat": 4,
                "kernel_size": 3,
                "stride": 2,
                "expand_ratio": 6,
                "id_skip": True,
                "se_ratio": 0.25
            }, {
                "out_dim": 112,
                "repeat": 2,
                "kernel_size": 5,
                "stride": 1,
                "expand_ratio": 6,
                "id_skip": True,
                "se_ratio": 0.25
            }, {
                "out_dim": 192,
                "repeat": 4,
                "kernel_size": 5,
                "stride": 1,
                "expand_ratio": 6,
                "id_skip": True,
                "se_ratio": 0.25
            }, {
                "out_dim": 80,
                "repeat": 2,
                "kernel_size": 3,
                "stride": 1,
                "expand_ratio": 6,
                "id_skip": True,
                "se_ratio": 0.25
            }]

        self.stem_conv = ConvBlock(filters=32,
                                   kernel_size=3,
                                   strides=2,
                                   use_bias=False,
                                   conv_mode=conv_mode,
                                   name="init_conv")
        # self._base_down = tf.keras.Sequential(name='down_sampling')
        self._base_down = []
        for set_idx, setting in enumerate(self.settings):

            if set_idx == 0:
                blk_filters_in = self.stem_dim
            else:
                blk_filters_in = self.settings[set_idx - 1]['out_dim']
            for blk_idx in range(setting['repeat']):
                if blk_idx > 0:
                    strides = 1
                    blk_filters_in = setting['out_dim']
                else:
                    strides = setting['stride']
                blk_name = '%i_res_%i' % (set_idx, blk_idx)
                # self._base_down.add(
                # ResBlk(setting['out_dim'],
                #        name=blk_name,
                #        strides=strides,
                #        kernel_size=setting['kernel_size']))
                self._base_down.append(
                    IRes(filters_in=blk_filters_in,
                         filters_out=setting['out_dim'],
                         se_ratio=setting['se_ratio'],
                         strides=strides,
                         kernel_size=setting['kernel_size'],
                         id_skip=setting['id_skip'],
                         name=blk_name))
                # self._base_down.add()

    def call(self, inputs, training=None):
        x = self.stem_conv(inputs)
        for layer in self._base_down:
            x = layer(x)
        return x, "None"


class IRes(tf.keras.layers.Layer):
    def __init__(self,
                 activation='swish',
                 drop_rate=0.,
                 name='',
                 filters_in=16,
                 filters_out=16,
                 kernel_size=3,
                 strides=1,
                 expand_ratio=1,
                 se_ratio=0.,
                 id_skip=True,
                 project=True,
                 regularizer=None):

        super().__init__()
        filters = int(filters_in * expand_ratio)
        self.filters_in = filters_in
        self.filters_out = filters_out
        self.strides = strides
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.id_skip = id_skip
        self.project = project
        # Expansion phase
        if expand_ratio != 1:
            self.bneck = ConvBlock(filters=filters,
                                   kernel_size=1,
                                   use_bias=False,
                                   norm_method="bn",
                                   activation=activation,
                                   name=name + 'expand_conv')

        self.dw = DW(kernel_size=kernel_size,
                     strides=self.strides,
                     n1=activation)
        # Squeeze and Excitation phase

        if 0 < self.se_ratio <= 1:
            filters_se = max(1, int(filters_in * self.se_ratio))
            self.se_squeeze = tf.keras.layers.GlobalAveragePooling2D(
                name=name + 'se_squeeze')
            self.se_reduce = ConvBlock(filters=filters_se,
                                       kernel_size=1,
                                       strides=1,
                                       activation=activation,
                                       use_bias=False,
                                       name=name + 'se_reduce')
            self.se_expand = ConvBlock(filters=filters,
                                       kernel_size=1,
                                       strides=1,
                                       activation="sigmoid",
                                       use_bias=False,
                                       name=name + 'se_expand')

        # Output phase
        if self.project:
            self.output_phase = ConvBlock(filters=filters_out,
                                          kernel_size=1,
                                          strides=1,
                                          use_bias=False,
                                          norm_method="bn",
                                          activation=None,
                                          name=name + 'project_conv')

    @tf.function
    def call(self, inputs, training=None):
        if self.expand_ratio != 1:
            x = self.bneck(inputs)
        else:
            x = inputs
        x = self.dw(x)
        # squeeze and excitation
        if 0 < self.se_ratio <= 1:
            se = self.se_squeeze(x)
            _, c = se.get_shape().as_list()
            se = tf.reshape(se, [-1, 1, 1, c], name=self.name + 'se_reshape')
            se = self.se_reduce(se)
            se = self.se_expand(se)
            x = tf.keras.layers.multiply([x, se], name=self.name + 'se_excite')
        #output phase
        if self.project:
            x = self.output_phase(x)
        if self.id_skip and self.strides == 1 and self.filters_in == self.filters_out and self.project:
            x = x + inputs
        return x


class ResBlk(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 out_dim,
                 strides=1,
                 rate=1,
                 kernel_size=3,
                 identity_skip=False,
                 use_bottle=False,
                 norm_method="bn",
                 **kwargs):
        super(ResBlk, self).__init__(**kwargs)
        self.identity_skip = identity_skip
        self.use_bottle = use_bottle
        self.kernel_size = kernel_size
        if identity_skip:
            self.skip = tf.identity
        else:
            self.skip = ConvBlock(filters=out_dim,
                                  kernel_size=1,
                                  strides=strides,
                                  use_bias=False,
                                  padding='same',
                                  norm_method="bn",
                                  activation=None)
        if self.use_bottle:
            self.main_path_1k1 = tf.keras.Sequential(name='main_path_1k1')
            self.main_path_1k1.add(
                ConvBlock(filters=out_dim / 2,
                          kernel_size=1,
                          strides=strides,
                          activation='relu',
                          use_bias=False))
            self.main_path_1k1.add(
                ConvBlock(filters=out_dim / 2,
                          kernel_size=self.kernel_size,
                          dilation_rate=2,
                          strides=1,
                          activation='relu',
                          use_bias=False))
            self.main_path_1k1.add(
                ConvBlock(filters=out_dim,
                          kernel_size=1,
                          strides=strides,
                          norm_method="bn",
                          activation=None,
                          use_bias=False))
        else:
            self.main_path_kk = tf.keras.Sequential(name='main_path_kk')
            self.main_path_kk.add(
                ConvBlock(filters=out_dim,
                          kernel_size=self.kernel_size,
                          strides=strides,
                          activation='relu',
                          norm_method="bn",
                          use_bias=False))
            self.main_path_kk.add(
                ConvBlock(filters=out_dim,
                          kernel_size=self.kernel_size,
                          strides=1,
                          activation=None,
                          norm_method="bn",
                          use_bias=False))

    def call(self, data_in, training=None):
        if self.identity_skip:
            skip = self.skip(data_in)
        else:
            skip = tf.identity(data_in)
        if self.use_bottle:
            out = self.main_path_1k1(data_in)
        else:
            out = self.main_path_kk(data_in)
        out = tf.math.add_n([skip, out], name='merge')
        return tf.nn.relu(out)


def Evo(input_shape, pooling, kernel_initializer):
    kernel_initializer = KernelInitializers().get_initializer(
        kernel_initializer)
    evo_model = EvoModel(arch="base")
    image_inputs = tf.keras.Input(shape=input_shape, name='image_inputs')
    fmaps = evo_model(image_inputs)

    return tf.keras.Model(image_inputs, fmaps, name='backbone')
