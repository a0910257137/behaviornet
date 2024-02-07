import tensorflow as tf
from .kernel_initializers import KernelInitializers
from ..utils.conv_module import ConvBlock
from pprint import pprint
import math
from keras_flops import get_flops

BN_MOMENTUM = 0.999
BN_EPSILON = 1e-3


class SEModule(tf.keras.layers.Layer):
    """
    A squeeze and excite module
    """

    def __init__(self, filters, ratio):
        super(SEModule, self).__init__()
        self.pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.reshape = tf.keras.layers.Lambda(self._reshape)

        self.conv1 = ConvBlock(filters=int(filters / ratio),
                               kernel_size=1,
                               strides=1,
                               norm_method=None,
                               activation="relu",
                               use_bias=False)
        self.conv2 = ConvBlock(filters=int(filters),
                               kernel_size=1,
                               strides=1,
                               norm_method=None,
                               activation=None,
                               use_bias=False)
        self.hard_sigmoid = tf.keras.layers.Activation('hard_sigmoid')

    @staticmethod
    def _reshape(x):
        return tf.keras.layers.Reshape((1, 1, int(x.shape[1])))(x)

    @staticmethod
    def _excite(x, excitation):
        """
        Multiply by an excitation factor

        :param x: A Tensorflow Tensor
        :param excitation: A float between 0 and 1
        :return:
        """
        return x * excitation

    def call(self, inputs):
        x = self.reshape(self.pooling(inputs))
        x = self.conv1(x)
        excitation = self.hard_sigmoid(self.conv2(x))
        x = tf.keras.layers.Lambda(self._excite,
                                   arguments={'excitation':
                                              excitation})(inputs)
        return x


class GhostModule(tf.keras.layers.Layer):
    """
    The main Ghost module
    """

    def __init__(self, out, ratio, convkernel, dwkernel):
        super(GhostModule, self).__init__()
        self.ratio = ratio
        self.out = out
        self.conv_out_channel = math.ceil(self.out * 1.0 / ratio)
        self.conv = ConvBlock(filters=int(self.conv_out_channel),
                              kernel_size=convkernel,
                              strides=1,
                              norm_method="bn",
                              activation=None,
                              use_bias=False)

        self.depthconv = ConvBlock(filters=None,
                                   kernel_size=dwkernel,
                                   strides=1,
                                   norm_method=None,
                                   activation=None,
                                   use_bias=False,
                                   conv_mode="dw_conv2d")

        self.slice = tf.keras.layers.Lambda(
            self._return_slices,
            arguments={'channel': int(self.out - self.conv_out_channel)})
        self.concat = tf.keras.layers.Concatenate()

    @staticmethod
    def _return_slices(x, channel):
        return x[:, :, :, :channel]

    def call(self, inputs):
        x = self.conv(inputs)
        if self.ratio == 1:
            return x
        dw = self.depthconv(x)
        dw = self.slice(dw)
        output = self.concat([x, dw])
        return output


class GBNeck(tf.keras.layers.Layer):
    """
    The GhostNet Bottleneck
    """

    def __init__(self, dwkernel, strides, exp, out, ratio, use_se):
        super(GBNeck, self).__init__()
        self.strides = strides
        self.use_se = use_se
        self.conv_layers = []
        self.conv = ConvBlock(filters=out,
                              kernel_size=1,
                              strides=1,
                              norm_method="bn",
                              activation=None,
                              use_bias=False)
        self.depth_conv1 = ConvBlock(filters=None,
                                     kernel_size=dwkernel,
                                     strides=strides,
                                     norm_method="bn",
                                     activation=None,
                                     use_bias=False,
                                     conv_mode="dw_conv2d")

        self.depth_conv2 = ConvBlock(filters=None,
                                     kernel_size=dwkernel,
                                     strides=strides,
                                     norm_method="bn",
                                     activation="relu",
                                     use_bias=False,
                                     conv_mode="dw_conv2d")
        self.ghost1 = GhostModule(exp, ratio, 1, 3)
        self.ghost2 = GhostModule(out, ratio, 1, 3)
        self.bn1 = tf.keras.layers.BatchNormalization(name='bn')
        self.bn2 = tf.keras.layers.BatchNormalization(name='bn')
        self.relu1 = tf.keras.layers.Activation(activation='relu',
                                                name='act_relu')
        self.se = SEModule(exp, ratio)

    def call(self, inputs):
        x = self.depth_conv1(inputs)
        x = self.conv(x)
        y = self.relu1(self.bn1(self.ghost1(inputs)))
        # Extra depth conv if strides higher than 1
        if self.strides > 1:
            y = self.depth_conv2(y)
        # Squeeze and excite
        if self.use_se:
            y = self.se(y)
        y = self.bn2(self.ghost2(y))
        # Skip connection
        return tf.keras.layers.add([x, y])


class GhostNet(tf.keras.Model):

    def __init__(self, config, kernel_initializer, *args, **kwargs):
        super(GhostNet, self).__init__(*args, **kwargs)
        self.config = config
        self.dwkernels = self.config.dwkernels
        self.strides = self.config.strides
        self.exps = self.config.exps
        self.outs = self.config.outs
        self.use_ses = self.config.use_ses
        self.ratios = [2] * 16
        self.stage_layers = []
        input_channel = 16
        self.output_idx = [2, 4, 10, 15]
        self.stem = ConvBlock(filters=input_channel,
                              kernel_size=3,
                              strides=2,
                              use_bias=False,
                              norm_method="bn",
                              activation="relu",
                              name='conv_stem')

        for i, args in enumerate(
                zip(self.dwkernels, self.strides, self.exps, self.outs,
                    self.ratios, self.use_ses)):
            self.stage_layers.append(GBNeck(*args))

    @tf.function
    def call(self, x):
        output = []
        x = self.stem(x)
        for i, layer in enumerate(self.stage_layers):
            x = layer(x)
            if i in self.output_idx:
                output.append(x)
        return tuple(output)


def ghostnet(config, input_shape, kernel_initializer):
    kernel_initializer = KernelInitializers().get_initializer(
        kernel_initializer)
    mobilenextnet = GhostNet(config=config,
                             kernel_initializer=kernel_initializer)
    image_inputs = tf.keras.Input(shape=input_shape, name='image_inputs')
    fmaps = mobilenextnet(image_inputs)
    return tf.keras.Model(image_inputs, fmaps)
