from models.module import mobilenet
import tensorflow as tf
from .kernel_initializers import KernelInitializers
from ..utils.conv_module import ConvBlock, BottleNeck
from pprint import pprint


class MobileNet(tf.keras.Model):
    def __init__(self, arch, kernel_initializer, *args, **kwargs):
        super(MobileNet, self).__init__(*args, **kwargs)
        # implemnt bottleneck
        self.bases = []
        if arch == 'v3_large':
            filters = [
                16, 24, 24, 40, 40, 40, 80, 80, 80, 80, 112, 112, 160, 160, 160
            ]
            k_sizes = [3, 3, 3, 5, 5, 5, 3, 3, 3, 3, 3, 3, 5, 5, 5]
            es = [
                16, 64, 72, 72, 120, 120, 240, 200, 184, 184, 480, 672, 672,
                960, 960
            ]
            ss = [1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1]
            is_squeezes = [
                False, False, False, True, True, True, False, False, False,
                False, True, True, True, True
            ]
            n1s = [
                'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'HS', 'HS',
                'HS', 'HS', 'HS', 'HS', 'HS', 'HS', 'HS'
            ]

            self.init_conv = ConvBlock(filters=16,
                                       kernel_size=3,
                                       strides=2,
                                       name='init_conv',
                                       kernel_initializer=kernel_initializer,
                                       activation='HS')
            for i, (filter, k_size, e, s, is_squeeze, n1) in enumerate(
                    zip(filters, k_sizes, es, ss, is_squeezes, n1s)):
                if i == 0:
                    input_chs = 16
                else:
                    input_chs = filters[i - 1]
                self.bases += [
                    BottleNeck(input_chs=input_chs,
                               filters=filter,
                               kernel=k_size,
                               e=e,
                               s=s,
                               is_squeeze=is_squeeze,
                               nl=n1)
                ]
            xxx
            self.trans_conv = ConvBlock(filters=960,
                                        kernel_size=1,
                                        strides=1,
                                        name='trans_conv',
                                        kernel_initializer=kernel_initializer,
                                        activation='HS')
            self.last_glbap = tf.keras.layers.GlobalAveragePooling2D()
            self.last_conv = ConvBlock(filters=1280,
                                       kernel_size=1,
                                       strides=1,
                                       norm_method=None,
                                       name='last_conv',
                                       kernel_initializer=kernel_initializer,
                                       activation='HS')

    def call(self, x):
        m = len(self.base)
        x = self.init_conv(x)
        for base_op in self.bases:
            x = base_op(x)
            print(x)
            xxx
        x = self.trans_conv(x)
        x = tf.reshape((x, [1, 1, 960]))
        x = self.last_conv(x)
        return x


def MobileNetV3(input_shape, kernel_initializer):
    kernel_initializer = KernelInitializers().get_initializer(
        kernel_initializer)
    mobilenet = MobileNet(arch='v3_large',
                          kernel_initializer=kernel_initializer)
    image_inputs = tf.keras.Input(shape=input_shape, name='image_inputs')
    fmaps = mobilenet(image_inputs)
    return tf.keras.Model(image_inputs, mobilenet)
