from models.module import mobilenet
import tensorflow as tf
from .kernel_initializers import KernelInitializers
from ..utils.conv_module import ConvBlock, BottleNeck


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
            for filter, k_size, e, s, is_squeeze, n1 in zip(
                    filters, k_sizes, es, ss, is_squeezes, n1s):
                self.bases += [
                    BottleNeck(input_shape=1,
                               filters=filter,
                               kernel=k_size,
                               e=e,
                               s=s,
                               is_squeeze=is_squeeze,
                               nl=n1)
                ]
            xxx

    def call(self, x):
        m = len(self.base)
        x = self.init_conv(x)
        for base_op in self.bases:
            x = base_op(x)
            print(x)
            xxx

        return x


def MobileNetV3(input_shape, kernel_initializer):
    kernel_initializer = KernelInitializers().get_initializer(
        kernel_initializer)
    mobilenet = MobileNet(arch='v3_large',
                          kernel_initializer=kernel_initializer)
    image_inputs = tf.keras.Input(shape=input_shape, name='image_inputs')
    fmaps = mobilenet(image_inputs)
    return tf.keras.Model(image_inputs, mobilenet)
