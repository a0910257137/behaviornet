import math
import tensorflow as tf

from ..utils.conv_module import ConvBlock
from ..utils.layers import MBConvBlock, FusedMBConvBlock
from .kernel_initializers import KernelInitializers


class InitBlk(tf.keras.layers.Layer):
    def __init__(self, stem_out, initial_layer, **kwargs):
        super().__init__(**kwargs)
        self.initial_layer = initial_layer
        self.stem_out = stem_out
        self.init_conv1 = ConvBlock(filters=self.stem_out,
                                    kernel_size=3,
                                    strides=2,
                                    kernel_initializer=self.initial_layer,
                                    norm_method='bn',
                                    activation='silu',
                                    use_bias=False)

    def call(self, x, training=None):
        x = self.init_conv1(x)
        return x


class EfficientNetV2(tf.keras.Model):
    def __init__(self, arch, kernel_initializer, **kwargs):
        super().__init__(**kwargs)
        self._blocks = []
        # Stem part.
        self.width_coefficient = 1.0
        self.depth_coefficient = 1.0
        self.depth_divisor = 8
        self.min_depth = 8
        if arch == 'efficientnetv2-s':
            self.stem_output = 24
            self.conv_types = [1, 1, 1, 1, 1]
            self.expand_ratios = [1, 4, 4, 4, 6, 6]
            self.input_filters = [24, 24, 48, 64, 128, 160]
            self.kernel_sizes = [3, 3, 3, 3, 3, 3]
            self.num_repeats = [2, 4, 4, 6, 9, 15]
            self.output_filters = [24, 48, 64, 128, 160, 256]
            self.se_ratios = [None, None, None, 0.25, 0.25, 0.25]
            self.strides = [1, 2, 2, 2, 1, 2]
        self._stem = InitBlk(self.stem_output, kernel_initializer)
        for block_infos in zip(self.conv_types, self.expand_ratios,
                               self.kernel_sizes, self.num_repeats,
                               self.input_filters, self.output_filters,
                               self.se_ratios, self.strides):
            conv_type, expand_ratio, kernel_size, num_repeats, input_filter, output_filter, se_ratio, stride = block_infos
            assert num_repeats > 0
            input_filters = self.round_filters(input_filters)
            output_filter = self.round_filters(output_filter)
            repeats = self.round_repeats(num_repeats, self.depth_coefficient)
            conv_block = {0: MBConvBlock(), 1: FusedMBConvBlock()}[conv_type]
            print(conv_block)
            xxx

    def call(self, x):
        return x

    def round_filters(self, filters, skip=False):
        """Round number of filters based on depth multiplier."""
        multiplier = self.width_coefficient
        divisor = self.depth_divisor
        min_depth = self.min_depth
        if skip or not multiplier:
            return filters

        filters *= multiplier
        min_depth = min_depth or divisor
        new_filters = max(min_depth,
                          int(filters + divisor / 2) // divisor * divisor)
        return int(new_filters)

    def round_repeats(self, repeats, multiplier, skip=False):
        """Round number of filters based on depth multiplier."""
        if skip or not multiplier:
            return repeats
        return int(math.ceil(multiplier * repeats))


def EfficientNet(input_shape, kernel_initializer):
    kernel_initializer = KernelInitializers().get_initializer(
        kernel_initializer)
    mobilenet = EfficientNetV2(arch='efficientnetv2-s',
                               kernel_initializer=kernel_initializer)
    image_inputs = tf.keras.Input(shape=input_shape, name='image_inputs')
    fmaps = mobilenet(image_inputs)
    return tf.keras.Model(image_inputs, fmaps)