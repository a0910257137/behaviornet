import tensorflow as tf
from .kernel_initializers import KernelInitializers
from ..utils.conv_module import ConvBlock
from pprint import pprint


class MobileNetModel(tf.keras.Model):

    def __init__(self,
                 config,
                 kernel_initializer,
                 num_stages=4,
                 out_indices=(0, 1, 2, 3),
                 *args,
                 **kwargs):
        super(MobileNetModel, self).__init__(*args, **kwargs)
        self.config = config
        self.out_indices = out_indices
        if self.config.block_cfg is None:
            stage_planes = [8, 16, 32, 64, 128, 256]  #0.25 default
            stage_blocks = [2, 4, 4, 2]
        else:
            stage_planes = self.config.block_cfg['stage_planes']
            stage_blocks = self.config.block_cfg['stage_blocks']
        assert len(stage_planes) == 6
        assert len(stage_blocks) == 4
        self.bases = []
        self.stem = [
            ConvBlock(filters=stage_planes[0],
                      kernel_size=3,
                      use_bias=False,
                      strides=2,
                      kernel_initializer=tf.keras.initializers.HeNormal,
                      norm_method='bn'),
            ConvBlock(filters=stage_planes[1],
                      kernel_size=3,
                      use_bias=False,
                      strides=1,
                      conv_mode='sp_conv2d',
                      kernel_initializer=tf.keras.initializers.HeNormal,
                      norm_method='bn')
        ]
        # implemnt bottleneck
        self.stage_layers = []
        for i, num_blocks in enumerate(stage_blocks):
            _layers = []
            layer_name = f'layer{i + 1}'
            for n in range(num_blocks):
                if n == 0:
                    _layer = ConvBlock(
                        filters=stage_planes[i + 2],
                        kernel_size=3,
                        use_bias=False,
                        strides=2,
                        conv_mode='sp_conv2d',
                        kernel_initializer=tf.keras.initializers.HeNormal,
                        norm_method='bn',
                        name=layer_name)
                else:
                    _layer = ConvBlock(
                        filters=stage_planes[i + 2],
                        kernel_size=3,
                        use_bias=False,
                        strides=1,
                        conv_mode='sp_conv2d',
                        kernel_initializer=tf.keras.initializers.HeNormal,
                        norm_method='bn',
                        name=layer_name)
                _layers.append(_layer)
            _block = [*_layers]
            self.stage_layers.append(_block)

    @tf.function
    def call(self, x):
        output = []
        for stem_layer in self.stem:
            x = stem_layer(x)
        for i, stage_layers in enumerate(self.stage_layers):
            for stage_layer in stage_layers:
                x = stage_layer(x)
            if i in self.out_indices:
                output.append(x)
        return tuple(output)


def mobilenet(config, input_shape, kernel_initializer):
    kernel_initializer = KernelInitializers().get_initializer(
        kernel_initializer)
    mobilenet = MobileNetModel(config=config,
                               kernel_initializer=kernel_initializer)
    image_inputs = tf.keras.Input(shape=input_shape, name='image_inputs')
    fmaps = mobilenet(image_inputs)
    return tf.keras.Model(image_inputs, fmaps)