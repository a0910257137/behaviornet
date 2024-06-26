import tensorflow as tf
from .kernel_initializers import KernelInitializers
from ..utils.conv_module import ConvBlock
from pprint import pprint

conv_mode = 'sp_conv2d'


# as public function
def channel_shuffle(x, groups):
    _, height, width, num_channels = x.get_shape().as_list()
    channels_per_group = num_channels // groups
    # reshape
    x = tf.reshape(x, [-1, height, width, groups, channels_per_group])
    x = tf.transpose(x, [0, 1, 2, 4, 3])
    # flatten
    x = tf.reshape(x, [-1, height, width, num_channels])
    return x


class ShuffleV2Block(tf.keras.layers.Layer):

    def __init__(self, inp, oup, stride, *args, **kwargs):
        super(ShuffleV2Block, self).__init__(*args, **kwargs)

        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride
        self.padding = 'same'
        self.inp_shape = inp
        self.oup_shape = oup
        self.branch_features = oup // 2
        # self.initial_layer = tf.keras.initializers.RandomNormal(mean=0,
        #                                                         stddev=.1)
        self.initial_layer = tf.keras.initializers.HeNormal()

        if self.stride > 1:
            self.branch1 = tf.keras.Sequential(name='branch1')
            self.branch1.add(
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=3,
                    strides=(self.stride, self.stride),
                    padding=self.padding,
                    depthwise_initializer=self.initial_layer,
                    use_bias=False))
            self.branch1.add(tf.keras.layers.BatchNormalization())
            self.branch1.add(
                tf.keras.layers.Conv2D(filters=self.branch_features,
                                       kernel_size=1,
                                       strides=(1, 1),
                                       padding=self.padding,
                                       kernel_initializer=self.initial_layer,
                                       use_bias=False))
            self.branch1.add(tf.keras.layers.BatchNormalization())
            self.branch1.add(tf.keras.layers.Activation(activation="swish"))

        self.branch2 = tf.keras.Sequential(name='branch2')
        self.branch2.add(
            tf.keras.layers.Conv2D(filters=self.branch_features,
                                   kernel_size=1,
                                   strides=(1, 1),
                                   padding=self.padding,
                                   kernel_initializer=self.initial_layer,
                                   use_bias=False))
        self.branch2.add(tf.keras.layers.BatchNormalization())
        self.branch2.add(tf.keras.layers.Activation(activation="swish"))
        self.branch2.add(
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3,
                strides=(self.stride, self.stride),
                padding=self.padding,
                depthwise_initializer=self.initial_layer,
                use_bias=False))
        self.branch2.add(tf.keras.layers.BatchNormalization())
        self.branch2.add(
            tf.keras.layers.Conv2D(filters=self.branch_features,
                                   kernel_size=1,
                                   strides=(1, 1),
                                   padding=self.padding,
                                   kernel_initializer=self.initial_layer,
                                   use_bias=False))
        self.branch2.add(tf.keras.layers.BatchNormalization())
        self.branch2.add(tf.keras.layers.Activation(activation="swish"))

    def call(self, x):
        if self.stride == 1:
            x1, x2 = x[..., :self.branch_features], x[...,
                                                      self.branch_features:]
            out = tf.concat([x1, self.branch2(x2)], axis=-1)
        else:
            out = tf.concat([self.branch1(x), self.branch2(x)], axis=-1)
        out = channel_shuffle(out, 2)

        return out


def get_downsampling(down_method):
    if down_method == "avg_pool":
        return tf.keras.layers.AveragePooling2D(strides=2, padding='same')
    elif down_method == "max_pool":
        return tf.keras.layers.MaxPooling2D(strides=2, padding='same')
    elif down_method == "dwconv":
        return tf.keras.layers.DepthwiseConv2D(kernel_size=2,
                                               strides=2,
                                               padding='same')


class Stem(tf.keras.layers.Layer):

    def __init__(self, out_chs, *args, **kwargs):
        super(Stem, self).__init__(*args, **kwargs)
        self.out_chs = out_chs
        self.conv_mode = 'sp_conv2d'
        self.stem_1 = ConvBlock(
            filters=out_chs[0],
            kernel_size=3,
            strides=2,
            use_bias=False,
            conv_mode=self.conv_mode,
            kernel_initializer=tf.keras.initializers.HeNormal(),
            activation="swish",
            name='init_conv1')
        self.stem_2a = ConvBlock(
            filters=out_chs[1],
            kernel_size=1,
            strides=1,
            use_bias=False,
            conv_mode=self.conv_mode,
            kernel_initializer=tf.keras.initializers.HeNormal(),
            activation="swish",
            name='init_conv2')

        self.stem_2b = ConvBlock(
            filters=out_chs[2],
            kernel_size=3,
            strides=2,
            use_bias=False,
            conv_mode=self.conv_mode,
            kernel_initializer=tf.keras.initializers.HeNormal(),
            activation="swish",
            name='init_conv3')
        self.stem_2p = get_downsampling(down_method='max_pool')
        self.stem_3 = ConvBlock(
            filters=out_chs[3],
            kernel_size=1,
            strides=1,
            use_bias=False,
            conv_mode=self.conv_mode,
            kernel_initializer=tf.keras.initializers.HeNormal(),
            activation="swish",
            name='init_conv4')

    def call(self, x, training=None):
        stem_1_out = self.stem_1(x)
        stem_2a_out = self.stem_2a(stem_1_out)
        stem_2b_out = self.stem_2b(stem_2a_out)
        stem_2p_out = self.stem_2p(stem_1_out)
        out = self.stem_3(tf.concat([stem_2b_out, stem_2p_out], axis=-1))
        return out


class ShuffleNetV2(tf.keras.Model):

    def __init__(self, arch, kernel_initializer, *args, **kwargs):
        super(ShuffleNetV2, self).__init__(*args, **kwargs)

        # out_stages can only be a subset of (2, 3, 4)
        self.out_stages = [2, 3, 4]

        assert set(self.out_stages).issubset((2, 3, 4))
        # self.model_size = '1.0x'
        self.model_size = 'yolo5n'
        self.stage_repeats = [4, 8, 4]
        # self.stage_repeats = [4, 8, 4]
        self.with_last_conv = False
        self.kernal_size = 3

        if self.model_size == "0.5x":
            self._stage_out_channels = [24, 48, 96, 192, 1024]
        elif self.model_size == "1.0x":
            self._stage_out_channels = [24, 116, 232, 464, 1024]
        elif self.model_size == "1.5x":
            self._stage_out_channels = [24, 176, 352, 704, 1024]
        elif self.model_size == "2.0x":
            self._stage_out_channels = [24, 244, 488, 976, 2048]
        elif self.model_size == "yolo5n":
            # yoylo do stem in first
            self.stem_channels = [32, 16, 32, 32]
            self._stage_out_channels = [128, 256, 512]
            self.last_channels = 128
        else:
            raise NotImplementedError
        # building first layer
        # stem
        if 'yolo' in self.model_size:
            self.init_conv = Stem(self.stem_channels)
            input_channels = self.stem_channels[-1]
            output_channels = self._stage_out_channels[0]
        else:
            output_channels = self._stage_out_channels[0]
            self.init_conv = ConvBlock(
                filters=output_channels,
                kernel_size=3,
                strides=2,
                use_bias=False,
                conv_mode='sp_conv2d',
                kernel_initializer=tf.keras.initializers.RandomNormal(
                    mean=0, stddev=0.01),
                name='init_conv',
                activation='leaky_relu')
            input_channels = output_channels
        self.maxpool = get_downsampling('max_pool')
        # stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        self.base = []
        for i, (repeats, output_channels) in enumerate(
                zip(self.stage_repeats, self._stage_out_channels)):
            prefix = 'stage_{}'.format(i + 2)
            sufix = '_init'
            seq = []
            seq += [
                ShuffleV2Block(input_channels,
                               output_channels,
                               2,
                               name=prefix + sufix)
            ]
            #TODO: check conv list and do p4 - p6
            for i in range(repeats - 1):
                sufix = '_repeat{}'.format(i + 1)
                seq.append(
                    ShuffleV2Block(input_channels,
                                   output_channels,
                                   1,
                                   name=prefix + sufix))
            self.base += [seq]

            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        if self.with_last_conv:
            self.last_module = tf.keras.Sequential(name='last_prediction')
            self.last_module.add(
                tf.keras.layers.Conv2D(
                    filters=self.last_channels,
                    kernel_size=1,
                    strides=(1, 1),
                    padding='same',
                    kernel_initializer=tf.keras.initializers.RandomNormal(
                        mean=0, stddev=.1),
                    use_bias=False))
            self.last_module.add(tf.keras.layers.BatchNormalization())
            self.last_module.add(
                tf.keras.layers.Activation(activation="swish"))

        self.skip_layers = [
            "stage_0",
            "stage_1_repreat_3",
            "stage_2_repreat_7",
            "stage_3_repreat_3",
        ]

    def call(self, x, training=None):
        skip_connections = {}
        x = self.init_conv(x)
        skip_connections[self.skip_layers[0]] = x
        for i, layer_ops in enumerate(self.base):
            for layer_op in layer_ops:
                x = layer_op(x)
            if i + 2 in self.out_stages:
                skip_connections[self.skip_layers[i + 1]] = x
        return x, skip_connections


def SuffleNet(input_shape, kernel_initializer):
    kernel_initializer = KernelInitializers().get_initializer(
        kernel_initializer)
    sufflenet = ShuffleNetV2(arch='1x0', kernel_initializer=kernel_initializer)
    image_inputs = tf.keras.Input(shape=input_shape, name='image_inputs')
    fmaps = sufflenet(image_inputs)
    return tf.keras.Model(image_inputs, fmaps)
