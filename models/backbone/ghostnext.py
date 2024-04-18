import tensorflow as tf
from .kernel_initializers import KernelInitializers
from ..utils.conv_module import ConvBlock
from pprint import pprint
from keras_flops import get_flops

BN_MOMENTUM = 0.999
BN_EPSILON = 1e-3


def _make_divisible(channels, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_channels = max(min_value,
                       int(channels + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_channels < 0.9 * channels:
        new_channels += divisor
    return new_channels


class SEModule(tf.keras.layers.Layer):
    """
    A squeeze and excite module
    """

    def __init__(self,
                 in_channels,
                 se_ratio=0.25,
                 divisor=8,
                 limit_round_down=0.9,
                 name=""):
        super(SEModule, self).__init__()
        """Squeeze-and-Excitation block, arxiv: https://arxiv.org/pdf/1709.01507.pdf"""
        reduction = _make_divisible(in_channels * se_ratio, divisor,
                                    limit_round_down)
        self.reshape = tf.keras.layers.Lambda(self._reshape)
        self.pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.conv1 = ConvBlock(filters=reduction,
                               kernel_size=1,
                               strides=1,
                               norm_method=None,
                               activation="relu",
                               use_bias=False,
                               name=name + "/1x1conv_1")

        self.conv2 = ConvBlock(filters=in_channels,
                               kernel_size=1,
                               strides=1,
                               norm_method=None,
                               activation="hard_sigmoid",
                               use_bias=False,
                               name=name + "/1x1conv_2")
        self.mul = tf.keras.layers.Multiply(name=name and name + "/se_output")

    @staticmethod
    def _reshape(x):
        return tf.keras.layers.Reshape((1, 1, int(x.shape[1])))(x)

    def call(self, inputs):
        x = self.reshape(self.pooling(inputs))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.mul([inputs, x])
        return x


class GhostModule(tf.keras.layers.Layer):

    def __init__(self,
                 out_channel,
                 kernel_size=3,
                 strides=1,
                 activation=None,
                 is_ghost_1=False,
                 name=""):
        super(GhostModule, self).__init__()
        ratio = 2
        hidden_channel = int(tf.math.ceil(float(out_channel) / ratio))
        self.is_ghost_1 = is_ghost_1
        if 'ghost_1' in name and self.is_ghost_1:
            name_tmp = "/prim_dw" if strides == 1 else "/prim_down"
            self.dw1 = ConvBlock(filters=None,
                                 kernel_size=kernel_size,
                                 strides=strides,
                                 use_bias=False,
                                 norm_method="bn",
                                 activation=activation,
                                 conv_mode="dw_conv2d",
                                 name=name + name_tmp)
        # 1x1 conv
        self.primary_conv = ConvBlock(filters=hidden_channel,
                                      kernel_size=1,
                                      strides=1,
                                      use_bias=False,
                                      norm_method="bn",
                                      activation=activation,
                                      name=name + "/prim")

        self.cheap_conv = ConvBlock(filters=None,
                                    kernel_size=kernel_size,
                                    strides=1,
                                    use_bias=False,
                                    norm_method="bn",
                                    activation=None,
                                    conv_mode="dw_conv2d",
                                    name=name + "/cheap")

    def __call__(self, x):
        if self.is_ghost_1:
            x = self.dw1(x)
        primary_x = self.primary_conv(x)
        cheap_x = self.cheap_conv(primary_x)
        return tf.concat([primary_x, cheap_x], axis=-1)


class GhostModuleMultiply(tf.keras.layers.Layer):
    """
    The main Ghost module
    """

    def __init__(self,
                 out_channel,
                 kernel_size,
                 strides=1,
                 activation=None,
                 name=""):
        super(GhostModuleMultiply, self).__init__()
        self.strides = strides
        self.ghost = GhostModule(out_channel,
                                 kernel_size,
                                 strides=self.strides,
                                 activation=activation,
                                 is_ghost_1=True,
                                 name=name + "/ghost_1")

        # DFC attention (Decouple Fully Connection)
        self.avg = tf.keras.layers.AveragePooling2D(pool_size=2,
                                                    strides=2,
                                                    name=name + '/AvgPool')
        self.dfc_1 = ConvBlock(filters=out_channel,
                               kernel_size=1,
                               strides=1,
                               use_bias=False,
                               norm_method="bn",
                               activation=None,
                               name=name + "/DFC_1")
        self.dfc_2 = ConvBlock(filters=out_channel,
                               kernel_size=(1, 5),
                               strides=1,
                               use_bias=False,
                               norm_method="bn",
                               activation=None,
                               conv_mode="dw_conv2d",
                               name=name + "/DFC_2")

        self.dfc_3 = ConvBlock(filters=out_channel,
                               kernel_size=(5, 1),
                               strides=1,
                               use_bias=False,
                               norm_method="bn",
                               activation=None,
                               conv_mode="dw_conv2d",
                               name=name + "/DFC_1")
        self.sigmoid = tf.keras.layers.Activation(activation="sigmoid",
                                                  name=name + "/DFC_sigmoid")
        if self.strides == 1:
            self.resize_image = tf.image.resize

    def __call__(self, inputs):
        x = self.ghost(inputs)
        shortcut = self.avg(inputs)
        shortcut = self.dfc_1(shortcut)

        shortcut = self.dfc_2(shortcut)
        shortcut = self.dfc_3(shortcut)

        shortcut = self.sigmoid(shortcut)
        if self.strides == 1:
            shortcut = self.resize_image(shortcut,
                                         tf.shape(inputs)[1:-1],
                                         antialias=False,
                                         method="bilinear")
        return tf.keras.layers.Multiply()([shortcut, x])


class GBNeck(tf.keras.layers.Layer):
    """
    The GhostNet Bottleneck
    """

    def __init__(self,
                 out_channel,
                 first_ghost_channel,
                 kernel_size=3,
                 strides=1,
                 se_ratio=0,
                 tensor_multiplier=0.5,
                 shortcut=True,
                 use_ghost_module_multiply=False,
                 name=""):
        super(GBNeck, self).__init__()
        self.se_ratio = se_ratio
        self.tensor_multiplier = tensor_multiplier
        self.shortcut = shortcut
        self.use_ghost_module_multiply = use_ghost_module_multiply
        if self.shortcut:
            self.dw_shortcut = ConvBlock(filters=None,
                                         kernel_size=kernel_size,
                                         strides=strides,
                                         norm_method="bn",
                                         activation=None,
                                         use_bias=False,
                                         conv_mode="dw_conv2d")
            self.pw_shortcut = ConvBlock(filters=out_channel,
                                         kernel_size=1,
                                         strides=1,
                                         norm_method="bn",
                                         activation=None,
                                         use_bias=False)

        if self.use_ghost_module_multiply:
            self.ghost_module_multiply = GhostModuleMultiply(
                first_ghost_channel,
                kernel_size,
                strides=strides,
                activation="relu",
                name=name + "/module_mul")
        else:
            self.ghost_module_1 = GhostModule(first_ghost_channel,
                                              kernel_size,
                                              strides=strides,
                                              activation="relu",
                                              is_ghost_1=True,
                                              name=name + "/ghost_1")
        if self.se_ratio > 0:
            self.se_module = SEModule(first_ghost_channel,
                                      se_ratio=self.se_ratio,
                                      divisor=4,
                                      name=name + "/se_module")
        self.ghost_moudule_2 = GhostModule(out_channel,
                                           kernel_size,
                                           strides=1,
                                           activation=None,
                                           is_ghost_1=False,
                                           name=name + "/ghost_2")

    @staticmethod
    def _return_slices(x, channel):
        return x[:, :, :, :channel]

    def call(self, inputs):
        if self.shortcut:
            shortcut = self.dw_shortcut(inputs)
            shortcut = self.pw_shortcut(shortcut)
        else:
            shortcut = inputs

        if self.use_ghost_module_multiply:
            x = self.ghost_module_multiply(inputs)
        else:
            x = self.ghost_module_1(inputs)
        if self.se_ratio > 0:
            x = self.se_module(x)
        x = self.ghost_moudule_2(x)
        if 0. < float(self.tensor_multiplier) < 1.0:
            ip = round(tf.shape(x)[-1] * self.tensor_multiplier)
            shortcut_portion = tf.keras.layers.Lambda(
                lambda z: z[:, :, :, 0:ip])(shortcut)
            x_hat = tf.keras.layers.Lambda(lambda z: z[:, :, :, 0:ip])(x)
            x_tail = tf.keras.layers.Lambda(lambda z: z[:, :, :, ip:])(x)
            # portion of channel are sum up
            x_sum = shortcut_portion + x_hat
            return tf.concat([x_sum, x_tail], axis=-1)

        else:
            return shortcut + x


class GhostNext(tf.keras.Model):

    def __init__(self, config, kernel_initializer, *args, **kwargs):
        super(GhostNext, self).__init__(*args, **kwargs)
        self.config = config
        self.kernel_sizes = self.config.kernel_size
        self.first_ghost_channels = self.config.first_ghost_channels
        self.out_channels = self.config.out_channels
        self.se_ratios = self.config.se_ratios
        self.strides = self.config.strides
        self.stage_one_width = self.config.stage_one_width
        self.stage_one_strides = self.config.stage_one_strides
        self.num_ghost_module_v1_stacks = self.config.num_ghost_module_v1_stacks
        self.stage_layers = []
        self.output_idx = [2, 4, 10, 15]
        tensor_multiplier = 1
        width_mul = 1.3
        """stage 1"""
        stem_width = _make_divisible(self.stage_one_width * width_mul,
                                     divisor=4)
        self.stem_conv = ConvBlock(filters=stem_width,
                                   kernel_size=3,
                                   strides=2,
                                   use_bias=False,
                                   activation="relu",
                                   norm_method="bn",
                                   name='conv_stem')
        """ stages2~15 """
        in_channel = stem_width
        for stack_id, (kernel_size, stride, first_ghost, out_channel,
                       se_ratio) in enumerate(
                           zip(self.kernel_sizes, self.strides,
                               self.first_ghost_channels, self.out_channels,
                               self.se_ratios)):
            stage_name = "stage{}".format(stack_id + 2)
            out_channel = _make_divisible(out_channel * width_mul, 4)
            first_ghost_channel = _make_divisible(first_ghost * width_mul, 4)
            use_ghost_module_multiply = True if self.num_ghost_module_v1_stacks >= 0 and stack_id >= self.num_ghost_module_v1_stacks else False
            shortcut = False if out_channel == in_channel and stride == 1 else True
            layer = GBNeck(out_channel,
                           first_ghost_channel,
                           kernel_size,
                           stride,
                           se_ratio,
                           tensor_multiplier,
                           shortcut,
                           use_ghost_module_multiply,
                           name=stage_name)
            in_channel = out_channel
            self.stage_layers.append(layer)

    @tf.function
    def call(self, x):
        output = []
        x = self.stem_conv(x)
        for i, layer in enumerate(self.stage_layers):
            x = layer(x)
            if i in self.output_idx:
                output.append(x)
        return tuple(output)


def ghostnext(config, input_shape, kernel_initializer):
    kernel_initializer = KernelInitializers().get_initializer(
        kernel_initializer)
    mobilenextnet = GhostNext(config=config,
                              kernel_initializer=kernel_initializer)
    image_inputs = tf.keras.Input(shape=input_shape, name='image_inputs')
    fmaps = mobilenextnet(image_inputs)
    # fully_models = tf.keras.Model(image_inputs, fmaps, name='fully')
    # flops = get_flops(fully_models, batch_size=1)
    # print(f"FLOPS: {flops / 10 ** 9:.03} G")
    # exit(1)
    return tf.keras.Model(image_inputs, fmaps)
