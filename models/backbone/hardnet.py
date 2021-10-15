import imp
from .kernel_initializers import KernelInitializers
from ..utils.conv_module import ConvBlock
import tensorflow as tf
from pprint import pprint

conv_mode = 'sp_conv2d'


class HardBlock(tf.keras.layers.Layer):
    def __init__(self,
                 in_channels,
                 growth_rate,
                 grmul,
                 n_layers,
                 kernel_initializer,
                 keepBase=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.keepBase = keepBase
        self.links = []
        self.layers = []
        self.out_channels = 0
        for i in range(n_layers):
            outch, inch, link = self._get_link(i + 1, in_channels, growth_rate,
                                               grmul)
            self.links.append(link)
            self.layers.append(
                ConvBlock(outch,
                          kernel_size=3,
                          strides=1,
                          use_bias=False,
                          conv_mode=conv_mode,
                          kernel_initializer=kernel_initializer))
            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch

    @property
    def get_out_ch(self):
        return self.out_channels

    def _get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2**i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch, _, _ = self._get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def call(self, inputs):
        layers_ = [inputs]
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                inputs = tf.concat(tin, 3)
            else:
                inputs = tin[0]
            out = self.layers[layer](inputs)
            layers_.append(out)
        t = len(layers_)
        out_ = []
        for i in range(t):
            if (i == 0 and self.keepBase) or (i == t - 1) or (i % 2 == 1):
                out_.append(layers_[i])
        out = tf.concat(out_, 3)
        return out


class AvgPoolConcat(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        self.avg9x9 = tf.keras.layers.AveragePooling2D(pool_size=(9, 9),
                                                       strides=1,
                                                       padding='same')
        super().__init__(**kwargs)

    def call(self, input, **kwargs):
        x2 = self.avg9x9(input)
        x3 = input / (
            tf.reduce_sum(input_tensor=x2, axis=[1, 2], keepdims=True) + 0.1)
        return tf.concat([input, x2, x3], axis=-1)


class HardNet(tf.keras.Model):
    def __init__(self, pooling, arch, kernel_initializer, *args, **kwargs):
        def get_downsampling(down_method, name):
            if down_method == "avg_pool":
                return tf.keras.layers.AveragePooling2D(strides=2,
                                                        padding='same',
                                                        name=name)
            elif down_method == "max_pool":
                return tf.keras.layers.MaxPooling2D(strides=2,
                                                    padding='same',
                                                    name=name)
            elif down_method == "dwconv":
                return tf.keras.layers.DepthwiseConv2D(kernel_size=2,
                                                       strides=2,
                                                       padding='same',
                                                       name=name)

        """
        Arguments:
        - *arch*: provide HardNet arhictecture 68 and 85
        - *pooling*: type is  strings. There are avg_pool/max_pool/dw different down-sampling methods.
        """
        super(HardNet, self).__init__(*args, **kwargs)
        self._base = []
        self._shortcut_layers = []
        grmul = 1.6
        if arch == 68:
            #HarDNet68
            first_ch = [32, 64]
            ch_list = [128, 256, 320, 640]
            gr = [14, 16, 20, 40]
            n_layers = [8, 16, 16, 16]
            downSamp = [1, 0, 1, 1]
            last_proj_ch = 192
            last_blk = [
                576, 72, 8
            ]  # as hardblock input channel, groth_rate, number of layers
            blks = len(n_layers)

        elif arch == 39:
            #HarDNet68
            first_ch = [24, 48]
            ch_list = [96, 320, 640]
            gr = [14, 18, 54]
            n_layers = [4, 16, 8]
            downSamp = [1, 1, 1]
            last_proj_ch = 128
            last_blk = [
                496, 64, 8
            ]  # as hardblock input channel, groth_rate, number of layers
            blks = len(n_layers)

        elif arch == 85:
            #HarDNet85
            first_ch = [48, 96]
            ch_list = [192, 256, 320, 480, 720, 1280]
            gr = [24, 24, 28, 36, 48, 256]
            n_layers = [8, 16, 16, 16, 16, 4]
            downSamp = [1, 0, 1, 0, 1, 0]
            last_proj_ch = 256

            last_blk = [
                768, 80, 8
            ]  # as hardblock input channel, groth_rate, number of layers
            blks = len(n_layers)
        else:
            raise ValueError(
                "Unsupported HarDNet version with number of layers {}".format(
                    arch))

        self._base.append(
            ConvBlock(first_ch[0],
                      kernel_size=3,
                      strides=2,
                      use_bias=False,
                      conv_mode=conv_mode,
                      name='init_conv1'))
        self._base.append(
            ConvBlock(first_ch[1],
                      kernel_size=3,
                      strides=1,
                      use_bias=False,
                      conv_mode=conv_mode,
                      name='init_conv2'))
        self._shortcut_layers.append(len(self._base) - 1)
        self._base.append(
            tf.keras.layers.AveragePooling2D(pool_size=(3, 3),
                                             strides=2,
                                             padding='same',
                                             name='init_avg_pool'))
        ch = first_ch[1]
        for i in range(blks):
            blk = HardBlock(ch,
                            gr[i],
                            grmul,
                            n_layers[i],
                            kernel_initializer,
                            name='down_hard_blk{}'.format(i + 1))
            ch = blk.get_out_ch
            self._base.append(blk)
            if i != blks - 1:
                self._base.append(
                    ConvBlock(ch_list[i],
                              kernel_size=1,
                              use_bias=False,
                              conv_mode=conv_mode,
                              name='down_trans{}'.format(i + 1)))
            ch = ch_list[i]
            if downSamp[i] == 1:
                self._shortcut_layers.append(len(self._base) - 1)
                # "max_pool", // "avg_pool" , "dwconv"
                if i != blks - 1:
                    self._base.append(
                        get_downsampling(pooling,
                                         name='down' + '_' + pooling +
                                         '{}'.format(i + 1)))
            if i == blks - 1:
                self._base.append(
                    ConvBlock(last_proj_ch,
                              kernel_size=1,
                              use_bias=False,
                              conv_mode=conv_mode,
                              name='down_last_trans{}'.format(i + 1)))
                self._base.append(
                    get_downsampling(pooling,
                                     name='down' + '_' + pooling +
                                     '{}'.format(i + 1)))
                self._base.append(
                    AvgPoolConcat(name='down_avg_concat{}'.format(i + 1)))
                blk = HardBlock(last_blk[0],
                                last_blk[1],
                                grmul,
                                last_blk[2],
                                kernel_initializer,
                                name='down_last_hard_blk{}'.format(i + 1))
                ch = blk.get_out_ch
                self._base.append(blk)
        # hard code architecture 39/68 for skip connections
        # hardblk output will be the next fpn
        if arch == 39:
            self._shortcut_layers[1:3] = [3, 6]
        elif arch == 68:
            self._shortcut_layers[1:3] = [3, 8]

    def call(self, x):
        skip_connections = {}
        for i in range(len(self._base)):
            x = self._base[i](x)
            if i in self._shortcut_layers:
                skip_connections[x.name] = x
        return x, skip_connections


def HardNet39(input_shape, pooling, kernel_initializer):
    kernel_initializer = KernelInitializers().get_initializer(
        kernel_initializer)
    hardnet = HardNet(pooling=pooling,
                      arch=39,
                      kernel_initializer=kernel_initializer)
    image_inputs = tf.keras.Input(shape=input_shape, name='image_inputs')
    fmaps = hardnet(image_inputs)

    return tf.keras.Model(image_inputs, fmaps, name='backbone')


def HardNet68(input_shape, pooling, kernel_initializer):
    kernel_initializer = KernelInitializers().get_initializer(
        kernel_initializer)
    hardnet = HardNet(pooling=pooling,
                      arch=68,
                      kernel_initializer=kernel_initializer)
    image_inputs = tf.keras.Input(shape=input_shape, name='image_inputs')
    fmaps = hardnet(image_inputs)
    return tf.keras.Model(image_inputs, fmaps, name='backbone')


def HardNet85(input_shape, pooling, kernel_initializer):
    hardnet = HardNet(pooling=pooling,
                      arch=85,
                      kernel_initializer=kernel_initializer)
    image_inputs = tf.keras.Input(shape=input_shape, name='image_inputs')
    fmaps = hardnet(image_inputs)
    return tf.keras.Model(inputs=image_inputs, outputs=fmaps, name='backbone')