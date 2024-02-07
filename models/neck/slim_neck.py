import tensorflow as tf
from ..utils.conv_module import *

# class Bottleneck(nn.Module):
#     # Standard bottleneck
#     def __init__(self,
#                  c1,
#                  c2,
#                  shortcut=True,
#                  g=1,
#                  e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c_, c2, 3, 1, g=g)
#         self.add = shortcut and c1 == c2

#     def forward(self, x):
#         return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class GBottleneck(tf.keras.layers.Layer):

    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super(GBottleneck, self).__init__()
        c_ = int(c2 * e)
        self.conv1 = ConvBlock(filters=c_,
                               kernel_size=1,
                               strides=1,
                               norm_method="bn",
                               activation="relu")
        self.conv2 = ConvBlock(filters=c2,
                               kernel_size=3,
                               strides=1,
                               norm_method="bn",
                               activation="relu")
        self.add = shortcut and c1 == c2

    def __call__(self, x):
        short_x = self.conv2(self.conv1(x)) if self.add else self.conv2(
            self.conv1(x))
        return x + short_x


class C3(tf.keras.layers.Layer):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c2, n=1, shortcut=True, e=0.5):
        super(C3, self).__init__()
        c_ = int(c2 * e)
        self.conv1 = ConvBlock(filters=c_,
                               kernel_size=1,
                               strides=1,
                               use_bias=False,
                               norm_method="bn",
                               activation="relu")
        self.conv2 = ConvBlock(filters=c_,
                               kernel_size=1,
                               strides=1,
                               use_bias=False,
                               norm_method="bn",
                               activation="relu")
        self.conv3 = ConvBlock(filters=c2,
                               kernel_size=1,
                               strides=1,
                               use_bias=False,
                               norm_method="bn",
                               activation="relu")
        self.m = [GBottleneck(c_, c_, shortcut, e=1.0) for _ in range(n)]

    def __call__(self, x):
        x1 = self.conv1(x)
        for layer in self.m:
            x1 = layer(x1)
        x2 = self.conv2(x)
        x = tf.concat([x1, x2], axis=-1)
        x = self.conv3(x)
        return x


class GSConv(tf.keras.layers.Layer):

    def __init__(self, c2, k=1, s=1, activate="relu"):
        # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
        super(GSConv, self).__init__()
        c_ = c2 // 2
        self.conv = ConvBlock(
            filters=c_,
            kernel_size=k,
            strides=s,
            use_bias=False,
            norm_method="bn",
            activation=activate,
            conv_mode="conv2d",
        )
        self.dw = ConvBlock(
            filters=None,
            kernel_size=5,
            strides=1,
            use_bias=False,
            norm_method="bn",
            activation=activate,
            conv_mode="dw_conv2d",
        )

    def __call__(self, x):
        x1 = self.conv(x)
        x2 = tf.concat([x1, self.dw(x1)], axis=-1)
        # suffle
        b, h, w, c = [tf.shape(x2)[i] for i in range(4)]
        x2 = tf.transpose(x2, [0, 3, 1, 2])
        b_c = b * c // 2
        y = tf.reshape(x2, [b_c, 2, h * w])
        y = tf.transpose(y, [1, 0, 2])
        y = tf.reshape(y, [2, -1, c // 2, h, w])
        y = tf.concat([y[0], y[1]], axis=1)
        y = tf.transpose(y, [0, 2, 3, 1])
        return y


class SlimNeck(tf.keras.Model):

    def __init__(self, config, end_level=-1, **kwargs):
        super(SlimNeck, self).__init__(**kwargs)
        self.in_channels = config.in_channels
        l_out_channels = config.l_out_channels
        self.num_outs = config.num_outs
        gs_out_channels = [128, 64, 64, 128]
        gs_kernel_size = [1, 1, 3, 3]
        gs_strides = [1, 1, 2, 2]
        # ---------------------------------
        c3_out_channels = [128, 64, 128, 256]
        start_level = config.start_level
        self.num_ins = len(self.in_channels)
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert self.num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(self.in_channels)
            assert self.num_outs == end_level - start_level
        self.start_level, self.end_level = start_level, end_level
        self.lateral_convs, self.gs_convs, self.c3_convs = [], [], []
        for i in range(4):
            self.gs_convs.append(
                GSConv(c2=gs_out_channels[i],
                       k=gs_kernel_size[i],
                       s=gs_strides[i]))
            self.c3_convs.append(
                C3(c2=c3_out_channels[i], n=1, shortcut=False, e=1))
            if i < 3:
                l_conv = ConvBlock(
                    filters=l_out_channels,
                    kernel_size=1,
                    strides=1,
                    kernel_initializer=tf.keras.initializers.HeNormal,
                    norm_method=None,
                    activation=None,
                    use_bias=False)
                self.lateral_convs.append(l_conv)
        self.resize = tf.image.resize
        self.last_c3 = C3(c2=256, n=1, shortcut=False, e=1)

    def call(self, x):
        assert len(x) == len(self.in_channels)
        # build laterals

        laterals = [
            lateral_conv(x[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # build top-down path
        used_backbone_levels = len(laterals)
        up_lateral_feats = []
        x = laterals[2]
        for i, j in zip(range(used_backbone_levels - 1, 0, -1), range(2)):
            prev_shape = tf.shape(laterals[i - 1])[1:3]
            x = self.gs_convs[j](x)
            up_lateral_feats.append(x)
            x = self.resize(images=x,
                            size=prev_shape,
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            lateral_x = laterals[i - 1]
            x = tf.concat([lateral_x, x], axis=-1)
            x = self.c3_convs[j](x)

        # build outputs
        # part 2: concat bottom-up path
        outs = []
        for i, j in zip(range(2, 4), range(1, -1, -1)):
            x = self.c3_convs[i](x)
            outs.append(x)
            x = self.gs_convs[i](x)
            later_x = up_lateral_feats[j]
            x = tf.concat([later_x, x], axis=-1)
        x = self.last_c3(x)
        outs.append(x)
        return outs[0], outs[1], outs[2]
