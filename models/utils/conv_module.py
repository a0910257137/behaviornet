import tensorflow as tf
from pprint import pprint


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 use_bias=True,
                 strides=1,
                 bias_initializer='zeros',
                 kernel_initializer=tf.keras.initializers.HeUniform(),
                 activation='relu',
                 norm_method='bn',
                 conv_mode='conv2d',
                 name=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.activation = activation
        self.norm_method = norm_method
        self.kernel_size = (kernel_size, kernel_size)
        self.strides = (strides, strides)

        if conv_mode == 'conv2d':
            self.conv = tf.keras.layers.Conv2D(
                filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                padding='same',
                name='conv')
        elif conv_mode == 'sp_conv2d':
            self.conv = tf.keras.layers.SeparableConv2D(
                filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                padding='same',
                name='sp_conv')

        if norm_method == 'bn':
            self.norm = tf.keras.layers.BatchNormalization(name='bn')

        if activation == 'relu':
            self.relu = tf.keras.layers.Activation(activation='relu',
                                                   name='act_relu')
        elif activation == 'hs':
            self.hs = tf.keras.layers.Activation(activation='swish',
                                                 name='act_hs')
        elif activation == 'leaky_relu':
            self.l_relu = tf.keras.layers.LeakyReLU()
        elif activation == 'sigmoid':
            self.sigmoid = tf.keras.layers.Activation(activation='sigmoid',
                                                      name='act_sigmoid')
        elif activation == 'softmax':
            self.softmax = tf.keras.layers.Activation(activation='softmax',
                                                      name='act_softmax')

    def call(self, input):
        output = self.conv(input)
        if self.norm_method == 'bn':
            output = self.norm(output)
        if self.activation == 'relu':
            output = self.relu(output)
        elif self.activation == 'leaky_relu':
            output = self.l_relu(output)
        elif self.activation == 'sigmoid':
            output = self.sigmoid(output)
        elif self.activation == 'softmax':
            output = self.softmax(output)
        elif self.activation == 'hs':
            output = self.hs(output)
        return output


class TransitionUp(tf.keras.layers.Layer):
    def __init__(self, filters, up_method, scale, concat, name, **kwargs):
        super().__init__(**kwargs)
        if up_method == 'bilinear':
            self.up_sample = tf.keras.layers.UpSampling2D(
                size=scale, interpolation='bilinear')
        elif up_method == 'nearest':
            self.up_sample = tf.keras.layers.UpSampling2D(
                size=(2, 2), interpolation='nearest')
        elif up_method == 'trans_deconv':
            self.up_sample = tf.keras.layers.Conv2DTranspose(
                filters=filters,
                kernel_size=(2, 2),
                strides=(scale, scale),
                use_bias=False,
                name='deconv_%s' % name)
        self.concat = concat

    def call(self, inputs, skip=None, **kwargs):
        out = self.up_sample(inputs)
        if self.concat:
            out = tf.concat([out, skip], axis=-1)
        return out


class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, input_chs, filters, kernel_size, e, s, is_squeeze, nl,
                 **kwargs):
        super(BottleNeck, self).__init__(**kwargs)
        """
            Bottleneck
            This function defines a basic bottleneck structure.
            # Arguments
                inputs: Tensor, input tensor of conv layer.
                filters: Integer, the dimensionality of the output space.
                kernel: An integer or tuple/list of 2 integers, specifying the
                    width and height of the 2D convolution window.
                e: Integer, expansion factor.
                    t is always applied to the input size.
                s: An integer or tuple/list of 2 integers,specifying the strides
                    of the convolution along the width and height.Can be a single
                    integer to specify the same value for all spatial dimensions.
                squeeze: Boolean, Whether to use the squeeze.
                nl: String, nonlinearity activation type.
            # Returns
                Output tensor.
        """
        self.alpha = 1.0
        self.nl = nl
        self.is_squeeze = is_squeeze
        tchannel = int(e)
        cchannel = int(self.alpha * filters)
        self.r = s == 1 and input_chs == filters
        self.bneck_conv = ConvBlock(filters=tchannel,
                                    kernel_size=1,
                                    strides=1,
                                    name=None,
                                    activation=self.nl,
                                    norm_method='bn')
        self.dw = DW(kernel_size=kernel_size, stride=s, n1=self.nl)
        if self.is_squeeze:
            self.se_blk = SE(tchannel)

        self.bneck_tran_conv = ConvBlock(filters=cchannel,
                                         kernel_size=1,
                                         strides=1,
                                         name='bneck_trans_conv',
                                         norm_method='bn',
                                         activation=None)
        if self.r:
            self.add = tf.keras.layers.Add()

    def call(self, x):
        inputs = x
        x = self.bneck_conv(x)
        x = self.dw(x)
        if self.is_squeeze:
            x = self.se_blk(x)
        x = self.bneck_tran_conv(x)
        if self.r:
            x = self.add([x, inputs])
        return x


class DW(tf.keras.layers.Layer):
    def __init__(self, kernel_size, stride, n1, **kwargs):
        super(DW, self).__init__(**kwargs)
        self.n1 = n1

        self.dw = tf.keras.layers.DepthwiseConv2D(kernel_size,
                                                  strides=(stride, stride),
                                                  depth_multiplier=1,
                                                  name='dw_conv',
                                                  padding='same')

        self.bn = tf.keras.layers.BatchNormalization(name='dw_bn')
        self.relu = tf.keras.layers.ReLU(max_value=6.0)

    def call(self, x):
        x = self.dw(x)
        x = self.bn(x)
        if self.n1 == 'relu':
            """
            Relu 6
            """
            x = self.relu(x)

        elif self.n1 == 'HS':
            """
            Hard swish
            """
            x = x * self.relu(x + 3.0) / 6.0
        return x


class SE(tf.keras.layers.Layer):
    def __init__(self, input_chs, **kwargs):

        super(SE, self).__init__(**kwargs)
        self.input_chs = input_chs
        self.glbap = tf.keras.layers.GlobalAvgPool2D()
        self.relu = tf.keras.layers.Dense(units=input_chs, activation='relu')
        self.hs = tf.keras.layers.Dense(units=input_chs,
                                        activation='hard_sigmoid')
        self.mlty = tf.keras.layers.Multiply()

    def call(self, x):
        inputs = x
        x = self.glbap(x)
        x = self.relu(x)
        x = self.hs(x)
        x = tf.reshape(x, [-1, 1, 1, self.input_chs])
        x = self.mlty([inputs, x])
        return x
