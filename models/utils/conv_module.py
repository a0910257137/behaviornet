import tensorflow as tf
import numpy as np


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 use_bias=True,
                 strides=1,
                 name=None,
                 activation='relu',
                 norm_method='bn',
                 **kwargs):
        super().__init__(**kwargs)

        self.activation = activation
        self.norm_method = norm_method
        self.kernel_size = (kernel_size, kernel_size)
        self.strides = (strides, strides)

        self.conv = tf.keras.layers.Conv2D(
            filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            use_bias=use_bias,
            kernel_initializer=tf.keras.initializers.HeUniform(),
            padding='same',
            name='conv')
        if norm_method == 'bn': 
            self.norm = tf.keras.layers.BatchNormalization(name='bn')

        if activation == 'relu':
            self.relu = tf.keras.layers.Activation(activation='relu',
                                                   name='act_relu')
        elif activation == 'sigmoid':
            self.sigmoid = tf.keras.layers.Activation(activation='sigmoid',
                                                      name='act_sigmoid')
        elif activation == 'softmax':
            self.softmax = tf.keras.layers.Activation(activation='softmax',
                                                      name='act_softmax')

    def call(self, input, **kwargs):
        output = self.conv(input)
        if self.norm_method == 'bn':
            output = self.norm(output)
        if self.activation == 'relu':
            output = self.relu(output)
        elif self.activation == 'sigmoid':
            output = self.sigmoid(output)
        elif self.activation == 'softmax':
            output = self.softmax(output)
        return output


class TransitionUp(tf.keras.layers.Layer):
    def __init__(self, filters, up_method, scale, name, **kwargs):
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

    def call(self, inputs, skip=None, concat=True, **kwargs):
        out = self.up_sample(inputs)
        if concat:
            out = tf.concat([out, skip], axis=-1)
        return out
