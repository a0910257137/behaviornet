import tensorflow as tf
import math
from .conv_module import ConvBlock

conv_mode = "sp_conv2d"


class SPPF(tf.keras.layers.Layer):

    def __init__(self, out_channels, name, **kwargs):
        super().__init__(**kwargs)
        c_ = out_channels // 2
        self.cv1 = ConvBlock(c_,
                             kernel_size=1,
                             strides=1,
                             use_bias=False,
                             activation="relu")
        self.cv2 = ConvBlock(out_channels,
                             kernel_size=1,
                             strides=1,
                             use_bias=False,
                             activation="relu")
        self.max_pool2d = tf.keras.layers.MaxPool2D(pool_size=(5, 5),
                                                    strides=1,
                                                    padding='same')

    def __call__(self, x):
        x = self.cv1(x)
        y1 = self.max_pool2d(x)
        y2 = self.max_pool2d(y1)
        y3 = self.max_pool2d(y2)
        out = self.cv2(tf.concat([x, y1, y2, y3], axis=-1))
        return out
