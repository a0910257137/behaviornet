import tensorflow as tf
from ..utils.conv_module import ConvBlock
from pprint import pprint


class YDetHead(tf.keras.Model):
    def __init__(self, config, *args, **kwargs):
        super(YDetHead, self).__init__(*args, **kwargs)
        self.config = config
        self.num_heads = 3
        self.head_convs = []
        for i in range(self.num_heads):
            self.head_convs.append(
                ConvBlock(filters=48,
                          kernel_size=1,
                          strides=1,
                          kernel_initializer=tf.keras.initializers.HeNormal(),
                          norm_method=None,
                          name="P{}".format(6 - i)))

    @tf.function
    def call(self, feats):
        for i in range(self.num_heads):
            feats[i] = self.head_convs[i](feats[i])
        return feats
