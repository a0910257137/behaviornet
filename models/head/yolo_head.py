import tensorflow as tf
from ..utils.conv_module import ConvBlock
from pprint import pprint


class YDetHead(tf.keras.Model):
    def __init__(self, config, *args, **kwargs):
        super(YDetHead, self).__init__(*args, **kwargs)
        self.config = config
        self.pred_cfg = self.config.head.pred_layer
        self.pred_branches = {k: None for k in self.pred_cfg if "P" in k}
        self.num_heads = len(self.pred_branches.keys())
        self.keys = ["P6", "P5", "P4"]
        self.head_convs = []
        for i in range(self.num_heads):
            self.head_convs.append(
                ConvBlock(filters=self.pred_cfg[self.keys[i]],
                          kernel_size=1,
                          strides=1,
                          kernel_initializer=tf.keras.initializers.HeNormal(),
                          norm_method=None,
                          name="P{}".format(6 - i)))

    @tf.function
    def call(self, feats):
        for i in range(self.num_heads):
            self.pred_branches[self.keys[i]] = self.head_convs[i](feats[i])
        return self.pred_branches