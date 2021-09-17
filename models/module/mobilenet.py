import tensorflow as tf
from ..utils.conv_module import ConvBlock
from pprint import pprint


class MobileNet(tf.keras.Model):
    def __init__(self, config, backbone, head, **kwargs):
        super(MobileNet, self).__init__(**kwargs)
        self.config = config
        self.backbone = backbone
        self.head = head

    @tf.function
    def call(self, x):
        feats = self.backbone(x)
        preds = self.head(feats)
        return preds
