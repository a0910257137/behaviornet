import tensorflow as tf
from ..utils.conv_module import ConvBlock
from ..utils.ops import *
from pprint import pprint


class ObjDet(tf.keras.Model):
    def __init__(self, config, backbone, neck, head, **kwargs):
        super(ObjDet, self).__init__(**kwargs)
        self.config = config
        self.backbone = backbone
        self.neck = neck
        self.head = head

    @tf.function
    def call(self, x):
        down_x, skip_connections = self.backbone(x)
        fpn_x = self.neck([down_x, skip_connections])
        preds = self.head(fpn_x)
        return preds
