from re import X
import tensorflow as tf
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
        x, skip_connections = self.backbone(x)
        if self.neck is None:
            pass
        else:
            x = self.neck([x, skip_connections])
        preds = self.head(x)
        return preds
