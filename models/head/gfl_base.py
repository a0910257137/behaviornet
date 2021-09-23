import tensorflow as tf


class GFLBase(tf.keras.layers.Layer):
    def __init__(self, config, *args, **kwargs):
        super(GFLBase, self).__init__(*args, **kwargs)
        self.config = config

    def call(self):
        return
