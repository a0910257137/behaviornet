import tensorflow as tf
import numpy as np
from pprint import pprint


class Histogram(tf.keras.callbacks.Callback):
    def __init__(self, config, writers, update_freq, feed_inputs_display=None):
        super(Histogram, self).__init__()

        self.train_seen = 0
        self.eval_seen = 0
        self.update_freq = update_freq
        self.feed_inputs_display = feed_inputs_display
        self.config = config
        self.writers = writers

    def on_train_begin(self, logs=None):
        pass

    def on_test_begin(self, logs=None):
        pass

    def on_epoch_end(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.train_seen += 1
        if batch % self.update_freq == 0:
            with self.writers['train'].as_default():
                for tf_var in self.model.model.trainable_weights:
                    if tf_var.name in [
                            'hard_net/conv_block_1/dw_conv/depthwise_kernel:0',
                            'hard_net/conv_block_2/conv/kernel:0',
                            'conv_block_42/conv/kernel:0',
                            'conv_block_43/conv/kernel:0',
                            'conv_block_44/conv/kernel:0',
                            'conv_block_45/conv/kernel:0'
                    ]:
                        tf.summary.histogram(tf_var.name,
                                             tf_var.numpy(),
                                             step=self.train_seen)
            self.writers["train"].flush()

    def on_test_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.eval_seen += 1
        if batch % self.update_freq == 0:
            with self.writers['validation'].as_default():
                for tf_var in self.model.model.trainable_weights:
                    if tf_var.name in [
                            'hard_net/conv_block_1/dw_conv/depthwise_kernel:0',
                            'hard_net/conv_block_2/conv/kernel:0',
                            'conv_block_42/conv/kernel:0',
                            'conv_block_43/conv/kernel:0',
                            'conv_block_44/conv/kernel:0',
                            'conv_block_45/conv/kernel:0'
                    ]:
                        tf.summary.histogram(tf_var.name,
                                             tf_var.numpy(),
                                             step=self.eval_seen)
            self.writers["validation"].flush()
