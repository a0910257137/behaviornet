import tensorflow as tf

from monitor import logger
import time
import datetime


class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_train_batch_begin(self, batch, logs=None):
        self.start_time = time.time()

    def on_train_batch_end(self, batch, logs=None):
        now = datetime.datetime.now()
        self.end_time = time.time()
        train_info = '\n[TRAIN_LOSSES] %s: %d, %s: %d, %s: %.4fs, %s: %s; ' % (
            'epoch', self.epoch, 'step', batch, 'duration',
            self.end_time - self.start_time, 'Date', now)
        for key in logs.keys():
            train_info += '%s: %.6f, ' % (key, logs[key])
        logger.info(train_info)

    def on_test_batch_begin(self, batch, logs=None):
        self.start_time = time.time()

    def on_test_batch_end(self, batch, logs=None):
        self.end_time = time.time()
        now = datetime.datetime.now()
        eval_info = '\n[EVAL_LOSSES] %s: %d, %s: %d, %s: %.4fs,%s: %s; ' % (
            'epoch', self.epoch, 'step', batch, 'duration',
            self.end_time - self.start_time, 'Date', now)
        for key in logs.keys():
            eval_info += ' %s: %.6f, ' % (key, logs[key])
        logger.info(eval_info)
