import tensorflow as tf
from pathlib import Path
import os
from monitor import logger
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from pprint import pprint


class CheckpointManagerCallback(tf.keras.callbacks.Callback):
    """
    Callback wraping `tf.train.CheckpointManager`.
    Restores previous checkpoint `on_train_begin`
    Example usage:
    ```python
    model = get_model(...)
    model.compile(optimizer=optimizer, ...)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(
        checkpoint, '/tmp/my_model', max_to_keep=5)
    callback = CheckpointManagerCallback(checkpoint, manager, period=1)
    model.fit(..., callbacks=[callbacks])
    ```
    """
    def __init__(self,
                 checkpoint,
                 manager,
                 model,
                 directory,
                 period=1,
                 save_on_train_end=True):
        self._manager = manager
        self._checkpoint = checkpoint
        self._period = period
        self._save_on_train_end = save_on_train_end
        self._restored = False
        self._epoch_count = None
        self._last_save = None
        self.directory = directory
        self.model = model

    # def on_batch_begin(self, epoch, logs=None):
    #     tf.saved_model.save(self.model.model,
    #                         '/aidata/anders/objects/hico/models/base')
    #     xxcxcx

    def on_epoch_end(self, epoch, logs=None):
        epochs_finished = epoch + 1
        self._epoch_count = epochs_finished
        if epochs_finished % self._period == 0:
            self._save()

    def on_train_end(self, logs=None):
        if self._save_on_train_end:
            self._save()

    def _save(self):
        if self._epoch_count is None:
            return
        if self._last_save != self._epoch_count:
            # save per epoch
            # if self._epoch_count % 5 == 0:
            # filename = "cp-{epoch:04d}.ckpt"
            # filepath = os.path.join(self.directory,
            #                         filename.format(epoch=self._epoch_count))
            # self.model.save_weights(filepath=filepath, save_format='tf')
            tf.saved_model.save(self.model.model, self.directory)
            # self._manager.save(self._epoch_count)
            self._last_save = self._epoch_count
