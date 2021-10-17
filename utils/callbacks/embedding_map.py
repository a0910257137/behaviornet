import tensorflow as tf
import os
import time
import numpy as np
import io
import cv2
import matplotlib.pyplot as plt
from pprint import pprint


class EmbeddingMap(tf.keras.callbacks.Callback):
    def __init__(self,
                 config,
                 train_datasets,
                 test_datasets,
                 update_freq,
                 feed_inputs_display=None):
        super(EmbeddingMap, self).__init__()

        self.train_seen = 0
        self.eval_seen = 0
        self.update_freq = update_freq
        self.feed_inputs_display = feed_inputs_display
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets
        self.cates = self._get_cates(config.data_reader.tasks)
        self.train_writer = tf.summary.create_file_writer(
            os.path.join(config.summary.log_dir, 'train'))
        self.eval_writer = tf.summary.create_file_writer(
            os.path.join(config.summary.log_dir, 'validation'))
        self.writers = {
            'train': self.train_writer,
            'validation': self.eval_writer
        }

    def _get_cates(self, tasks):
        def read(path):
            with open(path) as f:
                return [x.strip() for x in f.readlines()]

        return {
            infos['preprocess']: read(infos["category_path"])
            for infos in tasks
        }

    def on_train_begin(self, logs=None):
        self.train_iter_ds = iter(self.train_datasets)

    def on_test_begin(self, logs=None):
        self.test_iter_ds = iter(self.test_datasets)

    def on_epoch_end(self, batch, logs=None):
        self.train_iter_ds = iter(self.train_datasets)

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.train_seen += 1
        if batch % self.update_freq == 0:
            batch_images, batch_labels = next(self.train_iter_ds)

            fmaps = self.model.model(batch_images, training=False)
            for lb_name in batch_labels:
                if 'heat_map' in lb_name:
                    pred_hms = tf.cast(fmaps[lb_name], tf.float16)
                    gt_hms = tf.cast(batch_labels[lb_name], tf.float16)
                    self._summary_hms(self.train_writer, batch_images, gt_hms,
                                      pred_hms, lb_name, self.cates,
                                      self.train_seen)

    def _summary_hms(self, writer, batch_images, gt_hms, pred_hms, lb_name,
                     task_cates, saved_step):
        """
            Args:
                batch_images: g.t. images
                gt_hms:  g.t. labeld hms
                pred_hms: pred. hms
                task_cates:   {task_name1:[], task_name2:[]}
                saved_step: int saved step
        """
        with writer.as_default():
            tf.summary.image(name='Input',
                             data=batch_images,
                             step=saved_step,
                             max_outputs=2)
            # list type for task_cates
            for task_name in task_cates:
                if 'rel' in task_name and 'rel' in lb_name:
                    cates = task_cates[task_name]
                    for j, cate in enumerate(cates):
                        pred_hm = tf.expand_dims(pred_hms[..., j], axis=-1)
                        gt_hm = tf.expand_dims(gt_hms[..., j], axis=-1)
                        tf.summary.image(name='Predict ' + cate.upper(),
                                         data=pred_hm,
                                         step=saved_step,
                                         max_outputs=2)
                        tf.summary.image(name='Input ' + cate.upper(),
                                         data=gt_hm,
                                         step=saved_step,
                                         max_outputs=2)
                elif 'obj' in task_name and 'obj' in lb_name:
                    cates = task_cates[task_name]
                    for j, cate in enumerate(cates):
                        pred_hm = tf.expand_dims(pred_hms[..., j], axis=-1)
                        gt_hm = tf.expand_dims(gt_hms[..., j], axis=-1)
                        tf.summary.image(name='Predict ' + cate.upper(),
                                         data=pred_hm,
                                         step=saved_step,
                                         max_outputs=2)
                        tf.summary.image(name='Input ' + cate.upper(),
                                         data=gt_hm,
                                         step=saved_step,
                                         max_outputs=2)

        writer.flush()

    def on_test_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.eval_seen += 1
        if batch % self.update_freq == 0:
            batch_images, batch_labels = next(self.test_iter_ds)
            fmaps = self.model.model(batch_images, training=False)
            for lb_name in batch_labels:
                if 'heat_map' in lb_name:
                    pred_hms = tf.cast(fmaps[lb_name], tf.float32)
                    gt_hms = tf.cast(batch_labels[lb_name], tf.float32)
                    self._summary_hms(self.eval_writer, batch_images, gt_hms,
                                      pred_hms, lb_name, self.cates,
                                      self.eval_seen)
