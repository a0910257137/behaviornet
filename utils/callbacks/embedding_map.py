import tensorflow as tf
import os
import time
import numpy as np
import cv2
from pprint import pprint
import copy


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
        self.config = config
        # self.num_lnmks = self.config.models.head.pred_layer.num_landmarks
        self.num_lnmks = 42
        self.data_cfg = self.config.data_reader
        self.max_obj_num = self.data_cfg.max_obj_num
        self.resize_size = tf.cast(self.data_cfg.resize_size, tf.float32)
        self.cates = self._get_cates(self.data_cfg.tasks)
        self.train_writer = tf.summary.create_file_writer(
            os.path.join(self.config.summary.log_dir, 'train'))
        self.eval_writer = tf.summary.create_file_writer(
            os.path.join(self.config.summary.log_dir, 'validation'))
        self.writers = {
            'train': self.train_writer,
            'validation': self.eval_writer
        }
        self.keys = ['center', 'nose_lnmk']
        # self.keys = ['center']

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

                elif 'landmarks' in lb_name or 'keypoints' in lb_name:
                    pred_lnmks = tf.reshape(fmaps[lb_name],
                                            [-1, self.num_lnmks, 2])
                    pred_lnmks = np.einsum('b c d, d->b c d', pred_lnmks,
                                           self.resize_size)
                    pred_lnmks = tf.cast(pred_lnmks, tf.int32)
                    gt_lnmks = batch_labels[lb_name]
                    valid_mask = tf.math.reduce_all(tf.math.is_finite(
                        gt_lnmks[..., 0]),
                                                    axis=-1)
                    gt_lnmks = gt_lnmks[valid_mask]
                    gt_lnmks = np.einsum('b c d, d->b c d', gt_lnmks,
                                         self.resize_size)

                    gt_lnmks = tf.cast(gt_lnmks, tf.int32)
                    gt_lnmks = gt_lnmks.numpy()
                    pred_lnmks = pred_lnmks.numpy()
                    batch_images = batch_images.numpy() * 255
                    batch_gt_lnmks = gt_lnmks
                    batch_pred_lnmks = pred_lnmks
                    self._summary_lnmks(self.train_writer, batch_images,
                                        batch_gt_lnmks, batch_pred_lnmks,
                                        self.train_seen)

    def _summary_lnmks(self, writer, batch_images, batch_gt_lnmks,
                       batch_pred_lnmks, saved_step):
        with writer.as_default():
            original_imgs = np.asarray(batch_images[:2]).astype(np.uintc)
            tf.summary.image(name='Input',
                             data=tf.cast(original_imgs, tf.uint8),
                             step=saved_step,
                             max_outputs=2)
            gt_imgs, pred_imgs = [], []
            for image, gt_lnmks, pred_lnmks in zip(batch_images[:2],
                                                   batch_gt_lnmks[:2],
                                                   batch_pred_lnmks[:2]):
                for gt_lnmk, pred_lnmk in zip(gt_lnmks, pred_lnmks):
                    # green is ground truth
                    gt_image = cv2.circle(image, tuple(gt_lnmk[::-1]), 2,
                                          (0, 255, 0), -1)

                    pred_image = cv2.circle(image, tuple(pred_lnmk[::-1]), 2,
                                            (255, 0, 0), -1)
                gt_imgs.append(gt_image)
                pred_imgs.append(pred_image)
            gt_imgs = np.asarray(gt_imgs).astype(np.uintc)
            pred_imgs = np.asarray(pred_imgs).astype(np.uintc)
            gt_imgs = tf.cast(gt_imgs, tf.uint8)
            pred_imgs = tf.cast(pred_imgs, tf.uint8)

            tf.summary.image(name='Predict ',
                             data=pred_imgs,
                             step=saved_step,
                             max_outputs=2)
            tf.summary.image(name='Ground Truth',
                             data=gt_imgs,
                             step=saved_step,
                             max_outputs=2)
        writer.flush()

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
                    pred_hms = pred_hms[:10, ...]
                    gt_hms = gt_hms[:10, ...]
                    for j, key in enumerate(self.keys):
                        pred_hm = tf.expand_dims(pred_hms[..., j], axis=-1)
                        gt_hm = tf.expand_dims(gt_hms[..., j], axis=-1)
                        tf.summary.image(name='Predict ' + key,
                                         data=pred_hm,
                                         step=saved_step,
                                         max_outputs=2)
                        tf.summary.image(name='Input ' + key,
                                         data=gt_hm,
                                         step=saved_step,
                                         max_outputs=2)

                    # for j, cate in enumerate(cates):
                    #     pred_hm = tf.expand_dims(pred_hms[..., j], axis=-1)
                    #     gt_hm = tf.expand_dims(gt_hms[..., j], axis=-1)
                    #     tf.summary.image(name='Predict ' + cate.upper(),
                    #                      data=pred_hm,
                    #                      step=saved_step,
                    #                      max_outputs=2)
                    #     tf.summary.image(name='Input ' + cate.upper(),
                    #                      data=gt_hm,
                    #                      step=saved_step,
                    #                      max_outputs=2)

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
                elif 'landmarks' in lb_name or 'keypoints' in lb_name:
                    pred_lnmks = tf.reshape(fmaps[lb_name],
                                            [-1, self.num_lnmks, 2])
                    pred_lnmks = np.einsum('b c d, d->b c d', pred_lnmks,
                                           self.resize_size)
                    pred_lnmks = tf.cast(pred_lnmks, tf.int32)
                    gt_lnmks = batch_labels[lb_name]
                    valid_mask = tf.math.reduce_all(tf.math.is_finite(
                        gt_lnmks[..., 0]),
                                                    axis=-1)
                    gt_lnmks = gt_lnmks[valid_mask]
                    gt_lnmks = np.einsum('b c d, d->b c d', gt_lnmks,
                                         self.resize_size)

                    gt_lnmks = tf.cast(gt_lnmks, tf.int32)
                    gt_lnmks = gt_lnmks.numpy()
                    pred_lnmks = pred_lnmks.numpy()
                    batch_images = batch_images.numpy() * 255
                    batch_gt_lnmks = gt_lnmks
                    batch_pred_lnmks = pred_lnmks
                    self._summary_lnmks(self.eval_writer, batch_images,
                                        batch_gt_lnmks, batch_pred_lnmks,
                                        self.train_seen)
