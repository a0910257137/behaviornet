import tensorflow as tf
import os
import time
import numpy as np
import cv2
from pprint import pprint
import copy
from utils.io import load_BFM


class EmbeddingMap(tf.keras.callbacks.Callback):

    def __init__(self,
                 config,
                 writers,
                 train_datasets,
                 test_datasets,
                 update_freq,
                 feed_inputs_display=None):
        super(EmbeddingMap, self).__init__()

        self.train_seen = 0
        self.eval_seen = 0
        self.epoch = 0
        self.update_freq = update_freq
        self.feed_inputs_display = feed_inputs_display
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets
        self.config = config
        self.data_cfg = self.config.data_reader
        self.max_obj_num = self.data_cfg.max_obj_num
        self.resize_size = tf.cast(self.data_cfg.resize_size, tf.float32)
        self.cates = self._get_cates(self.data_cfg.tasks)
        self.writers = writers
        self.keys = ["center"]
        self.batch_size = config.models.batch_size
        #NOTE: Sperate by different task for 3dmm

        tdmm = self.config.models['3dmm']
        self.n_s, self.n_R = tdmm["n_s"], tdmm["n_R"]
        self.n_shp, self.n_exp = tdmm["n_shp"], tdmm["n_exp"]
        self.head_model = load_BFM(tdmm['model_path'])

        self.shapeMU = tf.cast(self.head_model['shapeMU'], tf.float32)
        self.shapePC = tf.cast(self.head_model['shapePC'][:, :self.n_shp],
                               tf.float32)
        self.expPC = tf.cast(self.head_model['expPC'][:, :self.n_exp],
                             tf.float32)
        pms = np.load(tdmm['pms_path'])
        self.train_mean_std = tf.cast(pms[:2, ], tf.float32)
        self.test_mean_std = tf.cast(pms[2:, ], tf.float32)

    def _get_cates(self, tasks):

        def read(path):
            with open(path) as f:
                return [x.strip() for x in f.readlines()]

        return {
            infos['preprocess']: read(infos["category_path"])
            for infos in tasks
        }

    # def on_train_begin(self, logs=None):
    #     self.train_iter_ds = iter(self.train_datasets)

    # def on_test_begin(self, logs=None):
    #     self.test_iter_ds = iter(self.test_datasets)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    # def on_epoch_end(self, batch, logs=None):
    #     self.train_iter_ds = iter(self.train_datasets)

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.train_seen += 1
        if batch % self.update_freq == 0:
            train_iter_ds = iter(self.train_datasets)
            batch_images, batch_labels = next(train_iter_ds)
            idx = np.random.choice(self.batch_size, 1, replace=False)
            img = tf.expand_dims(batch_images[idx[0]], axis=0)
            # consider the speed only choose oen image to feed into model
            fmaps = self.model.model(img, training=False)
            for lb_name in batch_labels:
                if 'heat_map' in lb_name:
                    pred_hms = tf.cast(fmaps[lb_name], tf.float16)
                    gt_hms = tf.expand_dims(tf.cast(
                        batch_labels[lb_name][idx[0]], tf.float16),
                                            axis=0)
                    self._summary_hms(self.writers['train'], img, gt_hms,
                                      pred_hms, lb_name, self.cates,
                                      self.train_seen)
                if 'param' in lb_name:
                    pred_hms = tf.cast(fmaps['obj_heat_map'], tf.float32)
                    pred_params = tf.cast(fmaps['obj_param_map'], tf.float32)
                    gt_params = tf.cast(batch_labels[lb_name], tf.float32)
                    self._summary_3dmm(self.writers['train'], gt_params,
                                       pred_params, pred_hms, self.train_seen)

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
                    self._summary_hms(self.writers['validation'], batch_images,
                                      gt_hms, pred_hms, lb_name, self.cates,
                                      self.eval_seen)

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
                        tf.summary.image(name='Predict/{}'.format(cate.upper()),
                                         data=pred_hm,
                                         step=saved_step,
                                         max_outputs=2)
                        tf.summary.image(name='Input/{}'.format(cate.upper()),
                                         data=gt_hm,
                                         step=saved_step,
                                         max_outputs=2)
                elif 'obj' in task_name and 'obj' in lb_name:
                    cates = task_cates[task_name]
                    for j, key in enumerate(self.keys):
                        pred_hm = tf.expand_dims(pred_hms[..., j], axis=-1)
                        gt_hm = tf.expand_dims(gt_hms[..., j], axis=-1)
                        tf.summary.image(name='Predict/{}'.format(key),
                                         data=pred_hm,
                                         step=saved_step,
                                         max_outputs=2)
                        tf.summary.image(name='Input/{}'.format(key),
                                         data=gt_hm,
                                         step=saved_step,
                                         max_outputs=2)
        writer.flush()

    def _summary_3dmm(self, writer, gt_params, pred_params, pred_hms,
                      saved_step):

        def top_k_loc(hms, k, h, w, c):
            flat_hms = tf.reshape(hms, [-1, h * w, c])
            flat_hms = tf.transpose(flat_hms, [0, 2, 1])
            scores, indices = tf.math.top_k(flat_hms, k, sorted=False)
            xs = tf.expand_dims(indices % w, axis=-1)
            ys = tf.expand_dims(indices // w, axis=-1)
            b_coors = tf.concat([ys, xs], axis=-1)
            return b_coors

        def apply_max_pool(data_in):
            kp_peak = tf.nn.max_pool(input=data_in,
                                     ksize=3,
                                     strides=1,
                                     padding='SAME',
                                     name='hm_nms')
            kps_mask = tf.cast(tf.equal(data_in, kp_peak), tf.float32)
            kps = data_in * kps_mask
            return kps

        pred_hms = apply_max_pool(pred_hms)

        b, h, w, c = [tf.shape(pred_hms)[i] for i in range(4)]
        top_k = 10
        b_coors = top_k_loc(pred_hms, top_k, h, w, c)
        b_idxs = tf.tile(
            tf.range(0, b, dtype=tf.int32)[:, tf.newaxis, tf.newaxis,
                                           tf.newaxis],
            [1, c, top_k, 1],
        )
        b_infos = tf.concat([b_idxs, b_coors], axis=-1)
        b_params = tf.gather_nd(pred_params, b_infos)
        b_scores = tf.gather_nd(pred_hms, b_infos)
        b_mask = tf.squeeze(b_scores > 0.5, axis=-1)
        b_params = b_params[b_mask]
        b_params = tf.reshape(b_params, (1, -1, tf.shape(b_params)[-1]))
        b_coors = tf.reshape(b_coors[b_mask], (1, -1, tf.shape(b_coors)[-1]))

        self.shapeMU
        self.shapePC
        self.expPC
        pred_params
        for n_coors, n_params in zip(b_coors, b_params):
            for coors, params in zip(n_coors, n_params):
                s, R, shp, exp = params[0], params[1:10], params[
                    10:60, np.newaxis], params[60:, np.newaxis]
                R = np.reshape(R, (3, 3))
                vertices = self.shapeMU + tf.linalg.matmul(
                    self.shapePC, shp) + tf.linalg.matmul(self.expPC, exp)
                vertices = tf.reshape(vertices,
                                      [3, int(tf.shape(vertices)[0] / 3)])
                vertices = tf.transpose(vertices.shape)
                print(vertices)
