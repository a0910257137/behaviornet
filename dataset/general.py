import os
import tensorflow as tf
import multiprocessing
import numpy as np
import cv2
from .general_task import GeneralTasks
from box import Box
from pprint import pprint
from glob import glob
from utils.io import load_BFM
from utils.mesh.transform import *
from utils.mesh.render import *

threads = multiprocessing.cpu_count()


class GeneralDataset:

    def __init__(self, config, mirrored_strategy):

        def read_cates(category_path):
            with open(category_path) as f:
                return [x.strip() for x in f.readlines()]

        self.config = config
        self.batch_size = config.batch_size * mirrored_strategy.num_replicas_in_sync
        self.tasks = config.tasks
        self.epochs = config.epochs
        for task in config.tasks:
            task['cates'] = read_cates(task['category_path'])
        self.config = Box(self.config)
        self.gener_task = GeneralTasks(self.config, self.batch_size)

    def _dataset(self, is_train):
        datasets = []
        for task in self.config.tasks:
            if is_train:
                filenames = glob(os.path.join(task.train_folder,
                                              '*.tfrecords'))
                num_files = len(filenames)
                ds = tf.data.TFRecordDataset(filenames,
                                             num_parallel_reads=threads)
            else:
                filenames = glob(os.path.join(task.test_folder, '*.tfrecords'))
                num_files = len(filenames)
                ds = tf.data.TFRecordDataset(filenames,
                                             num_parallel_reads=threads)
            datasets.append(ds)
        datasets = tf.data.TFRecordDataset.zip(tuple(datasets))
        if self.config.shuffle:
            datasets = datasets.shuffle(buffer_size=10000)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        datasets = datasets.with_options(options)
        datasets = datasets.batch(self.batch_size, drop_remainder=True)
        # load BFM
        # head_model = load_BFM('/aidata/anders/3D-head/3DDFA/BFM/BFM.mat')
        # kpt_ind = head_model['kpt_ind']
        # X_ind_all = np.stack([kpt_ind * 3, kpt_ind * 3 + 1, kpt_ind * 3 + 2])
        # X_ind_all = tf.concat([
        #     X_ind_all[:, :17], X_ind_all[:, 17:27], X_ind_all[:, 36:48],
        #     X_ind_all[:, 27:36], X_ind_all[:, 48:68]
        # ],
        #                       axis=-1)
        # valid_ind = tf.reshape(tf.transpose(X_ind_all), (-1))
        # shapeMU = tf.gather(tf.cast(head_model['shapeMU'], tf.float32),
        #                     valid_ind).numpy()
        # shapeMU = tf.reshape(shapeMU, (tf.shape(shapeMU)[0] // 3, 3))
        # # mean = tf.math.reduce_mean(shapeMU, axis=-2, keepdims=True)
        # # shapeMU -= mean
        # shapeMU = tf.reshape(shapeMU, (tf.shape(shapeMU)[0] * 3, 1))
        # shapePC = tf.gather(tf.cast(head_model['shapePC'][:, :40], tf.float32),
        #                     valid_ind).numpy()
        # expPC = tf.gather(tf.cast(head_model['expPC'][:, :11], tf.float32),
        #                   valid_ind).numpy()
        # for ds in datasets:
        #     b_img, targets = self.gener_task.build_maps(ds)
        #     b_img = b_img.numpy() * 255.
        #     b_bboxes = targets['b_bboxes'].numpy()
        #     b_kps = targets['b_kps'].numpy()
        #     b_params = targets['params'].numpy()
        #     b_origin_sizes = targets['b_origin_sizes'].numpy()
        #     b_resized = b_origin_sizes[:, ::-1] / np.array([320., 320.])
        #     for i, (img, params, bboxes, kps, resized) in enumerate(
        #             zip(b_img, b_params, b_bboxes, b_kps, b_resized)):
        #         origin_size = (resized * np.array([320., 320.])).astype(
        #             np.int32)
        #         img = cv2.resize(img, tuple(origin_size))
        #         for param, bbox, kp in zip(params, bboxes, kps):
        #             if np.any(bbox == np.inf):
        #                 continue
        #             R = param[:9]
        #             shp = param[9:9 + 40]
        #             exp = param[9 + 40:]
        #             kp = kp * resized
        #             R = np.reshape(R, (3, 3))
        #             vertices = shapeMU + shapePC.dot(shp[:, None]) + expPC.dot(
        #                 exp[:, None])
        #             vertices = np.asarray(vertices)
        #             vertices = vertices.reshape(68, 3)
        #             vertices = (vertices.dot(R.T))[:, :2]
        #             lnmk_tl, lnmk_br = np.min(vertices,
        #                                       axis=-2), np.max(vertices,
        #                                                        axis=-2)
        #             bbox = np.einsum('n d, d -> n d', bbox, resized)
        #             tl, br = bbox[0], bbox[1]
        #             bbox_wh = br - tl
        #             lnmk_wh = (lnmk_br - lnmk_tl)
        #             scales = bbox_wh / lnmk_wh
        #             vertices = np.einsum('n d, d -> n d', vertices,
        #                                  scales) + kp[None, :]
        #             img = cv2.circle(img, tuple(kp.astype(np.int32)), 3,
        #                              (255, 0, 0), -1)
        #             vertices = vertices.astype(np.int32)
        #             for kp in vertices:
        #                 img = cv2.circle(img, tuple(kp), 5, (0, 255, 0), -1)
        #         cv2.imwrite("output_{}.jpg".format(i), img[..., ::-1])
        #     xxxx
        datasets = datasets.map(
            lambda *x: self.gener_task.build_maps(x),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        datasets = datasets.prefetch(tf.data.experimental.AUTOTUNE)
        return datasets, num_files

    def get_datasets(self):
        training_ds, num_training_ds = self._dataset(True)
        testing_ds, num_testing_ds = self._dataset(False)
        return {
            "train": training_ds,
            "test": testing_ds,
            "training_length": num_training_ds,
            "testing_length": num_testing_ds
        }
