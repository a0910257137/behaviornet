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
                filenames = glob(os.path.join(task.train_folder, '*.tfrecords'))
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
        # shapeMU = tf.reshape(shapeMU, (tf.shape(shapeMU)[0] * 3, 1))
        # shapePC = tf.gather(tf.cast(head_model['shapePC'][:, :50], tf.float32),
        #                     valid_ind).numpy()
        # expPC = tf.gather(tf.cast(head_model['expPC'][:, :29], tf.float32),
        #                   valid_ind).numpy()
        # for ds in datasets:
        #     b_img, targets = self.gener_task.build_maps(ds)
        #     b_img = b_img.numpy() * 255
        #     b_coords = targets['b_coords'].numpy()
        #     b_params = targets['params'].numpy()
        #     b_size_vals = targets['size_vals'].numpy()
        #     b_origin_sizes = targets['b_origin_sizes'].numpy()
        #     b_lnmks = targets['b_lnmks'].numpy()

        #     b_coords = b_coords.astype(np.float32)
        #     for i, (img, origin_sizes, n_params, n_coords, n_lnmks,
        #             n_size_vals) in enumerate(
        #                 zip(b_img, b_origin_sizes, b_params, b_coords, b_lnmks,
        #                     b_size_vals)):
        #         mask = np.all(np.isfinite(n_coords), axis=-1)
        #         n_coords = n_coords[mask]
        #         n_lnmks = n_lnmks[mask]
        #         n_size_vals = n_size_vals[mask]
        #         n_coords = np.reshape(n_coords, (-1, 2))
        #         n_size_vals = np.reshape(n_size_vals, [-1, 2])

        #         mask = np.all(np.isfinite(n_params), axis=-1)
        #         n_params = np.reshape(n_params[mask], (-1, n_params.shape[-1]))
        #         for params, coords, lnmks, size_vals in zip(
        #                 n_params, n_coords, n_lnmks, n_size_vals):
        #             R, shp, exp = params[:9], params[9:100 +
        #                                              9], params[100 + 9:100 +
        #                                                         9 + 29]
        #             t3d = params[..., -3:]
        #             R = np.reshape(R, (3, 3))
        #             vertices = shapeMU + shapePC.dot(shp[:, None]) + expPC.dot(
        #                 exp[:, None])
        #             vertices = np.asarray(vertices)
        #             vertices = vertices.reshape(68, 3)
        #             # vertices = s * vertices.dot(R.T)
        #             vertices = vertices.dot(R.T)
        #             tl, br = np.min(vertices[..., :2],
        #                             axis=0), np.max(vertices[..., :2], axis=0)
        #             resized = origin_sizes[::-1] / np.array([320., 192.])
        #             size_vals = size_vals[..., ::-1] * resized
        #             scales = size_vals / (br - tl)
        #             lnmks = lnmks[:, ::-1]
        #             lnmks = vertices[:, :2] * scales + coords[::-1] * resized
        #             img = cv2.resize(img,
        #                              tuple(origin_sizes[::-1]),
        #                              interpolation=cv2.INTER_AREA)
        #             for j, kp in enumerate(lnmks):
        #                 kp = kp.astype(np.int32)
        #                 if j < 17:
        #                     img = cv2.circle(img, tuple(kp), 3, (255, 0, 255),
        #                                      -1)
        #                 else:
        #                     img = cv2.circle(img, tuple(kp), 3, (0, 255, 0), -1)
        #             cv2.imwrite("./{}.jpg".format(i), img[..., ::-1])
        #     exit(1)
        # for ds in datasets:
        #     b_img, targets = self.gener_task.build_maps(ds)
        #     b_img = b_img.numpy() * 255.
        #     b_bboxes = targets['b_bboxes'].numpy()
        #     b_lnmks = targets['b_lnmks'].numpy()
        #     b_params = targets['params']
        #     b_origin_sizes = targets['b_origin_sizes'].numpy()
        #     resized = b_origin_sizes[:, ::-1] / np.array([320., 320.])
        #     resized = np.expand_dims(resized, axis=[1, 2])
        #     for i, (img, lnmks, bboxes,
        #             params) in enumerate(zip(b_img, b_lnmks, b_bboxes,
        #                                      b_params)):
        #         for bbox, lnmk, param in zip(bboxes, lnmks, params):
        #             if np.any(bbox == np.inf):
        #                 continue
        #             R, shp, exp = param[:9], param[9:59], param[59:]
        #             R = np.reshape(R, (3, 3))
        #             vertices = shapeMU + shapePC.dot(shp[:, None]) + expPC.dot(
        #                 exp[:, None])
        #             vertices = np.asarray(vertices)
        #             vertices = vertices.reshape(68, 3)
        #             vertices = vertices.dot(R.T)
        #             vertices = vertices[:, :2]
        #             tl, br = bbox.astype(np.int32)
        #             bbox_wh = br - tl
        #             img = cv2.rectangle(img, tuple(tl), tuple(br), (0, 255, 0),
        #                                 2)
        #             box_centers = (tl + br) / 2
        #             tl, br = np.min(vertices, axis=0), np.max(vertices, axis=0)
        #             lnmk_wh = br - tl
        #             scale = bbox_wh / lnmk_wh
        #             vertices = scale * vertices
        #             tl, br = np.min(vertices, axis=0), np.max(vertices, axis=0)
        #             lnmk_centers = (tl + br) / 2
        #             shifts = box_centers - lnmk_centers
        #             vertices = vertices + shifts[None, :]
        #             vertices = vertices.astype(np.int32)
        #             for kp in vertices:
        #                 img = cv2.circle(img, tuple(kp), 2, (0, 255, 0), -1)
        #         cv2.imwrite("output_{}.jpg".format(i), img[..., ::-1])
        #     xxx
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
