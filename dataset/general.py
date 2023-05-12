import os
import tensorflow as tf
import multiprocessing
import numpy as np
import cv2
from .general_task import GeneralTasks
from box import Box
from pprint import pprint
from glob import glob
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
                ds = tf.data.TFRecordDataset(filenames,
                                             num_parallel_reads=threads)
            else:
                filenames = glob(os.path.join(task.test_folder, '*.tfrecords'))
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
        # for ds in datasets:
        #     b_img, targets = self.gener_task.build_maps(ds)
        #     b_img = b_img.numpy() * 255.
        #     b_keypoints = targets['b_keypoints'].numpy()
        #     b_bboxes = targets['b_bboxes'].numpy()
        #     b_scale_factors = targets['b_scale_factors'].numpy()
        #     b_origin_sizes = targets['b_origin_sizes'].numpy()
        #     resized = b_origin_sizes[:, ::-1] / np.array([640., 640.])
        #     resized = np.expand_dims(resized, axis=[1, 2])
        #     # b_bboxes = b_bboxes[..., ::-1]
        #     b_keypoints = b_keypoints[np.all(np.isfinite(b_keypoints), axis=-1)]
        #     b_keypoints = np.reshape(b_keypoints, [48, 5, -1])
        #     b_bboxes = b_bboxes[np.all(np.isfinite(b_bboxes), axis=-1)]
        #     b_bboxes = np.reshape(b_bboxes, [48, 2, 2])
        #     for img, origin_sizes, bboxes, keypoints in zip(
        #             b_img, b_origin_sizes, b_bboxes, b_keypoints):
        #         tl, br = bboxes.astype(np.int32)
        #         img = cv2.rectangle(img, tuple(tl), tuple(br), (0, 255, 0), 3)
        #         # img = cv2.resize(img,
        #         #                  tuple(origin_sizes[::-1]),
        #         #                  interpolation=cv2.INTER_AREA)
        #         for j, kp in enumerate(keypoints):
        #             kp = kp.astype(np.int32)[:2]
        #             img = cv2.circle(img, tuple(kp), 2, (255, 0, 255), -1)
        #         cv2.imwrite("output.jpg", img[..., ::-1])
        #         xxx
        datasets = datasets.map(
            lambda *x: self.gener_task.build_maps(x),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        datasets = datasets.prefetch(tf.data.experimental.AUTOTUNE)
        return datasets

    def get_datasets(self):
        return {"train": self._dataset(True), "test": self._dataset(False)}
