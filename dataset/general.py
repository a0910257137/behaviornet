import os
import tensorflow as tf
import multiprocessing
from .general_task import GeneralTasks
from box import Box
from pprint import pprint
from glob import glob
import numpy as np
import cv2

threads = multiprocessing.cpu_count()


class GeneralDataset:
    def __init__(self, config):
        def read_cates(category_path):
            with open(category_path) as f:
                return [x.strip() for x in f.readlines()]

        self.config = config
        self.train_batch_size = config.train_batch_size
        self.test_batch_size = config.test_batch_size
        self.tasks = config.tasks
        self.epochs = config.epochs
        for task in config.tasks:
            task['cates'] = read_cates(task['category_path'])
        self.config = Box(self.config)
        self.gener_task = GeneralTasks(self.config)

    def _dataset(self, mirrored_strategy, is_train):
        datasets = []
        for task in self.config.tasks:
            if is_train:
                filenames = glob(os.path.join(task.train_folder,
                                              '*.tfrecords'))
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
        if not is_train:
            batch_size = mirrored_strategy.num_replicas_in_sync * self.test_batch_size
        batch_size = mirrored_strategy.num_replicas_in_sync * self.train_batch_size
        datasets = datasets.batch(batch_size, drop_remainder=True)
        # for ds in datasets:
        #     b_img, targets = self.gener_task.build_maps(batch_size, ds)
        #     offset_idxs = targets["offset_idxs"]
        #     offset_vals = targets['offset_vals'].numpy()
        #     valid_mask = tf.math.reduce_all(tf.math.is_finite(offset_idxs),
        #                                     axis=-1)
        #     valid_n = tf.math.reduce_sum(tf.cast(valid_mask, tf.float32),
        #                                  axis=-1)
        #     offset_vals = offset_vals[valid_mask]
        #     offset_vals = tf.reshape(offset_vals, [36, -1, 10])
        #     b_idx = tf.reshape(offset_idxs, [36, -1, 2])
        #     size_idxs = targets['size_idxs']
        #     valid_mask = tf.math.reduce_all(tf.math.is_finite(size_idxs),
        #                                     axis=-1)
        #     size_idxs = targets['size_idxs'].numpy()
        #
        #     b_img = b_img.numpy() * 255.

        #     mask = np.all(np.isfinite(size_idxs), axis=-1)
        #     size_idxs = size_idxs[mask]
        #     offset_vals = offset_vals[mask]
        #     img = b_img[0]
        #     offset_val = offset_vals[0]
        #     kp = size_idxs[0]
        #     for offset_val in offset_val:
        #         sh_kp = (kp + offset_val).astype(np.int32)[::-1]
        #         img = cv2.circle(img, tuple(sh_kp), 3, (0, 0, 255), -1)

        #     for kp in size_idxs:
        #         kp = kp.astype(int)[::-1]
        #         img = cv2.circle(img, tuple(kp), 1, (0, 255, 0), -1)
        #         cv2.imwrite("./output.jpg", img[..., ::-1])
        #         exit(1)
        xxxx
        datasets = datasets.map(
            lambda *x: self.gener_task.build_maps(batch_size, x),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        datasets = datasets.prefetch(tf.data.experimental.AUTOTUNE)
        return datasets

    def get_datasets(self, mirrored_strategy):
        return {
            "train": self._dataset(mirrored_strategy, True),
            "test": self._dataset(mirrored_strategy, False)
        }
