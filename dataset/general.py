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

from skimage import io

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
        #     targets = self.gener_task.build_maps(ds)
        datasets = datasets.map(
            lambda *x: self.gener_task.build_maps(x),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        datasets = datasets.prefetch(tf.data.experimental.AUTOTUNE)
        return datasets

    def get_datasets(self):
        return {"train": self._dataset(True), "test": self._dataset(False)}
