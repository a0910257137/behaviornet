import os
from .general_task import GeneralTasks
from box import Box
from pprint import pprint
import tensorflow as tf
import numpy as np
import cv2
import time


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
                ds = tf.data.Dataset.list_files(task.train_folder + "/*.npy",
                                                shuffle=False)
            else:
                ds = tf.data.Dataset.list_files(task.test_folder + "/*.npy",
                                                shuffle=False)
            datasets.append(ds)
        datasets = tf.data.Dataset.zip(tuple(datasets))
        if self.config.shuffle:
            datasets = datasets.shuffle(buffer_size=10000)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        datasets = datasets.with_options(options)
        if not is_train:
            batch_size = mirrored_strategy.num_replicas_in_sync * self.test_batch_size
            # batch_size = self.test_batch_size
        # batch_size = self.train_batch_size
        batch_size = mirrored_strategy.num_replicas_in_sync * self.train_batch_size
        datasets = datasets.batch(batch_size, drop_remainder=True)
        # for ds in datasets:
        #     b_img, targets = self.gener_task.build_maps(batch_size, ds)
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
