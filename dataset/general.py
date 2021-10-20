import os
import tensorflow as tf
import multiprocessing
import math
from .general_task import GeneralTasks
from box import Box
from pprint import pprint
from glob import glob
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
                step_per_epoch = math.ceil(
                    len(filenames)) / (mirrored_strategy.num_replicas_in_sync *
                                       self.train_batch_size)
                ds = tf.data.TFRecordDataset(filenames,
                                             num_parallel_reads=threads)
            else:
                filenames = glob(os.path.join(task.test_folder, '*.tfrecords'))
                step_per_epoch = math.ceil(
                    len(filenames) / mirrored_strategy.num_replicas_in_sync *
                    self.test_batch_size)
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
        # datasets = datasets.apply(
        #     tf.data.experimental.prefetch_to_device("/gpu:0"))
        datasets = datasets.batch(batch_size, drop_remainder=True)

        # for ds in datasets:
        #     b_img, targets = self.gener_task.build_maps(batch_size, ds)
        #     b_img = b_img.numpy() * 255.
        #     b_idxs = targets["size_idxs"].numpy()
        #     b_size_vals = targets['size_vals'].numpy()
        #     for i, (img, idxs,
        #             size_vals) in enumerate(zip(b_img, b_idxs, b_size_vals)):
        #         valid_mask = np.all(np.isfinite(idxs), axis=-1)
        #         idxs, size_vals = idxs[valid_mask], size_vals[valid_mask]
        #         for idx, size_val in zip(idxs, size_vals):
        #             idx = idx
        #             size_val = size_val
        #             tl = (idx - size_val / 2).astype(int)
        #             br = (idx + size_val / 2).astype(int)
        #             img = cv2.rectangle(img, tuple(tl[::-1]), tuple(br[::-1]),
        #                                 (0, 255, 0), 2)
        #         cv2.imwrite("./output_" + str(i) + ".jpg", img[..., ::-1])

        datasets = datasets.map(
            lambda *x: self.gener_task.build_maps(batch_size, x),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        datasets = datasets.prefetch(tf.data.experimental.AUTOTUNE)
        return datasets, step_per_epoch

    def get_datasets(self, mirrored_strategy):
        return {
            "train": self._dataset(mirrored_strategy, True),
            "test": self._dataset(mirrored_strategy, False)
        }
