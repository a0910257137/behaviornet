import random
import tensorflow as tf
import numpy as np
from .mosaic import Mosaic
from .base import Base
from ..preprocess.utils import Tensorpack


class Augmentation(Base):
    def __init__(self, config, img_resize_size, batch_size):
        super(Augmentation, self).__init__()
        self.config = config
        self.augments = self.config.augments
        self.is_do_filp = self.augments.do_flip
        self.img_resize_size = img_resize_size
        self.batch_size = batch_size
        self.tensorpack = Tensorpack(self.augments.tensorpack_chains)

    def __call__(self, b_imgs, b_coors, b_origin_sizes):
        # comfirm coords data type

        b_coors = tf.cast(b_coors, tf.float32)
        #----------------------augmentations only influce image---------------------
        do_clc, flip_probs, do_ten_pack = self.random_param()
        # if self.is_do_filp:
        #     filp_imgs = tf.image.flip_left_right(b_imgs)
        #     tmp_logic = tf.tile(
        #         flip_probs[:, None, None, None],
        #         [1, self.img_resize_size[0], self.img_resize_size[1], 3])
        #     b_imgs = tf.where(tf.math.logical_not(tmp_logic), b_imgs,
        #                       filp_imgs)

        if len(self.augments.color_chains) != 0:
            aug_imgs = self.color_aug(b_imgs, self.augments.color_chains)
            tmp_logic = tf.tile(
                do_clc[:, None, None, None],
                [1, self.img_resize_size[0], self.img_resize_size[1], 3])
            b_imgs = tf.where(tf.math.logical_not(tmp_logic), b_imgs, aug_imgs)

        if len(self.augments.album_chains.keys()) != 0:
            b_imgs = self.album_augs(self.augments.album_chains, b_imgs)
        #----------------------b_ccords will be changed in augmentations---------------------
        # TODO: Implement rand past
        if len(self.augments.tensorpack_chains) != 0:
            b_imgs, b_coors = tf.py_function(
                self.tensorpack_augs,
                inp=[
                    b_coors, b_imgs, b_origin_sizes, self.img_resize_size[0],
                    self.img_resize_size[1], do_ten_pack,
                    self.augments.tensorpack_chains
                ],
                Tout=[tf.float32, tf.float32])
            imgs = tf.reshape(b_imgs, [
                self.batch_size, self.img_resize_size[0],
                self.img_resize_size[1], 3
            ])
        imgs = imgs / 255
        return b_imgs, b_coors, flip_probs

    def random_param(self):
        col_thre = 0.5 if len(self.augments.color_chains) else 0.0
        do_col = tf.random.uniform(
            shape=[self.batch_size], maxval=1, dtype=tf.float16) < col_thre
        flip_thre = 0.5 if self.augments.do_flip else 0.0

        do_flip = tf.random.uniform(
            shape=[self.batch_size], maxval=1, dtype=tf.float16) < flip_thre

        ten_pack_thres = 0.5 if len(self.augments.tensorpack_chains) else 0.0
        do_ten_pack = tf.random.uniform(shape=[self.batch_size],
                                        maxval=1,
                                        dtype=tf.float16) < ten_pack_thres
        return do_col, do_flip, do_ten_pack