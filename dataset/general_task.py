import os
import tensorflow as tf
import numpy as np
from functools import partial
from .utils import *
from pprint import pprint
from .preprocess import OFFER_ANNOS_FACTORY
from .preprocess.utils import Tensorpack
import time
from .augmentation.augmentation import Augmentation


class GeneralTasks:
    def __init__(self, config):
        self.config = config
        self.task_configs = config['tasks']
        self.model_name = self.config.model_name
        self.map_height, self.map_width = tf.cast(
            self.config.resize_size, tf.float32) * self.config.img_down_ratio
        self.is_do_filp = self.config.augments.do_flip
        self.img_resize_size = tf.cast(self.config.resize_size, dtype=tf.int32)
        self.coors_down_ratio = tf.cast(self.config.coors_down_ratio,
                                        dtype=tf.float32)
        self.tensorpack = Tensorpack(self.config.augments.tensorpack_chains)
        self.max_obj_num = self.config.max_obj_num
        self.features = {
            "origin_height": tf.io.FixedLenFeature([], dtype=tf.int64),
            "origin_width": tf.io.FixedLenFeature([], dtype=tf.int64),
            "b_images": tf.io.FixedLenFeature([], dtype=tf.string),
            "b_coords": tf.io.FixedLenFeature([], dtype=tf.string)
        }

    def build_maps(self, batch_size, task_infos):
        targets = {}
        self.batch_size = batch_size
        # self.do_clc, self.flip_probs, self.do_ten_pack = self.random_param()
        for task_infos, infos in zip(self.task_configs, task_infos):
            task, branch_names, m_cates = task_infos['preprocess'], task_infos[
                'branches'], len(task_infos['cates'])

            b_coords, b_imgs, b_origin_sizes = self._parse_TFrecord(
                task, infos)
            # b_coords, new_imgs = self._augments(b_coords, b_imgs,
            #                                     b_origin_sizes)
            b_coords, down_ratios = self._resize_coors(b_coords,
                                                       b_origin_sizes,
                                                       self.img_resize_size,
                                                       self.coors_down_ratio)
            _multi_aug_funcs = Augmentation(self.config, self.img_resize_size,
                                            self.batch_size)
            b_imgs, b_coords, self.flip_probs = _multi_aug_funcs(
                b_imgs, b_coords, b_origin_sizes)
            # b_coords, new_imgs = self._augments(b_coords, b_imgs,
            #                                     b_origin_sizes)
            offer_kps_func = OFFER_ANNOS_FACTORY[task]().offer_kps
            b_objs_kps, b_cates = b_coords[..., :-1], b_coords[..., -1][..., 0]
            b_obj_sizes = self._obj_sizes(b_objs_kps, task)
            b_round_kp_idxs, b_kp_idxs, b_coors, offset_vals = offer_kps_func(
                self.batch_size, b_objs_kps, self.map_height, self.map_width,
                b_obj_sizes, self.flip_probs, self.is_do_filp, branch_names)
            if task == "obj_det":
                b_hms = tf.py_function(self._draw_kps,
                                       inp=[
                                           b_round_kp_idxs, b_obj_sizes,
                                           self.map_height, self.map_width,
                                           m_cates, b_cates
                                       ],
                                       Tout=tf.float32)

                targets['size_idxs'] = b_round_kp_idxs
                targets['size_vals'] = tf.where(tf.math.is_nan(b_obj_sizes),
                                                np.inf, b_obj_sizes)
                targets['obj_heat_map'] = b_hms

        return tf.cast(b_imgs, dtype=tf.float32), targets

    def _resize_coors(self, annos, original_sizes, resize_size,
                      coors_down_ratio):
        img_down_ratio = resize_size / original_sizes
        img_down_ratio = tf.cast(img_down_ratio, tf.float32)
        down_ratios = tf.cast(coors_down_ratio * img_down_ratio, tf.float32)
        annos, cates = annos[..., :-1], annos[..., -1:]

        annos = tf.einsum('b n c d, b  d ->b n c d', annos, down_ratios)
        annos = tf.concat([annos, cates], axis=-1)

        return annos, down_ratios

    def random_param(self):
        col_thre = 0.5 if len(self.config.augments.color_chains) else 0.0
        do_col = tf.random.uniform(
            shape=[self.batch_size], maxval=1, dtype=tf.float32) < col_thre
        flip_thre = 0.5 if self.config.augments.do_flip else 0.0

        do_flip = tf.random.uniform(
            shape=[self.batch_size], maxval=1, dtype=tf.float32) < flip_thre

        ten_pack_thres = 0.5 if len(
            self.config.augments.tensorpack_chains) else 0.0
        do_ten_pack = tf.random.uniform(shape=[self.batch_size],
                                        maxval=1,
                                        dtype=tf.float32) < ten_pack_thres
        return do_col, do_flip, do_ten_pack

    def _augments(self, b_coors, imgs, img_sizes):
        def color_aug(imgs):
            aug_map = {
                "bright":
                partial(tf.image.random_brightness, max_delta=0.1),
                "saturat":
                partial(tf.image.random_saturation, lower=0.4, upper=1.8),
                "hue":
                partial(tf.image.random_hue, max_delta=0.1),
                "contrast":
                partial(tf.image.random_contrast, lower=0.6, upper=1.4),
            }
            if not self.config.augments.color_chains or len(
                    self.config.augments.color_chains) == 0:
                return imgs
            for aug_name in self.config.augments.color_chains:
                imgs = aug_map[aug_name](imgs)
            return imgs

        def filp_img(imgs):
            return tf.image.flip_left_right(imgs)

        if self.is_do_filp:
            filp_imgs = tf.image.flip_left_right(imgs)
            tmp_logic = tf.tile(
                self.flip_probs[:, None, None, None],
                [1, self.img_resize_size[0], self.img_resize_size[1], 3])
            imgs = tf.where(tf.math.logical_not(tmp_logic), imgs, filp_imgs)
        if len(self.config.augments.color_chains) != 0:
            aug_imgs = color_aug(imgs)
            tmp_logic = tf.tile(
                self.do_clc[:, None, None, None],
                [1, self.img_resize_size[0], self.img_resize_size[1], 3])
            imgs = tf.where(tf.math.logical_not(tmp_logic), imgs, aug_imgs)
        imgs = imgs / 255
        return b_coors, imgs

    def _draw_mask(self, b_objs_kps, b_cates, h, w, flip_probs, is_do_filp):
        def draw(objs_kps, cates):
            kp_mask = ~tf.math.is_inf(objs_kps)[:, 0, 0]
            objs_kps, cates = objs_kps[kp_mask], cates[kp_mask]

            mask = tf.zeros(shape=(h, w, 1))
            down_ratio = tf.constant(1 / 4, shape=[2])
            objs_kps = tf.einsum('n c d,  d ->n c d', objs_kps, down_ratio)
            for obj_kps, cate in zip(objs_kps, cates):

                obj_kps = tf.cast((obj_kps + .5), tf.int32)
                tl_kps, br_kps = obj_kps
                obj_h, obj_w = (br_kps - tl_kps)
                h_index, w_index = tf.range(obj_h), tf.range(obj_w)
                h_padding = tl_kps[0] + h_index
                w_padding = tl_kps[1] + w_index
                h_padding = tf.tile(h_padding[:, None],
                                    [1, w_padding.shape[0]])[..., None]

                h_padding = tf.transpose(h_padding, [1, 0, 2])
                w_padding = tf.tile(w_padding[:, None],
                                    [1, h_padding.shape[1]])[..., None]
                indexes = tf.concat([h_padding, w_padding], axis=-1)
                c = tf.tile(
                    tf.cast(0, tf.int32)[None, None, None],
                    [indexes.shape[0], indexes.shape[1], 1])

                indexes = tf.concat([indexes, c], axis=-1)
                fills = tf.ones(shape=indexes.shape[:2])
                mask = tf.tensor_scatter_nd_update(mask, indexes, fills)

            return mask

        b_fg_mask = tf.map_fn(lambda x: draw(x[0], x[1]),
                              (b_objs_kps, b_cates),
                              parallel_iterations=self.batch_size,
                              fn_output_signature=tf.float32)

        b_bg_mask = tf.ones(shape=(self.batch_size, h, w, 1))
        b_bg_mask = b_bg_mask - b_fg_mask
        return tf.concat([b_fg_mask, b_bg_mask], axis=-1)

    def _obj_sizes(self, b_objs_kps, task):
        if 'obj_det' in str(task):
            # B, N, 2, 2
            b_obj_sizes = b_objs_kps[:, :, 1, :] - b_objs_kps[:, :, 0, :]
        return b_obj_sizes

    def _rounding_offset(self, b_kp_idxs, b_round_kp_idxs):
        return b_kp_idxs - b_round_kp_idxs

    @tf.function
    def _one_hots(self, b_cates, m_cates):
        rel_classes = tf.zeros(shape=(self.batch_size, self.max_obj_num,
                                      m_cates),
                               dtype=tf.dtypes.float32)
        is_finites = ~tf.math.is_inf(b_cates)
        b_index = tf.where(is_finites)
        valid_counts = tf.where(is_finites, 1, 0)
        valid_counts = tf.math.reduce_sum(valid_counts)
        class_idx = tf.gather_nd(b_cates, b_index)
        b_index = tf.cast(b_index, tf.float32)
        b_index = tf.concat([b_index, class_idx[:, None]], axis=-1)
        b_index = tf.cast(b_index, tf.int32)
        one_hot_code = tf.tensor_scatter_nd_update(rel_classes, b_index,
                                                   tf.ones(shape=valid_counts))
        return one_hot_code

    def _draw_kps(self, b_round_kps, b_obj_sizes, h, w, m, b_cates):
        def draw(kps, sigmas, cates):
            mask = ~tf.math.is_inf(kps)[:, 0]
            kps, sigmas, cates = kps[mask], sigmas[mask], cates[mask]
            shape = [int(m), int(h), int(w)]
            hms = np.zeros(shape=shape, dtype=np.float32)
            for kp, sigma, cate in zip(kps, sigmas, cates):
                if tf.math.is_inf(sigma) or tf.math.reduce_any(
                        tf.math.is_inf(kp)):
                    continue
                hms[int(cate)] = draw_msra_gaussian(
                    hms[int(cate)], np.asarray(kp, dtype=np.float32),
                    np.asarray(sigma, dtype=np.float32))
            return hms

        b_sigmas = gaussian_radius(b_obj_sizes)
        b_hms = tf.map_fn(lambda x: draw(x[0], x[1], x[2]),
                          (b_round_kps, b_sigmas, b_cates),
                          back_prop=False,
                          fn_output_signature=tf.float32)
        return tf.transpose(b_hms, [0, 2, 3, 1])

    def _parse_TFrecord(self, task, infos):
        if task == "keypoint":
            anno_shape = [-1, self.max_obj_num, 4, 2]
        elif task == "obj_det":
            anno_shape = [-1, self.max_obj_num, 2, 3]
        elif task == "humankeypoint":
            anno_shape = [-1, self.max_obj_num, 13, 3]

        parse_vals = tf.io.parse_example(infos, self.features)
        b_images = tf.io.decode_raw(parse_vals['b_images'], tf.uint8)
        b_coords = tf.io.decode_raw(parse_vals['b_coords'], tf.float32)
        b_images = tf.reshape(b_images,
                              [-1, self.map_height, self.map_width, 3])
        b_coords = tf.reshape(b_coords, anno_shape)
        origin_height = tf.reshape(parse_vals['origin_height'], (-1, 1))
        origin_width = tf.reshape(parse_vals['origin_width'], (-1, 1))
        b_origin_sizes = tf.concat([origin_height, origin_width], axis=-1)
        b_origin_sizes = tf.cast(b_origin_sizes, tf.int32)
        return b_coords, b_images, b_origin_sizes
