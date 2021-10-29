import os
import tensorflow as tf
import numpy as np
import time
import cv2
from functools import partial

from .utils import *
from pprint import pprint
from .preprocess import OFFER_ANNOS_FACTORY
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
        for task_infos, infos in zip(self.task_configs, task_infos):
            task, branch_names, m_cates = task_infos['preprocess'], task_infos[
                'branches'], len(task_infos['cates'])
            b_coords, b_imgs, b_origin_sizes = self._parse_TFrecord(
                task, infos)
            b_coords, down_ratios = self._resize_coors(b_coords,
                                                       b_origin_sizes,
                                                       self.img_resize_size,
                                                       self.coors_down_ratio,
                                                       task)
            _multi_aug_funcs = Augmentation(self.config, self.img_resize_size,
                                            self.batch_size, task)

            b_imgs, b_coords, self.flip_probs = _multi_aug_funcs(
                b_imgs, b_coords, b_origin_sizes)
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
            elif task == "keypoint":
                print(b_coors)
                xxxx

        return tf.cast(b_imgs, dtype=tf.float32), targets

    def _resize_coors(self, annos, original_sizes, resize_size,
                      coors_down_ratio, task):
        img_down_ratio = resize_size / original_sizes
        img_down_ratio = tf.cast(img_down_ratio, tf.float32)
        annos, cates = annos[..., :-1], annos[..., -1:]
        if task == "obj_det":
            down_ratios = tf.cast(coors_down_ratio * img_down_ratio,
                                  tf.float32)
        elif task == "keypoint":
            #normalizer via image high and width for each y x
            down_ratios = img_down_ratio[::-1]
            # annos = tf.einsum('b n c d, b  d ->b n c d', annos, img_down_ratio)

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
        elif 'keypoint' in str(task):
            # B, N, 2, 2
            # make pseudo keypoint for top-left and bottom-right
            tl, br = b_objs_kps[:, :, 0, :], b_objs_kps[:, :, 1, :]
            b_obj_sizes = br - tl
        b_obj_sizes = tf.where(tf.math.is_nan(b_obj_sizes), np.inf,
                               b_obj_sizes)
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
            anno_shape = [-1, self.max_obj_num, 70, 3]
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
        # img = b_images.numpy()[0]
        # b_coords = b_coords.numpy()[0, 0]
        # b_coords = b_coords[..., :2]
        # b_origin_sizes = b_origin_sizes.numpy()[0]
        # resized_ratio = np.array([256., 256]) / b_origin_sizes
        # filp_imgs = tf.image.flip_left_right(b_images)
        # b_coords = np.einsum('c d, d->c d ', b_coords, resized_ratio[::-1])
        # np.save("kp.npy", b_coords)
        # cv2.imwrite('test.jpg', img[..., ::-1])
        # for kp in b_coords:
        #     kp = kp.astype(int)
        #     img = cv2.circle(img, tuple(kp), 3, (0, 255, 0), -1)
        # cv2.imwrite('non-flip_img.jpg', img[..., ::-1])
        return b_coords, b_images, b_origin_sizes
