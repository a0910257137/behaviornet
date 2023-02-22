import tensorflow as tf
import numpy as np
from .utils import *
from pprint import pprint
from .preprocess import OFFER_ANNOS_FACTORY
from .augmentation.augmentation import Augmentation
from .tdmm import MorphabelModel
import cv2


class GeneralTasks:

    def __init__(self, config, batch_size):
        self.config = config
        self.task_configs = config['tasks']
        self.model_name = self.config.model_name
        self.map_height, self.map_width = tf.cast(
            self.config.resize_size, tf.float32) * self.config.img_down_ratio
        self.is_do_filp = self.config.augments.do_flip
        self.img_resize_size = tf.cast(self.config.resize_size, dtype=tf.int32)
        self.max_obj_num = self.config.max_obj_num
        self.batch_size = batch_size
        self.img_channel = 3
        self.features = {
            "origin_height": tf.io.FixedLenFeature([], dtype=tf.int64),
            "origin_width": tf.io.FixedLenFeature([], dtype=tf.int64),
            "b_images": tf.io.FixedLenFeature([], dtype=tf.string),
            "b_coords": tf.io.FixedLenFeature([], dtype=tf.string),
            "is_masks": tf.io.FixedLenFeature([], dtype=tf.string)
        }
        self._multi_aug_funcs = Augmentation(self.config, self.batch_size,
                                             self.img_resize_size)
        self.MorphabelModel = MorphabelModel(self.batch_size, self.max_obj_num,
                                             self.config["3dmm"])

    def build_maps(self, task_infos):
        targets = {}
        for task_infos, infos in zip(self.task_configs, task_infos):
            self.num_lnmks = task_infos.num_lnmks
            task, m_cates = task_infos['preprocess'], len(task_infos['cates'])
            b_coords, b_face_masks, b_imgs, b_origin_sizes = self._parse_TFrecord(
                task, infos)

            b_imgs, b_coords = self._multi_aug_funcs(b_imgs, b_coords,
                                                     self.num_lnmks, task)
            offer_kps_func = OFFER_ANNOS_FACTORY[task]().offer_kps
            b_lnmks = tf.identity(b_coords)
            b_objs_kps, b_cates = b_coords[..., :-1], b_coords[..., -1][..., 0]
            b_obj_sizes = self._obj_sizes(b_objs_kps, task)
            b_round_kp_idxs, b_kp_idxs, b_coords, b_offset_vals = offer_kps_func(
                b_objs_kps, self.map_height, self.map_width)
            if task == "obj_det":
                targets['b_coords'] = b_coords[:, :, 1:, :]
                b_keypoints = tf.concat(
                    [b_coords[:, :, :1, :], b_coords[:, :, 3:4, :]], axis=-2)
                b_keypoints = b_coords[:, :, :1, :]
                b_hms = tf.py_function(self._draw_kps,
                                       inp=[
                                           b_keypoints, b_obj_sizes,
                                           self.map_height, self.map_width,
                                           m_cates, b_cates
                                       ],
                                       Tout=tf.float32)

                targets['size_idxs'] = b_coords[:, :, 0, :]
                targets['size_vals'] = tf.where(tf.math.is_nan(b_obj_sizes),
                                                np.inf, b_obj_sizes)
                targets['obj_heat_map'] = b_hms
                targets['offset_vals'] = b_offset_vals
                targets['offset_idxs'] = b_coords[:, :, 3, :]

            elif task == "tdmm":
                mean = tf.math.reduce_mean(b_lnmks[:, :, 2:, :2],
                                           axis=-2,
                                           keepdims=True)
                b_keypoints = mean
                targets['b_coords'] = tf.squeeze(mean, axis=-2)

                b_resized = tf.cast(b_origin_sizes, tf.float32) / tf.constant(
                    [192., 320.], dtype=tf.float32)
                b_lnmks = tf.einsum('b n k c, b c -> b n k c',
                                    b_lnmks[:, :, 2:, :2], b_resized)
                mean = tf.math.reduce_mean(b_lnmks, axis=-2, keepdims=True)
                b_lnmks -= mean
                b_lnmks = tf.where(tf.math.is_nan(b_lnmks), np.inf, b_lnmks)
                targets['b_lnmks'] = b_lnmks
                params = self.MorphabelModel.fit_points(b_lnmks, b_origin_sizes)
                targets['obj_heat_map'] = tf.py_function(
                    self._draw_kps,
                    inp=[
                        b_keypoints, b_face_masks, b_obj_sizes, self.map_height,
                        self.map_width, m_cates, b_cates
                    ],
                    Tout=tf.float32)
                targets['b_origin_sizes'] = b_origin_sizes
                targets['params'] = tf.cast(params, tf.float32)

        return tf.cast(b_imgs, dtype=tf.float32), targets

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

        b_fg_mask = tf.map_fn(lambda x: draw(x[0], x[1]), (b_objs_kps, b_cates),
                              parallel_iterations=self.batch_size,
                              fn_output_signature=tf.float32)

        b_bg_mask = tf.ones(shape=(self.batch_size, h, w, 1))
        b_bg_mask = b_bg_mask - b_fg_mask
        return tf.concat([b_fg_mask, b_bg_mask], axis=-1)

    @tf.function
    def _obj_sizes(self, b_objs_kps, task):
        if 'obj_det' in str(task) or 'tdmm' in str(task):
            # B, N, 2, 2
            b_obj_sizes = b_objs_kps[:, :, 1, :] - b_objs_kps[:, :, 0, :]
        b_obj_sizes = tf.where(tf.math.is_nan(b_obj_sizes), np.inf, b_obj_sizes)
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

    def _draw_kps(self, b_round_kps, b_face_masks, b_obj_sizes, h, w, m,
                  b_cates):

        b_sigmas = gaussian_radius(b_obj_sizes)
        b_hms = []
        with tf.device('CPU'):
            b_round_kps = b_round_kps.numpy()
            b_face_masks = b_face_masks.numpy()

            b_sigmas = b_sigmas.numpy()
            b_cates = b_cates.numpy()
            m = 2
            for kps, face_masks, sigmas, cates in zip(b_round_kps, b_face_masks,
                                                      b_sigmas, b_cates):
                mask = np.all(np.isfinite(kps), axis=-1)[:, 0]
                kps, sigmas, cates = kps[mask], sigmas[mask], cates[mask]
                face_masks = face_masks[mask]
                shape = [int(m), int(h), int(w)]
                hms = np.zeros(shape=shape, dtype=np.float32)
                for kp, face_mask, sigma, cate in zip(kps, face_masks, sigmas,
                                                      cates):
                    if np.isinf(sigma) or np.any(np.isinf(kp)):
                        continue
                    i = int(face_mask)
                    for i_kp in kp:
                        hms[int(i)] = draw_msra_gaussian(
                            hms[int(i)], np.asarray(i_kp, dtype=np.float32),
                            np.asarray(sigma, dtype=np.float32))
                    # for i, i_kp in enumerate(kp):
                    #     hms[int(i)] = draw_msra_gaussian(
                    #         hms[int(i)], np.asarray(i_kp, dtype=np.float32),
                    #         np.asarray(sigma, dtype=np.float32))
                b_hms.append(hms)
            b_hms = np.stack(b_hms)
        return np.transpose(b_hms, [0, 2, 3, 1])

    def _parse_TFrecord(self, task, infos):
        if task == "obj_det" or task == "tdmm":
            anno_shape = [-1, self.max_obj_num, self.num_lnmks, 3]
        b_coords, b_images, b_origin_sizes = None, None, None
        parse_vals = tf.io.parse_example(infos, self.features)
        b_images = tf.io.decode_raw(parse_vals['b_images'], tf.uint8)
        b_images = tf.reshape(
            b_images, [-1, self.map_height, self.map_width, self.img_channel])
        b_coords = tf.io.decode_raw(parse_vals['b_coords'], tf.float32)

        b_coords = tf.reshape(b_coords, anno_shape)
        b_face_masks = tf.io.decode_raw(parse_vals['is_masks'], tf.float32)
        b_face_masks = tf.reshape(b_face_masks, [-1, self.max_obj_num])
        origin_height = tf.reshape(parse_vals['origin_height'], (-1, 1))
        origin_width = tf.reshape(parse_vals['origin_width'], (-1, 1))
        b_origin_sizes = tf.concat([origin_height, origin_width], axis=-1)
        b_origin_sizes = tf.cast(b_origin_sizes, tf.int32)
        return b_coords, b_face_masks, b_images, b_origin_sizes
