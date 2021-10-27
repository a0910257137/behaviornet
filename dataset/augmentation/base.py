import tensorflow as tf
import numpy as np
import cv2
import random
from functools import partial
from tensorpack.dataflow import *
from albumentations import Compose, CoarseDropout, GridDropout
import math
import copy


class Base:
    def __init__(self):
        self.clc_aug_funcs = {
            "bright": partial(tf.image.random_brightness, max_delta=0.1),
            "saturat": partial(tf.image.random_saturation,
                               lower=0.4,
                               upper=1.8),
            "hue": partial(tf.image.random_hue, max_delta=0.1),
            "contrast": partial(tf.image.random_contrast, lower=0.6,
                                upper=1.4),
        }

    def color_aug(self, b_imgs, aug_chains):
        if not aug_chains or len(aug_chains) == 0:
            return b_imgs
        for aug_name in self.config.augments.color_chains:
            b_imgs = self.clc_aug_funcs[aug_name](b_imgs)
        return b_imgs

    def album_augs(self, album_config, b_imgs):
        def album_aug(images):
            aug_imgs = np.asarray(
                list(map(lambda x: transforms(image=x)["image"],
                         images))).astype(np.uint8)
            return aug_imgs

        transform_list = []
        for aug_name in album_config:
            if aug_name == "gridmask":
                transform_list.append(
                    GridDropout(
                        ratio=album_config.gridmask.ratio,
                        unit_size_min=album_config.gridmask.unit_size_min,
                        unit_size_max=album_config.gridmask.unit_size_max,
                        random_offset=album_config.gridmask.random_offset))
            if aug_name == "cutout":
                transform_list.append(
                    CoarseDropout(max_holes=album_config.cutout.max_holes,
                                  max_height=album_config.cutout.max_height,
                                  max_width=album_config.cutout.max_width,
                                  min_holes=album_config.cutout.min_holes,
                                  min_height=album_config.cutout.min_height,
                                  min_width=album_config.cutout.min_width,
                                  fill_value=album_config.cutout.fill_value))
            transforms = Compose(transform_list)
        b_imgs = tf.numpy_function(func=album_aug, inp=[b_imgs], Tout=tf.uint8)
        return b_imgs

    def tensorpack_augs(self, b_coors, b_imgs, b_img_sizes, max_obj_num,
                        do_ten_pack, tensorpack_chains):
        f'''
            Do random ratation, crop and resize by using "tensorpack" augmentation class.
                https://github.com/tensorpack/tensorpack
            1. resize the annotation to be the same as original image size.
            2. rotating at the image center by small degree.
            3. correct the minus points due to rotation by function correct_out_point.
            4. cropping ratio is 0.8.
            5. resize back.
            return :  B, N, [tl, br], [y, x, c] [B, N, C, D]


        '''
        # preprocess for different task annos
        b_coors = np.asarray(b_coors).astype(np.float32)
        b_img_sizes = np.asarray(b_img_sizes).astype(np.int32)
        b_imgs = np.asarray(b_imgs).astype(np.uint8)
        _, h, w, c = b_imgs.shape
        aug_prob = .5
        tmp_coors = []
        tmp_imgs = []
        for img, coors, is_do in zip(b_imgs, b_coors, do_ten_pack):
            if not is_do:
                tmp_imgs.append(img)
                tmp_coors.append(coors)
                continue
            valid_mask = np.all(np.isfinite(coors), axis=-1)
            valid_mask = valid_mask[:, 0]
            coors = coors[valid_mask]
            annos = coors[..., :2]
            annos = annos[..., ::-1]
            cates = coors[..., -1:]
            coors = np.concatenate([annos, cates], axis=-1)
            for tensorpack_aug in tensorpack_chains:
                if tensorpack_aug == "CropTransform":
                    # do crop transform
                    if random.random() < aug_prob:
                        img, coors, cates = self.crop_transform(
                            img, coors, h, w)
                        coors = np.concatenate([coors, cates], axis=-1)

                elif tensorpack_aug == "RandomPaste":
                    # do random paste
                    if random.random() < aug_prob:
                        img, coors, cates = self.random_paste(img, coors, h, w)
                        coors = np.concatenate([coors, cates], axis=-1)
                elif tensorpack_aug == "WarpAffineTransform":
                    if random.random() < aug_prob:
                        img, coors, cates = self.warp_affine_transform(
                            img, coors, h, w)
                        coors = np.concatenate([coors, cates], axis=-1)
            tmp_imgs.append(img)
            # coors = coors[..., :2]
            # print(coors)
            # for coor in coors:
            #     # coor = coor[:, :2]
            #     tl = coor[0].astype(int)
            #     br = coor[1].astype(int)
            #     cv2.rectangle(img, tuple(tl), tuple(br), (0, 255, 0), 1)
            # cv2.imwrite('output.jpg', img[..., ::-1])
            # flip to y, x coordinate
            # coors = np.concatenate([coors[..., ::-1], cates], axis=-1)
            annos = coors[..., :2]
            annos = annos[..., ::-1]
            coors = np.concatenate([annos, coors[..., -1:]], axis=-1)

            n, c, d = coors.shape
            complement = np.empty([max_obj_num - n, c, d])
            complement.fill(np.inf)
            complement = complement.astype(np.float32)
            coors = np.concatenate([coors, complement], axis=0)
            tmp_coors.append(coors)
        b_coors = np.stack(tmp_coors)

        b_imgs = np.stack(tmp_imgs)
        return b_imgs, b_coors

    def crop_transform(self, img, coors, h, w):
        base_ratio = 0.7
        crop_ratio = np.random.randint(low=1, high=20) / 100.0
        crop_ratio = base_ratio + crop_ratio
        h_crop, w_crop = int(round(h * crop_ratio)), int(round(w * crop_ratio))

        h1, w1 = np.random.randint(low=0, high=h - h_crop +
                                   1), np.random.randint(low=0,
                                                         high=w - w_crop + 1)
        crop_transform = imgaug.CropTransform(h1, w1, h_crop, w_crop)
        img_out = crop_transform.apply_image(img)
        annos, cates = coors[..., :-1], coors[..., -1:]
        annos_out = crop_transform.apply_coords(annos)

        resize_transform = imgaug.ResizeTransform(h_crop, w_crop, h, w,
                                                  cv2.INTER_CUBIC)
        img_out = resize_transform.apply_image(img_out)
        annos_out = resize_transform.apply_coords(annos_out)
        annos_out = self.correct_out_point(annos_out, cates, 0, 0, h, w)
        # coors = annos_out[..., :2]
        # print(coors)
        # for coor in coors:
        #     # coor = coor[:, :2]
        #     tl = coor[0].astype(int)
        #     br = coor[1].astype(int)
        #     cv2.rectangle(img_out, tuple(tl), tuple(br), (0, 255, 0), 1)
        # cv2.imwrite('output.jpg', img_out[..., ::-1])
        # xxxx
        if annos_out.any():
            return img_out, annos_out[..., :-1], annos_out[..., -1:]
        else:
            return img, annos, cates

    def random_paste(self, img, coors, h, w):
        annos, cates = coors[..., :-1], coors[..., -1:]
        base_ratio = 1.0
        bg_ratio = base_ratio + np.random.random_sample() * 0.4
        bg_h, bg_w = round(h * bg_ratio), round(w * bg_ratio)
        obj = RandomPasetWithMeanBackground((bg_h, bg_w))
        l = obj.get_transform(img)
        if l == False:
            return img, annos, cates
        img_out = obj._impl(img, l)
        annos_out = annos + l
        if annos_out.any():
            resize_transform = imgaug.ResizeTransform(bg_h, bg_w, h, w,
                                                      cv2.INTER_CUBIC)
            img_out = resize_transform.apply_image(img_out)
            annos_out = resize_transform.apply_coords(annos_out)
            annos_out = self.correct_out_point(annos_out, cates, 0, 0, h, w)
            if annos_out.any():
                return img_out, annos_out[..., :-1], annos_out[..., -1:]
            else:
                return img, annos, cates
        else:
            return img, annos, cates

    def warp_affine_transform(self, img, coors, h, w):
        annos, cates = coors[..., :-1], coors[..., -1:]
        center_kps = (annos[:, 1, :] + annos[:, 0, :]) / 2
        wh = annos[:, 1, :] - annos[:, 0, :]
        rotation_angle = np.random.randint(low=-8, high=8)
        img_center = (w / 2, h / 2)
        mat = cv2.getRotationMatrix2D(img_center, rotation_angle, 1)
        affine = WarpAffineTransform(mat, (w, h))
        img_out = affine.apply_image(img)

        center_kps = affine.apply_coords(center_kps)
        tls = center_kps - wh / 2
        brs = center_kps + wh / 2
        annos_out = np.concatenate([tls[:, None, :], brs[:, None, :]], axis=-2)
        annos_out = self.correct_out_point(annos_out, cates, 0, 0, h, w)

        if annos_out.any():
            return img_out, annos_out[..., :2], annos_out[..., -1:]
        else:
            return img, annos, cates

    def correct_out_point(self, annos, cates, h1, w1, h2, w2):
        def gen_boolean_mask(check):
            check = np.all(check, axis=-1)
            return check

        valid_indice_x = np.where(annos[..., 0] < w1)
        valid_indice_y = np.where(annos[..., 1] < h1)
        annos[:, :, 0][valid_indice_x] = w1
        annos[:, :, 1][valid_indice_y] = h1
        valid_indice_x = np.where(annos[..., 0] > w2)
        valid_indice_y = np.where(annos[..., 1] > h2)
        annos[:, :, 0][valid_indice_x] = w2
        annos[:, :, 1][valid_indice_y] = h2

        annos = np.concatenate([annos, cates], axis=-1)
        _, _, c = annos.shape
        axis_check = annos[:, 0, :2] != annos[:, 1, :2]
        if np.any(axis_check == False):
            axis_check = gen_boolean_mask(axis_check)
        annos = annos[axis_check]
        annos = annos.reshape((-1, 2, c))
        if len(annos) > 0:
            height_width = annos[:, 1, :2] - annos[:, 0, :2]
            area_mask = height_width[:, 0] * height_width[:, 1] > 10
            annos = annos[area_mask]
            return annos
        else:
            return annos


class RandomPasetWithMeanBackground(imgaug.RandomPaste):
    def get_transform(self, img):
        img_shape = img.shape[:2]
        if self.background_shape[0] > img_shape[0] and self.background_shape[
                1] > img_shape[1]:
            y0 = np.random.randint(self.background_shape[0] - img_shape[0])
            x0 = np.random.randint(self.background_shape[1] - img_shape[1])
            l = int(x0), int(y0)
            return l
        else:
            return False

    def _impl(self, img, loc):
        x0, y0 = loc
        img_shape = img.shape[:2]
        self.background_shape = np.asarray(self.background_shape).astype(
            np.int32)
        self.background_shape = tuple(self.background_shape)
        background = self.background_filler.fill(self.background_shape, img)
        image_mean = img.mean(axis=(0, 1))
        background[:, :] = image_mean
        background[y0:y0 + img_shape[0], x0:x0 + img_shape[1]] = img
        return background


class WarpAffineTransform:
    def __init__(self,
                 mat,
                 dsize,
                 interp=cv2.INTER_LINEAR,
                 borderMode=cv2.BORDER_CONSTANT,
                 borderValue=0):
        self.mat = mat
        self.dsize = dsize
        self.interp = interp
        self.borderMode = borderMode
        self.borderValue = borderValue

    def apply_image(self, img):
        ret = cv2.warpAffine(img,
                             self.mat,
                             self.dsize,
                             flags=self.interp,
                             borderMode=self.borderMode,
                             borderValue=self.borderValue)
        if img.ndim == 3 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        return ret

    def apply_coords(self, coords):
        n, d = coords.shape
        expand_ones = np.ones((n, 1), dtype='f4')
        coords = np.concatenate((coords, expand_ones), axis=-1)
        rotate_matrices = self.mat.T
        coords = np.dot(coords, rotate_matrices)
        return coords


#TODO: add features: mosaic
