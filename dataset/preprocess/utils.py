import numpy as np
import cv2
from tensorpack.dataflow import *
import tensorflow as tf



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


class Tensorpack:
    def __init__(self, tensorpack_aug_chain):
        self.chains = tensorpack_aug_chain

    def do_augment(self, b_coors, b_raw_img, b_img_sizes, anno_height,
                   anno_width, do_ten_pack):
        '''
        Do random ratation, crop and resize by using "tensorpack" augmentation class.
            https://github.com/tensorpack/tensorpack
        1. resize the annotation to be the same as image size.
        2. rotating at the image center by small degree.
        3. correct the minus points due to rotation by function correct_out_point.
        4. cropping ratio is 0.8.
        5. resize back.
        '''
        # preprocess for different task annos
        b_coors = b_coors.numpy()
        b_img_sizes = b_img_sizes.numpy().astype(np.int32)
        b_h, b_w = b_img_sizes[:, 0], b_img_sizes[:, 1]
        b_trans_img = np.asarray(b_raw_img)
        b_annos, b_cates = b_coors[..., :-1], b_coors[..., -1:]
        trans_annos = b_annos[:, ::-1]
        aug_prob = .5
        tmp_img, tmp_annos = [], []

        for img, annos, cates, h, w, is_do in zip(b_trans_img, b_annos,
                                                  b_cates, b_h, b_w,
                                                  do_ten_pack):

            img = np.asarray(img).astype(np.float32)
            img = cv2.resize(img, (w, h))
            mask = ~np.isinf(annos)[:, 0, 0]
            annos, cates = annos[mask], cates[mask]
            random_prob = np.random.random_sample()
            if is_do:
                for tensorpack_aug in self.chains:
                    if tensorpack_aug == "WarpAffineTransform":
                        # do random rotation
                        if random_prob < aug_prob:
                            trans_img, trans_annos, cates = self.warp_affine_transform(
                                trans_img, annos, cates)

                    elif tensorpack_aug == "CropTransform":
                        # do random crop
                        if random_prob < aug_prob:
                            img, annos, cates = self.crop_transform(
                                img, annos, cates, h, w)
                    elif tensorpack_aug == "RandomPaste":
                        # do random paste
                        aug_prob = 1.
                        if random_prob < aug_prob:
                            img, annos, cates = self.random_paste(
                                img, annos, cates, h, w)
                    elif tensorpack_aug == "JpegNoise":
                        # do random jpeg noise
                        if random_prob < aug_prob:
                            img = self.jpeg_noise(img)

                    elif tensorpack_aug == "GaussianNoise":
                        # do random gaussian noise
                        if random_prob < aug_prob:
                            img = self.gaussian_noise(img)

                    elif tensorpack_aug == "SaltPepperNoise":
                        # do random salt pepper noise
                        if random_prob < aug_prob:
                            img = self.salt_pepper_noise(img)

            annos = np.concatenate([annos[:, :, ::-1], cates], axis=-1)
            n = len(annos)
            inf_array = np.full((15 - n, 2, 3), np.inf)
            annos = np.concatenate((annos, inf_array))
            tmp_annos.append([annos])
            img = cv2.resize(img, (640, 384))
            tmp_img.append([img])
        b_annos = np.concatenate(tmp_annos, axis=0)
        b_imgs = np.concatenate(tmp_img, axis=0)
        return b_annos, b_imgs

    def jpeg_noise(self, img):
        obj = imgaug.JpegNoise((10, 50))
        q = obj._get_augment_params(img)
        return obj._augment(img, q)

    def gaussian_noise(self, img):
        obj = imgaug.GaussianNoise(sigma=0.5)
        q = obj._get_augment_params(img)
        return obj._augment(img, q)

    def salt_pepper_noise(self, img):
        obj = imgaug.SaltPepperNoise(white_prob=0.01, black_prob=0.01)
        q = obj._get_augment_params(img)
        return obj._augment(img, q)

    def warp_affine_transform(self, img, annos, cates, max_tries):
        for _ in range(max_tries):
            rotation_angle = np.random.randint(low=-10, high=10)
            center = (w // 2, h // 2)
            mat = cv2.getRotationMatrix2D(center, rotation_angle, 1)
            affine = imgaug.WarpAffineTransform(mat, (w, h))
            img_out = affine.apply_image(img)
            annos_out = affine.apply_coords(annos)
            annos_out, cates_out = correct_out_point(annos_out, cates, 0, 0, h,
                                                     w, task)
            if annos_out.any():
                return img_out, annos_out, cates_out
        else:
            return img, annos, cates

    def random_paste(self, img, annos, cates, h, w):
        bg_ratio = 1.1 + np.random.random_sample() * 0.5
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
            return oringal_img, original_coor, original_cates, original_record_m

    def crop_transform(self, img, annos, cates, h, w):
        crop_ratio = np.random.randint(low=1, high=2) / 10.0
        h_crop, w_crop = int(round(h * crop_ratio)), int(round(w * crop_ratio))
        h1, w1 = np.random.randint(low=0, high=h - h_crop +
                                   1), np.random.randint(low=0,
                                                         high=w - w_crop + 1)
        crop_transform = imgaug.CropTransform(h1, w1, h_crop, w_crop)
        img_out = crop_transform.apply_image(img)
        annos_out = crop_transform.apply_coords(annos)
        annos_out = self.correct_out_point(annos_out, cates, 0, 0, h_crop,
                                           w_crop)
        if annos_out.any():
            resize_transform = imgaug.ResizeTransform(h_crop, w_crop, h, w,
                                                      cv2.INTER_CUBIC)
            img_out = resize_transform.apply_image(img_out)
            annos_out = resize_transform.apply_coords(annos_out)
            annos_out = self.correct_out_point(annos_out[..., :2],
                                               annos_out[..., -1:], 0, 0, h, w)
            if annos_out.any():
                return img_out, annos_out[..., :-1], annos_out[..., -1:]
            else:
                return img, annos, cates
        else:
            return img, annos, cates

    def correct_out_point(self, annos, cates, h1, w1, h2, w2):
        def gen_boolean_mask(check):
            check = np.all(check, axis=-1)
            return check

        annos[np.where(annos[:, 1] < h1)[0], 1] = h1
        annos[np.where(annos[:, 1] > h2)[0], 1] = h2
        annos[np.where(annos[:, 0] < w1)[0], 0] = w1
        annos[np.where(annos[:, 0] > w2)[0], 0] = w2
        annos = np.concatenate([annos, cates], axis=-1)
        _, _, c = annos.shape
        axis_check = annos[:, 0, :2] != annos[:, 1, :2]
        if np.any(axis_check == False):
            axis_check = gen_boolean_mask(axis_check)
        annos = annos[axis_check]
        annos = annos.reshape((-1, 2, c))
        if len(annos) > 0:
            height_width = annos[:, 1, :2] - annos[:, 0, :2]
            area_mask = height_width[:, 0] * height_width[:, 1] > 20
            if np.any(area_mask == False):
                area_mask = gen_boolean_mask(area_mask)
            annos = annos[area_mask]
            return annos
        else:
            return annos


@tf.function
def _flip(b_objs_kps, b_obj_wid, w, channel_names, do_flip):
    tmp_coors = []
    for i, channel_name in enumerate(channel_names):
        kp = b_objs_kps[:, :, i, :]
        if 'tl' in channel_name or 'top_left' in channel_name:
            flip_x = w - kp[:, :, 1] - b_obj_wid
        elif 'br' in channel_name or 'bottom_right' in channel_name:
            flip_x = w - kp[:, :, 1] + b_obj_wid
        elif 'st' in channel_name or 'side_top' in channel_name:
            flip_x = w - kp[:, :, 1] + b_obj_wid
        elif 'sb' in channel_name or 'side_bottom' in channel_name:
            flip_x = w - kp[:, :, 1] + b_obj_wid
        else:
            flip_x = w - kp[:, :, 1]
        flip_kp = tf.concat([kp[:, :, 0, None], flip_x[..., None]], axis=-1)
        tmp_coors.append([flip_kp])
    b_objs_kps = tf.concat(tmp_coors, axis=0)
    return tf.transpose(b_objs_kps, [1, 2, 0, 3])


def _flip_human(kps, w):
    dist = np.float32(w / 2 - kps[:, 1])
    x = np.float32(kps[:, 1] + 2 * dist)
    x[np.isnan(x)] = np.inf
    kps[:, 1] = x
    head = np.expand_dims(kps[0], axis=0)
    body = kps[1:]
    body = body.reshape((6, 2, 2))
    body = np.flip(body, axis=1)
    body = body.reshape((12, 2))
    assert len(head.shape) == len(body.shape), f'{head.shape}_{body.shape}'
    kps = np.concatenate([head, body])
    return kps


def _coor_clip(kps, h_thres, w_thres):
    # clipped the key point(x,y) on the coordinate
    inf_mask = tf.math.is_inf(kps)
    y, x = kps[..., 0], kps[..., 1]
    y = tf.where(y < 0., 0., y)
    x = tf.where(x < 0., 0., x)
    y = tf.expand_dims(tf.where(y < h_thres, y, h_thres), axis=-1)
    x = tf.expand_dims(tf.where(x < w_thres, x, w_thres), axis=-1)
    result = tf.concat([y, x], axis=-1)
    result = tf.where(inf_mask, np.inf, result)
    return result


def debug_kps(img, coors, origin_img_size, coor_resize, task, name='output.jpg'):
    img = img[..., ::-1]
    # origin_img_size = origin_img_size.numpy()
    if img.shape[0] != origin_img_size[0] or img.shape[1] != origin_img_size[1]:
        img = cv2.resize(img, tuple(origin_img_size[::-1]))
    # img = np.asarray(img)*255.0
    img = np.asarray(img) * 1.0
    coors = coors.numpy()
    resize_factor = (1 / coor_resize)
    for obj_kps in coors:
        if np.all(np.isinf(obj_kps)) or np.all(np.isinf(obj_kps)):
            continue
        if 'rel' in str(task):
            cates = obj_kps[..., -1:]
            obj_kps = obj_kps[..., :4].reshape([-1, 2])
        obj_kps = np.einsum('n d, d ->n d', obj_kps, resize_factor)
        obj_kps = (obj_kps + 0.5).astype(int)
        coor_n, yx = obj_kps.shape
        if coor_n == 2:
            tl, br = obj_kps
            img = cv2.rectangle(img, tuple(
                tl[::-1]), tuple(br[::-1]), (255, 0, 0), 3)
        elif coor_n == 3:
            center, tl, br = obj_kps
            img = cv2.circle(img, tuple(center[::-1]), 3, (0, 255, 0), -1)
        elif coor_n == 4:
            obj_tl, obj_br = obj_kps[0], obj_kps[1]
            sub_tl, sub_br = obj_kps[2], obj_kps[3]
            img = cv2.rectangle(img, tuple(
                obj_tl[::-1]), tuple(obj_br[::-1]), (255, 0, 0), 3)
            img = cv2.rectangle(img, tuple(
                sub_tl[::-1]), tuple(sub_br[::-1]), (0, 255, 0), 3)
            # img = cv2.circle(img, tuple(center[::-1]), 3, (0, 255, 0), -1)
    cv2.imwrite(name, img)
