import re
import numpy as np
from itertools import permutations
import random
import cv2
import copy


def draw_l_box(imgs, obj_kps, clr=(0, 255, 0)):
    """
    imgs = [img]
    kps: [[[tl, br, st, sb], [tl, br, st, sb]],
          [[tl, br, st, sb], [tl, br]]]
    """
    def is_at_left_side(kp):
        if kp[0][1] < kp[2][1]:
            return True
        return False

    def is_3d(kp):
        if len(kp) > 2:
            return True
        return False

    def draw_3d(img, kp):
        tl, br = tuple(kp[0][::-1].astype(np.int32)), tuple(kp[1][::-1].astype(
            np.int32))
        st, sb = tuple(kp[2][::-1].astype(np.int32)), tuple(kp[3][::-1].astype(
            np.int32))
        img = cv2.rectangle(img, tl, br, clr, 1)
        img = cv2.line(img, st, sb, clr, 1)
        if is_at_left_side(kp):
            tr = (br[0], tl[1])
            img = cv2.line(img, tr, st, clr, 1)
            img = cv2.line(img, br, sb, clr, 1)
        else:
            bl = (tl[0], br[1])
            img = cv2.line(img, tl, st, clr, 1)
            img = cv2.line(img, bl, sb, clr, 1)
        return img

    def draw_2d(img, kp):
        tl, br = tuple(kp[0][::-1].astype(np.int32)), tuple(kp[1][::-1].astype(
            np.int32))
        img = cv2.rectangle(img, tl, br, clr, 1)
        return img

    # obj_kps = np.asarray(obj_kps)
    result = []
    count = 1
    for img, obj_kp in zip(imgs, obj_kps):
        for kp in obj_kp:
            kp = np.asarray(kp)
            if is_3d(kp):
                img = draw_3d(img, kp)
            else:
                img = draw_2d(img, kp)
        result.append(img)
        count += 1
    return result


def draw_box2d(imgs, obj_kps, target_dict, clr=(0, 255, 0)):
    def draw_2d(img, kp, category):
        tl, br = tuple(kp[:2][::-1].astype(np.int32)), tuple(
            kp[2:4][::-1].astype(np.int32))
        img = cv2.rectangle(img, tl, br, clr, 2)
        center = (tl[0] + 20, tl[1] - 20)
        img = cv2.putText(img, category, center, cv2.FONT_HERSHEY_SIMPLEX, 1,
                          (0, 0, 225), 2, cv2.LINE_AA)
        return img

    result = []
    b_bboxes = np.asarray(obj_kps)

    for img, bboxes in zip(imgs, b_bboxes):

        for bbox in bboxes:
            if any(np.isinf(bbox)):
                continue
            category_index = int(bbox[..., -1])
            category = target_dict[category_index]
            score = bbox[..., -2]
            kp = bbox[:4]
            img = draw_2d(img, kp, category)
            result.append(img)
    return result


def draw_rel(imgs,
             list_objs_rels,
             tar_obj_cates,
             tar_rel_cates,
             clr=(0, 255, 0)):
    def draw_2d(img, kp, img_size, category, clr=(0, 255, 0)):
        kp = tuple(kp[:2][::-1].astype(np.int32))
        h, w = img_size
        shift_x, shift_y = -55, 25
        if kp[0] - 55 < 0:
            shift_x = 45
        if kp[1] + 25 > h:
            shift_y = -25
        center = (kp[0] + shift_x, kp[1] + shift_y)
        img = cv2.circle(img, (kp), 4, clr, -1)
        img = cv2.putText(img, category, center, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                          clr, 2, cv2.LINE_AA)
        return img

    def rel_position(img, sub, obj, img_size, category, clr=(255, 0, 0)):
        kp = (sub + obj) / 2
        h, w = img_size
        kp = tuple(kp[:2][::-1].astype(np.int32))
        img = cv2.circle(img, (kp), 4, clr, -1)
        center = (kp[0] + 10, kp[1] - 10)
        img = cv2.putText(img, category, center, cv2.FONT_HERSHEY_SIMPLEX, 1,
                          clr, 2, cv2.LINE_AA)
        return img

    def draw_box(img, kp, img_size, score, clc, category):
        h, w = img_size
        shift_x, shift_y = -55, 25
        if kp[0] - 55 < 0:
            shift_x = 45
        if kp[1] + 25 > h:
            shift_y = -25
        kp = tuple(kp[:2][::-1].astype(np.int32))
        img = cv2.circle(img, kp, 3, clc, -1)
        center = (kp[0] + shift_x, kp[1] + shift_y)
        img = cv2.putText(img, category, center, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                          clc, 2, cv2.LINE_AA)
        return img

    def draw_relation(img, kp, img_size, score, shift, kp_size, clc, category):
        h, w = img_size
        shift_x, shift_y = shift
        if kp[0] + shift_x < 0:
            # shift_x is negative
            shift_x = -shift_x
        if kp[1] + shift_y > h:
            # shift_x is positive
            shift_y = -shift_y
        kp = tuple(kp[:2][::-1].astype(np.int32))
        clc = (int(clc[0]), int(clc[1]), int(clc[2]))
        img = cv2.circle(img, kp, kp_size, clc, -1)
        center = (kp[0] + shift_x, kp[1] + shift_y)
        if 'person' in category:
            img = cv2.putText(img, category, center, cv2.FONT_HERSHEY_SIMPLEX,
                              0.6, clc, 2, cv2.LINE_AA)
        else:
            img = cv2.putText(img, category, center, cv2.FONT_HERSHEY_SIMPLEX,
                              0.6, clc, 2, cv2.LINE_AA)
        return img

    def draw_line(img, sub_kp, obj_kp, obj_cate, rel_cate, output_string,
                  draw_position, clc):
        clc = (int(clc[0]), int(clc[1]), int(clc[2]))
        sub_kp = tuple(sub_kp[:2][::-1].astype(np.int32))
        obj_kp = tuple(obj_kp[:2][::-1].astype(np.int32))

        img = cv2.circle(img, sub_kp, 3, (255, 0, 0), -1)
        img = cv2.circle(img, obj_kp, 3, (0, 255, 0), -1)
        text_pos = (sub_kp[0] + 15, sub_kp[1] + 15)
        img = cv2.putText(img, rel_cate, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                          0.8, clc, 2, cv2.LINE_AA)
        text_pos = tuple([obj_kp[0] + 15, obj_kp[1] + 20])
        img = cv2.putText(img, obj_cate, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                          0.8, clc, 2, cv2.LINE_AA)
        img = cv2.line(img, sub_kp, obj_kp, clc, thickness=2)
        return img

    def draw_point_rels(img, sub_kp, obj_kp, obj_cate, rel_cate, clc):
        clc = (int(clc[0]), int(clc[1]), int(clc[2]))
        sub_kp = tuple(sub_kp[:2][::-1].astype(np.int32))
        img = cv2.circle(img, sub_kp, 3, (255, 0, 0), -1)
        text_pos = (sub_kp[0] + 15, sub_kp[1] + 20)
        img = cv2.putText(img, rel_cate, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                          0.8, clc, 2, cv2.LINE_AA)
        text_pos = (obj_kp[0] + 15, obj_kp[1] + 20)
        img = cv2.putText(img, obj_cate, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                          0.8, clc, 2, cv2.LINE_AA)
        return img

    def draw_bbox_rels(img, sub_kp, obj_kp, sub_cate, obj_cate, rel_cate):
        sub_tl_kp = tuple(sub_kp[:2][::-1])
        sub_br_kp = tuple(sub_kp[2:4][::-1])

        obj_tl_kp = tuple(obj_kp[:2][::-1])
        obj_br_kp = tuple(obj_kp[2:4][::-1])

        img = cv2.rectangle(img, sub_tl_kp, sub_br_kp, (255, 255, 0), 3)
        # relation category for subject
        text_pos = (sub_tl_kp[0] + 5, sub_tl_kp[1] + 20)
        img = cv2.putText(img, rel_cate, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                          0.8, (255, 255, 0), 2, cv2.LINE_AA)
        # subject category
        text_pos = (sub_br_kp[0] - 5, sub_br_kp[1] - 20)
        img = cv2.putText(img, sub_cate, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                          0.8, (255, 255, 0), 2, cv2.LINE_AA)

        text_pos = (obj_tl_kp[0] + 5, obj_tl_kp[1] + 20)
        img = cv2.putText(img, obj_cate, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                          0.8, (0, 255, 0), 2, cv2.LINE_AA)
        img = cv2.rectangle(img, obj_tl_kp, obj_br_kp, (0, 255, 0), 3)
        return img

    obj_det_res = []
    b_bboxes, pair_results = list_objs_rels
    b_bboxes = np.asarray(b_bboxes)
    deep_images = copy.deepcopy(imgs)
    for img, bboxes in zip(imgs, b_bboxes):
        h, w, _ = img.shape
        img_size = (h, w)
        valid_mask = np.all(~np.isinf(bboxes), axis=-1)
        bboxes = bboxes[valid_mask]

        for bbox in bboxes:
            cate_ind = int(bbox[-1])
            category = tar_obj_cates[cate_ind]
            kp = bbox[:4]
            center_kp = (kp[:2] + kp[2:4]) / 2
            score = bbox[..., -2]
            img = draw_2d(img, center_kp, img_size, category)
        obj_det_res.append(img)

    rel_det_res = []
    pair_results = np.asarray(pair_results)

    # indices = [1., 2., 7., 9., 10., 11.]
    # pairs_rules = [[3.], [3., 8.], [4.], [3., 7.], [3.], [3.]]

    pairing_rules = {
        "1": [3],
        "2": [3, 8],
        "7": [4],
        "9": [3, 7],
        "10": [3],
        "11": [3]
    }

    for i, (img, subs_rels_objs) in enumerate(zip(deep_images, pair_results)):
        valid_mask = np.all(~np.isinf(subs_rels_objs), axis=-1)
        subs_rels_objs = subs_rels_objs[valid_mask]
        subs_rels_objs = subs_rels_objs.astype(int)
        bboxes = b_bboxes[i]
        valid_mask = np.all(~np.isinf(bboxes), axis=-1)
        bboxes = bboxes[valid_mask]
        tmp = []
        for sub_rel_obj in subs_rels_objs:
            sub_idx = sub_rel_obj[0]
            rel_idx = sub_rel_obj[1]
            obj_idx = sub_rel_obj[2]
            if obj_idx >= bboxes.shape[0]:
                continue
            sub_bbox = bboxes[sub_idx]
            obj_bbox = bboxes[obj_idx]
            sub_index = int(sub_bbox[..., -1])
            if str(sub_index) not in list(pairing_rules.keys()):
                continue
            rules = pairing_rules[str(sub_index)]
            obj_index = int(obj_bbox[..., -1])
            if obj_index not in rules:
                continue
            sub_cate = tar_obj_cates[int(sub_bbox[..., -1])]
            rel_cate = tar_rel_cates[rel_idx]
            obj_cate = tar_obj_cates[int(obj_bbox[..., -1])]
            copied_img = copy.deepcopy(img)
            copied_img = draw_bbox_rels(copied_img, sub_bbox[:4].astype(int),
                                        obj_bbox[:4].astype(int), sub_cate,
                                        obj_cate, rel_cate)
            tmp.append(copied_img)
        rel_det_res.append(tmp)

    return zip(obj_det_res, rel_det_res)
