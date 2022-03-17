from pprint import pprint
import cv2
import numpy as np
import copy


def draw_box2d(b_imgs, b_obj_kps, target_dict, clr=(0, 255, 0)):
    def draw_2d(img, kp, score, category):
        tl, br = tuple(kp[:2][::-1].astype(np.int32)), tuple(
            kp[2:4][::-1].astype(np.int32))
        img = cv2.rectangle(img, tl, br, clr, 2)
        center = (tl[0] + 20, tl[1] - 20)
        img = cv2.putText(img, ('%3f' % score), center,
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), 2,
                          cv2.LINE_AA)
        return img

    result = []
    b_bboxes, b_lnmks, b_nose_scores = b_obj_kps
    b_nose_scores = b_nose_scores.numpy()
    b_lnmks = b_lnmks.numpy()
    b_bboxes = b_bboxes.numpy()
    for img, objs_kps, lnmks, nose_scores in zip(b_imgs, b_bboxes, b_lnmks,
                                                 b_nose_scores):
        valid_mask = np.all(np.isfinite(objs_kps), axis=-1)
        objs_kps = objs_kps[valid_mask]
        hws = objs_kps[:, 2:4] - objs_kps[:, :2]
        areas = hws[:, 0] * hws[:, 1]
        if len(areas) != 0:
            idx = np.argmax(areas, axis=0)
            objs_kps = np.reshape(objs_kps[idx], (-1, 6))
            for obj_kps in objs_kps:
                category_index = int(obj_kps[..., -1])
                category = target_dict[category_index]

                score = obj_kps[..., -2]
                kp = obj_kps[:4]
                tl, br = kp[:2], kp[2:]
                y1, x1 = tl
                y2, x2 = br
                nose_lnmks = lnmks[:, 2, :]
                logical_y = np.logical_and(y1 < nose_lnmks[:, :1],
                                           nose_lnmks[:, :1] < y2)
                logical_x = np.logical_and(x1 < nose_lnmks[:, 1:],
                                           nose_lnmks[:, 1:] < x2)

                logical_yx = np.concatenate([logical_y, logical_x], axis=-1)

                logical_yx = np.all(logical_yx, axis=-1)
                lnmks = lnmks[logical_yx]
                nose_scores = nose_scores[logical_yx]
                if len(nose_scores) != 0:
                    max_idx = np.argmax(nose_scores)
                    n_lnmk = np.reshape(lnmks[max_idx], (5, 2))
                    for lnmk in n_lnmk:
                        lnmk = lnmk.astype(np.int32)
                        img = cv2.circle(img, tuple(lnmk[::-1]), 3,
                                         (0, 255, 0), -1)
                img = draw_2d(img, kp, score, category)
        result.append(img)
    return result


def draw_offset_v1(b_imgs, b_obj_kps, target_dict, clr=(0, 255, 0)):
    def draw_2d(img, kp, score, category):
        tl, br = tuple(kp[:2][::-1].astype(np.int32)), tuple(
            kp[2:4][::-1].astype(np.int32))
        img = cv2.rectangle(img, tl, br, clr, 2)
        center = (tl[0] + 20, tl[1] - 20)
        img = cv2.putText(img, ('%3f' % score), center,
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), 2,
                          cv2.LINE_AA)
        return img

    result = []
    b_bboxes = b_obj_kps.numpy()
    for img, objs_kps in zip(b_imgs, b_bboxes):
        valid_mask = np.all(np.isfinite(objs_kps), axis=-1)
        objs_kps = objs_kps[valid_mask]
        for obj_kps in objs_kps:
            offset_kps = obj_kps[-10:]
            offset_kps = np.reshape(offset_kps, (-1, 2))
            offset_kps = offset_kps.astype(np.int32)
            for kp in offset_kps:
                img = cv2.circle(img, tuple(kp[::-1]), 3, (0, 0, 255), -1)
            obj_kps = obj_kps[:-10]
            category_index = int(obj_kps[..., -1])
            category = target_dict[category_index]
            score = obj_kps[..., -2]
            kp = obj_kps[:4]
            img = draw_2d(img, kp, score, category)
        result.append(img)
    return result


def draw_landmark(b_orig_imgs, b_landmarks):
    b_landmarks = b_landmarks.numpy()
    outputs_imgs = []
    for img, landmarks in zip(b_orig_imgs, b_landmarks):
        for lnmk in landmarks:
            lnmk = lnmk.astype(int)
            img = cv2.circle(img, tuple(lnmk[::-1]), 5, (0, 255, 0), -1)
        outputs_imgs.append(img)
    return outputs_imgs