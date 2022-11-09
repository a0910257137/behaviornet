from pprint import pprint
import cv2
import numpy as np
import sys
from pathlib import Path
from skimage import io

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.mesh.transform import *
from utils.mesh.render import *


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
                        img = cv2.circle(img, tuple(lnmk[::-1]), 3, (0, 255, 0),
                                         -1)
                img = draw_2d(img, kp, score, category)
        result.append(img)
    return result


def draw_tdmm(b_orig_imgs, b_rets):
    b_bboxes, b_lnmks, b_poses = b_rets[0].numpy(), b_rets[1].numpy(
    ), b_rets[2].numpy()
    outputs_imgs = []
    for i, (img, n_bboxes, n_lnmks,
            n_poses) in enumerate(zip(b_orig_imgs, b_bboxes, b_lnmks, b_poses)):
        mask = np.all(np.isfinite(n_bboxes), axis=-1)
        n_bboxes = n_bboxes[mask]
        mask = np.all(np.isfinite(n_lnmks), axis=-1)
        n_lnmks = np.reshape(n_lnmks[mask], (-1, 68, 2))
        mask = np.all(np.isfinite(n_poses), axis=-1)
        n_poses = np.reshape(n_poses[mask], (-1, 3))

        # resized = np.asarray(img.shape[:2]) / np.array([192., 320.])
        # n_lnmks = np.einsum('n k c, c -> n k c', n_lnmks, resized[::-1])
        for (bbox, lnmks, pose) in zip(n_bboxes, n_lnmks, n_poses):
            tl, br, scores = bbox[:2][::-1], bbox[2:4][::-1], bbox[-2]
            tl, br = tuple(tl.astype(np.int32)), tuple(br.astype(np.int32))
            img = cv2.rectangle(img, tl, br, (0, 255, 255), 2)
            lnmks = lnmks[:, ::-1]
            for j, kp in enumerate(lnmks):
                kp = kp.astype(np.int32)
                if j < 17:
                    img = cv2.circle(img, tuple(kp), 3, (255, 0, 255), -1)
                else:
                    img = cv2.circle(img, tuple(kp), 3, (0, 255, 255), -1)
            cv2.imwrite("output_{}.jpg".format(i), img)
        xxxx
        outputs_imgs.append(img)

    return outputs_imgs