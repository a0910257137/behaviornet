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

    def crop(img, tl, br):
        tl = tl.astype(np.int32)
        br = br.astype(np.int32)
        copped_img = img[tl[0]:br[0], tl[1]:br[1], :]
        return copped_img

    result, cropped_results = [], []
    obj_kps_results = []
    b_obj_kps = b_obj_kps.numpy()

    for img, obj_kps in zip(b_imgs, b_obj_kps):
        valid_mask = np.all(np.isfinite(obj_kps), axis=-1)
        obj_kps = obj_kps[valid_mask]
        obj_kps_results.append(obj_kps)
        for_copy_img = copy.deepcopy(img)
        for obj_kp in obj_kps:
            category_index = int(obj_kp[..., -1])
            category = target_dict[category_index]
            score = obj_kp[..., -2]
            kp = obj_kp[:4]
            cropped_img = crop(for_copy_img, kp[:2], kp[2:4])
            img = draw_2d(img, kp, score, category)
            cropped_results.append(cropped_img)
        result.append(img)
    return result, cropped_results, obj_kps_results


def draw_landmark(b_orig_imgs, b_landmarks):
    b_landmarks = b_landmarks.numpy()
    outputs_imgs = []
    for img, landmarks in zip(b_orig_imgs, b_landmarks):
        for lnmk in landmarks:
            lnmk = lnmk.astype(int)
            img = cv2.circle(img, tuple(lnmk[::-1]), 5, (0, 255, 0), -1)
        outputs_imgs.append(img)
    return outputs_imgs