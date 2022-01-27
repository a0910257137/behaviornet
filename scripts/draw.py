from pprint import pprint
import cv2
import numpy as np
import copy
from sklearn.cluster import KMeans, DBSCAN


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
    b_bboxes, b_ENM = b_obj_kps
    b_ENM = b_ENM.numpy()
    b_bboxes = b_bboxes.numpy()
    for img, objs_kps, ENMs in zip(b_imgs, b_bboxes, b_ENM):
        valid_mask = np.all(np.isfinite(objs_kps), axis=-1)
        objs_kps = objs_kps[valid_mask]
        ENMs = ENMs[valid_mask].astype(np.int32)

        for obj_kps, ENM in zip(objs_kps, ENMs):
            for kp in ENM:
                img = cv2.circle(img, tuple(kp[::-1]), 3, (0, 0, 255), -1)
            category_index = int(obj_kps[..., -1])
            category = target_dict[category_index]
            score = obj_kps[..., -2]
            kp = obj_kps[:4]
            img = draw_2d(img, kp, score, category)
        result.append(img)
    return result
    # ----------------------------------------heatmap version---------------------------------------
    # b_bboxes, b_lnmks = b_obj_kps
    # b_lnmks, b_lnmk_scores, b_lnmk_mask = b_lnmks
    # b_lnmks, b_lnmk_scores, b_lnmk_mask = b_lnmks.numpy(), b_lnmk_scores.numpy(
    # ), b_lnmk_mask.numpy()
    # b_bboxes = b_bboxes.numpy()

    # for img, obj_kps, lnmks, lnmk_scores, lnmk_masks in zip(
    #         b_imgs, b_bboxes, b_lnmks, b_lnmk_scores, b_lnmk_mask):
    #     for lnmk, lnmk_score, lnmk_mask in zip(lnmks, lnmk_scores, lnmk_masks):
    #         lnmk = lnmk[lnmk_mask]
    #         lnmk_score = lnmk_score[lnmk_mask]
    #         if len(lnmk) == 0:
    #             continue
    #         # kmeans = KMeans(n_clusters=2).fit(lnmk)
    #         clustering = DBSCAN(eps=2, min_samples=1).fit(lnmk)
    #         lbs = clustering.labels_
    #         items = np.unique(lbs)
    #         for i in items:
    #             if i == -1:
    #                 continue
    #             label_mask = lbs == i
    #             kps = lnmk[label_mask]
    #             score = lnmk_score[label_mask]
    #             kp = np.mean(kps, dtype=np.int32, axis=0)

    #             img = cv2.circle(img, tuple(kp[::-1]), 3, (0, 0, 255), -1)

    #     valid_mask = np.all(np.isfinite(obj_kps), axis=-1)
    #     obj_kps = obj_kps[valid_mask]

    #     for obj_kp in obj_kps:
    #         category_index = int(obj_kp[..., -1])
    #         category = target_dict[category_index]
    #         score = obj_kp[..., -2]
    #         kp = obj_kp[:4]
    #         img = draw_2d(img, kp, score, category)
    #     result.append(img)
    # return result


# ----------------------------------------heatmap version---------------------------------------


def draw_landmark(b_orig_imgs, b_landmarks):
    b_landmarks = b_landmarks.numpy()
    outputs_imgs = []
    for img, landmarks in zip(b_orig_imgs, b_landmarks):
        for lnmk in landmarks:
            lnmk = lnmk.astype(int)
            img = cv2.circle(img, tuple(lnmk[::-1]), 5, (0, 255, 0), -1)
        outputs_imgs.append(img)
    return outputs_imgs