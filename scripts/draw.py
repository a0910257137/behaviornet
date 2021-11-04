import cv2
import tensorflow as tf
import numpy as np


def draw_box2d(b_orig_imgs, b_landmarks):
    return


def draw_landmark(b_orig_imgs, b_landmarks):
    b_landmarks = b_landmarks.numpy()
    outputs_imgs = []
    for img, landmarks in zip(b_orig_imgs, b_landmarks):
        for lnmk in landmarks:
            lnmk = lnmk.astype(int)
            img = cv2.circle(img, tuple(lnmk[::-1]), 5, (0, 255, 0), -1)
        outputs_imgs.append(img)
    return outputs_imgs