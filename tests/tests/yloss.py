from re import L
import tensorflow as tf
import numpy as np
import argparse
import commentjson
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.loss.loss_functions import wing_loss


def yloss(loss_cfg):
    # for setting pesudo loss logist and target
    P6 = np.random.normal(size=[1, 3, 8, 8, 140])
    P5 = np.random.normal(size=[1, 3, 16, 16, 140])
    P4 = np.random.normal(size=[1, 3, 32, 32, 140])

    init_box2d = np.array([0.5, 0.5, 1., 1.])
    # for initializing

    indices, tbbox, anch, = [], [], []
    landmarks_list, lmks_mask = [], []
    keys = ["P4", "P5", "P6"]
    logists = {"P4": P4, "P5": P5, "P6": P6}
    landmarks = np.load("./test_data/b_kps.npy")
    anchors = loss_cfg["anchors"]
    strides = loss_cfg["strides"]
    anchors = np.asarray(anchors)
    anchors = anchors.reshape([3, 3, 2])
    strides = np.asarray(strides)[:, np.newaxis, np.newaxis]
    anchors = anchors / strides

    valid_maks = np.all(np.isfinite(landmarks), axis=-1)
    # landmarks as target
    landmarks = landmarks[valid_maks]
    landmarks = landmarks.reshape([-1, 68 * 2])
    init_box2d = np.tile(np.reshape(init_box2d, [-1, 4]),
                         [landmarks.shape[0], 1])
    landmarks = np.concatenate([init_box2d, landmarks], axis=-1)
    num_anchors = len(anchors)
    num_targets = landmarks.shape[0]
    gain = np.ones(shape=[1])
    ai = np.reshape(np.arange(num_anchors),
                    (num_anchors, 1)).repeat(1, num_targets)

    landmarks = np.repeat(landmarks[None, :, :], repeats=3, axis=0)
    landmarks = np.concatenate([landmarks, ai[:, :, None]], axis=-1)

    off = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]])
    g = 0.5
    # already kown values [128, 128, 256, 256]
    # if normalize values [0.5, 0.5, 1, 1]
    for i in range(num_anchors):
        anchor = anchors[i]
        print(anchor)
        xxxx
        pred = logists[keys[i]]
        b, d, h, w, c = pred.shape
        gain = np.array([w, h])
        gain = np.tile(gain, [70])
        gain = np.concatenate([gain, np.ones(shape=(1))], axis=0)
        landmarks = gain * landmarks
        # if only one anchors fit in
        one_anchor_tars = landmarks[2]
        if num_targets:
            # Assume no pass to anchor boxes
            # offset
            gxy = one_anchor_tars[:, :2]
            gxi = gain[:2] - one_anchor_tars[:, 2:4]
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T

            j = np.stack([np.ones_like(j), j, k, l, m])
            tars = np.repeat(one_anchor_tars[np.newaxis, :, :],
                             repeats=5,
                             axis=0)
            tars = tars[j]

            # offsets
            offsets = (np.zeros_like(gxy)[None] + off[:, None])[j]

            gxy = tars[:, :2]  # grid x y
            gwh = tars[:, 2:4]  # grid w h
            gij = (gxy - offsets).astype(np.float32)
            gi, gj = gij.T
            a = tars[:, -1]
            b = np.zeros(shape=[tars.shape[0]])
            indices.append([
                b, a,
                np.clip(
                    gj,
                    a_min=0.,
                    a_max=gain[0] - 1,
                ),
                np.clip(
                    gi,
                    a_min=0.,
                    a_max=gain[0] - 1,
                )
            ])
            tbbox.append(np.concatenate([gxy - gij, gwh], axis=-1))
            anch.append(anchor[a.astype(int)])
            tar_lmks = tars[:, 4:140]

            landmarks_mask = np.where(tar_lmks < 0.0, 0., 1.0)
            anchor_lv, c = tar_lmks.shape
            d = 2
            tar_lmks = np.reshape(tar_lmks, [anchor_lv, c // 2, d])
            gij = np.tile(gij[:, None, :], [1, c // 2, 1])
            tar_lmks = tar_lmks - gij
            landmarks_list += [tar_lmks]
            lmks_mask += [landmarks_mask]
    tbbox, indices, anch, landmarks_list, lmks_mask
    # Loss part
    # did wing_loss
    # wing_loss(pred, truel, mask)
    nt = 0
    for i, k in enumerate(logists):
        b, a, gj, gi = indices[i]  # image , anchor, gridy, gridx
        pi = logists[k]
        # print(pred.shape)

        n = 1
        if n:
            nt += n
            b = b.astype(int)
            a = a.astype(int)
            gj = gj.astype(int)
            gi = gi.astype(int)
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targes
            # lankmark loss
            # Regression
            print(ps.shape)
            print(anch[i].shape)
            xxx

    return


def parse_config():
    parser = argparse.ArgumentParser('Argparser for test yolo loss')
    parser.add_argument('--config')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_config()
    with open(args.config) as f:
        config = commentjson.loads(f.read())
    print('Test YOLO loss for landmarks')

    loss = yloss(config["models"]["loss"])