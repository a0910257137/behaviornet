import tensorflow as tf
import commentjson
from box import Box
from pprint import pprint
import os
from glob import glob
import numpy as np
from models.loss.anchor_loss import AnchorLoss
from utils.io import *

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

config = Box(read_commentjson(path="./config/scrfd.json"))
config['models']['loss'].anchor_generator = config['models'].anchor_generator
config['models']['loss'].resize_size = config['data_reader'].resize_size
config['models']['loss'].batch_size = config['batch_size']
anchor_func = AnchorLoss(config['models']['loss'])


def get_gt_preds(gt_dir, pred_dir):

    def compl(values):

        if len(values.shape) == 2:
            n, c = values.shape
            complement = np.empty([15 - n, c])
        elif len(values.shape) == 1:
            n, = values.shape
            complement = np.empty([15 - n])
        elif len(values.shape) == 3:
            n, c, d = values.shape
            complement = np.empty([15 - n, c, d])
        complement.fill(np.inf)
        values = np.concatenate([values, complement], axis=0)
        return values

    batch, lv_lens = 16, 3
    anchor_list, valid_flag_list = [], []
    gt_bboxes, gt_keypointss, gt_bboxes_ignore, gt_labels = [], [], [], []
    for i in range(batch):
        tmp = []
        for j in range(lv_lens):
            tmp.append(
                np.load(os.path.join(gt_dir, "anchor_b{}_lv{}.npy".format(i,
                                                                          j))))
        tmp = np.concatenate(tmp, axis=0)
        anchor_list.append(tmp)
        tmp = []
        for j in range(lv_lens):
            tmp.append(
                np.load(
                    os.path.join(gt_dir, "valid_flag_b{}_lv{}.npy".format(i,
                                                                          j))))
        tmp = np.concatenate(tmp, axis=0)
        valid_flag_list.append(tmp)
        values = np.load(os.path.join(gt_dir, "gt_bbox_{}.npy").format(i))
        values = compl(values)
        gt_bboxes.append(values)
        values = np.load(os.path.join(gt_dir, "gt_keypointss_{}.npy").format(i))
        values = compl(values)
        gt_keypointss.append(values)
        values = np.load(
            os.path.join(gt_dir, "gt_bboxes_ignore_{}.npy").format(i))

        gt_bboxes_ignore.append(values)
        values = np.load(os.path.join(gt_dir, "gt_labels_{}.npy".format(i)))
        values = compl(values)
        gt_labels.append(values)
    anchor_list = np.stack(anchor_list, axis=0)
    valid_flag_list = np.stack(valid_flag_list, axis=0)
    gt_bboxes = np.stack(gt_bboxes, axis=0)
    gt_keypointss = np.stack(gt_keypointss, axis=0)
    gt_labels = np.stack(gt_labels, axis=0)

    cls_scores, bbox_preds, kps_preds = [], [], []
    for i in range(3):

        value = np.load(os.path.join(pred_dir, "cls_score_{}.npy".format(i)))
        value = np.transpose(value, [0, 2, 3, 1])
        cls_scores.append(value)
        value = np.load(os.path.join(pred_dir, "bbox_pred_{}.npy".format(i)))
        value = np.transpose(value, [0, 2, 3, 1])
        bbox_preds.append(value)
        value = np.load(os.path.join(pred_dir, "kps_pred_{}.npy".format(i)))
        value = np.transpose(value, [0, 2, 3, 1])
        kps_preds.append(value)
    return anchor_list, valid_flag_list, gt_bboxes, gt_keypointss, gt_bboxes_ignore, gt_labels, cls_scores, bbox_preds, kps_preds


def run(anchor_list, valid_flag_list, gt_bboxes, gt_keypoints, gt_bboxes_ignore,
        gt_labels, cls_scores, bbox_preds, kps_preds):
    target_num_lv_anchors = tf.tile(tf.constant([[12800, 3200, 800]]), [16, 1])
    b_anchors = tf.cast(anchor_list, tf.float32)
    # is the anchors the same between torch and tensorflow ?
    b_flags = tf.cast(valid_flag_list, tf.bool)
    b_gt_bboxes = tf.cast(gt_bboxes, tf.float32)
    b_gt_bboxes = tf.reshape(b_gt_bboxes, [16, 15, 2, 2])
    b_gt_keypoints = tf.cast(gt_keypoints, tf.float32)
    b_gt_labels = tf.cast(gt_labels, tf.float32)

    with tf.device('CPU'):
        anchors_list, labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, keypoints_targets_list, keypoints_weights_list, num_total_samples = tf.py_function(
            anchor_func.get_targets,
            inp=[
                target_num_lv_anchors, b_anchors, b_flags, b_gt_bboxes,
                b_gt_keypoints, b_gt_labels
            ],
            Tout=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                  tf.float32, tf.float32, tf.float32))

    anchors_list = anchor_func.split2level(anchors_list)
    labels_list = anchor_func.split2level(labels_list)
    label_weights_list = anchor_func.split2level(label_weights_list)
    bbox_targets_list = anchor_func.split2level(bbox_targets_list)
    bbox_weights_list = anchor_func.split2level(bbox_weights_list)
    keypoints_targets_list = anchor_func.split2level(keypoints_targets_list)
    keypoints_weights_list = anchor_func.split2level(keypoints_weights_list)
    multi_lv_feats = list(zip(cls_scores, bbox_preds, kps_preds))

    loss_cls_list, loss_bbox_list, loss_kps_list = anchor_func.compute_loss(
        anchors_list, multi_lv_feats, labels_list, label_weights_list,
        bbox_targets_list, keypoints_targets_list, keypoints_weights_list,
        anchor_func.anchor_generator_strides, num_total_samples)

    return loss


if __name__ == "__main__":
    gt_dir = "/aidata/anders/3D-head/SCRFD/gt"
    pred_dir = "/aidata/anders/3D-head/SCRFD/logists"
    anchor_list, valid_flag_list, gt_bboxes, gt_keypointss, gt_bboxes_ignore, gt_labels, cls_scores, bbox_preds, kps_preds = get_gt_preds(
        gt_dir, pred_dir)

    loss = run(anchor_list, valid_flag_list, gt_bboxes, gt_keypointss,
               gt_bboxes_ignore, gt_labels, cls_scores, bbox_preds, kps_preds)
