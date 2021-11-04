import numpy as np


def IOU(gt, pred):
    if int(gt['x1']) < int(gt['x2']):
        gt_x1, gt_x2 = int(gt['x1']), int(gt['x2'])
    else:
        gt_x2, gt_x1 = int(gt['x1']), int(gt['x2'])

    if int(gt['y1']) < int(gt['y2']):
        gt_y1, gt_y2 = int(gt['y1']), int(gt['y2'])
    else:
        gt_y2, gt_y1 = int(gt['y1']), int(gt['y2'])

    if int(pred['x1']) < int(pred['x2']):
        pred_x1, pred_x2 = int(pred['x1']), int(pred['x2'])
    else:
        pred_x2, pred_x1 = int(pred['x1']), int(pred['x2'])
    if int(pred['y1']) < int(pred['y2']):
        pred_y1, pred_y2 = int(pred['y1']), int(pred['y2'])
    else:
        pred_y2, pred_y1 = int(pred['y1']), int(pred['y2'])

    x_left = max(gt_x1, pred_x1)
    y_top = max(gt_y1, pred_y1)
    x_right = min(gt_x2, pred_x2)
    y_bottom = min(gt_y2, pred_y2)
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    iou = intersection_area / float(gt_area + pred_area - intersection_area)
    return iou
