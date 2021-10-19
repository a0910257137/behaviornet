#%%
from tqdm import tqdm
import numpy as np
import os
import json
from scipy.optimize import linear_sum_assignment

from collections import Counter, defaultdict
from functools import partial
import copy
import pandas as pd

# gtjson = json.load(open("/aidata/anders/objects/WF/annos/BDD_val,json"))
# evaljson = json.load(
#     open("/aidata/anders/objects/WF/archive_model/blazenet/pred.json", "r"))
# cates_order = ["FACE"]


class BDD:
    def __init__(self, json_file):
        self.pathDict = {}
        self.pathSortedList = []
        self.category_statistic = []
        for frame in json_file['frame_list']:
            imgPath = os.path.join(frame['name'])
            labelsList = []
            for label in frame['labels']:
                self.category_statistic.append(label['category'].upper())
                labelsList.append(
                    dict(category=label['category'].upper(),
                         box2d=label['box2d']))
            self.pathDict[imgPath] = labelsList
        self.pathSortedList = sorted(self.pathDict)
        self.category_counts = Counter(self.category_statistic)
        self.category_iou = defaultdict(list)

    def __len__(self):
        return len(self.pathDict)

    def get_item(self, n):
        return self.pathSortedList[n]

    def get_labels(self, n):
        return self.pathDict[self.pathSortedList[n]]

    def get_cates(self):
        # just check the first frame
        return tuple(sorted(self.category_counts.keys()))

    def get_cates_counts(self):
        # just check the first frame
        return self.category_counts


class ComputeIOU:
    def __init__(self, gtbdd, evalbdd):
        gtbdd = BDD(gtbdd)
        evalbdd = BDD(evalbdd)
        gtbdd.get_cates_counts()
        evalbdd.get_cates_counts()
        self.gtbdd = gtbdd
        self.evalbdd = evalbdd
        assert len(gtbdd) == len(evalbdd), 'gt and eval frame mismatch'
        self.cates = set(gtbdd.get_cates()) | set(evalbdd.get_cates())
        self.iou_sum, self.gt_sum, self.pre_sum = self.compute()

    def compute(self):
        print('processing...')
        self.iou_sum = defaultdict(partial(np.ndarray, 0))
        self.gt_sum = defaultdict(int)
        self.pre_sum = defaultdict(int)
        for i in tqdm(range(len(self.gtbdd))):
            assert self.gtbdd.get_item(i) == self.evalbdd.get_item(i)
            gt_dict = self.sep_by_cates_as_key(self.gtbdd.get_labels(i))
            eval_dict = self.sep_by_cates_as_key(self.evalbdd.get_labels(i))
            cates = set(gt_dict) | set(eval_dict)
            for cate in cates:
                cost_matrics, gt_counts, pre_counts = self.obj_matching(
                    gt_dict[cate], eval_dict[cate])
                self.iou_sum[cate] = np.concatenate(
                    (copy.deepcopy(self.iou_sum[cate]), cost_matrics), axis=0)
                self.gt_sum[cate] += gt_counts
                self.pre_sum[cate] += pre_counts
        return self.iou_sum, self.gt_sum, self.pre_sum

    def report(self, A_type, theshold=0.5):
        recall_rate = defaultdict(float)
        precision_rate = defaultdict(float)
        epsilon = 1e-9
        for cate in self.cates:
            TP = (np.asarray(self.iou_sum[cate]) > theshold).sum()
            FN = self.gt_sum[cate] - TP
            FP = self.pre_sum[cate] - TP
            recall_rate[cate] = TP / max(TP + FN, epsilon)
            precision_rate[cate] = TP / max(TP + FP, epsilon)
        if A_type == 'AP':
            return precision_rate
        else:
            return recall_rate

    @staticmethod
    def sep_by_cates_as_key(labels):
        cates_boxes = defaultdict(list)
        for label in labels:
            cates_boxes[label['category']].append(label['box2d'])
        return cates_boxes

    def obj_matching(self, gt_labels, pred_labels):
        gt_counts = len(gt_labels)
        pre_counts = len(pred_labels)
        cost_matrics = np.zeros((gt_counts, pre_counts))
        for g in range(gt_counts):
            for p in range(pre_counts):
                cost_matrics[g, p] = self.IOU(gt_labels[g], pred_labels[p])
        r, c = linear_sum_assignment(cost_matrics, maximize=True)
        return cost_matrics[r, c], gt_counts, pre_counts

    @staticmethod
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
        iou = intersection_area / float(gt_area + pred_area -
                                        intersection_area)
        return iou


# iou = ComputeIOU(gtbdd, evalbdd)
