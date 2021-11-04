#%%
import numpy as np
import copy
import pandas as pd
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
from functools import partial
from .process.bdd_process import BDD

from .core.iou import IOU


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
                cost_matrics[g, p] = IOU(gt_labels[g], pred_labels[p])
        r, c = linear_sum_assignment(cost_matrics, maximize=True)
        return cost_matrics[r, c], gt_counts, pre_counts
