import argparse
import time
import numpy as np
from monitor import logger
from tqdm import tqdm
from pprint import pprint

import os
import sys
import cv2
import copy
import pandas as pd
from pathlib import Path
from utils.io import *
from utils.bdd_process import *
from metrics.compute import ComputeIOU

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, './utils/linker-metrics/linkermetrics')
# from linkermetrics.evaluator.bdd_metric_evaluator import BDDMetricEvaluator
from behavior_predictor.inference import BehaviorPredictor


class Eval:
    def __init__(self, model, config, eval_path, img_root, save_path,
                 batch_size):
        self.config = config
        self.eval_path = eval_path
        self.img_root = img_root
        self.save_path = save_path
        self.batch_size = batch_size
        self.mode = self.config['mode']
        self.predictor = model(self.config)

    def _get_conditions(self, cates, task):
        if task == 'TAIPOWER':
            tp_od_condition = dict(
                name='BDDConditionedContainer',
                category='HUMAN',
                frame=lambda frame: True,
                object_matcher=dict(
                    name='PointDistanceMatcher',
                    threshold=1000,
                    transform=[
                        dict(name='Box2DToKeyPointsWithCenter'),
                        dict(name='Numpify', target_key='shape'),
                        dict(name='GetShape')
                    ],
                ),
                object_calculator=[{
                    'name':
                    'PRFCalculator',
                    'metric': {
                        'name': 'PointDistance'
                    },
                    'transform': [
                        dict(name='Box2DToKeyPointsWithCenter'),
                        dict(name='Numpify', target_key='shape'),
                        dict(name='GetShape')
                    ],
                    'reporter': {
                        'name': 'EdgePRFReporter',
                        'threshold': 5
                    }
                }])
            __Conditions = []
            for cate in cates:
                eval_condition = copy.deepcopy(tp_od_condition)
                eval_condition['category'] = cate.upper()
                __Conditions.append(eval_condition)
        elif task == 'SAMSUNG':
            __Conditions = dict(
                name='BDDConditionedContainer',
                category='VEHICLE',
                object_matcher=dict(
                    name='IoU2DMatcher',
                    threshold=0.5,
                    transform=[
                        dict(name='PointArrangementSamsungCuboidShape'),
                        dict(name='MinimalBox2D'),
                        dict(name='Numpify', target_key='shape'),
                        dict(name='GetShape')
                    ]),
                object_calculator=[{
                    'name':
                    'PRFCalculator',
                    'metric': {
                        'name': 'PointAxialShift'
                    },
                    'transform': [
                        dict(name='PointArrangementSamsungCuboidShape'),
                        dict(name='Numpify', target_key='shape'),
                        dict(name='GetShape')
                    ],
                    'reporter': {
                        'name': 'EdgePRFReporter',
                        'threshold': 5
                    }
                }])

        return __Conditions

    def split_batchs(self, elems, n):
        for idx in range(0, len(elems), n):
            yield elems[idx:idx + n]

    def with_bddversion(self, input_json_path):
        psudo_bdd = {
            "bdd_version": "1.1.2",
            "company_code": 15,
            "inference_object": 2,
            "model_id": "",
            "frame_list": []
        }
        # psudo_bdd['frame_list'] = input_json_path['frame_list']
        psudo_bdd['frame_list'] = input_json_path
        return psudo_bdd

    def transform_pd_data(self, reports):
        tot_dict = {}
        pd_columns = ['precision', 'recall', 'accuracy']
        pd_rows = ['center_point', 'top_left', 'bottom_right']

        for report in reports:
            results = report['results']
            for res in results:
                obj_level = res['obj_level']
                if len(dict(obj_level).keys()) == 0:
                    df = pd.DataFrame({})
                    continue
                trans_dict = {}
                for pd_col in pd_columns:
                    trans_dict[pd_col] = {}
                    for pd_row in pd_rows:
                        if pd_row not in obj_level[pd_col].keys():
                            continue
                        trans_dict[pd_col][pd_row] = obj_level[pd_col][pd_row]
                df = pd.DataFrame(trans_dict)
            tot_dict[report['category']] = df
        return tot_dict

    def get_eval_path(self):
        eval_files = []
        if os.path.isfile(self.eval_path):
            eval_files.append(self.eval_path)
        else:
            eval_files = [
                os.path.join(self.eval_path, x)
                for x in os.listdir(self.eval_path)
            ]
        return eval_files

    def iou_report(self, iou, cates_order):
        precs, recs = [], []
        cates = []
        threshold = [0.95, 0.75, 0.5, 0.25, 0.1, 0.01]
        for thres in threshold:
            result_prec = iou.report('AP', thres)
            result_rec = iou.report('AR', thres)

            for cate in cates_order:
                try:
                    prec_val = result_prec[cate]
                    rec_val = result_rec[cate]
                except:
                    prec_val = '0'
                    rec_val = '0'
                precs.append(prec_val)
                recs.append(rec_val)
                cates.append(cate)

        dict = {
            "Category": cates,
            "Threshold": threshold,
            "Precision": precs,
            'Recall': recs
        }

        df = pd.DataFrame(dict)
        pprint(df)
        df.to_csv('infer_test_iou.csv', float_format='%.3f')

    def run(self):
        eval_files = self.get_eval_path()
        self.cates = ["FACE"]
        print('Eval categories')
        pprint(self.cates)
        total_imgs = 0
        batch_times = []
        for eval_file in eval_files:
            print('Evaluating with %s' % eval_file)
            gt_bdd_annos = load_json(eval_file)
            gt_bdd_list = gt_bdd_annos['frame_list']
            batch_frames = list(self.split_batchs(gt_bdd_list,
                                                  self.batch_size))
            batch_results = []
            for batch_frame in tqdm(batch_frames):
                path_batch = [
                    os.path.join(self.img_root, x['name']) for x in batch_frame
                ]
                imgs = [cv2.imread(x) for x in path_batch]
                img_origin_sizes = [img.shape[:2] for img in imgs]
                total_imgs += len(imgs)
                preds = self.predictor.pred(imgs, img_origin_sizes)
                # preds = preds.numpy()
                # for pred, img, frame in zip(preds, imgs, batch_frame):
                #     valid_mask = np.all(~np.isinf(pred), axis=-1)
                #     pred = pred[valid_mask]
                #     for p in pred:
                #         tl = p[:2].astype(int)
                #         tl = tl[::-1]
                #         br = p[2:4].astype(int)
                #         br = br[::-1]
                #         img = cv2.rectangle(img, tuple(tl), tuple(br),
                #                             (0, 255, 0), 3)
                #     cv2.imwrite(
                #         os.path.join(
                #             "/aidata/anders/objects/WF/model_imgs/train",
                #             frame["name"]), img)
                batch_results.append(preds)
            # to bdd annos by task
            eval_bdd_annos = to_tp_od_bdd(batch_results, batch_frames,
                                          self.cates)
            gt_bdd_annos, eval_bdd_annos = self.with_bddversion(
                gt_bdd_list), self.with_bddversion(
                    eval_bdd_annos['frame_list'])
            if self.config['eval_method'] == 'IoU':
                iou = ComputeIOU(gt_bdd_annos, eval_bdd_annos)
                self.iou_report(iou, self.cates)
            elif self.config['eval_method'] == 'keypoint':
                __Condistions = self._get_conditions(self.cates,
                                                     self.config['task'])
                evaluator = BDDMetricEvaluator(
                    frame_matcher=dict(name='BDDFrameToFrameMatcher'),
                    conditions=__Condistions)
                evaluator(dict(gt=gt_bdd_annos, eval=eval_bdd_annos))
                # report from linker metrics
                reports = evaluator.report
                # list of dictionary, each dictionary save one category
                tot_dict = self.transform_pd_data(reports)
                pprint(tot_dict)
        print('Finish evaluating')
        print(
            'Totoal speding %5fs, avg %5fs per batch with %i batch size, %i imgs'
            % (sum(batch_times), sum(batch_times) / len(batch_times),
               self.batch_size, total_imgs))


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate model performance')
    parser.add_argument('--config', default=None, help='eval config')

    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='batch size')
    parser.add_argument('--eval_path',
                        default=None,
                        help='eval data folder or file path')

    parser.add_argument('--img_root', help='eval images folder path')
    parser.add_argument('--save_path', help='save results in folder')
    parser.add_argument('--threshold',
                        type=int,
                        default=5,
                        help='edge metric threshold')

    parser.add_argument('--category', help='evaluate category')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print('Eval with %s' % args.config)
    if not os.path.isfile(args.config):
        raise FileNotFoundError('File %s does not exist.' % args.config)
    config = load_json(args.config)
    eval = Eval(BehaviorPredictor, config['predictor'], args.eval_path,
                args.img_root, args.save_path, args.batch_size)
    eval.run()
