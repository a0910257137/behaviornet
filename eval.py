import argparse
import time
import numpy as np
from monitor import logger
from tqdm import tqdm
from pprint import pprint
from box import Box
import os
import sys
import cv2
import copy
import pandas as pd
from pathlib import Path
from utils.io import *
from utils.bdd_process import *
from metrics.compute import ComputeIOU
from metrics.metric_evaluator import BDDMetricEvaluator

sys.path.insert(0, str(Path(__file__).parent.parent))

from behavior_predictor.inference import BehaviorPredictor


class Eval:
    def __init__(self, model, config, eval_path, img_root, save_path,
                 batch_size):
        self.pred_config = config['predictor']
        self.metric_config = config['metric']
        self.eval_path = eval_path
        self.img_root = img_root
        self.save_path = save_path
        self.batch_size = batch_size
        self.mode = self.pred_config['mode']
        self.metric_type = self.metric_config['metric_type']
        # self.predictor = model(self.pred_config)
        self.predictor = None

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
        # df.to_csv('infer_test_iou.csv', float_format='%.3f')

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
                break
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
            # batch_results = np.save('pred_results.npy',
            #                         np.asarray(batch_results))
            batch_results = np.load('pred_results.npy')
            if self.mode == 'centernet':
                eval_bdd_annos = to_tp_od_bdd(batch_results, batch_frames,
                                              self.cates)
            elif self.mode == 'landmark':
                eval_bdd_annos = to_lnmk_bdd(batch_results, batch_frames,
                                             self.cates)
            gt_bdd_annos, eval_bdd_annos = self.with_bddversion(
                gt_bdd_list), self.with_bddversion(
                    eval_bdd_annos['frame_list'])
            if self.metric_config['metric_type'] == 'IoU':
                # old version could parse all frame and calculate FP FN
                iou = ComputeIOU(gt_bdd_annos, eval_bdd_annos)
                self.iou_report(iou, self.cates)
            elif self.metric_type.lower() in ['keypoints', 'landmark', 'nle']:
                evaluator = BDDMetricEvaluator(self.metric_config)
            report_results = evaluator(gt_bdd_annos, eval_bdd_annos)
            if self.metric_type == 'NLE':
                dump_json(
                    path='/aidata/anders/objects/landmarks/metrics/nle.json',
                    data=report_results)
            else:
                obj_level_results = dict(report_results['obj_level'])
                df, mean_df = transform_pd_data(obj_level_results, True,
                                                self.metric_type)
                pprint(mean_df)
        print('Finish evaluating')
        print('Totoal speding %5fs, With %i batch size, %i imgs' %
              (sum(batch_times), self.batch_size, total_imgs))


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
    eval = Eval(BehaviorPredictor, Box(config), args.eval_path, args.img_root,
                args.save_path, args.batch_size)
    eval.run()
