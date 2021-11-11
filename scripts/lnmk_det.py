import sys
import cv2
import os
import json
import argparse
import commentjson
import numpy as np
from pprint import pprint
from pathlib import Path
from draw import *

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import load_text
from behavior_predictor.inference import BehaviorPredictor

BATCH_SIZE = 2


def dump_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)


def to_lnmk_bdd(batch_preds, batch_paths, batch_origin_shapes):
    batch_frames = []
    lnmk_infos = {
        "countour_face": 3,
        "left_eyebrow": 0,
        "right_eyebrow": 0,
        "left_eye": 6,
        "right_eye": 6,
        "nose": 2,
        "outer_lip": 8,
        "inner_lip": 0
    }
    batch_preds = batch_preds.reshape([-1, 25, 2]).astype(np.float16)
    for preds, paths, origin_shapes in zip(batch_preds, batch_paths,
                                           batch_origin_shapes):
        splitted_path = paths.split('/')
        pred_frame = {
            'dataset': 'Demo_Landmarks',
            'sequence': splitted_path[-3],
            'name': splitted_path[-1],
            'labels': []
        }

        preds = preds.tolist()
        h, w = origin_shapes
        pred_lb = {
            'keypoints': {},
            'category': 'FACE',
            'img_size': origin_shapes
        }

        for k in lnmk_infos.keys():
            num_lnmk = lnmk_infos[k]
            pred_lb['keypoints'][k] = preds[:num_lnmk]
            del preds[:num_lnmk]
        pred_frame['labels'].append(pred_lb)
        batch_frames.append(pred_frame)
    return batch_frames


def img_gen(config_path, img_path_root, save_annos):

    with open(config_path) as f:
        config = commentjson.loads(f.read())

    print('Restore model')
    predictor = BehaviorPredictor(config['predictor'])
    print(predictor)
    img_names = list(filter(lambda x: 'jpg' in x, os.listdir(img_path_root)))
    img_paths = list(map(lambda x: os.path.join(img_path_root, x), img_names))

    img_path_batchs = [
        img_paths[idx:idx + BATCH_SIZE]
        for idx in range(0, len(img_paths), BATCH_SIZE)
    ]
    img_name_batchs = [
        img_names[idx:idx + BATCH_SIZE]
        for idx in range(0, len(img_names), BATCH_SIZE)
    ]
    bdd_results = {"frame_list": []}
    for img_paths, img_names in zip(img_path_batchs, img_name_batchs):
        imgs, origin_shapes, orig_imgs = [], [], []
        for img_path in img_paths:
            print(img_path)
            img = cv2.imread(img_path)
            h, w, _ = img.shape
            origin_shapes.append((h, w))
            orig_imgs.append(img)
            imgs.append(img)
        rets = predictor.pred(imgs, origin_shapes)
        rets = rets.numpy()
        bdd_results['frame_list'] += to_lnmk_bdd(rets, img_paths,
                                                 origin_shapes)

    dump_json(path=save_annos, data=bdd_results)


def parse_config():
    parser = argparse.ArgumentParser('Argparser for model image generate')
    parser.add_argument('--config')
    parser.add_argument('--img_root')
    parser.add_argument('--save_annos')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_config()
    print('Result imgs generating')
    print(f"Use following config to produce tensorflow graph: {args.config}.")

    assert os.path.isfile(args.config)
    img_gen(args.config, args.img_root, args.save_annos)
