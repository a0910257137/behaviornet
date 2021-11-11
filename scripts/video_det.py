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
from glob import glob

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import load_text
from behavior_predictor.inference import BehaviorPredictor


def dump_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)


def make_dir(path):
    if not os.path.exists(path):
        os.umask(0)
        os.makedirs(path, mode=0o755)


def img_gen(config_path, video_root, save_root, anno_root):
    def _get_cates(path):
        cates = [x.strip() for x in load_text(path)]
        target_cat_dict = {i: k for i, k in enumerate(cates)}
        return target_cat_dict

    with open(config_path) as f:
        config = commentjson.loads(f.read())
    print('Restore model')
    predictor = BehaviorPredictor(config['predictor'])
    video_paths = glob(os.path.join(video_root, '*.MP4'))
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        frame_counter = 0
        directory = video_path.split('/')[-1]
        directory = directory.split('.')[0]
        directory = os.path.join(save_root, directory)
        crop_dir = os.path.join(directory, 'crop_imgs')
        make_dir(directory)
        make_dir(crop_dir)
        bdd_results = {'frame_list': []}
        while cap.isOpened():
            imgs, origin_shapes, orig_imgs = [], [], []
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            h, w, _ = frame.shape
            origin_shapes.append((h, w))
            orig_imgs.append(frame)
            imgs.append(frame)
            rets = predictor.pred(imgs, origin_shapes)
            target_dict = _get_cates(config['predictor']['cat_path'])
            imgs, cropped_results, obj_kps_results = draw_box2d(
                orig_imgs, rets, target_dict)
            frame_name = 'frame_{:0>6d}.jpg'.format(frame_counter)
            frame_infos = {
                "dataset": 'Demo_Box2d',
                "sequence": directory.split('/')[-1],
                "name": frame_name,
                "labels": []
            }
            for obj_kps in obj_kps_results:
                for kps in obj_kps:
                    tl = kps[:2]
                    br = kps[2:4]
                    frame_infos['labels'].append({
                        'box2d': {
                            'x1': float(tl[1]),
                            'y1': float(tl[0]),
                            'x2': float(br[1]),
                            'y2': float(br[0])
                        },
                        'score': float(kps[-2]),
                        'category': 'FACE'
                    })
            bdd_results['frame_list'].append(frame_infos)
            save_name = os.path.join(directory, frame_name)
            for img in imgs:
                print('writing %s' % save_name)
                cv2.imwrite(save_name, img)

            for j, img in enumerate(cropped_results):
                frame_name = 'frame_{:0>6d}_face_{:0>1d}.jpg'.format(
                    frame_counter, j)
                save_name = os.path.join(crop_dir, frame_name)
                cv2.imwrite(save_name, img)

            frame_counter += 1
        anno_name = '{}_box2d.json'.format(directory.split('/')[-1])
        dump_json(path=os.path.join(crop_dir, anno_name), data=bdd_results)


def parse_config():
    parser = argparse.ArgumentParser('Argparser for model image generate')
    parser.add_argument('--config')
    parser.add_argument('--video_root')
    parser.add_argument('--save_root')
    parser.add_argument('--anno_root')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_config()
    print('Result imgs generating')
    print(f"Use following config to produce tensorflow graph: {args.config}.")
    assert os.path.isfile(args.config)
    img_gen(args.config, args.video_root, args.save_root, args.anno_root)
