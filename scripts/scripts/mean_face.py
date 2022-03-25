import sys
import cv2
import os
import argparse
import commentjson
from tqdm import tqdm
import numpy as np
from pprint import pprint
from pathlib import Path
from draw import *
import base64
import io
import PIL.ExifTags
import PIL.Image
import PIL.ImageOps
from glob import glob

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import dump_json, load_json
from behavior_predictor.inference import BehaviorPredictor

BATCH_SIZE = 1

import codecs


def encodeImageForJson(image):
    img_pil = PIL.Image.fromarray(image, mode='RGB')
    f = io.BytesIO()
    img_pil.save(f, format='PNG')
    data = f.getvalue()
    encData = codecs.encode(data, 'base64').decode()
    encData = encData.replace('\n', '')
    return encData


def img_gen(config_path, img_root):

    with open(config_path) as f:
        config = commentjson.loads(f.read())

    print('Restore model')
    predictor = BehaviorPredictor(config['predictor'])

    lnmk_68 = [
        'countour_lnmk_0',
        'countour_lnmk_1',
        'countour_lnmk_2',
        'countour_lnmk_3',
        'countour_lnmk_4',
        'countour_lnmk_5',
        'countour_lnmk_6',
        'countour_lnmk_7',
        'countour_lnmk_8',
        'countour_lnmk_9',
        'countour_lnmk_10',
        'countour_lnmk_11',
        'countour_lnmk_12',
        'countour_lnmk_13',
        'countour_lnmk_14',
        'countour_lnmk_15',
        'countour_lnmk_16',
        'left_eyebrow_17',
        'left_eyebrow_18',
        'left_eyebrow_19',
        'left_eyebrow_20',
        'left_eyebrow_21',
        'right_eyebrow_22',
        'right_eyebrow_23',
        'right_eyebrow_24',
        'right_eyebrow_25',
        'right_eyebrow_26',
        'nose_lnmk_27',
        'nose_lnmk_28',
        'nose_lnmk_29',
        'nose_lnmk_30',
        'nose_lnmk_31',
        'nose_lnmk_32',
        'nose_lnmk_33',
        'nose_lnmk_34',
        'nose_lnmk_35',
        'left_eye_lnmk_36',
        'left_eye_lnmk_37',
        'left_eye_lnmk_38',
        'left_eye_lnmk_39',
        'left_eye_lnmk_40',
        'left_eye_lnmk_41',
        'right_eye_lnmk_42',
        'right_eye_lnmk_43',
        'right_eye_lnmk_44',
        'right_eye_lnmk_45',
        'right_eye_lnmk_46',
        'right_eye_lnmk_47',
        'outer_lip_lnmk_48',
        'outer_lip_lnmk_49',
        'outer_lip_lnmk_50',
        'outer_lip_lnmk_51',
        'outer_lip_lnmk_52',
        'outer_lip_lnmk_53',
        'outer_lip_lnmk_54',
        'outer_lip_lnmk_55',
        'outer_lip_lnmk_56',
        'outer_lip_lnmk_57',
        'outer_lip_lnmk_58',
        'outer_lip_lnmk_59',
        'inner_lip_lnmk_60',
        'inner_lip_lnmk_61',
        'inner_lip_lnmk_62',
        'inner_lip_lnmk_63',
        'inner_lip_lnmk_64',
        'inner_lip_lnmk_65',
        'inner_lip_lnmk_66',
        'inner_lip_lnmk_67',
    ]
    img_name_list = glob(os.path.join(img_root, '*.png'))

    for img_path in tqdm(img_name_list):
        img = cv2.imread(img_path)
        img = np.asarray(img)[..., ::-1]
        encData = encodeImageForJson(img)

        lanmd_infos = {
            "version": "4.5.13",
            "flags": {},
            "shapes": [],
            "imagePath": img_path.split('/')[-1],
            "imageData": encData,
            "imageHeight": 256,
            "imageWidth": 256
        }
        landmarks = predictor.pred([img], [img.shape[:2]])
        landmarks = landmarks[0].numpy()
        landmarks = landmarks[:, ::-1]
        for key, lnmk in zip(lnmk_68, landmarks):
            lanmd_infos['shapes'].append({
                "label": key,
                "points": [lnmk.tolist()],
                "group_id": None,
                "shape_type": "point",
                "flags": {}
            })
        save_anno = img_path.replace(".png", '.json')
        dump_json(path=save_anno, data=lanmd_infos)


def parse_config():
    parser = argparse.ArgumentParser('Argparser for model image generate')
    parser.add_argument('--config')
    parser.add_argument('--img_root')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_config()
    print('Result imgs generating')
    print(f"Use following config to produce tensorflow graph: {args.config}.")

    assert os.path.isfile(args.config)
    img_gen(args.config, args.img_root)
