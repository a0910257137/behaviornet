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
import copy

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import dump_json, load_text, load_json
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

    lnmk_25 = [
        'countour_lnmk_0',
        'countour_lnmk_8',
        'countour_lnmk_16',
        'nose_lnmk_27',
        'nose_lnmk_30',
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
        'outer_lip_lnmk_50',
        'outer_lip_lnmk_51',
        'outer_lip_lnmk_52',
        'outer_lip_lnmk_54',
        'outer_lip_lnmk_56',
        'outer_lip_lnmk_57',
        'outer_lip_lnmk_58',
    ]
    box2d_annos = load_json(
        '/aidata/anders/objects/box2d/demo/annos/BDD_test_landmark.json')

    # reference keypoint name
    img_name_list = sorted(list(box2d_annos.keys()))
    for img_name in tqdm(img_name_list):

        img_path = os.path.join(img_root, img_name)
        box2d_lbs = box2d_annos[img_name]
        img = cv2.imread(img_path)

        img = np.asarray(img)[..., ::-1]
        encData = encodeImageForJson(img)

        encoded = base64.b64encode(open(img_path, "rb").read())

        lanmd_infos = {
            "version": "4.5.13",
            "flags": {},
            "shapes": [],
            "imagePath": img_name,
            "imageData": encData,
            "imageHeight": 720,
            "imageWidth": 1280
        }
        for box2d in box2d_lbs:
            input_imgs, origin_shapes, orig_imgs = [], [], []
            box2d = box2d['bbox']
            x1 = box2d[0]
            y1 = box2d[1]
            w = box2d[2]
            h = box2d[3]
            tl = np.asarray((x1, y1)) - 1
            br = np.asarray((x1 + w, y1 + h)) + 1

            crop_img = copy.deepcopy(img[tl[1]:br[1], tl[0]:br[0], :])
            h, w, _ = crop_img.shape
            origin_shapes.append((h, w))
            if h == 0 or w == 0:
                continue
            input_imgs.append(crop_img)
            rets = predictor.pred(input_imgs, origin_shapes)
            landmarks = np.squeeze(rets.numpy(), axis=0)
            landmarks = landmarks[:, ::-1].astype(int)
            landmarks += tl

            for key, lnmk in zip(lnmk_68, landmarks):
                lanmd_infos['shapes'].append({
                    "label": key,
                    "points": [lnmk.tolist()],
                    "group_id": None,
                    "shape_type": "point",
                    "flags": {}
                })
        save_anno = os.path.join('/aidata/anders/objects/labels/images',
                                 img_name.replace(".jpg", '.json'))
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
