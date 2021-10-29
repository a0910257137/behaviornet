from pprint import pprint
from tqdm import tqdm
from box import Box
import numpy as np

import cv2
import os
import json
from tqdm import tqdm


def load_json(anno_path):
    with open(anno_path) as f:
        return json.loads(f.read())


path = "/aidata/anders/objects/ffhq/annos/BDD_box2d_kps.json"
save_root = "/aidata/anders/objects/ffhq/crop_imgs"
img_root = "/aidata/anders/objects/ffhq/imgs"
annos = load_json(path)
bdd_results = {"frame_list": []}
for frame in tqdm(annos['frame_list']):
    img_name = frame['name']
    img_path = os.path.join(img_root, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue
    for i, lb in enumerate(frame['labels']):
        box2d = lb['box2d']
        tl = np.array([box2d['x1'], box2d['y1']])
        br = np.array([box2d['x2'], box2d['y2']])
        tl = tl.astype(int) - 1
        br = br.astype(int) + 1
        cropped_img = img[tl[1]:br[1], tl[0]:br[0], :]
        splitted_img_name = img_name.split('.')
        pre_fix = splitted_img_name[0] + '_face_{}'.format(i)
        img_name = pre_fix + '.jpg'
        save_path = os.path.join(save_root, img_name)
        # cv2.imwrite(save_path, cropped_img)
