import sys
import cv2
import os
import argparse
import commentjson
import numpy as np
from pprint import pprint
from pathlib import Path
from draw import *

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import load_text
from behavior_predictor.inference import BehaviorPredictor
from utils.load import *

BATCH_SIZE = 1


def save_bin(arr, filename):
    output_file = open(filename, 'wb')
    arr.tofile(output_file)
    output_file.close


def img_gen(config_path, img_path_root, save_root):

    def _get_cates(path):
        cates = [x.strip() for x in load_text(path)]
        target_cat_dict = {i: k for i, k in enumerate(cates)}
        return target_cat_dict

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

    if config['predictor']['mode'] == "pose":

        param_mean_std = np.load(
            "/aidata/anders/objects/3D-head/LS3D-W/params/param_mean_std.npy")
        n_s, n_R, n_shp, n_exp = config['3dmm']['n_s'], config['3dmm'][
            'n_R'], config['3dmm']['n_shp'], config['3dmm']['n_exp']

        mean_s, mean_R, mean_shp, mean_exp = param_mean_std[
            0, :n_s], param_mean_std[0, n_s:n_s + n_R], param_mean_std[
                0, n_s + n_R:n_s + n_R + n_shp], param_mean_std[0,
                                                                209:209 + n_exp]

        std_s, std_R, std_shp, std_exp = param_mean_std[
            1, :n_s], param_mean_std[1, n_s:n_s + n_R], param_mean_std[
                1, n_s + n_R:n_s + n_R + n_shp], param_mean_std[1,
                                                                209:209 + n_exp]

        mean_s_R_shp_exp = np.concatenate([mean_s, mean_R, mean_shp, mean_exp],
                                          axis=0)
        std_s_R_shp_exp = np.concatenate([std_s, std_R, std_shp, std_exp],
                                         axis=0)

        head_model = load_BFM(config['3dmm']['model_path'])
        tex_para = np.random.rand(199, 1)
        colors = head_model['texMU'] + head_model['texPC'].dot(
            tex_para * head_model['texEV'])
        colors = np.reshape(colors, [int(3), int(len(colors) / 3)],
                            'F').T / 255.
        colors = np.minimum(np.maximum(colors, 0), 1)
        triangles = head_model['tri']

    for i, (img_paths,
            img_names) in enumerate(zip(img_path_batchs, img_name_batchs)):
        imgs, origin_shapes, orig_imgs = [], [], []
        for img_path in img_paths:
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            origin_shapes.append((h, w))
            orig_imgs.append(img)
            imgs.append(img)
        rets = predictor.pred(imgs, origin_shapes)
        if config['predictor']['mode'] == "centernet" or config['predictor'][
                'mode'] == "offset" or config['predictor']['mode'] == "tflite":
            target_dict = _get_cates(config['predictor']['cat_path'])
            imgs = draw_box2d(orig_imgs, rets, target_dict)
        elif config['predictor']['mode'] == "pose":
            imgs = draw_pose(head_model, colors, triangles, orig_imgs,
                             mean_s_R_shp_exp, std_s_R_shp_exp, rets)
        for img_name, img in zip(img_names, imgs):
            cv2.imwrite("output.jpg", img)
            exit(1)
            name = img_name.split('_')[-1]
            save_path = os.path.join(save_root, 'det_results')
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            print('writing %s' % os.path.join(save_path, img_name))
            cv2.imwrite(os.path.join(save_path, img_name), img)


def parse_config():
    parser = argparse.ArgumentParser('Argparser for model image generate')
    parser.add_argument('--config')
    parser.add_argument('--img_root')
    parser.add_argument('--save_root')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_config()
    print('Result imgs generating')
    print(f"Use following config to produce tensorflow graph: {args.config}.")

    assert os.path.isfile(args.config)
    img_gen(args.config, args.img_root, args.save_root)
