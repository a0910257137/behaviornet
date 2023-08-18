import sys
import argparse
import cv2
import yaml
import numpy as np
import os
from pathlib import Path
from glob import glob
from tqdm import tqdm
import torch
import copy

sys.path.insert(0, str(Path(__file__).parent.parent))
from third_party.TDDFA.FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
from third_party.TDDFA.TDDFA import TDDFA_ONNX
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from third_party.synthetics.trainer_synthetics import FaceSynthetics

from utils.io import dump_json, load_text

bdd_base = {
    "name":
    None,
    "dataset":
    None,
    "labels": [{
        "attributes": {
            "pose": {
                "roll": None,
                "pitch": None,
                "yaw": None
            }
        },
        "box2d": {
            'x1': None,
            'y1': None,
            'x2': None,
            'y2': None
        },
        "keypoints": {
            'countour_lnmk_0': None,
            'countour_lnmk_1': None,
            'countour_lnmk_2': None,
            'countour_lnmk_3': None,
            'countour_lnmk_4': None,
            'countour_lnmk_5': None,
            'countour_lnmk_6': None,
            'countour_lnmk_7': None,
            'countour_lnmk_8': None,
            'countour_lnmk_9': None,
            'countour_lnmk_10': None,
            'countour_lnmk_11': None,
            'countour_lnmk_12': None,
            'countour_lnmk_13': None,
            'countour_lnmk_14': None,
            'countour_lnmk_15': None,
            'countour_lnmk_16': None,
            'left_eyebrow_17': None,
            'left_eyebrow_18': None,
            'left_eyebrow_19': None,
            'left_eyebrow_20': None,
            'left_eyebrow_21': None,
            'right_eyebrow_22': None,
            'right_eyebrow_23': None,
            'right_eyebrow_24': None,
            'right_eyebrow_25': None,
            'right_eyebrow_26': None,
            'nose_lnmk_27': None,
            'nose_lnmk_28': None,
            'nose_lnmk_29': None,
            'nose_lnmk_30': None,
            'nose_lnmk_31': None,
            'nose_lnmk_32': None,
            'nose_lnmk_33': None,
            'nose_lnmk_34': None,
            'nose_lnmk_35': None,
            'left_eye_lnmk_36': None,
            'left_eye_lnmk_37': None,
            'left_eye_lnmk_38': None,
            'left_eye_lnmk_39': None,
            'left_eye_lnmk_40': None,
            'left_eye_lnmk_41': None,
            'right_eye_lnmk_42': None,
            'right_eye_lnmk_43': None,
            'right_eye_lnmk_44': None,
            'right_eye_lnmk_45': None,
            'right_eye_lnmk_46': None,
            'right_eye_lnmk_47': None,
            'outer_lip_lnmk_48': None,
            'outer_lip_lnmk_49': None,
            'outer_lip_lnmk_50': None,
            'outer_lip_lnmk_51': None,
            'outer_lip_lnmk_52': None,
            'outer_lip_lnmk_53': None,
            'outer_lip_lnmk_54': None,
            'outer_lip_lnmk_55': None,
            'outer_lip_lnmk_56': None,
            'outer_lip_lnmk_57': None,
            'outer_lip_lnmk_58': None,
            'outer_lip_lnmk_59': None,
            'inner_lip_lnmk_60': None,
            'inner_lip_lnmk_61': None,
            'inner_lip_lnmk_62': None,
            'inner_lip_lnmk_63': None,
            'inner_lip_lnmk_64': None,
            'inner_lip_lnmk_65': None,
            'inner_lip_lnmk_66': None,
            'inner_lip_lnmk_67': None,
        }
    }]
}


def get_lnmks_from_synth(img, model, tl, br, is_cuda):
    USE_FLIP = False
    input_size = 256
    flip_parts = ([1, 17], [2, 16], [3, 15], [4, 14], [5, 13], [6, 12], [7, 11],
                  [8, 10], [18, 27], [19, 26], [20, 25], [21, 24], [22, 23],
                  [32, 36], [33, 35], [37, 46], [38, 45], [39, 44], [40, 43],
                  [41, 48], [42, 47], [49, 55], [50, 54], [51, 53], [62, 64],
                  [61, 65], [68, 66], [59, 57], [60, 56])
    tl -= 10
    br += 10
    w, h = br - tl
    center = (tl + br) / 2
    rotate = 0
    _scale = input_size / (max(w, h) * 1.5)
    aimg, M = face_align.transform(img, center, input_size, _scale, rotate)
    aimg = cv2.cvtColor(aimg, cv2.COLOR_BGR2RGB)
    kps = None
    flips = [0, 1] if USE_FLIP else [0]
    for flip in flips:
        input = aimg.copy()
        if flip:
            input = input[:, ::-1, :].copy()
        input = np.transpose(input, (2, 0, 1))
        input = np.expand_dims(input, 0)
        if is_cuda:
            imgs = torch.Tensor(input).cuda()
        else:
            imgs = torch.Tensor(input).cpu()
        imgs.div_(255).sub_(0.5).div_(0.5)

        pred = model(imgs).detach().cpu().numpy().flatten().reshape((-1, 2))
        pred[:, 0:2] += 1
        pred[:, 0:2] *= (input_size // 2)
        if flip:
            pred_flip = pred.copy()
            pred_flip[:, 0] = input_size - 1 - pred_flip[:, 0]
            for pair in flip_parts:
                tmp = pred_flip[pair[0] - 1, :].copy()
                pred_flip[pair[0] - 1, :] = pred_flip[pair[1] - 1, :]
                pred_flip[pair[1] - 1, :] = tmp
            pred = pred_flip
        if kps is None:
            kps = pred
        else:
            kps += pred
            kps /= 2.0
    IM = cv2.invertAffineTransform(M)
    kps = face_align.trans_points(kps, IM)
    kps = np.concatenate([
        kps[:17, :], kps[17:27, :], kps[36:48, :], kps[27:36, :], kps[48:68, :]
    ],
                         axis=0)
    return kps


def run(root_dir, save_path, is_synthetics=False):
    is_cuda = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count()

    gt_text_lines = load_text(os.path.join(root_dir, "annos/output.txt"))
    gt_init_pose = np.load(os.path.join(root_dir, "initial_values.npy"))
    img_root = os.path.join(root_dir, "imgs")

    if is_synthetics:
        app = FaceAnalysis()
        app.prepare(ctx_id=0, det_size=(224, 224))
        path = "./third_party/synthetics/weights/synthetic_resnet50d.ckpt"
        if is_cuda:
            os.environ['CUDA_VISIBLE_DEVICES'] = "0"

            model = FaceSynthetics.load_from_checkpoint(path).cuda()
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
            model = FaceSynthetics.load_from_checkpoint(path).cpu()

    bdd_results = {"frame_list": []}
    cfg_dir = "./third_party/TDDFA/configs"
    cfg = yaml.load(open(os.path.join(cfg_dir, "resnet_120x120.yml")),
                    Loader=yaml.SafeLoader)
    # Init FaceBoxes and TDDFA, recommend using onnx flag
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OMP_NUM_THREADS'] = '4'
    face_boxes = FaceBoxes_ONNX()
    tddfa = TDDFA_ONNX(cfg_dir, **cfg)
    img_paths = glob(os.path.join(img_root, "*.jpg"))
    if len(img_paths) == 0:
        img_paths = glob(os.path.join(img_root, "*.png"))
    counts = 0
    m = len(img_paths)
    tmp = []

    lag_len = 3
    for i in range(m):
        if i < lag_len:
            continue
        tmp.append(os.path.join(img_root, "frame_{}.jpg".format(i)))

    img_paths = tmp
    progress = tqdm(total=m)
    for img_path, text_line in zip(img_paths, gt_text_lines):
        progress.update(1)
        tmp = []
        for i, line in enumerate(text_line.split(", ")):
            if i > 1:
                a = float(line)
                a -= gt_init_pose[i - 2]
                tmp.append(a)
        angles = np.asarray(tmp)
        name = img_path.split("/")[-1]
        bdd_infos = copy.deepcopy(bdd_base)
        bdd_infos['dataset'] = "pose"
        bdd_infos['name'] = name
        img = cv2.imread(img_path)
        # Detect faces, get 3DMM params and roi boxes
        boxes = np.asarray(face_boxes(img))
        n = len(boxes)
        if n == 0:
            counts += 1
            print(f'No face detected, exit')
            continue
        elif n > 1:
            whs = boxes[:, 2:4] - boxes[:, :2]
            areas = whs[:, 0] * whs[:, 1]
            idx = np.argmax(areas)
            boxes = [boxes[idx]]

        param_lst, roi_box_lst = tddfa(img, boxes)
        # Visualization and serialization
        dense_flag = False
        ver_lst = tddfa.recon_vers(param_lst,
                                   roi_box_lst,
                                   dense_flag=dense_flag)

        for i, (vers, roi_box) in enumerate(zip(ver_lst, roi_box_lst)):
            vers = np.concatenate([
                vers[:, :17], vers[:, 17:27], vers[:, 36:48], vers[:, 27:36],
                vers[:, 48:68]
            ],
                                  axis=-1)
            vers = vers.T
            vers = vers[:, :2]
            tl = np.min(vers, axis=0).astype(np.int32)
            br = np.max(vers, axis=0).astype(np.int32)
            lb_base = copy.deepcopy(bdd_infos['labels'][0])
            bdd_infos['labels'].pop()

            lb_base["attributes"] = {
                "pose": {
                    "roll": angles[0],
                    "pitch": angles[1],
                    "yaw": angles[2]
                },
                "valid": True,
                "small": False
            }
            # lb_base["attributes"]["pose"]["roll"] = angles[0]
            # lb_base["attributes"]["pose"]["pitch"] = angles[1]
            # lb_base["attributes"]["pose"]["yaw"] = angles[2]
            lb_base["box2d"]['x1'] = int(tl[0])
            lb_base["box2d"]['y1'] = int(tl[1])
            lb_base["box2d"]['x2'] = int(br[0])
            lb_base["box2d"]['y2'] = int(br[1])
            if is_synthetics:
                vers = get_lnmks_from_synth(img, model, boxes[i][:2],
                                            boxes[i][2:4], is_cuda)
            for i, key in enumerate(lb_base["keypoints"].keys()):
                # img = cv2.circle(img, tuple(vers[i].astype(np.int32)), 3,
                #                  (0, 255, 0), -1)
                lb_base['keypoints'][key] = vers[i][::-1].tolist()
            bdd_infos['labels'].append(lb_base)
        bdd_results["frame_list"].append(bdd_infos)
    print("Lost # of {} bboxes".format(counts))
    dump_json(path=save_path, data=bdd_results)


def parse_config():
    parser = argparse.ArgumentParser("Argparser for generating 3d landmarks")
    parser.add_argument('--root_dir')
    parser.add_argument('--save_path')
    parser.add_argument('--is_synthetic', action='store_true', default=False)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_config()
    bdd_annos = run(args.root_dir, args.save_path, args.is_synthetic)
