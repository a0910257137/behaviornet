import numpy as np
import sys
import argparse
import cv2
import os
import json
import tensorflow as tf
import math
from pprint import pprint
from tqdm import tqdm
from box import Box
from face_masker import FaceMasker
import random

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.morphable_model import MorphabelModel
from utils.mesh.transform import angle2matrix

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def build_keypoints(obj, obj_cates, img_info):
    kps = obj["keypoints"]
    obj_kp = []
    for key in list(kps.keys()):
        kp = kps[key]
        if np.any(kp == None):
            obj_kp.append([np.inf, np.inf])
        else:
            obj_kp.append(kp)

    obj_kp = np.asarray(obj_kp).astype(np.float32).reshape((-1, 2))
    flag_68_case = False if np.any(obj_kp == np.inf) else True

    obj_kp = np.where(obj_kp < 0., 0., obj_kp)
    obj_kp = np.where(obj_kp == np.inf, -1, obj_kp)

    obj_kp[:, 0] = np.where(obj_kp[:, 0] < img_info['height'], obj_kp[:, 0],
                            img_info['height'] - 1)

    obj_kp[:, 1] = np.where(obj_kp[:, 1] < img_info['width'], obj_kp[:, 1],
                            img_info['width'] - 1)
    bool_mask = np.isinf(obj_kp).astype(np.float)
    obj_kp = np.where(bool_mask, np.inf, obj_kp)
    cat_lb = obj_cates[obj['category']]
    cat_lb = np.expand_dims(np.asarray([cat_lb]), axis=-1)
    cat_lb = np.tile(cat_lb, [obj_kp.shape[0], 1])
    obj_kp = np.concatenate([obj_kp, cat_lb], axis=-1)

    return obj_kp, flag_68_case


def get_2d(box):
    tl = np.asarray([box['y1'], box['x1']])
    br = np.asarray([box['y2'], box['x2']])
    tl = np.expand_dims(tl, axis=-1)
    br = np.expand_dims(br, axis=-1)
    obj_kp = np.concatenate([tl, br], axis=-1)
    obj_kp = np.transpose(obj_kp, [1, 0])
    return obj_kp


def build_2d_obj(obj, obj_cates, img_info):
    obj_kp = get_2d(obj['box2d'])
    obj_name = obj['category'].split(' ')
    cat_key = str()
    for i, l in enumerate(obj_name):
        if i == 0:
            cat_key += l
        else:
            cat_key += '_' + l
    cat_lb = obj_cates[cat_key]
    cat_lb = np.expand_dims(np.asarray([cat_lb, cat_lb]), axis=-1)

    obj_kp = np.concatenate([obj_kp, cat_lb], axis=-1)
    bool_mask = np.isinf(obj_kp).astype(np.float)
    obj_kp = np.where(obj_kp >= 0., obj_kp, 0.)
    obj_kp[:, 0] = np.where(obj_kp[:, 0] < img_info['height'], obj_kp[:, 0],
                            img_info['height'] - 1)
    obj_kp[:, 1] = np.where(obj_kp[:, 1] < img_info['width'], obj_kp[:, 1],
                            img_info['width'] - 1)
    obj_kp = np.where(bool_mask, np.inf, obj_kp)
    return obj_kp


def get_mean_std(tmp_s, tmp_R, tmp_sp, tmp_ep, num_train_files):
    tmp_R = np.asarray(tmp_R)
    N = tmp_R.shape[0]
    tmp_s = np.expand_dims(np.stack(np.asarray(tmp_s)), axis=-1)
    tmp_R = np.stack(np.asarray(tmp_R)).reshape([N, -1])
    tmp_sp = np.squeeze(np.stack(np.asarray(tmp_sp)), axis=-1)
    tmp_ep = np.squeeze(np.stack(np.asarray(tmp_ep)), axis=-1)
    params = np.concatenate([tmp_s, tmp_R, tmp_sp, tmp_ep], axis=-1)
    mean = np.mean(params, axis=0)
    std = np.std(params, axis=0)
    return mean, std


def make_dir(path):
    if not os.path.exists(path):
        os.umask(0)
        os.makedirs(path, mode=0o755)


def complement(annos, max_num):
    n, c, d = annos.shape
    # assign numpy array to avoid > max_num case
    annos = np.asarray([x for x in annos if x.size != 0])
    if len(annos) < max_num:
        complement = max_num - len(annos)
        # number of keypoints
        # this 2 is coors; hard code term
        complement = np.empty([complement, c, d])
        complement.fill(np.inf)
        complement = complement.astype(np.float32)
        if len(annos) == 0:
            annos = complement
        else:
            annos = np.concatenate([annos, complement])
    else:
        annos = annos[:max_num, ...]
    return annos


def complement_masks(annos, max_num):
    annos = np.asarray([x for x in annos if x.size != 0])
    if len(annos) < max_num:
        complement = max_num - len(annos)
        complement = np.empty([complement])
        complement.fill(np.inf)
        complement = complement.astype(np.float32)
        if len(annos) == 0:
            annos = complement
        else:
            annos = np.concatenate([annos, complement])
    else:
        annos = annos[:max_num, ...]
    return annos


def get_coors(img_root,
              img_size,
              anno_path,
              min_num,
              max_obj=None,
              obj_classes=None,
              train_ratio=0.8,
              task='obj_det'):

    def is_img_valid(img_path):
        img_info = {}
        if not os.path.exists(img_path):
            print('%s not exist, bypass' % img_path)
            return None, img_info
        img = cv2.imread(img_path)
        if img is None:
            print('Can not read %s, bypass' % img_path)
            return None, img_info
        img_info['height'] = img.shape[0]
        img_info['width'] = img.shape[1]
        img_info['channel'] = img.shape[2]
        return img, img_info

    def load_json(anno_path):
        with open(anno_path) as f:
            return json.loads(f.read())

    anno = load_json(anno_path)
    discard_imgs = Box({
        'invalid': 0,
        'less_than': 0,
    })
    obj_counts = Box({'total_2d': 0, 'total_kps': 0})
    if obj_classes is not None:
        obj_cates = {k: i for i, k in enumerate(obj_classes)}

    num_frames = len(anno['frame_list'])
    num_train_files = math.ceil(num_frames * train_ratio)
    num_test_files = num_frames - num_train_files
    save_root = os.path.abspath(os.path.join(img_root, os.pardir, 'tf_records'))
    # save_root = os.path.join("/home2/user/anders/3D/total", 'tf_records')
    frame_count = 0
    bfm = MorphabelModel('/home3/user/anders/objects/3D-head/3DDFA/BFM/BFM.mat')
    tmp_s, tmp_R, tmp_sp, tmp_ep, = [], [], [], []
    for frame in tqdm(anno['frame_list']):
        num_train_files -= 1
        is_masks, frame_kps = [], []
        img_name = frame['name']
        dataset = frame['dataset']
        # img_path = os.path.join(img_root, dataset, "imgs", img_name)
        img_path = os.path.join(img_root, img_name)
        img, img_info = is_img_valid(img_path)
        if not img_info or len(frame['labels']) == 0 or img is None:
            discard_imgs.invalid += 1
            continue
        for obj in frame['labels']:
            if obj_classes is not None and task == 'obj_det':
                attributes = obj['attributes']
                is_masks.append(attributes['mask'])
                obj_kp = build_2d_obj(obj, obj_cates, img_info)
                obj_lnmks, flag_68_case = build_keypoints(
                    obj, obj_cates, img_info)

                obj_kps = obj_lnmks[:, :2]
                resized = np.array([192., 320.]) / np.array(
                    [img_info['height'], img_info['width']])
                obj_kps = np.einsum('n c, c -> n c', obj_kps, resized)
                fitted_sp, fitted_ep, fitted_s, fitted_angles, fitted_t = bfm.fit(
                    obj_kps[:, ::-1], bfm.kpt_ind, max_iter=5)
                R = angle2matrix(np.asarray(fitted_angles))
                tmp_s.append(fitted_s)
                tmp_R.append(R)
                tmp_sp.append(fitted_sp)
                tmp_ep.append(fitted_ep)
                obj_kp = np.concatenate([obj_kp, obj_lnmks], axis=0)
                frame_kps.append(obj_kp)
                obj_counts.total_2d += 1
                obj_counts.total_kps += 1
                if min_num > len(frame_kps):
                    discard_imgs.less_than += 1
                    continue

        if obj_classes is not None:
            # fill in keypoints
            frame_kps = complement(np.asarray(frame_kps, dtype=np.float32),
                                   max_obj)
            is_masks = complement_masks(np.asarray(is_masks, dtype=np.float32),
                                        max_obj)

        imgT = img[..., ::-1]
        imgT = np.asarray(imgT).astype(np.uint8)
        imgT = cv2.resize(imgT, img_size, interpolation=cv2.INTER_NEAREST)
        resized_shape = np.array(img_size) / np.array(
            [img_info['width'], img_info['height']])
        obj_kps, cates = frame_kps[..., :2], frame_kps[..., -1:]

        mask = obj_kps == -1
        obj_kps = np.einsum('n c d, d -> n c d', obj_kps, resized_shape[::-1])
        obj_kps[mask] = -1
        frame_kps = np.concatenate([obj_kps, cates], axis=-1).astype(np.float32)
        frame_kps = frame_kps.tostring()
        is_masks = is_masks.tostring()
        imgT = imgT.tostring()
        if img_path.split('/')[-1].split('.')[-1] == 'png':
            filename = img_path.split('/')[-1].replace('png', 'tfrecords')
        else:
            filename = img_path.split('/')[-1].replace('jpg', 'tfrecords')
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'origin_height': _int64_feature(img_info['height']),
                'origin_width': _int64_feature(img_info['width']),
                'b_images': _bytes_feature(imgT),
                'b_coords': _bytes_feature(frame_kps),
                'is_masks': _bytes_feature(is_masks)
            }))
        if num_train_files > 0:
            save_dir = os.path.join(save_root, 'train')
            make_dir(save_dir)
            writer = tf.io.TFRecordWriter(os.path.join(save_dir, filename))
            writer.write(example.SerializeToString())
        else:
            save_dir = os.path.join(save_root, 'test')
            make_dir(save_dir)
            writer = tf.io.TFRecordWriter(os.path.join(save_dir, filename))
            writer.write(example.SerializeToString())
        writer.close()
        frame_count += 1

    mean, std = get_mean_std(tmp_s, tmp_R, tmp_sp, tmp_ep, num_train_files)
    save_dir = os.path.join(save_root, 'params')
    make_dir(save_dir)

    np.save(os.path.join(save_dir, 'param_mean_std.npy'), np.stack([mean, std]))
    output = {
        'total_2d': obj_counts.total_2d,
        'total_kps': obj_counts.total_kps,
        'total_frames': num_frames,
        'train_frames': num_train_files,
        'test_frames': num_test_files,
        "discard_invalid_imgs": discard_imgs.invalid,
        "discard_less_than": discard_imgs.less_than
    }

    return output


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_root', type=str)
    parser.add_argument('--anno_file_names', default=None, nargs='+')
    parser.add_argument('--img_root', type=str)
    parser.add_argument('--obj_cate_file', type=str)
    parser.add_argument('--img_size', default=(320, 192), type=tuple)
    parser.add_argument('--max_obj', default=15, type=int)
    parser.add_argument('--min_num', default=1, type=int)
    parser.add_argument('--train_ratio', default=0.8, type=float)
    parser.add_argument('--task', default='obj_det', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_config()
    if args.anno_file_names is None:
        anno_file_names = [x for x in os.listdir(args.anno_root) if 'json' in x]
    else:
        anno_file_names = args.anno_file_names

    if len(anno_file_names) == 0:
        print('No annotations, exit')
        sys.exit(0)

    save_root = os.path.abspath(os.path.join(args.anno_root, os.pardir))
    total_discards = Box({
        'invalid': 0,
        'less_than': 0,
    })
    # read object categories
    if args.obj_cate_file:
        with open(args.obj_cate_file) as f:
            obj_cates = f.readlines()
            obj_cates = [x.strip() for x in obj_cates]
    else:
        obj_cates = None
    for anno_file_name in anno_file_names:
        print('Process %s' % anno_file_name)
        anno_path = os.path.join(args.anno_root, anno_file_name)

        output = get_coors(args.img_root,
                           args.img_size,
                           anno_path,
                           args.min_num,
                           max_obj=args.max_obj,
                           obj_classes=obj_cates,
                           train_ratio=args.train_ratio,
                           task=args.task)
        print('generated TF records are saved in %s' % save_root)
        print(
            'Total 2d objs: %i, Total landmark keypoints: %i, Total invalid objs: %i, Total less_than objs: %i'
            % (output['total_2d'], output['total_kps'],
               output['discard_invalid_imgs'], output['discard_less_than']))
