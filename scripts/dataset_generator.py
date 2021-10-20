from pprint import pprint
from tqdm import tqdm
from box import Box
import numpy as np
import sys
import argparse
import cv2
import os
import json
import tensorflow as tf
import math


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


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
    obj_kp = np.where(obj_kp > 0., obj_kp, 0.)
    obj_kp[:, 0] = np.where(obj_kp[:, 0] < img_info['height'], obj_kp[:, 0],
                            img_info['height'] - 1)
    obj_kp[:, 1] = np.where(obj_kp[:, 1] < img_info['width'], obj_kp[:, 1],
                            img_info['width'] - 1)
    obj_kp = np.where(bool_mask, np.inf, obj_kp)

    return obj_kp


def make_dir(path):
    if not os.path.exists(path):
        os.umask(0)
        os.makedirs(path, mode=0o755)


def build_human_keypoint(obj, kp_cates, img_info):
    def get_human_keypoint(obj, kp_cates):
        '''
        generate humankeypoint
        There are 13 joints:
        "head","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist",
        "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"
        for each joint has x, y, and v(visualization state)
        [y, x, v]
        '''
        visiable = 2
        obj_state = obj['attributes']['keypointState']
        obj_kp = obj['humanKeypoint']
        obj_state = np.asarray([obj_state[key] for key in obj_state])
        obj_kp = [obj_kp[key][::-1] for key in obj_kp]
        obj_kp = np.concatenate(obj_kp)
        obj_kp = obj_kp.reshape(13, 2)

        # number of keypoints
        cat = np.asarray(list(kp_cates.values()))
        cat = np.reshape(cat, [len(list(kp_cates.values())), 1])
        cat = cat.astype(np.float32)

        obj_kp = np.concatenate([obj_kp, cat], axis=-1)
        obj_kp[np.where(obj_state != visiable)] = np.inf
        return obj_kp

    obj_kp = get_human_keypoint(obj, kp_cates)
    bool_mask = np.isinf(obj_kp).astype(np.float)
    obj_kp = np.where(obj_kp > 0., obj_kp, 0.)
    obj_kp[:, 0] = np.where(obj_kp[:, 0] < img_info['height'], obj_kp[:, 0],
                            img_info['height'] - 1)
    obj_kp[:, 1] = np.where(obj_kp[:, 1] < img_info['width'], obj_kp[:, 1],
                            img_info['width'] - 1)
    obj_kp = np.where(bool_mask, np.inf, obj_kp)
    return obj_kp


def complement(annos, max_num):
    c = annos.shape[-1]
    # assign numpy array to avoid > max_num case
    annos = np.asarray([x for x in annos if x.size != 0])
    if len(annos) < max_num:
        complement = max_num - len(annos)
        # number of keypoints
        # this 2 is coors; hard code term
        complement = np.empty([complement, 2, c])
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
              exclude_cates,
              max_obj=None,
              obj_classes=None,
              train_ratio=0.8):
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
    batch_kps = []
    img_paths = []
    discard_imgs = Box({
        'invalid': 0,
        'less_than': 0,
    })
    obj_counts = Box({'total_2d': 0})
    if obj_classes is not None:
        obj_cates = {k: i for i, k in enumerate(obj_classes)}
    num_frames = len(anno['frame_list'])
    num_train_files = math.ceil(num_frames * train_ratio)
    num_test_files = num_frames - num_train_files
    save_root = os.path.abspath(os.path.join(img_root, os.pardir,
                                             'tf_records'))

    # gen btach frame list
    for frame in tqdm(anno['frame_list']):
        num_train_files -= 1
        frame_kps = []
        img_name = frame['name']
        img_path = os.path.join(img_root, img_name)
        if 'Wider' in frame['dataset']:
            img_path = os.path.join('/work/anders1234/WF/imgs', img_name)
        else:
            img_path = os.path.join(img_root, 'aug_' + img_name)
        img, img_info = is_img_valid(img_path)
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_NEAREST)
        if not img_info or len(frame['labels']) == 0:
            discard_imgs.invalid += 1
            continue
        for obj in frame['labels']:
            if exclude_cates and obj['category'].lower() in exclude_cates:
                continue
            if obj_classes is not None:
                obj_kp = build_2d_obj(obj, obj_cates, img_info)
                frame_kps.append(obj_kp)
                obj_counts.total_2d += 1
                if min_num > len(frame_kps):
                    discard_imgs.less_than += 1
                    continue

        if obj_classes is not None:
            frame_kps = np.asarray(frame_kps, dtype=np.float32)
            frame_kps = complement(frame_kps, max_obj)
            batch_kps.append(frame_kps)

        frame_kps = frame_kps.tostring()
        img = img[..., ::-1]
        img = img.tostring()
        filename = img_path.split('/')[-1].replace('jpg', 'tfrecords')

        example = tf.train.Example(features=tf.train.Features(
            feature={
                'origin_height': _int64_feature(img_info['height']),
                'origin_width': _int64_feature(img_info['width']),
                'b_images': _bytes_feature(img),
                'b_coords': _bytes_feature(frame_kps)
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
        img_paths.append(img_path)
    output = {
        'total_2d': obj_counts.total_2d,
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
    parser.add_argument('--img_size', default=[320, 192])
    parser.add_argument('--obj_cate_file', type=str)
    parser.add_argument('--max_obj', default=15, type=int)
    parser.add_argument('--exclude_cates', default=None, nargs='+')
    parser.add_argument('--min_num', default=1, type=int)
    parser.add_argument('--train_ratio', default=0.9, type=float)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_config()
    if args.exclude_cates:
        exclude_cates = [x.lower() for x in args.exclude_cates]
    else:
        exclude_cates = args.exclude_cates
    if args.anno_file_names is None:
        anno_file_names = [
            x for x in os.listdir(args.anno_root) if 'json' in x
        ]
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
    total_counts = Box({'total_2d': 0})
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
                           exclude_cates,
                           max_obj=args.max_obj,
                           obj_classes=obj_cates,
                           train_ratio=args.train_ratio)
        print('generated TF records are saved in %s' % save_root)
        print(
            'Total 2d objs: %i, Total invalid objs: %i, Total less_than objs: %i'
            % (output['total_2d'], output['discard_invalid_imgs'],
               output['discard_less_than']))
