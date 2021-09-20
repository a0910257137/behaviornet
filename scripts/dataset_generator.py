from pprint import pprint
from numpy.core.fromnumeric import reshape
from tqdm import tqdm
from box import Box
import numpy as np
import sys
import argparse
import cv2
import os
import json


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
              anno_path,
              min_num,
              exclude_cates,
              max_obj=None,
              obj_classes=None,
              max_human=None,
              human_classes=None):
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
    batch_kps, batch_rels, batch_human_kps = [], [], []
    img_paths, img_infos = [], []
    discard_imgs = Box({
        'invalid': 0,
        'less_than': 0,
    })
    obj_counts = Box({'total_2d': 0, 'total_3d': 0, 'total_humankp': 0})
    if obj_classes is not None:
        obj_cates = {k: i for i, k in enumerate(obj_classes)}

    if human_classes is not None:
        human_kp_cates = {k: i for i, k in enumerate(human_classes)}
    # gen btach frame list
    for frame in tqdm(anno['frame_list']):
        frame_kps, frame_rels, frame_human_kps = [], [], []
        if 'sequence' not in frame.keys():
            continue
        if len(frame['labels']) == 0:
            continue
        img_name = frame['name']
        # img_name = frame['name']
        img_path = os.path.join(img_root, img_name)
        img, img_info = is_img_valid(img_path)
        if not img_info:
            discard_imgs.invalid += 1
            continue
        if len(frame['labels']) == 0:
            continue
        img_infos.append(
            np.asarray([img_info['height'], img_info['width']], np.float32))

        for obj in frame['labels']:
            if 'box2d' not in list(obj.keys()):
                continue
            if exclude_cates and obj['category'].lower() in exclude_cates:
                continue
            if obj_classes is not None:
                obj_kp = build_2d_obj(obj, obj_cates, img_info)
                frame_kps.append(obj_kp)
                obj_counts.total_2d += 1
                if min_num > len(frame_kps):
                    discard_imgs.less_than += 1
                    continue
            if human_classes is not None:
                if sum(obj['attributes']['keypointState'].values()) == 0:
                    continue
                human_kp = build_human_keypoint(obj, human_kp_cates, img_info)
                obj_counts.total_humankp += 1
                frame_human_kps.append(human_kp)
        if obj_classes is not None:
            frame_kps = np.asarray(frame_kps, dtype=np.float32)
            frame_kps = complement(frame_kps, max_obj)
            c, _, _ = frame_kps.shape
            batch_kps.append(frame_kps)
        if human_classes is not None:
            frame_human_kps = np.asarray(frame_human_kps, dtype=np.float32)
            frame_human_kps = complement(frame_human_kps, max_obj)
            batch_human_kps.append(frame_human_kps)
        img_paths.append(img_path)
    total_frames = len(img_paths)
    output = {'imgs': img_paths, 'img_infos': img_infos}
    if obj_classes is not None:
        output['kps'] = batch_kps
    if human_classes is not None:
        output['human_kps'] = batch_human_kps
    return output, discard_imgs, obj_counts


def save(payload, save_roots, dataset_root, train_ratio):
    npy_names = [
        x.split('/')[-1].split('.')[0] + '.npy' for x in payload['imgs']
    ]
    split = int(len(npy_names) * train_ratio)
    img_info_root = os.path.join(dataset_root, 'img_infos')
    if not os.path.exists(img_info_root):
        os.makedirs(img_info_root)
    for idx in range(len(payload['imgs'])):
        info_name = npy_names[idx]
        elem = payload['img_infos'][idx]
        np.save(os.path.join(img_info_root, info_name), elem)
    payload.pop('imgs')
    payload.pop('img_infos')
    for key in payload:
        train_save_root = os.path.join(save_roots[key], 'train_%s' % key)
        test_save_root = os.path.join(save_roots[key], 'test_%s' % key)
        if not os.path.exists(train_save_root):
            os.makedirs(train_save_root)
        if not os.path.exists(test_save_root):
            os.makedirs(test_save_root)
        # train
        for idx in range(len(payload[key][:split])):
            elem = payload[key][:split][idx]
            npy_name = npy_names[:split][idx]
            np.save(os.path.join(train_save_root, npy_name), elem)
        # test
        for idx in range(len(payload[key][split:])):
            elem = payload[key][split:][idx]
            npy_name = npy_names[split:][idx]
            np.save(os.path.join(test_save_root, npy_name), elem)


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_root', type=str)
    parser.add_argument('--img_root', type=str)
    parser.add_argument('--anno_file_names', default=None, nargs='+')
    parser.add_argument('--obj_cate_file', type=str)
    parser.add_argument('--max_obj', default=15, type=int)
    parser.add_argument('--human_kps_cate_file', type=str, default=None)
    parser.add_argument('--max_human', default=15, type=int)
    parser.add_argument('--exclude_cates', default=None, nargs='+')
    parser.add_argument('--min_num', default=1, type=int)
    parser.add_argument('--train_ratio', default=0.8, type=float)
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
    kp_root_folder = os.path.join(save_root, 'kps')
    total_discards = Box({
        'invalid': 0,
        'less_than': 0,
    })
    total_counts = Box({'total_2d': 0, 'total_3d': 0, 'total_humankp': 0})
    # read object categories
    if args.obj_cate_file:
        with open(args.obj_cate_file) as f:
            obj_cates = f.readlines()
            obj_cates = [x.strip() for x in obj_cates]
    else:
        obj_cates = None

    if args.human_kps_cate_file:
        with open(args.human_kps_cate_file) as f:
            human_kp_cates = f.readlines()
            human_kp_cates = [x.strip() for x in human_kp_cates]
    else:
        human_kp_cates = None

    save_roots = {'kps': os.path.join(save_root, 'kps')}

    if human_kp_cates is not None:
        save_roots['human_kps'] = os.path.join(save_root, 'human_kps')

    for anno_file_name in anno_file_names:
        print('Process %s' % anno_file_name)
        anno_path = os.path.join(args.anno_root, anno_file_name)
        output, discard_imgs, obj_counts = get_coors(
            args.img_root,
            anno_path,
            args.min_num,
            exclude_cates,
            max_obj=args.max_obj,
            obj_classes=obj_cates,
            max_human=args.max_human,
            human_classes=human_kp_cates)
        save(output, save_roots, save_root, args.train_ratio)
        print('generated npy annos are saved in %s' % save_root)
    total_discards.invalid += discard_imgs.invalid
    total_discards.less_than += discard_imgs.less_than
    total_counts.total_2d += obj_counts.total_2d
    total_counts.total_3d += obj_counts.total_3d
    total_counts.total_humankp += obj_counts.total_humankp
    print('Total 2d objs: %i, total 3d objs: %i, total humankp objs: %i' %
          (total_counts.total_2d, total_counts.total_3d,
           total_counts.total_humankp))
