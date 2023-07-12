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
import random
from skimage.io import imread
import skimage.transform
from pathlib import Path
import copy

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.morphable_model import MorphabelModel
from utils.mesh import render
from utils.mesh.transform import angle2matrix
from utils.io import load_uv_coords

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
    obj_kp = np.where(obj_kp < 0., 0., obj_kp)
    obj_kp = np.where(obj_kp == np.inf, -1, obj_kp)

    obj_kp[:, 0] = np.where(obj_kp[:, 0] < img_info['height'], obj_kp[:, 0],
                            img_info['height'] - 1)

    obj_kp[:, 1] = np.where(obj_kp[:, 1] < img_info['width'], obj_kp[:, 1],
                            img_info['width'] - 1)
    bool_mask = np.isinf(obj_kp).astype(np.float32)
    obj_kp = np.where(bool_mask, np.inf, obj_kp)
    cat_lb = obj_cates[obj['category']]
    cat_lb = np.expand_dims(np.asarray([cat_lb]), axis=-1)
    cat_lb = np.tile(cat_lb, [obj_kp.shape[0], 1])
    obj_kp = np.concatenate([obj_kp, cat_lb], axis=-1)

    return obj_kp


def gen_vertices(bfm, fitted_s, fitted_angles, fitted_t, fitted_sp, fitted_ep):
    fitted_angles = np.asarray(fitted_angles)
    fitted_vertices = bfm.generate_vertices(fitted_sp, fitted_ep)
    transformed_vertices = bfm.transform(fitted_vertices, fitted_s,
                                         fitted_angles, fitted_t)
    return np.reshape(transformed_vertices, (-1))


def get_vertices(resolution, face_ind, pos):
    all_vertices = np.reshape(pos, [resolution**2, -1])
    vertices = all_vertices[face_ind, :]
    return vertices


def get_new_texture(ref_texture_src, uv_mask_src, texture):
    """Get new texture
        Mainly for data augmentation.
    """
    # random augmentation
    ref_texture = ref_texture_src
    uv_mask = uv_mask_src
    new_texture = texture * (1 - uv_mask[:, :, np.newaxis]
                             ) + ref_texture[:, :, :3] * uv_mask[:, :,
                                                                 np.newaxis]
    return new_texture


def get_colors_from_texture(resolution, face_ind, texture):

    all_colors = np.reshape(texture, [resolution**2, -1])
    colors = all_colors[face_ind, :]
    return colors


def process_uv(uv_coords, uv_h=256, uv_w=256):
    uv_coords[:, 0] = uv_coords[:, 0] * (uv_w - 1)
    uv_coords[:, 1] = uv_coords[:, 1] * (uv_h - 1)
    uv_coords[:, 1] = uv_h - uv_coords[:, 1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros(
        (uv_coords.shape[0], 1))))  # add z
    return uv_coords


def get_ref_texture_src(uv_face_path, mask_template_folder):
    template_name2ref_texture_src = {}
    template_name2uv_mask_src = {}
    mask_template_list = os.listdir(mask_template_folder)
    uv_face = imread(uv_face_path, as_gray=True) / 255.
    for mask_template in mask_template_list:
        mask_template_path = os.path.join(mask_template_folder, mask_template)

        # mask texture
        ref_texture_src = imread(mask_template_path, as_gray=False) / 255.

        if ref_texture_src.shape[
                2] == 4:  # must 4 channel, how about 3 channel?
            uv_mask_src = ref_texture_src[:, :, 3]
            ref_texture_src = ref_texture_src[:, :, :3]
        else:
            print('Fatal error!', mask_template_path)
        uv_mask_src[uv_face == 0] = 0
        template_name2ref_texture_src[mask_template] = ref_texture_src
        template_name2uv_mask_src[mask_template] = uv_mask_src
    return template_name2ref_texture_src, template_name2uv_mask_src


def aug_mask(img,
             bbox,
             vertices,
             full_triangles,
             bfm_uv_coords,
             face_ind,
             tri,
             template_name2ref_texture_src,
             template_name2uv_mask_src,
             aug_img_size=[256, 256],
             uv_size=[256, 256]):
    img = img / 255.
    h, w, _ = img.shape
    tl, br = bbox[:, ::-1]
    center = np.array(
        [br[0] - (br[0] - tl[0]) / 2.0, br[1] - (br[1] - tl[1]) / 2.0])
    old_size = (br[0] - tl[0] + br[1] - tl[1]) / 2
    size = int(old_size * 2)
    marg = old_size * 0.01
    t_x = np.random.rand() * marg * 2 - marg
    t_y = np.random.rand() * marg * 2 - marg
    size = size * (np.random.rand() * 0.2 + 0.9)
    center[0] = center[0] + t_x
    center[1] = center[1] + t_y

    # crop and record the transform parameters
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2],
                        [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, aug_img_size[0] - 1],
                        [aug_img_size[1] - 1, 0]])
    tform = skimage.transform.estimate_transform('similarity', src_pts, DST_PTS)
    # cropped_image will be as texture in uv map
    cropped_image = skimage.transform.warp(img,
                                           tform.inverse,
                                           output_shape=(aug_img_size[0],
                                                         aug_img_size[1]))
    # transform face position(image vertices) along with 2d facial image
    position = vertices
    position[:, 2] = 1
    position = np.dot(position, tform.params.T)
    position[:, 2] = vertices[:, 2] * tform.params[0, 0]  # scale z
    position[:, 2] = position[:, 2] - np.min(position[:, 2])  # translate
    # 4. uv position map: render position in uv space
    uv_position_map = copy.deepcopy(
        render.render_colors(bfm_uv_coords,
                             full_triangles,
                             position,
                             uv_size[0],
                             uv_size[1],
                             c=3))
    # restore from uv space
    cropped_vertices = np.reshape(uv_position_map, [-1, 3]).T
    z = cropped_vertices[2, :] / tform.params[0, 0]
    cropped_vertices[2, :] = 1
    vertices = np.dot(np.linalg.inv(tform.params),
                      cropped_vertices)  # convert to image size infos
    vertices = np.vstack((vertices[:2, :], z))
    pos = np.reshape(vertices.T, aug_img_size + [3])
    template_name = "{}.png".format(random.randint(0, 8 - 1))
    ref_texture_src = template_name2ref_texture_src[template_name]
    uv_mask_src = template_name2uv_mask_src[template_name]
    uv_texture_map = cv2.remap(cropped_image,
                               uv_position_map[:, :, :2].astype(np.float32),
                               None,
                               interpolation=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0))

    new_texture = get_new_texture(ref_texture_src, uv_mask_src, uv_texture_map)
    new_colors = get_colors_from_texture(256, face_ind, new_texture)
    vertices = get_vertices(256, face_ind, pos)
    vis_colors = np.ones(shape=(vertices.shape[0], 1))
    face_mask = render.render_mask_texture(vertices, vis_colors, tri, h, w, 1)
    new_image = render.render_mask_texture(vertices, new_colors, tri, h, w, 3)
    new_image = img * (1 - face_mask) + new_image * face_mask
    img = np.clip(new_image, -1, 1)  #must clip to (-1, 1)!
    img = img * 255
    return img


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
    # cat_lb = obj_cates[cat_key]
    cat_lb = 0
    cat_lb = np.expand_dims(np.asarray([cat_lb, cat_lb]), axis=-1)
    obj_kp = np.concatenate([obj_kp, cat_lb], axis=-1)
    bool_mask = np.isinf(obj_kp).astype(np.float32)
    obj_kp = np.where(obj_kp >= 0., obj_kp, 0.)
    obj_kp[:, 0] = np.where(obj_kp[:, 0] < img_info['height'], obj_kp[:, 0],
                            img_info['height'] - 1)
    obj_kp[:, 1] = np.where(obj_kp[:, 1] < img_info['width'], obj_kp[:, 1],
                            img_info['width'] - 1)
    obj_kp = np.where(bool_mask, np.inf, obj_kp)
    return obj_kp


def get_mean_std(tmp_R, tmp_sp, tmp_ep):
    tmp_R = np.asarray(tmp_R)
    N = tmp_R.shape[0]
    tmp_R = np.stack(np.asarray(tmp_R)).reshape([N, -1])
    tmp_sp = np.squeeze(np.stack(np.asarray(tmp_sp)), axis=-1)
    tmp_ep = np.squeeze(np.stack(np.asarray(tmp_ep)), axis=-1)
    params = np.concatenate([tmp_R, tmp_sp, tmp_ep], axis=-1)
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
    obj_cates = {k: i for i, k in enumerate(obj_classes)}

    num_frames = len(anno['frame_list'])
    num_train_files = math.ceil(num_frames * train_ratio)
    num_test_files = num_frames - num_train_files
    save_root = os.path.abspath(os.path.join(img_root, os.pardir, 'tf_records'))
    # save_root = os.path.abspath(
    #     os.path.join('/home2/user/anders/3D/WF', 'tf_records'))
    frame_count = 0
    root_dir = "/aidata/anders/3D-head/3DDFA"
    bfm_path = os.path.join(root_dir, "BFM/BFM.mat")
    uv_face_path = os.path.join(root_dir, "mask_data/uv-data/uv_face_mask.png")
    mask_template_folder = os.path.join(root_dir, "mask_data/mask-data")
    bfm = MorphabelModel(bfm_path)
    template_name2ref_texture_src, template_name2uv_mask_src = get_ref_texture_src(
        uv_face_path, mask_template_folder)
    bfm_uv_coords = load_uv_coords(os.path.join(root_dir, "BFM/BFM_UV.mat"))
    face_ind = np.loadtxt(
        os.path.join(root_dir,
                     "mask_data/uv-data/face_ind.txt")).astype(np.int32)
    tri = np.loadtxt(os.path.join(
        root_dir, "mask_data/uv-data/triangles.txt")).astype(np.int32)
    bfm_uv_coords = process_uv(bfm_uv_coords, uv_h=256, uv_w=256)
    tmp_R, tmp_sp, tmp_ep, = [], [], []
    tmp_params = []
    param_counts = 0
    for frame in tqdm(anno['frame_list']):
        num_train_files -= 1
        is_masks, frame_kps = [], []
        dataset = frame['dataset']
        img_name = frame['name']
        folder_names = img_name.split("_")
        # img_path = os.path.join(img_root, folder_names[0], folder_names[1],
        #                         folder_names[2], folder_names[3],
        #                         folder_names[4] + "_" + folder_names[5])
        # img_path = os.path.join(img_root, dataset, 'imgs', img_name)
        img_path = os.path.join(img_root, img_name)
        img, img_info = is_img_valid(img_path)
        if not img_info or len(frame['labels']) == 0 or img is None:
            discard_imgs.invalid += 1
            continue
        resized = np.array(img_size[::-1]) / np.array(
            [img_info['height'], img_info['width']])
        scale_factor = np.array(list(resized) + list(resized), dtype=np.float32)
        for obj in frame['labels']:
            bbox = build_2d_obj(obj, obj_cates, img_info)
            lnmks = np.zeros(shape=(68, 3))
            if task == 'keypoints' or task == 'obj_det':
                lnmks = build_keypoints(obj, obj_cates, img_info)

                fitted_sp, fitted_ep, fitted_s, fitted_angles, fitted_t = bfm.fit(
                    lnmks[:, :2][:, ::-1], bfm.kpt_ind, max_iter=3)
                params = np.concatenate([
                    np.squeeze(fitted_sp, axis=-1),
                    np.squeeze(fitted_ep, axis=-1), fitted_s[None],
                    fitted_angles, fitted_t
                ],
                                        axis=-1)
                tmp_params.append(params)
                if 0.0 > np.random.rand():
                    transformed_vertices = gen_vertices(bfm, fitted_s,
                                                        fitted_angles, fitted_t,
                                                        fitted_sp, fitted_ep)
                    transformed_vertices = transformed_vertices.reshape((-1, 3))
                    img = aug_mask(img, bbox[:, :2], transformed_vertices,
                                   bfm.full_triangles, bfm_uv_coords, face_ind,
                                   tri, template_name2ref_texture_src,
                                   template_name2uv_mask_src)
                    is_masks.append(True)

            param_counts += 1
            lnmks = np.concatenate([bbox, lnmks], axis=0)
            if task == 'obj_det':
                R = angle2matrix(np.asarray(fitted_angles))
                tmp_R.append(R)
                tmp_sp.append(fitted_sp)
                tmp_ep.append(fitted_ep)
            frame_kps.append(lnmks)
            obj_counts.total_2d += 1
            obj_counts.total_kps += 1
            if min_num > len(frame_kps):
                discard_imgs.less_than += 1
                continue
        # fill in keypoints
        frame_kps = complement(np.asarray(frame_kps, dtype=np.float32), max_obj)
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
        imgT = imgT.tobytes()
        frame_kps = frame_kps.tobytes()
        is_masks = is_masks.tobytes()
        scale_factor = scale_factor.tobytes()
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
                'is_masks': _bytes_feature(is_masks),
                'scale_factor': _bytes_feature(scale_factor)
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

    if task == 'obj_det':
        mean, std = get_mean_std(tmp_R, tmp_sp, tmp_ep)
        save_dir = os.path.join(save_root, 'params')
        make_dir(save_dir)
        np.save(os.path.join(save_dir, 'param_mean_std.npy'),
                np.stack([mean, std]))
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
    parser.add_argument('--img_size', default=(320, 320), type=tuple)
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
