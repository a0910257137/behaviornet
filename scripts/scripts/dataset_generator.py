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


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def calculate_pitch_yaw_roll(landmarks_2D,
                             cam_w=256,
                             cam_h=256,
                             radians=False):
    """ Return the the pitch  yaw and roll angles associated with the input image.
    @param radians When True it returns the angle in radians, otherwise in degrees.
    """

    assert landmarks_2D is not None, 'landmarks_2D is None'

    # Estimated camera matrix values.
    c_x = cam_w / 2
    c_y = cam_h / 2
    f_x = c_x / np.tan(60 / 2 * np.pi / 180)
    f_y = f_x
    camera_matrix = np.float32([[f_x, 0.0, c_x], [0.0, f_y, c_y],
                                [0.0, 0.0, 1.0]])
    camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])

    # dlib (68 landmark) trached points
    # TRACKED_POINTS = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]
    # wflw(98 landmark) trached points
    # TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
    # X-Y-Z with X pointing forward and Y on the left and Z up.
    # The X-Y-Z coordinates used are like the standard coordinates of ROS (robotic operative system)
    # OpenCV uses the reference usually used in computer vision:
    # X points to the right, Y down, Z to the front
    landmarks_3D = np.float32([
        [6.825897, 6.760612, 4.402142],  # LEFT_EYEBROW_LEFT, 
        [1.330353, 7.122144, 6.903745],  # LEFT_EYEBROW_RIGHT, 
        [-1.330353, 7.122144, 6.903745],  # RIGHT_EYEBROW_LEFT,
        [-6.825897, 6.760612, 4.402142],  # RIGHT_EYEBROW_RIGHT,
        [5.311432, 5.485328, 3.987654],  # LEFT_EYE_LEFT,
        [1.789930, 5.393625, 4.413414],  # LEFT_EYE_RIGHT,
        [-1.789930, 5.393625, 4.413414],  # RIGHT_EYE_LEFT,
        [-5.311432, 5.485328, 3.987654],  # RIGHT_EYE_RIGHT,
        [-2.005628, 1.409845, 6.165652],  # NOSE_LEFT,
        [-2.005628, 1.409845, 6.165652],  # NOSE_RIGHT,
        [2.774015, -2.080775, 5.048531],  # MOUTH_LEFT,
        [-2.774015, -2.080775, 5.048531],  # MOUTH_RIGHT,
        [0.000000, -3.116408, 6.097667],  # LOWER_LIP,
        [0.000000, -7.415691, 4.070434],  # CHIN
    ])
    landmarks_2D = np.asarray(landmarks_2D, dtype=np.float32).reshape(-1, 2)

    # Applying the PnP solver to find the 3D pose of the head from the 2D position of the landmarks.
    # retval - bool
    # rvec - Output rotation vector that, together with tvec, brings points from the world coordinate system to the camera coordinate system.
    # tvec - Output translation vector. It is the position of the world origin (SELLION) in camera co-ords

    _, rvec, tvec = cv2.solvePnP(landmarks_3D, landmarks_2D, camera_matrix,
                                 camera_distortion)
    # Get as input the rotational vector, Return a rotational matrix

    # const double PI = 3.141592653;
    # double thetaz = atan2(r21, r11) / PI * 180;
    # double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33*r33)) / PI * 180;
    # double thetax = atan2(r32, r33) / PI * 180;

    rmat, _ = cv2.Rodrigues(rvec)
    pose_mat = cv2.hconcat((rmat, tvec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    return map(lambda k: k[0],
               euler_angles)  # euler_angles contain (pitch, yaw, roll)


def transformation_from_points(points1, points2):
    '''0 - 先確定是float資料型別 '''
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    '''1 - 消除平移的影響 '''
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    '''2 - 消除縮放的影響 '''
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    '''3 - 計算矩陣M=BA^T 對矩陣M進行SVD分解；計算得到R '''
    # ||RA-B||; M=BA^T
    A = points1.T  # 2xN
    B = points2.T  # 2xN
    M = np.dot(B, A.T)
    U, S, Vt = np.linalg.svd(M)
    R = np.dot(U, Vt)

    s = s2 / s1
    sR = s * R
    c1 = c1.reshape(2, 1)
    c2 = c2.reshape(2, 1)
    T = c2 - np.dot(sR, c1)  # 模板人臉的中心位置減去 需要對齊的中心位置（經過旋轉和縮放之後）
    trans_mat = np.hstack([sR, T])  # 2x3

    return trans_mat, s


def build_eye_cls(obj_lnmks, obj):
    obj_lnmks[:2, 2] = -1
    # if obj['attributes']['eye_status'] == 'open':
    obj_lnmks[:2, 2] = True
    return obj_lnmks


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


def get_coors(img_root,
              img_size,
              anno_path,
              min_num,
              exclude_cates,
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
    save_root = os.path.abspath(os.path.join(img_root, os.pardir,
                                             'tf_records'))
    # save_root = os.path.join("/home2/user/anders/driv", 'tf_records')
    # gen btach frame list
    frame_count = 0
    mean_face = None
    face_masker = FaceMasker(
        is_aug=False,
        data_root="/aidata/anders/objects/landmarks/PRNet/Data",
        model_root="/aidata/anders/objects/landmarks/PRNet/model")
    for frame in tqdm(anno['frame_list']):
        num_train_files -= 1
        frame_kps, frame_theta = [], []
        img_name = frame['name']
        dataset = frame['dataset']
        img_path = os.path.join(img_root, img_name)
        # WFLW has sequence
        template_name = "{}.png".format(random.randint(0, 8 - 1))
        # if frame['sequence'] is not None and dataset == "WFLW":
        #     img_path = os.path.join(img_root, frame['sequence'], img_name)
        img, img_info = is_img_valid(img_path)
        if not img_info or len(frame['labels']) == 0 or img is None:
            discard_imgs.invalid += 1
            continue
        # if task == "obj_det":
        #     img = cv2.resize(img, img_size, interpolation=cv2.INTER_NEAREST)
        for obj in frame['labels']:
            if exclude_cates and obj['category'].lower() in exclude_cates:
                continue
            if obj_classes is not None and task == 'obj_det':
                obj_kp = build_2d_obj(obj, obj_cates, img_info)
                obj_lnmks, flag_68_case = build_keypoints(
                    obj, obj_cates, img_info)
                if frame_count == 0 and mean_face is None:
                    mean_face = obj_lnmks[:, :2]
                theta = 90.
                if flag_68_case:
                    trans_mat, scale = transformation_from_points(
                        obj_lnmks[:, :2][:, ::-1], mean_face[:, ::-1])
                    rotate_matrix = trans_mat[:, :2] / scale
                    theta = math.asin(rotate_matrix[0][1]) * 57.3
                    theta = np.round(theta, 3)
                frame_theta.append([theta])
                obj_lnmks = build_eye_cls(obj_lnmks, obj)
                obj_kp = np.concatenate([obj_kp, obj_lnmks], axis=0)
                frame_kps.append(obj_kp)
                obj_counts.total_2d += 1
                obj_counts.total_kps += 1
                if min_num > len(frame_kps):
                    discard_imgs.less_than += 1
                    continue
        if obj_classes is not None:
            frame_kps = np.asarray(frame_kps, dtype=np.float32)
            frame_kps = complement(frame_kps, max_obj)
            # fill in theta
            frame_theta = np.asarray(frame_theta)
            n, d = frame_theta.shape
            if len(frame_theta) < max_obj:
                # number of keypoints
                # this 2 is coors; hard code term
                comp = np.empty([max_obj - len(frame_theta), d])
                comp.fill(np.inf)
                comp = comp.astype(np.float32)
                if len(frame_theta) == 0:
                    frame_theta = comp
                else:
                    frame_theta = np.concatenate([frame_theta, comp])
            else:
                frame_theta = frame_theta[:max_obj, ...]
        imgT = img[..., ::-1]
        if random.random() < 0.6:
            imgT = face_masker.add_mask_one(imgT, frame_kps[:,
                                                            2:, :2][..., ::-1],
                                            template_name)
        imgT = np.asarray(imgT).astype(np.uint8)
        imgT = cv2.resize(imgT, img_size, interpolation=cv2.INTER_NEAREST)
        resized_shape = np.array(img_size) / np.array(
            [img_info['width'], img_info['height']])
        obj_kps, cates = frame_kps[..., :2], frame_kps[..., -1:]
        mask = obj_kps == -1

        obj_kps = np.einsum('n c d, d -> n c d', obj_kps, resized_shape[::-1])
        obj_kps[mask] = -1
        frame_kps = np.concatenate([obj_kps, cates], axis=-1)
        frame_kps = frame_kps.astype(np.float32)

        frame_kps = frame_kps.tostring()
        imgT = imgT.tostring()
        frame_theta = np.asarray(frame_theta).astype(np.float32).tostring()
        if img_path.split('/')[-1].split('.')[-1] == 'png':
            filename = img_path.split('/')[-1].replace('png', 'tfrecords')
        else:
            filename = img_path.split('/')[-1].replace('jpg', 'tfrecords')
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'origin_height': _int64_feature(img_info['height']),
                'origin_width': _int64_feature(img_info['width']),
                'b_theta': _bytes_feature(frame_theta),
                'b_images': _bytes_feature(imgT),
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
        frame_count += 1
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
    parser.add_argument('--img_size', default=(320, 192), type=tuple)
    parser.add_argument('--obj_cate_file', type=str)
    parser.add_argument('--max_obj', default=15, type=int)
    parser.add_argument('--exclude_cates', default=None, nargs='+')
    parser.add_argument('--min_num', default=1, type=int)
    parser.add_argument('--train_ratio', default=0.9, type=float)
    parser.add_argument('--task', default='obj_det', type=str)
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
                           train_ratio=args.train_ratio,
                           task=args.task)
        print('generated TF records are saved in %s' % save_root)
        print(
            'Total 2d objs: %i, Total landmark keypoints: %i, Total invalid objs: %i, Total less_than objs: %i'
            % (output['total_2d'], output['total_kps'],
               output['discard_invalid_imgs'], output['discard_less_than']))
