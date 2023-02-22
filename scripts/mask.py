import os, sys
import numpy as np
import skimage.transform
import cv2
import argparse
import random
from box import Box
from pathlib import Path
from skimage.io import imread
from utils.render import *
from pprint import pprint
import copy
from tqdm import tqdm
from utils.cython.render import render_cy

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import *
from utils.morphable_model import MorphabelModel
from utils.mesh import transform, render


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


def run_mask(
    save_root,
    bfm,
    bfm_uv_coords,
    face_ind,
    tri,
    template_name2ref_texture_src,
    template_name2uv_mask_src,
    anno_path,
    img_root,
    aug_img_size,
    uv_size,
):
    X_ind = bfm.kpt_ind
    X_ind_all = np.stack([X_ind * 3, X_ind * 3 + 1, X_ind * 3 + 2])
    X_ind_all = np.concatenate([
        X_ind_all[:, :17], X_ind_all[:, 17:27], X_ind_all[:, 36:48],
        X_ind_all[:, 27:36], X_ind_all[:, 48:68]
    ],
                               axis=-1)
    valid_ind = np.reshape(np.transpose(X_ind_all), (-1))
    bfm_uv_coords = process_uv(bfm_uv_coords, uv_h=uv_size[0], uv_w=uv_size[1])
    annos = load_json(anno_path)
    for frame in tqdm(annos["frame_list"]):
        name = frame["name"].replace(".json", '.jpg')
        img = cv2.imread(os.path.join(img_root, name)) / 255

        h, w, _ = img.shape
        for lb in frame['labels']:
            # attr = lb['attributes']
            # if attr['mask'] == True:
            box2d = lb['box2d'], 
            keypoints = lb['keypoints']
            kps = np.asarray([keypoints[k] for k in keypoints])
            fitted_sp, fitted_ep, fitted_s, fitted_angles, fitted_t = bfm.fit(
                kps[:, ::-1], X_ind, idxs=None, max_iter=20)
            print("-" * 100)
            print(fitted_angles)
            transformed_vertices = gen_vertices(bfm, fitted_s, fitted_angles,
                                                fitted_t, fitted_sp, fitted_ep)
            transformed_vertices = transformed_vertices.reshape((-1, 3))
            image_vertices = transformed_vertices
            
            # 3. crop image with bbox
            tl = (box2d['x1'], box2d['y1'])
            br = (box2d['x2'], box2d['y2'])
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
            tform = skimage.transform.estimate_transform(
                'similarity', src_pts, DST_PTS)
            # cropped_image will be as texture in uv map
            cropped_image = skimage.transform.warp(
                img,
                tform.inverse,
                output_shape=(aug_img_size[0], aug_img_size[1]))
            # transform face position(image vertices) along with 2d facial image
            position = image_vertices

            position[:, 2] = 1
            position = np.dot(position, tform.params.T)
            position[:,
                     2] = image_vertices[:, 2] * tform.params[0, 0]  # scale z
            position[:,
                     2] = position[:, 2] - np.min(position[:, 2])  # translate

            # 4. uv position map: render position in uv space
            uv_position_map = copy.deepcopy(
                render.render_colors(bfm_uv_coords,
                                     bfm.full_triangles,
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
                                       uv_position_map[:, :, :2].astype(
                                           np.float32),
                                       None,
                                       interpolation=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=(0))
            new_texture = get_new_texture(ref_texture_src, uv_mask_src,
                                          uv_texture_map)
            new_colors = get_colors_from_texture(256, face_ind, new_texture)
            vertices = get_vertices(256, face_ind, pos)
            vertices = vertices.T
            vertices_shape0 = vertices.shape[1]
            vis_colors = np.ones((1, vertices_shape0))
            face_mask = render_texture(vertices, vis_colors, tri.T, h, w, 1)
            new_image = render_texture(vertices, new_colors.T, tri.T, h, w, 3)
            new_image = img * (1 - face_mask) + new_image * face_mask
            img = np.clip(new_image, -1, 1)  #must clip to (-1, 1)!
        # cv2.imwrite(os.path.join(save_root, name), img * 255)


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_config()
    cfg = Box(load_json(args.config))
    bfm = MorphabelModel(cfg.BFM.BFM_PATH)
    bfm_uv_coords = load_uv_coords(cfg.BFM.BFM_UV_PATH)
    face_ind = np.loadtxt(
        os.path.join(cfg.MASK.data_root,
                     "uv-data/face_ind.txt")).astype(np.int32)
    tri = np.loadtxt(os.path.join(cfg.MASK.data_root,
                                  "uv-data/triangles.txt")).astype(np.int32)
    template_name2ref_texture_src, template_name2uv_mask_src = get_ref_texture_src(
        cfg.MASK.uv_face_path, cfg.MASK.mask_template_folder)
    run_mask(
        cfg.save_root,
        bfm,
        bfm_uv_coords,
        face_ind,
        tri,
        template_name2ref_texture_src,
        template_name2uv_mask_src,
        cfg.anno_path,
        cfg.img_root,
        cfg.img_size,
        cfg.uv_size,
    )
