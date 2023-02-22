import numpy as np
from utils.io import *
import argparse
from math import cos, sin
from matplotlib import pyplot as plt
from utils.mesh import transform, render
import cv2


def angle2matrix_3ddfa(angles):
    ''' get rotation matrix from three rotation angles(radian). The same as in 3DDFA.
    Args:
        angles: [3,]. x, y, z angles
        x: pitch.
        y: yaw. 
        z: roll. 
    Returns:
        R: 3x3. rotation matrix.
    '''
    x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(
        angles[2])
    # x, y, z = angles[0], angles[1], angles[2]

    # x
    Rx = np.array([[1, 0, 0], [0, cos(x), sin(x)], [0, -sin(x), cos(x)]])
    # y
    Ry = np.array([[cos(y), 0, -sin(y)], [0, 1, 0], [sin(y), 0, cos(y)]])
    # z
    Rz = np.array([[cos(z), sin(z), 0], [-sin(z), cos(z), 0], [0, 0, 1]])
    R = Rx.dot(Ry).dot(Rz)
    return R.astype(np.float32)


def run(bfm):
    kpt_ind = bfm['kpt_ind']
    X_ind_all = np.stack([kpt_ind * 3, kpt_ind * 3 + 1, kpt_ind * 3 + 2])
    X_ind_all = np.concatenate([
        X_ind_all[:, :17], X_ind_all[:, 17:27], X_ind_all[:, 36:48],
        X_ind_all[:, 27:36], X_ind_all[:, 48:68]
    ],
                               axis=-1)
    valid_ind = np.reshape(np.transpose(X_ind_all), (-1))
    shapeMU = bfm['shapeMU']
    shapeMU = np.reshape(shapeMU, (np.shape(shapeMU)[0] // 3, 3))
    mean = np.mean(shapeMU, axis=0, keepdims=True)
    shapeMU -= mean
    shapeMU = np.reshape(shapeMU, (np.shape(shapeMU)[0] * 3, 1))
    shapePC = bfm['shapePC']
    expPC = bfm['expPC']
    n_sp, n_ep = shapePC.shape[-1], expPC.shape[-1]
    sp = np.zeros((n_sp, 1), dtype=np.float32)
    ep = np.zeros((n_ep, 1), dtype=np.float32)
    tp = np.random.rand(1, 1)
    colors = bfm['texMU'] + bfm['texPC'].dot(tp * bfm['texEV'])
    colors = np.reshape(colors, [int(3), int(len(colors) / 3)], 'F').T / 255.
    colors = np.minimum(np.maximum(colors, 0), 1)

    vertices = shapeMU + shapePC.dot(sp) + expPC.dot(ep)
    vertices = np.reshape(vertices, (vertices.shape[0] // 3, 3))

    step = 15
    for angle_i in range(13):
        angles = np.array([0, 90 - step * angle_i, 0])
        R = angle2matrix_3ddfa(angles)
        transformed_vertices = 0.0013 * vertices.dot(R.T)
        image_vertices = transform.to_image(transformed_vertices, [128, 128],
                                            h=256)
        fitted_image = render.render_colors(image_vertices,
                                            bfm['tri'],
                                            colors,
                                            h=256,
                                            w=256)
        fitted_image *= 255
        X = np.reshape(image_vertices, (-1, 1))[valid_ind]
        X = np.reshape(X, (68, 3))
        for i, kp in enumerate(X):
            fitted_image = cv2.circle(fitted_image, (int(kp[0]), int(kp[1])), 3,
                                      (255, 255, 255), 1)
            if i < 17:
                fitted_image = cv2.circle(fitted_image,
                                          (int(kp[0]), int(kp[1])), 3,
                                          (205, 133, 63), -1)
                fitted_image = cv2.circle(fitted_image,
                                          (int(kp[0]), int(kp[1])), 3,
                                          (255, 255, 255), 1)
                if i < 16:
                    fitted_image = cv2.line(
                        fitted_image, (int(X[i][0]), int(X[i][1])),
                        (int(X[i + 1][0]), int(X[i + 1][1])), (0, 0, 0), 1)

            elif 17 <= i < 27:
                fitted_image = cv2.circle(fitted_image,
                                          (int(kp[0]), int(kp[1])), 2,
                                          (205, 186, 150), -1)
                if i < 26 and i != 21:
                    fitted_image = cv2.line(
                        fitted_image, (int(X[i][0]), int(X[i][1])),
                        (int(X[i + 1][0]), int(X[i + 1][1])), (0, 0, 0), 1)
            elif 27 <= i < 39:
                fitted_image = cv2.circle(fitted_image,
                                          (int(kp[0]), int(kp[1])), 2,
                                          (238, 130, 98), -1)
                if i < 38 and i != 32:
                    fitted_image = cv2.line(
                        fitted_image, (int(X[i][0]), int(X[i][1])),
                        (int(X[i + 1][0]), int(X[i + 1][1])), (0, 0, 0), 1)
            elif 39 <= i < 48:
                fitted_image = cv2.circle(fitted_image,
                                          (int(kp[0]), int(kp[1])), 2,
                                          (205, 96, 144), -1)
                if i < 47 and i != 42:
                    fitted_image = cv2.line(
                        fitted_image, (int(X[i][0]), int(X[i][1])),
                        (int(X[i + 1][0]), int(X[i + 1][1])), (0, 0, 0), 1)
            elif 48 <= i < 68:
                fitted_image = cv2.circle(fitted_image,
                                          (int(kp[0]), int(kp[1])), 2,
                                          (0, 191, 255), -1)
                if i < 67 and i != 60:
                    fitted_image = cv2.line(
                        fitted_image, (int(X[i][0]), int(X[i][1])),
                        (int(X[i + 1][0]), int(X[i + 1][1])), (0, 0, 0), 1)
        cv2.imwrite('{}.jpg'.format(-90 + step * angle_i),
                    fitted_image[..., ::-1])


def argparser():
    parser = argparse.ArgumentParser(description='Facial landmarks and bboxes')
    parser.add_argument(
        "--bfm_path",
        type=str,
        help="Configuration file to use",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = argparser()
    bfm = load_BFM(args.bfm_path)
    run(bfm)