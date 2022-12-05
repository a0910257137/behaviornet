import numpy as np
import math

import cv2


def isPointInTri(p, p0, p1, p2):

    v0 = p2[:2] - p0[:2]
    v1 = p1[:2] - p0[:2]
    v2 = p[:2] - p0[:2]

    dot00 = v0.dot(v0)

    dot01 = v0.dot(v1)
    dot02 = v0.dot(v2)
    dot11 = v1.dot(v1)
    dot12 = v1.dot(v2)

    if (dot00 * dot11 - dot01 * dot01 == 0):
        inverDeno = 0
    else:
        inverDeno = 1 / (dot00 * dot11 - dot01 * dot01)

    u = (dot11 * dot02 - dot01 * dot12) * inverDeno
    v = (dot00 * dot12 - dot01 * dot02) * inverDeno

    return (u >= 0) and (v >= 0) and (u + v < 1)


def get_point_weight(p, p0, p1, p2):

    v0 = p2[:2] - p0[:2]
    v1 = p1[:2] - p0[:2]
    v2 = p[:2] - p0[:2]

    dot00 = v0.dot(v0)
    dot01 = v0.dot(v1)
    dot02 = v0.dot(v2)
    dot11 = v1.dot(v1)
    dot12 = v1.dot(v2)

    inverDeno = 0.
    if (dot00 * dot11 - dot01 * dot01 == 0):
        inverDeno = 0
    else:
        inverDeno = 1 / (dot00 * dot11 - dot01 * dot01)

    u = (dot11 * dot02 - dot01 * dot12) * inverDeno
    v = (dot00 * dot12 - dot01 * dot02) * inverDeno
    return np.array([1 - u - v, v, u])


vertices = np.load("/aidata/anders/objects/3D-head/exp/image_vertices.npy")
# vertices = np.load("/aidata/anders/objects/3D-head/exp/uv_coords.npy")
triangles = np.load("/aidata/anders/objects/3D-head/exp/triangles.npy")
# colors = np.load("/aidata/anders/objects/3D-head/exp/attribute.npy")
colors = np.load("/aidata/anders/objects/3D-head/exp/position.npy")

h = w = 256
c = 3
image = np.zeros((h, w, c), dtype=np.float32)

depth_buffer = np.zeros([h, w], dtype=np.float32, order='C') - 999999.
N = triangles.shape[0]
p_x, p_y = 0, 0
triangles = np.reshape(triangles, [-1])
vertices = np.reshape(vertices, [-1])

for i in range(N):
    tri_p0_ind = triangles[3 * i]
    tri_p1_ind = triangles[3 * i + 1]
    tri_p2_ind = triangles[3 * i + 2]

    p0 = np.array([
        vertices[3 * tri_p0_ind], vertices[3 * tri_p0_ind + 1],
        vertices[3 * tri_p0_ind + 2]
    ])

    p1 = np.array([
        vertices[3 * tri_p1_ind], vertices[3 * tri_p1_ind + 1],
        vertices[3 * tri_p1_ind + 2]
    ])

    p2 = np.array([
        vertices[3 * tri_p2_ind], vertices[3 * tri_p2_ind + 1],
        vertices[3 * tri_p2_ind + 2]
    ])
    x_min = max(math.ceil(min(p0[0], min(p1[0], p2[0]))), 0)
    x_max = min(math.floor(max(p0[0], max(p1[0], p2[0]))), w - 1)
    #------------------------------------------------------------

    y_min = max(math.ceil(min(p0[1], min(p1[1], p2[1]))), 0)
    y_max = min(math.floor(max(p0[1], max(p1[1], p2[1]))), h - 1)

    if (x_max < x_min or y_max < y_min):
        continue

    # driscribe the pixel value
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            p = np.array([x, y])
            if (p[0] < 2 or p[0] > w - 3 or p[1] < 2 or p[1] > h - 3
                    or isPointInTri(p, p0, p1, p2)):
                weight = get_point_weight(p, p0, p1, p2)
                p_depth = weight[0] * p0[2] + weight[1] * p1[2] + weight[
                    2] * p2[2]
                if ((p_depth > depth_buffer[y, x])):
                    for k in range(c):
                        p0_color = colors[tri_p0_ind, k]
                        p1_color = colors[tri_p1_ind, k]
                        p2_color = colors[tri_p2_ind, k]

                        p_color = weight[0] * p0_color + weight[
                            1] * p1_color + weight[2] * p2_color
                        image[y, x, k] = p_color

                    depth_buffer[y, x] = p_depth

# image = image * 255
cv2.imwrite("output.jpg", image[..., ::-1])