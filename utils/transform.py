import numpy as np
from math import cos, sin


def angle2matrix(angles):
    ''' get rotation matrix from three rotation angles(degree). right-handed.
    Args:
        angles: [3,]. x, y, z angles
        x: pitch. positive for looking down.
        y: yaw. positive for looking left. 
        z: roll. positive for tilting head right. 
    Returns:
        R: [3, 3]. rotation matrix.
    '''
    # x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(
    #     angles[2])
    # x
    x, y, z = angles[0], angles[1], angles[2]
    Rx = np.array([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]])
    # y
    Ry = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
    # z
    Rz = np.array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]])

    R = Rz.dot(Ry.dot(Rx))
    return R.astype(np.float32)


def rotate(vertices, angles):
    ''' rotate vertices. 
    X_new = R.dot(X). X: 3 x 1   
    Args:
        vertices: [nver, 3]. 
        rx, ry, rz: degree angles
        rx: pitch. positive for looking down 
        ry: yaw. positive for looking left
        rz: roll. positive for tilting head right
    Returns:
        rotated vertices: [nver, 3]
    '''
    R = angle2matrix(angles)
    rotated_vertices = vertices.dot(R.T)

    return rotated_vertices
