from pprint import pprint
from utils.io import load_BFM
import numpy as np
import tensorflow as tf


class MorphabelModel:

    def __init__(self, batch_size, max_obj_num, config):
        """
        docstring for  MorphabelModel
            model: nver: number of vertices. ntri: number of triangles. *: must have. ~: can generate ones array for place holder.
                    'shapeMU': [3*nver, 1]. *
                    'shapePC': [3*nver, n_shape_para]. *
                    'shapeEV': [n_shape_para, 1]. ~
                    'expMU': [3*nver, 1]. ~ 
                    'expPC': [3*nver, n_exp_para]. ~
                    'expEV': [n_exp_para, 1]. ~
                    'texMU': [3*nver, 1]. ~
                    'texPC': [3*nver, n_tex_para]. ~
                    'texEV': [n_tex_para, 1]. ~
                    'tri': [ntri, 3] (start from 1, should sub 1 in python and c++). *
                    'tri_mouth': [114, 3] (start from 1, as a supplement to mouth triangles). ~
                    'kpt_ind': [68,] (start from 1). ~
        """
        self.batch_size = batch_size
        self.max_obj_num = max_obj_num
        self.model = load_BFM(config['model_path'])
        self.max_iter = config['max_iter']
        self.n_s = config['n_s']
        self.n_R = config['n_R']
        self.n_shp = config['n_shp']
        self.n_exp = config['n_exp']
        #-------------------- estimate --------------------
        self.kpt_ind = self.model['kpt_ind']
        X_ind_all = np.stack(
            [self.kpt_ind * 3, self.kpt_ind * 3 + 1, self.kpt_ind * 3 + 2])
        X_ind_all = tf.concat([
            X_ind_all[:, :17], X_ind_all[:, 17:27], X_ind_all[:, 36:48],
            X_ind_all[:, 27:36], X_ind_all[:, 48:68]
        ],
                              axis=-1)
        valid_ind = tf.reshape(tf.transpose(X_ind_all), (-1))

        self.shapeMU = tf.gather(tf.cast(self.model['shapeMU'], tf.float64),
                                 valid_ind)
        self.shapePC = tf.gather(
            tf.cast(self.model['shapePC'][:, :self.n_shp], tf.float64),
            valid_ind)
        self.expPC = tf.gather(
            tf.cast(self.model['expPC'][:, :self.n_exp], tf.float64), valid_ind)
        self.shapeMU = tf.tile(self.shapeMU[tf.newaxis, tf.newaxis, ...],
                               [self.batch_size, self.max_obj_num, 1, 1])

        self.shapeMU = tf.reshape(self.shapeMU,
                                  (self.batch_size, self.max_obj_num,
                                   tf.shape(self.shapeMU)[-2] // 3, 3))
        # mean = tf.math.reduce_mean(self.shapeMU, axis=-2, keepdims=True)
        # self.shapeMU -= mean
        self.shapeMU = tf.reshape(self.shapeMU,
                                  (self.batch_size, self.max_obj_num, -1, 1))
        self.shapePC = tf.tile(self.shapePC[tf.newaxis, tf.newaxis, ...],
                               [self.batch_size, self.max_obj_num, 1, 1])
        self.expPC = tf.tile(self.expPC[tf.newaxis, tf.newaxis, ...],
                             [self.batch_size, self.max_obj_num, 1, 1])

        self.expEV = tf.cast(self.model['expEV'][:self.n_exp, :], tf.float64)
        self.expEV = tf.tile(self.expEV[None, None, ...],
                             [self.batch_size, self.max_obj_num, 1, 1])

        self.expEV = tf.squeeze(self.expEV, axis=-1)
        self.shapeEV = tf.cast(self.model['shapeEV'][:self.n_shp, :],
                               tf.float64)
        self.shapeEV = tf.tile(self.shapeEV[None, None, ...],
                               [self.batch_size, self.max_obj_num, 1, 1])
        self.shapeEV = tf.squeeze(self.shapeEV, axis=-1)

    # ---------------- fit ----------------
    @tf.function
    def fit_points(self, x, b_origin_sizes):
        '''
        Args:
            x: (n, 2) image points
            X_ind: (n,) corresponding Model vertex indices
            model: 3DMM
            max_iter: iteration
        Returns:
            sp: (n_sp, 1). shape parameters
            ep: (n_ep, 1). exp parameters
            s, R, t
        '''
        # -- init --
        n_objs, k = tf.shape(x)[1], tf.shape(x)[-2]
        x = tf.cast(x[..., ::-1], tf.float64)
        sp = tf.zeros(shape=(self.n_shp, 1), dtype=tf.float64)
        ep = tf.zeros(shape=(self.n_exp, 1), dtype=tf.float64)
        mask = tf.math.reduce_all(tf.math.is_finite(x), axis=[-1, -2])
        mask_T = tf.tile(mask[:, :, None, None], (1, 1, 3, 3))
        mask_A = tf.tile(mask[:, :, None, None], (1, 1, 68 * 2, 8))
        mask_exp_eq_left = tf.tile(mask[:, :, None, None],
                                   (1, 1, self.n_exp, self.n_exp))
        mask_shp_eq_left = tf.tile(mask[:, :, None, None],
                                   (1, 1, self.n_shp, self.n_shp))
        # pre-process n_objs for shape, express
        for _ in range(self.max_iter):
            X = tf.math.add(
                self.shapeMU,
                tf.linalg.matmul(self.shapePC, sp) +
                tf.linalg.matmul(self.expPC, ep))
            X = tf.reshape(
                X, (self.batch_size, self.max_obj_num, tf.shape(X)[2] // 3, 3))
            P = self.estimate_affine_matrix_3d22d(X, x, mask_A, mask_T)
            s, R, t = self.P2sRt(P)
            #----- estimate shape
            # expression
            shape = tf.linalg.matmul(self.shapePC, sp)
            shape = tf.transpose(
                tf.reshape(shape, (self.batch_size, self.max_obj_num,
                                   tf.shape(shape)[2] // 3, 3)), (0, 1, 3, 2))
            ep = self.estimate_expression(x,
                                          mask_exp_eq_left,
                                          self.shapeMU,
                                          self.expPC,
                                          self.expEV,
                                          shape,
                                          s,
                                          R,
                                          t[:, :, :2],
                                          lamb=20.)
            # shape
            expression = tf.linalg.matmul(self.expPC, ep)
            expression = tf.transpose(
                tf.reshape(expression, (self.batch_size, n_objs, k, 3)),
                (0, 1, 3, 2))
            sp = self.estimate_shape(x,
                                     mask_shp_eq_left,
                                     self.shapeMU,
                                     self.shapePC,
                                     self.shapeEV,
                                     expression,
                                     s,
                                     R,
                                     t[:, :, :2],
                                     lamb=40.)
        angles = self.matrix2angle(R)
        R = self.angle2matrix(angles)
        sp, ep, s, R, t = self.valid_objs(sp, ep, s, R, t, mask)
        s = tf.reshape(s, (self.batch_size, n_objs, -1))
        R = tf.reshape(R, (self.batch_size, n_objs, -1))
        sp = tf.reshape(sp, (self.batch_size, n_objs, -1))
        ep = tf.reshape(ep, (self.batch_size, n_objs, -1))
        t = tf.reshape(t, (self.batch_size, n_objs, -1))
        if self.n_s == 0:
            params = tf.concat([R, sp, ep], axis=-1)
        else:
            params = tf.concat([s, R, sp, ep], axis=-1)
        return params

    def estimate_affine_matrix_3d22d(self, X, x, mask_A, mask_T):
        """
        Using Golden Standard Algorithm for estimating an affine camera
            matrix P from world to image correspondences.
            See Alg.7.2. in MVGCV 
            Code Ref: https://github.com/patrikhuber/eos/blob/master/include/eos/fitting/affine_camera_estimation.hpp
            x_homo = X_homo.dot(P_Affine)
        Args:
            X: [n, 3]. corresponding 3d points(fixed)
            x: [n, 2]. n>=4. 2d points(moving). x = PX
        Returns:
            P_Affine: [3, 4]. Affine camera matrix
            """
        #--- 1. normalization
        x = tf.transpose(x, (0, 1, 3, 2))  # B, n_objs, 2, 68
        X = tf.transpose(X, (0, 1, 3, 2))  # B, 3, 68

        _, n, c, k = x.get_shape().as_list()
        n_objs = tf.shape(x)[1]
        mean = tf.math.reduce_mean(x, keepdims=True, axis=-1)
        x = x - mean
        average_norm = tf.math.reduce_mean(tf.math.sqrt(
            tf.math.reduce_sum(x**2, axis=2)),
                                           axis=-1)  # B, n_objs
        scale = (tf.cast(tf.math.sqrt(2.), tf.float64) / average_norm)[:, :,
                                                                       None,
                                                                       None]
        x = scale * x
        # 2d points
        ms = -mean * scale

        row1 = tf.concat([
            tf.squeeze(scale, axis=2),
            tf.zeros(shape=(self.batch_size, n_objs, 1), dtype=tf.float64),
            ms[:, :, :1, 0]
        ],
                         axis=-1)
        row2 = tf.concat([
            tf.zeros(shape=(self.batch_size, n_objs, 1), dtype=tf.float64),
            tf.squeeze(scale, axis=2), ms[:, :, 1:, 0]
        ],
                         axis=-1)

        row3 = tf.concat([
            tf.zeros(shape=(self.batch_size, n_objs, 1), dtype=tf.float64),
            tf.zeros(shape=(self.batch_size, n_objs, 1), dtype=tf.float64),
            tf.ones(shape=(self.batch_size, n_objs, 1), dtype=tf.float64)
        ],
                         axis=-1)

        T = tf.concat(
            [row1[:, :, None, :], row2[:, :, None, :], row3[:, :, None, :]],
            axis=2)
        T = tf.cast(T, dtype=tf.float64)

        # 3d points
        X_homo = tf.concat([
            X,
            tf.ones(shape=(self.batch_size, n_objs, 1, k), dtype=tf.float64)
        ],
                           axis=2)

        mean = tf.math.reduce_mean(X, keepdims=True, axis=-1)
        X = X - mean

        average_norm = tf.math.reduce_mean(tf.math.sqrt(
            tf.math.reduce_sum(X**2, axis=2)),
                                           axis=-1)
        scale = (tf.cast(tf.math.sqrt(3.), tf.float64) / average_norm)[:, :,
                                                                       None,
                                                                       None]
        X = scale * X
        # 3D

        ms = -mean * scale
        row1 = tf.concat([
            tf.squeeze(scale, axis=2),
            tf.zeros(shape=(self.batch_size, n_objs, 1), dtype=tf.float64),
            tf.zeros(shape=(self.batch_size, n_objs, 1), dtype=tf.float64),
            ms[:, :, :1, 0]
        ],
                         axis=-1)

        row2 = tf.concat([
            tf.zeros(shape=(self.batch_size, n_objs, 1), dtype=tf.float64),
            tf.squeeze(scale, axis=2),
            tf.zeros(shape=(self.batch_size, n_objs, 1), dtype=tf.float64),
            ms[:, :, 1:2, 0]
        ],
                         axis=-1)

        row3 = tf.concat([
            tf.zeros(shape=(self.batch_size, n_objs, 1), dtype=tf.float64),
            tf.zeros(shape=(self.batch_size, n_objs, 1), dtype=tf.float64),
            tf.squeeze(scale, axis=2), ms[:, :, 2:, 0]
        ],
                         axis=-1)
        row4 = tf.concat([
            tf.zeros(shape=(self.batch_size, n_objs, 1), dtype=tf.float64),
            tf.zeros(shape=(self.batch_size, n_objs, 1), dtype=tf.float64),
            tf.zeros(shape=(self.batch_size, n_objs, 1), dtype=tf.float64),
            tf.ones(shape=(self.batch_size, n_objs, 1), dtype=tf.float64)
        ],
                         axis=-1)
        U = tf.concat([
            row1[:, :, None, :], row2[:, :, None, :], row3[:, :, None, :],
            row4[:, :, None, :]
        ],
                      axis=2)
        # --- 2. equations
        X_homo = tf.concat([
            X,
            tf.ones(shape=(self.batch_size, n_objs, 1, k), dtype=tf.float64)
        ],
                           axis=2)

        l = tf.concat([
            X_homo,
            tf.zeros(shape=(self.batch_size, n_objs, 4, 68), dtype=tf.float64)
        ],
                      axis=2)
        r = tf.concat([
            tf.zeros(shape=(self.batch_size, n_objs, 4, 68), dtype=tf.float64),
            X_homo
        ],
                      axis=2)
        A = tf.transpose(tf.concat([l, r], axis=-1), (0, 1, 3, 2))
        b = tf.reshape(x, (self.batch_size, n_objs, c * k, 1))
        # --- 3. solution
        # (B, N, 136, 8)
        # mask_A shape is (36, 15, 136, 8)
        true_idxs = tf.where(mask_A == True)
        valid_A = tf.reshape(tf.gather_nd(A, true_idxs), (-1, 136, 8))[None, :1,
                                                                       ...]
        # valid_A = tf.reshape(A[tf.reduce_all(mask_A, axis=(-2, -1))],
        #                      (self.batch_size, -1, 136, 8))[:, :1, :, :]

        A = tf.where(mask_A, A, tf.tile(valid_A, [self.batch_size, n, 1, 1]))
        p_8 = tf.linalg.matmul(tf.linalg.pinv(A), b)
        row1 = p_8[:, :, None, :4, 0]
        row2 = p_8[:, :, None, 4:, 0]

        row3 = tf.concat([
            tf.zeros(shape=(self.batch_size, n_objs, 1, 1), dtype=tf.float64),
            tf.zeros(shape=(self.batch_size, n_objs, 1, 1), dtype=tf.float64),
            tf.zeros(shape=(self.batch_size, n_objs, 1, 1), dtype=tf.float64),
            tf.ones(shape=(self.batch_size, n_objs, 1, 1), dtype=tf.float64)
        ],
                         axis=-1)
        P = tf.concat([row1, row2, row3], axis=2)
        # --- 4. denormalization
        true_idxs = tf.where(mask_T == True)
        valid_T = tf.reshape(tf.gather_nd(T, true_idxs), (-1, 3, 3))[None, :1,
                                                                     ...]
        # valid_T = tf.reshape(T[tf.reduce_all(mask_T, axis=(-2, -1))],
        #                      (self.batch_size, -1, 3, 3))[:, :1, :, :]
        T = tf.where(mask_T, T, tf.tile(valid_T, [self.batch_size, n, 1, 1]))
        P_Affine = tf.linalg.matmul(tf.linalg.inv(T), tf.linalg.matmul(P, U))
        return P_Affine

    def P2sRt(self, P):
        ''' decompositing camera matrix P
        Args: 
            P: (3, 4). Affine Camera Matrix.
        Returns:
            s: scale factor.
            R: (3, 3). rotation matrix.
            t: (3,). translation. 
        '''
        t = P[:, :, :, 3]
        R1 = P[:, :, 0:1, :3]
        R2 = P[:, :, 1:2, :3]
        s = (tf.linalg.norm(R1, axis=[2, 3]) +
             tf.linalg.norm(R2, axis=[2, 3])) / 2.0

        r1 = R1 / tf.linalg.norm(R1, axis=[2, 3], keepdims=True)
        r2 = R2 / tf.linalg.norm(R2, axis=[2, 3], keepdims=True)
        r3 = tf.linalg.cross(r1, r2)
        R = tf.concat([r1, r2, r3], axis=2)
        return s, R, t

    def matrix2angle(self, R):
        ''' get three Euler angles from Rotation Matrix
        Args:
            R: (3,3). rotation matrix
        Returns:
            x: pitch
            y: yaw
            z: roll
        '''
        sy = tf.math.sqrt(R[:, :, 0, 0] * R[:, :, 0, 0] +
                          R[:, :, 1, 0] * R[:, :, 1, 0])
        singular = sy < 1e-6
        rx = tf.where(singular, tf.math.atan2(-R[:, :, 1, 2], R[:, :, 1, 1]),
                      tf.math.atan2(R[:, :, 2, 1], R[:, :, 2, 2]))
        ry = tf.where(singular, tf.math.atan2(-R[:, :, 2, 0], sy),
                      tf.math.atan2(-R[:, :, 2, 0], sy))
        rz = tf.where(singular, tf.cast(0., tf.float64),
                      tf.math.atan2(R[:, :, 1, 0], R[:, :, 0, 0]))
        angles = tf.concat([rx[:, :, None], ry[:, :, None], rz[:, :, None]],
                           axis=-1)
        return angles

    def estimate_expression(self,
                            x,
                            mask_eq_left,
                            shapeMU,
                            expPC,
                            expEV,
                            shape,
                            s,
                            R,
                            t2d,
                            lamb=2000):
        '''
        Args:
            x: (2, n). image points (to be fitted)
            shapeMU: (3n, 1)
            expPC: (3n, n_ep)
            expEV: (n_ep, 1)
            shape: (3, n)
            s: scale
            R: (3, 3). rotation matrix
            t2d: (2,). 2d translation
            lambda: regulation coefficient

        Returns:
            exp_para: (n_ep, 1) shape parameters(coefficients)
        '''
        x = tf.transpose(x, (0, 1, 3, 2))  # B, n_objs, 2, 68
        dof = tf.shape(expPC)[-1]
        _, n, c, k = x.get_shape().as_list()
        n_objs = tf.shape(x)[1]
        sigma = expEV
        P = tf.constant([[1, 0, 0], [0, 1, 0]], dtype=tf.float64)
        P = tf.tile(P[None, None, :, :], [self.batch_size, n_objs, 1, 1])
        A = s[:, :, None, None] * tf.linalg.matmul(P, R)
        # --- calc pc
        # B, K, D, C
        pc_3d = tf.reshape(
            tf.transpose(expPC, [0, 1, 3, 2]),
            (self.batch_size, n_objs, dof, k, tf.shape(expPC)[2] // k))
        pc_3d = tf.reshape(pc_3d, (self.batch_size, n_objs, dof * k, 3))
        pc_2d = tf.linalg.matmul(pc_3d, A, transpose_b=(0, 1, 3, 2))

        pc = tf.transpose(
            tf.reshape(pc_2d, (self.batch_size, n_objs, dof, c * k)),
            (0, 1, 3, 2))  #B, n_objs, 2n x 29: n = 68
        # --- calc b
        # shapeMU
        mu_3d = tf.transpose(
            tf.reshape(shapeMU, (self.batch_size, n_objs, k, 3)),
            (0, 1, 3, 2))  # (batch, 3 x n)
        # expression
        shape_3d = shape
        b = tf.linalg.matmul(A, (mu_3d + shape_3d)) + tf.tile(
            t2d[..., tf.newaxis], (1, 1, 1, k))  # batch, n_objs, 2 x n
        b = tf.reshape(
            tf.transpose(b, (0, 1, 3, 2)),
            [self.batch_size, n_objs, -1, 1])  # batch, n_objs, 2n x 1
        # --- solve
        equation_left = tf.linalg.matmul(pc, pc, transpose_a=(
            0, 1, 3, 2)) + lamb * tf.linalg.diag(1 / sigma**2)
        x = tf.reshape(tf.transpose(x, (0, 1, 3, 2)),
                       (self.batch_size, n_objs, -1, 1))
        equation_right = tf.linalg.matmul(pc, x - b, transpose_a=(0, 1, 3, 2))

        true_idxs = tf.where(mask_eq_left == True)
        valid_left = tf.reshape(tf.gather_nd(equation_left, true_idxs),
                                (-1, self.n_exp, self.n_exp))[None, :1, ...]

        # valid_left = tf.reshape(
        #     equation_left[tf.reduce_all(mask_eq_left, axis=(-2, -1))],
        #     (self.batch_size, -1, self.n_exp, self.n_exp))[:, :1, :, :]
        equation_left = tf.where(
            mask_eq_left, equation_left,
            tf.tile(valid_left, (self.batch_size, n, 1, 1)))
        exp_para = tf.linalg.matmul(tf.linalg.inv(equation_left),
                                    equation_right)
        return exp_para

    def estimate_shape(self,
                       x,
                       mask_shp_eq_left,
                       shapeMU,
                       shapePC,
                       shapeEV,
                       expression,
                       s,
                       R,
                       t2d,
                       lamb=3000):
        '''
        Args:
            x: (2, n). image points (to be fitted)
            shapeMU: (3n, 1)
            shapePC: (3n, n_sp)
            shapeEV: (n_sp, 1)
            expression: (3, n)
            s: scale
            R: (3, 3). rotation matrix
            t2d: (2,). 2d translation
            lambda: regulation coefficient

        Returns:
            shape_para: (n_sp, 1) shape parameters(coefficients)
        '''

        x = tf.transpose(x, (0, 1, 3, 2))  # B, n_objs, 2, 68
        dof = tf.shape(shapePC)[-1]  # 199
        _, n, c, k = x.get_shape().as_list()  # 68
        n_objs = tf.shape(x)[1]
        sigma = shapeEV
        P = tf.constant([[1., 0., 0.], [0., 1., 0.]], tf.float64)
        P = tf.tile(P[None, None, :, :], (self.batch_size, n_objs, 1, 1))
        # P is (batch, 2, 3), (batch, 3, 3)
        A = s[:, :, None, None] * tf.linalg.matmul(P, R)
        # --- calc pc

        pc_3d = tf.reshape(
            tf.transpose(shapePC, (0, 1, 3, 2)),
            (self.batch_size, n_objs, dof, k, 3))  # batch, n_objs, 199 x n x 3

        pc_3d = tf.reshape(pc_3d, (self.batch_size, n_objs, dof * k, 3))

        pc_2d = tf.linalg.matmul(pc_3d, A, transpose_b=(0, 1, 3, 2))
        pc = tf.transpose(tf.reshape(pc_2d, (self.batch_size, n_objs, dof, -1)),
                          (0, 1, 3, 2))  # batch, n_objs, 2n x 199 : n = 68

        # --- calc b
        # shapeMU
        mu_3d = tf.transpose(
            tf.reshape(shapeMU, (self.batch_size, n_objs, k, 3)),
            (0, 1, 3, 2))  # batch, 3 x n
        # expression
        exp_3d = expression
        #
        b = tf.linalg.matmul(A, (mu_3d + exp_3d)) + tf.tile(
            t2d[..., np.newaxis], [1, 1, 1, k])  # batch, 2, 68 (2 x n)
        b = tf.reshape(tf.transpose(b, (0, 1, 3, 2)),
                       (self.batch_size, n_objs, -1, 1))  # batch, 2n,  1
        # --- solve
        equation_left = tf.linalg.matmul(pc, pc, transpose_a=(
            0, 1, 3, 2)) + lamb * tf.linalg.diag(1 / sigma**2)

        x = tf.reshape(tf.transpose(x, (0, 1, 3, 2)),
                       (self.batch_size, n_objs, -1, 1))
        #TODO:
        # valid right
        equation_right = tf.linalg.matmul(pc, x - b, transpose_a=(0, 1, 3, 2))

        true_idxs = tf.where(mask_shp_eq_left == True)
        valid_left = tf.reshape(tf.gather_nd(equation_left, true_idxs),
                                (-1, self.n_shp, self.n_shp))[None, :1, ...]

        # valid_left = tf.reshape(
        #     equation_left[tf.reduce_all(mask_shp_eq_left, axis=(-2, -1))],
        #     (self.batch_size, -1, self.n_shp, self.n_shp))[:, :1, :, :]
        equation_left = tf.where(
            mask_shp_eq_left, equation_left,
            tf.tile(valid_left, [self.batch_size, n, 1, 1]))
        shape_para = tf.linalg.matmul(tf.linalg.inv(equation_left),
                                      equation_right)

        return shape_para

    def valid_objs(self, sp, ep, s, R, t, mask):
        sp = tf.where(
            tf.tile(mask[..., tf.newaxis, tf.newaxis],
                    (1, 1, tf.shape(sp)[-2], tf.shape(sp)[-1])), sp, np.inf)
        ep = tf.where(
            tf.tile(mask[..., tf.newaxis, tf.newaxis],
                    (1, 1, tf.shape(ep)[-2], tf.shape(ep)[-1])), ep, np.inf)
        s = tf.where(mask, s, np.inf)

        R = tf.where(
            tf.tile(mask[..., tf.newaxis, tf.newaxis],
                    (1, 1, tf.shape(R)[-2], tf.shape(R)[-1])), R, np.inf)
        t = tf.where(tf.tile(mask[..., tf.newaxis], (1, 1, tf.shape(t)[-1])), t,
                     np.inf)
        return sp, ep, s, R, t

    def angle2matrix(self, angles):
        ''' get rotation matrix from three rotation angles(degree). right-handed.
        Args:
            angles: [3,]. x, y, z angles
            x: pitch. positive for looking down.
            y: yaw. positive for looking left. 
            z: roll. positive for tilting head right. 
        Returns:
            R: [3, 3]. rotation matrix.
        '''
        # use 1 rad =  57.3
        _, n, _ = angles.get_shape().as_list()
        x, y, z = angles[..., 0], angles[..., 1], angles[..., 2]
        # x, 3, 3
        # for Rx

        row1 = tf.constant([1., 0., 0.], shape=(1, 3), dtype=tf.float64)
        row1 = tf.tile(row1[None, None, :, :], (self.batch_size, n, 1, 1))
        row2 = tf.concat([
            tf.zeros(shape=(self.batch_size, n, 1, 1), dtype=tf.float64),
            tf.math.cos(x)[..., None, None], -tf.math.sin(x)[..., None, None]
        ],
                         axis=-1)
        row3 = tf.concat([
            tf.zeros(shape=(self.batch_size, n, 1, 1), dtype=tf.float64),
            tf.math.sin(x)[..., None, None],
            tf.math.cos(x)[..., None, None]
        ],
                         axis=-1)
        Rx = tf.concat([row1, row2, row3], axis=-2)
        # for Ry
        # y
        row1 = tf.concat([
            tf.math.cos(y)[..., None, None],
            tf.zeros(shape=(self.batch_size, n, 1, 1), dtype=tf.float64),
            tf.math.sin(y)[..., None, None]
        ],
                         axis=-1)
        row2 = tf.constant([0., 1., 0.], shape=(1, 3), dtype=tf.float64)
        row2 = tf.tile(row2[None, None, :, :], (self.batch_size, n, 1, 1))

        row3 = tf.concat([
            -tf.math.sin(y)[..., None, None],
            tf.zeros(shape=(self.batch_size, n, 1, 1), dtype=tf.float64),
            tf.math.cos(y)[..., None, None]
        ],
                         axis=-1)
        Ry = tf.concat([row1, row2, row3], axis=-2)
        # z
        row1 = tf.concat([
            tf.math.cos(z)[..., None, None], -tf.math.sin(z)[..., None, None],
            tf.zeros(shape=(self.batch_size, n, 1, 1), dtype=tf.float64)
        ],
                         axis=-1)
        row2 = tf.concat([
            tf.math.sin(z)[..., None, None],
            tf.math.cos(z)[..., None, None],
            tf.zeros(shape=(self.batch_size, n, 1, 1), dtype=tf.float64)
        ],
                         axis=-1)
        row3 = tf.constant([0., 0., 1.], shape=(1, 3), dtype=tf.float64)
        row3 = tf.tile(row3[None, None, :, :], (self.batch_size, n, 1, 1))
        Rz = tf.concat([row1, row2, row3], axis=-2)
        R = tf.linalg.matmul(Rz, tf.linalg.matmul(Ry, Rx))
        return R
