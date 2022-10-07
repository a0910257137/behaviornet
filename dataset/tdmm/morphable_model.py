from pprint import pprint
from utils.load import *
import numpy as np
import tensorflow as tf


class MorphabelModel:

    def __init__(self, batch_size, config):
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
        self.model = load_BFM(config['model_path'])
        self.max_iter = config['max_iter']
        # fixed attributes
        self.nver = self.model['shapePC'].shape[0] / 3
        self.ntri = self.model['tri'].shape[0]
        self.n_shape_para = self.model['shapePC'].shape[1]
        self.n_exp_para = self.model['expPC'].shape[1]
        self.n_tex_para = self.model['texMU'].shape[1]

        self.kpt_ind = self.model['kpt_ind']

        # need to rearrange index self.kpt_ind
        self.triangles = self.model['tri']
        self.full_triangles = tf.concat(
            [self.model['tri'], self.model['tri_mouth']], axis=0)
        #-------------------- estimate
        X_ind_all = np.stack(
            [self.kpt_ind * 3, self.kpt_ind * 3 + 1, self.kpt_ind * 3 + 2])
        # X_ind_all[:, :17]  #countour
        # X_ind_all[:, 17:27]  # eyebrows
        # X_ind_all[:, 27:36]  # nose
        # X_ind_all[:, 36:48]  # eyes
        # X_ind_all[:, 48:68]  # mouse
        X_ind_all = tf.concat([
            X_ind_all[:, :17], X_ind_all[:, 17:27], X_ind_all[:, 36:48],
            X_ind_all[:, 27:36], X_ind_all[:, 48:68]
        ],
                              axis=-1)

        valid_ind = tf.reshape(tf.transpose(X_ind_all), (-1))
        self.shapeMU = tf.gather(tf.cast(self.model['shapeMU'], tf.float32),
                                 valid_ind)
        self.shapePC = tf.gather(tf.cast(self.model['shapePC'], tf.float32),
                                 valid_ind)
        self.expPC = tf.gather(tf.cast(self.model['expPC'], tf.float32),
                               valid_ind)
        self.shapeMU = tf.tile(self.shapeMU[tf.newaxis, ...],
                               [self.batch_size, 1, 1])
        self.shapePC = tf.tile(self.shapePC[tf.newaxis, ...],
                               [self.batch_size, 1, 1])
        self.expPC = tf.tile(self.expPC[tf.newaxis, ...],
                             [self.batch_size, 1, 1])

        self.expEV = tf.cast(self.model['expEV'][:self.n_exp_para, :],
                             tf.float32)
        self.expEV = tf.tile(self.expEV[None, ...], [self.batch_size, 1, 1])

        self.expEV = tf.squeeze(self.expEV, axis=-1)
        self.shapeEV = tf.cast(self.model['shapeEV'][:self.n_shape_para, :],
                               tf.float32)
        self.shapeEV = tf.tile(self.shapeEV[None, ...], [self.batch_size, 1, 1])
        self.shapeEV = tf.squeeze(self.shapeEV, axis=-1)

        self.pseudo_T = np.load(config["pseudo_T_path"])
        self.pseudo_exp_eq_left = tf.cast(np.load(config["pseudo_exp_eq_left"]),
                                          tf.float32)

        self.pseudo_shp_eq_left = tf.cast(np.load(config["pseudo_shp_eq_left"]),
                                          tf.float32)
        self.pseudo_A = tf.cast(
            np.load(
                "/aidata/anders/objects/landmarks/3DDFA/pseudo_matrix/A.npy"),
            tf.float32)

    # ---------------- fit ----------------
    @tf.function
    def fit_points(self, x):
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
        x = x[..., ::-1]
        sp = tf.zeros(shape=(self.n_shape_para, 1), dtype=tf.float32)
        ep = tf.zeros(shape=(self.n_exp_para, 1), dtype=tf.float32)
        mask = tf.math.reduce_all(tf.math.is_finite(x), axis=[-1, -2])
        mask_T = tf.tile(mask[:, :, None, None], (1, 1, 3, 3))
        pseudo_T = tf.tile(self.pseudo_T[None, None, :, :],
                           (self.batch_size, n_objs, 1, 1))

        mask_A = tf.tile(mask[:, :, None, None], (1, 1, 136, 8))
        pseudo_A = tf.tile(self.pseudo_A[None, None, :, :],
                           (self.batch_size, n_objs, 1, 1))

        mask_exp_eq_left = tf.tile(mask[:, :, None, None], (1, 1, 29, 29))
        pseudo_exp_eq_left = tf.tile(self.pseudo_exp_eq_left[None, None, :, :],
                                     (self.batch_size, n_objs, 1, 1))

        mask_shp_eq_left = tf.tile(mask[:, :, None, None], (1, 1, 199, 199))
        pseudo_shp_eq_left = tf.tile(self.pseudo_shp_eq_left[None, None, :, :],
                                     (self.batch_size, n_objs, 1, 1))

        # pre-process n_objs for shape, express
        shapeMU = tf.tile(self.shapeMU[:, None, :, :],
                          [1, tf.shape(x)[1], 1, 1])
        shapePC = tf.tile(self.shapePC[:, None, :, :],
                          [1, tf.shape(x)[1], 1, 1])
        expPC = tf.tile(self.expPC[:, None, :, :], [1, tf.shape(x)[1], 1, 1])
        expEV = tf.tile(self.expEV[:, None, :], [1, tf.shape(x)[1], 1])
        shapeEV = tf.tile(self.shapeEV[:, None, :], [1, tf.shape(x)[1], 1])
        for _ in range(self.max_iter):
            X = tf.math.add(
                shapeMU,
                tf.linalg.matmul(shapePC, sp) + tf.linalg.matmul(expPC, ep))
            X = tf.reshape(X, (self.batch_size, n_objs, tf.shape(X)[2] // 3, 3))
            P = self.estimate_affine_matrix_3d22d(X, x, mask_A, pseudo_A,
                                                  mask_T, pseudo_T)
            s, R, t = self.P2sRt(P)
            #----- estimate shape
            # expression
            shape = tf.linalg.matmul(shapePC, sp)
            shape = tf.transpose(
                tf.reshape(
                    shape,
                    (self.batch_size, n_objs, tf.shape(shape)[2] // 3, 3)),
                (0, 1, 3, 2))
            ep = self.estimate_expression(x,
                                          mask_exp_eq_left,
                                          pseudo_exp_eq_left,
                                          shapeMU,
                                          expPC,
                                          expEV,
                                          shape,
                                          s,
                                          R,
                                          t[:, :, :2],
                                          lamb=20)
            # shape
            expression = tf.linalg.matmul(
                tf.tile(self.expPC[:, None, :, :], [1, n_objs, 1, 1]), ep)
            expression = tf.transpose(
                tf.reshape(expression, (self.batch_size, n_objs, k, 3)),
                (0, 1, 3, 2))
            sp = self.estimate_shape(x,
                                     mask_shp_eq_left,
                                     pseudo_shp_eq_left,
                                     shapeMU,
                                     shapePC,
                                     shapeEV,
                                     expression,
                                     s,
                                     R,
                                     t[:, :, :2],
                                     lamb=40)
        angles = self.matrix2angle(R)
        b_pitches = tf.where(angles[..., :1] > 0., np.pi - angles[..., :1],
                             -(np.pi + angles[..., :1]))
        angles = tf.concat([b_pitches, angles[..., 1:]], axis=-1)
        sp, ep, s, angles, t = self.valid_objs(sp, ep, s, angles, t, mask)
        #  might be different in the translation
        return sp, ep, s, angles, t

    def estimate_affine_matrix_3d22d(self, X, x, mask_A, pseudo_A, mask_T,
                                     pseudo_T):
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

        _, _, c, k = x.get_shape().as_list()
        n_objs = tf.shape(x)[1]
        mean = tf.math.reduce_mean(x, keepdims=True, axis=-1)
        x = x - mean
        average_norm = tf.math.reduce_mean(tf.math.sqrt(
            tf.math.reduce_sum(x**2, axis=2)),
                                           axis=-1)  # B, n_objs
        scale = (tf.math.sqrt(2.) / average_norm)[:, :, None, None]
        x = scale * x
        # 2d points

        ms = -mean * scale
        row1 = tf.concat([
            tf.squeeze(scale, axis=2),
            tf.zeros(shape=(self.batch_size, n_objs, 1)), ms[:, :, :1, 0]
        ],
                         axis=-1)
        row2 = tf.concat([
            tf.zeros(shape=(self.batch_size, n_objs, 1)),
            tf.squeeze(scale, axis=2), ms[:, :, 1:, 0]
        ],
                         axis=-1)

        row3 = tf.concat([
            tf.zeros(shape=(self.batch_size, n_objs, 1)),
            tf.zeros(shape=(self.batch_size, n_objs, 1)),
            tf.ones(shape=(self.batch_size, n_objs, 1))
        ],
                         axis=-1)

        T = tf.concat(
            [row1[:, :, None, :], row2[:, :, None, :], row3[:, :, None, :]],
            axis=2)
        # 3d points

        X_homo = tf.concat(
            [X, tf.ones(shape=(self.batch_size, n_objs, 1, k))], axis=2)

        mean = tf.math.reduce_mean(X, keepdims=True, axis=-1)
        X = X - mean

        average_norm = tf.math.reduce_mean(tf.math.sqrt(
            tf.math.reduce_sum(X**2, axis=2)),
                                           axis=-1)
        scale = (tf.math.sqrt(3.) / average_norm)[:, :, None, None]
        X = scale * X
        ms = -mean * scale
        row1 = tf.concat([
            tf.squeeze(scale, axis=2),
            tf.zeros(shape=(self.batch_size, n_objs, 1)),
            tf.zeros(shape=(self.batch_size, n_objs, 1)), ms[:, :, :1, 0]
        ],
                         axis=-1)

        row2 = tf.concat([
            tf.zeros(shape=(self.batch_size, n_objs, 1)),
            tf.squeeze(scale, axis=2),
            tf.zeros(shape=(self.batch_size, n_objs, 1)), ms[:, :, 1:2, 0]
        ],
                         axis=-1)

        row3 = tf.concat([
            tf.zeros(shape=(self.batch_size, n_objs, 1)),
            tf.zeros(shape=(self.batch_size, n_objs, 1)),
            tf.squeeze(scale, axis=2), ms[:, :, 2:, 0]
        ],
                         axis=-1)
        row4 = tf.concat([
            tf.zeros(shape=(self.batch_size, n_objs, 1)),
            tf.zeros(shape=(self.batch_size, n_objs, 1)),
            tf.zeros(shape=(self.batch_size, n_objs, 1)),
            tf.ones(shape=(self.batch_size, n_objs, 1))
        ],
                         axis=-1)
        U = tf.concat([
            row1[:, :, None, :], row2[:, :, None, :], row3[:, :, None, :],
            row4[:, :, None, :]
        ],
                      axis=2)

        # --- 2. equations
        # A = np.zeros((n * 2, 8), dtype=np.float32) # (136, 8)
        X_homo = tf.concat(
            [X, tf.ones(shape=(self.batch_size, n_objs, 1, k))], axis=2)

        l = tf.concat(
            [X_homo, tf.zeros(shape=(self.batch_size, n_objs, 4, 68))], axis=2)
        r = tf.concat(
            [tf.zeros(shape=(self.batch_size, n_objs, 4, 68)), X_homo], axis=2)
        A = tf.transpose(tf.concat([l, r], axis=-1), (0, 1, 3, 2))
        b = tf.reshape(x, (self.batch_size, n_objs, c * k, 1))
        # --- 3. solution
        # (B, N, 136, 8)
        A = tf.where(mask_A, A, pseudo_A)
        p_8 = tf.linalg.matmul(tf.linalg.pinv(A), b)
        row1 = p_8[:, :, None, :4, 0]
        row2 = p_8[:, :, None, 4:, 0]

        row3 = tf.concat([
            tf.zeros(shape=(self.batch_size, n_objs, 1, 1)),
            tf.zeros(shape=(self.batch_size, n_objs, 1, 1)),
            tf.zeros(shape=(self.batch_size, n_objs, 1, 1)),
            tf.ones(shape=(self.batch_size, n_objs, 1, 1))
        ],
                         axis=-1)
        P = tf.concat([row1, row2, row3], axis=2)
        # --- 4. denormalization
        T = tf.where(mask_T, T, pseudo_T)
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
        rz = tf.where(singular, 0., tf.math.atan2(R[:, :, 1, 0], R[:, :, 0, 0]))
        return tf.concat([rx[:, :, None], ry[:, :, None], rz[:, :, None]],
                         axis=-1)

    def estimate_expression(self,
                            x,
                            mask_eq_left,
                            pseudo_eq_left,
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
        _, _, c, k = x.get_shape().as_list()
        n_objs = tf.shape(x)[1]
        sigma = expEV
        P = tf.constant([[1, 0, 0], [0, 1, 0]], dtype=tf.float32)
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
        equation_left = tf.where(mask_eq_left, equation_left, pseudo_eq_left)
        exp_para = tf.linalg.matmul(tf.linalg.inv(equation_left),
                                    equation_right)
        return exp_para

    def estimate_shape(self,
                       x,
                       mask_shp_eq_left,
                       pseudo_shp_eq_left,
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
        _, _, c, k = x.get_shape().as_list()  # 68
        n_objs = tf.shape(x)[1]
        sigma = shapeEV
        P = tf.constant([[1., 0., 0.], [0., 1., 0.]], tf.float32)
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
        equation_right = tf.linalg.matmul(pc, x - b, transpose_a=(0, 1, 3, 2))
        equation_left = tf.where(mask_shp_eq_left, equation_left,
                                 pseudo_shp_eq_left)

        shape_para = tf.linalg.matmul(tf.linalg.inv(equation_left),
                                      equation_right)
        return shape_para

    def valid_objs(self, sp, ep, s, angles, t, mask):
        sp = tf.where(
            tf.tile(mask[..., tf.newaxis, tf.newaxis],
                    (1, 1, tf.shape(sp)[-2], tf.shape(sp)[-1])), sp, np.inf)
        ep = tf.where(
            tf.tile(mask[..., tf.newaxis, tf.newaxis],
                    (1, 1, tf.shape(ep)[-2], tf.shape(ep)[-1])), ep, np.inf)
        s = tf.where(mask, s, np.inf)
        angles = tf.where(
            tf.tile(mask[..., tf.newaxis], (1, 1, tf.shape(angles)[-1])),
            angles, np.inf)
        t = tf.where(tf.tile(mask[..., tf.newaxis], (1, 1, tf.shape(t)[-1])), t,
                     np.inf)
        return sp, ep, s, angles, t
