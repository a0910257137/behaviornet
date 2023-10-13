from pprint import pprint
from utils.io import load_BFM
import numpy as np
import tensorflow as tf


class MorphabelModelNumpy:

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
        X_ind_all = np.tile(self.kpt_ind[np.newaxis, :], [3, 1]) * 3
        X_ind_all[1, :] += 1
        X_ind_all[2, :] += 2
        X_ind_all = np.concatenate([
            X_ind_all[:, :17], X_ind_all[:, 17:27], X_ind_all[:, 36:48],
            X_ind_all[:, 27:36], X_ind_all[:, 48:68]
        ],
                                   axis=-1)

        valid_ind = X_ind_all.flatten('F')
        self.shapeMU = self.model['shapeMU'][valid_ind, :].astype(np.float32)
        self.shapePC = self.model['shapePC'][valid_ind, :self.n_shp].astype(
            np.float32)
        self.expPC = self.model['expPC'][valid_ind, :self.n_exp].astype(
            np.float32)

    # ---------------- fit ----------------

    def fit_points(self, b_xs):
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
        b_xs = b_xs.numpy()
        b_xs = b_xs[..., ::-1]
        b_obj_params = []
        for xs in b_xs:
            obj_tmp = []
            for x in xs:
                mask = np.all(np.isfinite(x))
                if (~mask):
                    continue
                x = x.copy().T
                #-- init
                sp = np.zeros((self.n_shp, 1), dtype=np.float32)
                ep = np.zeros((self.n_exp, 1), dtype=np.float32)

                # pre-process n_objs for shape, express
                for _ in range(self.max_iter):
                    X = self.shapeMU + self.shapePC.dot(sp) + self.expPC.dot(ep)
                    X = np.reshape(X, [int(len(X) / 3), 3]).T
                    # ----- estimate pose-----
                    P = self.estimate_affine_matrix_3d22d(X.T, x.T)

                    s, R, t = self.P2sRt(P)
                    shape = self.shapePC.dot(sp)
                    shape = np.reshape(shape, [int(len(shape) / 3), 3]).T
                    #----- estimate shape
                    # expression
                    ep = self.estimate_expression(
                        x,
                        self.shapeMU,
                        self.expPC,
                        self.model['expEV'][:self.n_exp, :],
                        shape,
                        s,
                        R,
                        t[:2],
                        lamb=20)
                    # shape
                    expression = self.expPC.dot(ep)
                    expression = np.reshape(expression,
                                            [int(len(expression) / 3), 3]).T
                    sp = self.estimate_shape(
                        x,
                        self.shapeMU,
                        self.shapePC,
                        self.model['shapeEV'][:self.n_shp, :],
                        expression,
                        s,
                        R,
                        t[:2],
                        lamb=40)
                # sp, ep, s, R, t
                s = np.array([s])
                R = np.reshape(R, (9))
                t = t[:2]
                params = np.concatenate(
                    [s, R, t,
                     np.squeeze(sp, axis=-1),
                     np.squeeze(ep, axis=-1)],
                    axis=0)
                obj_tmp.append(params)
            obj_tmp = np.asarray(obj_tmp)
            obj_tmp = self.complement_params(obj_tmp)
            b_obj_params.append(obj_tmp)
        return np.asarray(b_obj_params, dtype=np.float32)

    def estimate_affine_matrix_3d22d(self, X, x):
        ''' Using Golden Standard Algorithm for estimating an affine camera
            matrix P from world to image correspondences.
            See Alg.7.2. in MVGCV 
            Code Ref: https://github.com/patrikhuber/eos/blob/master/include/eos/fitting/affine_camera_estimation.hpp
            x_homo = X_homo.dot(P_Affine)
        Args:
            X: [n, 3]. corresponding 3d points(fixed)
            x: [n, 2]. n>=4. 2d points(moving). x = PX
        Returns:
            P_Affine: [3, 4]. Affine camera matrix
        '''
        X = X.T
        x = x.T
        assert (x.shape[1] == X.shape[1])
        n = x.shape[1]
        assert (n >= 4)

        #--- 1. normalization
        # 2d points
        mean = np.mean(x, 1)  # (2,)
        x = x - np.tile(mean[:, np.newaxis], [1, n])
        average_norm = np.mean(np.sqrt(np.sum(x**2, 0)))

        scale = np.sqrt(2) / average_norm
        x = scale * x

        T = np.zeros((3, 3), dtype=np.float32)

        T[0, 0] = T[1, 1] = scale
        T[:2, 2] = -mean * scale
        T[2, 2] = 1

        # 3d points
        X_homo = np.vstack((X, np.ones((1, n))))
        mean = np.mean(X, 1)  # (3,)
        X = X - np.tile(mean[:, np.newaxis], [1, n])

        m = X_homo[:3, :] - X
        average_norm = np.mean(np.sqrt(np.sum(X**2, 0)))
        scale = np.sqrt(3) / average_norm
        X = scale * X

        U = np.zeros((4, 4), dtype=np.float32)
        U[0, 0] = U[1, 1] = U[2, 2] = scale
        U[:3, 3] = -mean * scale
        U[3, 3] = 1

        # --- 2. equations
        A = np.zeros((n * 2, 8), dtype=np.float32)
        X_homo = np.vstack((X, np.ones((1, n)))).T
        A[:n, :4] = X_homo
        A[n:, 4:] = X_homo
        b = np.reshape(x, [-1, 1])
        # --- 3. solution

        p_8 = np.linalg.pinv(A).dot(b)
        P = np.zeros((3, 4), dtype=np.float32)
        P[0, :] = p_8[:4, 0]
        P[1, :] = p_8[4:, 0]
        P[-1, -1] = 1
        # --- 4. denormalization
        P_Affine = np.linalg.inv(T).dot(P.dot(U))
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
        t = P[:, 3]
        R1 = P[0:1, :3]
        R2 = P[1:2, :3]
        s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
        r1 = R1 / np.linalg.norm(R1)
        r2 = R2 / np.linalg.norm(R2)
        r3 = np.cross(r1, r2)
        R = np.concatenate((r1, r2, r3), 0)
        return s, R, t

    def estimate_expression(self,
                            x,
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
        x = x.copy()
        assert (shapeMU.shape[0] == expPC.shape[0])
        assert (shapeMU.shape[0] == x.shape[1] * 3)

        dof = expPC.shape[1]
        n = x.shape[1]

        sigma = expEV
        t2d = np.array(t2d)
        P = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

        A = s * P.dot(R)

        # --- calc pc
        pc_3d = np.resize(expPC.T, [dof, n, 3])
        pc_3d = np.reshape(pc_3d, [dof * n, 3])
        pc_2d = pc_3d.dot(A.T)

        pc = np.reshape(pc_2d, [dof, -1]).T  # 2n x 29

        # --- calc b
        # shapeMU
        mu_3d = np.resize(shapeMU, [n, 3]).T  # 3 x n
        # expression
        shape_3d = shape
        #
        b = A.dot(mu_3d + shape_3d) + np.tile(t2d[:, np.newaxis],
                                              [1, n])  # 2 x n

        b = np.reshape(b.T, [-1, 1])  # 2n x 1
        # --- solve
        # np.save("sigma.npy", lamb * np.diagflat(1 / sigma**2))
        equation_left = np.dot(pc.T, pc) + lamb * np.diagflat(1 / sigma**2)

        x = np.reshape(x.T, [-1, 1])
        equation_right = np.dot(pc.T, x - b)

        exp_para = np.dot(np.linalg.inv(equation_left), equation_right)

        return exp_para

    def estimate_shape(self,
                       x,
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
        x = x.copy()
        assert (shapeMU.shape[0] == shapePC.shape[0])
        assert (shapeMU.shape[0] == x.shape[1] * 3)

        dof = shapePC.shape[1]
        n = x.shape[1]
        sigma = shapeEV
        t2d = np.array(t2d)
        P = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        A = s * P.dot(R)

        # --- calc pc
        pc_3d = np.resize(shapePC.T, [dof, n, 3])  # 199 x n x 3
        pc_3d = np.reshape(pc_3d, [dof * n, 3])
        pc_2d = pc_3d.dot(A.T.copy())  # 199 x n x 2

        pc = np.reshape(pc_2d, [dof, -1]).T  # 2n x 199
        # --- calc b
        # shapeMU
        mu_3d = np.resize(shapeMU, [n, 3]).T  # 3 x n
        # expression
        exp_3d = expression
        #
        b = A.dot(mu_3d + exp_3d) + np.tile(t2d[:, np.newaxis], [1, n])  # 2 x n

        b = np.reshape(b.T, [-1, 1])  # 2n x 1

        # --- solve
        equation_left = np.dot(pc.T, pc) + lamb * np.diagflat(1 / sigma**2)

        x = np.reshape(x.T, [-1, 1])
        equation_right = np.dot(pc.T, x - b)
        shape_para = np.dot(np.linalg.inv(equation_left), equation_right)
        return shape_para

    def complement_params(self, params):
        n, d = params.shape
        params = np.asarray([x for x in params if x.size != 0])
        complement = self.max_obj_num - n
        complement = np.empty([complement, d])
        complement.fill(np.inf)
        complement = complement.astype(np.float32)
        if n == 0:
            annos = complement
        else:
            annos = np.concatenate([params, complement])

        return annos