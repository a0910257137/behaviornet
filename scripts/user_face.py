import numpy as np
import cv2
import json
import os
from pprint import pprint
from math import cos, sin
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.morphable_model import MorphabelModel

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def angle2matrix(x, y, z):
    # x
    Rx = np.array([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]])

    # y
    Ry = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
    # z
    Rz = np.array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]])

    R = Rz.dot(Ry.dot(Rx))
    return R.astype(np.float32)


def get_Rt(params):
    angles, t = params[:3], params[3:]
    R = angle2matrix(*angles)
    return np.concatenate([R, t], axis=-1)


def camera_func(x3d, K, Rt):

    x3d = np.concatenate([x3d, np.ones(shape=(x3d.shape[0], 1))], axis=-1)
    preds = x3d.dot(Rt.T)
    preds = preds.dot(K.T)
    preds = preds / preds[:, -1:]
    return preds[:, :2]


root_dir = "/aidata/anders/3D-head/calibration"


class LM:

    def __init__(self):
        self.iteration = 0  # iteration counter
        self.func_calls = 0  # running count of function evaluations
        self.MaxIter = 5000
        self.trans_constraint = 100

    def __call__(self, x2d, x3d, K, angles, params, img_wh):
        x2d = x2d / img_wh
        x2d = x2d.reshape([-1])
        # define eps (not available in python)
        eps = 2**(-52)
        # params = np.concatenate([angles, params], axis=0)
        # number of parameters
        Npar = len(params)
        # number of data points
        Npnt = len(x2d)
        # previous set of parameters
        p_old = np.zeros((Npar, 1))
        # previous model, y_old = y_hat(t,p_old)
        y_old = np.zeros((Npnt, 1))
        # a really big initial Chi-sq value
        X2 = 1e-3 / eps
        # a really big initial Chi-sq value
        X2_old = 1e-3 / eps
        # Jacobian matrix
        J = np.zeros((Npnt, Npar))
        # statistical degrees of freedom
        DoF = np.array([[Npnt - Npar + 1]])
        weight = 1 / (x2d.T @ x2d)

        # fractional increment of 'p' for numerical derivatives
        dp = [-0.0001]
        # # # upper bounds for parameter values
        epsilon_1 = 1e-8  # convergence tolerance for gradient
        epsilon_2 = 1e-8  # convergence tolerance for parameters
        epsilon_4 = 1e-1  # determines acceptance of a L-M step
        lambda_0 = 1e-3  # initial value of damping paramter, lambda
        lambda_UP_fac = 11  # factor for increasing lambda
        lambda_DN_fac = 9  # factor for decreasing lambda
        Update_Type = 1  # 1: Levenberg-Marquardt lambda update, 2: Quadratic update, 3: Nielsen's lambda update equations
        if len(dp) == 1:
            dp = dp * np.ones((Npar, 1))

        idx = np.arange(len(dp))  # indices of the parameters to be fit
        stop = 0  # termination flag

        # identical weights vector
        if np.var(weight) == 0:
            weight = abs(weight) * np.ones((Npnt, 1))
            print('Using uniform weights for error analysis')
        else:
            weight = abs(weight)

        # initialize Jacobian with finite difference calculation
        JtWJ, JtWdy, X2, preds, J = self.lm_matx(x3d, K, img_wh, p_old, y_old,
                                                 1, J, angles, params, x2d,
                                                 weight, dp)

        if np.abs(JtWdy).max() < epsilon_1:
            print('*** Your Initial Guess is Extremely Close to Optimal ***')
        lambda_0 = np.atleast_2d([lambda_0])

        # Marquardt: init'l lambda
        if Update_Type == 1:
            lambda_ = lambda_0
        # Quadratic and Nielsen
        else:
            lambda_ = lambda_0 * max(np.diag(JtWJ))
            nu = 2

        # previous value of X2
        X2_old = X2
        # initialize convergence history
        cvg_hst = np.ones((self.MaxIter, Npar + 2))
        # -------- Start Main Loop ----------- #
        while not stop and self.iteration <= self.MaxIter:
            self.iteration = self.iteration + 1
            # incremental change in parameters
            # Marquardt
            if Update_Type == 1:
                h = np.linalg.solve((JtWJ + lambda_ * np.diag(np.diag(JtWJ))),
                                    JtWdy)
            # Quadratic and Nielsen
            else:
                h = np.linalg.solve((JtWJ + lambda_ * np.eye(Npar)), JtWdy)
            # update the [idx] elements

            params_try = params + h[idx]
            # apply constraints

            params_try = np.minimum(
                np.maximum(-self.trans_constraint, params_try),
                self.trans_constraint)

            # residual error using p_try
            preds = self.lm_func(x3d, K, img_wh, angles,
                                 params_try).reshape([-1])
            delta_y = np.array([x2d - preds]).T

            # floating point error; break
            if not all(np.isfinite(delta_y)):
                stop = 1
                break

            self.func_calls = self.func_calls + 1
            # Chi-squared error criteria
            X2_try = delta_y.T @ (delta_y * weight)
            # % Quadratic
            if Update_Type == 2:
                # One step of quadratic line update in the h direction for minimum X2
                alpha = np.divide(JtWdy.T @ h,
                                  ((X2_try - X2) / 2 + 2 * JtWdy.T @ h))
                h = alpha * h

                # % update only [idx] elements
                params_try = params + h[idx]
                # % apply constraints
                params_try = np.minimum(
                    np.maximum(-self.trans_constraint, params_try),
                    self.trans_constraint)
                # % residual error using p_try
                delta_y = x2d - self.lm_func(x3d, K, img_wh, angles,
                                             params_try).reshape([-1])
                delta_y = delta_y[:, None]

                self.func_calls = self.func_calls + 1
                # % Chi-squared error criteria
                X2_try = delta_y.T @ (delta_y * weight)
            rho = np.matmul(h.T @ (lambda_ * h + JtWdy),
                            np.linalg.inv(X2 - X2_try + 1e-6))

            # it IS significantly better
            if (rho > epsilon_4):
                dX2 = X2 - X2_old
                X2_old = X2
                p_old = params
                y_old = preds
                # % accept p_try
                params = params_try
                JtWJ, JtWdy, X2, preds, J = self.lm_matx(
                    x3d, K, img_wh, p_old, y_old, dX2, J, angles, params, x2d,
                    weight, dp)

                # % decrease lambda ==> Gauss-Newton method
                # % Levenberg
                if Update_Type == 1:
                    lambda_ = max(lambda_ / lambda_DN_fac, 1.e-7)
                # % Quadratic
                elif Update_Type == 2:
                    lambda_ = max(lambda_ / (1 + alpha), 1.e-7)
                # % Nielsen
                else:
                    lambda_ = lambda_ * max(1 / 3, 1 - (2 * rho - 1)**3)
                    nu = 2

            # it IS NOT better
            else:
                # % do not accept p_try
                X2 = X2_old
                if not np.remainder(self.iteration, 2 * Npar):
                    JtWJ, JtWdy, dX2, preds, J = self.lm_matx(
                        x3d, K, img_wh, p_old, y_old, -1, J, angles, params,
                        x2d, weight, dp)

                # % increase lambda  ==> gradient descent method
                # % Levenberg
                if Update_Type == 1:
                    lambda_ = min(lambda_ * lambda_UP_fac, 1.e7)
                # % Quadratic
                elif Update_Type == 2:
                    lambda_ = lambda_ + abs((X2_try - X2) / 2 / alpha)
                # % Nielsen
                else:
                    lambda_ = lambda_ * nu
                    nu = 2 * nu
            # update convergence history ... save _reduced_ Chi-square
            cvg_hst[self.iteration - 1, 0] = self.func_calls
            cvg_hst[self.iteration - 1, 1] = X2 / DoF

            for i in range(Npar):
                cvg_hst[self.iteration - 1, i + 2] = params.T[0][i]

            if (max(abs(JtWdy)) < epsilon_1 and self.iteration > 2):
                print('**** Convergence in r.h.s. ("JtWdy")  ****')
                stop = 1

            if (max(abs(h) / (abs(params) + 1e-12)) < epsilon_2
                    and self.iteration > 2):
                print('**** Convergence in Parameters ****')
                stop = 1

            if (self.iteration == self.MaxIter):
                print(
                    '!! Maximum Number of Iterations Reached Without Convergence !!'
                )
                stop = 1

            # --- End of Main Loop --- #
            # --- convergence achieved, find covariance and confidence intervals

        #  ---- Error Analysis ----
        #  recompute equal weights for paramter error analysis
        if np.var(weight) == 0:
            weight = DoF / (delta_y.T @ delta_y) * np.ones((Npnt, 1))

        # % reduced Chi-square
        redX2 = X2 / DoF
        # params[2] = params[2] + 3
        JtWJ, JtWdy, X2, preds, J = self.lm_matx(x3d, K, img_wh, p_old, y_old,
                                                 -1, J, angles, params, x2d,
                                                 weight, dp)
        # standard error of parameters
        covar_p = np.linalg.inv(JtWJ)
        sigma_p = np.sqrt(np.diag(covar_p))
        error_p = sigma_p / params
        # standard error of the fit
        sigma_y = np.zeros((Npnt, 1))
        for i in range(Npnt):
            sigma_y[i, 0] = J[i, :] @ covar_p @ J[i, :].T

        sigma_y = np.sqrt(sigma_y)

        # parameter correlation matrix
        corr_p = covar_p / [sigma_p @ sigma_p.T]

        # coefficient of multiple determination
        R_sq = np.correlate(x2d, preds)
        R_sq = 0
        # convergence history
        cvg_hst = cvg_hst[:self.iteration, :]
        iters = list(range(cvg_hst.shape[0]))
        # from matplotlib import pyplot as plt
        # for i in range(cvg_hst.shape[1] - 1):
        #     plt.plot(iters, cvg_hst[:, i + 1], label="list_{}".format(i))
        # plt.legend()
        # plt.savefig("foo.jpg")
        return preds, params, redX2, sigma_p, sigma_y, corr_p, R_sq, cvg_hst

    def lm_matx(self, x3d, K, img_wh, p_old, y_old, dX2, J, angles, params, x2d,
                weight, dp):
        """
        Evaluate the linearized fitting matrix, JtWJ, and vector JtWdy, and
        calculate the Chi-squared error function, Chi_sq used by Levenberg-Marquardt
        algorithm (lm).

        Parameters
        ----------
        x3d    :     independent variables used as arg to lm_func (m x 1)
        p_old  :     previous parameter values (n x 1)
        y_old  :     previous model ... y_old = y_hat(t,p_old) (m x 1)
        dX2    :     previous change in Chi-squared criteria (1 x 1)
        J      :     Jacobian of model, y_hat, with respect to parameters, p (m x n)
        params :     current parameter values (n x 1)
        x2d    :     data to be fit by func(t,p,c) (m x 1)
        weight :     the weighting vector for least squares fit inverse of
                    the squared standard measurement errors
        dp     :     fractional increment of 'p' for numerical derivatives
                    - dp(j)>0 central differences calculated
                    - dp(j)<0 one sided differences calculated
                    - dp(j)=0 sets corresponding partials to zero; i.e. holds p(j) fixed

        Returns
        -------
        JtWJ   :     linearized Hessian matrix (inverse of covariance matrix) (n x n)
        JtWdy  :     linearized fitting vector (n x m)
        Chi_sq :     Chi-squared criteria: weighted sum of the squared residuals WSSR
        preds  :     model evaluated with parameters 'p' (m x 1)
        J :          Jacobian of model, y_hat, with respect to parameters, p (m x n)

        """
        # number of parameters
        Npar = len(params)
        # evaluate model using parameters 'p'
        preds = self.lm_func(x3d, K, img_wh, angles, params).reshape([-1])
        self.func_calls = self.func_calls + 1

        if not np.remainder(self.iteration, 2 * Npar) or dX2 > 0:
            # finite difference
            J = self.lm_FD_J(x3d, K, img_wh, angles, params, preds, dp)
        else:
            # rank-1 update
            J = self.lm_Broyden_J(p_old, y_old, J, angles, params, preds)
        # residual error between model and data

        delta_y = np.array([x2d - preds]).T
        # Chi-squared error criteria
        Chi_sq = delta_y.T @ (delta_y * weight)
        JtWJ = J.T @ (J * (weight * np.ones((1, Npar))))
        JtWdy = J.T @ (weight * delta_y)
        return JtWJ, JtWdy, Chi_sq, preds, J

    def lm_func(self, x3d, K, img_wh, angles, params):
        """
        Define model function used for nonlinear least squares curve-fitting.

        Parameters
        ----------
        params     : parameter values , n = 4 in these examples             (n x 1)

        Returns
        -------
        preds : curve-fit fctn evaluated at points t and with parameters p (m x 1)

        """

        params = np.concatenate([angles, params], axis=0)
        Rt = get_Rt(params)
        preds = camera_func(x3d, K, Rt)
        preds = preds / img_wh
        return preds

    def lm_FD_J(self, x3d, K, img_wh, angles, params, preds, dp):
        """

        Computes partial derivates (Jacobian) dy/dp via finite differences.

        Parameters
        ----------

        trans_params  :     current translation parameter values (n x 1), n = 3 for translating x, y, z
        preds         :     use camera model func(x3d, K, img_wh, p,c) initialised by user before each call to lm_FD_J (m x 1)
        dp            :     fractional increment of p for numerical derivatives
                            - dp(j)>0 central differences calculated
                            - dp(j)<0 one sided differences calculated
                            - dp(j)=0 sets corresponding partials to zero; i.e. holds p(j) fixed

        Returns
        -------
        J :      Jacobian Matrix (n x m)

        """

        # number of data points
        m = len(preds)
        # number of parameters
        n = len(params)

        # initialize Jacobian to Zero
        ps = params
        J = np.zeros((m, n))
        del_ = np.zeros((n, 1))
        # START --- loop over all parameters
        for j in range(n):
            # parameter perturbation

            del_[j, 0] = dp[j, 0] * (1 + abs(params[j, 0]))
            # perturb parameter p(j)
            params[j, 0] = ps[j, 0] + del_[j, 0]

            if del_[j, 0] != 0:
                y1 = self.lm_func(x3d, K, img_wh, angles, params).reshape([-1])
                self.func_calls = self.func_calls + 1

                if dp[j, 0] < 0:
                    # backwards difference
                    J[:, j] = (y1 - preds) / del_[j, 0]
                else:
                    # central difference, additional func call
                    params[j, 0] = ps[j, 0] - del_[j]
                    J[:, j] = (y1 - self.lm_func(x3d, K, img_wh, angles,
                                                 params)) / (2 * del_[j, 0])
                    self.func_calls = self.func_calls + 1
            # restore p(j)
            params[j, 0] = ps[j, 0]
        return J

    def lm_Broyden_J(self, p_old, y_old, J, angles, p, y):
        """
        Carry out a rank-1 update to the Jacobian matrix using Broyden's equation.

        Parameters
        ----------
        p_old :     previous set of parameters (n x 1)
        y_old :     model evaluation at previous set of parameters, y_hat(t,p_old) (m x 1)
        J     :     current version of the Jacobian matrix (m x n)
        p     :     current set of parameters (n x 1)
        y     :     model evaluation at current  set of parameters, y_hat(t,p) (m x 1)

        Returns
        -------
        J     :     rank-1 update to Jacobian Matrix J(i,j)=dy(i)/dp(j) (m x n)

        """
        h = p - p_old
        a = (np.array([y - y_old]).T - J @ h) @ h.T
        b = h.T @ h
        # Broyden rank-1 update eq'n
        J = J + a / b
        return J


def main(annos_path, img_root):
    print('-' * 100)
    print('initialize bfm model success')
    lm = LM()
    bfm = MorphabelModel('/aidata/anders/3D-head/3DDFA/BFM/BFM.mat')
    X_ind = bfm.kpt_ind
    X_ind_all = np.stack([X_ind * 3, X_ind * 3 + 1, X_ind * 3 + 2])
    X_ind_all = np.concatenate([
        X_ind_all[:, :17], X_ind_all[:, 17:27], X_ind_all[:, 36:48],
        X_ind_all[:, 27:36], X_ind_all[:, 48:68]
    ],
                               axis=-1)
    valid_ind = np.reshape(np.transpose(X_ind_all), (-1))
    shapeMU = bfm.model['shapeMU'][valid_ind, :]
    shapePC = bfm.model['shapePC'][valid_ind, :]
    expPC = bfm.model['expPC'][valid_ind, :]
    annos = load_json(annos_path)
    print('-' * 100)
    print("Start calibrating and calculating depth")
    K = np.fromfile(
        "/aidata/anders/3D-head/user_depth/calibrate_images/params/bins/cmat_f64.bin"
    ).reshape([3, -1])
    for frame in annos["frame_list"]:
        name = frame["name"]
        img_path = os.path.join(img_root, name)
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        img_wh = np.array([w, h])[None, :]
        params = np.array([5, 5, 50])[:, None]
        for lb in frame["labels"]:
            box2d = lb["box2d"]
            keypoints = lb["keypoints"]
            kps = np.asarray([keypoints[key] for key in keypoints.keys()])
            x2d = kps[:, ::-1]  # convert to x, y format
            fitted_sp, fitted_ep, fitted_s, fitted_angles, fitted_t = bfm.fit(
                kps[:, ::-1], X_ind, idxs=None, max_iter=5)
            landmarks = shapeMU + shapePC.dot(fitted_sp) + expPC.dot(fitted_ep)
            landmarks = np.reshape(landmarks, (landmarks.shape[0] // 3, 3))
            x3d = 1e-4 * landmarks  # convert to centimeter
            angles = np.asarray(fitted_angles)[:, None]
            preds, p_fit, Chi_sq, sigma_p, sigma_y, corr, R_sq, cvg_hst = lm(
                x2d, x3d, K, angles, params, img_wh)
            preds = preds.reshape([68, 2]) * img_wh
            kps = preds.astype(np.int32)
            for l in range(kps.shape[0]):
                if 0 <= l < 17:
                    color = [205, 133, 63]
                elif 17 <= l < 27:
                    # eyebrows
                    color = [205, 186, 150]
                elif 27 <= l < 39:
                    # eyes
                    color = [238, 130, 98]
                elif 39 <= l < 48:
                    # nose
                    color = [205, 96, 144]
                elif 48 <= l < 68:
                    color = [0, 191, 255]
                if l in [0, 8, 16, 27, 30, 33, 36]:
                    color = [0, 0, 255]
                cv2.circle(img, (kps[l][0], kps[l][1]), 10, color, -1)
                cv2.circle(img, (kps[l][0], kps[l][1]), 10, (255, 255, 255), -1)
                line_width = 5
                if l not in [16, 21, 26, 32, 38, 42, 47, 59, 67]:
                    start_point = (kps[l][0], kps[l][1])
                    end_point = (kps[l + 1][0], kps[l + 1][1])
                    cv2.line(img, start_point, end_point, (0, 0, 0), line_width)
                elif l == 32:
                    start_point = (kps[l][0], kps[l][1])
                    end_point = (kps[27][0], kps[27][1])
                    cv2.line(img, start_point, end_point, (0, 0, 0), line_width)
                elif l == 38:
                    start_point = (kps[l][0], kps[l][1])
                    end_point = (kps[33][0], kps[33][1])
                    cv2.line(img, start_point, end_point, (0, 0, 0), line_width)
                elif l == 59:
                    start_point = (kps[l][0], kps[l][1])
                    end_point = (kps[48][0], kps[48][1])
                    cv2.line(img, start_point, end_point, (0, 0, 0), line_width)
                elif l == 67:
                    start_point = (kps[l][0], kps[l][1])
                    end_point = (kps[60][0], kps[60][1])
                    cv2.line(img, start_point, end_point, (0, 0, 0), line_width)
            tl, br = (int(box2d['x1']), int(box2d['y1'])), (int(box2d['x2']),
                                                            int(box2d['y2']))
            img = cv2.rectangle(img, tl, br, (0, 255, 255), 2)
            # for kp in preds:
            #     kp = kp.astype(np.int32)
            #     img = cv2.circle(img, kp, 10, (0, 255, 0), -1)
            # cv2.imwrite("output.jpg", img)
            print('\nLM fitting results:')
            for i in range(len(p_fit)):
                print('----------------------------- ')
                print('parameter      = p%i' % (i + 1))
                print('fitted value   = %0.4f' % p_fit[i, 0])
            # print('standard error = %0.2f %%' % error_p[i, 0])
    # lm(x2d, x3d, K, angles, params, img_wh)
    return p_fit, Chi_sq, sigma_p, sigma_y, corr, R_sq, cvg_hst


def parse_config():
    parser = argparse.ArgumentParser('Argparser for calibrating 3D model')
    parser.add_argument('--anno_path')
    parser.add_argument('--img_root')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_config()
    p_fit, Chi_sq, sigma_p, sigma_y, corr, R_sq, cvg_hst = main(
        args.anno_path, args.img_root)
