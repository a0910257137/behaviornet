import numpy as np
import tensorflow as tf


@tf.function
def lnmk_loss(b_gt_lnmks, b_gt_params, b_pred_Z_params, u_base, shp_base,
              exp_base, pms, weights):
    pred_params = b_pred_Z_params * pms[1][None, :] + pms[0][None, :]
    b_gt_R = b_gt_params[:, :9]
    b_pred_R = pred_params[:, :9]
    pred_shp = pred_params[:, 9:49]
    pred_exp = pred_params[:, 49:]
    b_gt_R = tf.reshape(b_gt_R, [-1, 3, 3])
    b_pred_R = tf.reshape(b_pred_R, [-1, 3, 3])
    pred_vertices = u_base + tf.linalg.matmul(
        shp_base, pred_shp[..., None]) + tf.linalg.matmul(
            exp_base, pred_exp[..., None])
    pred_vertices = tf.reshape(pred_vertices,
                               (-1, tf.shape(pred_vertices)[1] // 3, 3))
    b_pred_lnmks = pred_vertices[:, :68, :]
    s = 0.001
    b_gt_lnmks = s * tf.linalg.matmul(b_gt_lnmks, b_gt_R, transpose_b=(0, 2, 1))
    b_pred_lnmks = s * tf.linalg.matmul(
        b_pred_lnmks, b_pred_R, transpose_b=(0, 2, 1))
    b_diffs = (b_pred_lnmks[..., :2] - b_gt_lnmks[..., :2])
    loss = tf.math.sqrt(
        tf.math.reduce_sum(tf.math.square(b_diffs), axis=-1) + 1e-9)
    loss = weights * tf.math.reduce_sum(loss, axis=-1)
    loss = tf.math.reduce_mean(loss)
    loss = 0.5 * tf.where(tf.math.is_nan(loss), 0., loss)
    return loss


class WPDCLoss:

    def __init__(self, cfg, eps=1e-6):
        self.eps = eps
        self.n_s = cfg["n_s"]
        self.n_R = cfg["n_R"]
        self.n_t3d = cfg["n_t3d"]
        self.n_shp = cfg["n_shp"]
        self.n_exp = cfg["n_exp"]

    @tf.function
    def __call__(self,
                 gt_params,
                 pms,
                 pred_Z_params,
                 u_base,
                 shp_base,
                 exp_base,
                 weight=None,
                 avg_factor=None,
                 **kwargs):
        loss, gt_vertices = self.wpdc(gt_params, pms, pred_Z_params, u_base,
                                      shp_base, exp_base)
        loss = weight * tf.reduce_sum(loss, axis=-1)
        loss = tf.math.reduce_mean(loss)
        loss = 1.0 * tf.where(tf.math.is_nan(loss), 0., loss)

        return loss, gt_vertices

    def wpdc(self, gt_params, pms, pred_Z_params, u_base, shp_base, exp_base):

        def get_Rt_weights(gt_vertices, pred_Rt, gt_Rt):
            pose_diffs = tf.math.abs(pred_Rt - gt_Rt)
            pv_nomrs = tf.norm(gt_vertices, ord=2, axis=-2)
            # N_vertices = tf.cast(tf.shape(gt_vertices)[-2], tf.float32)
            # offset_norm = tf.math.sqrt(N_vertices)
            weights = []
            for ind in range(9):
                if ind in [0, 1, 2]:
                    w = pose_diffs[:, ind] * pv_nomrs[:, 0]
                elif ind in [3, 4, 5]:
                    w = pose_diffs[:, ind] * pv_nomrs[:, 1]
                elif ind in [6, 7, 8]:
                    w = pose_diffs[:, ind] * pv_nomrs[:, 2]
                # else:
                #     w = pose_diffs[:, ind] * offset_norm
                weights.append(w)
            return tf.stack(weights)

        def get_shp_exp_weights(shp_base, exp_base, pred_shp_exp, gt_shp_exp):
            w_norm = tf.norm(tf.concat((shp_base, exp_base), axis=-1),
                             ord=2,
                             axis=-2)
            return w_norm * tf.math.abs(pred_shp_exp - gt_shp_exp)

        gt_Rt = gt_params[:, self.n_s:self.n_s + self.n_R]
        gt_shp_exp = gt_params[:, self.n_s + self.n_R:]
        shp_base = tf.tile(shp_base[None, :, :],
                           [tf.shape(gt_shp_exp)[0], 1, 1])
        exp_base = tf.tile(exp_base[None, :, :],
                           [tf.shape(gt_shp_exp)[0], 1, 1])
        gt_vertices = u_base[None, :, :] + tf.linalg.matmul(
            shp_base, gt_shp_exp[:, :self.n_shp, None]) + tf.linalg.matmul(
                exp_base, gt_shp_exp[:, self.n_shp:, None])
        gt_vertices = tf.reshape(gt_vertices,
                                 [-1, tf.shape(gt_vertices)[-2] // 3, 3])
        with tf.name_scope('wpdc_loss'):
            # pred_params are reconstruct params
            pred_params = pred_Z_params * pms[1][None, :] + pms[0][None, :]
            pred_Rt = pred_params[:, self.n_s:self.n_s + self.n_R]
            pred_shp_exp = pred_params[:, self.n_s + self.n_R:]
            Rt_weights = get_Rt_weights(gt_vertices, pred_Rt, gt_Rt)
            shp_exp_weights = get_shp_exp_weights(shp_base, exp_base,
                                                  pred_shp_exp, gt_shp_exp)
            Rt_weights = tf.transpose(Rt_weights, (1, 0))
            weights = tf.concat([Rt_weights, shp_exp_weights], axis=-1)
            weights = (weights + self.eps) / tf.math.reduce_max(
                weights, keepdims=True, axis=-1)
            gt_Z_params = (gt_params - pms[:1, :]) / pms[1:, :]
            loss = weights * tf.math.square(pred_Z_params[:, self.n_s:] -
                                            gt_Z_params[:, self.n_s:])

        b_lnmks = gt_vertices[:, :68, :]
        return loss, b_lnmks