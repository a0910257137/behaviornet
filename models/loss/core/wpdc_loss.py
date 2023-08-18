import numpy as np
import tensorflow as tf
from .utils import weight_reduce_loss


class WPDCLoss:

    def __init__(self, n_R, n_shp, n_exp, eps=1e-6):
        self.eps = eps
        self.n_R = n_R
        self.n_shp = n_shp
        self.n_exp = n_exp

    # @tf.function
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
        loss, pred_params = self.wpdc(gt_params, pms, pred_Z_params, u_base,
                                      shp_base, exp_base)
        loss = weight * tf.reduce_sum(loss, axis=-1)
        loss = tf.math.reduce_mean(loss)
        loss = tf.where(tf.math.is_nan(loss), 0., loss)
        return loss, pred_params

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

        gt_R, gt_shp_exp = gt_params[:, :self.n_R], gt_params[:, self.n_R:]
        # gt_t = gt_params[:, -3:]
        # gt_Rt = tf.concat([gt_R, gt_t], axis=-1)
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
            pred_R, pred_shp_exp = pred_params[:, :self.
                                               n_R], pred_params[:, self.n_R:]
            # pred_t = pred_params[:, -3:]
            # pred_Rt = tf.concat([pred_R, pred_t], axis=-1)
            R_weights = get_Rt_weights(gt_vertices, pred_R, gt_R)

            shp_exp_weights = get_shp_exp_weights(shp_base, exp_base,
                                                  pred_shp_exp, gt_shp_exp)
            R_weights = tf.transpose(R_weights, (1, 0))
            # R_weights, t_weights = Rt_weights[:, :9], Rt_weights[:, -3:-1]
            weights = tf.concat([R_weights, shp_exp_weights], axis=-1)
            weights = (weights + self.eps) / tf.math.reduce_max(
                weights, keepdims=True, axis=-1)
            gt_Z_params = (gt_params - pms[:1, :]) / pms[1:, :]
            # loss = tf.math.square(pred_Z_params - gt_Z_params)
            loss = weights * tf.math.square(pred_Z_params - gt_Z_params)
        return loss, pred_params
