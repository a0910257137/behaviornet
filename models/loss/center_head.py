import tensorflow as tf
from .loss_base import LossBase
from .loss_functions import *
from pprint import pprint
from utils.io import load_BFM


class CenterHeadLoss(LossBase):

    def __init__(self, config):
        self.config = config

        self.loss_cfg = self.config.loss
        self.meta_joint = self.loss_cfg.meta_joint
        # self.is_wpdc = self.loss_cfg["wpdc"]

        self.head_cfg = self.config.head
        self.max_obj_num = self.config.max_obj_num
        self.keys = ["obj_heat_map", "param"]
        self.n_s, self.n_R = config['3dmm']["n_s"], config['3dmm']["n_R"]
        self.n_shp, self.n_exp = config['3dmm']["n_shp"], config['3dmm'][
            "n_exp"]
        self.head_model = load_BFM(config['3dmm']['model_path'])
        self.shapeMU = tf.cast(self.head_model['shapeMU'], tf.float32)
        self.shapePC = tf.cast(self.head_model['shapePC'][:, :self.n_shp],
                               tf.float32)
        self.expPC = tf.cast(self.head_model['expPC'][:, :self.n_exp],
                             tf.float32)

        kpt_ind = self.head_model['kpt_ind']
        kpt_ind = np.stack([kpt_ind * 3, kpt_ind * 3 + 1, kpt_ind * 3 + 2])
        self.kpt_ind = tf.concat([
            kpt_ind[:, :17], kpt_ind[:, 17:27], kpt_ind[:, 36:48],
            kpt_ind[:, 27:36], kpt_ind[:, 48:68]
        ],
                                 axis=-1)

    def build_loss(self, logits, targets, batch, training):

        def fn_loss(x):
            return x

        with tf.name_scope("losses_collections"):
            losses = {k: None for k in self.keys}
            gt_idxs = targets["b_coords"]
            losses['obj_heat_map'] = penalty_reduced_focal_loss(
                targets['obj_heat_map'], logits['obj_heat_map'])
            u_base, shp_base, exp_base, mean = self.resample(
                batch, self.max_obj_num, self.shapeMU, self.shapePC, self.expPC,
                targets['params'], self.kpt_ind)

            gt_shp_exp = targets['params'][..., self.n_R + self.n_s:]
            gt_vertices = u_base + tf.linalg.matmul(
                shp_base,
                gt_shp_exp[:, :, :self.n_shp, None]) + tf.linalg.matmul(
                    exp_base, gt_shp_exp[:, :, self.n_shp:, None])

            gt_vertices = tf.reshape(
                gt_vertices,
                [batch, self.max_obj_num,
                 tf.shape(gt_vertices)[2] // 3, 3])
            w_loss, v_loss, l_loss = None, None, None
            w_loss, pred_params = wpdc_loss(batch, self.max_obj_num, gt_idxs,
                                            gt_vertices, targets['params'],
                                            targets['Z_params'],
                                            targets["mean_std"],
                                            logits["obj_param_map"], self.n_s,
                                            self.n_R, shp_base, exp_base)
            if self.meta_joint:
                v_loss, pred_lnmks = vdc_loss(batch, self.max_obj_num, gt_idxs,
                                              gt_vertices, targets['params'],
                                              targets["mean_std"], pred_params,
                                              self.n_s, self.n_R, self.n_shp,
                                              u_base, shp_base, exp_base)

                pred_lnmks -= mean[..., :2]
                l_loss = lrr_loss(batch, self.max_obj_num, pred_lnmks,
                                  targets['b_lnmks'])
                w_loss = tf.math.reduce_mean(w_loss)
                v_loss = tf.math.reduce_mean(v_loss)
                l_loss = tf.math.reduce_mean(l_loss)
                # w_loss, w_test_loss = tf.math.reduce_mean(
                #     w_loss[:-1, ...]), w_loss[-1:, ...]

                # v_loss, v_test_loss = tf.math.reduce_mean(
                #     v_loss[:-1]), tf.math.reduce_mean(v_loss[-1:])

                # l_loss, l_test_loss = tf.math.reduce_mean(
                #     l_loss[:-1]), l_loss[-1:]

                # loss = tf.cond(
                #     w_test_loss > (w_test_loss / l_test_loss) * v_test_loss,
                #     lambda: fn_loss(w_loss), lambda: fn_loss(
                #         (w_loss / l_loss) * v_loss))
                # vanilla meta joint
                beta = 0.7
                loss = beta * w_loss + (1 - beta) * (w_loss / l_loss) * v_loss
            else:
                loss = tf.math.reduce_mean(w_loss)
            losses["param"] = loss
            losses["total"] = losses['obj_heat_map'] + losses["param"]
        return losses

    def resample(self, batch_size, max_obj_num, shapeMU, shapePC, expPC,
                 gt_params, kpt_ind):
        # resmpale different vertices for 68 landmarks and 132 random samples
        # NOTE: randomly sampling method
        index = tf.random.shuffle(tf.range(start=0, limit=53215,
                                           dtype=tf.int32))[:396]
        keypoints_resample = tf.stack([3 * index, 3 * index + 1, 3 * index + 2])
        keypoints_mix = tf.concat([kpt_ind, keypoints_resample], axis=-1)
        n_objs = tf.shape(gt_params)[1]
        keypoints_mix = tf.reshape(tf.transpose(keypoints_mix), [-1])
        u_base = tf.tile(
            tf.gather(shapeMU, keypoints_mix)[None, None, :, :],
            [batch_size, n_objs, 1, 1])
        u_vertices = tf.reshape(
            u_base, [batch_size, max_obj_num,
                     tf.shape(u_base)[2] // 3, 3])
        mean = tf.math.reduce_mean(u_vertices[..., :68, :],
                                   axis=-2,
                                   keepdims=True)
        shp_base = tf.tile(
            tf.gather(shapePC, keypoints_mix)[None, None, :, :],
            [batch_size, n_objs, 1, 1])
        exp_base = tf.tile(
            tf.gather(expPC, keypoints_mix)[None, None, :, :],
            (batch_size, n_objs, 1, 1))
        return u_base, shp_base, exp_base, mean
