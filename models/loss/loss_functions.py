from sklearn.metrics import log_loss
import tensorflow as tf
import numpy as np
from tensorflow._api.v2 import math
from tensorflow.python import tf2


def rel_cross_entropy(batch_size, max_obj_num, entropy_func, rel_cates, logits,
                      b_idxs, tars):
    b_idxs = tf.reshape(b_idxs, [batch_size, max_obj_num, 2])
    is_finite = tf.math.is_finite(b_idxs)
    valid_pos = tf.where(tf.math.reduce_any(is_finite, axis=-1), 1., 0.)
    b_idxs = tf.where(is_finite, b_idxs, tf.zeros_like(b_idxs))
    batch_idx = tf.tile(
        tf.reshape(tf.range(batch_size, dtype=tf.float32), (batch_size, 1, 1)),
        [1, max_obj_num, 1])
    b_idxs = tf.concat([batch_idx, b_idxs], axis=-1)
    b_idxs = tf.cast(b_idxs, tf.int32)
    peaks = tf.gather_nd(logits, b_idxs)
    peaks = tf.cast(peaks, tf.float32)
    peaks = tf.einsum('b n c, b n ->b n c', peaks, valid_pos)

    # hyperparameters
    beta = 0.9999
    gamma = 2.
    num_of_classes = 7
    b_index = tf.where(tf.math.is_finite(rel_cates))
    class_idx = tf.gather_nd(rel_cates, b_index)
    y, idx, counts = tf.unique_with_counts(class_idx)
    num_cls_smaples = tf.zeros(shape=(num_of_classes, 1),
                               dtype=tf.dtypes.float32)

    y = tf.cast(tf.expand_dims(y, axis=-1), tf.int32)
    r = tf.zeros_like(y, dtype=tf.int32)
    y = tf.concat([y, r], axis=-1)
    counts = tf.reshape(tf.cast(counts, tf.float32), [-1])
    num_cls_smaples = tf.tensor_scatter_nd_update(num_cls_smaples, y, counts)
    effective_num = 1.0 - tf.math.pow(beta, num_cls_smaples)
    weights = (1.0 - beta) / effective_num
    weights = tf.where(weights == np.inf, 0., weights)

    weights = weights / tf.math.reduce_sum(weights) * num_of_classes
    weights = tf.squeeze(weights, axis=-1)
    # (7, 1)

    weights = tf.tile(weights[None, None, :], [batch_size, max_obj_num, 1])
    weights = weights * tars
    weights = tf.math.reduce_sum(weights, axis=-1, keepdims=True)
    weights = tf.tile(weights, [1, 1, num_of_classes])

    entropy_loss = entropy_func(tars, peaks)
    modulator = tf.math.exp(-gamma * tars * peaks -
                            gamma * tf.math.log(1 + tf.math.exp(-1.0 * peaks)))

    modulator = tf.math.reduce_sum(modulator, axis=-1)
    loss = modulator * entropy_loss
    # modulator *

    weighted_loss = tf.math.reduce_sum(weights, axis=-1) * loss
    focal_loss = tf.math.reduce_sum(weighted_loss)
    focal_loss /= tf.math.reduce_sum(tars)
    return focal_loss


def cross_entropy(soft_obj, targets):
    """
    calculate cross entropy and channel represent class
    """
    _, H, W, C = soft_obj.get_shape().as_list()
    soft_obj = tf.reshape(soft_obj, shape=[-1, H * W, C])
    targets = tf.reshape(targets, shape=[-1, H * W, C])
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=targets,
                                                   logits=soft_obj)
    loss = tf.reduce_mean(loss)
    return loss


def penalty_reduced_focal_loss(targets, logits, alpha=2, beta=4):
    """
    Penalty-reduced pixelwise logistic regression with focal loss.
    from paper: [CenterNet: Objects as Points](https://arxiv.org/pdf/1904.07850.pdf)
    """
    loss = 0
    pos_idxs = tf.where(tf.equal(targets, 1.0), tf.ones_like(targets),
                        tf.zeros_like(targets))
    penalty_reduced = tf.math.pow(1 - targets, beta)
    pos_loss = tf.math.pow(1 - logits, alpha) * tf.math.log(logits) * pos_idxs
    neg_loss = (penalty_reduced * tf.math.pow(logits, alpha) *
                tf.math.log(1 - logits) * (1 - pos_idxs))

    num_pos = tf.reduce_mean(tf.math.reduce_sum(pos_idxs, axis=[1, 2, 3]))
    pos_loss = tf.reduce_mean(tf.math.reduce_sum(pos_loss, axis=[1, 2, 3]))
    neg_loss = tf.reduce_mean(tf.math.reduce_sum(neg_loss, axis=[1, 2, 3]))
    loss = tf.cond(
        tf.equal(num_pos, 0.0),
        lambda: loss - neg_loss,
        lambda: loss - (pos_loss + neg_loss) / num_pos,
    )
    return loss


def l1_loss(b_idx, b_sms, tar_vals, batch_size, max_obj_num):

    def make_valid_mask(b_idx):
        is_finite = tf.math.is_finite(b_idx)
        b_idx = tf.where(is_finite, b_idx, tf.zeros_like(b_idx))
        valid_pos_mask = tf.cast(tf.reduce_all(is_finite, axis=-1), tf.float32)
        valid_pos_mask = tf.transpose(valid_pos_mask, [0, 2, 1])
        valid_pos = tf.cast(tf.reduce_any(valid_pos_mask > 0, axis=[1, 2]),
                            tf.float32)
        batchwise_N = tf.reduce_sum((valid_pos), axis=0)
        return valid_pos_mask, batchwise_N

    b_idx = tf.expand_dims(b_idx, axis=-1)
    # B C N 1
    b_idx = tf.reshape(b_idx, [batch_size, max_obj_num, 2, 1])
    b_idx = tf.transpose(b_idx, [0, 3, 1, 2])
    _, _, n, c = b_idx.get_shape().as_list()

    # In order to get 2 channels info
    b_idx = tf.tile(b_idx, [1, c, 1, 1])
    valid_pos_mask, batchwise_N = make_valid_mask(b_idx)
    batch_idx = tf.tile(
        tf.reshape(tf.range(batch_size, dtype=tf.float32), (batch_size, 1, 1)),
        [1, c, n])
    channel_idx = tf.tile(tf.reshape(tf.range(c, dtype=tf.float32), (1, c, 1)),
                          [batch_size, 1, n])
    b_idx = tf.concat([batch_idx[..., None], b_idx, channel_idx[..., None]],
                      axis=-1)
    b_idx = tf.cast(b_idx, tf.int32)
    pred_vals = tf.gather_nd(b_sms, b_idx)
    pred_vals = tf.transpose(pred_vals, [0, 2, 1]) * valid_pos_mask
    batch_loss_matrix = tf.cast(
        tf.abs(pred_vals - tar_vals) * valid_pos_mask, tf.float32)
    batch_loss_matrix = tf.where(tf.math.is_nan(batch_loss_matrix),
                                 tf.zeros_like(batch_loss_matrix),
                                 batch_loss_matrix)
    batch_loss_matrix = tf.reduce_sum(batch_loss_matrix, axis=[1, 2])
    loss = tf.reduce_mean(batch_loss_matrix) / batchwise_N
    return loss


def batch_pull_loss(b_idx, b_kp_hms, batch_size, max_obj_num, filter_channels):
    """ By using euclidean distance, we calculate the pull for a batch.
        The “pull” loss train the network to group the itself embedding vector.
        The idx contains some invalid locations which should be ignored.
        Arguments:
            kp_hms -- BHWC
            [
                [tl_kp_hm],
                [br_kp_hm],
                [st_kp_hm],
                [sb_kp_hm]
            ]
            idxs  -- BN42
            [
                [tl, br, st, sb],
                [tl, br, st, sb],
                .
                .
                .
            ]
            batch_size -- int
            max_obj_num -- number of objects in a batch
            filter_channels --  embedding vectors channels
        return:
            loss: scalar: (b,)
            ek: obj embeddings, (b, ch, N)
            valid_ek_mask: (0., 1.) mask for valid ek locations, (b, N)
    """
    # B N 4 2
    tmp = []
    b_idx = tf.reshape(b_idx, [batch_size, max_obj_num, 2, 2])
    b_idx = tf.transpose(b_idx, [0, 2, 1, 3])

    b_idx = concat_b_idx(b_idx)
    _, c, n, _ = b_idx.get_shape().as_list()

    is_finite = tf.math.is_finite(b_idx)
    b_idx = tf.where(is_finite, b_idx, tf.zeros_like(b_idx))
    batch_idx = tf.tile(
        tf.reshape(tf.range(batch_size, dtype=tf.float32), (batch_size, 1, 1)),
        [1, c, n])
    channel_idx = tf.tile(tf.reshape(tf.range(c, dtype=tf.float32), (1, c, 1)),
                          [batch_size, 1, n])
    # B 8 N 4
    b_idx = tf.concat([batch_idx[..., None], b_idx, channel_idx[..., None]],
                      axis=-1)

    tmp_idx = tf.cast(b_idx, tf.int32)

    valid_pos_mask = tf.cast(tf.reduce_all(is_finite, axis=-1), tf.float32)
    # B 8 N
    # B, 8, N, 4
    peaks = tf.gather_nd(b_kp_hms, tmp_idx) * valid_pos_mask
    # tl as anchor point
    # each point has two embedding channels
    ek = peaks[:, :2, :]
    tmp_valid = []
    for i in range(c // filter_channels):
        dist = tf.abs(peaks[:, i * 2:i * 2 + 2, :] - ek)
        valid = valid_pos_mask[:, i * 2:i * 2 + 1, :]
        sum_channel = tf.sqrt(
            tf.reduce_sum(tf.pow(dist, 2), axis=1, keepdims=True) + 1e-10)
        tmp.append(sum_channel)
        tmp_valid.append(valid)

    dist_diff = tf.concat(tmp, axis=1)
    valid_diff = tf.concat(tmp_valid, axis=1)
    mask_exceeds = tf.where(dist_diff > 4, 4 * tf.ones_like(dist_diff),
                            dist_diff)
    batch_loss_matrix = valid_diff * \
        (tf.math.exp(-0.5 * tf.square(mask_exceeds - 4)))
    batch_loss_matrix = tf.reduce_sum(batch_loss_matrix, axis=[1, 2])
    valid_ek_mask = tf.cast(tf.reduce_any(valid_pos_mask > 0, axis=1),
                            tf.float32)
    batchwise_N = tf.reduce_sum(valid_ek_mask, axis=1)
    loss = tf.reduce_mean(batch_loss_matrix / (batchwise_N + 1e-7))
    return loss, ek, valid_ek_mask


def batch_push_loss(ek, valid_pos, filter_channels):
    """ By using the the euclidean formula, we calculate the batch-wisely push loss.
        The “push” loss is isolated from other's embedding vector except itself.
    Arguments:
        ek               -- object embedding/tag/encoding  N objects in batches (b, N)
        valid_pos        -- valid_ek location, (0., 1.) mask, (b, N)
        filter_channels  -- heat maps channels
    Returns:
        loss             --  loss: scalar: (b,)
    """
    sum_sqr_dist = 0
    for ch in range(filter_channels):
        # ek's shape (b,  N)
        embedding_tl = ek[:, ch, :]
        embedding_tl = tf.math.square(embedding_tl[:, None, :] -
                                      embedding_tl[:, :, None])
        sum_sqr_dist += embedding_tl
    sum_sqr_dist = tf.sqrt(sum_sqr_dist)
    sigma = 1
    batch_loss_matrix = tf.math.exp(-0.5 * tf.square(sum_sqr_dist))
    # get the valid position in the cases of 2 pts, 4 pts or all np.inf
    valid_mask = valid_pos[:, None, :] * valid_pos[:, :, None]
    # mask invalids
    batch_loss_matrix = batch_loss_matrix * valid_mask

    eye = tf.eye(tf.shape(valid_pos)[1])[None, ...]
    _diagonal = batch_loss_matrix * eye
    batch_loss_matrix = batch_loss_matrix - _diagonal
    batchwise_N = tf.reduce_sum(valid_pos, axis=1)
    loss = tf.reduce_sum(batch_loss_matrix,
                         axis=[1, 2]) / (batchwise_N * (batchwise_N - 1) + 1e-7)
    return tf.reduce_mean(loss)


def pull_push_loss(idxs, kp_hms, batch_size, max_obj_num):
    """pull loss
    kp_hms: BWHC
    [
        [tl_kp_hm],
        [br_kp_hm],
        [st_kp_hm],
        [sb_kp_hm],
    ]
    idxs: BN42
    [
        [tl, br, st, sb],
        [tl, br, st, sb],
        .
        .
        .
    ]
    """
    # kp_idx is gt keypoints
    # logits is preded embeds

    filter_channels = get_filter_channels(kp_hms)
    with tf.name_scope("pull_loss"):
        pull_loss, ek, valid_ek_mask = batch_pull_loss(idxs, kp_hms, batch_size,
                                                       max_obj_num,
                                                       filter_channels)
    with tf.name_scope("push_loss"):
        push_loss = batch_push_loss(ek, valid_ek_mask, filter_channels)
    return pull_loss, push_loss


def get_filter_channels(kp_hms):
    _, _, _, hms_channels = kp_hms.get_shape().as_list()
    return hms_channels // 2


def concat_b_idx(b_idx):
    tmp = []
    _, pts_channels, _, _ = b_idx.get_shape().as_list()
    for channel in range(pts_channels):
        new_b_idxes = tf.expand_dims(b_idx[:, channel, :, :], axis=1)
        new_b_idxes = tf.tile(new_b_idxes, [1, 2, 1, 1])
        tmp.append(new_b_idxes)
        if channel == pts_channels - 1:
            b_idx = tf.concat(tmp, axis=1)
    return b_idx


def infonce(b_idxs, b_kp_hms, batch_size, max_obj_num, tau=0.2):
    with tf.name_scope("info_nce_loss"):
        b_idxs = tf.reshape(b_idxs, [batch_size, max_obj_num, 2, 2])
        _, n, c, d = [tf.shape(b_idxs)[i] for i in range(4)]
        indices = tf.tile(
            tf.reshape(tf.range(batch_size, dtype=tf.float32),
                       (batch_size, 1, 1, 1)),
            [1, n, c, 1],
        )
        coors_masks = tf.reduce_all(tf.math.is_finite(b_idxs), axis=-1)
        coors_masks = tf.reduce_any(coors_masks, axis=-1)
        b_idxs = b_idxs[coors_masks]
        indices = indices[coors_masks]
        b_idxs = tf.concat([indices, b_idxs], axis=-1)

        b_idxs = tf.cast(b_idxs, dtype=tf.int32)
        b_logists = tf.gather_nd(b_kp_hms, b_idxs)
        b_logists = tf.math.reduce_sum(b_logists, axis=1) / 2
        b_logists = tf.math.l2_normalize(b_logists, axis=-1)
        n = tf.shape(b_logists)[0]
        labels = tf.eye(n)
        b_logists = tf.linalg.matmul(b_logists, b_logists, transpose_b=True)

        loss = tf.keras.losses.CategoricalCrossentropy(
            reduction=tf.compat.v1.losses.Reduction.NONE)(labels,
                                                          b_logists / tau)
        loss = tf.reduce_sum(loss, axis=-1) / tf.cast(n, tf.float32)
        loss += 2 * tau * loss
        return loss


def offset_loss(b_idx, b_oms, tar_vals, batch_size, max_obj_num):
    with tf.name_scope('offset_loss'):
        # B C N
        valid_mask = tf.math.reduce_all(tf.math.is_finite(b_idx), axis=-1)
        valid_mask = tf.cast(valid_mask, tf.float32)
        valid_n = tf.math.reduce_sum(valid_mask, axis=-1)
        b_info_idx = tf.tile(
            tf.range(batch_size, dtype=tf.int32)[:, None, None],
            (1, max_obj_num, 1))
        b_idx = tf.cast(b_idx, tf.int32)
        b_idx = tf.concat([b_info_idx, b_idx], axis=-1)
        b_pred_off_vals = tf.reshape(tf.gather_nd(b_oms, b_idx),
                                     (batch_size, max_obj_num, 4, 2))
        b_tar_vals = tf.where(tf.math.is_inf(tar_vals), 0., tar_vals)
        b_pred_off_vals = tf.cast(b_pred_off_vals, tf.float32)

        batch_N_loss = tf.math.reduce_sum(
            tf.math.abs(b_tar_vals - b_pred_off_vals), axis=(2, 3)) * valid_mask
        batch_loss = tf.math.reduce_sum(batch_N_loss,
                                        axis=-1) / (valid_n + 1e-7)
        loss = tf.math.reduce_mean(batch_loss)
    return loss


def wpdc_loss(batch_size, max_obj_num, b_idx, gt_vertices, gt_params,
              gt_Z_params, pms, pred_opms, n_s, n_R, shp_base, exp_base):

    def get_Rt_weights(gt_vertices, pred_Rt, gt_Rt):
        pose_diffs = tf.reshape(tf.math.abs(pred_Rt - gt_Rt),
                                (batch_size, max_obj_num, -1))

        pv_nomrs = tf.norm(gt_vertices, ord=2, axis=2)
        # N_vertices = tf.cast(tf.shape(gt_vertices)[-2], tf.float32)
        # offset_norm = tf.math.sqrt(N_vertices)
        weights = []
        for ind in range(9):
            if ind in [0, 1, 2]:
                w = pose_diffs[..., ind] * pv_nomrs[..., 0]
            elif ind in [3, 4, 5]:
                w = pose_diffs[..., ind] * pv_nomrs[..., 1]
            elif ind in [6, 7, 8]:
                w = pose_diffs[..., ind] * pv_nomrs[..., 2]
            # else:
            #     w = pose_diffs[..., ind] * offset_norm
            weights.append(w)
        return tf.stack(weights)

    def get_shp_exp_weights(shp_base, exp_base, pred_shp_exp, gt_shp_exp):
        w_norm = tf.norm(tf.concat((shp_base, exp_base), axis=-1),
                         ord=2,
                         axis=2)
        shp_exp_diffs = tf.reshape(tf.math.abs(pred_shp_exp - gt_shp_exp),
                                   (batch_size, max_obj_num, -1))
        return w_norm * shp_exp_diffs

    eps = 1e-6
    gt_Rt, gt_shp_exp = gt_params[..., n_s:n_R + n_s], gt_params[...,
                                                                 n_R + n_s:]
    with tf.name_scope('wpdc_loss'):
        # B C N
        valid_mask = tf.math.reduce_all(tf.math.is_finite(b_idx), axis=-1)
        b_info_idx = tf.tile(
            tf.range(batch_size, dtype=tf.int32)[:, None, None],
            (1, max_obj_num, 1))
        b_idx = tf.cast(b_idx, tf.int32)
        b_idx = tf.concat([b_info_idx, b_idx], axis=-1)
        pred_Z_params = tf.gather_nd(pred_opms, b_idx)
        # pred_params are reconstruct params
        pred_params = pred_Z_params * pms[1][None, None, :] + pms[0][None,
                                                                     None, :]
        pred_Rt, pred_shp_exp = pred_params[..., n_s:n_s +
                                            n_R], pred_params[..., n_s + n_R:]

        Rt_weights = get_Rt_weights(gt_vertices, pred_Rt, gt_Rt)
        shp_exp_weights = get_shp_exp_weights(shp_base, exp_base, pred_shp_exp,
                                              gt_shp_exp)
        weights = tf.concat(
            [tf.transpose(Rt_weights, (1, 2, 0)), shp_exp_weights], axis=-1)

        weights = (weights + eps) / tf.math.reduce_max(
            weights, keepdims=True, axis=-1)

        R_shp_exp_loss = weights * tf.math.square(pred_Z_params[..., 1:] -
                                                  gt_Z_params[..., 1:])

        s_loss = tf.math.square(pred_Z_params[..., :1] - gt_Z_params[..., :1])
        loss = tf.concat([s_loss, R_shp_exp_loss], axis=-1)

        loss = loss * tf.cast(valid_mask[..., None], tf.float32)
        loss = tf.where(
            tf.tile(valid_mask[..., None], (1, 1, tf.shape(loss)[-1])), loss,
            0.)
        b_objs = tf.math.reduce_sum(tf.cast(valid_mask, tf.float32), axis=-1)
        loss = tf.reduce_sum(loss, axis=(-2, -1)) / b_objs

    return loss, pred_params


def vdc_loss(batch_size, max_obj_num, b_idx, gt_vertices, gt_params, pms,
             pred_params, n_s, n_R, n_shp, u_base, shp_base, exp_base):
    gt_s, gt_R = gt_params[..., :n_s], gt_params[..., n_s:n_s + n_R]
    with tf.name_scope('vdc_loss'):
        # B C N
        valid_mask = tf.math.reduce_all(tf.math.is_finite(b_idx), axis=-1)
        pred_s, pred_R = pred_params[..., :n_s], pred_params[..., n_s:n_s + n_R]
        pred_shp, pred_exp = pred_params[..., n_s + n_R:n_s + n_R +
                                         n_shp], pred_params[...,
                                                             n_s + n_R + n_shp:]
        pred_R = tf.reshape(pred_R, (batch_size, max_obj_num, 3, 3))
        gt_R = tf.reshape(gt_R, (batch_size, max_obj_num, 3, 3))
        pred_vertices = u_base + tf.linalg.matmul(
            shp_base, pred_shp[..., None]) + tf.linalg.matmul(
                exp_base, pred_exp[..., None])
        pred_vertices = tf.reshape(
            pred_vertices,
            (batch_size, max_obj_num, tf.shape(pred_vertices)[2] // 3, 3))
        pred_R = tf.reshape(pred_R, (batch_size, max_obj_num, 3, 3))
        pred_vertices = pred_s[..., None] * tf.linalg.matmul(
            pred_vertices, pred_R, transpose_b=(0, 1, 3, 2))
        pred_lnmks = pred_vertices[..., :68, :2]
        gt_R = tf.reshape(gt_R, (batch_size, max_obj_num, 3, 3))

        gt_vertices = gt_s[..., None] * tf.linalg.matmul(
            gt_vertices, gt_R, transpose_b=(0, 1, 3, 2))
        b_diffs = (gt_vertices - pred_vertices)**2
        b_loss = b_diffs[valid_mask]
    return b_loss, pred_lnmks


def lrr_loss(batch, max_obj_num, pred_lnmks, gt_lnmks):
    valid_mask = tf.math.reduce_all(tf.math.is_finite(gt_lnmks), axis=[-1, -2])
    mask = gt_lnmks == np.inf
    gt_lnmks = tf.where(tf.math.is_inf(gt_lnmks), 0., gt_lnmks)
    mean = tf.math.reduce_mean(gt_lnmks, axis=-2, keepdims=True)
    gt_lnmks -= mean
    gt_lnmks = gt_lnmks[..., ::-1]
    pred_lnmks = tf.reshape(pred_lnmks, [batch, max_obj_num, -1])

    gt_lnmks = tf.reshape(gt_lnmks, [batch, max_obj_num, -1])
    diff = pred_lnmks - gt_lnmks
    loss = tf.math.sqrt(
        tf.math.reduce_sum(tf.math.square(diff), axis=-1) + 1e-9)

    valid_mask = tf.cast(valid_mask, tf.float32)
    loss = loss * valid_mask
    loss = tf.math.reduce_sum(loss, axis=-1) / tf.math.reduce_sum(valid_mask,
                                                                  axis=-1)
    
    return loss
