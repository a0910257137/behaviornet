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


def wpdc_loss(b_idx, gt_params, param_mean_std, pred_opms, n_s, n_R, n_shp,
              n_exp, kpt_ind, shapeMU, shapePC, expPC, batch_size, max_obj_num):

    def resample(gt_params, kpt_ind):
        # resmpale different vertices for 68 landmarks and 132 random samples
        index = tf.random.shuffle(tf.range(start=0, limit=53215,
                                           dtype=tf.int32))[:132]
        index = tf.reshape(index, (-1, 1))
        keypoints_resample = tf.concat(
            [3 * index, 3 * index + 1, 3 * index + 2], axis=-1)
        keypoints_mix = tf.concat([kpt_ind, keypoints_resample], axis=0)
        n_objs = tf.shape(gt_params)[1]
        keypoints_mix = tf.reshape(keypoints_mix, [-1])
        u_base = tf.tile(
            tf.gather(shapeMU, keypoints_mix)[None, None, :, :],
            [batch_size, n_objs, 1, 1])
        w_shp_base = tf.tile(
            tf.gather(shapePC, keypoints_mix)[None, None, :, :],
            [batch_size, n_objs, 1, 1])
        w_exp_base = tf.tile(
            tf.gather(expPC, keypoints_mix)[None, None, :, :],
            (batch_size, n_objs, 1, 1))
        return u_base, w_shp_base, w_exp_base

    u_base, w_shp_base, w_exp_base = resample(gt_params, kpt_ind)
    gt_s, gt_R, gt_shp_exp = gt_params[..., :n_s], gt_params[
        ..., n_s:n_R + n_s], gt_params[..., n_R + n_s:]

    mean_s, mean_R, mean_shp, mean_exp = param_mean_std[0, :1], param_mean_std[
        0, 1:10], param_mean_std[0, 10:209], param_mean_std[0, 209:238]
    std_s, std_R, std_shp, std_exp = param_mean_std[1, :1], param_mean_std[
        1, 1:10], param_mean_std[1, 10:209], param_mean_std[1, 209:238]
    mean_shp, mean_exp = mean_shp[:n_shp], mean_exp[:n_exp]
    std_shp, std_exp = std_shp[:n_shp], std_exp[:n_exp]

    gt_vertices = u_base + tf.linalg.matmul(
        w_shp_base, gt_shp_exp[:, :, :n_shp, tf.newaxis]) + tf.linalg.matmul(
            w_exp_base, gt_shp_exp[:, :, n_shp:, tf.newaxis])
    gt_vertices = tf.reshape(gt_vertices, [batch_size, max_obj_num, 200, 3])
    with tf.name_scope('wpdc_loss'):
        # B C N
        valid_mask = tf.math.reduce_all(tf.math.is_finite(b_idx), axis=-1)
        b_info_idx = tf.tile(
            tf.range(batch_size, dtype=tf.int32)[:, None, None],
            (1, max_obj_num, 1))
        b_idx = tf.cast(b_idx, tf.int32)
        b_idx = tf.concat([b_info_idx, b_idx], axis=-1)
        pred_params = tf.gather_nd(pred_opms, b_idx)

        pred_s, pred_R, pred_shp_exp = pred_params[..., :n_s], pred_params[
            ..., n_s:n_s + n_R], pred_params[..., n_s + n_R:]

        pose_diffs = tf.reshape(tf.math.abs(pred_R - gt_R),
                                (batch_size, max_obj_num, -1))
        pv_nomrs = tf.norm(gt_vertices, ord=2, axis=2)
        weights = []
        for ind in range(9):
            if ind in [0, 3, 6]:
                w = pose_diffs[..., ind] * pv_nomrs[..., 0]
            elif ind in [1, 4, 7]:
                w = pose_diffs[..., ind] * pv_nomrs[..., 1]
            elif ind in [2, 5, 8]:
                w = pose_diffs[..., ind] * pv_nomrs[..., 2]
            weights.append(w)
        weights = tf.stack(weights)
        magic_number = 0.00057339936  # scale
        w_norm = tf.norm(tf.concat((w_shp_base, w_exp_base), axis=-1),
                         ord=2,
                         axis=2)
        shp_exp_diffs = tf.reshape(tf.math.abs(pred_shp_exp - gt_shp_exp),
                                   (batch_size, max_obj_num, -1))
        eps = 1e-6
        weights = tf.concat(
            [tf.transpose(weights, (1, 2, 0)), gt_s * shp_exp_diffs * w_norm],
            axis=-1)
        weights = (weights + eps) / tf.math.reduce_max(
            weights, keepdims=True, axis=-1)
        # Z-score as noirmalization ~ N(0, 1)
        mean_s_R_shp_exp = tf.concat([mean_s, mean_R, mean_shp, mean_exp],
                                     axis=0)
        std_s_R_shp_exp = tf.concat([std_s, std_R, std_shp, std_exp], axis=0)
        Z = (gt_params -
             mean_s_R_shp_exp[None, None, :]) / std_s_R_shp_exp[None, None, :]
        R_shp_exp_loss = weights * tf.math.square(pred_params[..., 1:] -
                                                  Z[..., 1:])
        s_loss = tf.math.square(pred_params[..., :1] - Z[..., :1])
        loss = tf.concat([s_loss, R_shp_exp_loss], axis=-1)
        loss = loss * tf.cast(valid_mask[..., None], tf.float32)
        loss = tf.where(
            tf.tile(valid_mask[..., None], (1, 1, tf.shape(loss)[-1])), loss,
            0.)
        b_objs = tf.math.reduce_sum(tf.cast(valid_mask, tf.float32), axis=-1)
        loss = tf.reduce_sum(loss, axis=(-2, -1)) / b_objs
    return tf.math.reduce_mean(loss)


def vdc_loss(b_idx, param_u_std, b_opms, shapeMU, shapePC, expPC, kpt_ind,
             shape_params, expression_params, scale, angles, tanslations,
             batch_size, max_obj_num):

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
        # use 1 rad =  57.3
        x, y, z = angles[..., 0], angles[..., 1], angles[..., 2]
        # x, 3, 3
        # for Rx
        row1 = tf.constant([1., 0., 0.], shape=(3, 1))
        row1 = tf.tile(row1[None, None, :, :], (batch_size, max_obj_num, 1, 1))
        row2 = tf.concat([
            tf.zeros(shape=(batch_size, max_obj_num, 1, 1)),
            tf.math.cos(x)[..., None, None], -tf.math.sin(x)[..., None, None]
        ],
                         axis=-2)
        row3 = tf.concat([
            tf.zeros(shape=(batch_size, max_obj_num, 1, 1)),
            tf.math.sin(x)[..., None, None],
            tf.math.cos(x)[..., None, None]
        ],
                         axis=-2)
        Rx = tf.concat([row1, row2, row3], axis=-1)
        # for Ry
        # y
        row1 = tf.concat([
            tf.math.cos(y)[..., None, None],
            tf.zeros(shape=(batch_size, max_obj_num, 1, 1)),
            tf.math.sin(y)[..., None, None]
        ],
                         axis=-2)
        row2 = tf.constant([0., 1., 0.], shape=(3, 1))
        row2 = tf.tile(row2[None, None, :, :], (batch_size, max_obj_num, 1, 1))

        row3 = tf.concat([
            -tf.math.sin(y)[..., None, None],
            tf.zeros(shape=(batch_size, max_obj_num, 1, 1)),
            tf.math.cos(y)[..., None, None]
        ],
                         axis=-2)
        Ry = tf.concat([row1, row2, row3], axis=-1)
        # z
        row1 = tf.concat([
            tf.math.cos(z)[..., None, None], -tf.math.sin(z)[..., None, None],
            tf.zeros(shape=(batch_size, max_obj_num, 1, 1))
        ],
                         axis=-2)
        row2 = tf.concat([
            tf.math.sin(z)[..., None, None],
            tf.math.cos(z)[..., None, None],
            tf.zeros(shape=(batch_size, max_obj_num, 1, 1))
        ],
                         axis=-2)
        row3 = tf.constant([0., 0., 1.], shape=(3, 1))
        row3 = tf.tile(row3[None, None, :, :], (batch_size, max_obj_num, 1, 1))
        Rz = tf.concat([row1, row2, row3], axis=-1)
        # R = tf.linalg.matmul(Rz, tf.linalg.matmul(Ry, Rx))
        return (Rx, Ry, Rz)

    index = tf.random.shuffle(tf.range(start=0, limit=53215,
                                       dtype=tf.int32))[:132]
    index = tf.reshape(index, (-1, 1))
    keypoints_resample = tf.concat([3 * index, 3 * index + 1, 3 * index + 2],
                                   axis=-1)
    keypoints_mix = tf.concat([kpt_ind, keypoints_resample], axis=0)
    n_objs = tf.shape(shape_params)[1]
    keypoints_mix = tf.reshape(keypoints_mix, [-1])
    u_base = tf.tile(
        tf.gather(shapeMU, keypoints_mix)[None, None, :, :],
        [batch_size, n_objs, 1, 1])
    w_shp_base = tf.tile(
        tf.gather(shapePC, keypoints_mix)[None, None, :, :],
        [batch_size, n_objs, 1, 1])
    w_exp_base = tf.tile(
        tf.gather(expPC, keypoints_mix)[None, None, :, :],
        (batch_size, n_objs, 1, 1))

    vertices = u_base + tf.linalg.matmul(w_shp_base,
                                         shape_params) + tf.linalg.matmul(
                                             w_exp_base, expression_params)
    # pvs = tf.reshape(pvs, [batch_size, n_objs, 200, 3])
    vertices = tf.transpose(
        tf.reshape(vertices, (batch_size, max_obj_num, 3, 200)), (0, 1, 3, 2))
    with tf.name_scope('vdc_loss'):
        # B C N
        valid_mask = tf.math.reduce_all(tf.math.is_finite(b_idx), axis=-1)
        b_info_idx = tf.tile(
            tf.range(batch_size, dtype=tf.int32)[:, None, None],
            (1, max_obj_num, 1))
        b_idx = tf.cast(b_idx, tf.int32)
        b_idx = tf.concat([b_info_idx, b_idx], axis=-1)
        b_idx = tf.where(b_idx < -1000, 0, b_idx)
        b_pred_pose_vals = tf.gather_nd(b_opms, b_idx)
        pred_rx, pred_ry, pred_rz = angle2matrix(b_pred_pose_vals)
        gt_rx, gt_ry, gt_rz = angle2matrix(angles)
        pred_R = tf.linalg.matmul(pred_rz, tf.linalg.matmul(pred_ry, pred_rx))
        gt_R = tf.linalg.matmul(gt_rz, tf.linalg.matmul(gt_ry, gt_rx))
        pred_vertices = scale[..., None, None] * tf.linalg.matmul(
            vertices, pred_R,
            transpose_b=(0, 1, 3, 2)) + tanslations[:, :, tf.newaxis, :]
        gt_vertices = scale[..., None, None] * tf.linalg.matmul(
            vertices, gt_R,
            transpose_b=(0, 1, 3, 2)) + tanslations[:, :, tf.newaxis, :]
        b_diffs = (gt_vertices - pred_vertices)**2
        b_loss = tf.where(
            tf.tile(valid_mask[..., None, None],
                    (1, 1, tf.shape(b_diffs)[-2], 1)), b_diffs, 0.)
        b_objs = tf.math.reduce_sum(tf.cast(valid_mask, tf.float32), axis=-1)
        b_loss = tf.reduce_sum(b_loss, axis=[1, 2, 3]) / b_objs
        loss = tf.math.reduce_mean(b_loss)
    return loss