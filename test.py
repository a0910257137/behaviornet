import numpy as np
import tensorflow as tf

overlaps_inf = np.load(
    '/Users/poanchen/Desktop/projects/nanodet/overlaps_inf.npy')
index = np.load('/Users/poanchen/Desktop/projects/nanodet/index.npy')
overlap = np.load('/Users/poanchen/Desktop/projects/nanodet/overlap.npy')

aft_overlaps_inf = np.load(
    '/Users/poanchen/Desktop/projects/nanodet/aft_overlaps_inf.npy')
overlaps_inf = tf.convert_to_tensor(overlaps_inf)
overlap = tf.convert_to_tensor(overlap)
index = tf.convert_to_tensor(index)

overlap = tf.reshape(tf.transpose(overlap), [-1])
ov_vals = tf.gather(overlap, index)


tf.tensor_scatter_nd_update(overlaps_inf, index[:, None], ov_vals)
# overlaps_inf[index] = ov_vals
print(overlaps_inf)
print('-' * 100)
print()
xxx
# print('-' * 100)

print('Answer')
# v = tf.math.equal(candidate_overlaps1, candidate_overlaps2)
print(v)
