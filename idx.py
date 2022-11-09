import tensorflow as tf
from utils.io import load_BFM
import os
import numpy as np
from matplotlib import pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
BFM = load_BFM("/aidata/anders/objects/3D-head/3DDFA/BFM/BFM.mat")
kpt_ind = tf.cast(BFM['kpt_ind'], tf.float32)
normal_val = tf.random.normal(shape=(3, 68),
                              mean=0.0,
                              stddev=5.0,
                              dtype=tf.dtypes.float32)

X_ind_all = tf.stack([kpt_ind * 3, kpt_ind * 3 + 1, kpt_ind * 3 + 2])
X_ind_all = tf.concat([
    X_ind_all[:, :17], X_ind_all[:, 17:27], X_ind_all[:, 36:48],
    X_ind_all[:, 27:36], X_ind_all[:, 48:68]
],
                      axis=-1)

additional_idx = (X_ind_all + normal_val) + 0.5
X_ind_all = tf.concat([X_ind_all, additional_idx], axis=0)
n_shp = 50
n_exp = 29
X_ind_all = tf.cast(X_ind_all, tf.int32)
valid_ind = tf.reshape(tf.transpose(X_ind_all), (-1))
shapeMU = tf.gather(tf.cast(BFM['shapeMU'], tf.float32), valid_ind)
shapePC = tf.gather(tf.cast(BFM['shapePC'][:, :n_shp], tf.float32), valid_ind)
expPC = tf.gather(tf.cast(BFM['expPC'][:, :n_exp], tf.float32), valid_ind)

sp = tf.zeros(shape=(n_shp, 1), dtype=tf.float32)
ep = tf.zeros(shape=(n_exp, 1), dtype=tf.float32)
head = tf.math.add(shapeMU,
                   tf.linalg.matmul(shapePC, sp) + tf.linalg.matmul(expPC, ep))
lnmks = tf.reshape(head, (tf.shape(head)[0] // 3, 3))
lnmks = lnmks.numpy()

fig = plt.figure()
for i, kp in enumerate(lnmks[:, :2]):
    if i < 68:
        plt.scatter(kp[0], kp[1], color="blue")
    else:
        plt.scatter(kp[0], kp[1], color="green")

plt.savefig("foo.png")
