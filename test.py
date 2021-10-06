import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

h = 40
w = 40
stride = 8

y_range = (np.arange(h) + 0.5) * stride
x_range = (np.arange(w) + 0.5) * stride
y, x = tf.meshgrid(y_range, x_range)
y = tf.transpose(y)
x = tf.transpose(x)

y = np.reshape(y, [-1])
x = np.reshape(x, [-1])
y = tf.expand_dims(tf.cast(y, tf.float32), axis=-1)
x = tf.expand_dims(tf.cast(x, tf.float32), axis=-1)
grid_cells = tf.concat([y, x], axis=-1)
grid_cells = np.asarray(grid_cells)
for cell in grid_cells[:300]:
    # print(cell)
    plt.scatter(x=cell[1], y=cell[0], s=50, marker='*', c='magenta')
plt.grid()
plt.show()
#TODO: know no transpose and transpose problems
