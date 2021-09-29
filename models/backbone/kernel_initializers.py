import tensorflow as tf


class KernelInitializers:
    def __init__(self):
        self.mapping = {
            'glorot_normal': tf.keras.initializers.GlorotNormal(),
            'glorot_uniform': tf.keras.initializers.GlorotUniform(),
            'he_normal': tf.keras.initializers.HeNormal(),
            'he_uniform': tf.keras.initializers.HeUniform()
        }

    def get_initializer(self, key):
        if key in self.mapping:
            return self.mapping[key]
        print(
            '%s not in kernel initializers, using default glorot_normal init' %
            key)
        return self.mapping['glorot_normal']
