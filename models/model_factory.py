from .module import MODULE_FACTORY
from .neck import NECK_FACTORY
from .head import HEAD_FACTORY
from .loss import LOSS_FACTORY
import tensorflow as tf
from .network import Network
from .backbone.hardnet import *
# from .backbone.evo import *
# from .backbone.shufflenetv2 import SuffleNet

from pprint import pprint


class ModelFactory:

    def __init__(self, config, lr):
        self.config = config
        self._model_keys = ['backbone', 'neck', 'head']
        self.img_channel = 3
        self.backbone = HardNet39(input_shape=(self.config.resize_size[0],
                                               self.config.resize_size[1],
                                               self.img_channel),
                                  pooling='avg_pool',
                                  kernel_initializer='he_uniform')
        self.neck = None
        if self.config.neck.module_name is not None:
            self.neck = NECK_FACTORY.get(self.config.neck.module_name)(
                self.config, name='neck')
        self.head = HEAD_FACTORY.get(self.config.head.module_name)(self.config,
                                                                   name='head')
        self.loss = LOSS_FACTORY.get(self.config.loss.type)(
            self.config).build_loss
        self.modules = MODULE_FACTORY.get(self.config.model_name)(self.config,
                                                                  self.backbone,
                                                                  self.neck,
                                                                  self.head)

    def build_model(self):
        network = Network(self.config,
                          self.modules,
                          self._model_keys,
                          name='network')
        optimizers = self._optimizer()
        network.compile(optimizer=optimizers, loss=self.loss, run_eagerly=False)
        return network, optimizers

    def _optimizer(self):
        optimizers = {}
        if self.config.multi_optimizer:
            for model_key in self._model_keys:
                optimizer_key = self.config[model_key].optimizer
                lr = self.config[model_key].lr
                if optimizer_key == 'adam':
                    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
                elif optimizer_key == 'sgd':
                    optimizer = tf.keras.optimizers.SGD(learning_rate=lr,
                                                        momentum=0.9,
                                                        nesterov=True)
                elif optimizer_key == 'nesterov_mom':
                    optimizer = tf.train.MomentumOptimizer(learning_rate=lr,
                                                           momentum=0.9,
                                                           use_nesterov=True)
                optimizers[model_key] = optimizer
            return optimizers
        else:
            optimizer_key = self.config.optimizer
            lr = self.config.lr
            if optimizer_key == 'adam':
                return tf.keras.optimizers.Adam(learning_rate=lr)
            elif optimizer_key == 'sgd':
                return tf.keras.optimizers.SGD(learning_rate=lr,
                                               momentum=0.9,
                                               decay=1e-6,
                                               nesterov=True)
            elif optimizer_key == 'nesterov_mom':
                return tf.train.MomentumOptimizer(learning_rate=lr,
                                                  momentum=0.9,
                                                  use_nesterov=True)
