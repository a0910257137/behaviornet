import tensorflow as tf
from .module import MODULE_FACTORY
from .backbone.mobilenet import MobileNetV3
from .head import HEAD_FACTORY
from .loss import LOSS_FACTORY
from .network import Network
from pprint import pprint


class ModelFactory:
    def __init__(self, config, cp_path, lr):
        self.config = config
        self._model_keys = ['backbone', 'head']
        self.backbone = MobileNetV3(input_shape=(self.config.resize_size[0],
                                                 self.config.resize_size[1],
                                                 3),
                                    kernel_initializer='he_uniform')

        self.head = HEAD_FACTORY.get(self.config.head.module_name)(self.config,
                                                                   name='head')

        self.loss = LOSS_FACTORY.get(self.config.loss.type)(
            self.config).build_loss
        self.modules = MODULE_FACTORY.get(self.config.model_name)(
            self.config, self.backbone, self.head)

    def build_model(self):
        network = Network(self.config,
                          self.modules,
                          self._model_keys,
                          name='network')
        optimizers = self._optimizer()
        network.compile(optimizer=optimizers,
                        loss=self.loss,
                        run_eagerly=False)
        return network, optimizers

    def _optimizer(self):
        optimizers = {}
        for model_key in self._model_keys:
            optimizer_key = self.config[model_key].optimizer
            lr = self.config[model_key].lr
            if optimizer_key == 'adam':
                optimizer = tf.keras.optimizers.Adam(lr=lr)
            elif optimizer_key == 'sgd':
                optimizer = tf.keras.optimizers.SGD(lr=lr,
                                                    momentum=0.9,
                                                    nesterov=True)
            elif optimizer_key == 'nesterov_mom':
                optimizer = tf.train.MomentumOptimizer(lr,
                                                       0.9,
                                                       use_nesterov=True)
            optimizers[model_key] = optimizer
        return optimizers
