import tensorflow as tf
import numpy as np
from pprint import pprint
from keras_flops import get_flops


class Network(tf.keras.Model):

    def __init__(self, config, model, _model_keys, **kwargs):
        super(Network, self).__init__(**kwargs)
        self.model = model
        self.config = config
        self._model_keys = _model_keys
        self.task = self.config.tasks[0]['preprocess']
        self.epochs = 0
        if self.task == 'tdmm' or self.task == 'keypoint':
            # if self.task == 'tdmm':
            pms = np.load(self.config['3dmm']['pms_path'])
            n_s = self.config['3dmm']["n_s"]
            n_R = self.config['3dmm']["n_R"]
            n_shp = self.config['3dmm']["n_shp"]
            n_exp = self.config['3dmm']["n_exp"]
            n_t3d = self.config['3dmm']["n_t3d"]
            Rt = pms[:, n_s:n_s + n_R]
            shp = pms[:, n_s + n_R:n_s + n_R + n_shp]
            exp = pms[:, n_s + n_R + n_shp:n_s + n_R + n_shp + n_exp]

            pms = np.concatenate([Rt, shp, exp], axis=-1)
            self.train_mean_std = tf.cast(pms[
                :2,
            ], tf.float32)

    def compile(self, optimizer, loss, run_eagerly=None):
        super(Network, self).compile(optimizer=optimizer,
                                     run_eagerly=run_eagerly,
                                     metrics=['accuracy'])
        self._loss = loss
        self.optimizer = optimizer
        # image_inputs = tf.keras.Input(shape=(320, 320, 3), name='image_inputs')
        # preds = self.model(image_inputs, training=False)
        # model_dir = "/aidata/anders/data_collection/okay/total/archives/whole/test"
        # tf.keras.models.save_model(self.model, model_dir)
        # flops = get_flops(fully_models, batch_size=1)
        # print(f"FLOPS: {flops / 10 ** 9:.03} G")
        # exit(1)

    def train_step(self, data):
        training = True
        imgs, labels = data
        if self.task == 'tdmm' or self.task == 'keypoint':
            # if self.task == 'tdmm':
            labels['Z_params'] = (labels['params'] -
                                  self.train_mean_std[0][None, None, :]
                                  ) / self.train_mean_std[1][None, None, :]
            labels['mean_std'] = self.train_mean_std
        with tf.GradientTape() as tape:
            preds = self.model(imgs, training=training)
            loss = self._loss(self.epochs, preds, labels, training)

        if self.config.multi_optimizer:
            self._gradient(self.model, self.optimizer, loss['total'], tape)
        else:
            trainable_vars = self.model.trainable_variables
            grads = tape.gradient(loss['total'], trainable_vars)
            self.optimizer.apply_gradients(zip(grads, trainable_vars))
        return loss

    def test_step(self, data):
        training = False
        imgs, labels = data
        if self.task == 'tdmm' or self.task == 'keypoint':
            # if self.task == 'tdmm':
            labels['Z_params'] = (labels['params'] -
                                  self.train_mean_std[0][None, None, :]
                                  ) / self.train_mean_std[1][None, None, :]
            labels['mean_std'] = self.train_mean_std
        preds = self.model(imgs, training=training)
        loss = self._loss(self.epochs, preds, labels, training)
        return loss

    def get_trainable_variables(self, model):

        def disable_training(layer):
            for l in layer.layers:
                l.trainable = False
                if hasattr(l, "layers"):
                    disable_training(l)
                elif isinstance(l, tf.keras.layers.BatchNormalization):
                    l.trainable = False
                elif isinstance(l, tf.keras.layers.Conv2D):
                    l.trainable = False

        model_dicts = self.config
        model_keys = [l.name for l in model.layers]
        model_total_vars = []
        for key in self._model_keys:
            if self.config.model_name == 'detr' and self.config.frozen_backbone and key == 'backbone':
                disable_training(model.get_layer(key))
            if key not in model_keys:
                raise KeyError('Set wrong model key')
            layer_vars = model.get_layer(key).trainable_variables
            model_dicts[key]['num_vars'] = len(layer_vars)
            model_dicts[key]['variables'] = layer_vars
            model_total_vars += layer_vars
        return model_dicts, model_total_vars

    def _gradient(self, model, optimizers, total_loss, tape):
        model_dicts, model_total_vars = self.get_trainable_variables(model)
        gradients = tape.gradient(total_loss, model_total_vars)
        for key in self._model_keys:
            num_vars = model_dicts[key]['num_vars']
            train_vars = model_dicts[key]['variables']
            self.optimizer[key].apply_gradients(
                zip(gradients[:num_vars], train_vars))
            gradients = gradients[num_vars:]

    def _get_wpdc(self, x):
        x['total'] = x['obj_heat_map'] + x['wpdc']
        return x

    def _get_vdc(self, x):
        x['total'] = x['obj_heat_map'] + x['vdc']
        return x

    def get_config(self):
        return super().get_config()
