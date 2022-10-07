import tensorflow as tf
import numpy as np
from pprint import pprint
from monitor import logger


class Restore:

    def __init__(self, cp_dir, resize_size):
        self.cp_dir = cp_dir
        self.inp_size = resize_size

    def flatten_model(self, nested_model):

        def get_layers(layers):
            layers_flat = []
            for layer in layers:
                try:
                    layers_flat.extend(get_layers(layer.layers))
                except AttributeError:
                    layers_flat.append(layer)
            return layers_flat

        flat_model = get_layers(nested_model.layers)
        return flat_model

    def build_restoration(self, model, excluded_layers):
        # load by saved model
        restore_keys = ['backbone', 'neck', 'head']
        restored_model = tf.keras.models.load_model(self.cp_dir)
        model.model(tf.constant(0., shape=[1] + self.inp_size + [3]),
                    training=False)
        logger.info(f'Train from restoration')
        logger.info(f'Initialize for building')
        logger.info(f'Excluded {excluded_layers}'.format(
            excluded_layers=excluded_layers))
        for key in restore_keys:
            try:
                if excluded_layers is not None and key in excluded_layers:
                    continue
                elif key == 'backbone':
                    load_weights = restored_model.backbone.get_layer(
                        'hard_net').get_weights()
                    model.model.backbone.get_layer('hard_net').set_weights(
                        load_weights)
                elif key == 'neck':
                    restore_layers = self.flatten_model(
                        restored_model.get_layer(key))
                    model_layers = self.flatten_model(
                        model.model.get_layer(key))
                    for i, (restore_layer, model_layer) in enumerate(
                            zip(restore_layers, model_layers)):
                        model_layer.set_weights(restore_layer.get_weights())
                else:
                    load_weights = restored_model.get_layer(key).get_weights()
                    model.model.get_layer(key).set_weights(load_weights)

            except KeyError:
                print('Restore key error, please check you model')
        logger.info(f'Finish load-wights')
        return model
