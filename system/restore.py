import tensorflow as tf
import copy
import numpy as np
from pprint import pprint
from monitor import logger


class Restore:
    def __init__(self, cp_dir):
        self.cp_dir = cp_dir

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
        logger.info(f'Train from restoration')
        logger.info(f'Initialize for building')
        model.model(tf.constant(0., shape=(192, 320, 3), training=False))
        logger.info(f'Excluded {excluded_layers}'.format(
            excluded_layers=excluded_layers))
        for key in restore_keys:
            try:
                if key == 'backbone':
                    load_weights = restored_model.backbone.get_layer(
                        'hard_net').get_weights()
                    model.model.backbone.get_layer('hard_net').set_weights(
                        load_weights)
                elif excluded_layers is not None and key in excluded_layers:
                    continue
                else:
                    load_weights = restored_model.get_layer(key).get_weights()
                    model.model.get_layer(key).set_weights(
                        load_weights)
            except KeyError:
                print('Restore key error, please check you model')
        logger.info(f'Finish load-wights')
        return model
