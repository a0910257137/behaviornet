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
        restored_model = tf.keras.models.load_model(self.cp_dir)

        restore_weights = restored_model.backbone.get_layer(
            'hard_net').get_weights()

        model_backbone = model.model.backbone.get_layer('hard_net')
        model_backbone.set_weights(restore_weights)
        restore_weights = restored_model.get_layer('neck').get_weights()
        model_neck = model.model.get_layer('neck')
        model_neck.set_weights(restore_weights)
        # restored_layers = self.flatten_model(restored_model)
        # model_layers = self.flatten_model(model_backbone)
        # model_layers = self.flatten_model(model.model)
        # logger.info(f'Train from restoration')
        # logger.info(f'Excluded {excluded_layers}'.format(
        #     excluded_layers=excluded_layers))
        # for i, (model_layer,
        #         restored_layer) in enumerate(zip(model_layers,
        #                                          restored_layers)):
        #     try:

        #         if i in [47, 48, 49]:
        #             continue
        #         model_name = model_layer.name
        #         restored_name = restored_layer.name
        #         if excluded_layers is None or restored_name not in excluded_layers:
        #             model_layer.set_weights(restored_layer.get_weights())
        #         elif restored_name in excluded_layers:
        #             continue
        #     except KeyError:
        #         print('Restore key error, please check you model')
        # logger.info(f'Finish load-wights')
        return model
