import tensorflow as tf
import numpy as np
from pprint import pprint
from monitor import logger
import os


def bn_fusion(value_lists):
    kernel_weights, kernel_bias = value_lists[0], value_lists[1]
    bn_gamma, bn_beta = value_lists[2], value_lists[3]
    bn_moving_mean, bn_moving_variance = value_lists[4], value_lists[5]
    fused_weights = (kernel_weights * bn_gamma) / np.sqrt(bn_moving_variance)
    fused_bias = bn_beta + (bn_gamma / np.sqrt(bn_moving_variance)) * (
        kernel_bias - bn_moving_mean)
    return fused_weights, fused_bias


def save_weight_bias(weight, bias, save_root, name):
    if 'conv3x3_params' in name:
        save_path = os.path.join(save_root, "params",
                                 "weight_{}.npy".format(name))
        np.save(save_path, weight)
        save_path = os.path.join(save_root, "params",
                                 "bias_{}.npy".format(name))
        np.save(save_path, bias)
    elif "conv3x3_kps" in name:
        save_path = os.path.join(save_root, "kps", "weight_{}.npy".format(name))
        np.save(save_path, weight)
        save_path = os.path.join(save_root, "kps", "bias_{}.npy".format(name))
        np.save(save_path, bias)
    elif "pred_bbox" in name:
        save_path = os.path.join(save_root, "bbox",
                                 "weight_{}.npy".format(name))
        np.save(save_path, weight)
        save_path = os.path.join(save_root, "bbox", "bias_{}.npy".format(name))
        np.save(save_path, bias)

    elif "pred_params" in name:
        save_path = os.path.join(save_root, "params",
                                 "weight_{}.npy".format(name))
        np.save(save_path, weight)
        save_path = os.path.join(save_root, "params",
                                 "bias_{}.npy".format(name))
        np.save(save_path, bias)
    elif "pred_kp" in name:
        save_path = os.path.join(save_root, "kps", "weight_{}.npy".format(name))
        np.save(save_path, weight)
        save_path = os.path.join(save_root, "kps", "bias_{}.npy".format(name))
        np.save(save_path, bias)


def save_scales(restored_scales, model_scales, save_root):
    for restored_scale, model_scale in zip(restored_scales, model_scales):
        restored_scale = [restored_scale.numpy()]
        np.save(
            os.path.join(save_root, "bbox", "{}.npy".format(model_scale.name)),
            restored_scale)


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

    def build_restoration(self,
                          model,
                          excluded_layers,
                          model_type='mobile_next_net_model'):
        # load by saved model
        restore_keys = ['backbone', 'neck', 'head']
        restored_model = tf.keras.models.load_model(self.cp_dir)
        model.model(tf.constant(0., shape=[1] + self.inp_size + [3]),
                    training=False)
        # load_weights = restored_model.backbone.get_layer(
        #     'mobile_net_model').get_weights()
        # model.model.set_weights(restored_model.get_weights())
        logger.info(f'Train from restoration')
        logger.info(f'Initialize for building')
        logger.info(f'Excluded {excluded_layers}'.format(
            excluded_layers=excluded_layers))
        # weight_path = "/aidata/anders/data_collection/okay/total/archives/WF/scale_down/weights"
        for key in restore_keys:
            try:
                if excluded_layers is not None and key in excluded_layers:
                    continue
                elif key == 'backbone':

                    load_weights = restored_model.backbone.get_layer(
                        model_type).get_weights()
                    model.model.backbone.get_layer(model_type).set_weights(
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
                    # load_weights = restored_model.get_layer(key).get_weights()
                    restore_layers = self.flatten_model(
                        restored_model.get_layer(key))
                    model_layers = self.flatten_model(
                        model.model.get_layer(key))
                    # save_scales(
                    #     restored_model.get_layer(key).__dict__["scales"],
                    #     model.model.get_layer(key).__dict__["scales"],
                    #     weight_path)
                    # model_other_layers = model_layers[7:19] + model_layers[22:]
                    # restore_other_layers = restore_layers[
                    #     7:19] + restore_layers[22:]
                    # for restore_layer, model_layer in zip(
                    #         restore_other_layers, model_other_layers):
                    #     kernel_weight_bias = restore_layer.get_weights()
                    #     if "conv3x3" in model_layer.name:
                    #         weight, bias = bn_fusion(kernel_weight_bias)
                    #     else:
                    #         weight, bias = kernel_weight_bias
                    #     save_weight_bias(weight, bias, weight_path,
                    #                      model_layer.name)
                    # restore_layers = restore_layers[:7] + restore_layers[19:22]
                    for i, (restore_layer, model_layer) in enumerate(
                            zip(restore_layers, model_layers)):
                        model_layer.set_weights(restore_layer.get_weights())
            except KeyError:
                print('Restore key error, please check you model')
        logger.info(f'Finish load-wights')
        # model_dir = "/aidata/anders/data_collection/okay/total/archives/WF/scale_down"
        # tf.keras.models.save_model(model.model, model_dir)
        return model
