import tensorflow as tf
import numpy as np
from pprint import pprint
from monitor import logger
import os

is_shrink = False
is_one_branch = True


def bn_fusion(value_lists):
    kernel_weights, kernel_bias = value_lists[0], value_lists[1]
    bn_gamma, bn_beta = value_lists[2], value_lists[3]
    bn_moving_mean, bn_moving_variance = value_lists[4], value_lists[5]
    fused_weights = (kernel_weights * bn_gamma) / np.sqrt(bn_moving_variance +
                                                          0.001)
    fused_bias = bn_beta + (bn_gamma / np.sqrt(bn_moving_variance + 0.001)) * (
        kernel_bias - bn_moving_mean)
    return fused_weights, fused_bias


def save_weight_bias(weight, bias, save_root, name):
    weight = weight.astype(np.float32)
    weight = np.transpose(weight, axes=[0, 1, 3, 2])
    bias = bias.astype(np.float32)
    if 'conv3x3_params' in name:
        save_path = os.path.join(save_root, "params",
                                 "weight_{}.npy".format(name))
        np.save(save_path, weight)
        save_path = os.path.join(save_root, "params/binary",
                                 "weight_{}.bin".format(name))
        weight.tofile(save_path)
        save_path = os.path.join(save_root, "params",
                                 "bias_{}.npy".format(name))
        np.save(save_path, bias)
        save_path = os.path.join(save_root, "params/binary",
                                 "bias_{}.bin".format(name))
        bias.tofile(save_path)
    elif "conv3x3_kps" in name:
        save_path = os.path.join(save_root, "kps",
                                 "weight_{}.npy".format(name))
        np.save(save_path, weight)
        save_path = os.path.join(save_root, "kps/binary",
                                 "weight_{}.bin".format(name))
        weight.tofile(save_path)
        save_path = os.path.join(save_root, "kps", "bias_{}.npy".format(name))
        np.save(save_path, bias)
        save_path = os.path.join(save_root, "kps/binary",
                                 "bias_{}.bin".format(name))
        bias.tofile(save_path)
    elif "pred_bbox" in name:
        save_path = os.path.join(save_root, "bbox",
                                 "weight_{}.npy".format(name))
        np.save(save_path, weight)
        save_path = os.path.join(save_root, "bbox/binary",
                                 "weight_{}.bin".format(name))
        weight.tofile(save_path)
        save_path = os.path.join(save_root, "bbox", "bias_{}.npy".format(name))
        np.save(save_path, bias)
        save_path = os.path.join(save_root, "bbox/binary",
                                 "bias_{}.bin".format(name))
        bias.tofile(save_path)

    elif "pred_params" in name:
        save_path = os.path.join(save_root, "params",
                                 "weight_{}.npy".format(name))
        np.save(save_path, weight)
        save_path = os.path.join(save_root, "params/binary",
                                 "weight_{}.bin".format(name))
        weight.tofile(save_path)

        save_path = os.path.join(save_root, "params",
                                 "bias_{}.npy".format(name))
        np.save(save_path, bias)

        save_path = os.path.join(save_root, "params/binary",
                                 "bias_{}.bin".format(name))
        bias.tofile(save_path)

    elif "pred_kp" in name:
        save_path = os.path.join(save_root, "kps",
                                 "weight_{}.npy".format(name))
        np.save(save_path, weight)
        save_path = os.path.join(save_root, "kps/binary",
                                 "weight_{}.bin".format(name))
        weight.tofile(save_path)
        save_path = os.path.join(save_root, "kps", "bias_{}.npy".format(name))
        np.save(save_path, bias)

        save_path = os.path.join(save_root, "kps/binary",
                                 "bias_{}.bin".format(name))
        bias.tofile(save_path)


def save_scales(restored_scales, save_root):
    for restored_scale in restored_scales:
        s_vals = [restored_scale.numpy().astype(np.float32)]
        np.save(
            os.path.join(save_root, "bbox",
                         "{}.npy".format(restored_scale.name)), s_vals)
        restored_scale.numpy().astype(np.float32).tofile(
            os.path.join(save_root, "bbox/binary",
                         "{}.bin".format(restored_scale.name)))


def make_dir(path):
    if not os.path.exists(path):
        os.umask(0)
        os.makedirs(path, mode=0o755)


class Restore:

    def __init__(self, head_cfg, cp_dir, resize_size):
        self.cp_dir = cp_dir
        self.inp_size = resize_size
        self.head_branch = head_cfg.head_branch

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
        logger.info(f'Train from restoration')
        logger.info(f'Initialize for building')
        logger.info(f'Excluded {excluded_layers}'.format(
            excluded_layers=excluded_layers))
        if self.head_branch != None:
            logger.info(f'Enable branch model')
            model_dir = os.path.join(self.cp_dir, "branch")
            make_dir(model_dir)

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
                    restore_layers = self.flatten_model(
                        restored_model.get_layer(key))
                    model_layers = self.flatten_model(
                        model.model.get_layer(key))
                    if self.head_branch != None:
                        conv3x3_cls = restore_layers[:2]
                        conv3x3_params = restore_layers[6:8]
                        conv3x3_kps = restore_layers[12:14]
                        pred_cls = restore_layers[18]
                        pred_bbox = restore_layers[21]
                        pred_param = restore_layers[24]
                        pred_kps = restore_layers[27]
                        restore_layers = conv3x3_cls + conv3x3_params + conv3x3_kps + [
                            pred_cls
                        ] + [pred_bbox] + [pred_param] + [pred_kps]
                    model.model.get_layer(
                        key).scales[0]._values = restored_model.get_layer(
                            key).scales[1]._values
                    model.model.get_layer(
                        key).scales = restored_model.get_layer(key).scales
                    for i, (restore_layer, model_layer) in enumerate(
                            zip(restore_layers, model_layers)):
                        model_layer.set_weights(restore_layer.get_weights())
            except KeyError:
                print('Restore key error, please check you model')
        if self.head_branch != None:
            logger.info(f'Save branch model and exit ...')
            tf.keras.models.save_model(model.model, model_dir)
            exit(1)
        logger.info(f'Finish load-wights')
        return model
