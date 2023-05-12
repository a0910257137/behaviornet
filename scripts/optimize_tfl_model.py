import sys
import os
import argparse
import commentjson
from pathlib import Path
import numpy as np
import multiprocessing

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.module import MODULE_FACTORY
from models.neck import NECK_FACTORY
from models.head import HEAD_FACTORY
from models.backbone.hardnet import *
from box import Box
import cv2
from glob import glob

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
threads = multiprocessing.cpu_count()


def flatten_model(nested_model):

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


def _folding_bn(weights):
    depth_kernel, point_kernel, bias, gamma, beta, moving_mean, moving_variance = weights
    point_kernel = gamma * point_kernel / (np.sqrt(moving_variance +
                                                   tf.keras.backend.epsilon()))
    bias = beta + gamma * (bias - moving_mean) / (
        np.sqrt(moving_variance + tf.keras.backend.epsilon()))
    return depth_kernel, point_kernel, bias


def save(dir, depth_kernel, point_kernel, bias, conv_1x1_kernel, conv_1x1_bias):
    if not os.path.exists(dir):
        os.umask(0)
        os.makedirs(dir, mode=0o755)
    if depth_kernel is not None:
        np.save(os.path.join(dir, 'conv_3x3_depth.npy'), depth_kernel)
    if point_kernel is not None:
        np.save(os.path.join(dir, 'conv_3x3_point.npy'), point_kernel)
    np.save(os.path.join(dir, 'conv_3x3_bias.npy'), bias)
    np.save(os.path.join(dir, 'conv_1x1_kernel.npy'), conv_1x1_kernel)
    np.save(os.path.join(dir, 'conv_1x1_bias.npy'), conv_1x1_bias)

    # deal with binary files
    if depth_kernel is not None:
        depth_kernel.tofile(os.path.join(dir, "conv_3x3_depth.bin"))

    if point_kernel is not None:
        point_kernel.tofile(os.path.join(dir, "conv_3x3_point.bin"))

    bias.tofile(os.path.join(dir, "conv_3x3_bias.bin"))
    conv_1x1_kernel.tofile(os.path.join(dir, "conv_1x1_kernel.bin"))
    conv_1x1_bias.tofile(os.path.join(dir, "conv_1x1_bias.bin"))


def _experimentt_folding(save_root, weights):

    kernel, bias, gamma, beta, moving_mean, moving_variance = weights
    kernel = gamma * kernel / (np.sqrt(moving_variance +
                                       tf.keras.backend.epsilon()))
    bias = beta + gamma * (bias - moving_mean) / (
        np.sqrt(moving_variance + tf.keras.backend.epsilon()))
    save(os.path.join(save_root, 'experiment'), None, None, bias, kernel, bias)


def _restore(model, cp_dir, save_root):
    # load by saved model
    restore_keys = ['backbone', 'neck', 'head']

    restored_model = tf.keras.models.load_model(cp_dir)
    _ = restored_model(tf.constant(0., shape=(1, 192, 320, 3)), training=False)
    _ = model(tf.constant(0., shape=(1, 192, 320, 3)), training=False)
    for key in restore_keys:
        try:
            if key == 'head':
                # fetch model layers
                restore_layers = flatten_model(restored_model.get_layer(key))
                model_layers = flatten_model(model.get_layer(key))

                restore_offset_layers = restore_layers[2:7]
                restore_size_layers = restore_layers[7:9]
                experiment_layers = restore_layers[9:]
                # step 1 heat map
                for i, (restore_layer, model_layer) in enumerate(
                        zip(restore_layers[:2], model_layers[:2])):
                    if i == 1:
                        restore_weights = restore_layer.get_weights()
                        kernel_weights = np.concatenate(
                            [restore_weights[0],
                             np.zeros(shape=(1, 1, 32, 1))],
                            axis=-1)
                        kernel_bias = np.concatenate(
                            [restore_weights[1],
                             np.zeros(shape=(1, ))],
                            axis=-1)
                        model_layer.set_weights([kernel_weights, kernel_bias])
                    else:
                        model_layer.set_weights(restore_layer.get_weights())
                #step 1 heat map
                depth_kernel, point_kernel, bias = _folding_bn(
                    model_layers[0].get_weights())
                conv_1x1_kernel, conv_1x1_bias = model_layers[1].get_weights()
                save(os.path.join(save_root, 'heat'), depth_kernel,
                     point_kernel, bias, conv_1x1_kernel, conv_1x1_bias)
                # step 2 size map

                depth_kernel, point_kernel, bias = _folding_bn(
                    restore_size_layers[0].get_weights())
                conv_1x1_kernel, conv_1x1_bias = restore_size_layers[
                    1].get_weights()

                save(os.path.join(save_root, 'size'), depth_kernel,
                     point_kernel, bias, conv_1x1_kernel, conv_1x1_bias)

                depth_kernel, point_kernel, bias = _folding_bn(
                    restore_offset_layers[0].get_weights())
                conv_1x1_kernel, conv_1x1_bias = [], []
                for layer in restore_offset_layers[1:]:
                    conv_1x1_kernel += [layer.get_weights()[0]]
                    conv_1x1_bias += [layer.get_weights()[1]]
                conv_1x1_kernel = np.concatenate(conv_1x1_kernel, axis=-1)
                conv_1x1_bias = np.concatenate(conv_1x1_bias, axis=-1)

                save(os.path.join(save_root, 'offset'), depth_kernel,
                     point_kernel, bias, conv_1x1_kernel, conv_1x1_bias)
                # step 3 save experiments
                _experimentt_folding(save_root,
                                     experiment_layers[1].get_weights())
                model_layers[2].set_weights(experiment_layers[0].get_weights())

            elif key == 'backbone':
                load_weights = restored_model.backbone.get_layer(
                    'hard_net').get_weights()
                model.backbone.get_layer('hard_net').set_weights(load_weights)
            elif key == 'neck':
                load_weights = restored_model.get_layer(key).get_weights()
                model.get_layer(key).set_weights(load_weights)

        except KeyError:
            print('Restore key error, please check you model ...')

    print(f'Finish load-wights ...')
    return model


def convert2tfl(img_root, save_root, model, tfl_format):

    def representative_dataset_gen():
        jpg_files = glob(os.path.join(img_root, '*.jpg'))
        pbg_file = glob(os.path.join(img_root, '*.png'))
        filenames = list(jpg_files) + list(pbg_file)
        num_files = len(filenames)
        idx = np.arange(num_files)
        idx = np.random.shuffle(idx)
        filenames = np.asarray(filenames)[idx]
        for i, file_path in enumerate(filenames[0][:500]):
            img = cv2.imread(file_path)
            img = img[..., ::-1] / 255
            img = np.asarray(img).astype(np.float32)
            img = cv2.resize(img, (320, 192))
            b_imgs = img[np.newaxis, ...]
            yield [b_imgs]

    print("Start converting tf-lite model and the format is {} ...".format(
        tfl_format))
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if tfl_format == 'fp32':
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.target_spec.supported_types = [tf.float32]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        save_tfl_name = 'FP32.tflite'
    elif tfl_format == 'int8':
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        converter.target_spec.supported_types = [tf.int8, tf.uint8]
        converter.representative_dataset = representative_dataset_gen
        save_tfl_name = 'INT8.tflite'
    tflite_model = converter.convert()
    with open(os.path.join(save_root, save_tfl_name), 'wb') as f:
        f.write(tflite_model)
    print("Finish tf-lite and successfully save tf-lite model at the path {}".
          format(save_root))


def run(cfg_path, save_root, img_root, tfl_format):
    # prune offset and size branch for optimizing
    # You should mually change the model architecture
    with open(cfg_path) as f:
        config = commentjson.loads(f.read())
    config = Box(config)
    backbone = HardNet39(input_shape=(192, 320, 3),
                         pooling='avg_pool',
                         kernel_initializer='he_uniform')
    neck = NECK_FACTORY.get(config.models.neck.module_name)(config.models,
                                                            name='neck')
    head = HEAD_FACTORY.get(config.models.head.module_name)(config.models,
                                                            name='head')
    model = MODULE_FACTORY.get(config.models.model_name)(config.models,
                                                         backbone, neck, head)
    cp_dir = config.model_path
    model = _restore(model, cp_dir, save_root)
    _ = model(tf.constant(0., shape=(1, 192, 320, 3)), training=False)
    inputs = tf.keras.Input(shape=(192, 320, 3), batch_size=1, name='input')
    preds = model(inputs, training=False)
    model = tf.keras.Model(inputs, preds)
    convert2tfl(img_root, save_root, model, tfl_format)
    tf.keras.models.save_model(model, save_root)


def parse_config():

    parser = argparse.ArgumentParser('Argparser for optimizing model')
    parser.add_argument('--config')
    parser.add_argument('--save_root')
    parser.add_argument('--img_root')
    parser.add_argument('--tfl_format')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_config()
    print(f"Use following config to produce tensorflow graph: {args.config}.")

    run(args.config, args.save_root, args.img_root, args.tfl_format)
